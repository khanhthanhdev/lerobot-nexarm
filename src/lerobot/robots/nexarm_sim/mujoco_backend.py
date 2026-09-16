# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np
import numpy.typing as npt

from lerobot.motors.nexarm.nexarm import JOINT_NAMES, POSITION_MAX, POSITION_MIN

MUJOCO_JOINTS = {
    "shoulder_pan": "joint_1_base_to_link_1",
    "shoulder_lift": "joint_2_link_1_to_link_2",
    "elbow_flex": "joint_3_link_2_to_link_3",
    "wrist_flex": "joint_4_link_3_to_link_4",
    "wrist_roll": "joint_5_link_4_to_link_5",
    "gripper": "right_jaw_slide_joint",
}

# The physical follower accepts 0..4095 for every servo. Its leader mapping
# deliberately restricts the useful gripper command range to 1195..2833.
RAW_RANGES: dict[str, tuple[int, int]] = dict.fromkeys(JOINT_NAMES[:-1], (POSITION_MIN, POSITION_MAX))
RAW_RANGES["gripper"] = (1195, 2833)

HOME_POSITIONS: dict[str, float] = dict.fromkeys(JOINT_NAMES[:-1], 2048.0)
HOME_POSITIONS["gripper"] = 2833.0


def resolve_model_path(model_path: Path | str) -> Path:
    """Resolve a model path from either the current directory or checkout root."""

    model_path = Path(model_path).expanduser()
    if model_path.is_absolute():
        resolved = model_path
    elif model_path.exists():
        resolved = model_path.resolve()
    else:
        checkout_root = Path(__file__).resolve().parents[4]
        resolved = checkout_root / model_path
    if not resolved.is_file():
        raise FileNotFoundError(
            f"NexArm MuJoCo model not found at {resolved}. "
            "Export the Fusion model first or pass --robot.model_path."
        )
    return resolved


class NexArmMujocoBackend:
    """MuJoCo state, control conversion, stepping, and camera rendering."""

    def __init__(
        self,
        model_path: Path,
        fps: int,
        camera_width: int,
        camera_height: int,
        camera_names: tuple[str, ...],
        action_delay_steps: int = 0,
        enable_domain_randomization: bool = False,
    ) -> None:
        self.model_path = resolve_model_path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.fps = fps
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_names = camera_names
        self.action_delay_steps = max(0, int(action_delay_steps))
        self.enable_domain_randomization = bool(enable_domain_randomization)
        self._action_queue: deque[dict[str, float]] = deque()
        self._renderer: mujoco.Renderer | None = None

        self._joint_ids: dict[str, int] = {}
        self._actuator_ids: dict[str, int] = {}
        actuator_by_joint = {
            int(self.model.actuator_trnid[actuator_id, 0]): actuator_id
            for actuator_id in range(self.model.nu)
            if self.model.actuator_trnid[actuator_id, 0] >= 0
        }

        for feature_name, mujoco_joint_name in MUJOCO_JOINTS.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_joint_name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo model is missing required joint {mujoco_joint_name!r}")
            if joint_id not in actuator_by_joint:
                raise ValueError(f"MuJoCo joint {mujoco_joint_name!r} has no actuator")
            self._joint_ids[feature_name] = joint_id
            self._actuator_ids[feature_name] = actuator_by_joint[joint_id]

        missing_cameras = [
            name
            for name in camera_names
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0
        ]
        if missing_cameras:
            raise ValueError(f"MuJoCo model is missing configured cameras: {missing_cameras}")

        self._camera_ids: dict[str, int] = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) for name in camera_names
        }

        # Cache nominal parameters for resetting after domain randomization
        self._nominal_cam_pos: dict[str, np.ndarray] = {
            name: self.model.cam_pos[cid].copy() for name, cid in self._camera_ids.items()
        }
        self._nominal_cam_quat: dict[str, np.ndarray] = {
            name: self.model.cam_quat[cid].copy() for name, cid in self._camera_ids.items()
        }
        self._nominal_cam_fovy: dict[str, float] = {
            name: float(self.model.cam_fovy[cid]) for name, cid in self._camera_ids.items()
        }
        self._nominal_light_ambient = self.model.light_ambient.copy()
        self._nominal_light_diffuse = self.model.light_diffuse.copy()
        self._nominal_dof_damping = self.model.dof_damping.copy()
        self._nominal_dof_frictionloss = self.model.dof_frictionloss.copy()
        self._nominal_geom_friction = self.model.geom_friction.copy()
        self._nominal_geom_rgba = self.model.geom_rgba.copy()
        self._nominal_body_mass = self.model.body_mass.copy()

        control_period = 1.0 / fps
        self.steps_per_action = max(1, round(control_period / self.model.opt.timestep))

    def _control_range(self, feature_name: str) -> tuple[float, float]:
        actuator_id = self._actuator_ids[feature_name]
        low, high = self.model.actuator_ctrlrange[actuator_id]
        return float(low), float(high)

    def raw_to_control(self, feature_name: str, raw_position: float) -> float:
        raw_low, raw_high = RAW_RANGES[feature_name]
        raw_position = float(np.clip(raw_position, raw_low, raw_high))
        control_low, control_high = self._control_range(feature_name)
        ratio = (raw_position - raw_low) / (raw_high - raw_low)
        return control_low + ratio * (control_high - control_low)

    def control_to_raw(self, feature_name: str, control_position: float) -> float:
        control_low, control_high = self._control_range(feature_name)
        control_position = float(np.clip(control_position, control_low, control_high))
        raw_low, raw_high = RAW_RANGES[feature_name]
        ratio = (control_position - control_low) / (control_high - control_low)
        return raw_low + ratio * (raw_high - raw_low)

    def randomize_domain(self, rng: np.random.Generator | None = None) -> None:
        """Apply Visual Domain Randomization (VDR) and Dynamics Domain Randomization (DDR)."""
        if rng is None:
            rng = np.random.default_rng()

        # 1. Camera extrinsics & intrinsics perturbation
        for name, cid in self._camera_ids.items():
            self.model.cam_pos[cid] = self._nominal_cam_pos[name] + rng.uniform(-0.012, 0.012, size=3)
            self.model.cam_fovy[cid] = self._nominal_cam_fovy[name] + float(rng.uniform(-1.5, 1.5))
            euler_noise = rng.uniform(-0.035, 0.035, size=3)
            delta_q = np.array([1.0, euler_noise[0] / 2, euler_noise[1] / 2, euler_noise[2] / 2])
            delta_q /= np.linalg.norm(delta_q)
            w1, x1, y1, z1 = self._nominal_cam_quat[name]
            w2, x2, y2, z2 = delta_q
            perturbed_q = np.array(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ]
            )
            self.model.cam_quat[cid] = perturbed_q / np.linalg.norm(perturbed_q)

        # 2. Lighting intensity & color variation
        light_scale = float(rng.uniform(0.75, 1.35))
        for i in range(self.model.nlight):
            self.model.light_ambient[i] = np.clip(
                self._nominal_light_ambient[i] * light_scale * rng.uniform(0.9, 1.1, size=3),
                0.05,
                0.95,
            )
            self.model.light_diffuse[i] = np.clip(
                self._nominal_light_diffuse[i] * light_scale * rng.uniform(0.9, 1.1, size=3),
                0.1,
                1.0,
            )

        # 3. Floor & Cube visual / physical properties
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            self.model.geom_rgba[floor_id, :3] = rng.uniform(0.65, 0.95, size=3)

        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_collision")
        if cube_geom_id >= 0:
            self.model.geom_rgba[cube_geom_id, :3] = np.clip(
                self._nominal_geom_rgba[cube_geom_id, :3] + rng.uniform(-0.12, 0.12, size=3),
                0.0,
                1.0,
            )
            self.model.geom_friction[cube_geom_id, 0] = float(rng.uniform(0.8, 2.2))

        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if cube_body_id >= 0:
            self.model.body_mass[cube_body_id] = self._nominal_body_mass[cube_body_id] * float(
                rng.uniform(0.8, 1.25)
            )

        # 4. Joint dynamics (damping & frictionloss)
        for _feature_name, joint_id in self._joint_ids.items():
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            damping_scale = float(rng.uniform(0.8, 1.25))
            friction_scale = float(rng.uniform(0.75, 1.30))
            self.model.dof_damping[dof_adr] = self._nominal_dof_damping[dof_adr] * damping_scale
            self.model.dof_frictionloss[dof_adr] = self._nominal_dof_frictionloss[dof_adr] * friction_scale

    def reset_domain(self) -> None:
        """Restore nominal camera, lighting, material, and dynamics parameters."""
        for name, cid in self._camera_ids.items():
            self.model.cam_pos[cid] = self._nominal_cam_pos[name].copy()
            self.model.cam_quat[cid] = self._nominal_cam_quat[name].copy()
            self.model.cam_fovy[cid] = self._nominal_cam_fovy[name]
        self.model.light_ambient[:] = self._nominal_light_ambient
        self.model.light_diffuse[:] = self._nominal_light_diffuse
        self.model.dof_damping[:] = self._nominal_dof_damping
        self.model.dof_frictionloss[:] = self._nominal_dof_frictionloss
        self.model.geom_friction[:] = self._nominal_geom_friction
        self.model.geom_rgba[:] = self._nominal_geom_rgba
        self.model.body_mass[:] = self._nominal_body_mass

    def reset(self, settle_steps: int = 0, rng: np.random.Generator | None = None) -> None:
        if self.enable_domain_randomization:
            self.randomize_domain(rng)
        else:
            self.reset_domain()

        mujoco.mj_resetData(self.model, self.data)
        home_action: dict[str, float] = {}
        for feature_name in JOINT_NAMES:
            target = self.raw_to_control(feature_name, HOME_POSITIONS[feature_name])
            actuator_id = self._actuator_ids[feature_name]
            joint_id = self._joint_ids[feature_name]
            self.data.ctrl[actuator_id] = target
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = target
            home_action[f"{feature_name}.pos"] = HOME_POSITIONS[feature_name]

        self._action_queue.clear()
        for _ in range(self.action_delay_steps):
            self._action_queue.append(dict(home_action))

        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def set_action(self, action: Mapping[str, float]) -> dict[str, float]:
        missing = [f"{name}.pos" for name in JOINT_NAMES if f"{name}.pos" not in action]
        if missing:
            raise KeyError(f"NexArm simulation action is missing keys: {missing}")

        sent: dict[str, float] = {}
        for feature_name in JOINT_NAMES:
            key = f"{feature_name}.pos"
            raw_low, raw_high = RAW_RANGES[feature_name]
            raw_position = float(np.clip(float(action[key]), raw_low, raw_high))
            self.data.ctrl[self._actuator_ids[feature_name]] = self.raw_to_control(feature_name, raw_position)
            sent[key] = raw_position
        return sent

    def step(self, action: Mapping[str, float] | None = None) -> dict[str, float] | None:
        if action is not None:
            if self.action_delay_steps > 0:
                self._action_queue.append(
                    {f"{name}.pos": float(action[f"{name}.pos"]) for name in JOINT_NAMES}
                )
                action_to_apply = self._action_queue.popleft()
                sent = self.set_action(action_to_apply)
            else:
                sent = self.set_action(action)
        else:
            sent = None
        mujoco.mj_step(self.model, self.data, self.steps_per_action)
        return sent

    def joint_positions(self) -> dict[str, float]:
        positions: dict[str, float] = {}
        for feature_name in JOINT_NAMES:
            joint_id = self._joint_ids[feature_name]
            qpos = float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
            positions[f"{feature_name}.pos"] = self.control_to_raw(feature_name, qpos)
        return positions

    def solve_ik(
        self,
        target_xyz: npt.ArrayLike,
        *,
        site_name: str = "gripper_frame",
        max_iterations: int = 120,
        tolerance_m: float = 0.003,
        restarts: int = 8,
        seed: int = 0,
    ) -> dict[str, float] | None:
        """Solve position-only IK for deterministic scripted data generation."""

        target = np.asarray(target_xyz, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError(f"target_xyz must have shape (3,), got {target.shape}")
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"MuJoCo model is missing required IK site {site_name!r}")

        arm_names = JOINT_NAMES[:-1]
        qpos_addresses = [
            int(self.model.jnt_qposadr[self._joint_ids[feature_name]]) for feature_name in arm_names
        ]
        dof_addresses = [
            int(self.model.jnt_dofadr[self._joint_ids[feature_name]]) for feature_name in arm_names
        ]
        rng = np.random.default_rng(seed)
        for restart in range(restarts + 1):
            working = mujoco.MjData(self.model)
            working.qpos[:] = self.data.qpos
            if restart:
                for feature_name, qpos_address in zip(arm_names, qpos_addresses, strict=True):
                    low, high = self.model.jnt_range[self._joint_ids[feature_name]]
                    working.qpos[qpos_address] = rng.uniform(low * 0.8, high * 0.8)

            for _ in range(max_iterations):
                mujoco.mj_forward(self.model, working)
                error = target - working.site_xpos[site_id]
                if np.linalg.norm(error) <= tolerance_m:
                    return {
                        feature_name: self.control_to_raw(
                            feature_name,
                            float(working.qpos[qpos_address]),
                        )
                        for feature_name, qpos_address in zip(
                            arm_names,
                            qpos_addresses,
                            strict=True,
                        )
                    }

                jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, working, jacobian, None, site_id)
                selected = jacobian[:, dof_addresses]
                damping = 1e-3
                delta = selected.T @ np.linalg.solve(
                    selected @ selected.T + damping * np.eye(3),
                    error,
                )
                delta = np.clip(delta, -0.15, 0.15)
                for index, (feature_name, qpos_address) in enumerate(
                    zip(arm_names, qpos_addresses, strict=True)
                ):
                    low, high = self.model.jnt_range[self._joint_ids[feature_name]]
                    working.qpos[qpos_address] = np.clip(
                        working.qpos[qpos_address] + delta[index],
                        low,
                        high,
                    )
        return None

    def site_position(self, name: str) -> npt.NDArray[np.float64]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"MuJoCo model is missing site {name!r}")
        return self.data.site_xpos[site_id].copy()

    def body_position(self, name: str) -> npt.NDArray[np.float64]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing body {name!r}")
        return self.data.xpos[body_id].copy()

    def geom_position(self, name: str) -> npt.NDArray[np.float64]:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"MuJoCo model is missing geom {name!r}")
        return self.data.geom_xpos[geom_id].copy()

    def render(self, camera_name: str) -> npt.NDArray[np.uint8]:
        if camera_name not in self.camera_names:
            raise KeyError(f"Camera {camera_name!r} is not configured")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.camera_height,
                width=self.camera_width,
            )
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
