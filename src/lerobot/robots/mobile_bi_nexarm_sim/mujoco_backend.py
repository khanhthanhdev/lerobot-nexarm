# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import numpy.typing as npt

from .contract import ARM_FEATURES, MobileNexArmContract, checkout_root, load_contract


def resolve_checkout_path(path: Path) -> Path:
    expanded = path.expanduser()
    resolved = (
        expanded.resolve() if expanded.is_absolute() or expanded.exists() else checkout_root() / expanded
    )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


@dataclass(frozen=True)
class TaskStatus:
    success: bool
    terminated: bool
    reason: str | None
    hold_time_s: float


class MobileBiNexArmMujocoBackend:
    """Name-driven MuJoCo implementation of the canonical 16-D contract."""

    def __init__(
        self,
        *,
        model_path: Path,
        spec_path: Path,
        fps: int,
        camera_width: int,
        camera_height: int,
        camera_names: tuple[str, ...],
        seed: int,
        first_task_mode: bool,
        episode_time_s: float,
    ) -> None:
        self.model_path = resolve_checkout_path(model_path)
        self.contract: MobileNexArmContract = load_contract(resolve_checkout_path(spec_path))
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.fps = fps
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_names = camera_names
        self.first_task_mode = first_task_mode
        self.episode_time_s = episode_time_s
        self._renderer: mujoco.Renderer | None = None
        self._episode_seed = seed
        self._rng = np.random.default_rng(seed)
        self._hold_steps = 0
        self._failure_reason: str | None = None
        self._last_action = self.contract.home_action

        names = (
            "base_x",
            "base_y",
            "base_yaw",
            "lift_axis",
            "cube_joint",
            *[f"{side}_{feature}" for side in ("left", "right") for feature in ARM_FEATURES],
        )
        self._joint_ids = {name: self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in names}
        self._actuator_ids = {
            f"{side}_{feature}": self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_{feature}_control")
            for side in ("left", "right")
            for feature in ARM_FEATURES
        }
        self._actuator_ids["lift_axis"] = self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_axis_control")
        self._body_ids = {
            name: self._required_id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("cube", "target_zone", "left_gripper", "right_gripper")
        }
        self._site_ids = {
            side: self._required_id(mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp") for side in ("left", "right")
        }
        missing_cameras = [
            name
            for name in camera_names
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0
        ]
        if missing_cameras:
            raise ValueError(f"MuJoCo model is missing configured cameras: {missing_cameras}")
        self.steps_per_action = max(1, round((1 / fps) / self.model.opt.timestep))

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo model is missing required {object_type.name}: {name}")
        return object_id

    def _qpos_address(self, joint_name: str) -> int:
        return int(self.model.jnt_qposadr[self._joint_ids[joint_name]])

    def _qvel_address(self, joint_name: str) -> int:
        return int(self.model.jnt_dofadr[self._joint_ids[joint_name]])

    def reset(self, *, seed: int | None = None, settle_steps: int = 0) -> dict[str, float]:
        if seed is not None:
            self._episode_seed = seed
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._hold_steps = 0
        self._failure_reason = None
        self._last_action = self.contract.home_action
        lift_home = self.contract.payload["lift"]["home_mm"] / 1000
        self.data.qpos[self._qpos_address("lift_axis")] = lift_home
        self.data.ctrl[self._actuator_ids["lift_axis"]] = lift_home
        for side in ("left", "right"):
            for feature, joint in self.contract.joint_by_feature.items():
                value = joint.raw_to_sim(joint.home_raw)
                self.data.qpos[self._qpos_address(f"{side}_{feature}")] = value
                self.data.ctrl[self._actuator_ids[f"{side}_{feature}"]] = value

        cube_x = self._rng.uniform(0.48, 0.62)
        cube_y = self._rng.uniform(0.08, 0.20)
        target_x = self._rng.uniform(0.48, 0.62)
        target_y = self._rng.uniform(-0.08, 0.02)
        cube_qpos = self._qpos_address("cube_joint")
        table_top = self.contract.payload["task"]["table_top_m"]
        cube_half = self.contract.payload["task"]["cube_half_size_m"]
        self.data.qpos[cube_qpos : cube_qpos + 7] = [
            cube_x,
            cube_y,
            table_top + cube_half,
            1,
            0,
            0,
            0,
        ]
        self.model.body_pos[self._body_ids["target_zone"], :3] = [
            target_x,
            target_y,
            table_top + 0.002,
        ]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
        return self.joint_state()

    @property
    def episode_seed(self) -> int:
        return self._episode_seed

    def raw_to_sim(self, feature: str, raw: float) -> float:
        return self.contract.joint_by_feature[feature].raw_to_sim(raw)

    def sim_to_raw(self, feature: str, value: float) -> float:
        return self.contract.joint_by_feature[feature].sim_to_raw(value)

    def solve_ik(
        self,
        side: str,
        target_xyz: npt.ArrayLike,
        *,
        max_iterations: int = 100,
        tolerance_m: float = 0.003,
        restarts: int = 8,
    ) -> dict[str, float] | None:
        """Damped least-squares position IK for scripted data generation."""
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        target = np.asarray(target_xyz, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError(f"target_xyz must have shape (3,), got {target.shape}")
        joint_names = [f"{side}_{feature}" for feature in ARM_FEATURES[:-1]]
        qpos_addresses = [self._qpos_address(name) for name in joint_names]
        dof_addresses = [self._qvel_address(name) for name in joint_names]
        ik_rng = np.random.default_rng(self._episode_seed + int(np.sum(np.abs(target) * 1000)))
        for restart in range(restarts + 1):
            working = mujoco.MjData(self.model)
            working.qpos[:] = self.data.qpos
            if restart:
                for feature, qpos_address in zip(ARM_FEATURES[:-1], qpos_addresses, strict=True):
                    low, high = self.contract.joint_by_feature[feature].sim_range
                    working.qpos[qpos_address] = ik_rng.uniform(low * 0.7, high * 0.7)
            for _ in range(max_iterations):
                mujoco.mj_forward(self.model, working)
                error = target - working.site_xpos[self._site_ids[side]]
                if np.linalg.norm(error) <= tolerance_m:
                    return {
                        feature: self.sim_to_raw(feature, float(working.qpos[qpos_address]))
                        for feature, qpos_address in zip(ARM_FEATURES[:-1], qpos_addresses, strict=True)
                    }
                jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(
                    self.model,
                    working,
                    jacobian,
                    None,
                    self._site_ids[side],
                )
                selected = jacobian[:, dof_addresses]
                damping = 1e-3
                delta = selected.T @ np.linalg.solve(selected @ selected.T + damping * np.eye(3), error)
                delta = np.clip(delta, -0.15, 0.15)
                for index, (feature, qpos_address) in enumerate(
                    zip(ARM_FEATURES[:-1], qpos_addresses, strict=True)
                ):
                    limits = self.contract.joint_by_feature[feature].sim_range
                    working.qpos[qpos_address] = np.clip(working.qpos[qpos_address] + delta[index], *limits)
        return None

    def body_position(self, name: str) -> npt.NDArray[np.float64]:
        return self.data.xpos[self._body_ids[name]].copy()

    def set_action(self, action: Mapping[str, float]) -> dict[str, float]:
        missing = [name for name in self.contract.feature_names if name not in action]
        if missing:
            raise KeyError(f"mobile NexArm action is missing keys: {missing}")
        sent: dict[str, float] = {}
        home = self.contract.home_action
        for side in ("left", "right"):
            for feature, joint in self.contract.joint_by_feature.items():
                key = f"{side}_{feature}.pos"
                requested = home[key] if self.first_task_mode and side == "right" else float(action[key])
                raw = float(np.clip(requested, *joint.raw_range))
                self.data.ctrl[self._actuator_ids[f"{side}_{feature}"]] = joint.raw_to_sim(raw)
                sent[key] = raw

        limits = self.contract.payload["base"]["velocity_limits"]
        base_values = {
            "x.vel": float(np.clip(float(action["x.vel"]), *limits["x_m_s"])),
            "y.vel": float(np.clip(float(action["y.vel"]), *limits["y_m_s"])),
            "theta.vel": float(np.clip(float(action["theta.vel"]), *limits["theta_deg_s"])),
        }
        if self.first_task_mode:
            base_values = dict.fromkeys(base_values, 0.0)
        sent.update(base_values)
        lift_range = self.contract.payload["lift"]["range_mm"]
        lift = float(np.clip(float(action["lift_axis.height_mm"]), *lift_range))
        if self.first_task_mode:
            lift = float(self.contract.payload["lift"]["home_mm"])
        self.data.ctrl[self._actuator_ids["lift_axis"]] = lift / 1000
        sent["lift_axis.height_mm"] = lift
        self._last_action = sent
        return sent

    def _integrate_virtual_base(self) -> None:
        yaw_addr = self._qpos_address("base_yaw")
        yaw = float(self.data.qpos[yaw_addr])
        vx_body = self._last_action["x.vel"]
        vy_body = self._last_action["y.vel"]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        vx_world = cos_yaw * vx_body - sin_yaw * vy_body
        vy_world = sin_yaw * vx_body + cos_yaw * vy_body
        dt = 1 / self.fps
        self.data.qpos[self._qpos_address("base_x")] += vx_world * dt
        self.data.qpos[self._qpos_address("base_y")] += vy_world * dt
        self.data.qpos[yaw_addr] += np.deg2rad(self._last_action["theta.vel"]) * dt
        self.data.qvel[self._qvel_address("base_x")] = vx_world
        self.data.qvel[self._qvel_address("base_y")] = vy_world
        self.data.qvel[self._qvel_address("base_yaw")] = np.deg2rad(self._last_action["theta.vel"])

    def step(self, action: Mapping[str, float]) -> dict[str, float]:
        if self.task_status().terminated:
            raise RuntimeError("episode has terminated; reset before sending another action")
        sent = self.set_action(action)
        self._integrate_virtual_base()
        mujoco.mj_step(self.model, self.data, self.steps_per_action)
        self._update_task_status()
        return sent

    def joint_state(self) -> dict[str, float]:
        state: dict[str, float] = {}
        for side in ("left", "right"):
            for feature in ARM_FEATURES:
                value = float(self.data.qpos[self._qpos_address(f"{side}_{feature}")])
                state[f"{side}_{feature}.pos"] = self.sim_to_raw(feature, value)
        state.update(
            {
                "x.vel": float(self._last_action["x.vel"]),
                "y.vel": float(self._last_action["y.vel"]),
                "theta.vel": float(self._last_action["theta.vel"]),
                "lift_axis.height_mm": float(self.data.qpos[self._qpos_address("lift_axis")] * 1000),
            }
        )
        return state

    def _update_task_status(self) -> None:
        cube = self.data.xpos[self._body_ids["cube"]]
        target = self.data.xpos[self._body_ids["target_zone"]]
        task = self.contract.payload["task"]
        inside = (
            np.linalg.norm(cube[:2] - target[:2]) <= task["target_radius_m"] - 0.005
            and cube[2] >= task["table_top_m"] + task["cube_half_size_m"] - 0.005
        )
        gripper_spec = self.contract.payload["joints"][5]
        released = self._last_action["left_gripper.pos"] <= (
            gripper_spec["open_raw"] + 0.1 * (gripper_spec["closed_raw"] - gripper_spec["open_raw"])
        )
        cube_joint = self._joint_ids["cube_joint"]
        dof = int(self.model.jnt_dofadr[cube_joint])
        cube_speed = np.linalg.norm(self.data.qvel[dof : dof + 3])
        if inside and released and cube_speed < 0.08:
            self._hold_steps += 1
        else:
            self._hold_steps = 0
        if cube[2] < 0.05:
            self._failure_reason = "dropped_cube"
        elif self._has_forbidden_collision():
            self._failure_reason = "forbidden_collision"
        elif self.data.time >= self.episode_time_s:
            self._failure_reason = "unreleased_grasp" if inside and not released else "timeout"

    def _has_forbidden_collision(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            names = {
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
            }
            if "table_collision" in names and any(
                name is not None and (name.startswith("left_") or name.startswith("right_")) for name in names
            ):
                return True
            sides = {
                "left" if name and name.startswith("left_") else "right"
                for name in names
                if name and (name.startswith("left_") or name.startswith("right_"))
            }
            if sides == {"left", "right"}:
                return True
        return False

    def task_status(self) -> TaskStatus:
        success = self._hold_steps >= round(self.contract.payload["task"]["success_hold_s"] * self.fps)
        reason = "success" if success else self._failure_reason
        return TaskStatus(
            success=success,
            terminated=success or self._failure_reason is not None,
            reason=reason,
            hold_time_s=self._hold_steps / self.fps,
        )

    def render(self, camera_name: str) -> npt.NDArray[np.uint8]:
        if camera_name not in self.camera_names:
            raise KeyError(f"camera {camera_name!r} is not configured")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
