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

from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.motors.nexarm.mujoco_mapping import load_joint_mapping
from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim.calibration import (
    SimCalibration,
    apply_calibration,
    resolve_action_delay_steps,
)
from lerobot.robots.nexarm_sim.mujoco_backend import (
    RAW_RANGES,
    NexArmMujocoBackend,
    resolve_model_path,
)
from lerobot.robots.nexarm_sim.pick_place_task import NexArmPickPlaceTask
from lerobot.robots.nexarm_sim.stack_bowls_task import (
    GRIPPER_ASSIST_ENGAGE_POS,
    REAL_TELEOP_ASSIST_RELEASE_POS,
    STAGES,
    NexArmStackBowlsStatus,
    NexArmStackBowlsTask,
    get_task_instruction,
)

DEFAULT_SCENE_PATH = Path("sim/description/mjcf/scene.xml")
DEFAULT_STACK_BOWLS_SCENE_PATH = Path("sim/description/mjcf/bowl_stack_scene.xml")


class _NexArmMujocoEnv(gym.Env):
    """Shared Gymnasium plumbing for the NexArm MuJoCo tasks.

    Actions and ``agent_pos`` use the real follower's units (raw servo ticks, or those ticks normalized to
    [-1, 1]), and the three cameras match the real rig, so a policy trained here sees the same features as
    one trained on real recordings. Subclasses create the task and define the environment state and reward.
    """

    default_scene_path: Path = DEFAULT_SCENE_PATH
    environment_state_dim: int = 6
    task_description: str = ""

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 30,
    }

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        obs_type: Literal["pixels_agent_pos", "pixels", "state"] = "pixels_agent_pos",
        render_mode: Literal["rgb_array", "human"] | None = "rgb_array",
        fps: int = 30,
        observation_width: int = 640,
        observation_height: int = 480,
        max_episode_steps: int = 400,
        reward_type: Literal["dense", "sparse"] = "dense",
        success_hold_s: float = 0.5,
        control_mode: Literal["raw", "normalized"] = "raw",
        calibration_path: Path | str | None = None,
        enable_domain_randomization: bool = False,
        action_delay_steps: int | None = None,
        init_qpos_noise_rad: float = 0.0,
        joint_mapping_path: Path | str | None = None,
    ) -> None:
        super().__init__()
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.fps = fps
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.success_hold_s = success_hold_s
        self.control_mode = control_mode

        resolved_path = resolve_model_path(
            Path(model_path) if model_path is not None else self.default_scene_path
        )

        self.backend = NexArmMujocoBackend(
            model_path=resolved_path,
            fps=self.fps,
            camera_width=self.observation_width,
            camera_height=self.observation_height,
            camera_names=("front", "wrist", "top"),
            action_delay_steps=resolve_action_delay_steps(action_delay_steps, None),
            enable_domain_randomization=enable_domain_randomization,
            init_qpos_noise_rad=init_qpos_noise_rad,
            joint_mapping=load_joint_mapping(joint_mapping_path),
        )
        if calibration_path is not None:
            try:
                apply_calibration(
                    self.backend,
                    SimCalibration.load(calibration_path, self.backend.joint_mapping.version),
                    action_delay_steps=action_delay_steps,
                )
            except BaseException:
                self.backend.close()
                raise
        try:
            self.task = self._make_task()
        except BaseException:
            self.backend.close()
            raise

        # Action Space & Observation Space bounds
        if self.control_mode == "raw":
            raw_low = np.array([min(RAW_RANGES[name]) for name in JOINT_NAMES], dtype=np.float32)
            raw_high = np.array([max(RAW_RANGES[name]) for name in JOINT_NAMES], dtype=np.float32)
            self.action_space = spaces.Box(
                low=raw_low,
                high=raw_high,
                shape=(6,),
                dtype=np.float32,
            )
            agent_pos_space = spaces.Box(
                low=raw_low,
                high=raw_high,
                shape=(6,),
                dtype=np.float32,
            )
        elif self.control_mode == "normalized":
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(6,),
                dtype=np.float32,
            )
            agent_pos_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(6,),
                dtype=np.float32,
            )
        else:
            raise ValueError(
                f"Unsupported control_mode: '{self.control_mode}'. Expected 'raw' or 'normalized'."
            )
        front_img_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.observation_height, self.observation_width, 3),
            dtype=np.uint8,
        )
        wrist_img_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.observation_height, self.observation_width, 3),
            dtype=np.uint8,
        )
        top_img_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.observation_height, self.observation_width, 3),
            dtype=np.uint8,
        )

        if self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_pos_space,
                    "pixels": spaces.Dict(
                        {
                            "front": front_img_space,
                            "wrist": wrist_img_space,
                            "top": top_img_space,
                        }
                    ),
                    "pixels/front": front_img_space,
                    "pixels/wrist": wrist_img_space,
                    "pixels/top": top_img_space,
                }
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "front": front_img_space,
                            "wrist": wrist_img_space,
                            "top": top_img_space,
                        }
                    ),
                    "pixels/front": front_img_space,
                    "pixels/wrist": wrist_img_space,
                    "pixels/top": top_img_space,
                }
            )
        elif self.obs_type == "state":
            env_state_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.environment_state_dim,),
                dtype=np.float32,
            )
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_pos_space,
                    "environment_state": env_state_space,
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

        self._step_count = 0
        self._viewer: Any | None = None

    def _make_task(self) -> Any:
        raise NotImplementedError

    def _environment_state(self) -> np.ndarray:
        raise NotImplementedError

    def _reward_components(self, status: Any) -> dict[str, float]:
        """Dense reward terms by name; the dense reward is their sum."""
        raise NotImplementedError

    def _compute_reward(self, status: Any) -> float:
        if self.reward_type == "sparse":
            self._last_reward_components = {"success": 1.0 if status.success else 0.0}
        else:
            self._last_reward_components = self._reward_components(status)
        return float(sum(self._last_reward_components.values()))

    def _info(self, status: Any) -> dict[str, Any]:
        raise NotImplementedError

    def _get_agent_pos(self) -> np.ndarray:
        """Returns joint positions (raw [0, 4095] or normalized [-1.0, 1.0])."""
        raw_positions = self.backend.joint_positions()
        if self.control_mode == "raw":
            return np.array([raw_positions[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)

        normalized = np.zeros(6, dtype=np.float32)
        for i, name in enumerate(JOINT_NAMES):
            raw = raw_positions[f"{name}.pos"]
            low, high = RAW_RANGES[name]
            # Map [low, high] to [-1.0, 1.0]
            val = 2.0 * (raw - low) / (high - low) - 1.0
            normalized[i] = float(np.clip(val, -1.0, 1.0))
        return normalized

    def _get_obs(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        if self.obs_type in ("pixels_agent_pos", "state"):
            obs["agent_pos"] = self._get_agent_pos()

        if self.obs_type in ("pixels_agent_pos", "pixels"):
            front_img = self.backend.render("front")
            wrist_img = self.backend.render("wrist")
            top_img = self.backend.render("top")
            obs["pixels"] = {"front": front_img, "wrist": wrist_img, "top": top_img}
            obs["pixels/front"] = front_img
            obs["pixels/wrist"] = wrist_img
            obs["pixels/top"] = top_img

        if self.obs_type == "state":
            obs["environment_state"] = self._environment_state().astype(np.float32)

        return obs

    def _action_to_raw(self, action: np.ndarray) -> dict[str, float]:
        """Converts action to raw actuator targets."""
        raw_targets: dict[str, float] = {}
        if self.control_mode == "raw":
            for i, name in enumerate(JOINT_NAMES):
                low, high = RAW_RANGES[name]
                raw_min, raw_max = min(low, high), max(low, high)
                raw_val = float(action[i])
                raw_targets[f"{name}.pos"] = float(np.clip(raw_val, raw_min, raw_max))
        else:
            action = np.clip(action, -1.0, 1.0)
            for i, name in enumerate(JOINT_NAMES):
                low, high = RAW_RANGES[name]
                raw_min, raw_max = min(low, high), max(low, high)
                norm_val = float(action[i])
                raw_val = low + 0.5 * (norm_val + 1.0) * (high - low)
                raw_targets[f"{name}.pos"] = float(np.clip(raw_val, raw_min, raw_max))
        return raw_targets

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        int_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        status = self.task.reset(seed=int_seed, settle_steps=10)
        self._step_count = 0
        return self._get_obs(), self._info(status)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        raw_action = self._action_to_raw(action)
        # Step through the backend so calibrated action latency applies, then evaluate the task.
        self.backend.step(raw_action)
        status = self.task.observe()

        obs = self._get_obs()
        reward = self._compute_reward(status)
        terminated = bool(status.terminated)
        truncated = bool(self._step_count >= self.max_episode_steps)
        info = self._info(status)
        info["reward_components"] = dict(self._last_reward_components)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self.backend.render("front")
        elif self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.backend.model, self.backend.data)
            if self._viewer is not None:
                self._viewer.sync()
            return None
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self.backend.close()


class NexArmPickPlaceEnv(_NexArmMujocoEnv):
    """Pick the red cube and place it in the green target zone."""

    task_description = "Pick up the red cube and place it in the green target zone."

    def __init__(self, *, target_radius_m: float = 0.05, **kwargs: Any) -> None:
        self.target_radius_m = target_radius_m
        super().__init__(**kwargs)

    def _make_task(self) -> NexArmPickPlaceTask:
        return NexArmPickPlaceTask(
            self.backend,
            target_radius_m=self.target_radius_m,
            success_hold_s=self.success_hold_s,
            timeout_s=float(self.max_episode_steps) / float(self.fps),
        )

    def _environment_state(self) -> np.ndarray:
        return np.concatenate([self.task.cube_position, self.task.target_position])

    def _reward_components(self, status: Any) -> dict[str, float]:
        gripper_pos = self.backend.site_position("gripper_frame")
        cube_pos = self.task.cube_position
        target_pos = self.task.target_position

        # 1. Reach bonus
        dist_to_cube = float(np.linalg.norm(gripper_pos - cube_pos))
        reach_reward = 1.0 - float(np.tanh(10.0 * dist_to_cube))

        # 2. Grasp bonus
        grasp_reward = 1.0 if status.is_grasped else 0.0

        # 3. Lift bonus
        cube_z = float(cube_pos[2])
        lift_reward = float(np.clip((cube_z - 0.015) / 0.05, 0.0, 1.0)) if status.is_grasped else 0.0

        # 4. Target alignment bonus
        dist_cube_to_target = float(np.linalg.norm(cube_pos[:2] - target_pos[:2]))
        target_reward = (
            (1.0 - float(np.tanh(5.0 * dist_cube_to_target)))
            if (status.is_grasped and cube_z > 0.03)
            else 0.0
        )

        # 5. Placement & release bonus
        place_reward = 2.0 if (status.is_inside_target and status.is_released) else 0.0

        # 6. Task success bonus
        success_reward = 5.0 if status.success else 0.0

        return {
            "reach": reach_reward,
            "grasp": grasp_reward,
            "lift": lift_reward,
            "target": target_reward,
            "place": place_reward,
            "success": success_reward,
        }

    def _info(self, status: Any) -> dict[str, Any]:
        return {
            "is_grasped": status.is_grasped,
            "is_released": status.is_released,
            "is_inside_target": status.is_inside_target,
            "success": status.success,
            "is_success": status.success,
            "reason": status.reason,
        }


class NexArmStackBowlsEnv(_NexArmMujocoEnv):
    """Stack the red, blue and black bowls in the order named by the episode's instruction.

    This is the task of the real ``nexarm_stack_bowls`` recordings. Each reset draws the bowl layout and the
    stacking order from the seed, and ``task_description`` holds the current episode's instruction. By
    default the grasp is physical: the jaws hold a bowl by friction alone, as in the scripted dataset
    generator. ``grasp_mode="contact_assisted"`` instead carries a bowl rigidly once both jaws touch it. Either
    way a bowl counts as released once the gripper command opens below ``assist_release_pos``; the default
    lets go at the partial openings real operators use.

    The dense reward is potential-based: each shaping term pays the change in its potential, so over an
    episode the terms sum to the final potential minus the first. Holding a bowl therefore earns nothing per
    step, and every stage, release included, raises the potential, so only finishing pays more.
    """

    default_scene_path = DEFAULT_STACK_BOWLS_SCENE_PATH
    # Bottom, middle and top bowl positions, in the order the instruction names them.
    environment_state_dim = 9
    # Larger than the most the shaping terms can pay over an episode (two bowls, 5 each).
    success_reward = 10.0

    def __init__(
        self,
        *,
        max_episode_steps: int = 900,
        position_jitter_m: float = 0.04,
        assist_engage_pos: float = GRIPPER_ASSIST_ENGAGE_POS,
        assist_release_pos: float = REAL_TELEOP_ASSIST_RELEASE_POS,
        grasp_mode: str = "physical",
        **kwargs: Any,
    ) -> None:
        self.position_jitter_m = position_jitter_m
        self.grasp_mode = grasp_mode
        self.assist_engage_pos = assist_engage_pos
        self.assist_release_pos = assist_release_pos
        self._last_potential: dict[str, float] | None = None
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)
        self.task_description = get_task_instruction(*self.task.current_order)

    def _make_task(self) -> NexArmStackBowlsTask:
        return NexArmStackBowlsTask(
            self.backend,
            success_hold_s=self.success_hold_s,
            timeout_s=float(self.max_episode_steps) / float(self.fps),
            position_jitter_m=self.position_jitter_m,
            assist_engage_pos=self.assist_engage_pos,
            assist_release_pos=self.assist_release_pos,
            grasp_mode=self.grasp_mode,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.task_description = info["task"]
        self._last_potential = self._potential()
        return obs, info

    def _environment_state(self) -> np.ndarray:
        return np.concatenate([self.task.bowl_position(color) for color in self.task.current_order])

    def _potential(self) -> dict[str, float]:
        """Task progress, split into the shaping terms; their sum rises strictly from stage to stage.

        A bowl is placed once it is nested and let go. Each placed bowl is worth 5: 1 each for reach, grasp
        and carry, and 2 for stage (nested, then released). The bowl being moved scores reach, grasp and carry
        in [0, 1] each, plus 1 for stage once nested; a nested bowl keeps full reach, grasp and carry until it
        is let go, so opening the gripper never lowers the potential.
        """
        task = self.task
        bottom, middle, top = task.current_order
        placed = 0
        for lower, upper in ((bottom, middle), (middle, top)):
            if not (task.is_nested(lower, upper) and task.held_bowl != upper):
                break
            placed += 1
        reach = grasp = carry = stage = 0.0
        if placed < 2:
            base, mover = (bottom, middle) if placed == 0 else (middle, top)
            mover_pos, base_pos = task.bowl_position(mover), task.bowl_position(base)
            if task.is_nested(base, mover):
                reach = grasp = carry = stage = 1.0
            else:
                gripper_pos = self.backend.site_position("gripper_frame")
                reach = 1.0 - float(np.tanh(10.0 * np.linalg.norm(gripper_pos - mover_pos)))
                if task.held_bowl == mover:
                    grasp = 1.0
                    carry = 1.0 - float(np.tanh(5.0 * np.linalg.norm(mover_pos[:2] - base_pos[:2])))
        return {
            "reach": placed + reach,
            "grasp": placed + grasp,
            "carry": placed + carry,
            "stage": 2.0 * placed + stage,
        }

    def _reward_components(self, status: NexArmStackBowlsStatus) -> dict[str, float]:
        potential = self._potential()
        previous = self._last_potential if self._last_potential is not None else potential
        self._last_potential = potential
        components = {name: potential[name] - previous[name] for name in potential}
        components["success"] = self.success_reward if status.success else 0.0
        return components

    def _info(self, status: NexArmStackBowlsStatus) -> dict[str, Any]:
        return {
            "held_bowl": self.task.held_bowl or "",
            "is_concentric": status.is_concentric,
            "is_nested_height": status.is_nested_height,
            "is_released": status.is_released,
            "is_arm_static": status.is_arm_static,
            "success": status.success,
            "is_success": status.success,
            "reason": status.reason,
            "task": get_task_instruction(*status.order),
            # Each stage latches once reached, so its value at episode end says how far the episode got.
            **{f"stage/{stage}": status.stages.get(stage, False) for stage in STAGES},
            "failed_stage": status.failed_stage or "",
        }


# Register the environment with Gymnasium
gym.register(
    id="NexArmPickPlace-v0",
    entry_point="lerobot.envs.nexarm:NexArmPickPlaceEnv",
)

gym.register(
    id="gym_nexarm/NexArmPickPlace-v0",
    entry_point="lerobot.envs.nexarm:NexArmPickPlaceEnv",
)

gym.register(
    id="NexArmStackBowls-v0",
    entry_point="lerobot.envs.nexarm:NexArmStackBowlsEnv",
)

gym.register(
    id="gym_nexarm/NexArmStackBowls-v0",
    entry_point="lerobot.envs.nexarm:NexArmStackBowlsEnv",
)
