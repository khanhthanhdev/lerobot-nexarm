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

from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim.mujoco_backend import (
    RAW_RANGES,
    NexArmMujocoBackend,
    resolve_model_path,
)
from lerobot.robots.nexarm_sim.pick_place_task import NexArmPickPlaceTask

DEFAULT_SCENE_PATH = Path("sim/description/mjcf/scene.xml")
FALLBACK_SCENE_PATH = Path("sim/fusion_export/scene.xml")


class NexArmPickPlaceEnv(gym.Env):
    """Gymnasium environment for the Hiwonder NexArm 6-DoF tabletop pick-and-place task.

    This environment wraps the high-fidelity MuJoCo physics model of the NexArm robot
    and complies 100% with the Gymnasium API standards.
    """

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
        target_radius_m: float = 0.05,
        success_hold_s: float = 0.5,
    ) -> None:
        super().__init__()
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.fps = fps
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.target_radius_m = target_radius_m
        self.success_hold_s = success_hold_s

        if model_path is not None:
            resolved_path = resolve_model_path(Path(model_path))
        else:
            try:
                resolved_path = resolve_model_path(DEFAULT_SCENE_PATH)
            except FileNotFoundError:
                resolved_path = resolve_model_path(FALLBACK_SCENE_PATH)

        self.backend = NexArmMujocoBackend(
            model_path=resolved_path,
            fps=self.fps,
            camera_width=self.observation_width,
            camera_height=self.observation_height,
            camera_names=("front", "wrist"),
        )
        self.task = NexArmPickPlaceTask(
            self.backend,
            target_radius_m=self.target_radius_m,
            success_hold_s=self.success_hold_s,
            timeout_s=float(self.max_episode_steps) / float(self.fps),
        )

        # Action Space: 6 continuous normalized actions in [-1.0, 1.0]
        # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # Observation Space
        agent_pos_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
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

        if self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_pos_space,
                    "pixels/front": front_img_space,
                    "pixels/wrist": wrist_img_space,
                }
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels/front": front_img_space,
                    "pixels/wrist": wrist_img_space,
                }
            )
        elif self.obs_type == "state":
            env_state_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(6,),
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

    def _get_agent_pos(self) -> np.ndarray:
        """Returns normalized joint positions in [-1.0, 1.0]."""
        raw_positions = self.backend.joint_positions()
        normalized = np.zeros(6, dtype=np.float32)
        for i, name in enumerate(JOINT_NAMES):
            raw = raw_positions[f"{name}.pos"]
            low, high = RAW_RANGES[name]
            # Map [low, high] to [-1.0, 1.0]
            val = 2.0 * (raw - low) / (high - low) - 1.0
            normalized[i] = float(np.clip(val, -1.0, 1.0))
        return normalized

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        if self.obs_type in ("pixels_agent_pos", "state"):
            obs["agent_pos"] = self._get_agent_pos()

        if self.obs_type in ("pixels_agent_pos", "pixels"):
            obs["pixels/front"] = self.backend.render("front")
            obs["pixels/wrist"] = self.backend.render("wrist")

        if self.obs_type == "state":
            cube_pos = self.task.cube_position
            target_pos = self.task.target_position
            env_state = np.concatenate([cube_pos, target_pos]).astype(np.float32)
            obs["environment_state"] = env_state

        return obs

    def _action_to_raw(self, action: np.ndarray) -> dict[str, float]:
        """Converts normalized [-1.0, 1.0] action to raw actuator targets."""
        action = np.clip(action, -1.0, 1.0)
        raw_targets: dict[str, float] = {}
        for i, name in enumerate(JOINT_NAMES):
            low, high = RAW_RANGES[name]
            norm_val = float(action[i])
            raw_val = low + 0.5 * (norm_val + 1.0) * (high - low)
            raw_targets[f"{name}.pos"] = float(np.clip(raw_val, low, high))
        return raw_targets

    def _compute_reward(self, status: Any) -> float:
        if self.reward_type == "sparse":
            return 1.0 if status.success else 0.0

        # Dense shaped reward
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

        return reach_reward + grasp_reward + lift_reward + target_reward + place_reward + success_reward

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

        obs = self._get_obs()
        info = {
            "is_grasped": status.is_grasped,
            "is_released": status.is_released,
            "is_inside_target": status.is_inside_target,
            "success": status.success,
            "reason": status.reason,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        raw_action = self._action_to_raw(action)
        self.backend.set_action(raw_action)
        status = self.task.step()

        obs = self._get_obs()
        reward = self._compute_reward(status)
        terminated = bool(status.terminated)
        truncated = bool(self._step_count >= self.max_episode_steps)

        info = {
            "is_grasped": status.is_grasped,
            "is_released": status.is_released,
            "is_inside_target": status.is_inside_target,
            "success": status.success,
            "reason": status.reason,
        }

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


# Register the environment with Gymnasium
gym.register(
    id="NexArmPickPlace-v0",
    entry_point="lerobot.envs.nexarm:NexArmPickPlaceEnv",
)

gym.register(
    id="gym_nexarm/NexArmPickPlace-v0",
    entry_point="lerobot.envs.nexarm:NexArmPickPlaceEnv",
)
