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

from functools import cached_property

from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .calibration import SimCalibration, apply_calibration, resolve_action_delay_steps
from .config_nexarm_sim import NexArmSimConfig
from .mujoco_backend import NexArmMujocoBackend


class NexArmSim(Robot):
    """LeRobot follower backed by the NexArm MuJoCo model.

    ``name`` selects this simulated implementation, while ``robot_type`` reports the physical embodiment
    (``nexarm_follower``) so simulated and real datasets and policies stay compatible.
    """

    config_class = NexArmSimConfig
    name = "nexarm_sim"

    def __init__(self, config: NexArmSimConfig):
        super().__init__(config)
        self.robot_type = "nexarm_follower"
        self.config = config
        self._backend: NexArmMujocoBackend | None = None

    @property
    def backend(self) -> NexArmMujocoBackend:
        if self._backend is None:
            raise ConnectionError("NexArm simulation is not connected")
        return self._backend

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {f"{name}.pos": float for name in JOINT_NAMES}
        features.update(
            dict.fromkeys(
                self.config.camera_names,
                (self.config.camera_height, self.config.camera_width, 3),
            )
        )
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @property
    def is_connected(self) -> bool:
        return self._backend is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        backend = NexArmMujocoBackend(
            model_path=self.config.model_path,
            fps=self.config.fps,
            camera_width=self.config.camera_width,
            camera_height=self.config.camera_height,
            camera_names=self.config.camera_names,
            action_delay_steps=resolve_action_delay_steps(self.config.action_delay_steps, None),
            enable_domain_randomization=self.config.enable_domain_randomization,
        )
        if self.config.calibration_path is not None:
            try:
                apply_calibration(
                    backend,
                    SimCalibration.load(self.config.calibration_path),
                    action_delay_steps=self.config.action_delay_steps,
                )
            except Exception:
                backend.close()
                raise
        self._backend = backend
        self.configure()

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        self.backend.reset(settle_steps=self.config.settle_steps)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation: RobotObservation = self.backend.joint_positions()
        for camera_name in self.config.camera_names:
            observation[camera_name] = self.backend.render(camera_name)
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command a joint target and return it clamped, like ``NexArmFollower.send_action``.

        With ``action_delay_steps > 0`` the physics applies an older command this tick; the returned value
        is still this tick's command, which is what a dataset labels as the action.
        """
        commanded = self.backend.clamp_action(action)
        self.backend.step(commanded)
        return commanded

    @check_if_not_connected
    def reset(self) -> None:
        self.backend.reset(settle_steps=self.config.settle_steps)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.backend.close()
        self._backend = None
