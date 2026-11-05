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

import math
from dataclasses import dataclass
from pathlib import Path

from lerobot.robots.config import RobotConfig

from .mujoco_backend import SERVO_MAX_ACCEL_TICKS_S2, SERVO_MAX_SPEED_TICKS_S


@RobotConfig.register_subclass("nexarm_sim")
@dataclass
class NexArmSimConfig(RobotConfig):
    """Configuration for the MuJoCo-backed NexArm follower."""

    model_path: Path = Path("sim/description/mjcf/scene.xml")
    fps: int = 30
    camera_width: int = 640
    camera_height: int = 480
    camera_names: tuple[str, ...] = ("front", "wrist", "top")
    settle_steps: int = 100
    # None takes the calibrated latency (or 0 without a calibration); an explicit value always wins.
    action_delay_steps: int | None = None
    enable_domain_randomization: bool = False
    calibration_path: Path | None = None
    # Real-to-sim joint sign and offset from tools/nexarm/probe_joint_mapping.py; None is the identity.
    joint_mapping_path: Path | None = None
    # Servo setpoint limits in raw ticks, in JOINT_NAMES order. Arm speeds are the HX spec (the real demos never
    # saturate an arm joint); the gripper speed is the real saturated speed. The acceleration cap (None per joint
    # or for all: no cap) keeps the stiff arm's jerk from shaking objects out of a friction grasp.
    max_speed_ticks_s: tuple[float, ...] = SERVO_MAX_SPEED_TICKS_S
    max_accel_ticks_s2: tuple[float | None, ...] | None = SERVO_MAX_ACCEL_TICKS_S2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps cannot be negative")
        if self.action_delay_steps is not None and self.action_delay_steps < 0:
            raise ValueError("action_delay_steps cannot be negative")
        joint_count = len(SERVO_MAX_SPEED_TICKS_S)
        if len(self.max_speed_ticks_s) != joint_count or any(
            not (math.isfinite(speed) and speed > 0) for speed in self.max_speed_ticks_s
        ):
            raise ValueError(f"max_speed_ticks_s needs {joint_count} positive values")
        if self.max_accel_ticks_s2 is not None and (
            len(self.max_accel_ticks_s2) != joint_count
            or any(
                accel is not None and not (math.isfinite(accel) and accel > 0)
                for accel in self.max_accel_ticks_s2
            )
        ):
            raise ValueError(f"max_accel_ticks_s2 needs {joint_count} positive values or None")
