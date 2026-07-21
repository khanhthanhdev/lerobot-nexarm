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

from dataclasses import dataclass
from pathlib import Path

from ..config import RobotConfig


@RobotConfig.register_subclass("nexarm_sim")
@dataclass
class NexArmSimConfig(RobotConfig):
    """Configuration for the MuJoCo-backed NexArm follower."""

    model_path: Path = Path("sim/fusion_export/scene.xml")
    fps: int = 30
    camera_width: int = 640
    camera_height: int = 480
    camera_names: tuple[str, ...] = ("front", "wrist")
    settle_steps: int = 100

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps cannot be negative")
