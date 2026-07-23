# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass
from pathlib import Path

from ..config import RobotConfig


@RobotConfig.register_subclass("mobile_bi_nexarm_sim")
@dataclass
class MobileBiNexArmSimConfig(RobotConfig):
    """Backend-neutral 16-D mobile bimanual NexArm simulation config."""

    backend: str = "mujoco"
    model_path: Path = Path("sim/mobile_nexarm/generated/mobile_bi_nexarm.xml")
    spec_path: Path = Path("sim/mobile_nexarm/spec.json")
    fps: int = 30
    camera_width: int = 640
    camera_height: int = 480
    camera_names: tuple[str, ...] = ("front", "left_wrist")
    seed: int = 0
    settle_steps: int = 50
    first_task_mode: bool = True
    auto_reset: bool = False
    episode_time_s: float = 20.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend != "mujoco":
            raise ValueError("the LeRobot Robot adapter currently supports backend='mujoco'")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps cannot be negative")
        if self.episode_time_s <= 0:
            raise ValueError("episode_time_s must be positive")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names cannot contain duplicates")
        unknown = set(self.camera_names) - {"front", "left_wrist", "right_wrist"}
        if unknown:
            raise ValueError(f"unknown mobile NexArm cameras: {sorted(unknown)}")
