# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Builds the real NexArm leader, follower and raw buses from a robot profile.

Sessions and the calibration check get this object injected, so tests substitute fakes without
touching serial ports.
"""

from typing import Any

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors.nexarm import NexArmMotorsBus
from lerobot.nexarm_service.profiles import CameraProfile, RobotProfile
from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig
from lerobot.teleoperators.nexarm_leader import NexArmLeader, NexArmLeaderConfig


def opencv_camera_config(cam: CameraProfile) -> OpenCVCameraConfig:
    source: int | str = int(cam.source) if cam.source.isdigit() else cam.source
    return OpenCVCameraConfig(index_or_path=source, width=cam.width, height=cam.height, fps=cam.fps)


def follower_config(profile: RobotProfile, with_cameras: bool = False) -> NexArmFollowerConfig:
    return NexArmFollowerConfig(
        id=profile.name,
        port=profile.follower_port,
        baudrate=profile.baudrate,
        # The session owns idle safety (lease + watchdog); the driver's auto-relax must not cut torque
        # mid-session, and rest return plus torque-off is done by the worker.
        idle_timeout_s=0.0,
        disable_torque_on_disconnect=False,
        cameras={k: opencv_camera_config(c) for k, c in profile.cameras.items()} if with_cameras else {},
    )


def leader_config(profile: RobotProfile) -> NexArmLeaderConfig:
    if profile.leader_port is None:
        raise ValueError(f"profile {profile.name!r} has no leader_port")
    return NexArmLeaderConfig(id=profile.name, port=profile.leader_port, baudrate=profile.baudrate)


class Hardware:
    def make_follower(self, profile: RobotProfile, with_cameras: bool = False) -> Any:
        return NexArmFollower(follower_config(profile, with_cameras))

    def make_leader(self, profile: RobotProfile) -> Any:
        return NexArmLeader(leader_config(profile))

    def make_bus(self, port: str, baudrate: int) -> NexArmMotorsBus:
        return NexArmMotorsBus(port=port, baudrate=baudrate)
