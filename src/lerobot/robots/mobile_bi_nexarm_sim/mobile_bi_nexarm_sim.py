# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from functools import cached_property

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_mobile_bi_nexarm_sim import MobileBiNexArmSimConfig
from .contract import load_contract
from .mujoco_backend import MobileBiNexArmMujocoBackend, TaskStatus


class MobileBiNexArmSim(Robot):
    """Direct LeRobot Robot adapter for the mobile bimanual simulation."""

    config_class = MobileBiNexArmSimConfig
    name = "mobile_bi_nexarm_sim"

    def __init__(self, config: MobileBiNexArmSimConfig):
        super().__init__(config)
        self.config = config
        self.contract = load_contract(config.spec_path)
        self._backend: MobileBiNexArmMujocoBackend | None = None
        self._next_seed = config.seed

    @property
    def backend(self) -> MobileBiNexArmMujocoBackend:
        if self._backend is None:
            raise ConnectionError("mobile bimanual NexArm simulation is not connected")
        return self._backend

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = dict.fromkeys(self.contract.feature_names, float)
        features.update(
            dict.fromkeys(
                self.config.camera_names,
                (self.config.camera_height, self.config.camera_width, 3),
            )
        )
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self.contract.feature_names, float)

    @property
    def is_connected(self) -> bool:
        return self._backend is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._backend = MobileBiNexArmMujocoBackend(
            model_path=self.config.model_path,
            spec_path=self.config.spec_path,
            fps=self.config.fps,
            camera_width=self.config.camera_width,
            camera_height=self.config.camera_height,
            camera_names=self.config.camera_names,
            seed=self.config.seed,
            first_task_mode=self.config.first_task_mode,
            episode_time_s=self.config.episode_time_s,
        )
        self.reset(seed=self.config.seed)

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation: RobotObservation = self.backend.joint_state()
        for camera_name in self.config.camera_names:
            observation[camera_name] = self.backend.render(camera_name)
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        status = self.backend.task_status()
        if status.terminated and self.config.auto_reset:
            self.reset()
        return self.backend.step(action)

    @check_if_not_connected
    def reset(self, seed: int | None = None) -> None:
        episode_seed = self._next_seed if seed is None else seed
        self.backend.reset(seed=episode_seed, settle_steps=self.config.settle_steps)
        self._next_seed = episode_seed + 1

    @check_if_not_connected
    def task_status(self) -> TaskStatus:
        return self.backend.task_status()

    @check_if_not_connected
    def disconnect(self) -> None:
        self.backend.close()
        self._backend = None
