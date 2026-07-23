# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.nexarm.generate_sim_dataset import generate_episode
from lerobot.robots.nexarm_sim import NexArmPickPlaceTask, NexArmSim, NexArmSimConfig

MODEL_PATH = Path(__file__).resolve().parents[2] / "sim" / "fusion_export" / "scene.xml"


def test_scripted_episode_passes_physical_success_gate() -> None:
    robot = NexArmSim(
        NexArmSimConfig(
            id="synthetic_test",
            model_path=MODEL_PATH,
            camera_names=(),
            settle_steps=0,
        )
    )
    robot.connect()
    try:
        task = NexArmPickPlaceTask(robot.backend)
        success, reason = generate_episode(robot, task, seed=0)

        assert success
        assert reason == "success"
        assert np.linalg.norm(task.cube_position[:2] - task.target_position[:2]) <= task.target_radius_m
    finally:
        robot.disconnect()
