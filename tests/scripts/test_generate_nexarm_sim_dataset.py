# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from examples.nexarm import generate_sim_dataset as generator_module
from examples.nexarm.generate_sim_dataset import (
    ARM_JOINTS,
    EpisodeMetrics,
    FrameRecorder,
    QualityThresholds,
    TrajectoryVariation,
    arm_raw_limits,
    check_quality,
    generate_episode,
    main as generator_main,
)
from lerobot.motors.nexarm.mujoco_mapping import JOINT_MAPPING_VERSION
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


def _connected_robot(**config) -> NexArmSim:
    robot = NexArmSim(
        NexArmSimConfig(id="synthetic_test", model_path=MODEL_PATH, camera_names=(), settle_steps=0, **config)
    )
    robot.connect()
    return robot


def test_varied_trajectories_still_succeed() -> None:
    robot = _connected_robot(enable_domain_randomization=True)
    try:
        task = NexArmPickPlaceTask(robot.backend)
        results = [
            generate_episode(robot, task, seed=seed, variation=TrajectoryVariation.sample(seed))[0]
            for seed in range(5)
        ]
        assert sum(results) >= 4
    finally:
        robot.disconnect()


def test_recorded_actions_are_commands_not_delayed_actions() -> None:
    def first_actions(delay: int) -> np.ndarray:
        robot = _connected_robot(action_delay_steps=delay)
        try:
            frames: list[dict[str, float]] = []
            generate_episode(
                robot,
                NexArmPickPlaceTask(robot.backend),
                seed=0,
                record_frame=lambda _obs, action: frames.append(dict(action)),
            )
            return np.array([[a[f"{n}.pos"] for n in ARM_JOINTS] for a in frames[:10]])
        finally:
            robot.disconnect()

    # The first stage starts from the same home pose, so labels must not depend on the transport delay.
    np.testing.assert_allclose(first_actions(0), first_actions(2))


def test_domain_randomization_is_reproducible_from_seed() -> None:
    robot = _connected_robot(enable_domain_randomization=True)
    try:
        task = NexArmPickPlaceTask(robot.backend)
        task.reset(seed=3)
        first = robot.backend.last_domain_params
        task.reset(seed=4)
        other = robot.backend.last_domain_params
        task.reset(seed=3)
        assert robot.backend.last_domain_params == first
        assert other != first
        assert first["joints"]
    finally:
        robot.disconnect()


def test_quality_gate_rejects_bad_episodes() -> None:
    limits = QualityThresholds()
    good = EpisodeMetrics(
        frames=400,
        max_action_delta=100,
        max_action_jerk=40,
        min_limit_margin=500,
        final_place_error_m=0.005,
        final_cube_speed=0.0,
    )
    assert check_quality(good, limits) is None
    assert check_quality(replace(good, max_action_jerk=1000), limits) == "quality_jerk_spike"
    assert check_quality(replace(good, final_place_error_m=0.04), limits) == "quality_placement_error"
    assert check_quality(replace(good, final_place_error_m=float("nan")), limits) == "quality_placement_error"
    assert check_quality(replace(good, frames=10), limits) == "quality_too_short"


def test_generator_writes_dataset_report_and_episode_log(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    code = generator_main(
        [
            "--repo-id",
            "local/test_nexarm_sim_gen",
            "--root",
            str(root),
            "--episodes",
            "2",
            "--no-video",
            "--camera-width",
            "64",
            "--camera-height",
            "48",
            "--model",
            str(MODEL_PATH),
        ]
    )
    assert code == 0
    report = json.loads((root / "generation_report.json").read_text())
    assert len(report["accepted_seeds"]) == 2
    assert report["validation"]["episodes"] == 2
    assert report["provenance"]["model_sha256"]
    rows = [json.loads(line) for line in (root / "generation_episodes.jsonl").read_text().splitlines()]
    assert sum(row["accepted"] for row in rows) == 2
    assert all(row["dr_params"] for row in rows)


def test_limit_margin_uses_reachable_joint_range() -> None:
    robot = _connected_robot()
    try:
        limits = arm_raw_limits(robot.backend)
        recorder = FrameRecorder(raw_limits=limits)
        command = {f"{name}.pos": 2048.0 for name in ARM_JOINTS}
        command["shoulder_pan.pos"] = limits["shoulder_pan"][0] + 10.0
        recorder({}, command)
        # The pan joint saturates near 512 ticks, not at the 0..4095 command range.
        assert limits["shoulder_pan"][0] > 0
        assert recorder.metrics.min_limit_margin == pytest.approx(10.0)
    finally:
        robot.disconnect()


def test_generator_without_cameras_writes_follower_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        generator_module,
        "NexArmSimConfig",
        lambda **kwargs: NexArmSimConfig(**{**kwargs, "camera_names": ()}),
    )
    root = tmp_path / "dataset"
    code = generator_main(
        ["--repo-id", "local/test_nexarm_sim_gen", "--root", str(root), "--episodes", "1", "--no-video"]
        + ["--model", str(MODEL_PATH)]
    )
    assert code == 0
    assert json.loads((root / "meta" / "info.json").read_text())["robot_type"] == "nexarm_follower"
    report = json.loads((root / "generation_report.json").read_text())
    assert report["domain_randomization"] is True
    assert report["validation"]["episodes"] == 1

    resume_args = ["--repo-id", "local/test_nexarm_sim_gen", "--root", str(root), "--episodes", "1"]
    assert generator_main([*resume_args, "--no-video", "--model", str(MODEL_PATH), "--resume"]) == 0
    report = json.loads((root / "generation_report.json").read_text())
    assert report["joint_mapping"] == JOINT_MAPPING_VERSION
    assert report["validation"]["episodes"] == 2
    assert len(set(report["accepted_seeds"])) == 2
