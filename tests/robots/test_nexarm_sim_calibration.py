# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples.nexarm.calibrate_sim import Episode, Simulator, fit_calibration, rmse
from lerobot.motors.nexarm.mujoco_mapping import JOINT_MAPPING_VERSION
from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim import (
    JointCalibration,
    NexArmSim,
    NexArmSimConfig,
    SimCalibration,
)

MODEL_PATH = Path(__file__).resolve().parents[2] / "sim" / "fusion_export" / "scene.xml"


def test_calibration_round_trip(tmp_path: Path) -> None:
    calibration = SimCalibration(
        action_delay_steps=3,
        joints={"shoulder_lift": JointCalibration(kp_scale=0.5, damping_spread=0.2)},
        source={"repo_id": "x"},
    )
    path = tmp_path / "calibration.json"
    calibration.save(path)
    assert SimCalibration.load(path) == calibration


def test_calibration_validation() -> None:
    with pytest.raises(ValueError):
        JointCalibration(kp_scale=0.0)
    with pytest.raises(ValueError):
        SimCalibration(action_delay_steps=-1)
    with pytest.raises(ValueError):
        SimCalibration(joints={"not_a_joint": JointCalibration()})


def test_apply_calibration_scales_model_and_recenters_dr(tmp_path: Path) -> None:
    calibration = SimCalibration(
        action_delay_steps=2,
        joints={"shoulder_lift": JointCalibration(kp_scale=2.0, damping_scale=1.5, damping_spread=0.1)},
    )
    path = tmp_path / "calibration.json"
    calibration.save(path)

    plain = NexArmSim(NexArmSimConfig(id="a", model_path=MODEL_PATH, camera_names=(), settle_steps=0))
    tuned = NexArmSim(
        NexArmSimConfig(id="b", model_path=MODEL_PATH, camera_names=(), settle_steps=0, calibration_path=path)
    )
    plain.connect()
    tuned.connect()
    try:
        joint = plain.backend._joint_ids["shoulder_lift"]
        dof = int(plain.backend.model.jnt_dofadr[joint])
        actuator = plain.backend._actuator_ids["shoulder_lift"]
        assert tuned.backend.model.dof_damping[dof] == pytest.approx(
            1.5 * plain.backend.model.dof_damping[dof]
        )
        assert tuned.backend.model.actuator_gainprm[actuator, 0] == pytest.approx(
            2.0 * plain.backend.model.actuator_gainprm[actuator, 0]
        )
        assert tuned.backend.action_delay_steps == 2
        assert tuned.backend._nominal_dof_damping[dof] == pytest.approx(tuned.backend.model.dof_damping[dof])
        assert tuned.backend.dr_ranges.joint_damping_scale["shoulder_lift"] == pytest.approx((0.9, 1.1))
    finally:
        plain.disconnect()
        tuned.disconnect()


def _synthetic_episode(sim: Simulator, truth: SimCalibration, seed: int, frames: int = 80) -> Episode:
    rng = np.random.default_rng(seed)
    t = np.arange(frames)
    actions = np.tile([2048.0] * 5 + [2833.0], (frames, 1))
    for column in range(5):
        actions[:, column] += rng.uniform(150, 350) * np.sin(
            2 * np.pi * t / rng.uniform(30, 60) + rng.uniform(0, 6)
        )
    episode = Episode(states=np.zeros_like(actions), actions=actions)
    episode.states[0] = actions[0]
    sim.configure(truth)
    episode.states[1:] = sim.rollout(episode)
    return episode


def test_fit_recovers_known_delay_and_gain() -> None:
    sim = Simulator(MODEL_PATH, fps=30)
    truth = SimCalibration(action_delay_steps=2, joints={"shoulder_lift": JointCalibration(kp_scale=0.5)})
    episodes = [_synthetic_episode(sim, truth, seed) for seed in range(2)]

    fitted = fit_calibration(sim, episodes, ["shoulder_lift"], range(0, 5))

    assert fitted.action_delay_steps == 2
    assert 0.4 <= fitted.joints["shoulder_lift"].kp_scale <= 0.6
    column = JOINT_NAMES.index("shoulder_lift")
    before = rmse(sim, episodes, SimCalibration())[column]
    after = rmse(sim, episodes, fitted)[column]
    assert after < 0.2 * before


def _sim(path: Path | None = None, **overrides) -> NexArmSim:
    config = NexArmSimConfig(
        id="cal", model_path=MODEL_PATH, camera_names=(), settle_steps=0, calibration_path=path, **overrides
    )
    return NexArmSim(config)


def test_calibration_with_other_fps_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    SimCalibration(action_delay_steps=1, fps=15).save(path)
    robot = _sim(path)
    with pytest.raises(ValueError, match="15 fps"):
        robot.connect()
    assert not robot.is_connected


def test_action_delay_precedence_config_over_calibration_over_default(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    SimCalibration(action_delay_steps=3, fps=30).save(path)
    for robot, expected in (
        (_sim(path), 3),
        (_sim(path, action_delay_steps=1), 1),
        (_sim(path, action_delay_steps=0), 0),
        (_sim(None), 0),
        (_sim(None, action_delay_steps=2), 2),
    ):
        robot.connect()
        try:
            assert robot.backend.action_delay_steps == expected
        finally:
            robot.disconnect()


def test_apply_calibration_only_recenters_dr_of_fitted_joints(tmp_path: Path) -> None:
    calibration = SimCalibration(
        joints={name: JointCalibration(damping_spread=0.1, frictionloss_spread=0.1) for name in JOINT_NAMES},
        fitted_joints=["shoulder_lift"],
    )
    path = tmp_path / "calibration.json"
    calibration.save(path)
    plain, tuned = _sim(None), _sim(path)
    plain.connect()
    tuned.connect()
    try:
        assert tuned.backend.dr_ranges.joint_damping_scale["shoulder_lift"] == pytest.approx((0.9, 1.1))
        for name in JOINT_NAMES:
            if name == "shoulder_lift":
                continue
            assert (
                tuned.backend.dr_ranges.joint_damping_scale[name]
                == plain.backend.dr_ranges.joint_damping_scale[name]
            )
            assert (
                tuned.backend.dr_ranges.joint_friction_scale[name]
                == plain.backend.dr_ranges.joint_friction_scale[name]
            )
    finally:
        plain.disconnect()
        tuned.disconnect()


def test_older_calibration_files_read_fps_and_fitted_joints_from_source(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "action_delay_steps": 2,
                "joints": {name: {} for name in JOINT_NAMES},
                "source": {"fps": 30, "fitted_joints": ["shoulder_pan", "elbow_flex"]},
                "joint_mapping": JOINT_MAPPING_VERSION,
            }
        ),
        encoding="utf-8",
    )
    calibration = SimCalibration.load(path)
    assert calibration.fps == 30
    assert calibration.fitted_joint_names() == ["shoulder_pan", "elbow_flex"]


@pytest.mark.parametrize("joint_mapping", [None, "per_joint_ctrlrange_v0"])
def test_calibration_fitted_with_another_joint_mapping_is_refused(tmp_path: Path, joint_mapping) -> None:
    payload = {"action_delay_steps": 1, "joints": {name: {} for name in JOINT_NAMES}}
    if joint_mapping is not None:
        payload["joint_mapping"] = joint_mapping
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="joint mapping"):
        SimCalibration.load(path)


def test_saved_calibration_records_the_joint_mapping(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    SimCalibration().save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["joint_mapping"] == JOINT_MAPPING_VERSION
