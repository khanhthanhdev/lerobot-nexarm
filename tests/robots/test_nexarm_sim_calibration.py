# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from examples.nexarm.calibrate_sim import Episode, Simulator, fit_calibration, rmse
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
