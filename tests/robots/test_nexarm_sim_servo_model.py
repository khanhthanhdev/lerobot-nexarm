# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""The sim servo model: setpoint speed and acceleration caps, gripper stroke, holding stiffness, scripted tasks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lerobot_robot_nexarm.motors.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS, JOINT_NAMES
from lerobot_robot_nexarm.nexarm_sim import NexArmPickPlaceTask, NexArmSimConfig
from lerobot_robot_nexarm.nexarm_sim.mujoco_backend import (
    SERVO_MAX_ACCEL_TICKS_S2,
    SERVO_MAX_SPEED_TICKS_S,
    NexArmMujocoBackend,
)

SCENE = Path(__file__).resolve().parents[2] / "sim" / "description" / "mjcf" / "scene.xml"


@pytest.fixture
def backend() -> NexArmMujocoBackend:
    backend = NexArmMujocoBackend(
        model_path=SCENE, fps=30, camera_width=64, camera_height=48, camera_names=()
    )
    backend.reset()
    yield backend
    backend.close()


def _sustained_speed(
    backend: NexArmMujocoBackend, start: dict[str, float], target: dict[str, float], key: str
):
    """Best 200 ms average speed (ticks/s) after stepping from ``start`` to ``target``."""
    for _ in range(40):
        backend.step(start)
    trace = [backend.joint_positions()[key]]
    for _ in range(2 * backend.fps):
        backend.step(target)
        trace.append(backend.joint_positions()[key])
    window = round(0.2 * backend.fps)
    return max(abs(trace[i + window] - trace[i]) for i in range(len(trace) - window)) / (window / backend.fps)


@pytest.mark.parametrize("name", ["shoulder_pan", "wrist_roll"])
def test_unloaded_joint_moves_at_its_speed_cap(backend: NexArmMujocoBackend, name: str) -> None:
    start = dict(backend.joint_positions())
    target = dict(start)
    target[f"{name}.pos"] += 1200.0
    cap = SERVO_MAX_SPEED_TICKS_S[JOINT_NAMES.index(name)]
    assert _sustained_speed(backend, start, target, f"{name}.pos") == pytest.approx(cap, rel=0.10)


def test_gravity_loaded_joint_never_exceeds_its_speed_cap(backend: NexArmMujocoBackend) -> None:
    start = dict(backend.joint_positions())
    target = dict(start)
    target["elbow_flex.pos"] -= 1200.0
    cap = SERVO_MAX_SPEED_TICKS_S[JOINT_NAMES.index("elbow_flex")]
    assert _sustained_speed(backend, start, target, "elbow_flex.pos") <= cap * 1.02


@pytest.mark.parametrize(
    ("start", "end"), [(GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS), (GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS)]
)
def test_full_gripper_stroke_takes_the_real_duration(
    backend: NexArmMujocoBackend, start: float, end: float
) -> None:
    pose = dict(backend.joint_positions())
    pose["gripper.pos"] = start
    for _ in range(30):
        backend.step(pose)
    origin = backend.joint_positions()["gripper.pos"]
    pose["gripper.pos"] = end
    frames = 0
    while frames < 60 and abs(backend.joint_positions()["gripper.pos"] - origin) < 0.95 * abs(end - origin):
        backend.step(pose)
        frames += 1
    assert 0.35 <= frames / backend.fps <= 0.5


def test_setpoint_acceleration_respects_the_cap(backend: NexArmMujocoBackend) -> None:
    pose = dict(backend.joint_positions())
    pose["shoulder_pan.pos"] += 1200.0
    pose["gripper.pos"] = GRIPPER_OPEN_POS
    dt = backend.model.opt.timestep
    per_tick = backend._servo_max_velocity / np.asarray(SERVO_MAX_SPEED_TICKS_S)
    backend.set_action(pose)
    previous = backend._servo_velocity.copy()
    jumps = []
    for _ in range(backend.steps_per_action * 10):
        backend._advance_servo_setpoint(dt)
        jumps.append(np.abs(backend._servo_velocity - previous) / dt / per_tick)
        previous = backend._servo_velocity.copy()
    peak = np.max(jumps, axis=0)
    pan = JOINT_NAMES.index("shoulder_pan")
    assert peak[pan] <= SERVO_MAX_ACCEL_TICKS_S2[pan] * (1 + 1e-9)
    # The gripper is uncapped: its setpoint reaches full speed in one substep.
    assert SERVO_MAX_ACCEL_TICKS_S2[-1] is None
    assert peak[-1] > 10 * SERVO_MAX_ACCEL_TICKS_S2[pan]


def test_the_arm_holds_grasp_poses_within_2_mrad(backend: NexArmMujocoBackend) -> None:
    task = NexArmPickPlaceTask(backend)
    worst = 0.0
    for seed in range(3):
        task.reset(seed=seed, settle_steps=25)
        solution = backend.solve_ik(backend.body_position("cube") + np.array([0.0, 0.0, 0.1]), seed=seed)
        assert solution is not None
        command = dict(backend.joint_positions())
        command.update({f"{name}.pos": value for name, value in solution.items()})
        for _ in range(int(1.5 * backend.fps)):
            backend.step(command)
        now = backend.joint_positions()
        error = max(abs(now[f"{n}.pos"] - command[f"{n}.pos"]) for n in JOINT_NAMES[:-1])
        worst = max(worst, error * 2 * np.pi / 4096 * 1000)
    assert worst < 2.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_speed_ticks_s": (3413.0,) * 5},
        {"max_speed_ticks_s": (3413.0,) * 5 + (0.0,)},
        {"max_accel_ticks_s2": (20000.0,) * 5},
        {"max_accel_ticks_s2": (20000.0,) * 5 + (-1.0,)},
    ],
)
def test_invalid_servo_caps_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        NexArmSimConfig(id="test", **kwargs)
    with pytest.raises(ValueError):
        NexArmMujocoBackend(
            model_path=SCENE, fps=30, camera_width=64, camera_height=48, camera_names=(), **kwargs
        )


def test_accel_cap_can_be_turned_off() -> None:
    backend = NexArmMujocoBackend(
        model_path=SCENE, fps=30, camera_width=64, camera_height=48, camera_names=(), max_accel_ticks_s2=None
    )
    try:
        assert backend.max_accel_ticks_s2 == (None,) * len(JOINT_NAMES)
    finally:
        backend.close()
