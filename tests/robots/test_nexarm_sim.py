# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim import (
    NexArmPickPlaceTask,
    NexArmSim,
    NexArmSimConfig,
)
from lerobot.robots.nexarm_sim.mujoco_backend import HOME_POSITIONS, RAW_RANGES, NexArmMujocoBackend

MODEL_PATH = Path(__file__).resolve().parents[2] / "sim" / "fusion_export" / "scene.xml"


@pytest.fixture
def backend() -> NexArmMujocoBackend:
    backend = NexArmMujocoBackend(
        model_path=MODEL_PATH,
        fps=30,
        camera_width=160,
        camera_height=120,
        camera_names=("front", "wrist"),
    )
    backend.reset()
    yield backend
    backend.close()


def test_raw_control_round_trip(backend: NexArmMujocoBackend) -> None:
    for name in JOINT_NAMES:
        low, high = RAW_RANGES[name]
        for raw in (low, (low + high) / 2, high):
            actual = backend.control_to_raw(name, backend.raw_to_control(name, raw))
            assert actual == pytest.approx(raw)


def test_backend_steps_and_renders(backend: NexArmMujocoBackend) -> None:
    action = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
    action["shoulder_pan.pos"] = 3072.0
    sent = backend.step(action)

    assert sent == action
    assert backend.data.time > 0
    for camera_name in ("front", "wrist"):
        image = backend.render(camera_name)
        assert image.shape == (120, 160, 3)
        assert image.dtype == np.uint8
        assert image.std() > 0


def test_gripper_collision_geometry_contacts_cube(backend: NexArmMujocoBackend) -> None:
    model = backend.model
    data = backend.data
    action = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
    for _ in range(60):
        backend.step(action)

    jaw_center = _jaw_center(backend)
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qpos_address = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos_address : cube_qpos_address + 7] = [
        *jaw_center,
        1,
        0,
        0,
        0,
    ]
    mujoco.mj_forward(model, data)

    for _ in range(10):
        backend.step(action)

    cube_contact_geoms: set[str] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
        }
        if "cube_collision" in geom_names:
            cube_contact_geoms.update(geom_names)

    assert "link_6_left_jaw_collision_0" in cube_contact_geoms
    assert "link_6_right_jaw_collision_0" in cube_contact_geoms


def test_gripper_opening_increases_jaw_separation(backend: NexArmMujocoBackend) -> None:
    model = backend.model
    left_geom = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "link_6_left_jaw_collision_0",
    )
    right_geom = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "link_6_right_jaw_collision_0",
    )
    action = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}

    for _ in range(60):
        backend.step(action)
    closed_separation = abs(backend.data.geom_xpos[right_geom, 0] - backend.data.geom_xpos[left_geom, 0])

    action["gripper.pos"] = RAW_RANGES["gripper"][0]
    for _ in range(60):
        backend.step(action)
    open_separation = abs(backend.data.geom_xpos[right_geom, 0] - backend.data.geom_xpos[left_geom, 0])

    assert open_separation > closed_separation
    assert open_separation - closed_separation == pytest.approx(0.051, abs=0.002)


def test_robot_defaults_and_collision_masks_are_active(backend: NexArmMujocoBackend) -> None:
    model = backend.model
    arm_joint_names = (
        "joint_1_base_to_link_1",
        "joint_2_link_1_to_link_2",
        "joint_3_link_2_to_link_3",
        "joint_4_link_3_to_link_4",
        "joint_5_link_4_to_link_5",
    )
    arm_dofs = [
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in arm_joint_names
    ]
    assert np.all(model.dof_damping[arm_dofs] > 0)
    assert np.all(model.dof_frictionloss[arm_dofs] > 0)
    assert np.all(model.dof_armature[arm_dofs] > 0)

    left_jaw = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "link_6_left_jaw_collision_0",
    )
    right_jaw = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "link_6_right_jaw_collision_0",
    )
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_collision")
    assert model.geom_contype[left_jaw] & model.geom_conaffinity[right_jaw]
    assert model.geom_contype[left_jaw] & model.geom_conaffinity[cube]


def test_sim_robot_matches_physical_feature_contract() -> None:
    robot = NexArmSim(
        NexArmSimConfig(
            id="test",
            model_path=MODEL_PATH,
            camera_width=160,
            camera_height=120,
            settle_steps=0,
        )
    )
    expected_joint_keys = {f"{name}.pos" for name in JOINT_NAMES}
    assert set(robot.action_features) == expected_joint_keys
    assert set(robot.observation_features) == expected_joint_keys | {"front", "wrist"}

    robot.connect()
    try:
        observation = robot.get_observation()
        assert set(observation) == expected_joint_keys | {"front", "wrist"}

        action = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
        action["gripper.pos"] = 1000.0
        sent = robot.send_action(action)
        assert sent["gripper.pos"] == RAW_RANGES["gripper"][0]
    finally:
        robot.disconnect()

    assert not robot.is_connected


def _set_cube_position(backend: NexArmMujocoBackend, xyz: tuple[float, float, float]) -> None:
    cube_joint_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    qpos_address = backend.model.jnt_qposadr[cube_joint_id]
    backend.data.qpos[qpos_address : qpos_address + 7] = [*xyz, 1, 0, 0, 0]
    dof_address = backend.model.jnt_dofadr[cube_joint_id]
    backend.data.qvel[dof_address : dof_address + 6] = 0
    mujoco.mj_forward(backend.model, backend.data)


def _jaw_center(backend: NexArmMujocoBackend) -> tuple[float, float, float]:
    positions = []
    for name in ("link_6_left_jaw_collision_0", "link_6_right_jaw_collision_0"):
        geom_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        positions.append(backend.data.geom_xpos[geom_id])
    return tuple(np.mean(positions, axis=0))


def test_pick_place_task_reset_is_seeded(backend: NexArmMujocoBackend) -> None:
    task = NexArmPickPlaceTask(backend)
    task.reset(seed=17)
    first_cube = task.cube_position
    first_target = task.target_position

    task.reset(seed=17)
    assert task.cube_position == pytest.approx(first_cube)
    assert task.target_position == pytest.approx(first_target)
    assert np.linalg.norm(first_cube[:2] - first_target[:2]) >= task.target_radius_m + 0.04


def test_pick_place_task_accepts_stable_released_placement(
    backend: NexArmMujocoBackend,
) -> None:
    task = NexArmPickPlaceTask(backend)
    task.reset(seed=2)
    target = task.target_position
    _set_cube_position(backend, (float(target[0]), float(target[1]), 0.01))
    action = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
    action["gripper.pos"] = RAW_RANGES["gripper"][0]
    for _ in range(60):
        backend.step(action)

    status = task.observe()
    for _ in range(25):
        status = task.step()
        if status.success:
            break

    assert status.success
    assert status.reason == "success"
    assert status.is_inside_target
    assert status.is_released
    assert status.hold_time_s >= 0.5


def test_pick_place_task_rejects_outside_unreleased_drop_and_timeout(
    backend: NexArmMujocoBackend,
) -> None:
    task = NexArmPickPlaceTask(backend, timeout_s=1)

    task.reset(seed=3)
    target = task.target_position
    _set_cube_position(backend, (float(target[0] + 0.08), float(target[1]), 0.018))
    assert not task.observe().is_inside_target

    task.reset(seed=4)
    target = task.target_position
    _set_cube_position(backend, (float(target[0]), float(target[1]), 0.018))
    backend.data.time = 1
    status = task.observe()
    assert status.terminated
    assert status.reason == "unreleased_grasp"

    task.reset(seed=5)
    _set_cube_position(backend, (0.53, -0.17, -0.03))
    status = task.observe()
    assert status.terminated
    assert status.reason == "dropped_cube"

    task.reset(seed=6)
    backend.data.time = 1
    status = task.observe()
    assert status.terminated
    assert status.reason == "timeout"


def test_pick_place_task_reports_two_jaw_grasp(backend: NexArmMujocoBackend) -> None:
    task = NexArmPickPlaceTask(backend)
    task.reset(seed=7)
    action = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
    for _ in range(60):
        backend.step(action)
    _set_cube_position(backend, _jaw_center(backend))
    for _ in range(10):
        backend.step(action)

    assert task.status().is_grasped
