# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim import NexArmSim, NexArmSimConfig
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
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qpos_address = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos_address : cube_qpos_address + 7] = [
        0.53937,
        -0.03991,
        0.2305,
        1,
        0,
        0,
        0,
    ]
    mujoco.mj_forward(model, data)

    action = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
    for _ in range(60):
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
