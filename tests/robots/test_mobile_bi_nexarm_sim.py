# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from lerobot.robots.mobile_bi_nexarm_sim import (
    ARM_FEATURES,
    STATE_ACTION_NAMES,
    MobileBiNexArmSim,
    MobileBiNexArmSimConfig,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "sim/mobile_nexarm/generated/mobile_bi_nexarm.xml"
URDF = ROOT / "sim/mobile_nexarm/generated/mobile_bi_nexarm.urdf"
USD = ROOT / "sim/mobile_nexarm/generated/mobile_bi_nexarm.usda"


@pytest.fixture
def robot() -> MobileBiNexArmSim:
    value = MobileBiNexArmSim(
        MobileBiNexArmSimConfig(
            id="test",
            model_path=MODEL,
            camera_width=64,
            camera_height=48,
            settle_steps=0,
            episode_time_s=1,
        )
    )
    value.connect()
    yield value
    value.disconnect()


def test_canonical_contract_order_and_round_trip() -> None:
    contract = load_contract()
    assert len(STATE_ACTION_NAMES) == 16
    assert STATE_ACTION_NAMES[:6] == tuple(f"left_{name}.pos" for name in ARM_FEATURES)
    assert STATE_ACTION_NAMES[12:] == (
        "x.vel",
        "y.vel",
        "theta.vel",
        "lift_axis.height_mm",
    )
    for joint in contract.joints:
        for raw in (joint.raw_range[0], joint.home_raw, joint.raw_range[1]):
            assert joint.sim_to_raw(joint.raw_to_sim(raw)) == pytest.approx(raw)
    home_fk = contract.forward_kinematics(
        "left", {joint.feature: joint.home_raw for joint in contract.joints}
    )
    assert np.isfinite(home_fk).all()
    assert home_fk.shape == (4, 4)


def test_generated_formats_share_joint_limits_axes_mounts_and_mimic() -> None:
    contract = load_contract()
    mjcf = ET.parse(MODEL).getroot()
    urdf = ET.parse(URDF).getroot()
    for side in ("left", "right"):
        mjcf_mount = mjcf.find(f".//body[@name='{side}_arm_mount']")
        urdf_mount = urdf.find(f".//joint[@name='{side}_arm_mount_joint']/origin")
        assert mjcf_mount is not None and urdf_mount is not None
        expected_mount = contract.payload["arms"][side]["mount_xyz_m"]
        assert [float(value) for value in mjcf_mount.attrib["pos"].split()] == pytest.approx(expected_mount)
        assert [float(value) for value in urdf_mount.attrib["xyz"].split()] == pytest.approx(expected_mount)
        for joint in contract.joints:
            name = f"{side}_{joint.feature}"
            mjcf_joint = mjcf.find(f".//joint[@name='{name}']")
            urdf_joint = urdf.find(f".//joint[@name='{name}']")
            assert mjcf_joint is not None and urdf_joint is not None
            assert [float(value) for value in mjcf_joint.attrib["axis"].split()] == pytest.approx(joint.axis)
            assert [float(value) for value in mjcf_joint.attrib["range"].split()] == pytest.approx(
                joint.sim_range
            )
            assert [
                float(urdf_joint.find("limit").attrib[key]) for key in ("lower", "upper")
            ] == pytest.approx(joint.sim_range)
    equalities = {item.attrib["name"] for item in mjcf.findall(".//equality/joint")}
    assert equalities == {
        "left_gripper_mimic_constraint",
        "right_gripper_mimic_constraint",
    }
    usd = USD.read_text(encoding="utf-8")
    for name in STATE_ACTION_NAMES[:12]:
        assert f'"{name.removesuffix(".pos")}"' in usd


def test_seeded_reset_and_first_task_clamps(robot: MobileBiNexArmSim) -> None:
    robot.reset(seed=41)
    cube_id = mujoco.mj_name2id(robot.backend.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    first = robot.backend.data.xpos[cube_id].copy()
    robot.reset(seed=41)
    assert robot.backend.data.xpos[cube_id] == pytest.approx(first)
    action = robot.contract.home_action | {
        "right_shoulder_pan.pos": 4095,
        "x.vel": 0.4,
        "lift_axis.height_mm": 0,
    }
    sent = robot.send_action(action)
    assert sent["right_shoulder_pan.pos"] == robot.contract.home_action["right_shoulder_pan.pos"]
    assert sent["x.vel"] == 0
    assert sent["lift_axis.height_mm"] == robot.contract.payload["lift"]["home_mm"]


def test_body_frame_base_velocity_when_mobile() -> None:
    robot = MobileBiNexArmSim(
        MobileBiNexArmSimConfig(
            id="mobile",
            model_path=MODEL,
            settle_steps=0,
            first_task_mode=False,
            camera_names=(),
        )
    )
    robot.connect()
    try:
        yaw_address = robot.backend._qpos_address("base_yaw")
        robot.backend.data.qpos[yaw_address] = np.pi / 2
        x_before = robot.backend.data.qpos[robot.backend._qpos_address("base_x")]
        y_before = robot.backend.data.qpos[robot.backend._qpos_address("base_y")]
        action = robot.contract.home_action | {"x.vel": 0.3}
        robot.send_action(action)
        x_delta = robot.backend.data.qpos[robot.backend._qpos_address("base_x")] - x_before
        y_delta = robot.backend.data.qpos[robot.backend._qpos_address("base_y")] - y_before
        assert abs(y_delta) > abs(x_delta)
        assert y_delta > 0
        assert robot.backend.joint_state()["x.vel"] == pytest.approx(0.3)
    finally:
        robot.disconnect()


def test_task_success_drop_and_timeout(robot: MobileBiNexArmSim) -> None:
    target = robot.backend.data.xpos[robot.backend._body_ids["target_zone"]].copy()
    cube_qpos = robot.backend._qpos_address("cube_joint")
    robot.backend.data.qpos[cube_qpos : cube_qpos + 3] = [*target[:2], 0.50]
    robot.backend.data.qpos[cube_qpos + 3 : cube_qpos + 7] = [1, 0, 0, 0]
    robot.backend._last_action["left_gripper.pos"] = 1195
    mujoco.mj_forward(robot.backend.model, robot.backend.data)
    for _ in range(round(0.5 * robot.backend.fps)):
        robot.backend._update_task_status()
    assert robot.task_status().success

    robot.reset(seed=2)
    robot.backend.data.qpos[cube_qpos + 2] = 0.01
    mujoco.mj_forward(robot.backend.model, robot.backend.data)
    robot.backend._update_task_status()
    assert robot.task_status().reason == "dropped_cube"

    robot.reset(seed=3)
    robot.backend.data.time = robot.backend.episode_time_s
    robot.backend._update_task_status()
    assert robot.task_status().reason == "timeout"
