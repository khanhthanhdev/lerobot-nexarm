#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hardware-free lifecycle tests for NexArmLeader."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from lerobot.motors.nexarm.nexarm import JOINT_NAMES  # noqa: E402
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError  # noqa: E402
from lerobot_teleoperator_nexarm import NexArmLeader, NexArmLeaderConfig  # noqa: E402


def make_bus(positions: list[int] | None = None) -> MagicMock:
    bus = MagicMock(name="NexArmMotorsBus")
    bus.is_connected = False
    bus.read_positions.return_value = list(positions or [2048] * 6)
    bus.connect.side_effect = lambda: setattr(bus, "is_connected", True)
    bus.disconnect.side_effect = lambda: setattr(bus, "is_connected", False)
    return bus


@pytest.fixture
def leader(tmp_path):
    bus = make_bus()
    with patch("lerobot_teleoperator_nexarm.nexarm_leader.NexArmMotorsBus", return_value=bus):
        config = NexArmLeaderConfig(port="/dev/null", id="test_leader", calibration_dir=tmp_path)
        teleop = NexArmLeader(config)
        yield teleop, bus
        if teleop.is_connected:
            teleop.disconnect()


def test_leader_connect_configures_torque_off(leader) -> None:
    teleop, bus = leader
    teleop.connect(calibrate=False)

    assert teleop.is_connected
    bus.connect.assert_called_once()
    bus.set_torque.assert_called_once_with(False)


def test_leader_connection_state_guards(leader) -> None:
    teleop, _ = leader

    with pytest.raises(DeviceNotConnectedError):
        teleop.get_action()

    with pytest.raises(DeviceNotConnectedError):
        teleop.disconnect()

    teleop.connect(calibrate=False)
    with pytest.raises(DeviceAlreadyConnectedError):
        teleop.connect(calibrate=False)


def test_leader_get_action_maps_positions(leader) -> None:
    teleop, bus = leader
    bus.read_positions.return_value = [2048, 1000, 2048, 2048, 2048, 2048]
    teleop.connect(calibrate=False)

    action = teleop.get_action()
    assert isinstance(action, dict)
    assert len(action) == len(JOINT_NAMES)

    # Joint 1 (shoulder_lift): 4096 - 1000 = 3096
    assert action["shoulder_lift.pos"] == 3096.0
    # Joint 5 (gripper): 2833 + (2048 - 2048) * 4 = 2833.0
    assert action["gripper.pos"] == 2833.0


def test_leader_follow_api_inverts_the_follower_mapping(leader) -> None:
    from lerobot.common.control_utils import teleop_supports_feedback
    from lerobot.motors.nexarm.nexarm import (
        GRIPPER_CLOSED_POS,
        GRIPPER_OPEN_POS,
        map_follower_to_leader,
        map_leader_to_follower,
    )

    teleop, bus = leader
    teleop.connect(calibrate=False)
    assert teleop_supports_feedback(teleop)
    assert set(teleop.feedback_features) == {f"{n}.pos" for n in JOINT_NAMES}

    # forward then inverse recovers the leader pose, for every leader gripper position it can produce
    for leader_pose in ([2048, 2048, 1900, 2100, 2048, 2048], [100, 3000, 2048, 1000, 3500, 1700]):
        follower_pose = map_leader_to_follower(leader_pose)
        back = map_follower_to_leader(follower_pose)
        assert back[:5] == leader_pose[:5]
        assert abs(back[5] - leader_pose[5]) <= 1
    # the full follower gripper range lands inside the leader's reachable range
    assert map_follower_to_leader([2048] * 5 + [GRIPPER_CLOSED_POS])[5] == 2048
    assert map_follower_to_leader([2048] * 5 + [GRIPPER_OPEN_POS])[5] == 1638

    bus.reset_mock()
    teleop.enable_torque()
    teleop.send_feedback(
        {f"{n}.pos": 2048.0 for n in JOINT_NAMES} | {"shoulder_lift.pos": 1500.0, "gripper.pos": 2833.0}
    )
    teleop.disable_torque()
    bus.set_torque.assert_any_call(True)
    bus.set_torque.assert_any_call(False)
    written = bus.write_positions.call_args.args[0]
    assert written == [2048, 4096 - 1500, 2048, 2048, 2048, 2048]


def test_leader_feedback_honours_a_stored_offset(leader) -> None:
    from lerobot.motors import MotorCalibration

    teleop, bus = leader
    teleop.connect(calibrate=False)
    teleop.calibration = {
        "elbow_flex": MotorCalibration(
            id=3, drive_mode=0, homing_offset=2048 + 50, range_min=0, range_max=4095
        )
    }
    teleop.send_feedback({f"{n}.pos": 2048.0 for n in JOINT_NAMES})
    # get_action adds the offset, so feedback must subtract it to round-trip
    assert bus.write_positions.call_args.args[0][2] == 2048 - 50
