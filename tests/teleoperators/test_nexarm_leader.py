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
from lerobot.teleoperators.nexarm_leader import NexArmLeader, NexArmLeaderConfig  # noqa: E402
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError  # noqa: E402


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
    with patch("lerobot.teleoperators.nexarm_leader.nexarm_leader.NexArmMotorsBus", return_value=bus):
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
