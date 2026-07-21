#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hardware-free lifecycle and safety tests for NexArmFollower."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from lerobot.motors.nexarm.nexarm import JOINT_NAMES  # noqa: E402
from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig  # noqa: E402
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError  # noqa: E402


def make_bus(positions: list[int] | None = None) -> MagicMock:
    bus = MagicMock(name="NexArmMotorsBus")
    bus.is_connected = False
    bus.read_positions.return_value = list(positions or [2048] * 6)
    bus.connect.side_effect = lambda: setattr(bus, "is_connected", True)
    bus.disconnect.side_effect = lambda: setattr(bus, "is_connected", False)
    return bus


@pytest.fixture
def follower(tmp_path):
    bus = make_bus()
    with (
        patch("lerobot.robots.nexarm_follower.nexarm_follower.NexArmMotorsBus", return_value=bus),
        patch("lerobot.robots.nexarm_follower.nexarm_follower.time.sleep"),
    ):
        config = NexArmFollowerConfig(port="/dev/null", id="test", calibration_dir=tmp_path)
        robot = NexArmFollower(config)
        yield robot, bus
        if robot.is_connected:
            robot.disconnect()


def connect(robot: NexArmFollower) -> None:
    robot.connect(calibrate=False)


def test_connect_configures_bridge_motion_and_torque(follower) -> None:
    robot, bus = follower
    connect(robot)

    assert robot.is_connected
    bus.connect.assert_called_once()
    bus.enter_lerobot_mode.assert_called_once()
    bus.write_motion_params.assert_called_once_with(acc=100, speed=2000)
    bus.set_torque.assert_called_once_with(True)


def test_connection_state_guards(follower) -> None:
    robot, _ = follower
    action = {f"{name}.pos": 2048.0 for name in JOINT_NAMES}

    with pytest.raises(DeviceNotConnectedError):
        robot.get_observation()
    with pytest.raises(DeviceNotConnectedError):
        robot.send_action(action)
    with pytest.raises(DeviceNotConnectedError):
        robot.disconnect()

    connect(robot)
    with pytest.raises(DeviceAlreadyConnectedError):
        connect(robot)


def test_observation_returns_float_joint_values(follower) -> None:
    robot, bus = follower
    bus.read_positions.return_value = [100, 200, 300, 400, 500, 600]
    connect(robot)

    observation = robot.get_observation()

    assert observation == {
        f"{name}.pos": float(position)
        for name, position in zip(JOINT_NAMES, [100, 200, 300, 400, 500, 600], strict=True)
    }


def test_observation_reuses_previous_values_for_corrupt_reads(follower) -> None:
    robot, bus = follower
    bus.read_positions.side_effect = [
        [100, 200, 300, 400, 500, 600],
        [0, 4095, 301, 401, 501, 601],
    ]
    connect(robot)
    robot.get_observation()

    observation = robot.get_observation()

    assert observation["shoulder_pan.pos"] == 100.0
    assert observation["shoulder_lift.pos"] == 200.0
    assert observation["elbow_flex.pos"] == 301.0


def test_send_action_clamps_rounds_and_returns_command(follower) -> None:
    robot, bus = follower
    connect(robot)
    values = [-100.0, 5000.0, 2048.7, 400.0, 500.0, 600.0]
    action = {f"{name}.pos": value for name, value in zip(JOINT_NAMES, values, strict=True)}

    result = robot.send_action(action)

    expected = [0, 4095, 2049, 400, 500, 600]
    bus.write_positions.assert_called_once_with(expected)
    assert result == {f"{name}.pos": float(value) for name, value in zip(JOINT_NAMES, expected, strict=True)}
    bus.read_positions.assert_not_called()


def test_disconnect_holds_position_before_disabling_torque(follower) -> None:
    robot, bus = follower
    positions = [100, 200, 300, 400, 500, 600]
    bus.read_positions.return_value = positions
    connect(robot)
    bus.reset_mock()

    robot.disconnect()

    bus.read_positions.assert_called_once()
    bus.write_positions.assert_called_once_with(positions)
    bus.set_torque.assert_called_once_with(False)
    bus.exit_lerobot_mode.assert_called_once()
    bus.disconnect.assert_called_once()
    assert not robot.is_connected


def test_config_defaults() -> None:
    config = NexArmFollowerConfig(port="COM19")

    assert config.port == "COM19"
    assert config.baudrate == 1_000_000
    assert config.disable_torque_on_disconnect is True
    assert (config.motion_acc, config.motion_speed) == (100, 2000)
    assert config.cameras == {}
