#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hardware-free tests for the NexArm CommProtocol motor bus."""

import struct
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from lerobot.motors.nexarm.nexarm import (  # noqa: E402
    CMD_LEROBOT_MODE,
    CMD_READ_POS,
    CMD_SET_MOTION_PARAMS,
    CMD_TORQUE,
    CMD_WRITE_POS,
    DEFAULT_BAUDRATE,
    JOINT_COUNT,
    JOINT_NAMES,
    POSITION_MAX,
    POSITION_MIN,
    SYSTEM_ID,
    NexArmMotorsBus,
    build_frame,
    map_leader_to_follower,
    parse_frame,
)


def test_frame_round_trip_and_checksum() -> None:
    payload = bytes([1, 2, 3])
    frame = build_frame(SYSTEM_ID, CMD_TORQUE, payload)

    assert frame[:2] == b"\xff\xff"
    assert frame[3] == len(payload) + 2
    assert frame[-1] == (~sum(frame[2:-1])) & 0xFF
    assert parse_frame(frame) == (SYSTEM_ID, CMD_TORQUE, payload)


@pytest.mark.parametrize(
    "frame",
    [
        b"\xff\xff\x01",
        b"\x00\x00\xff\x02\x60\x9e",
        build_frame(SYSTEM_ID, CMD_READ_POS, bytes(12))[:-3],
    ],
)
def test_parse_frame_rejects_malformed_frames(frame: bytes) -> None:
    assert parse_frame(frame) is None


def test_parse_frame_rejects_bad_checksum() -> None:
    frame = bytearray(build_frame(SYSTEM_ID, CMD_READ_POS))
    frame[-1] ^= 0xFF
    assert parse_frame(bytes(frame)) is None


def test_parse_frame_accepts_firmware_short_checksum() -> None:
    args = bytes(range(JOINT_COUNT * 2))
    length = len(args) + 2
    data = bytes([SYSTEM_ID, length, CMD_READ_POS]) + args
    firmware_checksum = (~sum(data[:3])) & 0xFF

    assert parse_frame(b"\xff\xff" + data + bytes([firmware_checksum])) == (
        SYSTEM_ID,
        CMD_READ_POS,
        args,
    )


def test_leader_to_follower_mapping() -> None:
    result = map_leader_to_follower([100, 1000, 300, 400, 500, 2048])

    assert result == [100, 3096, 300, 400, 500, 2833]
    assert map_leader_to_follower([-100, 5000, 2048, 2048, 2048, 2500]) == [
        POSITION_MIN,
        1,
        2048,
        2048,
        2048,
        2833,
    ]


def test_constants_match_six_joint_protocol() -> None:
    assert JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    assert len(JOINT_NAMES) == JOINT_COUNT == 6
    assert (POSITION_MIN, POSITION_MAX) == (0, 4095)


def test_bus_connect_disconnect() -> None:
    mock_serial = MagicMock(is_open=True)
    with (
        patch("lerobot.motors.nexarm.nexarm.serial.Serial", return_value=mock_serial) as serial_cls,
        patch("lerobot.motors.nexarm.nexarm.time.sleep"),
    ):
        bus = NexArmMotorsBus("/dev/null")
        bus.connect()

        assert bus.is_connected
        serial_cls.assert_called_once_with(
            port="/dev/null",
            baudrate=DEFAULT_BAUDRATE,
            timeout=bus.timeout,
            write_timeout=bus.timeout,
        )

        bus.disconnect()

    assert not bus.is_connected
    mock_serial.close.assert_called_once()


def test_read_positions_decodes_and_clamps_reply() -> None:
    bus = NexArmMotorsBus("/dev/null")
    positions = [-100, 1024, 3072, 512, 5000, 2000]
    reply = b"".join(struct.pack("<h", position) for position in positions)

    with patch.object(bus, "_send", return_value=(CMD_READ_POS, reply)):
        assert bus.read_positions() == [0, 1024, 3072, 512, 4095, 2000]


def test_read_positions_retries_then_times_out() -> None:
    bus = NexArmMotorsBus("/dev/null")
    with (
        patch.object(bus, "_send", return_value=None) as send,
        patch("lerobot.motors.nexarm.nexarm.time.sleep"),
        pytest.raises(TimeoutError, match="No position reply"),
    ):
        bus.read_positions(retries=3)

    assert send.call_count == 3


def test_write_positions_clamps_and_encodes() -> None:
    bus = NexArmMotorsBus("/dev/null")
    with patch.object(bus, "_send") as send:
        bus.write_positions([-100, 5000, 100, 200, 300, 400])

    frame = send.call_args.args[0]
    assert send.call_args.kwargs == {"expect_reply": False}
    _, cmd, args = parse_frame(frame)  # type: ignore[misc]
    assert cmd == CMD_WRITE_POS
    assert list(struct.unpack("<6h", args)) == [0, 4095, 100, 200, 300, 400]


@pytest.mark.parametrize(
    ("method", "value", "command", "expected_args"),
    [
        ("set_torque", True, CMD_TORQUE, b"\x01"),
        ("set_torque", False, CMD_TORQUE, b"\x00"),
        ("enter_lerobot_mode", None, CMD_LEROBOT_MODE, b"\x01"),
        ("exit_lerobot_mode", None, CMD_LEROBOT_MODE, b"\x00"),
    ],
)
def test_control_commands(method: str, value: bool | None, command: int, expected_args: bytes) -> None:
    bus = NexArmMotorsBus("/dev/null")
    with patch.object(bus, "_send") as send, patch("lerobot.motors.nexarm.nexarm.time.sleep"):
        if value is None:
            getattr(bus, method)()
        else:
            getattr(bus, method)(value)

    _, cmd, args = parse_frame(send.call_args.args[0])  # type: ignore[misc]
    assert (cmd, args) == (command, expected_args)
    assert send.call_args.kwargs == {"expect_reply": False}


def test_write_motion_params_clamps_and_encodes() -> None:
    bus = NexArmMotorsBus("/dev/null")
    with patch.object(bus, "_send") as send:
        bus.write_motion_params(acc=999, speed=9999)

    _, cmd, args = parse_frame(send.call_args.args[0])  # type: ignore[misc]
    assert cmd == CMD_SET_MOTION_PARAMS
    assert args == bytes([254, 0x48, 0x0D])
    assert send.call_args.kwargs == {"expect_reply": False}


def test_read_requires_connection() -> None:
    bus = NexArmMotorsBus("/dev/null")
    with pytest.raises(ConnectionError, match="Serial port not open"):
        bus.read_positions(retries=1)
