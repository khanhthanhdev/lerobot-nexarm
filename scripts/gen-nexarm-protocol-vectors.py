#!/usr/bin/env python

# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Write tests/fixtures/nexarm_protocol_vectors.json: golden NexArm CommProtocol frames.

Both this repo (pytest) and the desktop app (cargo test) check every vector, so the two implementations of
the wire protocol cannot drift apart. The desktop copy is refreshed with its scripts/sync-protocol-vectors.

    uv run python scripts/gen-nexarm-protocol-vectors.py > tests/fixtures/nexarm_protocol_vectors.json
"""

import json
import struct
import sys

from lerobot.motors.nexarm.nexarm import (
    CMD_LEROBOT_MODE,
    CMD_READ_POS,
    CMD_SET_MOTION_PARAMS,
    CMD_TORQUE,
    CMD_WRITE_POS,
    SYSTEM_ID,
    build_frame,
)


def positions(values: list[int]) -> bytes:
    return b"".join(struct.pack("<h", v) for v in values)


def short_checksum(frame: bytes) -> bytes:
    """The leader firmware's reply checksum covers only ID, LEN, CMD (tx_packet_complete bug)."""
    return frame[:-1] + bytes([(~sum(frame[2:5])) & 0xFF])


def vector(name: str, kind: str, cmd: int, args: bytes, checksum: str = "full") -> dict:
    frame = build_frame(SYSTEM_ID, cmd, args)
    if checksum == "short":
        frame = short_checksum(frame)
    elif checksum == "bad":
        frame = frame[:-1] + bytes([(frame[-1] + 1) & 0xFF])
    return {
        "name": name,
        "kind": kind,
        "id": SYSTEM_ID,
        "cmd": cmd,
        "args": args.hex(),
        "frame": frame.hex(),
        "checksum": checksum,
    }


def build_vectors() -> list[dict]:
    centre = positions([2048] * 6)
    mixed = positions([0, 4095, 1195, 2833, 100, 3000])
    return [
        vector(
            "set_motion_params_acc100_speed2000", "request", CMD_SET_MOTION_PARAMS, bytes([100, 0xD0, 0x07])
        ),
        vector("bridge_mode_on", "request", CMD_LEROBOT_MODE, bytes([1])),
        vector("bridge_mode_off", "request", CMD_LEROBOT_MODE, bytes([0])),
        vector("read_positions", "request", CMD_READ_POS, b""),
        vector("write_positions_centre", "request", CMD_WRITE_POS, centre),
        vector("write_positions_mixed", "request", CMD_WRITE_POS, mixed),
        vector("torque_on", "request", CMD_TORQUE, bytes([1])),
        vector("torque_off", "request", CMD_TORQUE, bytes([0])),
        vector("read_positions_reply_full", "reply", CMD_READ_POS, mixed),
        vector("read_positions_reply_short_checksum", "reply", CMD_READ_POS, mixed, "short"),
        vector("torque_reply_short_checksum", "reply", CMD_TORQUE, bytes([1]), "short"),
        vector("bridge_mode_reply_short_checksum", "reply", CMD_LEROBOT_MODE, bytes([1]), "short"),
        vector("read_positions_reply_corrupt", "reply", CMD_READ_POS, mixed, "bad"),
    ]


def main() -> None:
    sys.stdout.write(json.dumps({"vectors": build_vectors()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
