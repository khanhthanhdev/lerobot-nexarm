# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Golden CommProtocol frames shared with the desktop app (nexarm-desktop/tests/protocol_vectors.rs)."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from lerobot.motors.nexarm.nexarm import build_frame, parse_frame  # noqa: E402

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "nexarm_protocol_vectors.json"
VECTORS = json.loads(FIXTURE.read_text(encoding="utf-8"))["vectors"]


@pytest.mark.parametrize("v", VECTORS, ids=[v["name"] for v in VECTORS])
def test_every_vector_round_trips_through_the_driver(v):
    frame, args = bytes.fromhex(v["frame"]), bytes.fromhex(v["args"])
    if v["kind"] == "request":
        assert build_frame(v["id"], v["cmd"], args) == frame
    if v["checksum"] == "bad":
        assert parse_frame(frame) is None
    else:
        assert parse_frame(frame) == (v["id"], v["cmd"], args)


def test_the_fixture_covers_every_command_the_driver_sends_and_both_checksum_forms():
    cmds = {v["cmd"] for v in VECTORS}
    assert {56, 68, 96, 97, 98} <= cmds
    assert {v["checksum"] for v in VECTORS} == {"full", "short", "bad"}


def test_the_committed_fixture_matches_its_generator():
    spec = importlib.util.spec_from_file_location(
        "gen_vectors", ROOT / "scripts" / "gen-nexarm-protocol-vectors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build_vectors() == VECTORS, (
        "tests/fixtures/nexarm_protocol_vectors.json is stale; regenerate it with "
        "scripts/gen-nexarm-protocol-vectors.py and refresh the desktop copy"
    )
