# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Hardware-zero policy: engine calibration files are identity.

Zero positions live in the servos (desktop Servo Advanced page for the follower, Synchronizer page
for the leader). The engine never applies a software ``homing_offset``; ``check`` only reports how far
the leader and follower are from each other at the vendor Home pose (every joint at 2048 ticks).
"""

import json
import logging
import shutil
import time
from pathlib import Path

from pydantic import BaseModel

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS, TELEOPERATORS
from lerobot_robot_nexarm.motors.nexarm import (
    JOINT_NAMES,
    POSITION_CENTER,
    POSITION_MAX,
    POSITION_MIN,
    map_leader_to_follower,
)

logger = logging.getLogger(__name__)

# Per-joint |leader - follower| above this at Home means a servo offset needs fixing in hardware.
DEFAULT_TOLERANCE_TICKS = 40


def identity_calibration() -> dict[str, dict[str, int]]:
    return {
        name: {
            "id": i + 1,
            "drive_mode": 0,
            "homing_offset": POSITION_CENTER,
            "range_min": POSITION_MIN,
            "range_max": POSITION_MAX,
        }
        for i, name in enumerate(JOINT_NAMES)
    }


def is_identity(calibration: dict) -> bool:
    expected = identity_calibration()
    return all(
        name in calibration and all(calibration[name].get(k) == v for k, v in fields.items())
        for name, fields in expected.items()
    )


def nexarm_calibration_files(root: Path = HF_LEROBOT_CALIBRATION) -> list[Path]:
    dirs = [root / ROBOTS / "nexarm_follower", root / TELEOPERATORS / "nexarm_leader"]
    return sorted(p for d in dirs if d.is_dir() for p in d.glob("*.json"))


def reset_non_identity_files(root: Path = HF_LEROBOT_CALIBRATION) -> list[Path]:
    """Back up then overwrite every NexArm calibration file that carries a software offset.

    Returns the files that were reset. The backup sits next to the original as ``<name>.json.bak-<ts>``.
    """
    reset = []
    for path in nexarm_calibration_files(root):
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable NexArm calibration file %s; leaving it alone", path)
            continue
        if is_identity(current):
            continue
        backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        path.write_text(json.dumps(identity_calibration(), indent=4), encoding="utf-8")
        logger.warning("reset software calibration offsets in %s (backup: %s)", path, backup)
        reset.append(path)
    return reset


class JointDelta(BaseModel):
    joint: str
    leader_raw: int
    leader_mapped: int
    follower: int
    delta: int
    ok: bool


class CalibrationReport(BaseModel):
    tolerance_ticks: int
    joints: list[JointDelta]
    ok: bool
    reset_files: list[str]
    hint: str


def build_report(
    leader_raw: list[int],
    follower: list[int],
    reset_files: list[Path],
    tolerance: int = DEFAULT_TOLERANCE_TICKS,
) -> CalibrationReport:
    mapped = map_leader_to_follower(leader_raw)
    joints = [
        JointDelta(
            joint=name,
            leader_raw=leader_raw[i],
            leader_mapped=mapped[i],
            follower=follower[i],
            delta=follower[i] - mapped[i],
            ok=abs(follower[i] - mapped[i]) <= tolerance,
        )
        for i, name in enumerate(JOINT_NAMES)
    ]
    ok = all(j.ok for j in joints)
    hint = (
        "Leader and follower agree at Home."
        if ok
        else "Fix zero offsets in hardware: follower on the Servo Advanced page, leader on the Synchronizer page."
    )
    return CalibrationReport(
        tolerance_ticks=tolerance, joints=joints, ok=ok, reset_files=[str(p) for p in reset_files], hint=hint
    )
