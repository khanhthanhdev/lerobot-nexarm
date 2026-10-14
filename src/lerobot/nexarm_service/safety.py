# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Arm safety helpers shared by every session kind.

NexArm servos drop when torque is cut, so a session never releases the arm where it happens to be:
it captures a validated rest pose at start and ramps back to it before torque off.
"""

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence

from lerobot.motors.nexarm.nexarm import (
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    JOINT_NAMES,
    POSITION_MAX,
    radians_to_raw,
)

logger = logging.getLogger(__name__)

# URDF joint limits (sim/description/urdf/nexarm.urdf, joints 1-5) in radians. Converted to ticks below.
# HARDWARE VALIDATION NEEDED: these are model limits, not yet confirmed against the physical arm.
_URDF_LIMITS_RAD = {
    "shoulder_pan": 2.35619,
    "shoulder_lift": 2.0944,
    "elbow_flex": 2.35619,
    "wrist_flex": 1.74533,
    "wrist_roll": 3.14159,
}


# The firmware occasionally returns values pinned near 0 or 4095 on a dropped packet. Reads within this
# margin of either end are treated as corrupt, so no soft limit may reach into it: an arm legitimately parked
# there could never be read back and returned to rest.
CORRUPT_MARGIN = 5


def _soft_limits() -> dict[str, tuple[int, int]]:
    limits = {}
    for name, rad in _URDF_LIMITS_RAD.items():
        low = max(CORRUPT_MARGIN + 1, math.ceil(radians_to_raw(-rad)))
        high = min(POSITION_MAX - CORRUPT_MARGIN - 1, math.floor(radians_to_raw(rad)))
        limits[name] = (low, high)
    limits["gripper"] = (GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS)
    return limits


SOFT_LIMITS: dict[str, tuple[int, int]] = _soft_limits()


def clamp_action(action: Mapping[str, float]) -> dict[str, float]:
    """Clamp every ``<joint>.pos`` value to its soft limit; unknown keys are rejected."""
    out: dict[str, float] = {}
    for name in JOINT_NAMES:
        low, high = SOFT_LIMITS[name]
        out[f"{name}.pos"] = float(max(low, min(high, action[f"{name}.pos"])))
    return out


def is_valid_read(positions: Sequence[float]) -> bool:
    return len(positions) == len(JOINT_NAMES) and all(
        CORRUPT_MARGIN < p < POSITION_MAX - CORRUPT_MARGIN for p in positions
    )


def capture_rest(
    read: Callable[[], Sequence[int]],
    samples: int = 5,
    tolerance: int = 20,
    attempts: int = 30,
    sleep: Callable[[float], None] = time.sleep,
) -> list[int]:
    """Return a rest pose built only from consecutive valid reads that agree within ``tolerance`` ticks.

    Raises RuntimeError when the arm never settles on valid reads: starting a session without a
    trustworthy rest pose would mean returning the arm to garbage at stop.
    """
    window: list[list[int]] = []
    for _ in range(attempts):
        try:
            positions = list(read())
        except TimeoutError:
            positions = []
        if is_valid_read(positions) and all(
            abs(a - b) <= tolerance
            for a, b in zip(positions, window[-1] if window else positions, strict=True)
        ):
            window.append(positions)
            if len(window) >= samples:
                return [round(sum(col) / len(col)) for col in zip(*window, strict=True)]
        else:
            window = [positions] if is_valid_read(positions) else []
        sleep(0.02)
    raise RuntimeError("could not capture a stable rest pose: no run of valid, steady position reads")


def ramp_to(
    read: Callable[[], Sequence[int]],
    write: Callable[[list[int]], None],
    target: Sequence[int],
    max_step: int = 40,
    dt: float = 0.03,
    tolerance: int = 12,
    max_steps: int = 400,
    give_up_s: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Move every joint toward ``target`` by at most ``max_step`` ticks per ``dt``.

    The ramp starts from one validated read and then steps from its own last command, never from a later
    raw read: a corrupt read (pinned near 0 or 4095) or a dropped one cannot make a joint jump. Invalid
    or timed-out reads are skipped; the ramp gives up only after ``give_up_s`` without a valid read, and
    raises if the target is not reached within ``max_steps`` ticks.
    """
    command: list[int] | None = None
    bad_since: float | None = None
    for _ in range(max_steps):
        if command is None:
            try:
                current = list(read())
            except TimeoutError:
                current = []
            if not is_valid_read(current):
                bad_since = clock() if bad_since is None else bad_since
                if clock() - bad_since > give_up_s:
                    raise TimeoutError("no valid position read; cannot return the arm to rest")
                sleep(dt)
                continue
            command = current
        deltas = [t - c for t, c in zip(target, command, strict=True)]
        if all(abs(d) <= tolerance for d in deltas):
            write(list(target))
            return
        command = [c + max(-max_step, min(max_step, d)) for c, d in zip(command, deltas, strict=True)]
        write(list(command))
        sleep(dt)
    raise TimeoutError("arm did not reach the rest pose")
