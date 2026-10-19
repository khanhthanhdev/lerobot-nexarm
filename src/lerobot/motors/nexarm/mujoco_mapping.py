# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single conversion between NexArm raw servo positions and MuJoCo joint positions.

Arm joints use the servo constant (4096 ticks per revolution around the 2048 center) and are clamped to the
MuJoCo joint range, so the joint range acts like a hardware limit. The gripper maps the follower's raw
range linearly and inverted onto the jaw slide joint: ``GRIPPER_CLOSED_POS`` is the slide's lower limit
(jaws closed) and ``GRIPPER_OPEN_POS`` its upper limit (jaws fully open).

By default raw 2048 lands on the exported CAD pose, which is the real arm's Home pose (``servo_4096_v2``;
``v1`` used an earlier CAD zero with the arm bent differently). A :class:`JointMapping` measured on the real follower
with ``examples/nexarm/probe_joint_mapping.py`` adds a per-joint sign and zero offset, so the sim reaches the
same pose as the real arm for the same raw values.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.motors.nexarm.nexarm import (
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    JOINT_NAMES,
    POSITION_CENTER,
    POSITION_MAX,
    POSITION_MIN,
    radians_to_raw,
    raw_to_radians,
)

# Bumped whenever the raw <-> joint mapping changes meaning. Calibrations and generated datasets record it
# so data encoded with a different mapping is refused instead of silently mixed.
JOINT_MAPPING_VERSION = "servo_4096_v2"


@dataclass(frozen=True)
class JointMapping:
    """Per-arm-joint ``qpos = sign * (raw - 2048) * 2*pi/4096 + offset_rad``; the default is the identity.

    Joints left out keep sign +1 and offset 0. The gripper always uses its linear slide mapping.
    """

    signs: Mapping[str, int] = field(default_factory=dict)
    offsets_rad: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = (set(self.signs) | set(self.offsets_rad)) - set(JOINT_NAMES[:-1])
        if unknown:
            raise ValueError(f"Joint mapping names unknown arm joints: {sorted(unknown)}")
        bad = {name: sign for name, sign in self.signs.items() if sign not in (1, -1)}
        if bad:
            raise ValueError(f"Joint mapping signs must be +1 or -1, got {bad}")

    def sign(self, feature_name: str) -> int:
        return int(self.signs.get(feature_name, 1))

    def offset(self, feature_name: str) -> float:
        return float(self.offsets_rad.get(feature_name, 0.0))

    @property
    def is_identity(self) -> bool:
        return all(self.sign(name) == 1 and self.offset(name) == 0.0 for name in JOINT_NAMES[:-1])

    @property
    def version(self) -> str:
        """Recorded by calibrations and datasets; differs for every distinct mapping."""
        if self.is_identity:
            return JOINT_MAPPING_VERSION
        values = [[self.sign(name), round(self.offset(name), 6)] for name in JOINT_NAMES[:-1]]
        digest = hashlib.sha256(json.dumps(values).encode()).hexdigest()[:12]
        return f"{JOINT_MAPPING_VERSION}+probe-{digest}"

    @classmethod
    def load(cls, path: Path | str) -> JointMapping:
        """Read the JSON written by ``examples/nexarm/probe_joint_mapping.py``."""
        joints = json.loads(Path(path).read_text(encoding="utf-8"))["joints"]
        return cls(
            signs={name: int(joint["sign"]) for name, joint in joints.items()},
            offsets_rad={name: float(joint["offset_rad"]) for name, joint in joints.items()},
        )


IDENTITY_MAPPING = JointMapping()


def load_joint_mapping(path: Path | str | None) -> JointMapping:
    """The probe mapping at ``path``, or the identity when no path is given."""
    return IDENTITY_MAPPING if path is None else JointMapping.load(path)


MUJOCO_JOINTS = {
    "shoulder_pan": "joint_1_base_to_link_1",
    "shoulder_lift": "joint_2_link_1_to_link_2",
    "elbow_flex": "joint_3_link_2_to_link_3",
    "wrist_flex": "joint_4_link_3_to_link_4",
    "wrist_roll": "joint_5_link_4_to_link_5",
    "gripper": "right_jaw_slide_joint",
}

# The physical follower accepts 0..4095 for every servo. Its leader mapping
# deliberately restricts the useful gripper command range to open..closed.
RAW_RANGES: dict[str, tuple[int, int]] = dict.fromkeys(JOINT_NAMES[:-1], (POSITION_MIN, POSITION_MAX))
RAW_RANGES["gripper"] = (GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS)

HOME_POSITIONS: dict[str, float] = dict.fromkeys(JOINT_NAMES[:-1], float(POSITION_CENTER))
HOME_POSITIONS["gripper"] = float(GRIPPER_CLOSED_POS)


def _checked_range(joint_range: Sequence[float]) -> tuple[float, float]:
    low, high = float(joint_range[0]), float(joint_range[1])
    if high <= low:
        raise ValueError(f"MuJoCo joint range must be limited with low < high, got ({low}, {high})")
    return low, high


def raw_to_joint_position(
    feature_name: str,
    raw_position: float,
    joint_range: Sequence[float],
    mapping: JointMapping = IDENTITY_MAPPING,
) -> float:
    """Convert a raw servo position to a MuJoCo joint position (rad, or m for the gripper slide)."""
    raw_low, raw_high = RAW_RANGES[feature_name]
    raw = min(max(float(raw_position), raw_low), raw_high)
    low, high = _checked_range(joint_range)
    if feature_name == "gripper":
        ratio = (GRIPPER_CLOSED_POS - raw) / (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
        return low + ratio * (high - low)
    joint_position = mapping.sign(feature_name) * raw_to_radians(raw) + mapping.offset(feature_name)
    return min(max(joint_position, low), high)


def joint_position_to_raw(
    feature_name: str,
    joint_position: float,
    joint_range: Sequence[float],
    mapping: JointMapping = IDENTITY_MAPPING,
) -> float:
    """Inverse of :func:`raw_to_joint_position`; positions outside the joint range are clamped first."""
    low, high = _checked_range(joint_range)
    value = min(max(float(joint_position), low), high)
    if feature_name == "gripper":
        ratio = (value - low) / (high - low)
        return GRIPPER_CLOSED_POS - ratio * (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
    # The sign is +1 or -1, so it is its own inverse.
    return radians_to_raw(mapping.sign(feature_name) * (value - mapping.offset(feature_name)))


def reachable_raw_range(
    feature_name: str, joint_range: Sequence[float], mapping: JointMapping = IDENTITY_MAPPING
) -> tuple[float, float]:
    """Raw interval that maps inside the joint range (ascending)."""
    low, high = _checked_range(joint_range)
    ends = sorted(
        (
            joint_position_to_raw(feature_name, low, (low, high), mapping),
            joint_position_to_raw(feature_name, high, (low, high), mapping),
        )
    )
    raw_low, raw_high = RAW_RANGES[feature_name]
    return max(ends[0], raw_low), min(ends[1], raw_high)
