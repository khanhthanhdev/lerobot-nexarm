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
"""

from __future__ import annotations

from collections.abc import Sequence

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
JOINT_MAPPING_VERSION = "servo_4096_v1"

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


def raw_to_joint_position(feature_name: str, raw_position: float, joint_range: Sequence[float]) -> float:
    """Convert a raw servo position to a MuJoCo joint position (rad, or m for the gripper slide)."""
    raw_low, raw_high = RAW_RANGES[feature_name]
    raw = min(max(float(raw_position), raw_low), raw_high)
    low, high = _checked_range(joint_range)
    if feature_name == "gripper":
        ratio = (GRIPPER_CLOSED_POS - raw) / (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
        return low + ratio * (high - low)
    return min(max(raw_to_radians(raw), low), high)


def joint_position_to_raw(feature_name: str, joint_position: float, joint_range: Sequence[float]) -> float:
    """Inverse of :func:`raw_to_joint_position`; positions outside the joint range are clamped first."""
    low, high = _checked_range(joint_range)
    value = min(max(float(joint_position), low), high)
    if feature_name == "gripper":
        ratio = (value - low) / (high - low)
        return GRIPPER_CLOSED_POS - ratio * (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
    return radians_to_raw(value)


def reachable_raw_range(feature_name: str, joint_range: Sequence[float]) -> tuple[float, float]:
    """Raw interval that maps inside the joint range (ascending)."""
    low, high = _checked_range(joint_range)
    ends = sorted(
        (
            joint_position_to_raw(feature_name, low, (low, high)),
            joint_position_to_raw(feature_name, high, (low, high)),
        )
    )
    raw_low, raw_high = RAW_RANGES[feature_name]
    return max(ends[0], raw_low), min(ends[1], raw_high)
