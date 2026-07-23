# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

ARM_FEATURES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
BASE_FEATURES = ("x.vel", "y.vel", "theta.vel", "lift_axis.height_mm")
STATE_ACTION_NAMES = tuple(
    [f"left_{name}.pos" for name in ARM_FEATURES]
    + [f"right_{name}.pos" for name in ARM_FEATURES]
    + list(BASE_FEATURES)
)
DEFAULT_CAMERAS = ("front", "left_wrist")


def checkout_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_spec_path() -> Path:
    return checkout_root() / "sim" / "mobile_nexarm" / "spec.json"


@dataclass(frozen=True)
class JointContract:
    feature: str
    source_joint: str
    axis: tuple[float, float, float]
    sim_range: tuple[float, float]
    raw_range: tuple[float, float]
    home_raw: float
    mimic_joint: str | None = None
    mimic_multiplier: float | None = None

    def raw_to_sim(self, raw: float) -> float:
        raw_low, raw_high = self.raw_range
        sim_low, sim_high = self.sim_range
        ratio = (min(max(float(raw), raw_low), raw_high) - raw_low) / (raw_high - raw_low)
        return sim_low + ratio * (sim_high - sim_low)

    def sim_to_raw(self, value: float) -> float:
        sim_low, sim_high = self.sim_range
        raw_low, raw_high = self.raw_range
        ratio = (min(max(float(value), sim_low), sim_high) - sim_low) / (sim_high - sim_low)
        return raw_low + ratio * (raw_high - raw_low)


@dataclass(frozen=True)
class MobileNexArmContract:
    payload: dict[str, Any]
    joints: tuple[JointContract, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return STATE_ACTION_NAMES

    @property
    def joint_by_feature(self) -> dict[str, JointContract]:
        return {joint.feature: joint for joint in self.joints}

    @property
    def home_action(self) -> dict[str, float]:
        action = {
            f"{side}_{joint.feature}.pos": joint.home_raw
            for side in ("left", "right")
            for joint in self.joints
        }
        action.update(
            {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
                "lift_axis.height_mm": float(self.payload["lift"]["home_mm"]),
            }
        )
        return action

    @property
    def dataset_profile(self) -> dict[str, Any]:
        return dict(self.payload["dataset_profile"])

    def forward_kinematics(
        self,
        side: str,
        raw_positions: dict[str, float],
        *,
        lift_height_mm: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Reference end-effector transform used for cross-backend validation."""
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        mount = self.payload["arms"][side]
        lift = self.payload["lift"]["home_mm"] if lift_height_mm is None else float(lift_height_mm)
        transform = _translation(
            np.asarray(mount["mount_xyz_m"], dtype=np.float64)
            + np.asarray([0.0, 0.0, lift / 1000], dtype=np.float64)
        ) @ _rpy_transform(mount["mount_rpy_rad"])
        for joint, length in zip(self.joints[:5], self.payload["kinematics"]["link_lengths_m"], strict=True):
            angle = joint.raw_to_sim(raw_positions[joint.feature])
            transform = (
                transform
                @ _axis_angle_transform(np.asarray(joint.axis), angle)
                @ _translation(np.asarray([0.0, length, 0.0]))
            )
        return transform


def _translation(xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = xyz
    return result


def _axis_angle_transform(axis: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.asarray(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    return result


def _rpy_transform(rpy: list[float]) -> npt.NDArray[np.float64]:
    roll, pitch, yaw = rpy
    return (
        _axis_angle_transform(np.asarray([0.0, 0.0, 1.0]), yaw)
        @ _axis_angle_transform(np.asarray([0.0, 1.0, 0.0]), pitch)
        @ _axis_angle_transform(np.asarray([1.0, 0.0, 0.0]), roll)
    )


@lru_cache(maxsize=4)
def load_contract(path: str | Path | None = None) -> MobileNexArmContract:
    spec_path = Path(path).expanduser() if path is not None else default_spec_path()
    if not spec_path.is_absolute() and not spec_path.is_file():
        spec_path = checkout_root() / spec_path
    with spec_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    joints = tuple(
        JointContract(
            feature=item["feature"],
            source_joint=item["source_joint"],
            axis=tuple(item["axis"]),
            sim_range=tuple(item["range"]),
            raw_range=tuple(item["raw_range"]),
            home_raw=float(item["home_raw"]),
            mimic_joint=item.get("mimic_joint"),
            mimic_multiplier=item.get("mimic_multiplier"),
        )
        for item in payload["joints"]
    )
    if tuple(joint.feature for joint in joints) != ARM_FEATURES:
        raise ValueError("spec joint order must match the canonical NexArm arm feature order")
    if len(STATE_ACTION_NAMES) != 16 or len(set(STATE_ACTION_NAMES)) != 16:
        raise AssertionError("mobile NexArm contract must contain 16 unique features")
    return MobileNexArmContract(payload=payload, joints=joints)
