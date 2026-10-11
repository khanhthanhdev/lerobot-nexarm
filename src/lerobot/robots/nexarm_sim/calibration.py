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

"""Real-to-sim calibration parameters for the NexArm MuJoCo twin."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lerobot.motors.nexarm.mujoco_mapping import JOINT_MAPPING_VERSION
from lerobot.motors.nexarm.nexarm import JOINT_NAMES

from .mujoco_backend import NexArmMujocoBackend

MIN_SPREAD = 0.10


@dataclass
class JointCalibration:
    """Multiplicative corrections to the model's joint dynamics.

    ``*_spread`` is the fractional half-range used for domain randomization around the calibrated value.
    """

    kp_scale: float = 1.0
    damping_scale: float = 1.0
    frictionloss_scale: float = 1.0
    damping_spread: float = 0.25
    frictionloss_spread: float = 0.25

    def __post_init__(self) -> None:
        for name in ("kp_scale", "damping_scale", "frictionloss_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("damping_spread", "frictionloss_spread"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")


@dataclass
class SimCalibration:
    """Calibrated action latency and per-joint dynamics fitted from real recordings.

    ``fps`` is the control rate the latency (in frames) was fitted at; ``fitted_joints`` lists the joints whose
    dynamics were actually fitted. Both are optional so older files still load (they fall back to ``source``).
    """

    action_delay_steps: int = 0
    joints: dict[str, JointCalibration] = field(
        default_factory=lambda: {name: JointCalibration() for name in JOINT_NAMES}
    )
    source: dict[str, Any] = field(default_factory=dict)
    fps: int | None = None
    fitted_joints: list[str] | None = None
    joint_mapping: str = JOINT_MAPPING_VERSION

    def __post_init__(self) -> None:
        if self.action_delay_steps < 0:
            raise ValueError("action_delay_steps cannot be negative")
        unknown = set(self.joints) - set(JOINT_NAMES)
        if unknown:
            raise ValueError(f"Unknown joints in calibration: {sorted(unknown)}")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.fitted_joints is not None:
            unknown = set(self.fitted_joints) - set(self.joints)
            if unknown:
                raise ValueError(f"fitted_joints not present in calibration joints: {sorted(unknown)}")

    def fitted_joint_names(self) -> list[str]:
        """Joints whose fitted values should replace the defaults (all listed joints when not recorded)."""
        return list(self.fitted_joints) if self.fitted_joints is not None else list(self.joints)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> SimCalibration:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        joint_mapping = raw.get("joint_mapping")
        if joint_mapping != JOINT_MAPPING_VERSION:
            raise ValueError(
                f"Calibration {path} was fitted with joint mapping {joint_mapping or 'the pre-servo-constant mapping'} "
                f"but this code uses {JOINT_MAPPING_VERSION}. Re-run examples/nexarm/calibrate_sim.py."
            )
        source = raw.get("source", {})
        fps = raw.get("fps") if raw.get("fps") is not None else source.get("fps")
        fitted_joints = (
            raw.get("fitted_joints") if raw.get("fitted_joints") is not None else source.get("fitted_joints")
        )
        return cls(
            action_delay_steps=int(raw.get("action_delay_steps", 0)),
            joints={name: JointCalibration(**values) for name, values in raw.get("joints", {}).items()},
            source=source,
            fps=int(fps) if fps is not None else None,
            fitted_joints=list(fitted_joints) if fitted_joints is not None else None,
            joint_mapping=joint_mapping,
        )


def check_calibration_fps(calibration: SimCalibration, fps: int) -> None:
    """Refuse a calibration fitted at another control rate (its latency is counted in frames)."""
    if calibration.fps is not None and calibration.fps != fps:
        raise ValueError(
            f"Simulation calibration was fitted at {calibration.fps} fps but this run uses {fps} fps. "
            "Re-run examples/nexarm/calibrate_sim.py at the run fps or match --fps to the calibration."
        )


def resolve_action_delay_steps(explicit: int | None, calibration: SimCalibration | None) -> int:
    """Explicit config/CLI value > calibration > default (no delay)."""
    if explicit is not None:
        return explicit
    if calibration is not None:
        return calibration.action_delay_steps
    return 0


def apply_calibration(
    backend: NexArmMujocoBackend,
    calibration: SimCalibration,
    *,
    action_delay_steps: int | None = None,
) -> None:
    """Scale the model's joint dynamics in place and re-center domain randomization on them.

    Only joints the calibration fitted get their randomization ranges replaced; the others keep the
    default ranges. ``action_delay_steps`` overrides the calibrated latency when given.
    """

    check_calibration_fps(calibration, backend.fps)
    model = backend.model
    fitted = set(calibration.fitted_joint_names())
    for name, joint in calibration.joints.items():
        joint_id = backend._joint_ids[name]
        dof = int(model.jnt_dofadr[joint_id])
        model.dof_damping[dof] *= joint.damping_scale
        model.dof_frictionloss[dof] *= joint.frictionloss_scale
        actuator_id = backend._actuator_ids[name]
        model.actuator_gainprm[actuator_id, 0] *= joint.kp_scale
        model.actuator_biasprm[actuator_id, 1] *= joint.kp_scale
        if name not in fitted:
            continue
        backend.dr_ranges.joint_damping_scale[name] = (
            1.0 - joint.damping_spread,
            1.0 + joint.damping_spread,
        )
        backend.dr_ranges.joint_friction_scale[name] = (
            1.0 - joint.frictionloss_spread,
            1.0 + joint.frictionloss_spread,
        )
    backend.snapshot_nominal_dynamics()
    backend.action_delay_steps = resolve_action_delay_steps(action_delay_steps, calibration)
