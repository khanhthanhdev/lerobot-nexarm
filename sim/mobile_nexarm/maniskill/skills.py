"""Deterministic staged pick/place policy with a bounded repair budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class Stage(StrEnum):
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSFER = "transfer"
    RELEASE = "release"
    RETREAT = "retreat"


class SkillBackend(Protocol):
    def object_pose(self) -> npt.NDArray[np.float64]: ...
    def target_pose(self) -> npt.NDArray[np.float64]: ...
    def solve_left_ik(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None: ...
    def execute_left_target(
        self, joints: npt.NDArray[np.float64], gripper_raw: float, steps: int
    ) -> dict[str, Any]: ...
    def evaluate(self) -> dict[str, Any]: ...


@dataclass
class Trace:
    stage: str
    attempt: int
    target_xyz: list[float]
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScriptedResult:
    success: bool
    traces: list[dict[str, Any]]
    repairs: list[dict[str, Any]]
    reason: str


class PickPlaceSkill:
    """Approach, grasp, lift, transfer, release, retreat with local repairs."""

    def __init__(self, *, max_repairs: int = 2, waypoint_steps: int = 12) -> None:
        if max_repairs < 0:
            raise ValueError("max_repairs cannot be negative")
        self.max_repairs = max_repairs
        self.waypoint_steps = waypoint_steps

    def run(self, backend: SkillBackend) -> ScriptedResult:
        traces: list[dict[str, Any]] = []
        repairs: list[dict[str, Any]] = []
        cube = np.asarray(backend.object_pose(), dtype=np.float64)[:3]
        target = np.asarray(backend.target_pose(), dtype=np.float64)[:3]
        stages = (
            (Stage.APPROACH, cube + [0, 0, 0.09], 1195.0),
            (Stage.GRASP, cube + [0, 0, 0.015], 2833.0),
            (Stage.LIFT, cube + [0, 0, 0.12], 2833.0),
            (Stage.TRANSFER, target + [0, 0, 0.12], 2833.0),
            (Stage.RELEASE, target + [0, 0, 0.045], 1195.0),
            (Stage.RETREAT, target + [0, 0, 0.15], 1195.0),
        )
        for stage, nominal, gripper in stages:
            succeeded = False
            for attempt in range(self.max_repairs + 1):
                offset = np.asarray([0.0, 0.0, 0.01 * attempt])
                waypoint = nominal + offset
                joints = backend.solve_left_ik(waypoint)
                if joints is None:
                    result = {"ok": False, "reason": "ik_failed"}
                else:
                    result = backend.execute_left_target(joints, gripper, self.waypoint_steps)
                traces.append(
                    {
                        "stage": stage.value,
                        "attempt": attempt,
                        "target_xyz": waypoint.tolist(),
                        "result": result,
                    }
                )
                if result.get("ok", False):
                    succeeded = True
                    break
                if attempt < self.max_repairs:
                    repairs.append(
                        {
                            "stage": stage.value,
                            "attempt": attempt + 1,
                            "reason": result.get("reason", "unknown"),
                            "z_offset_m": float(offset[2] + 0.01),
                        }
                    )
            if not succeeded:
                return ScriptedResult(False, traces, repairs, f"{stage.value}_failed")
        evaluation = backend.evaluate()
        return ScriptedResult(
            bool(evaluation.get("success", False)),
            traces,
            repairs,
            str(evaluation.get("reason", "success" if evaluation.get("success") else "task_gate_failed")),
        )
