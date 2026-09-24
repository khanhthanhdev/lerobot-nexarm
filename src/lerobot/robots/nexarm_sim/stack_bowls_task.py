# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import itertools
from dataclasses import dataclass

import mujoco
import numpy as np

from .mujoco_backend import NexArmMujocoBackend

COLORS = ("red", "blue", "black")
PERMUTATIONS: list[tuple[str, str, str]] = list(itertools.permutations(COLORS))
# 6 permutations:
# 1. ('red', 'blue', 'black')
# 2. ('red', 'black', 'blue')
# 3. ('blue', 'red', 'black')
# 4. ('blue', 'black', 'red')
# 5. ('black', 'red', 'blue')
# 6. ('black', 'blue', 'red')


def get_task_instruction(bottom: str, middle: str, top: str) -> str:
    """Format task instruction for the 3-bowl stacking task."""
    return f"Stack the bowls with {bottom} on bottom, {middle} in middle, and {top} on top."


@dataclass(frozen=True)
class NexArmStackBowlsStatus:
    """State of the NexArm 3-bowl stacking task."""

    success: bool
    terminated: bool
    reason: str | None
    hold_time_s: float
    order: tuple[str, str, str]
    is_concentric: bool
    is_nested_height: bool
    is_released: bool


class NexArmStackBowlsTask:
    """Seeded 3-bowl stacking task on the 6-DOF NexArm platform."""

    def __init__(
        self,
        backend: NexArmMujocoBackend,
        *,
        max_concentric_error_m: float = 0.035,
        min_nesting_delta_m: float = 0.010,
        max_nesting_delta_m: float = 0.038,
        success_hold_s: float = 0.5,
        timeout_s: float = 40.0,
    ) -> None:
        self.backend = backend
        self.max_concentric_error_m = max_concentric_error_m
        self.min_nesting_delta_m = min_nesting_delta_m
        self.max_nesting_delta_m = max_nesting_delta_m
        self.success_hold_s = success_hold_s
        self.timeout_s = timeout_s

        # Cache IDs
        self._body_ids = {c: self._required_id(mujoco.mjtObj.mjOBJ_BODY, f"bowl_{c}") for c in COLORS}
        self._joint_ids = {c: self._required_id(mujoco.mjtObj.mjOBJ_JOINT, f"bowl_{c}_joint") for c in COLORS}
        self._left_jaw_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "link_6_left_jaw_collision_0")
        self._right_jaw_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "link_6_right_jaw_collision_0")

        self.current_order: tuple[str, str, str] = PERMUTATIONS[0]
        self._start_time: float = 0.0
        self._hold_start_time: float | None = None
        self._failure_reason: str | None = None

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.backend.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"NexArm task model is missing {object_type.name}: {name}")
        return object_id

    def bowl_position(self, color: str) -> np.ndarray:
        return self.backend.data.xpos[self._body_ids[color]].copy()

    def bowl_velocity(self, color: str) -> np.ndarray:
        body_id = self._body_ids[color]
        # Linear velocity in world frame
        return self.backend.data.cvel[body_id, 3:6].copy()

    def reset(
        self,
        *,
        seed: int = 0,
        order: tuple[str, str, str] | None = None,
        settle_steps: int = 25,
    ) -> NexArmStackBowlsStatus:
        self.backend.reset(settle_steps=0)
        rng = np.random.default_rng(seed)

        # Select permutation (cycle uniformly through the 6 permutations by seed)
        if order is None:
            self.current_order = PERMUTATIONS[seed % len(PERMUTATIONS)]
        else:
            self.current_order = order

        # Spawn 3 bowls on table in non-overlapping reachable positions
        # Workspace: X in [-0.12, 0.12], Y in [-0.28, -0.17]
        # Bowls have radius ~62mm. Staggering ensures >= 140mm inter-bowl distance.
        base_positions = [
            np.array([-0.11, -0.27, 0.020]),
            np.array([0.00, -0.18, 0.020]),
            np.array([0.11, -0.27, 0.020]),
        ]
        # Shuffle slot assignment
        slot_order = rng.permutation(3)

        for i, color in enumerate(COLORS):
            slot = slot_order[i]
            # Add small random jitter (+-5mm in X, +-5mm in Y)
            jitter_x = rng.uniform(-0.005, 0.005)
            jitter_y = rng.uniform(-0.005, 0.005)
            yaw = rng.uniform(-np.pi, np.pi)
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

            qpos_adr = int(self.backend.model.jnt_qposadr[self._joint_ids[color]])
            pos = base_positions[slot] + np.array([jitter_x, jitter_y, 0.0])
            self.backend.data.qpos[qpos_adr : qpos_adr + 3] = pos
            self.backend.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

        # Forward kinematics
        mujoco.mj_forward(self.backend.model, self.backend.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.backend.model, self.backend.data)

        self._start_time = float(self.backend.data.time)
        self._hold_start_time = None
        self._failure_reason = None
        return self.status()

    def _is_gripper_disengaged(self) -> bool:
        """Check if gripper jaws are not contacting any bowl."""
        for i in range(self.backend.data.ncon):
            con = self.backend.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            is_jaw = g1 in (self._left_jaw_geom_id, self._right_jaw_geom_id) or g2 in (
                self._left_jaw_geom_id,
                self._right_jaw_geom_id,
            )
            if is_jaw:
                return False
        return True

    def status(self) -> NexArmStackBowlsStatus:
        bottom, middle, top = self.current_order
        pos_b = self.bowl_position(bottom)
        pos_m = self.bowl_position(middle)
        pos_t = self.bowl_position(top)

        # 1. Concentricity check
        dist_bm = float(np.linalg.norm(pos_m[:2] - pos_b[:2]))
        dist_mt = float(np.linalg.norm(pos_t[:2] - pos_m[:2]))
        is_concentric = dist_bm < self.max_concentric_error_m and dist_mt < self.max_concentric_error_m

        # 2. Nesting height check
        dz_bm = float(pos_m[2] - pos_b[2])
        dz_mt = float(pos_t[2] - pos_m[2])
        is_nested = (
            self.min_nesting_delta_m <= dz_bm <= self.max_nesting_delta_m
            and self.min_nesting_delta_m <= dz_mt <= self.max_nesting_delta_m
        )

        # 3. Gripper release check
        is_released = self._is_gripper_disengaged()

        # 4. Velocities check (stability)
        vel_b = np.linalg.norm(self.bowl_velocity(bottom))
        vel_m = np.linalg.norm(self.bowl_velocity(middle))
        vel_t = np.linalg.norm(self.bowl_velocity(top))
        is_stable = max(vel_b, vel_m, vel_t) < 0.05

        current_time = float(self.backend.data.time)
        is_stacked = is_concentric and is_nested and is_released and is_stable

        if is_stacked:
            if self._hold_start_time is None:
                self._hold_start_time = current_time
            hold_time = current_time - self._hold_start_time
        else:
            self._hold_start_time = None
            hold_time = 0.0

        success = hold_time >= self.success_hold_s
        elapsed = current_time - self._start_time
        timeout = elapsed >= self.timeout_s

        if timeout and not success:
            self._failure_reason = "timeout"

        terminated = success or timeout
        reason = "success" if success else self._failure_reason

        return NexArmStackBowlsStatus(
            success=success,
            terminated=terminated,
            reason=reason,
            hold_time_s=hold_time,
            order=self.current_order,
            is_concentric=is_concentric,
            is_nested_height=is_nested,
            is_released=is_released,
        )

    def observe(self) -> NexArmStackBowlsStatus:
        return self.status()

    def step(self, action: dict[str, float] | None = None) -> NexArmStackBowlsStatus:
        if action is None:
            action = self.backend.joint_positions()
        self.backend.step(action)
        return self.status()
