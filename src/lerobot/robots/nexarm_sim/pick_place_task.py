# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .mujoco_backend import RAW_RANGES, NexArmMujocoBackend


@dataclass(frozen=True)
class NexArmPickPlaceStatus:
    """State of the single-arm tabletop pick-and-place task."""

    success: bool
    terminated: bool
    reason: str | None
    hold_time_s: float
    is_grasped: bool
    is_released: bool
    is_inside_target: bool


class NexArmPickPlaceTask:
    """Seeded pick/place task layered on the existing six-DOF NexArm backend."""

    def __init__(
        self,
        backend: NexArmMujocoBackend,
        *,
        target_radius_m: float = 0.05,
        success_hold_s: float = 0.5,
        timeout_s: float = 20.0,
    ) -> None:
        if target_radius_m <= 0:
            raise ValueError("target_radius_m must be positive")
        if success_hold_s <= 0:
            raise ValueError("success_hold_s must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.backend = backend
        self.target_radius_m = target_radius_m
        self.success_hold_s = success_hold_s
        self.timeout_s = timeout_s
        self._cube_joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._cube_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._target_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        self._cube_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "cube_collision")
        self._cube_half_size_m = float(self.backend.model.geom_size[self._cube_geom_id, 0])
        self._left_jaw_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "link_6_left_jaw_collision_0")
        self._right_jaw_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "link_6_right_jaw_collision_0")
        cube_qpos = int(self.backend.model.jnt_qposadr[self._cube_joint_id])
        self._cube_spawn_xy = self.backend.model.qpos0[cube_qpos : cube_qpos + 2].copy()
        self._target_spawn_xy = self.backend.model.body_pos[self._target_body_id, :2].copy()
        self._start_time = 0.0
        self._hold_start_time: float | None = None
        self._failure_reason: str | None = None

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.backend.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"NexArm task model is missing {object_type.name}: {name}")
        return object_id

    @property
    def cube_position(self) -> np.ndarray:
        return self.backend.data.xpos[self._cube_body_id].copy()

    @property
    def target_position(self) -> np.ndarray:
        return self.backend.data.xpos[self._target_body_id].copy()

    def reset(self, *, seed: int = 0, settle_steps: int = 0) -> NexArmPickPlaceStatus:
        self.backend.reset(settle_steps=0)
        rng = np.random.default_rng(seed)
        # Randomize relative to the poses declared by the scene instead of
        # assuming the original Fusion world coordinates. This keeps the task
        # valid when the robot is mounted or the scene is recentered.
        cube_center = self._cube_spawn_xy + np.array([0.001, 0.01])
        target_center = self._target_spawn_xy + np.array([0.0, 0.01])
        cube_xy = cube_center + rng.uniform((-0.03, -0.04), (0.03, 0.04))
        target_xy = target_center + rng.uniform((-0.03, -0.04), (0.03, 0.04))
        if np.linalg.norm(cube_xy - target_xy) < self.target_radius_m + 0.04:
            target_xy[0] = target_center[0] - 0.03
        cube_qpos = int(self.backend.model.jnt_qposadr[self._cube_joint_id])
        self.backend.data.qpos[cube_qpos : cube_qpos + 7] = [
            cube_xy[0],
            cube_xy[1],
            self._cube_half_size_m,
            1,
            0,
            0,
            0,
        ]
        self.backend.model.body_pos[self._target_body_id, :3] = [
            target_xy[0],
            target_xy[1],
            0.002,
        ]
        mujoco.mj_forward(self.backend.model, self.backend.data)
        for _ in range(settle_steps):
            self.backend.step()
        self._start_time = float(self.backend.data.time)
        self._hold_start_time = None
        self._failure_reason = None
        return self.status()

    def step(self) -> NexArmPickPlaceStatus:
        """Advance one control period using the current MuJoCo actuator targets."""
        self.backend.step()
        return self.observe()

    def observe(self) -> NexArmPickPlaceStatus:
        """Update the success/failure gate without advancing physics."""
        cube = self.cube_position
        target = self.target_position
        inside = (
            np.linalg.norm(cube[:2] - target[:2]) <= self.target_radius_m - self._cube_half_size_m
            and cube[2] >= 0.8 * self._cube_half_size_m
        )
        released = self._is_released()
        cube_dof = int(self.backend.model.jnt_dofadr[self._cube_joint_id])
        cube_speed = np.linalg.norm(self.backend.data.qvel[cube_dof : cube_dof + 3])
        if inside and released and cube_speed < 0.05:
            if self._hold_start_time is None:
                self._hold_start_time = float(self.backend.data.time)
        else:
            self._hold_start_time = None
        if cube[2] < -0.02:
            self._failure_reason = "dropped_cube"
        elif self.backend.data.time - self._start_time >= self.timeout_s:
            self._failure_reason = "unreleased_grasp" if inside and not released else "timeout"
        return self.status()

    def status(self) -> NexArmPickPlaceStatus:
        cube = self.cube_position
        target = self.target_position
        inside = (
            np.linalg.norm(cube[:2] - target[:2]) <= self.target_radius_m - self._cube_half_size_m
            and cube[2] >= 0.8 * self._cube_half_size_m
        )
        released = self._is_released()
        hold_time = (
            0.0 if self._hold_start_time is None else float(self.backend.data.time - self._hold_start_time)
        )
        success = hold_time >= self.success_hold_s
        reason = "success" if success else self._failure_reason
        return NexArmPickPlaceStatus(
            success=success,
            terminated=success or self._failure_reason is not None,
            reason=reason,
            hold_time_s=hold_time,
            is_grasped=self._is_grasped(),
            is_released=released,
            is_inside_target=inside,
        )

    def _is_released(self) -> bool:
        gripper_joint_id = self.backend._joint_ids["gripper"]
        qpos = float(self.backend.data.qpos[self.backend.model.jnt_qposadr[gripper_joint_id]])
        raw = self.backend.control_to_raw("gripper", qpos)
        low, high = RAW_RANGES["gripper"]
        return raw <= low + 0.15 * (high - low)

    def _is_grasped(self) -> bool:
        touched: set[int] = set()
        for index in range(self.backend.data.ncon):
            contact = self.backend.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self._cube_geom_id not in pair:
                continue
            touched.update(pair)
        return self._left_jaw_geom_id in touched and self._right_jaw_geom_id in touched
