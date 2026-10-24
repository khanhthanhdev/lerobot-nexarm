# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from lerobot.motors.nexarm.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS

from .mujoco_backend import NexArmMujocoBackend

# Grasp-assist hysteresis on the commanded raw gripper target, inside GRIPPER_OPEN_POS..
# GRIPPER_CLOSED_POS. A command above the engage threshold (towards closed) may attach a bowl;
# one below the release threshold (towards open) drops it.
GRIPPER_ASSIST_ENGAGE_POS = 2400
GRIPPER_ASSIST_RELEASE_POS = 1600
# Real teleoperated releases open only to a median of 1989 (real-data report, Finding 6), so an env that should
# let go when a policy trained on real data opens the gripper releases at this higher value.
REAL_TELEOP_ASSIST_RELEASE_POS = 2300

COLORS = ("red", "blue", "black")
# Nominal bowl slots (x, y, z of the bowl body) on the table, before the seeded jitter. They are staggered so that
# bowls of radius ~62 mm stay at least MIN_BOWL_SEPARATION_M apart.
BOWL_SLOTS_M = ((-0.11, -0.27, 0.020), (0.00, -0.18, 0.020), (0.11, -0.27, 0.020))
MIN_BOWL_SEPARATION_M = 0.14
# The assist attaches a bowl only if its centre is this close to the gripper frame.
ASSIST_MAX_DISTANCE_M = 0.120
# A held bowl counts as picked once its centre is this far above where it rested at reset.
PICK_LIFT_M = 0.02
# The arm counts as static when no arm joint moves faster than this.
ARM_STATIC_MAX_SPEED_RAD_S = 0.1
# The task's stages, in the order a successful episode reaches them. Each latches once reached.
STAGES = ("picked_middle", "nested_middle", "released_middle", "picked_top", "nested_top", "released_top")
PERMUTATIONS: list[tuple[str, str, str]] = [(a, b, c) for a, b, c in itertools.permutations(COLORS)]
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
    # Diagnostic only; success does not require it.
    is_arm_static: bool = True
    # Each of STAGES, True once reached in this episode.
    stages: dict[str, bool] = field(default_factory=dict)
    # The first of STAGES not yet reached; None once all are.
    failed_stage: str | None = None


class NexArmStackBowlsTask:
    """Seeded 3-bowl stacking task on the 6-DOF NexArm platform."""

    def __init__(
        self,
        backend: NexArmMujocoBackend,
        *,
        max_concentric_error_m: float = 0.035,
        min_nesting_delta_m: float = 0.020,
        max_nesting_delta_m: float = 0.038,
        success_hold_s: float = 0.5,
        timeout_s: float = 40.0,
        grasp_mode: Literal["contact_assisted", "proximity_assisted", "physical"] = "contact_assisted",
        position_jitter_m: float = 0.005,
        assist_release_pos: float = GRIPPER_ASSIST_RELEASE_POS,
        assist_engage_pos: float = GRIPPER_ASSIST_ENGAGE_POS,
    ) -> None:
        if grasp_mode not in ("contact_assisted", "proximity_assisted", "physical"):
            raise ValueError(f"Unknown grasp mode: {grasp_mode}")
        if not np.isfinite(position_jitter_m) or not 0 <= position_jitter_m <= 0.05:
            raise ValueError("position_jitter_m must be between 0 and 0.05 meters")
        if not 0 < success_hold_s < timeout_s:
            raise ValueError("Require 0 < success_hold_s < timeout_s")
        if not GRIPPER_OPEN_POS < assist_engage_pos < GRIPPER_CLOSED_POS:
            raise ValueError(f"assist_engage_pos must be within ({GRIPPER_OPEN_POS}, {GRIPPER_CLOSED_POS})")
        # Real teleoperated releases often open only to about 1900-2000, so replaying them needs a higher value.
        if not GRIPPER_OPEN_POS <= assist_release_pos < assist_engage_pos:
            raise ValueError(f"assist_release_pos must be within [{GRIPPER_OPEN_POS}, {assist_engage_pos})")
        self.assist_release_pos = assist_release_pos
        self.assist_engage_pos = assist_engage_pos
        self.grasp_mode = grasp_mode
        self.position_jitter_m = position_jitter_m
        self.backend = backend
        self.max_concentric_error_m = max_concentric_error_m
        self.min_nesting_delta_m = min_nesting_delta_m
        self.max_nesting_delta_m = max_nesting_delta_m
        self.success_hold_s = success_hold_s
        self.timeout_s = timeout_s

        # Cache IDs
        self._body_ids = {c: self._required_id(mujoco.mjtObj.mjOBJ_BODY, f"bowl_{c}") for c in COLORS}
        self._arm_dof_ids = [
            int(backend.model.jnt_dofadr[joint_id])
            for name, joint_id in backend._joint_ids.items()
            if name != "gripper"
        ]
        self._joint_ids = {c: self._required_id(mujoco.mjtObj.mjOBJ_JOINT, f"bowl_{c}_joint") for c in COLORS}
        self._gripper_frame_site_id = self._required_id(mujoco.mjtObj.mjOBJ_SITE, "gripper_frame")
        self._left_jaw_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "link_6_left_jaw_collision_0")
        self._right_jaw_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "link_6_right_jaw_collision_0")
        # The thin rim and wall shell each bowl offers only to the jaws (it collides with nothing else).
        model = backend.model
        self._jaw_only_geom_ids = {
            c: [
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] == self._body_ids[c]
                and model.geom_contype[g]
                and not model.geom_conaffinity[g]
            ]
            for c in COLORS
        }
        self._jaw_only_contype = {
            c: model.geom_contype[ids].copy() for c, ids in self._jaw_only_geom_ids.items()
        }

        self.held_bowl: str | None = None
        self.held_rel_pos: np.ndarray | None = None
        self.held_rel_mat: np.ndarray | None = None
        self.target_bowl: str | None = None
        self.backend.step_callback = self._on_physics_step

        self.current_order: tuple[str, str, str] = PERMUTATIONS[0]
        self._start_time: float = 0.0
        self._hold_start_time: float | None = None
        self._failure_reason: str | None = None
        self._rest_z: dict[str, float] = {}
        self._stages: dict[str, bool] = dict.fromkeys(STAGES, False)

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
        if order is not None and order not in PERMUTATIONS:
            raise ValueError("order must contain red, blue, and black exactly once")
        self.backend.reset(settle_steps=0, rng=np.random.default_rng([seed, 1]))
        rng = np.random.default_rng(seed)

        # Select permutation (cycle uniformly through the 6 permutations by seed)
        if order is None:
            self.current_order = PERMUTATIONS[seed % len(PERMUTATIONS)]
        else:
            self.current_order = order

        # Spawn 3 bowls on table in non-overlapping reachable positions.
        # Reject overlapping layouts rather than allowing enlarged jitter to spawn collisions.
        for _ in range(1000):
            positions = np.array(BOWL_SLOTS_M)
            positions[:, :2] += rng.uniform(-self.position_jitter_m, self.position_jitter_m, (3, 2))
            if all(
                np.linalg.norm(positions[i, :2] - positions[j, :2]) >= MIN_BOWL_SEPARATION_M
                for i, j in itertools.combinations(range(3), 2)
            ):
                break
        else:
            raise RuntimeError("Could not sample a non-overlapping bowl layout")

        # Shuffle slot assignment
        slot_order = rng.permutation(3)

        for i, color in enumerate(COLORS):
            slot = slot_order[i]
            yaw = rng.uniform(-np.pi, np.pi)
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

            qpos_adr = int(self.backend.model.jnt_qposadr[self._joint_ids[color]])
            pos = positions[slot]
            self.backend.data.qpos[qpos_adr : qpos_adr + 3] = pos
            self.backend.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

        # Forward kinematics
        mujoco.mj_forward(self.backend.model, self.backend.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.backend.model, self.backend.data)

        self._start_time = float(self.backend.data.time)
        self._hold_start_time = None
        self._failure_reason = None
        self.held_bowl = None
        self.held_rel_pos = None
        self.held_rel_mat = None
        for color in COLORS:
            self._set_jaw_contacts(color, enabled=True)
        self.target_bowl = None
        self._rest_z = {color: float(self.bowl_position(color)[2]) for color in COLORS}
        self._stages = dict.fromkeys(STAGES, False)
        return self.status()

    def _on_physics_step(self) -> None:
        """Track or assist the grasp.

        ``physical`` only tracks it: the bowl is a free body held by jaw friction alone. The assisted modes
        carry a grasped bowl rigidly. Contact mode requires both jaws to touch the same bowl; proximity
        mode preserves the legacy behavior for comparison, not grasp validation.
        """
        gripper_ctrl = self.backend.control_to_raw(
            "gripper",
            float(self.backend.data.ctrl[self.backend._actuator_ids["gripper"]]),
        )
        site_id = self._gripper_frame_site_id
        site_pos = self.backend.data.site_xpos[site_id].copy()
        site_mat = self.backend.data.site_xmat[site_id].reshape(3, 3).copy()

        if self.grasp_mode == "physical":
            self._track_physical_grasp(gripper_ctrl, site_pos, site_mat)
            return

        if gripper_ctrl > self.assist_engage_pos:  # commanded towards closed
            if self.held_bowl is None:
                candidate_colors = [self.target_bowl] if self.target_bowl is not None else COLORS
                closest_c = None
                closest_dist = 999.0
                contacts = self._bowl_jaw_contacts()
                for c in candidate_colors:
                    if self.grasp_mode == "contact_assisted" and len(contacts[c]) != 2:
                        continue
                    bpos = self.bowl_position(c)
                    d = float(np.linalg.norm(bpos - site_pos))
                    if d < closest_dist:
                        closest_dist = d
                        closest_c = c
                if closest_dist < ASSIST_MAX_DISTANCE_M and closest_c is not None:
                    self.held_bowl = closest_c
                    bpos = self.bowl_position(self.held_bowl)
                    self.held_rel_pos = site_mat.T @ (bpos - site_pos)
                    bowl_mat = self.backend.data.xmat[self._body_ids[self.held_bowl]].reshape(3, 3)
                    self.held_rel_mat = site_mat.T @ bowl_mat
                    # The assist stands in for the grip. Pinned to the gripper, the bowl would turn any jaw overlap
                    # with its shell into a constant force that holds the arm off its target.
                    self._set_jaw_contacts(self.held_bowl, enabled=False)

            if self.held_bowl is not None:
                qadr = int(self.backend.model.jnt_qposadr[self._joint_ids[self.held_bowl]])
                dofadr = int(self.backend.model.jnt_dofadr[self._joint_ids[self.held_bowl]])
                target_bpos = site_pos + site_mat @ self.held_rel_pos
                self.backend.data.qpos[qadr : qadr + 3] = target_bpos
                # Carry the bowl rigidly: it turns with the gripper, as a held bowl does, so the gripper
                # never swings into its wall.
                if self.held_rel_mat is not None:
                    mujoco.mju_mat2Quat(
                        self.backend.data.qpos[qadr + 3 : qadr + 7], (site_mat @ self.held_rel_mat).flatten()
                    )
                self.backend.data.qvel[dofadr : dofadr + 6] = 0
        elif gripper_ctrl < self.assist_release_pos:  # commanded towards open
            if self.held_bowl is not None:
                self._set_jaw_contacts(self.held_bowl, enabled=True)
                self.held_bowl = None
                self.held_rel_pos = None
                self.held_rel_mat = None

    def _track_physical_grasp(self, gripper_ctrl: float, site_pos: np.ndarray, site_mat: np.ndarray) -> None:
        """A bowl is held while the gripper is commanded closed and a jaw still touches it.

        It is grasped when both jaws touch it, and dropped when the command opens or the jaws lose it. The
        relative pose is re-measured every step, so it follows the bowl as it slips or swings in the jaws.
        """
        if gripper_ctrl < self.assist_release_pos:
            self.held_bowl = None
        else:
            contacts = self._bowl_jaw_contacts()
            if self.held_bowl is not None and not contacts[self.held_bowl]:
                self.held_bowl = None
            if self.held_bowl is None and gripper_ctrl > self.assist_engage_pos:
                candidates = [self.target_bowl] if self.target_bowl is not None else COLORS
                self.held_bowl = next((c for c in candidates if len(contacts[c]) == 2), None)
        if self.held_bowl is None:
            self.held_rel_pos = None
            self.held_rel_mat = None
            return
        body = self._body_ids[self.held_bowl]
        self.held_rel_pos = site_mat.T @ (self.bowl_position(self.held_bowl) - site_pos)
        self.held_rel_mat = site_mat.T @ self.backend.data.xmat[body].reshape(3, 3)

    def _set_jaw_contacts(self, color: str, *, enabled: bool) -> None:
        ids = self._jaw_only_geom_ids[color]
        self.backend.model.geom_contype[ids] = self._jaw_only_contype[color] if enabled else 0

    def _bowl_jaw_contacts(self) -> dict[str, set[int]]:
        """Return penetrating jaw contacts grouped by bowl (ignores contact margins)."""
        contacts: dict[str, set[int]] = {color: set() for color in COLORS}
        jaws = {self._left_jaw_geom_id, self._right_jaw_geom_id}
        bodies = {body_id: color for color, body_id in self._body_ids.items()}
        for con in self.backend.data.contact[: self.backend.data.ncon]:
            if con.dist > 0:
                continue
            for jaw, other in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
                if jaw in jaws:
                    color = bodies.get(int(self.backend.model.geom_bodyid[other]))
                    if color is not None:
                        contacts[color].add(int(jaw))
        return contacts

    def _is_gripper_disengaged(self) -> bool:
        """Require no assisted attachment and no jaw contact with any bowl."""
        return self.held_bowl is None and not any(self._bowl_jaw_contacts().values())

    def _is_concentric(self, lower: str, upper: str) -> bool:
        offset = self.bowl_position(upper)[:2] - self.bowl_position(lower)[:2]
        return float(np.linalg.norm(offset)) < self.max_concentric_error_m

    def _is_nested_height(self, lower: str, upper: str) -> bool:
        dz = float(self.bowl_position(upper)[2] - self.bowl_position(lower)[2])
        return self.min_nesting_delta_m <= dz <= self.max_nesting_delta_m

    def is_nested(self, lower: str, upper: str) -> bool:
        """Whether bowl ``upper`` sits centred in bowl ``lower`` at the nesting height."""
        return self._is_concentric(lower, upper) and self._is_nested_height(lower, upper)

    def _is_picked(self, color: str) -> bool:
        lift = float(self.bowl_position(color)[2]) - self._rest_z.get(color, np.inf)
        return self.held_bowl == color and lift >= PICK_LIFT_M

    def _update_stages(self, is_released: bool) -> None:
        bottom, middle, top = self.current_order
        middle_nested = self.is_nested(bottom, middle)
        top_nested = middle_nested and self.is_nested(middle, top)
        reached = {
            "picked_middle": self._is_picked(middle),
            "nested_middle": middle_nested,
            "released_middle": middle_nested
            and self.held_bowl != middle
            and not self._bowl_jaw_contacts()[middle],
            "picked_top": self._is_picked(top),
            "nested_top": top_nested,
            "released_top": top_nested and is_released,
        }
        for stage, now in reached.items():
            self._stages[stage] = self._stages[stage] or now

    def _is_arm_static(self) -> bool:
        speeds = np.abs(self.backend.data.qvel[self._arm_dof_ids])
        return bool(speeds.max(initial=0.0) < ARM_STATIC_MAX_SPEED_RAD_S)

    def status(self) -> NexArmStackBowlsStatus:
        bottom, middle, top = self.current_order

        # 1. Concentricity check
        is_concentric = self._is_concentric(bottom, middle) and self._is_concentric(middle, top)

        # 2. Nesting height check
        is_nested = self._is_nested_height(bottom, middle) and self._is_nested_height(middle, top)

        # 3. Gripper release check
        is_released = self._is_gripper_disengaged()
        self._update_stages(is_released)

        # 4. Velocities check (stability)
        vel_b = np.linalg.norm(self.bowl_velocity(bottom))
        vel_m = np.linalg.norm(self.bowl_velocity(middle))
        vel_t = np.linalg.norm(self.bowl_velocity(top))
        is_stable = max(vel_b, vel_m, vel_t) < 0.05
        # A spinning or tipped stack must not pass just because its centers align.
        for color in COLORS:
            body_id = self._body_ids[color]
            angular_speed = np.linalg.norm(self.backend.data.cvel[body_id, :3])
            upright = self.backend.data.xmat[body_id].reshape(3, 3)[2, 2] > np.cos(np.deg2rad(15))
            is_stable = is_stable and angular_speed < 0.5 and upright

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
            is_arm_static=self._is_arm_static(),
            stages=dict(self._stages),
            failed_stage=next((stage for stage in STAGES if not self._stages[stage]), None),
        )

    def observe(self) -> NexArmStackBowlsStatus:
        return self.status()

    def step(self, action: dict[str, float] | None = None) -> NexArmStackBowlsStatus:
        if action is None:
            action = self.backend.joint_positions()
        self.backend.step(action)
        return self.status()
