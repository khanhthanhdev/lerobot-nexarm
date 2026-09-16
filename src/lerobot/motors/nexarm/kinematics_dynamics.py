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

"""C-accelerated 6-DOF Kinematics, Jacobian & Dynamics for NexArm.

Provides Forward Kinematics (FK), Full Spatial Jacobian (6x5), 6-DOF Inverse
Kinematics (IK) with position + orientation targets, Mass Matrix (M(q)), and
Gravity Compensation Torques (g(q) = tau_grav) for compliant control.
"""

# ruff: noqa: N803, N806

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from lerobot.motors.nexarm.nexarm import JOINT_NAMES, POSITION_MAX, POSITION_MIN

MUJOCO_JOINTS = {
    "shoulder_pan": "joint_1_base_to_link_1",
    "shoulder_lift": "joint_2_link_1_to_link_2",
    "elbow_flex": "joint_3_link_2_to_link_3",
    "wrist_flex": "joint_4_link_3_to_link_4",
    "wrist_roll": "joint_5_link_4_to_link_5",
    "gripper": "right_jaw_slide_joint",
}

RAW_RANGES: dict[str, tuple[int, int]] = dict.fromkeys(JOINT_NAMES[:-1], (POSITION_MIN, POSITION_MAX))
RAW_RANGES["gripper"] = (1195, 2833)

HOME_POSITIONS: dict[str, float] = dict.fromkeys(JOINT_NAMES[:-1], 2048.0)
HOME_POSITIONS["gripper"] = 2833.0


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Compute 3x3 rotation matrix from Roll-Pitch-Yaw angles in radians."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def rotation_matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Extract Roll-Pitch-Yaw angles (radians) from 3x3 rotation matrix."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


class NexArmKinematicsDynamics:
    """Kinematics, Jacobian, and Generalized Dynamics solver for NexArm."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        site_name: str = "gripper_frame",
    ) -> None:
        if model_path is None:
            checkout_root = Path(__file__).resolve().parents[4]
            model_path = checkout_root / "sim" / "fusion_export" / "scene.xml"
        model_path = Path(model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"NexArm model file not found: {model_path}")

        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site {site_name!r} not found in model {model_path}")

        self._joint_ids: dict[str, int] = {}
        for feature_name, jname in MUJOCO_JOINTS.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Joint {jname!r} not found in model")
            self._joint_ids[feature_name] = jid

        self.arm_names = JOINT_NAMES[:-1]
        self.qpos_addrs = [int(self.model.jnt_qposadr[self._joint_ids[name]]) for name in self.arm_names]
        self.dof_addrs = [int(self.model.jnt_dofadr[self._joint_ids[name]]) for name in self.arm_names]

    def raw_to_control(self, feature_name: str, raw_position: float) -> float:
        raw_low, raw_high = RAW_RANGES[feature_name]
        raw_position = float(np.clip(raw_position, raw_low, raw_high))
        jid = self._joint_ids[feature_name]
        low, high = self.model.jnt_range[jid]
        ratio = (raw_position - raw_low) / (raw_high - raw_low)
        return float(low + ratio * (high - low))

    def control_to_raw(self, feature_name: str, control_position: float) -> float:
        jid = self._joint_ids[feature_name]
        low, high = self.model.jnt_range[jid]
        control_position = float(np.clip(control_position, low, high))
        raw_low, raw_high = RAW_RANGES[feature_name]
        ratio = (control_position - low) / (high - low)
        return float(raw_low + ratio * (raw_high - raw_low))

    def _set_qpos(self, joint_positions: Mapping[str, float] | npt.ArrayLike) -> None:
        if isinstance(joint_positions, Mapping):
            for name, qpos_addr in zip(self.arm_names, self.qpos_addrs, strict=True):
                raw_val = joint_positions.get(f"{name}.pos", joint_positions.get(name, HOME_POSITIONS[name]))
                self.data.qpos[qpos_addr] = self.raw_to_control(name, float(raw_val))
            if "gripper.pos" in joint_positions or "gripper" in joint_positions:
                grip_val = joint_positions.get(
                    "gripper.pos", joint_positions.get("gripper", HOME_POSITIONS["gripper"])
                )
                grip_jid = self._joint_ids["gripper"]
                self.data.qpos[self.model.jnt_qposadr[grip_jid]] = self.raw_to_control(
                    "gripper", float(grip_val)
                )
        else:
            arr = np.asarray(joint_positions, dtype=np.float64)
            for i, qpos_addr in enumerate(self.qpos_addrs):
                self.data.qpos[qpos_addr] = arr[i]

    def forward_kinematics(
        self,
        joint_positions: Mapping[str, float] | npt.ArrayLike,
    ) -> dict[str, Any]:
        """Compute end-effector Cartesian pose (position, rotation matrix, RPY, quat)."""
        self._set_qpos(joint_positions)
        mujoco.mj_forward(self.model, self.data)

        pos = self.data.site_xpos[self.site_id].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        rpy_rad = rotation_matrix_to_rpy(R)
        rpy_deg = np.rad2deg(rpy_rad)

        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])

        return {
            "position": pos,
            "rotation_matrix": R,
            "quaternion_wxyz": quat,
            "rpy_rad": rpy_rad,
            "rpy_deg": rpy_deg,
        }

    def compute_jacobian(
        self,
        joint_positions: Mapping[str, float] | npt.ArrayLike,
    ) -> np.ndarray:
        """Compute full 6x5 geometric Jacobian (3 translation + 3 rotation)."""
        self._set_qpos(joint_positions)
        mujoco.mj_forward(self.model, self.data)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

        J_pos = jacp[:, self.dof_addrs]
        J_rot = jacr[:, self.dof_addrs]
        return np.vstack([J_pos, J_rot])

    def gravity_compensation_torques(
        self,
        joint_positions: Mapping[str, float] | npt.ArrayLike,
    ) -> dict[str, float]:
        """Compute static gravity compensation torques tau_grav = g(q) in Nm."""
        self._set_qpos(joint_positions)
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        torques: dict[str, float] = {}
        for name, dof_addr in zip(self.arm_names, self.dof_addrs, strict=True):
            torques[name] = float(self.data.qfrc_bias[dof_addr])
        return torques

    def mass_matrix(
        self,
        joint_positions: Mapping[str, float] | npt.ArrayLike,
    ) -> np.ndarray:
        """Compute generalized inertia matrix M(q) for the 5 arm joints."""
        self._set_qpos(joint_positions)
        mujoco.mj_forward(self.model, self.data)

        M_full = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, M_full, self.data.qM)
        return M_full[np.ix_(self.dof_addrs, self.dof_addrs)].copy()

    def inverse_kinematics(
        self,
        target_xyz: npt.ArrayLike,
        target_rpy: npt.ArrayLike | None = None,
        current_joint_positions: Mapping[str, float] | None = None,
        *,
        tolerance_m: float = 0.002,
        tolerance_rad: float = 0.05,
        max_iterations: int = 150,
        damping: float = 1e-3,
        nullspace_weight: float = 0.1,
    ) -> dict[str, float] | None:
        """Solve 6-DOF Inverse Kinematics using Damped Least Squares."""
        target_pos = np.asarray(target_xyz, dtype=np.float64)
        if target_pos.shape != (3,):
            raise ValueError(f"target_xyz must have shape (3,), got {target_pos.shape}")

        working = mujoco.MjData(self.model)
        if current_joint_positions is not None:
            for name, qpos_addr in zip(self.arm_names, self.qpos_addrs, strict=True):
                raw = current_joint_positions.get(
                    f"{name}.pos", current_joint_positions.get(name, HOME_POSITIONS[name])
                )
                working.qpos[qpos_addr] = self.raw_to_control(name, float(raw))
        else:
            for name, qpos_addr in zip(self.arm_names, self.qpos_addrs, strict=True):
                working.qpos[qpos_addr] = self.raw_to_control(name, HOME_POSITIONS[name])

        target_R = None
        if target_rpy is not None:
            rpy = np.asarray(target_rpy, dtype=np.float64)
            target_R = rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])

        q_home = np.array([self.raw_to_control(name, HOME_POSITIONS[name]) for name in self.arm_names])

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, working)
            cur_pos = working.site_xpos[self.site_id]
            pos_err = target_pos - cur_pos

            mujoco.mj_jacSite(self.model, working, jacp, jacr, self.site_id)
            J_pos = jacp[:, self.dof_addrs]

            if target_R is not None:
                cur_R = working.site_xmat[self.site_id].reshape(3, 3)
                rot_err = 0.5 * (
                    np.cross(cur_R[:, 0], target_R[:, 0])
                    + np.cross(cur_R[:, 1], target_R[:, 1])
                    + np.cross(cur_R[:, 2], target_R[:, 2])
                )
                err = np.concatenate([pos_err, rot_err * 0.4])
                J = np.vstack([J_pos, jacr[:, self.dof_addrs] * 0.4])
                pos_conv = np.linalg.norm(pos_err) <= tolerance_m
                rot_conv = np.linalg.norm(rot_err) <= tolerance_rad
                if pos_conv and rot_conv:
                    return {
                        name: self.control_to_raw(name, float(working.qpos[addr]))
                        for name, addr in zip(self.arm_names, self.qpos_addrs, strict=True)
                    }
            else:
                err = pos_err
                J = J_pos
                if np.linalg.norm(pos_err) <= tolerance_m:
                    return {
                        name: self.control_to_raw(name, float(working.qpos[addr]))
                        for name, addr in zip(self.arm_names, self.qpos_addrs, strict=True)
                    }

            # Damped Least-Squares solve: delta_q = J^T (J J^T + lambda^2 I)^-1 err
            n_rows = J.shape[0]
            lambda_ident = damping * np.eye(n_rows)
            delta_primary = J.T @ np.linalg.solve(J @ J.T + lambda_ident, err)

            # Null-space posture projection: (I - J# J) * (q_home - q)
            q_current = np.array([working.qpos[addr] for addr in self.qpos_addrs])
            nullspace_proj = np.eye(5) - np.linalg.pinv(J) @ J
            delta_null = nullspace_weight * (nullspace_proj @ (q_home - q_current))

            delta = np.clip(delta_primary + delta_null, -0.15, 0.15)

            for i, (name, addr) in enumerate(zip(self.arm_names, self.qpos_addrs, strict=True)):
                low, high = self.model.jnt_range[self._joint_ids[name]]
                working.qpos[addr] = np.clip(working.qpos[addr] + delta[i], low, high)

        return None
