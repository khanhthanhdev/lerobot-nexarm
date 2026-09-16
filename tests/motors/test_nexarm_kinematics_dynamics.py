# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

# ruff: noqa: N806

from __future__ import annotations

import numpy as np
import pytest

from lerobot.motors.nexarm import NexArmKinematicsDynamics
from lerobot.motors.nexarm.kinematics_dynamics import HOME_POSITIONS, JOINT_NAMES


@pytest.fixture
def kd() -> NexArmKinematicsDynamics:
    return NexArmKinematicsDynamics()


def test_forward_kinematics_computes_valid_pose(kd: NexArmKinematicsDynamics) -> None:
    fk = kd.forward_kinematics(HOME_POSITIONS)

    assert "position" in fk
    assert "rotation_matrix" in fk
    assert "quaternion_wxyz" in fk
    assert "rpy_rad" in fk

    pos = fk["position"]
    assert pos.shape == (3,)
    # End-effector should be above ground and in front of base
    assert pos[2] > 0.1

    quat = fk["quaternion_wxyz"]
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-4)

    R = fk["rotation_matrix"]
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-4)


def test_jacobian_dimensions_and_rank(kd: NexArmKinematicsDynamics) -> None:
    J = kd.compute_jacobian(HOME_POSITIONS)

    # 6-DOF spatial velocity (3 trans + 3 rot) for 5 arm joints
    assert J.shape == (6, 5)
    # Should be full column rank (5) at home posture
    rank = np.linalg.matrix_rank(J)
    assert rank == 5


def test_inverse_kinematics_recovers_reachable_target(kd: NexArmKinematicsDynamics) -> None:
    # 1. Forward pass from a known realistic posture
    test_q = {name: HOME_POSITIONS[name] + 50.0 for name in JOINT_NAMES}
    fk_target = kd.forward_kinematics(test_q)
    target_xyz = fk_target["position"]

    # 2. Solve IK starting from home
    solution = kd.inverse_kinematics(
        target_xyz,
        current_joint_positions=HOME_POSITIONS,
        tolerance_m=0.002,
    )

    assert solution is not None
    # 3. Check forward kinematics of solution reaches target
    fk_solved = kd.forward_kinematics(solution)
    error = np.linalg.norm(fk_solved["position"] - target_xyz)
    assert error <= 0.003


def test_gravity_compensation_torques(kd: NexArmKinematicsDynamics) -> None:
    torques = kd.gravity_compensation_torques(HOME_POSITIONS)

    assert len(torques) == 5
    for name in JOINT_NAMES[:-1]:
        assert name in torques
        assert isinstance(torques[name], float)

    # Shoulder lift (joint 2) and elbow flex (joint 3) must bear arm weight against gravity
    # Total gravity torque should be non-zero
    total_magnitude = sum(abs(v) for v in torques.values())
    assert total_magnitude > 0.05


def test_mass_matrix_positive_definite(kd: NexArmKinematicsDynamics) -> None:
    M = kd.mass_matrix(HOME_POSITIONS)

    assert M.shape == (5, 5)
    # Mass matrix must be symmetric positive-definite
    assert np.allclose(M, M.T, atol=1e-5)
    eigenvalues = np.linalg.eigvalsh(M)
    assert np.all(eigenvalues > 0.0)
