#!/usr/bin/env python3

"""Generate successful scripted NexArm 3-Bowl Stacking episodes as a LeRobot dataset.

Example:
    uv run python examples/nexarm/generate_stack_bowls_dataset.py \
        --repo-id local/nexarm_stack_bowls \
        --root outputs/datasets/nexarm_stack_bowls \
        --episodes 50 --position-jitter-m 0.02

Calibrated to a real arm (see calibrate_sim.py), sharded over 2 GPUs, held-out eval split:
    uv run python examples/nexarm/generate_stack_bowls_dataset.py --calibration outputs/calibration/nexarm.json \
        --episodes 200 --gpus 0,1 --split eval

Domain randomization (visual, dynamics and every bowl's friction/mass/color) is on by default; --no-dr
disables it. Grasps are physical by default: the jaws hold each bowl by friction alone. Use
--grasp-mode contact_assisted to carry a bowl rigidly once both jaws touch it, and proximity_assisted only
to reproduce the legacy shortcut.
Provenance goes to ``generation_report.json`` and one line per attempt to ``generation_episodes.jsonl``
at the dataset root, as in generate_sim_dataset.py.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

# Select an offscreen renderer before MuJoCo/GLFW are imported on headless Linux hosts.
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

try:  # imported as a package module (tests) or run as a script
    from examples.nexarm import sim_dataset_utils as utils
except ModuleNotFoundError:
    import sim_dataset_utils as utils  # type: ignore[no-redef]

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot_robot_nexarm.motors.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS, JOINT_NAMES
from lerobot_robot_nexarm.nexarm_sim import (
    NexArmSim,
    NexArmSimConfig,
    NexArmStackBowlsTask,
    get_task_instruction,
)
from lerobot_robot_nexarm.nexarm_sim.mujoco_backend import HOME_POSITIONS, MUJOCO_JOINTS, resolve_model_path
from lerobot_robot_nexarm.nexarm_sim.stack_bowls_task import (
    GRIPPER_ASSIST_ENGAGE_POS,
    GRIPPER_ASSIST_RELEASE_POS,
    PERMUTATIONS,
)

OPEN_GRIPPER = float(GRIPPER_OPEN_POS)  # wide open: 67.4mm gap
CLOSED_GRIPPER = float(GRIPPER_CLOSED_POS)  # pinched closed: 16.7mm gap
BOWL_RIM_RADIUS = 0.0615  # Rim wall mid-plane radius in meters
BOWL_RIM_Z_OFFSET = 0.0185  # Height of the rim top above bowl body center
BOWL_RIM_OUTER_RADIUS = 0.0625  # Outside face of the rim wall
# Grasp points tried around the rim, nearest the far side first, and the gap the open jaws must keep from any
# other bowl's rim there. The jaws meet the wall obliquely off the far side, yet forcing every grasp to any one
# angle from 20 to 40 degrees either way still stacks 58-62 of 64 seeds, as many as the far side does.
GRASP_RIM_ANGLES_RAD = np.deg2rad([0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 40.0, -40.0])
JAW_CLEARANCE_M = 0.005
# The jaws come down almost vertically onto the thin rim, as in the real demos; fully vertical (90) leaves far
# bowls out of reach and fails more often.
GRASP_PITCH_DEG = 80.0
JAW_CENTER_OFFSET = np.array([0.0, 0.033, 0.0])  # Jaw collision pad center in gripper_frame
# The arm sags about 1.4 cm below its IK target at the rim. The descent aims that much higher and then re-aims
# on the measured jaw height, so the jaws never dip below the rim, where an open jaw can hook a neighbor bowl.
RIM_DESCENT_SAG_M = 0.014
RIM_DESCENT_ATTEMPTS = 4
RIM_DESCENT_TOLERANCE_M = 0.003
# Like a teleoperator, the expert moves in straight segments: up to a transit height, across, then down. In
# transit, the carried bowl's bottom (or the open jaws' tips) clears the highest rim of every other bowl by this
# much. Held by its rim top, the bowl hangs low in the jaws, so higher transits leave far placements out of reach.
TRANSIT_CLEARANCE_M = 0.065
JAW_TIP_BELOW_CENTER_M = 0.015  # Open jaw tips below the jaw pad center
CARTESIAN_SPACING_M = 0.02  # IK waypoint spacing along a straight segment
CARTESIAN_SPEED_M = 0.006  # Travel per frame along a segment
# Carried bowl center above the bowl it goes onto at release.
RELEASE_HEIGHT_M = 0.025


def _get_rim_and_unit_vector(bowl_pos: np.ndarray, angle_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute a rim point and the 2D radial unit direction from the robot base to it.

    ``angle_rad`` turns the point around the bowl from the far side, the rim point straight out from the base.
    """
    u = bowl_pos[:2] / np.linalg.norm(bowl_pos[:2])
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    outward = np.array([cos * u[0] - sin * u[1], sin * u[0] + cos * u[1]])
    rim_point = np.array(
        [
            bowl_pos[0] + BOWL_RIM_RADIUS * outward[0],
            bowl_pos[1] + BOWL_RIM_RADIUS * outward[1],
            bowl_pos[2] + BOWL_RIM_Z_OFFSET,
        ]
    )
    return rim_point, rim_point[:2] / np.linalg.norm(rim_point[:2])


def _open_jaw_room(backend, solution: dict[str, float], centers: list[np.ndarray]) -> float:
    """Horizontal gap between the open jaws at ``solution`` and the nearest rim of bowls centered at ``centers``."""
    model = backend.model
    data = mujoco.MjData(model)
    data.qpos[:] = _solution_qpos(backend, solution)
    data.qpos[model.jnt_qposadr[backend._joint_ids["gripper"]]] = backend.raw_to_control(
        "gripper", OPEN_GRIPPER
    )
    mujoco.mj_kinematics(model, data)
    room = np.inf
    for name in ("link_6_left_jaw_collision_0", "link_6_right_jaw_collision_0"):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        frame, center, half = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom], model.geom_size[geom]
        for bowl_center in centers:
            # Alternate projections between the jaw box and the bowl's vertical axis to find their closest points.
            point = center.copy()
            for _ in range(50):
                on_axis = np.array([bowl_center[0], bowl_center[1], point[2]])
                point = center + frame @ np.clip(frame.T @ (on_axis - center), -half, half)
            room = min(room, float(np.linalg.norm(point[:2] - bowl_center[:2])) - BOWL_RIM_OUTER_RADIUS)
    return room


def _grasp_rim(
    robot: NexArmSim, task: NexArmStackBowlsTask, source: str, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rim point to grasp ``source`` at, and the radial direction the jaws close along.

    The far side, straight out from the base, is the default. When the open outer jaw would land on another
    bowl's rim there (a finished stack beside the top bowl, say), the grasp moves around the rim, away from it,
    by the smallest angle that clears it, as an operator grabs a bowl where there is room for the fingers.
    """
    bowl = task.bowl_position(source)
    others = [task.bowl_position(c) for c in task.current_order if c != source]
    best, best_room = _get_rim_and_unit_vector(bowl), -np.inf
    for angle in GRASP_RIM_ANGLES_RAD:
        rim, u = _get_rim_and_unit_vector(bowl, angle)
        sol = solve_ik_jaw(robot.backend, rim, u, seed=seed)
        if not sol:
            continue
        room = _open_jaw_room(robot.backend, sol, others)
        if room >= JAW_CLEARANCE_M:
            return rim, u
        if room > best_room:
            best, best_room = (rim, u), room
    return best


def solve_ik_jaw(
    backend,
    target_pos: np.ndarray,
    u_rad: np.ndarray,
    pitch_deg: float = GRASP_PITCH_DEG,
    roll_val_rad: float = np.deg2rad(80),
    max_iterations: int = 150,
    tolerance_pos: float = 0.002,
    tolerance_dir: float = 0.05,
    restarts: int = 5,
    seed: int = 0,
    init_qpos: np.ndarray | None = None,
    point_offset: np.ndarray = JAW_CENTER_OFFSET,
) -> dict[str, float] | None:
    """Solve IK positioning the jaw collision center with radial rim straddle orientation.

    The gripper pitches downward at `pitch_deg` (with adaptive shallow fallbacks if needed)
    and rotates `wrist_roll` by ~80 degrees so the jaw closing axis is strictly radial
    across the thin rim wall. ``init_qpos`` starts the search there instead of at the current state, so the
    waypoints of one path solve to neighbouring joint configurations. ``point_offset`` (gripper frame) picks
    another point to position, such as a held bowl's center.
    """
    model = backend.model
    data = backend.data
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame")

    arm4 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
    joint_ids = {
        k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MUJOCO_JOINTS[k]) for k in JOINT_NAMES
    }
    qpos_addrs = [int(model.jnt_qposadr[joint_ids[k]]) for k in arm4]
    dof_addrs = [int(model.jnt_dofadr[joint_ids[k]]) for k in arm4]
    roll_qpos = int(model.jnt_qposadr[joint_ids["wrist_roll"]])

    target_pos = np.asarray(target_pos, dtype=np.float64)
    u_rad = np.asarray(u_rad, dtype=np.float64) / np.linalg.norm(u_rad)

    # Steeper is a squarer grip on the rim; far targets fall back to shallower pitches, down to 60 degrees.
    candidate_pitches = [pitch_deg - step for step in np.arange(0.0, max(pitch_deg - 60.0, 15.0) + 1e-6, 5.0)]
    for p in candidate_pitches:
        theta = np.deg2rad(p)
        target_dir = np.array([np.cos(theta) * u_rad[0], np.cos(theta) * u_rad[1], -np.sin(theta)])
        rng = np.random.default_rng(seed)

        for restart in range(restarts + 1):
            working = mujoco.MjData(model)
            working.qpos[:] = data.qpos if init_qpos is None else init_qpos
            working.qpos[roll_qpos] = roll_val_rad
            if restart:
                for k in arm4:
                    low, high = model.jnt_range[joint_ids[k]]
                    working.qpos[model.jnt_qposadr[joint_ids[k]]] = rng.uniform(low * 0.8, high * 0.8)

            for _ in range(max_iterations):
                mujoco.mj_forward(model, working)
                site_pos = working.site_xpos[site_id]
                site_mat = working.site_xmat[site_id].reshape(3, 3)
                current_jaw = site_pos + site_mat @ point_offset
                approach_dir = -site_mat[:, 1]
                pos_err = target_pos - current_jaw
                dir_err = np.cross(approach_dir, target_dir)

                if (
                    np.linalg.norm(pos_err) < tolerance_pos
                    and np.linalg.norm(approach_dir - target_dir) < tolerance_dir
                ):
                    sol = {
                        k: backend.control_to_raw(k, working.qpos[model.jnt_qposadr[joint_ids[k]]])
                        for k in arm4
                    }
                    sol["wrist_roll"] = backend.control_to_raw("wrist_roll", roll_val_rad)
                    return sol

                jac_p = np.zeros((3, model.nv), dtype=np.float64)
                jac_r = np.zeros((3, model.nv), dtype=np.float64)
                # Differentiate the jaw target, which is offset from the frame site.
                mujoco.mj_jac(model, working, jac_p, jac_r, current_jaw, int(model.site_bodyid[site_id]))
                jac = np.vstack([jac_p[:, dof_addrs], 0.2 * jac_r[:, dof_addrs]])
                damping = 1e-3
                err = np.concatenate([pos_err, 0.2 * dir_err])
                delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
                delta = np.clip(delta, -0.15, 0.15)
                for i, qaddr in enumerate(qpos_addrs):
                    low, high = model.jnt_range[joint_ids[arm4[i]]]
                    working.qpos[qaddr] = np.clip(working.qpos[qaddr] + delta[i], low, high)

    return None


def solve_ik_held_bowl(
    robot: NexArmSim,
    task: NexArmStackBowlsTask,
    target_bowl_xyz: np.ndarray,
    u_rad: np.ndarray,
    pitch_deg: float = GRASP_PITCH_DEG,
    seed: int = 0,
    init_qpos: np.ndarray | None = None,
) -> dict[str, float] | None:
    """Solve IK placing the held bowl's center, which moves rigidly with the gripper, at target_bowl_xyz."""
    if task.held_rel_pos is None:
        return None
    return solve_ik_jaw(
        robot.backend,
        target_bowl_xyz,
        u_rad,
        pitch_deg=pitch_deg,
        seed=seed,
        init_qpos=init_qpos,
        point_offset=task.held_rel_pos,
    )


def _solution_qpos(backend, solution: dict[str, float], base: np.ndarray | None = None) -> np.ndarray:
    """Full qpos with the solved joints set, the rest taken from ``base`` (default: the current state)."""
    qpos = np.array(backend.data.qpos if base is None else base, dtype=np.float64)
    for k, v in solution.items():
        qpos[backend.model.jnt_qposadr[backend._joint_ids[k]]] = backend.raw_to_control(k, v)
    return qpos


def _jaw_center(backend) -> np.ndarray:
    site_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame")
    site_mat = backend.data.site_xmat[site_id].reshape(3, 3)
    return backend.data.site_xpos[site_id] + site_mat @ JAW_CENTER_OFFSET


def _descend_to_rim(
    robot: NexArmSim,
    task: NexArmStackBowlsTask,
    rim: np.ndarray,
    u_rim: np.ndarray,
    *,
    seed: int = 0,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None = None,
) -> dict[str, float] | None:
    """Lower the open jaws onto the rim, re-aiming until the gravity-sagged arm really holds them there.

    Uncorrected, the closing inner jaw lands on the bowl's curved wall instead of straddling the vertical rim.
    """
    target = rim + [0, 0, RIM_DESCENT_SAG_M]
    sol = None
    for attempt in range(RIM_DESCENT_ATTEMPTS):
        sol = solve_ik_jaw(robot.backend, target, u_rim, seed=seed)
        if not sol:
            return None
        _interpolate_stage(
            robot,
            task,
            sol,
            OPEN_GRIPPER,
            OPEN_GRIPPER,
            steps=15 if attempt == 0 else 5,
            settle_steps=2,
            record_frame=record_frame,
        )
        sag = rim[2] - _jaw_center(robot.backend)[2]
        if abs(sag) < RIM_DESCENT_TOLERANCE_M:
            break
        target[2] += sag
    return sol


def _min_jerk(steps: int) -> np.ndarray:
    """Progress 0..1 over ``steps`` frames with zero start and end velocity, like a teleoperator's bell-shaped move."""
    t = np.linspace(0.0, 1.0, steps, endpoint=True)
    return t**3 * (10 - 15 * t + 6 * t**2)


def _interpolate_stage(
    robot: NexArmSim,
    task: NexArmStackBowlsTask,
    solution: dict[str, float],
    start_gripper: float,
    target_gripper: float,
    *,
    steps: int,
    settle_steps: int = 10,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None = None,
) -> bool:
    """Move joints to the target solution on a minimum-jerk profile and hold for settle_steps."""
    start = robot.backend.joint_positions()
    target = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
    target.update({f"{name}.pos": value for name, value in solution.items()})
    target["gripper.pos"] = target_gripper

    # Scale steps if large joint travel is required (prevent PD motor lag on long transits)
    max_delta = max(abs(start[k] - target[k]) for k in target if k != "gripper.pos")
    min_steps = int(np.ceil(max_delta / 25.0))
    effective_steps = max(steps, min_steps)

    for alpha in _min_jerk(effective_steps):
        action = {key: float((1 - alpha) * start[key] + alpha * target[key]) for key in target}
        action["gripper.pos"] = float((1 - alpha) * start_gripper + alpha * target_gripper)
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(action)
        if record_frame is not None:
            record_frame(observation, sent)
        if task.observe().terminated:
            # Stop here: settling would jump the command straight to the target in one frame.
            return True

    # Motor lag can leave a carried bowl centimeters from its IK target. Keep
    # commanding the target before releasing or starting the next motion.
    arm_keys = [k for k in target if k != "gripper.pos"]  # closed jaws stop on the rim, short of their target
    previous = robot.backend.joint_positions()
    for settle_index in range(max(settle_steps, robot.config.fps)):
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(target)
        if record_frame is not None:
            record_frame(observation, sent)
        if task.observe().terminated:
            break
        positions = robot.backend.joint_positions()
        arm_error = max(abs(positions[k] - target[k]) for k in arm_keys)
        # A loaded arm sags a few ticks under gravity and never closes the gap; stopped near the target is there.
        arm_stopped = max(abs(positions[k] - previous[k]) for k in arm_keys) < 1.0 and arm_error < 25.0
        previous = positions
        if settle_index + 1 >= settle_steps and (arm_error < 8.0 or arm_stopped):
            break

    return True


def _follow_path(
    robot: NexArmSim,
    task: NexArmStackBowlsTask,
    solutions: list[dict[str, float]],
    gripper: float,
    *,
    steps: int,
    settle_steps: int = 2,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None = None,
) -> bool:
    """Sweep through joint waypoints in one minimum-jerk move, without stopping at the intermediate ones.

    Real operators lift a grasped bowl and swing it toward the target in a single motion; stopping at each
    waypoint would look robotic. Progress is spread along the path by joint distance.
    """
    start = robot.backend.joint_positions()
    keys = [key for key in start if key != "gripper.pos"]
    points = [np.array([start[key] for key in keys])]
    for solution in solutions:
        target = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
        target.update({f"{name}.pos": value for name, value in solution.items()})
        points.append(np.array([target[key] for key in keys]))
    lengths = [float(np.abs(b - a).max()) for a, b in zip(points[:-1], points[1:], strict=True)]
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if cumulative[-1] <= 0.0:
        return True
    effective_steps = max(steps, int(np.ceil(cumulative[-1] / 25.0)))
    for alpha in _min_jerk(effective_steps):
        distance = alpha * cumulative[-1]
        segment = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(lengths) - 1)
        local = (distance - cumulative[segment]) / max(lengths[segment], 1e-9)
        joints = points[segment] + np.clip(local, 0.0, 1.0) * (points[segment + 1] - points[segment])
        action = {key: float(value) for key, value in zip(keys, joints, strict=True)}
        action["gripper.pos"] = gripper
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(action)
        if record_frame is not None:
            record_frame(observation, sent)
        if task.observe().terminated:
            return True
    final = {key: float(value) for key, value in zip(keys, points[-1], strict=True)}
    final["gripper.pos"] = gripper
    for _ in range(settle_steps):
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(final)
        if record_frame is not None:
            record_frame(observation, sent)
        if task.observe().terminated:
            break
    return True


def _transit_height(task: NexArmStackBowlsTask, *, held: bool) -> float:
    """Jaw center height (or, when ``held``, carried bowl center height) that clears every other bowl's rim."""
    tops = [task.bowl_position(c)[2] + BOWL_RIM_Z_OFFSET for c in task.current_order if c != task.held_bowl]
    below = BOWL_RIM_Z_OFFSET if held else JAW_TIP_BELOW_CENTER_M
    return max(tops) + TRANSIT_CLEARANCE_M + below


def _approach_direction(backend) -> np.ndarray:
    """Horizontal direction the jaws currently point along, which the IK keeps radial."""
    site_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame")
    approach = -backend.data.site_xmat[site_id].reshape(3, 3)[:, 1]
    return approach[:2] / np.linalg.norm(approach[:2])


def _move_straight(
    robot: NexArmSim,
    task: NexArmStackBowlsTask,
    start: np.ndarray,
    end: np.ndarray,
    u_start: np.ndarray,
    u_end: np.ndarray,
    gripper: float,
    *,
    seed: int,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None = None,
) -> dict[str, float] | None:
    """Move the jaw center, or the carried bowl's center while one is held, along the straight line start->end.

    The line is cut into IK waypoints, each solved from the previous one so the joints stay on one branch, and
    swept in one minimum-jerk motion. The jaw direction turns from ``u_start`` to ``u_end`` along the way.
    Returns the final waypoint's solution, or None if any waypoint is out of reach.
    """
    count = max(1, int(np.ceil(np.linalg.norm(end - start) / CARTESIAN_SPACING_M)))
    qpos = robot.backend.data.qpos.copy()
    solutions = []
    for fraction in np.arange(1, count + 1) / count:
        point = start + fraction * (end - start)
        u = (1 - fraction) * u_start + fraction * u_end
        if task.held_bowl is None:
            sol = solve_ik_jaw(robot.backend, point, u, seed=seed, init_qpos=qpos)
        else:
            sol = solve_ik_held_bowl(robot, task, point, u, seed=seed, init_qpos=qpos)
        if not sol:
            return None
        qpos = _solution_qpos(robot.backend, sol, qpos)
        solutions.append(sol)
    steps = max(12, int(np.ceil(np.linalg.norm(end - start) / CARTESIAN_SPEED_M)))
    _follow_path(robot, task, solutions, gripper, steps=steps, settle_steps=2, record_frame=record_frame)
    return solutions[-1]


def _pick_and_place(
    robot: NexArmSim,
    task: NexArmStackBowlsTask,
    source: str,
    target: str,
    *,
    from_home: bool,
    seed: int,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None = None,
) -> str | None:
    """Stack ``source`` onto ``target`` in straight segments, the way the real demos were teleoperated.

    Open jaws: up to the transit height, across to above the source rim, straight down onto it; grasp. Carried
    bowl: straight up to the transit height, across to above the target, straight down to release height;
    release. Open jaws: straight back up. Returns the failing step's name, or None on success.
    """
    task.target_bowl = source
    rim, u_rim = _grasp_rim(robot, task, source, seed=seed)

    # Open jaws to above the source rim. From home, which is high and clear of the bowls, go there directly.
    transit = _transit_height(task, held=False)
    above_rim = np.array([rim[0], rim[1], transit])
    if from_home:
        sol = solve_ik_jaw(robot.backend, above_rim, u_rim, seed=seed)
        if not sol:
            return "approach"
        _interpolate_stage(
            robot, task, sol, OPEN_GRIPPER, OPEN_GRIPPER, steps=20, settle_steps=2, record_frame=record_frame
        )
    else:
        jaw = _jaw_center(robot.backend)
        u_jaw = _approach_direction(robot.backend)
        up = np.array([jaw[0], jaw[1], max(jaw[2], transit)])
        if up[2] > jaw[2] + 1e-3 and not _move_straight(
            robot, task, jaw, up, u_jaw, u_jaw, OPEN_GRIPPER, seed=seed, record_frame=record_frame
        ):
            return "rise"
        if not _move_straight(
            robot, task, up, above_rim, u_jaw, u_rim, OPEN_GRIPPER, seed=seed, record_frame=record_frame
        ):
            return "approach"

    # Straight down onto the rim, then re-aim for the arm's sag.
    if not _move_straight(
        robot,
        task,
        above_rim,
        rim + [0, 0, RIM_DESCENT_SAG_M],
        u_rim,
        u_rim,
        OPEN_GRIPPER,
        seed=seed,
        record_frame=record_frame,
    ):
        return "descend"
    sol = _descend_to_rim(robot, task, rim, u_rim, seed=seed, record_frame=record_frame)
    if not sol:
        return "pre_grasp"

    # Grasp rim (pinch fingers to straddle rim wall)
    _interpolate_stage(
        robot, task, sol, OPEN_GRIPPER, CLOSED_GRIPPER, steps=10, settle_steps=2, record_frame=record_frame
    )
    if task.held_bowl is None:
        return "grasp_missed"

    # Carried bowl: straight up off its neighbours, across above the target, straight down to release height.
    picked = task.bowl_position(source)
    carry_z = _transit_height(task, held=True)
    lifted = np.array([picked[0], picked[1], carry_z])
    if not _move_straight(
        robot, task, picked, lifted, u_rim, u_rim, CLOSED_GRIPPER, seed=seed, record_frame=record_frame
    ):
        return "lift"
    target_pos = task.bowl_position(target)
    _, u_target = _get_rim_and_unit_vector(target_pos)
    above_target = np.array([target_pos[0], target_pos[1], carry_z])
    if not _move_straight(
        robot,
        task,
        lifted,
        above_target,
        u_rim,
        u_target,
        CLOSED_GRIPPER,
        seed=seed,
        record_frame=record_frame,
    ):
        return "carry"
    # Re-measure the target for a concentric placement.
    release = task.bowl_position(target) + [0, 0, RELEASE_HEIGHT_M]
    sol = _move_straight(
        robot,
        task,
        above_target,
        release,
        u_target,
        u_target,
        CLOSED_GRIPPER,
        seed=seed,
        record_frame=record_frame,
    )
    if not sol:
        return "lower"
    # Motor lag leaves the loaded arm short of the last waypoint; hold it there before letting go.
    _interpolate_stage(
        robot, task, sol, CLOSED_GRIPPER, CLOSED_GRIPPER, steps=5, settle_steps=3, record_frame=record_frame
    )

    # Release
    _interpolate_stage(
        robot, task, sol, CLOSED_GRIPPER, OPEN_GRIPPER, steps=15, settle_steps=3, record_frame=record_frame
    )

    # Open jaws straight back up out of the stack.
    jaw = _jaw_center(robot.backend)
    up = np.array([jaw[0], jaw[1], _transit_height(task, held=False)])
    if not _move_straight(
        robot, task, jaw, up, u_target, u_target, OPEN_GRIPPER, seed=seed, record_frame=record_frame
    ):
        return "retreat"
    return None


def generate_episode(
    robot: NexArmSim,
    task: NexArmStackBowlsTask,
    *,
    seed: int,
    order: tuple[str, str, str] | None = None,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None = None,
) -> tuple[bool, str, str]:
    """Run one scripted 3-bowl stacking attempt and return (success, reason, task_prompt)."""
    task.reset(seed=seed, order=order, settle_steps=25)
    bottom, middle, top = task.current_order
    task_prompt = get_task_instruction(bottom, middle, top)

    # --- Phase 1: Stack Middle onto Bottom ---
    failed = _pick_and_place(
        robot, task, middle, bottom, from_home=True, seed=seed, record_frame=record_frame
    )
    if failed:
        return False, f"ik_p1_{failed}", task_prompt

    # Settle between phases
    current_joints = robot.backend.joint_positions()
    for _ in range(5):
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(current_joints)
        if record_frame is not None:
            record_frame(observation, sent)

    # --- Phase 2: Stack Top onto Middle ---
    failed = _pick_and_place(robot, task, top, middle, from_home=False, seed=seed, record_frame=record_frame)
    if failed:
        return False, f"ik_p2_{failed}", task_prompt

    task.target_bowl = None

    # Stability hold until success or timeout
    hold_action = robot.backend.joint_positions()
    for _ in range(35):
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(hold_action)
        if record_frame is not None:
            record_frame(observation, sent)
        status = task.observe()
        if status.terminated:
            return (
                status.success,
                status.reason or ("success" if status.success else "terminated"),
                task_prompt,
            )

    status = task.status()
    return status.success, status.reason or ("success" if status.success else "task_gate_failed"), task_prompt


def _build_dataset(robot: NexArmSim, args: argparse.Namespace) -> LeRobotDataset:
    features = {
        **hw_to_dataset_features(robot.action_features, ACTION, args.video),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, args.video),
    }
    return utils.open_dataset(
        args,
        features,
        robot.robot_type,
        encoder_queue_maxsize=240,
        encoder_threads=4 if args.video else None,
        image_writer_threads=4 if not args.video else 0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", default="local/nexarm_stack_bowls")
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/nexarm_stack_bowls"))
    parser.add_argument("--episodes", type=int, default=20, help="Number of accepted episodes to write")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--cameras",
        type=str,
        default="front,wrist,top",
        help="Comma-separated list of cameras (e.g. 'front,wrist,top')",
    )
    parser.add_argument("--model", type=Path, default=Path("sim/description/mjcf/bowl_stack_scene.xml"))
    parser.add_argument(
        "--grasp-mode",
        choices=("contact_assisted", "proximity_assisted", "physical"),
        default="physical",
        help="physical holds bowls by jaw friction alone; the assisted modes carry a grasped bowl rigidly.",
    )
    parser.add_argument(
        "--assist-release-pos",
        type=float,
        default=GRIPPER_ASSIST_RELEASE_POS,
        help="Raw gripper command below which the grasp assist drops a held bowl.",
    )
    parser.add_argument(
        "--position-jitter-m",
        type=float,
        default=0.04,
        help="Per-axis bowl position variation around nominal slots, in meters (0 to 0.05). The default "
        "matches NexArmStackBowls-v0, so policies are evaluated on the layouts they were trained on.",
    )
    parser.add_argument(
        "--order",
        choices=[",".join(order) for order in PERMUTATIONS],
        help="Fixed bottom,middle,top order; otherwise cycle all six task instructions.",
    )
    utils.add_generation_args(parser)
    args = parser.parse_args(argv)
    if not np.isfinite(args.position_jitter_m) or not 0 <= args.position_jitter_m <= 0.05:
        parser.error("--position-jitter-m must be between 0 and 0.05")
    if not GRIPPER_OPEN_POS <= args.assist_release_pos < GRIPPER_ASSIST_ENGAGE_POS:
        parser.error(f"--assist-release-pos must be within [{GRIPPER_OPEN_POS}, {GRIPPER_ASSIST_ENGAGE_POS})")
    camera_names = [name.strip() for name in args.cameras.split(",")]
    if any(not name for name in camera_names) or len(set(camera_names)) != len(camera_names):
        parser.error("--cameras must contain distinct nonempty names")
    args.camera_names = tuple(camera_names)
    utils.finalize_generation_args(parser, args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)

    if args.workers > 1:
        return utils.run_sharded(
            Path(__file__).resolve(),
            raw_argv,
            repo_id=args.repo_id,
            root=args.root,
            episodes=args.episodes,
            seed_start=args.seed_start,
            max_attempts=args.max_attempts,
            workers=args.workers,
            gpus=args.gpus,
            split=args.split,
            resume=args.resume,
        )
    if args.gpus:
        return utils.run_on_gpu(Path(__file__).resolve(), raw_argv, args.gpus[0])

    robot = NexArmSim(
        NexArmSimConfig(
            id="synthetic_bowl_generator",
            model_path=args.model,
            fps=args.fps,
            camera_names=args.camera_names,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            settle_steps=0,
            action_delay_steps=args.action_delay_steps,
            enable_domain_randomization=args.domain_randomization,
            calibration_path=args.calibration,
            joint_mapping_path=args.joint_mapping,
        )
    )
    previous_report = utils.read_report(args.root) if args.resume else None
    dataset = _build_dataset(robot, args)
    existing_episodes = dataset.meta.total_episodes
    episodes_path = args.root / utils.EPISODES_NAME
    if not args.resume:
        episodes_path.unlink(missing_ok=True)
    base_delay, delay_range = utils.delay_schedule(args)
    order = tuple(args.order.split(",")) if args.order is not None else None

    print(
        f"[INFO] Generating {args.episodes} episodes of 3-bowl stacking dataset "
        f"(resume={args.resume}, starting_seed={args.seed_start}, total_existing={existing_episodes})..."
    )

    accepted = 0
    attempts = 0
    accepted_seeds: list[int] = []
    rejected_attempts: list[dict[str, int | str]] = []
    try:
        robot.connect()
        task = NexArmStackBowlsTask(
            robot.backend,
            grasp_mode=args.grasp_mode,
            position_jitter_m=args.position_jitter_m,
            assist_release_pos=args.assist_release_pos,
        )
        while accepted < args.episodes and attempts < args.max_attempts:
            seed = args.seed_start + attempts
            delay = utils.episode_delay(seed, base_delay, delay_range)
            robot.backend.action_delay_steps = delay

            bottom, middle, top = order or PERMUTATIONS[seed % len(PERMUTATIONS)]
            current_prompt = get_task_instruction(bottom, middle, top)

            def record_frame(
                observation: dict[str, object],
                action: dict[str, float],
                prompt: str = current_prompt,
            ) -> None:
                observation_frame = build_dataset_frame(dataset.features, observation, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
                dataset.add_frame({**observation_frame, **action_frame, "task": prompt})

            success, reason, _ = generate_episode(
                robot,
                task,
                seed=seed,
                order=order,
                record_frame=record_frame,
            )
            attempts += 1
            utils.append_jsonl(
                episodes_path,
                {
                    "seed": seed,
                    "accepted": success,
                    "reason": reason,
                    "episode_index": existing_episodes + accepted if success else None,
                    "action_delay_steps": delay,
                    "order": task.current_order,
                    "prompt": current_prompt,
                    "grasp_mode": args.grasp_mode,
                    "assist_engage_pos": task.assist_engage_pos,
                    "assist_release_pos": task.assist_release_pos,
                    "dr_params": robot.backend.last_domain_params,
                },
            )
            if success:
                dataset.save_episode()
                accepted += 1
                accepted_seeds.append(seed)
                print(
                    f"Accepted episode {accepted}/{args.episodes} (seed={seed}, prompt='{current_prompt}')",
                    flush=True,
                )
            else:
                dataset.clear_episode_buffer()
                rejected_attempts.append({"seed": seed, "reason": reason})
                print(f"Rejected attempt {attempts} (seed={seed}, reason='{reason}')", flush=True)
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            dataset.finalize()

    total_episodes = existing_episodes + accepted
    report = utils.base_report(
        "examples/nexarm/generate_stack_bowls_dataset.py",
        args,
        model_path=resolve_model_path(args.model),
        attempts=attempts,
        accepted_seeds=accepted_seeds,
        rejected_attempts=rejected_attempts,
    )
    report.update(
        {
            "cameras": list(args.camera_names),
            "grasp_mode": args.grasp_mode,
            "assist_engage_pos": GRIPPER_ASSIST_ENGAGE_POS,
            "assist_release_pos": args.assist_release_pos,
            "position_jitter_m": args.position_jitter_m,
            "order": order,
        }
    )
    if previous_report is not None:
        report = utils.combine_reports([previous_report, report])
    if total_episodes:
        report["validation"] = utils.validate_dataset(args.repo_id, args.root, total_episodes)
    utils.write_report(args.root, report)

    print(
        f"[INFO] Finished: {accepted}/{args.episodes} episodes written to {args.root} "
        f"(total dataset episodes: {total_episodes})",
        flush=True,
    )
    print(f"[INFO] Rejection counts: {dict(Counter(a['reason'] for a in rejected_attempts))}", flush=True)
    return 0 if accepted >= args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
