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
disables it. Grasps default to contact-gated assistance (both jaws must touch the bowl).
This still assists bowl transport and is not a physical grasp benchmark.
Use --grasp-mode proximity_assisted only to reproduce the legacy shortcut.
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

from lerobot.motors.nexarm.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS, JOINT_NAMES
from lerobot.robots.nexarm_sim import (
    NexArmSim,
    NexArmSimConfig,
    NexArmStackBowlsTask,
    get_task_instruction,
)
from lerobot.robots.nexarm_sim.mujoco_backend import HOME_POSITIONS, MUJOCO_JOINTS, resolve_model_path
from lerobot.robots.nexarm_sim.stack_bowls_task import PERMUTATIONS
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

OPEN_GRIPPER = float(GRIPPER_OPEN_POS)  # wide open: 67.4mm gap
CLOSED_GRIPPER = float(GRIPPER_CLOSED_POS)  # pinched closed: 16.7mm gap
BOWL_RIM_RADIUS = 0.061  # Outer rim radius in meters
BOWL_RIM_Z_OFFSET = 0.0215  # Height of bowl rim above bowl body center
JAW_CENTER_OFFSET = np.array([0.0, 0.033, 0.0])  # Jaw collision pad center in gripper_frame


def _get_rim_and_unit_vector(bowl_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial outward rim point and 2D unit direction from robot base."""
    u = bowl_pos[:2] / np.linalg.norm(bowl_pos[:2])
    rim_point = np.array(
        [
            bowl_pos[0] + BOWL_RIM_RADIUS * u[0],
            bowl_pos[1] + BOWL_RIM_RADIUS * u[1],
            bowl_pos[2] + BOWL_RIM_Z_OFFSET,
        ]
    )
    return rim_point, u


def solve_ik_jaw(
    backend,
    target_pos: np.ndarray,
    u_rad: np.ndarray,
    pitch_deg: float = 75.0,
    roll_val_rad: float = np.deg2rad(80),
    max_iterations: int = 150,
    tolerance_pos: float = 0.002,
    tolerance_dir: float = 0.05,
    restarts: int = 5,
    seed: int = 0,
) -> dict[str, float] | None:
    """Solve IK positioning the jaw collision center with radial rim straddle orientation.

    The gripper pitches downward at `pitch_deg` (with adaptive shallow fallbacks if needed)
    and rotates `wrist_roll` by ~80 degrees so the jaw closing axis is strictly radial
    across the thin rim wall.
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

    candidate_pitches = [pitch_deg, pitch_deg - 5.0, pitch_deg - 10.0, pitch_deg - 15.0]
    for p in candidate_pitches:
        theta = np.deg2rad(p)
        target_dir = np.array([np.cos(theta) * u_rad[0], np.cos(theta) * u_rad[1], -np.sin(theta)])
        rng = np.random.default_rng(seed)

        for restart in range(restarts + 1):
            working = mujoco.MjData(model)
            working.qpos[:] = data.qpos
            working.qpos[roll_qpos] = roll_val_rad
            if restart:
                for k in arm4:
                    low, high = model.jnt_range[joint_ids[k]]
                    working.qpos[model.jnt_qposadr[joint_ids[k]]] = rng.uniform(low * 0.8, high * 0.8)

            for _ in range(max_iterations):
                mujoco.mj_forward(model, working)
                site_pos = working.site_xpos[site_id]
                site_mat = working.site_xmat[site_id].reshape(3, 3)
                current_jaw = site_pos + site_mat @ JAW_CENTER_OFFSET
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
    pitch_deg: float = 75.0,
    seed: int = 0,
) -> dict[str, float] | None:
    """Iteratively solve IK to position the held bowl center at target_bowl_xyz."""
    if task.held_rel_pos is None:
        return None

    site_id = mujoco.mj_name2id(robot.backend.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame")
    target_jaw = target_bowl_xyz.copy()

    for _ in range(6):
        sol = solve_ik_jaw(robot.backend, target_jaw, u_rad, pitch_deg=pitch_deg, seed=seed)
        if sol is None:
            return None
        working = mujoco.MjData(robot.backend.model)
        working.qpos[:] = robot.backend.data.qpos
        for k, v in sol.items():
            working.qpos[robot.backend.model.jnt_qposadr[robot.backend._joint_ids[k]]] = (
                robot.backend.raw_to_control(k, v)
            )
        mujoco.mj_forward(robot.backend.model, working)
        site_pos = working.site_xpos[site_id].copy()
        site_mat = working.site_xmat[site_id].reshape(3, 3).copy()
        pred_bowl_pos = site_pos + site_mat @ task.held_rel_pos
        err = target_bowl_xyz - pred_bowl_pos
        if np.linalg.norm(err) < 0.002:
            return sol
        target_jaw += err

    return None


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
    """Linearly interpolate joints to target solution and hold for settle_steps."""
    start = robot.backend.joint_positions()
    target = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
    target.update({f"{name}.pos": value for name, value in solution.items()})
    target["gripper.pos"] = target_gripper

    # Scale steps if large joint travel is required (prevent PD motor lag on long transits)
    max_delta = max(abs(start[k] - target[k]) for k in target if k != "gripper.pos")
    min_steps = int(np.ceil(max_delta / 25.0))
    effective_steps = max(steps, min_steps)

    for alpha in np.linspace(0.0, 1.0, effective_steps, endpoint=True):
        action = {key: float((1 - alpha) * start[key] + alpha * target[key]) for key in target}
        action["gripper.pos"] = float((1 - alpha) * start_gripper + alpha * target_gripper)
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(action)
        if record_frame is not None:
            record_frame(observation, sent)
        if task.observe().terminated:
            break

    # Motor lag can leave a carried bowl centimeters from its IK target. Keep
    # commanding the target before releasing or starting the next motion.
    for settle_index in range(max(settle_steps, robot.config.fps)):
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(target)
        if record_frame is not None:
            record_frame(observation, sent)
        if task.observe().terminated:
            break
        positions = robot.backend.joint_positions()
        if settle_index + 1 >= settle_steps and max(abs(positions[k] - target[k]) for k in target) < 8.0:
            break

    return True


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

    site_id = mujoco.mj_name2id(robot.backend.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame")

    # --- Phase 1: Stack Middle onto Bottom ---
    pos_m = task.bowl_position(middle)
    pos_b = task.bowl_position(bottom)
    rim_m, u_m = _get_rim_and_unit_vector(pos_m)
    _, u_b = _get_rim_and_unit_vector(pos_b)
    task.target_bowl = middle

    # Approach from +0.08m above rim
    sol = solve_ik_jaw(robot.backend, rim_m + [0, 0, 0.08], u_m, seed=seed)
    if not sol:
        return False, "ik_p1_approach", task_prompt
    _interpolate_stage(
        robot, task, sol, OPEN_GRIPPER, OPEN_GRIPPER, steps=20, settle_steps=2, record_frame=record_frame
    )

    # Descend to rim
    sol = solve_ik_jaw(robot.backend, rim_m, u_m, seed=seed)
    if not sol:
        return False, "ik_p1_pre_grasp", task_prompt
    _interpolate_stage(
        robot, task, sol, OPEN_GRIPPER, OPEN_GRIPPER, steps=15, settle_steps=2, record_frame=record_frame
    )

    # Grasp rim (pinch fingers to straddle rim wall)
    _interpolate_stage(
        robot, task, sol, OPEN_GRIPPER, CLOSED_GRIPPER, steps=15, settle_steps=3, record_frame=record_frame
    )
    if task.held_bowl is None:
        return False, "ik_p1_grasp_missed", task_prompt

    # Lift bowl vertically
    sol = solve_ik_jaw(robot.backend, rim_m + [0, 0, 0.12], u_m, seed=seed)
    if not sol:
        return False, "ik_p1_lift", task_prompt
    _interpolate_stage(
        robot, task, sol, CLOSED_GRIPPER, CLOSED_GRIPPER, steps=20, settle_steps=2, record_frame=record_frame
    )

    # Carry above bottom bowl
    sol = solve_ik_held_bowl(robot, task, pos_b + [0, 0, 0.12], u_b, seed=seed)
    if not sol:
        return False, "ik_p1_carry", task_prompt
    _interpolate_stage(
        robot, task, sol, CLOSED_GRIPPER, CLOSED_GRIPPER, steps=25, settle_steps=2, record_frame=record_frame
    )

    # Lower into bottom bowl
    sol = solve_ik_held_bowl(robot, task, pos_b + [0, 0, 0.035], u_b, seed=seed)
    if not sol:
        return False, "ik_p1_lower", task_prompt
    _interpolate_stage(
        robot, task, sol, CLOSED_GRIPPER, CLOSED_GRIPPER, steps=15, settle_steps=3, record_frame=record_frame
    )

    # Release
    _interpolate_stage(
        robot, task, sol, CLOSED_GRIPPER, OPEN_GRIPPER, steps=15, settle_steps=3, record_frame=record_frame
    )

    # Retreat upward along reverse approach angle
    site_pos = robot.backend.data.site_xpos[site_id].copy()
    site_mat = robot.backend.data.site_xmat[site_id].reshape(3, 3).copy()
    jaw_curr = site_pos + site_mat @ JAW_CENTER_OFFSET
    sol = solve_ik_jaw(robot.backend, jaw_curr + [0, 0, 0.08], u_b, seed=seed)
    if not sol:
        return False, "ik_p1_retreat", task_prompt
    _interpolate_stage(
        robot, task, sol, OPEN_GRIPPER, OPEN_GRIPPER, steps=15, settle_steps=2, record_frame=record_frame
    )

    # Settle between phases
    current_joints = robot.backend.joint_positions()
    for _ in range(5):
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(current_joints)
        if record_frame is not None:
            record_frame(observation, sent)

    # --- Phase 2: Stack Top onto Middle ---
    pos_t = task.bowl_position(top)
    pos_m_now = task.bowl_position(middle)
    rim_t, u_t = _get_rim_and_unit_vector(pos_t)
    _, u_m_now = _get_rim_and_unit_vector(pos_m_now)
    task.target_bowl = top

    # Approach top bowl (transit across table)
    sol = solve_ik_jaw(robot.backend, rim_t + [0, 0, 0.08], u_t, seed=seed)
    if not sol:
        return False, "ik_p2_approach", task_prompt
    _interpolate_stage(
        robot, task, sol, OPEN_GRIPPER, OPEN_GRIPPER, steps=25, settle_steps=2, record_frame=record_frame
    )

    # Descend to rim
    sol = solve_ik_jaw(robot.backend, rim_t, u_t, seed=seed)
    if not sol:
        return False, "ik_p2_pre_grasp", task_prompt
    _interpolate_stage(
        robot, task, sol, OPEN_GRIPPER, OPEN_GRIPPER, steps=15, settle_steps=2, record_frame=record_frame
    )

    # Grasp rim
    _interpolate_stage(
        robot, task, sol, OPEN_GRIPPER, CLOSED_GRIPPER, steps=15, settle_steps=3, record_frame=record_frame
    )
    if task.held_bowl is None:
        return False, "ik_p2_grasp_missed", task_prompt

    # Lift top bowl
    sol = solve_ik_jaw(robot.backend, rim_t + [0, 0, 0.12], u_t, seed=seed)
    if not sol:
        return False, "ik_p2_lift", task_prompt
    _interpolate_stage(
        robot, task, sol, CLOSED_GRIPPER, CLOSED_GRIPPER, steps=20, settle_steps=2, record_frame=record_frame
    )

    # Carry above middle bowl
    sol = solve_ik_held_bowl(robot, task, pos_m_now + [0, 0, 0.12], u_m_now, seed=seed)
    if not sol:
        return False, "ik_p2_carry", task_prompt
    _interpolate_stage(
        robot, task, sol, CLOSED_GRIPPER, CLOSED_GRIPPER, steps=25, settle_steps=2, record_frame=record_frame
    )

    # Re-measure middle bowl position for sub-millimeter concentric placement
    pos_m_now = task.bowl_position(middle)
    sol = solve_ik_held_bowl(robot, task, pos_m_now + [0, 0, 0.035], u_m_now, seed=seed)
    if not sol:
        return False, "ik_p2_lower", task_prompt
    _interpolate_stage(
        robot, task, sol, CLOSED_GRIPPER, CLOSED_GRIPPER, steps=15, settle_steps=3, record_frame=record_frame
    )

    # Release
    _interpolate_stage(
        robot, task, sol, CLOSED_GRIPPER, OPEN_GRIPPER, steps=15, settle_steps=3, record_frame=record_frame
    )

    # Retreat upward along reverse approach angle
    site_pos = robot.backend.data.site_xpos[site_id].copy()
    site_mat = robot.backend.data.site_xmat[site_id].reshape(3, 3).copy()
    jaw_curr = site_pos + site_mat @ JAW_CENTER_OFFSET
    sol = solve_ik_jaw(robot.backend, jaw_curr + [0, 0, 0.08], u_m_now, seed=seed)
    if not sol:
        return False, "ik_p2_retreat", task_prompt
    _interpolate_stage(
        robot, task, sol, OPEN_GRIPPER, OPEN_GRIPPER, steps=15, settle_steps=2, record_frame=record_frame
    )

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
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/bowl_stack_scene.xml"))
    parser.add_argument(
        "--grasp-mode",
        choices=("contact_assisted", "proximity_assisted"),
        default="contact_assisted",
        help="Both modes assist transport; contact_assisted requires both jaws touching the bowl.",
    )
    parser.add_argument(
        "--position-jitter-m",
        type=float,
        default=0.005,
        help="Per-axis bowl position variation around nominal slots, in meters (0 to 0.05).",
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
