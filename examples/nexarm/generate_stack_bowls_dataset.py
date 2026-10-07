#!/usr/bin/env python3

"""Generate successful scripted NexArm 3-Bowl Stacking episodes as a LeRobot dataset.

Example:
    uv run python examples/nexarm/generate_stack_bowls_dataset.py \
        --repo-id local/nexarm_stack_bowls \
        --root outputs/datasets/nexarm_stack_bowls \
        --episodes 50 --position-jitter-m 0.02 --domain-randomization

Grasps default to contact-gated assistance (both jaws must touch the bowl).
This still assists bowl transport and is not a physical grasp benchmark.
Use --grasp-mode proximity_assisted only to reproduce the legacy shortcut.
Generation provenance is stored in meta/generation.jsonl alongside the dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

# Select an offscreen renderer before MuJoCo/GLFW are imported on headless hosts.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim import (
    NexArmSim,
    NexArmSimConfig,
    NexArmStackBowlsTask,
    get_task_instruction,
)
from lerobot.robots.nexarm_sim.mujoco_backend import HOME_POSITIONS, MUJOCO_JOINTS, RAW_RANGES
from lerobot.robots.nexarm_sim.stack_bowls_task import PERMUTATIONS
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

OPEN_GRIPPER = float(RAW_RANGES["gripper"][0])  # 1195.0 (wide open: 67.4mm gap)
CLOSED_GRIPPER = float(RAW_RANGES["gripper"][1])  # 2833.0 (pinched closed: 16.7mm gap)
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
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        **hw_to_dataset_features(robot.action_features, ACTION, args.video),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, args.video),
    }
    if args.resume:
        print(f"[INFO] Resuming existing dataset at {args.root}...")
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=args.root,
            streaming_encoding=args.video,
            encoder_queue_maxsize=240,
            encoder_threads=4 if args.video else None,
            image_writer_threads=4 if not args.video else 0,
        )
        try:
            if dataset.fps != args.fps:
                raise ValueError(f"Existing dataset uses {dataset.fps} FPS; requested {args.fps}")
            for key, feature in features.items():
                actual = dataset.features.get(key, {})
                for field in ("dtype", "shape", "names"):
                    if actual.get(field) != feature.get(field):
                        raise ValueError(f"Existing dataset feature {key} has incompatible {field}")
            recorded_keys = {key for key in dataset.features if key.startswith((f"{OBS_STR}.", f"{ACTION}"))}
            if recorded_keys != set(features):
                raise ValueError("Existing dataset cameras or action/state features do not match")
        except BaseException:
            dataset.finalize()
            raise
        return dataset

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type=robot.name,
        features=features,
        use_videos=args.video,
        streaming_encoding=args.video,
        encoder_queue_maxsize=240,
        encoder_threads=4 if args.video else None,
        image_writer_threads=4 if not args.video else 0,
    )


def _generation_log(root: Path) -> Path:
    return root / "meta" / "generation.jsonl"


def _next_seed(root: Path, fallback: int) -> int:
    """Resume beyond reserved attempt ranges, including rejected or interrupted attempts."""
    path = _generation_log(root)
    if not path.exists():
        return fallback
    with path.open() as stream:
        return max([fallback, *(int(json.loads(line).get("next_seed", 0)) for line in stream)])


def _log_generation(root: Path, record: dict[str, object]) -> None:
    path = _generation_log(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def _merge_generation_logs(datasets: list[LeRobotDataset], root: Path) -> None:
    offset = 0
    for dataset in datasets:
        path = _generation_log(Path(dataset.root))
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get("episode_index") is not None:
                        record["episode_index"] += offset
                    _log_generation(root, record)
        offset += dataset.meta.total_episodes


def _run_multi_gpu(args: argparse.Namespace, gpu_list: list[str]) -> int:
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    num_workers = len(gpu_list)
    episodes_per_worker = [args.episodes // num_workers] * num_workers
    for i in range(args.episodes % num_workers):
        episodes_per_worker[i] += 1
    attempts_per_worker = [args.max_attempts * n // args.episodes for n in episodes_per_worker]
    for i in range(args.max_attempts - sum(attempts_per_worker)):
        attempts_per_worker[i] += 1

    temp_shards: list[tuple[str, Path]] = []
    procs: list[subprocess.Popen] = []

    existing_ds = None
    if args.resume:
        if not args.root.is_dir():
            raise FileNotFoundError(f"Cannot resume missing dataset: {args.root}")
        existing_ds = LeRobotDataset(args.repo_id, root=args.root)
    elif args.root.exists():
        raise FileExistsError(f"Dataset already exists: {args.root}. Use --resume to append.")

    base_seed = args.seed_start
    if base_seed is None:
        base_seed = _next_seed(args.root, existing_ds.meta.total_episodes) if existing_ds is not None else 0
    # Each worker can consume its entire attempt budget without overlapping seeds.
    args.root.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix=f"{args.root.name}_workers_", dir=args.root.parent))

    print(
        f"[INFO] Launching {num_workers} parallel workers on GPUs {gpu_list} to generate "
        f"{args.episodes} episodes total ({episodes_per_worker} per worker)..."
    )

    for worker_idx, (gpu_id, worker_eps) in enumerate(zip(gpu_list, episodes_per_worker, strict=True)):
        if worker_eps <= 0:
            continue
        shard_root = work_root / f"shard_{worker_idx}"
        shard_repo = f"{args.repo_id}_shard_{worker_idx}"
        temp_shards.append((shard_repo, shard_root))

        worker_seed = base_seed + sum(attempts_per_worker[:worker_idx])

        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--repo-id",
            shard_repo,
            "--root",
            str(shard_root),
            "--episodes",
            str(worker_eps),
            "--max-attempts",
            str(attempts_per_worker[worker_idx]),
            "--seed-start",
            str(worker_seed),
            "--fps",
            str(args.fps),
            "--camera-width",
            str(args.camera_width),
            "--camera-height",
            str(args.camera_height),
            "--cameras",
            args.cameras,
            "--model",
            str(args.model),
            "--grasp-mode",
            args.grasp_mode,
            "--position-jitter-m",
            str(args.position_jitter_m),
        ]
        if args.order is not None:
            cmd.extend(["--order", args.order])
        if args.domain_randomization:
            cmd.append("--domain-randomization")
        if not args.video:
            cmd.append("--no-video")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["MUJOCO_GL"] = "egl"
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    failed = False
    try:
        for p in procs:
            if p.wait() != 0:
                failed = True
    except BaseException:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            p.wait()
        raise

    if failed:
        print(f"[ERROR] One or more workers failed. Partial datasets preserved at {work_root}")
        return 1

    print(f"[INFO] All workers completed successfully. Merging {len(temp_shards)} shards into {args.root}...")
    try:
        shard_datasets = [LeRobotDataset(repo_id, root=shard_root) for repo_id, shard_root in temp_shards]
        for shard, count in zip(shard_datasets, (n for n in episodes_per_worker if n > 0), strict=True):
            if shard.meta.total_episodes != count:
                raise RuntimeError(f"Worker wrote {shard.meta.total_episodes} episodes; expected {count}")
        all_to_merge = ([existing_ds] if existing_ds is not None else []) + shard_datasets
        merged_root = work_root / "merged"
        merge_datasets(all_to_merge, output_repo_id=args.repo_id, output_dir=merged_root)
        _merge_generation_logs(all_to_merge, merged_root)
        # Keep the original available until the complete merged dataset is ready.
        backup_root = work_root / "original"
        if existing_ds is not None:
            args.root.rename(backup_root)
        try:
            merged_root.rename(args.root)
        except BaseException:
            if backup_root.exists():
                backup_root.rename(args.root)
            raise
    except BaseException:
        print(f"[ERROR] Merge failed. Worker datasets preserved at {work_root}")
        raise
    shutil.rmtree(work_root)
    print(f"[INFO] Successfully merged into final dataset at {args.root}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/nexarm_stack_bowls")
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/nexarm_stack_bowls"))
    parser.add_argument("--episodes", type=int, default=20, help="Number of accepted episodes to write")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to parallelize across (e.g. '0,7' or '0,1').",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume/append new episodes to an existing dataset without overwriting or deleting it.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Starting random seed. If not specified, starts from 0 (or from existing episode count if --resume).",
    )
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
        "--domain-randomization",
        "--dr",
        action="store_true",
        help="Enable Visual and Dynamics Domain Randomization per episode",
    )
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
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.set_defaults(video=True)
    args = parser.parse_args()
    if args.max_attempts is None:
        args.max_attempts = args.episodes * 3
    for name in ("episodes", "max_attempts", "fps", "camera_width", "camera_height"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.position_jitter_m) or not 0 <= args.position_jitter_m <= 0.05:
        parser.error("--position-jitter-m must be between 0 and 0.05")
    camera_names = [name.strip() for name in args.cameras.split(",")]
    if any(not name for name in camera_names) or len(set(camera_names)) != len(camera_names):
        parser.error("--cameras must contain distinct nonempty names")
    if args.seed_start is not None and args.seed_start < 0:
        parser.error("--seed-start cannot be negative")
    if args.max_attempts < args.episodes:
        parser.error("--max-attempts must be at least --episodes")
    if args.gpus is not None:
        gpu_list = [g.strip() for g in args.gpus.split(",")]
        if any(not g.isdecimal() for g in gpu_list) or len(set(gpu_list)) != len(gpu_list):
            parser.error("--gpus must contain distinct nonnegative GPU IDs")
    return args


def main() -> int:
    args = parse_args()

    if args.gpus is not None:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
        return _run_multi_gpu(args, gpu_list)

    cam_names = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    robot = NexArmSim(
        NexArmSimConfig(
            id="synthetic_bowl_generator",
            model_path=args.model,
            fps=args.fps,
            camera_names=cam_names,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            settle_steps=0,
            enable_domain_randomization=args.domain_randomization,
        )
    )
    dataset = _build_dataset(robot, args)
    accepted = 0
    attempts = 0

    if args.seed_start is None:
        base_seed = _next_seed(args.root, dataset.meta.total_episodes) if args.resume else 0
    else:
        base_seed = args.seed_start

    print(
        f"[INFO] Generating {args.episodes} episodes of 3-bowl stacking dataset "
        f"(resume={args.resume}, starting_seed={base_seed}, total_existing={dataset.meta.total_episodes})..."
    )

    failures: Counter[str] = Counter()
    order = tuple(args.order.split(",")) if args.order is not None else None
    try:
        # Reserve the full seed range before starting, so interrupted runs cannot reuse it.
        _log_generation(
            args.root,
            {
                "type": "run",
                "next_seed": base_seed + args.max_attempts,
                "seed_start": base_seed,
                "grasp_mode": args.grasp_mode,
                "position_jitter_m": args.position_jitter_m,
                "order": order,
                "domain_randomization": args.domain_randomization,
                "model": str(args.model),
                "fps": args.fps,
                "cameras": cam_names,
                "camera_width": args.camera_width,
                "camera_height": args.camera_height,
            },
        )
        robot.connect()
        task = NexArmStackBowlsTask(
            robot.backend,
            grasp_mode=args.grasp_mode,
            position_jitter_m=args.position_jitter_m,
        )
        while accepted < args.episodes and attempts < args.max_attempts:
            seed = base_seed + attempts

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

            episode_index = dataset.meta.total_episodes if success else None
            if success:
                dataset.save_episode()
                accepted += 1
                print(
                    f"Accepted episode {accepted}/{args.episodes} (seed={seed}, prompt='{current_prompt}')",
                    flush=True,
                )
            else:
                failures[reason] += 1
                dataset.clear_episode_buffer()
                print(f"Rejected attempt {attempts} (seed={seed}, reason='{reason}')", flush=True)
            _log_generation(
                args.root,
                {
                    "type": "attempt",
                    "seed": seed,
                    "episode_index": episode_index,
                    "success": success,
                    "reason": reason,
                    "grasp_mode": args.grasp_mode,
                    "order": task.current_order,
                },
            )
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            dataset.finalize()

    print(
        f"[INFO] Finished: {accepted}/{args.episodes} episodes written to {args.root} (total dataset episodes: {dataset.meta.total_episodes})",
        flush=True,
    )
    print(f"[INFO] Rejection counts: {dict(failures)}", flush=True)
    return 0 if accepted >= args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
