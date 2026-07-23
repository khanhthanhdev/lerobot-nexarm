#!/usr/bin/env python

"""Generate successful scripted NexArm MuJoCo episodes as a LeRobot dataset.

Example:
    MUJOCO_GL=egl uv run python examples/nexarm/generate_sim_dataset.py \
        --repo-id local/nexarm_sim_pick_place \
        --root outputs/datasets/nexarm_sim_pick_place \
        --episodes 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim import NexArmPickPlaceTask, NexArmSim, NexArmSimConfig
from lerobot.robots.nexarm_sim.mujoco_backend import HOME_POSITIONS, RAW_RANGES, resolve_model_path
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

DEFAULT_TASK = "Pick up the red cube, place it in the green target zone, and release it."
OPEN_GRIPPER = float(RAW_RANGES["gripper"][0])
CLOSED_GRIPPER = float(RAW_RANGES["gripper"][1])


def _interpolate_stage(
    robot: NexArmSim,
    task: NexArmPickPlaceTask,
    target_xyz: np.ndarray,
    gripper: float,
    *,
    seed: int,
    steps: int,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None,
    settle_steps: int = 10,
) -> bool:
    solution = robot.backend.solve_ik(
        target_xyz,
        seed=seed,
        tolerance_m=0.001,
        restarts=0,
    )
    if solution is None:
        return False

    start = robot.backend.joint_positions()
    target = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
    target.update({f"{name}.pos": value for name, value in solution.items()})
    target["gripper.pos"] = gripper

    for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
        action = {key: float((1 - alpha) * start[key] + alpha * target[key]) for key in target}
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(action)
        if record_frame is not None:
            record_frame(observation, sent)
        if task.observe().terminated:
            break
    for _ in range(settle_steps):
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(target)
        if record_frame is not None:
            record_frame(observation, sent)
        if task.observe().terminated:
            break
    return True


def generate_episode(
    robot: NexArmSim,
    task: NexArmPickPlaceTask,
    *,
    seed: int,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None = None,
    trace: bool = False,
) -> tuple[bool, str]:
    """Run one deterministic pick/place attempt and return its accepted status."""

    task.reset(seed=seed, settle_steps=25)
    cube = robot.backend.body_position("cube")
    target = robot.backend.body_position("target_zone")

    # Keep the jaw collision boxes off the floor while retaining vertical
    # overlap with the 20 mm cube.
    cube_grasp = cube + np.array([0.0, 0.0, 0.015])
    target_release = target + np.array([0.0, 0.0, 0.012])
    stages = (
        (cube_grasp + [0.0, 0.0, 0.12], OPEN_GRIPPER, 16),
        (cube_grasp, OPEN_GRIPPER, 18),
        (cube_grasp, CLOSED_GRIPPER, 80),
        (cube_grasp + [0.0, 0.0, 0.055], CLOSED_GRIPPER, 30),
        (cube_grasp + [0.0, 0.0, 0.09], CLOSED_GRIPPER, 50),
        (cube_grasp + [0.0, 0.0, 0.13], CLOSED_GRIPPER, 50),
        (target_release + [0.0, 0.0, 0.12], CLOSED_GRIPPER, 60),
        (target_release, CLOSED_GRIPPER, 18),
        (target_release, OPEN_GRIPPER, 24),
        (target_release + [0.0, 0.0, 0.14], OPEN_GRIPPER, 18),
    )

    for stage_index, (waypoint, gripper, steps) in enumerate(stages):
        if not _interpolate_stage(
            robot,
            task,
            np.asarray(waypoint),
            gripper,
            seed=seed * 100 + stage_index,
            steps=steps,
            record_frame=record_frame,
        ):
            return False, f"ik_failed_stage_{stage_index}"
        status = task.status()
        if trace:
            jaws = np.stack(
                [
                    robot.backend.geom_position("link_6_left_jaw_collision_0"),
                    robot.backend.geom_position("link_6_right_jaw_collision_0"),
                ]
            )
            print(
                f"seed={seed} stage={stage_index} site={robot.backend.site_position('gripper_frame')} "
                f"jaws={jaws} cube={robot.backend.body_position('cube')} grasped={status.is_grasped}"
            )
        if status.terminated:
            return status.success, status.reason or "terminated"

    # Hold after release so the task's stability gate can accept the placement.
    hold_action = robot.backend.joint_positions()
    for _ in range(20):
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(hold_action)
        if record_frame is not None:
            record_frame(observation, sent)
        status = task.observe()
        if status.terminated:
            return status.success, status.reason or "terminated"

    status = task.status()
    return status.success, status.reason or "task_gate_failed"


def _build_dataset(robot: NexArmSim, args: argparse.Namespace) -> LeRobotDataset:
    features = {
        **hw_to_dataset_features(robot.action_features, ACTION, args.video),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, args.video),
    }
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type=robot.name,
        features=features,
        use_videos=args.video,
        streaming_encoding=args.video,
        encoder_queue_maxsize=120,
        encoder_threads=2 if args.video else None,
        image_writer_threads=4 if not args.video else 0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/nexarm_sim_pick_place")
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/nexarm_sim_pick_place"))
    parser.add_argument("--episodes", type=int, default=20, help="Number of accepted episodes to write")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/scene.xml"))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.set_defaults(video=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.fps <= 0:
        parser.error("--episodes and --fps must be positive")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("camera dimensions must be positive")
    if args.max_attempts is None:
        args.max_attempts = args.episodes * 3
    if args.max_attempts < args.episodes:
        parser.error("--max-attempts cannot be smaller than --episodes")
    return args


def main() -> int:
    args = parse_args()
    robot = NexArmSim(
        NexArmSimConfig(
            id="synthetic_generator",
            model_path=args.model,
            fps=args.fps,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            settle_steps=0,
        )
    )
    dataset = _build_dataset(robot, args)
    robot.connect()
    task = NexArmPickPlaceTask(robot.backend)
    accepted = 0
    attempts = 0
    accepted_seeds: list[int] = []
    rejected_attempts: list[dict[str, int | str]] = []

    try:
        while accepted < args.episodes and attempts < args.max_attempts:
            seed = args.seed_start + attempts

            def record_frame(observation: dict[str, object], action: dict[str, float]) -> None:
                observation_frame = build_dataset_frame(dataset.features, observation, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
                dataset.add_frame({**observation_frame, **action_frame, "task": args.task})

            success, reason = generate_episode(
                robot,
                task,
                seed=seed,
                record_frame=record_frame,
                trace=args.trace,
            )
            attempts += 1
            if success:
                dataset.save_episode(parallel_encoding=False)
                accepted += 1
                accepted_seeds.append(seed)
                print(f"accepted seed={seed} ({accepted}/{args.episodes})")
            else:
                dataset.clear_episode_buffer()
                rejected_attempts.append({"seed": seed, "reason": reason})
                print(f"rejected seed={seed}: {reason}")
    finally:
        robot.disconnect()
        dataset.finalize()

    model_path = resolve_model_path(args.model)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report = {
        "generator": "examples/nexarm/generate_sim_dataset.py",
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "repo_id": args.repo_id,
        "fps": args.fps,
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "video": args.video,
        "requested_episodes": args.episodes,
        "attempts": attempts,
        "accepted_seeds": accepted_seeds,
        "rejected_attempts": rejected_attempts,
    }
    report_path = args.root / "generation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {accepted} episode(s) from {attempts} attempt(s) to {args.root}")
    return 0 if accepted == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
