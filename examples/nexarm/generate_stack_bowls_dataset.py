#!/usr/bin/env python3

"""Generate successful scripted NexArm 3-Bowl Stacking episodes as a LeRobot dataset.

Example:
    uv run python examples/nexarm/generate_stack_bowls_dataset.py \
        --repo-id local/nexarm_stack_bowls \
        --root outputs/datasets/nexarm_stack_bowls \
        --episodes 50
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

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
from lerobot.robots.nexarm_sim.mujoco_backend import HOME_POSITIONS, RAW_RANGES
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

OPEN_GRIPPER = float(RAW_RANGES["gripper"][0])
CLOSED_GRIPPER = float(RAW_RANGES["gripper"][1])


def _interpolate_stage(
    robot: NexArmSim,
    task: NexArmStackBowlsTask,
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
        tolerance_m=0.002,
        restarts=2,
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
    task: NexArmStackBowlsTask,
    *,
    seed: int,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None = None,
) -> tuple[bool, str, str]:
    """Run one scripted 3-bowl stacking attempt and return (success, reason, task_prompt)."""
    status = task.reset(seed=seed, settle_steps=25)
    bottom, middle, top = task.current_order
    task_prompt = get_task_instruction(bottom, middle, top)

    pos_b = task.bowl_position(bottom)
    pos_m = task.bowl_position(middle)
    pos_t = task.bowl_position(top)

    # Offset to grasp bowl rim (rim radius ~ 0.055m, near Y-min edge towards front)
    grasp_offset = np.array([0.0, -0.045, 0.012])
    nest_m_pos = pos_b + np.array([0.0, 0.0, 0.038])
    nest_t_pos = pos_b + np.array([0.0, 0.0, 0.055])

    stages = [
        # --- Phase 1: Stack Middle onto Bottom ---
        (pos_m + grasp_offset + [0.0, 0.0, 0.12], OPEN_GRIPPER, 20),
        (pos_m + grasp_offset, OPEN_GRIPPER, 18),
        (pos_m + grasp_offset, CLOSED_GRIPPER, 50),
        (pos_m + grasp_offset + [0.0, 0.0, 0.12], CLOSED_GRIPPER, 35),
        (nest_m_pos + grasp_offset + [0.0, 0.0, 0.10], CLOSED_GRIPPER, 45),
        (nest_m_pos + grasp_offset, CLOSED_GRIPPER, 25),
        (nest_m_pos + grasp_offset, OPEN_GRIPPER, 30),
        (nest_m_pos + grasp_offset + [0.0, 0.0, 0.12], OPEN_GRIPPER, 20),
        # --- Phase 2: Stack Top onto Middle ---
        (pos_t + grasp_offset + [0.0, 0.0, 0.12], OPEN_GRIPPER, 25),
        (pos_t + grasp_offset, OPEN_GRIPPER, 18),
        (pos_t + grasp_offset, CLOSED_GRIPPER, 50),
        (pos_t + grasp_offset + [0.0, 0.0, 0.12], CLOSED_GRIPPER, 35),
        (nest_t_pos + grasp_offset + [0.0, 0.0, 0.10], CLOSED_GRIPPER, 45),
        (nest_t_pos + grasp_offset, CLOSED_GRIPPER, 25),
        (nest_t_pos + grasp_offset, OPEN_GRIPPER, 30),
        (nest_t_pos + grasp_offset + [0.0, 0.0, 0.14], OPEN_GRIPPER, 20),
    ]

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
            return False, f"ik_failed_stage_{stage_index}", task_prompt

    # Stability hold
    hold_action = robot.backend.joint_positions()
    for _ in range(35):
        observation = robot.get_observation() if record_frame is not None else {}
        sent = robot.send_action(hold_action)
        if record_frame is not None:
            record_frame(observation, sent)
        status = task.observe()
        if status.terminated:
            return status.success, status.reason or "terminated", task_prompt

    status = task.status()
    return status.success, status.reason or "task_gate_failed", task_prompt


def _build_dataset(robot: NexArmSim, args: argparse.Namespace) -> LeRobotDataset:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
    parser.add_argument("--repo-id", default="local/nexarm_stack_bowls")
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/nexarm_stack_bowls"))
    parser.add_argument("--episodes", type=int, default=20, help="Number of accepted episodes to write")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/bowl_stack_scene.xml"))
    parser.add_argument(
        "--domain-randomization",
        "--dr",
        action="store_true",
        help="Enable Visual and Dynamics Domain Randomization per episode",
    )
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.set_defaults(video=True)
    args = parser.parse_args()
    if args.max_attempts is None:
        args.max_attempts = args.episodes * 3
    return args


def main() -> int:
    args = parse_args()
    robot = NexArmSim(
        NexArmSimConfig(
            id="synthetic_bowl_generator",
            model_path=args.model,
            fps=args.fps,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            settle_steps=0,
            enable_domain_randomization=args.domain_randomization,
        )
    )
    dataset = _build_dataset(robot, args)
    robot.connect()
    task = NexArmStackBowlsTask(robot.backend)
    accepted = 0
    attempts = 0

    print(f"[INFO] Generating {args.episodes} episodes of 3-bowl stacking dataset...")

    try:
        while accepted < args.episodes and attempts < args.max_attempts:
            seed = args.seed_start + attempts

            task.reset(seed=seed, settle_steps=25)
            bottom, middle, top = task.current_order
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
                record_frame=record_frame,
            )
            attempts += 1

            if success:
                dataset.save_episode()
                accepted += 1
                print(f"Accepted episode {accepted}/{args.episodes} (seed={seed}, prompt='{current_prompt}')")
            else:
                dataset.clear_episode_buffer()
                print(f"Rejected attempt {attempts} (seed={seed}, reason='{reason}')")
    finally:
        robot.disconnect()

    print(f"[INFO] Finished: {accepted}/{args.episodes} episodes saved to {args.root}")
    return 0 if accepted >= args.episodes else 1


if __name__ == "__main__":
    main()
