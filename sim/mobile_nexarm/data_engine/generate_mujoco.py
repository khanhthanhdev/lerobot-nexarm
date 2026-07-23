#!/usr/bin/env python
"""Generate deterministic MuJoCo pick/place episodes in the neutral format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.mobile_nexarm_artifact import EpisodeArtifactWriter, EpisodeMetadata
from lerobot.robots.mobile_bi_nexarm_sim import MobileBiNexArmSim, MobileBiNexArmSimConfig


def _move(
    robot: MobileBiNexArmSim,
    target_xyz: np.ndarray,
    gripper: float,
    *,
    stage: str,
    writer: EpisodeArtifactWriter,
    timestamps: list[float],
    traces: list[dict],
    steps: int = 12,
) -> bool:
    solution = robot.backend.solve_ik("left", target_xyz)
    if solution is None:
        traces.append({"stage": stage, "ok": False, "reason": "ik_failed", "target": target_xyz.tolist()})
        return False
    start = robot.backend.joint_state()
    target = robot.contract.home_action | {f"left_{name}.pos": value for name, value in solution.items()}
    target["left_gripper.pos"] = gripper
    for interpolation in np.linspace(0, 1, steps):
        action = dict(target)
        for name in solution:
            key = f"left_{name}.pos"
            action[key] = (1 - interpolation) * start[key] + interpolation * target[key]
        state_before = robot.backend.joint_state()
        sent = robot.send_action(action)
        cameras = {name: robot.backend.render(name) for name in robot.config.camera_names}
        writer.add_frame(
            timestamp=timestamps[0],
            state=state_before,
            action=sent,
            cameras=cameras,
        )
        timestamps[0] += 1 / robot.config.fps
        if robot.task_status().terminated:
            break
    status = robot.task_status()
    traces.append(
        {
            "stage": stage,
            "ok": status.success or not status.terminated,
            "target": target_xyz.tolist(),
        }
    )
    return status.success or not status.terminated


def generate_episode(output: Path, seed: int, *, width: int, height: int) -> bool:
    robot = MobileBiNexArmSim(
        MobileBiNexArmSimConfig(
            id=f"generator-{seed}",
            seed=seed,
            camera_width=width,
            camera_height=height,
            settle_steps=25,
            first_task_mode=True,
        )
    )
    robot.connect()
    traces: list[dict] = []
    repairs: list[dict] = []
    metadata = EpisodeMetadata(
        task="pick up the red cube with the left arm and place it in the green target zone",
        seed=seed,
        backend="mujoco",
        domain_parameters={"cube_target_pose_seed": seed},
        stage_traces=traces,
        repair_attempts=repairs,
    )
    writer = EpisodeArtifactWriter(output, metadata)
    timestamp = [0.0]
    try:
        cube = robot.backend.body_position("cube")
        target = robot.backend.body_position("target_zone")
        stages = (
            ("approach", cube + [0, 0, 0.10], 1195.0, 12),
            ("grasp", cube + [0, 0, 0.015], 2833.0, 18),
            ("lift", cube + [0, 0, 0.13], 2833.0, 12),
            ("transfer", target + [0, 0, 0.13], 2833.0, 12),
            ("release", target + [0, 0, 0.045], 1195.0, 18),
            ("retreat", target + [0, 0, 0.20], 1195.0, 12),
        )
        completed = True
        for stage, waypoint, gripper, steps in stages:
            if _move(
                robot,
                waypoint,
                gripper,
                stage=stage,
                writer=writer,
                timestamps=timestamp,
                traces=traces,
                steps=steps,
            ):
                if robot.task_status().success:
                    break
                continue
            completed = False
            if robot.task_status().terminated:
                break
            for attempt in range(1, 3):
                repaired = waypoint + [0, 0, 0.01 * attempt]
                repairs.append({"stage": stage, "attempt": attempt, "z_offset_m": 0.01 * attempt})
                if _move(
                    robot,
                    repaired,
                    gripper,
                    stage=stage,
                    writer=writer,
                    timestamps=timestamp,
                    traces=traces,
                ):
                    completed = True
                    break
            if not completed:
                break
        status = robot.task_status()
        success = completed and status.success
        writer.finalize(
            success=success,
            success_reason=status.reason or ("task_gate_failed" if completed else "repair_budget_exhausted"),
            source_assets=[
                robot.config.spec_path,
                robot.config.model_path,
            ],
        )
        return success
    finally:
        robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--minimum-success-rate", type=float, default=0.70)
    args = parser.parse_args()
    successes = 0
    for offset in range(args.episodes):
        seed = args.seed_start + offset
        successes += generate_episode(
            args.output / f"episode_{seed:06d}",
            seed,
            width=args.width,
            height=args.height,
        )
    rate = successes / args.episodes
    print(f"accepted {successes}/{args.episodes} episodes ({rate:.1%})")
    if rate < args.minimum_success_rate:
        raise SystemExit(f"raw scripted success {rate:.1%} is below {args.minimum_success_rate:.1%}")


if __name__ == "__main__":
    main()
