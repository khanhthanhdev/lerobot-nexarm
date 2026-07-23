#!/usr/bin/env python
"""Replay canonical 16-D artifacts in Isaac Sim 5.1 by joint name."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.mobile_nexarm_artifact import load_episode_artifact
from lerobot.robots.mobile_bi_nexarm_sim.contract import ARM_FEATURES, load_contract


def resolve_dof_indices(dof_names: list[str], required: list[str]) -> list[int]:
    by_name = {name: index for index, name in enumerate(dof_names)}
    missing = [name for name in required if name not in by_name]
    if missing:
        raise ValueError(f"Isaac articulation is missing DOFs: {missing}")
    return [by_name[name] for name in required]


def replay(
    *,
    robot_usd: Path,
    room_usd: Path,
    artifact: Path,
    headless: bool,
) -> dict[str, Any]:
    try:
        from isaacsim import SimulationApp
    except ImportError as exc:
        raise RuntimeError("run this script with the Isaac Sim 5.1 Python interpreter") from exc

    app = SimulationApp({"headless": headless})
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

        stage_utils.open_stage(str(room_usd))
        prim_path = "/World/mobile_bi_nexarm"
        stage_utils.add_reference_to_stage(str(robot_usd), prim_path)
        world = World(physics_dt=1 / 120, rendering_dt=1 / 30)
        robot = SingleArticulation(prim_path=prim_path, name="mobile_bi_nexarm")
        world.scene.add(robot)
        world.reset()
        robot.initialize()
        metadata, trajectory, _ = load_episode_artifact(artifact, decode_videos=False)
        contract = load_contract()
        required = [
            "base_x",
            "base_y",
            "base_yaw",
            "lift_axis",
            *[f"{side}_{feature}" for side in ("left", "right") for feature in ARM_FEATURES],
        ]
        indices = resolve_dof_indices(list(robot.dof_names), required)
        targets = np.zeros(len(required), dtype=np.float32)
        base_pose = np.zeros(3, dtype=np.float64)
        for action in trajectory["action"]:
            dt = 1 / metadata["fps"]
            base_pose += [action[12] * dt, action[13] * dt, np.deg2rad(action[14]) * dt]
            targets[:3] = base_pose
            targets[3] = action[15] / 1000
            cursor = 4
            for side_offset in (0, 6):
                for feature_offset, feature in enumerate(ARM_FEATURES):
                    targets[cursor] = contract.joint_by_feature[feature].raw_to_sim(
                        float(action[side_offset + feature_offset])
                    )
                    cursor += 1
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=targets, joint_indices=indices)
            )
            for _ in range(4):
                world.step(render=not headless)
        actual = np.asarray(robot.get_joint_positions(indices), dtype=np.float32)
        return {
            "backend": "isaac",
            "frames": int(len(trajectory["action"])),
            "final_joint_positions": actual.tolist(),
            "source_success": bool(metadata["success"]),
        }
    finally:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-usd", type=Path, required=True)
    parser.add_argument("--room-usd", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    result = replay(
        robot_usd=args.robot_usd.resolve(),
        room_usd=args.room_usd.resolve(),
        artifact=args.artifact.resolve(),
        headless=args.headless,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
