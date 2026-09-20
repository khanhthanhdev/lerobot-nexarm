#!/usr/bin/env python3
"""Interactive MuJoCo simulation for the NexArm 3-Bowl Stacking Task.

Visualizes the 3 bowls (Red, Blue, Black) on the table and allows interactive
joint control or passive physics inspection.

Usage:
  uv run python examples/nexarm/stack_bowls_sim.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer

from lerobot.robots.nexarm_sim import NexArmSim, NexArmSimConfig
from lerobot.utils.robot_utils import precise_sleep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NexArm 3-Bowl Stacking MuJoCo Simulation")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("sim/fusion_export/scene_stack_bowls.xml"),
        help="Path to MuJoCo scene XML",
    )
    parser.add_argument("--fps", type=int, default=30, help="Simulation frame rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = NexArmSimConfig(
        id="stack_bowls",
        model_path=args.model,
        fps=args.fps,
        camera_names=("front", "wrist"),
        camera_width=640,
        camera_height=480,
        settle_steps=0,
    )
    robot = NexArmSim(config)
    robot.connect()

    print("==================================================================")
    print("  NexArm 3-Bowl Stacking Task: Red, Blue, Black Bowls")
    print("  - Drag joints using MuJoCo UI sliders on the right")
    print("  - Right-click and drag to move camera; double-click object to track")
    print("==================================================================")

    try:
        with mujoco.viewer.launch_passive(robot.backend.model, robot.backend.data) as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()
                robot.backend.step(robot.backend.joint_positions())
                viewer.sync()
                precise_sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
