#!/usr/bin/env python

"""Play the single-arm NexArm pick-and-place task with MuJoCo sliders.

uv run python examples/nexarm/pick_place_sim.py --seed 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer

from lerobot.robots.nexarm_sim import (
    NexArmPickPlaceTask,
    NexArmSim,
    NexArmSimConfig,
)
from lerobot.utils.robot_utils import precise_sleep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use MuJoCo actuator sliders to place the red cube in the green zone."
    )
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/scene.xml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--auto-reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = NexArmSim(
        NexArmSimConfig(
            id="pick_place",
            model_path=args.model,
            fps=args.fps,
            camera_names=(),
            settle_steps=0,
        )
    )
    robot.connect()
    task = NexArmPickPlaceTask(robot.backend, timeout_s=args.timeout)
    task.reset(seed=args.seed, settle_steps=25)
    print("Move the red cube into the green target, open the gripper, and hold it there for 0.5 s.")
    try:
        with mujoco.viewer.launch_passive(robot.backend.model, robot.backend.data) as viewer:
            seed = args.seed
            while viewer.is_running():
                started = time.perf_counter()
                status = task.step()
                viewer.sync()
                if status.terminated:
                    print(f"episode ended: {status.reason}")
                    if not args.auto_reset:
                        break
                    seed += 1
                    task.reset(seed=seed, settle_steps=25)
                precise_sleep(max(0.0, 1 / args.fps - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
