#!/usr/bin/env python

"""Play with NexArm in MuJoCo, optionally using the physical leader arm.

Viewer sliders:
    uv run python examples/nexarm/simulate.py

Physical leader:
    uv run python examples/nexarm/simulate.py --leader-port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer

from lerobot.robots.nexarm_sim import NexArmSim, NexArmSimConfig
from lerobot.teleoperators.nexarm_leader import NexArmLeader, NexArmLeaderConfig
from lerobot.utils.robot_utils import precise_sleep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control the NexArm MuJoCo simulation")
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/scene.xml"))
    parser.add_argument("--leader-port", help="Optional NexArm leader serial port")
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = NexArmSim(
        NexArmSimConfig(
            id="play",
            model_path=args.model,
            fps=args.fps,
            camera_names=(),
        )
    )
    robot.connect()

    if args.leader_port is None:
        try:
            mujoco.viewer.launch(robot.backend.model, robot.backend.data)
        finally:
            robot.disconnect()
        return

    leader = NexArmLeader(NexArmLeaderConfig(id="sim_leader", port=args.leader_port))
    leader.connect()
    try:
        with mujoco.viewer.launch_passive(robot.backend.model, robot.backend.data) as viewer:
            while viewer.is_running():
                start = time.perf_counter()
                robot.send_action(leader.get_action())
                viewer.sync()
                precise_sleep(1.0 / args.fps - (time.perf_counter() - start))
    except KeyboardInterrupt:
        pass
    finally:
        leader.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
