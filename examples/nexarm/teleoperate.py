#!/usr/bin/env python

# Teleoperate NexArm: leader arm controls follower arm in real time.
#
# Usage (leader /dev/ttyUSB0 and follower /dev/ttyUSB1 are the defaults):
#   python examples/nexarm/teleoperate.py
#   python examples/nexarm/teleoperate.py --leader-port COM18 --follower-port COM19
#
# Optional flags:
#   --fps            Control loop rate (default: 30)
#   --front-cam      Front camera index or /dev/v4l/by-id path (default: 0)
#   --wrist-cam      Wrist camera index or /dev/v4l/by-id path (default: 1)
#   --front-fourcc / --wrist-fourcc  Camera pixel format, e.g. MJPG or YUYV (default: auto)
#   --top-cam        RealSense top camera serial (default: auto-detect exactly one)
#   --no-top-cam     Run without the top camera
#   --rerun-save-path outputs/rerun/nexarm_teleop.rrd  Save a replayable session
#   --no-display     Disable Rerun visualization

import argparse
import time

from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig
from lerobot.teleoperators.nexarm_leader import NexArmLeader, NexArmLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun

try:  # imported as a package module (tests) or run as a script
    from examples.nexarm.camera_config import CameraSetupError, add_camera_args, build_camera_configs
except ModuleNotFoundError:
    from camera_config import (  # type: ignore[no-redef]
        CameraSetupError,
        add_camera_args,
        build_camera_configs,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate NexArm")
    parser.add_argument("--leader-port", default="/dev/ttyUSB0", help="Serial port for the leader ESP32")
    parser.add_argument("--follower-port", default="/dev/ttyUSB1", help="Serial port for the follower ESP32")
    parser.add_argument("--fps", type=int, default=30)
    add_camera_args(parser)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--rerun-save-path",
        help="Optional .rrd path to save camera frames, joint observations, and actions",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_display and args.rerun_save_path:
        raise ValueError("--rerun-save-path requires Rerun visualization; omit --no-display.")

    camera_config = build_camera_configs(args, args.fps)
    follower_config = NexArmFollowerConfig(port=args.follower_port, cameras=camera_config)
    leader_config = NexArmLeaderConfig(port=args.leader_port)

    follower = NexArmFollower(follower_config)
    leader = NexArmLeader(leader_config)

    follower.connect()
    leader.connect()

    if not args.no_display:
        init_rerun(session_name="nexarm_teleoperate", save_path=args.rerun_save_path)

    print("Teleoperation started. Press Ctrl+C to stop.")
    try:
        while True:
            start = time.perf_counter()

            leader_pos = leader.get_action()
            follower.send_action(leader_pos)
            obs = follower.get_observation()

            if not args.no_display:
                log_rerun_data(observation=obs, action=leader_pos)

            precise_sleep(1.0 / args.fps - (time.perf_counter() - start))
    except KeyboardInterrupt:
        print("Stopping teleoperation.")
    finally:
        follower.disconnect()
        leader.disconnect()
        if not args.no_display:
            shutdown_rerun()


if __name__ == "__main__":
    try:
        main()
    except CameraSetupError as error:
        raise SystemExit(f"Teleoperation setup failed: {error}") from error
