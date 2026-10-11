#!/usr/bin/env python

# Record a demonstration dataset with NexArm.
#
# This is a convenience wrapper around the lerobot-record CLI.
# Replace YOUR_HF_USERNAME with your actual Hugging Face username
# (run `huggingface-cli whoami` to confirm).
#
# Usage (leader /dev/ttyUSB0 and follower /dev/ttyUSB1 are the defaults):
#   python examples/nexarm/record.py \
#       --repo-id YOUR_HF_USERNAME/nexarm_pick \
#       --rerun-save-path outputs/rerun/nexarm_pick.rrd \
#       --num-episodes 50 --episode-time 10 --reset-time 10
#
# Cameras: front (--front-cam, default 0), wrist (--wrist-cam, default 1) and the
# RealSense top camera (auto-detected; --top-cam <serial> to choose, --no-top-cam to disable).
#
# Keys during recording:
#   Enter    start / confirm next episode
#   ←        redo last episode
#   ESC      finish early and save
#
# Alternatively, use the CLI directly:
#   lerobot-record \
#       --robot.type=nexarm_follower \
#       --robot.port=/dev/ttyUSB1 \
#       --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"top":{"type":"intelrealsense","serial_number_or_name":"YOUR_SERIAL","width":640,"height":480,"fps":30}}' \
#       --teleop.type=nexarm_leader \
#       --teleop.port=/dev/ttyUSB0 \
#       --dataset.repo_id=YOUR_HF_USERNAME/nexarm_pick \
#       --dataset.single_task="Pick up the object" \
#       --dataset.num_episodes=50 \
#       --dataset.episode_time_s=10 \
#       --dataset.reset_time_s=10

import argparse
import json
import subprocess
import sys

try:  # imported as a package module (tests) or run as a script
    from examples.nexarm.camera_config import CameraSetupError, add_camera_args, build_camera_dicts
except ModuleNotFoundError:
    from camera_config import CameraSetupError, add_camera_args, build_camera_dicts  # type: ignore[no-redef]


def parse_args():
    parser = argparse.ArgumentParser(description="Record NexArm dataset")
    parser.add_argument("--leader-port", default="/dev/ttyUSB0")
    parser.add_argument("--follower-port", default="/dev/ttyUSB1")
    parser.add_argument("--repo-id", required=True, help="e.g. my_hf_user/nexarm_pick")
    parser.add_argument("--task", default="Pick up the object", help="One-sentence task description")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--episode-time", type=int, default=10, help="Seconds per episode")
    parser.add_argument("--reset-time", type=int, default=10, help="Seconds to reset between episodes")
    parser.add_argument("--fps", type=int, default=30)
    add_camera_args(parser)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--rerun-save-path",
        help="Optional .rrd path to save the live recording alongside the LeRobot dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cameras_json = json.dumps(build_camera_dicts(args, args.fps))

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_record",
        "--robot.type=nexarm_follower",
        f"--robot.port={args.follower_port}",
        f"--robot.cameras={cameras_json}",
        "--teleop.type=nexarm_leader",
        f"--teleop.port={args.leader_port}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.fps={args.fps}",
        f"--dataset.episode_time_s={args.episode_time}",
        f"--dataset.reset_time_s={args.reset_time}",
        "--display_data=true",
    ]

    cmd.append(f"--dataset.push_to_hub={'true' if args.push_to_hub else 'false'}")
    if args.rerun_save_path:
        cmd.append(f"--rerun_save_path={args.rerun_save_path}")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    try:
        main()
    except CameraSetupError as error:
        raise SystemExit(f"Recording setup failed: {error}") from error
