#!/usr/bin/env python

# Record a demonstration dataset with NexArm.
#
# Usage:
#   python examples/nexarm/record.py \
#       --follower-port COM19 --leader-port COM18 \
#       --repo-id <hf_username>/nexarm_pick \
#       --num-episodes 50 --episode-time 10 --reset-time 10
#
# Keys during recording:
#   Enter    start / confirm next episode
#   ←        redo last episode
#   ESC      finish early and save

import argparse

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig
from lerobot.scripts.lerobot_record import record
from lerobot.teleoperators.nexarm_leader import NexArmLeader, NexArmLeaderConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Record NexArm dataset")
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--repo-id", required=True, help="e.g. my_hf_user/nexarm_pick")
    parser.add_argument("--task", default="Pick up the object", help="One-sentence task description")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--episode-time", type=int, default=10, help="Seconds per episode")
    parser.add_argument("--reset-time", type=int, default=10, help="Seconds to reset between episodes")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--front-cam", type=int, default=0)
    parser.add_argument("--wrist-cam", type=int, default=1)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    camera_config = {
        "front": OpenCVCameraConfig(index_or_path=args.front_cam, width=640, height=480, fps=args.fps),
        "wrist": OpenCVCameraConfig(index_or_path=args.wrist_cam, width=640, height=480, fps=args.fps),
    }

    follower_config = NexArmFollowerConfig(port=args.follower_port, cameras=camera_config)
    leader_config = NexArmLeaderConfig(port=args.leader_port)

    follower = NexArmFollower(follower_config)
    leader = NexArmLeader(leader_config)

    record(
        robot=follower,
        teleop=leader,
        repo_id=args.repo_id,
        single_task=args.task,
        num_episodes=args.num_episodes,
        episode_time_s=args.episode_time,
        reset_time_s=args.reset_time,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
        display_data=True,
    )


if __name__ == "__main__":
    main()
