#!/usr/bin/env python

# Run inference with a trained policy on NexArm (no leader arm needed).
#
# Usage:
#   python examples/nexarm/rollout.py \
#       --follower-port COM19 \
#       --policy-path outputs/train/nexarm_act/checkpoints/last/pretrained_model
#
# Or use a policy from Hugging Face Hub:
#   python examples/nexarm/rollout.py \
#       --follower-port COM19 \
#       --policy-path <hf_username>/nexarm_act

import argparse

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig
from lerobot.scripts.lerobot_rollout import rollout


def parse_args():
    parser = argparse.ArgumentParser(description="Run NexArm inference")
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--policy-path", required=True, help="Local checkpoint dir or HF Hub repo id")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--front-cam", type=int, default=0)
    parser.add_argument("--wrist-cam", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--repo-id", default=None, help="Optional: save rollout as dataset (e.g. my_user/eval_nexarm)")
    return parser.parse_args()


def main():
    args = parse_args()

    camera_config = {
        "front": OpenCVCameraConfig(index_or_path=args.front_cam, width=640, height=480, fps=args.fps),
        "wrist": OpenCVCameraConfig(index_or_path=args.wrist_cam, width=640, height=480, fps=args.fps),
    }

    follower_config = NexArmFollowerConfig(port=args.follower_port, cameras=camera_config)
    follower = NexArmFollower(follower_config)

    rollout(
        robot=follower,
        policy_path=args.policy_path,
        fps=args.fps,
        num_episodes=args.num_episodes,
        repo_id=args.repo_id,
        display_data=True,
    )


if __name__ == "__main__":
    main()
