#!/usr/bin/env python

# Run inference with a trained policy on NexArm (no leader arm needed).
#
# Replace YOUR_HF_USERNAME with your actual Hugging Face username.
# The opened cameras must match the policy's image inputs: front, wrist and the RealSense top
# camera by default; pass --no-top-cam for a policy trained on front/wrist only.
#
# Usage — local checkpoint (follower /dev/ttyUSB1 is the default):
#   python examples/nexarm/rollout.py \
#       --policy-path outputs/train/nexarm_act/checkpoints/last/pretrained_model \
#       --rerun-save-path outputs/rerun/nexarm_rollout.rrd
#
# Usage — policy from Hugging Face Hub, front/wrist cameras only:
#   python examples/nexarm/rollout.py \
#       --follower-port COM19 --no-top-cam \
#       --policy-path YOUR_HF_USERNAME/nexarm_act
#
# Alternatively, use the CLI directly:
#   lerobot-rollout \
#       --strategy.type=base \
#       --policy.path=outputs/train/nexarm_act/checkpoints/last/pretrained_model \
#       --robot.type=nexarm_follower \
#       --robot.port=/dev/ttyUSB1 \
#       --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"top":{"type":"intelrealsense","serial_number_or_name":"YOUR_SERIAL","width":640,"height":480,"fps":30}}' \
#       --fps=30 \
#       --display_data=true

import argparse
import json
import subprocess
import sys

try:  # imported as a package module (tests) or run as a script
    from examples.nexarm.camera_config import (
        CameraSetupError,
        add_camera_args,
        build_camera_dicts,
        camera_names,
        check_policy_cameras,
    )
except ModuleNotFoundError:
    from camera_config import (  # type: ignore[no-redef]
        CameraSetupError,
        add_camera_args,
        build_camera_dicts,
        camera_names,
        check_policy_cameras,
    )

IMAGE_PREFIX = "observation.images."


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run NexArm inference")
    parser.add_argument("--follower-port", default="/dev/ttyUSB1")
    parser.add_argument("--policy-path", required=True, help="Local checkpoint dir or HF Hub repo id")
    parser.add_argument("--fps", type=int, default=30, help="Control loop and camera frame rate")
    add_camera_args(parser)
    parser.add_argument(
        "--strategy",
        default="base",
        choices=["base", "sentry", "highlight", "dagger"],
        help="Rollout strategy (default: base)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Dataset destination repository id (required for sentry/highlight/dagger, e.g. YOUR_HF_USERNAME/eval_nexarm)",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Optional language task instruction for language-conditioned policies (e.g. SmolVLA)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Rollout duration in seconds (0 = run indefinitely until stopped)",
    )
    parser.add_argument(
        "--inference-type",
        default="sync",
        choices=["sync", "rtc"],
        help="Inference backend (default: sync)",
    )
    parser.add_argument(
        "--rerun-save-path",
        help="Optional .rrd path to save camera frames, observations, and policy actions",
    )
    args = parser.parse_args(argv)
    if args.strategy in ("sentry", "highlight", "dagger") and not args.repo_id:
        parser.error(f"--strategy {args.strategy} requires --repo-id to specify the dataset destination")
    return args


def policy_camera_names(policy_path: str) -> list[str]:
    """Camera names (``front``, ``wrist``, ``top``) the policy expects as image inputs."""
    import lerobot.policies  # noqa: F401  # registers the policy config types (act, smolvla, ...)
    from lerobot.configs import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(policy_path)
    return [key.removeprefix(IMAGE_PREFIX) for key in config.image_features]


def main():
    args = parse_args()
    check_policy_cameras(camera_names(args), policy_camera_names(args.policy_path), args.policy_path)
    cameras_json = json.dumps(build_camera_dicts(args, args.fps))

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_rollout",
        f"--strategy.type={args.strategy}",
        f"--policy.path={args.policy_path}",
        "--robot.type=nexarm_follower",
        f"--robot.port={args.follower_port}",
        f"--robot.cameras={cameras_json}",
        f"--fps={args.fps}",
        f"--inference.type={args.inference_type}",
        "--display_data=true",
    ]

    if args.task:
        cmd.append(f"--task={args.task}")
    if args.duration > 0:
        cmd.append(f"--duration={args.duration}")
    if args.repo_id:
        cmd.append(f"--dataset.repo_id={args.repo_id}")
    if args.rerun_save_path:
        cmd.append(f"--rerun_save_path={args.rerun_save_path}")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    try:
        main()
    except CameraSetupError as error:
        raise SystemExit(f"Rollout setup failed: {error}") from error
