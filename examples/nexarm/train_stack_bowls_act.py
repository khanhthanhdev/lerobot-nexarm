#!/usr/bin/env python3
"""Prepare and launch ACT training on real NexArm stack-bowl demonstrations."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.motors.nexarm.nexarm import JOINT_NAMES as MOTOR_NAMES
from lerobot.policies.act import ACTConfig
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.feature_utils import dataset_to_policy_features

# Existing 2-camera real dataset; pass --repo-id thanhkt/nexarm_stack_bowls_top for the 3-camera root.
REPO_ID = "thanhkt/nexarm_stack_bowls"
CONFIG_PATH = Path(__file__).with_name("stack_bowls_act.yaml")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOINT_NAMES = [f"{name}.pos" for name in MOTOR_NAMES]
CAMERAS = ("observation.images.front", "observation.images.wrist")
OPTIONAL_CAMERAS = ("observation.images.top",)


def dataset_cameras(meta: LeRobotDatasetMetadata) -> tuple[str, ...]:
    """Required front/wrist cameras plus the top camera when the dataset has it."""
    return CAMERAS + tuple(key for key in OPTIONAL_CAMERAS if key in meta.features)


def camera_shape(meta: LeRobotDatasetMetadata) -> tuple[int, int, int]:
    """The (height, width, 3) every camera was recorded at, read from the dataset metadata."""
    shapes = {key: tuple(meta.features.get(key, {}).get("shape", ())) for key in dataset_cameras(meta)}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"All cameras must share one resolution; dataset metadata has {shapes}.")
    shape = next(iter(shapes.values()))
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"Expected (height, width, 3) RGB camera features; dataset metadata has {shapes}.")
    return shape


def validate_metadata(meta: LeRobotDatasetMetadata) -> None:
    """Fail early if the real dataset's robot or feature contract has changed.

    fps and camera resolution come from the dataset (collection --fps/--width/--height), so any
    consistent recording setup trains; only the robot, joints and camera set are fixed.
    """
    if meta.info.robot_type != "nexarm_follower":
        raise ValueError(f"Expected a nexarm_follower dataset, got robot_type={meta.info.robot_type!r}.")
    if not meta.fps or meta.fps <= 0:
        raise ValueError(f"Dataset metadata has no valid fps ({meta.fps!r}).")
    if meta.total_episodes < 2:
        raise ValueError("Need at least two episodes for training and held-out evaluation.")
    for key in ("observation.state", "action"):
        feature = meta.features.get(key, {})
        if list(feature.get("shape", [])) != [6] or feature.get("names") != JOINT_NAMES:
            raise ValueError(f"{key} must contain the six NexArm joints in order: {JOINT_NAMES}")
    cameras = dataset_cameras(meta)
    for key in cameras:
        if meta.features.get(key, {}).get("dtype") != "video":
            raise ValueError(f"Expected {key} as an RGB video feature.")
    camera_shape(meta)
    for key in ("action", "observation.state", *cameras):
        if key not in meta.stats:
            raise ValueError(f"Dataset normalization statistics missing for {key}.")


def resolve_policy(meta: LeRobotDatasetMetadata, pretrained: str | None) -> ACTConfig:
    """Keep checkpoint architecture intact and require matching robot/camera features."""
    features = dataset_to_policy_features(meta.features)
    inputs = {key: value for key, value in features.items() if value.type != FeatureType.ACTION}
    outputs = {key: value for key, value in features.items() if value.type == FeatureType.ACTION}
    if pretrained is None:
        return ACTConfig(input_features=inputs, output_features=outputs, device="cpu")
    cfg = PreTrainedConfig.from_pretrained(pretrained)
    if not isinstance(cfg, ACTConfig):
        raise ValueError("--pretrained must point to an ACT checkpoint.")
    if cfg.input_features != inputs or cfg.output_features != outputs:
        raise ValueError(
            "Checkpoint features must match front/wrist (and top, if recorded) cameras and six-dimensional state/action. "
            "Use a compatible NexArm ACT checkpoint, or omit --pretrained for initial training."
        )
    return cfg


def check_data(
    meta: LeRobotDatasetMetadata, cfg: ACTConfig, root: Path | None, *, local_files_only: bool = False
) -> None:
    """Decode real frames and verify episode-end action padding without running training."""
    dataset = LeRobotDataset(
        meta.repo_id,
        root=root,
        local_files_only=local_files_only,
        revision=meta.revision,
        episodes=[0],
        video_backend="pyav",
        delta_timestamps={"action": [i / meta.fps for i in cfg.action_delta_indices]},
    )
    height, width, channels = camera_shape(meta)
    for index in (0, len(dataset) - 1):
        sample = dataset[index]
        expected = {"observation.state": (6,), "action": (cfg.chunk_size, 6)}
        expected.update(dict.fromkeys(dataset_cameras(meta), (channels, height, width)))
        for key, shape in expected.items():
            if tuple(sample[key].shape) != shape or not torch.isfinite(sample[key]).all():
                raise ValueError(
                    f"Invalid real sample {index}: {key} expected finite values of shape {shape}"
                )
        mask = sample["action_is_pad"]
        if tuple(mask.shape) != (cfg.chunk_size,):
            raise ValueError("Invalid action padding mask.")
        if index == len(dataset) - 1 and cfg.chunk_size > 1 and not mask[1:].all():
            raise ValueError("Future actions must be padded at the episode boundary.")
    print("Real-data check passed: front/wrist(/top) videos, state, action chunks, and episode-end padding.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo-id", default=REPO_ID, help="NexArm dataset Hub ID")
    parser.add_argument(
        "--pretrained", help="Compatible ACT Hub model ID or local pretrained_model directory"
    )
    parser.add_argument("--dataset-root", type=Path, help="Existing local copy of the real dataset")
    parser.add_argument(
        "--revision", default="main", help="Dataset branch, tag, or commit; pinned before launch"
    )
    parser.add_argument(
        "--check-data", action="store_true", help="Decode episode 0 and exit without training"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check metadata/checkpoint and print the command"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs on this machine")
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="no", help="Accelerate precision"
    )
    parser.add_argument("--output-dir", type=Path)
    # Forward normal LeRobot flags, such as --steps=1000 and --batch_size=1.
    args, extra = parser.parse_known_args()
    if args.num_gpus < 1:
        parser.error("--num-gpus must be at least 1.")
    if args.num_gpus > 1 and args.device != "cuda":
        parser.error("Multiple GPUs require --device cuda.")
    if args.device == "cpu" and args.mixed_precision == "fp16":
        parser.error("FP16 training requires CUDA; use --mixed-precision no on CPU.")
    if args.pretrained and Path(args.pretrained).expanduser().exists():
        args.pretrained = str(Path(args.pretrained).expanduser().resolve())
    if any(not arg.startswith("--") or "=" not in arg for arg in extra):
        parser.error("Trainer overrides must use --key=value syntax.")
    protected = (
        "--dataset.repo_id=",
        "--dataset.root=",
        "--dataset.revision=",
        "--policy.",
        "--config_path=",
        "--resume=",
    )
    if any(arg.startswith(protected) for arg in extra):
        parser.error(
            "Use the launcher's dataset/checkpoint options; policy and resume overrides are unsupported."
        )
    root = args.dataset_root.expanduser().resolve() if args.dataset_root else None
    if root is not None and not (root / "meta" / "info.json").is_file():
        parser.error("--dataset-root must contain an existing LeRobot dataset with meta/info.json.")
    revision = args.revision
    if root is None:
        revision = HfApi().repo_info(args.repo_id, repo_type="dataset", revision=args.revision).sha
        # Avoid silently reading an older recording at HF_LEROBOT_HOME/repo_id.
        root = HF_LEROBOT_HOME / "prepared" / args.repo_id / revision
        snapshot_download(
            args.repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=root,
            allow_patterns="meta/**",
        )
    meta = LeRobotDatasetMetadata(
        args.repo_id, root=root, revision=revision, local_files_only=args.dataset_root is not None
    )
    validate_metadata(meta)
    cfg = resolve_policy(meta, args.pretrained)
    print(
        json.dumps(
            {
                "dataset": args.repo_id,
                "revision": meta.revision,
                "episodes": meta.total_episodes,
                "frames": meta.total_frames,
                "tasks": meta.total_tasks,
                "fps": meta.fps,
                "cameras": dataset_cameras(meta),
                "chunk_size": cfg.chunk_size,
                "pretrained": args.pretrained,
            },
            indent=2,
        )
    )
    if args.check_data:
        check_data(meta, cfg, root, local_files_only=args.dataset_root is not None)
        return
    command = [sys.executable, "-m"]
    if args.num_gpus > 1 or args.mixed_precision != "no":
        command.extend(
            [
                "accelerate.commands.launch",
                f"--num_processes={args.num_gpus}",
                "--num_machines=1",
                f"--mixed_precision={args.mixed_precision}",
                "--dynamo_backend=no",
            ]
        )
        if args.num_gpus > 1:
            command.append("--multi_gpu")
        elif args.device == "cpu":
            command.append("--cpu")
        command.append("--module")
    command.extend(
        [
            "lerobot.scripts.lerobot_train",
            f"--config_path={CONFIG_PATH}",
            f"--dataset.repo_id={args.repo_id}",
            f"--dataset.revision={meta.revision}",
            f"--policy.device={args.device}",
            "--policy.push_to_hub=false",
        ]
    )
    if root:
        command.append(f"--dataset.root={root}")
    if args.pretrained:
        command.append(f"--policy.path={args.pretrained}")
    if args.output_dir:
        command.append(f"--output_dir={args.output_dir.resolve()}")
    # The child runs from PROJECT_ROOT; preserve paths supplied from another directory.
    extra = [
        f"--output_dir={Path(arg.split('=', 1)[1]).expanduser().resolve()}"
        if arg.startswith("--output_dir=")
        else arg
        for arg in extra
    ]
    command.extend(extra)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Select --device cpu or run on a CUDA training machine.")
        if args.device == "cuda" and torch.cuda.device_count() < args.num_gpus:
            parser.error(
                f"Requested {args.num_gpus} GPUs, but only {torch.cuda.device_count()} are visible. "
                "Check CUDA_VISIBLE_DEVICES or run on a machine with enough GPUs."
            )
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
