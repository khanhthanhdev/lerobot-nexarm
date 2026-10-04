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
from lerobot.policies.act import ACTConfig
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.feature_utils import dataset_to_policy_features

REPO_ID = "thanhkt/nexarm_stack_bowls"
CONFIG_PATH = Path(__file__).with_name("stack_bowls_act.yaml")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CAMERAS = ("observation.images.front", "observation.images.wrist")


def validate_metadata(meta: LeRobotDatasetMetadata) -> None:
    """Fail early if the real dataset's robot or feature contract has changed."""
    if meta.info.robot_type != "nexarm_follower" or meta.fps != 30:
        raise ValueError("Expected a nexarm_follower dataset recorded at 30 FPS.")
    if meta.total_episodes < 2:
        raise ValueError("Need at least two episodes for training and held-out evaluation.")
    for key in ("observation.state", "action"):
        feature = meta.features.get(key, {})
        if list(feature.get("shape", [])) != [6] or feature.get("names") != JOINT_NAMES:
            raise ValueError(f"{key} must contain the six NexArm joints in order: {JOINT_NAMES}")
    for key in CAMERAS:
        feature = meta.features.get(key, {})
        if feature.get("dtype") != "video" or list(feature.get("shape", [])) != [480, 640, 3]:
            raise ValueError(f"Expected {key} as 480x640 RGB video.")
    for key in ("action", "observation.state", *CAMERAS):
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
            "Checkpoint features must match both front/wrist cameras and six-dimensional state/action. "
            "Use a compatible NexArm ACT checkpoint, or omit --pretrained for initial training."
        )
    return cfg


def check_data(meta: LeRobotDatasetMetadata, cfg: ACTConfig, root: Path | None) -> None:
    """Decode real frames and verify episode-end action padding without running training."""
    dataset = LeRobotDataset(
        REPO_ID,
        root=root,
        revision=meta.revision,
        episodes=[0],
        video_backend="pyav",
        delta_timestamps={"action": [i / meta.fps for i in cfg.action_delta_indices]},
    )
    for index in (0, len(dataset) - 1):
        sample = dataset[index]
        expected = {"observation.state": (6,), "action": (cfg.chunk_size, 6)}
        expected.update(dict.fromkeys(CAMERAS, (3, 480, 640)))
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
    print("Real-data check passed: front/wrist videos, state, action chunks, and episode-end padding.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output-dir", type=Path)
    # Forward normal LeRobot flags, such as --steps=1000 and --batch_size=1.
    args, extra = parser.parse_known_args()
    if args.pretrained and Path(args.pretrained).exists():
        args.pretrained = str(Path(args.pretrained).resolve())
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
    root = args.dataset_root.resolve() if args.dataset_root else None
    if root is not None and not (root / "meta" / "info.json").is_file():
        parser.error("--dataset-root must contain an existing LeRobot dataset with meta/info.json.")
    revision = args.revision
    if root is None:
        revision = HfApi().repo_info(REPO_ID, repo_type="dataset", revision=args.revision).sha
        # Avoid silently reading an older recording at HF_LEROBOT_HOME/repo_id.
        root = HF_LEROBOT_HOME / "prepared" / REPO_ID / revision
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=root,
            allow_patterns="meta/**",
        )
    meta = LeRobotDatasetMetadata(REPO_ID, root=root, revision=revision)
    validate_metadata(meta)
    cfg = resolve_policy(meta, args.pretrained)
    print(
        json.dumps(
            {
                "dataset": REPO_ID,
                "revision": meta.revision,
                "episodes": meta.total_episodes,
                "frames": meta.total_frames,
                "tasks": meta.total_tasks,
                "fps": meta.fps,
                "cameras": CAMERAS,
                "chunk_size": cfg.chunk_size,
                "pretrained": args.pretrained,
            },
            indent=2,
        )
    )
    if args.check_data:
        check_data(meta, cfg, root)
        return
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--config_path={CONFIG_PATH}",
        f"--dataset.revision={meta.revision}",
        f"--policy.device={args.device}",
    ]
    if root:
        command.append(f"--dataset.root={root}")
    if args.pretrained:
        command.append(f"--policy.path={args.pretrained}")
    if args.output_dir:
        command.append(f"--output_dir={args.output_dir.resolve()}")
    command.extend(extra)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Select --device cpu or run on a CUDA training machine.")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
