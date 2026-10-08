#!/usr/bin/env python3
"""Compare TurboVLA predictions with recorded actions around gripper transitions.

Runs offline, without connecting to a robot. Uses checkpoint statistics and the
training collator; excludes episode padding and reports errors in raw joint units.
Group metrics describe deliberately selected frames, not whole-dataset accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def select_groups(actions: np.ndarray, episodes: np.ndarray, horizon: int, limit: int, threshold: float):
    """Select pre-close chunks and open/closed controls without crossing episodes."""
    closed = actions[:, 5] > threshold
    crossings = np.flatnonzero(closed[1:] & ~closed[:-1] & (episodes[1:] == episodes[:-1])) + 1
    before = set()
    for crossing in crossings:
        for offset in (1, max(1, horizon // 2), horizon - 1):
            index = int(crossing - offset)
            if index >= 0 and episodes[index] == episodes[crossing]:
                before.add(index)
    candidates = {
        "before_close": np.array(sorted(before), dtype=int),
        "closed_control": np.flatnonzero(closed),
        "open_control": np.flatnonzero(~closed),
    }
    groups = {}
    for name, indices in candidates.items():
        selection = np.linspace(0, len(indices) - 1, min(limit, len(indices)), dtype=int)
        groups[name] = indices[selection].tolist()
    return groups, len(crossings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument(
        "--turbovla-repo", type=Path, default=Path(__file__).resolve().parents[2] / "TurboVLA"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples-per-group", type=int, default=24)
    parser.add_argument("--gripper-threshold", type=float, default=1800.0)
    parser.add_argument(
        "--cameras", help="Same camera selection/order as training; defaults to auto-detection"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/turbovla.json"))
    args = parser.parse_args()
    if args.samples_per_group < 1 or not np.isfinite(args.gripper_threshold):
        parser.error("Require positive samples-per-group and a finite gripper-threshold")
    if not (args.dataset_root / "meta/info.json").is_file():
        parser.error("dataset-root must contain meta/info.json")

    # Sibling scripts are importable when this file is executed directly.
    from rollout_turbovla import TurboVLAPolicyRunner
    from train_turbovla import TurboVLANexArmCollator, resolve_camera_keys

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    runner = TurboVLAPolicyRunner(
        args.checkpoint, args.turbovla_repo, args.stats_path, device=args.device, binarize_gripper=False
    )
    if not runner.stats:
        raise ValueError("Checkpoint normalization statistics are required for this diagnostic")
    horizon = runner.model.config.action.horizon
    if horizon < 2:
        raise ValueError("Transition diagnosis requires a horizon of at least 2")
    meta = LeRobotDatasetMetadata("local/diagnostic", root=args.dataset_root, local_files_only=True)
    dataset = LeRobotDataset(
        "local/diagnostic",
        root=args.dataset_root,
        delta_timestamps={"action": [i / meta.fps for i in range(horizon)]},
        local_files_only=True,
    )
    cameras = resolve_camera_keys(dataset.features, camera_hints=args.cameras)
    if len(cameras) != runner.num_views:
        raise ValueError(f"Dataset cameras {cameras} do not match checkpoint's {runner.num_views} views")
    # Match training's explicit resize, including checkpoints trained at non-default sizes.
    with (args.checkpoint.parent / "config.json").open() as handle:
        config = json.load(handle)
    image_size = config.get("vision", {}).get("image_size", 224)
    runner.image_processor.size = {"height": image_size, "width": image_size}
    collator = TurboVLANexArmCollator(cameras, runner.stats, runner.image_processor, horizon, dataset.meta)

    # Read numerical columns without decoding videos for the entire dataset.
    table = dataset.select_columns(["action", "episode_index"]).with_format("numpy")[:]
    actions = np.asarray(table["action"], dtype=np.float32)
    episodes = np.asarray(table["episode_index"]).reshape(-1)
    groups, crossings = select_groups(
        actions, episodes, horizon, args.samples_per_group, args.gripper_threshold
    )
    print(f"[DATA] {len(actions)} frames; {crossings} closing transitions; cameras={cameras}")
    if crossings == 0:
        print("[WARN] No closing transitions found. Check dataset actions and gripper threshold.")
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "cameras": cameras,
        "horizon": horizon,
        "closing_transitions": crossings,
        "gripper_threshold": args.gripper_threshold,
        "groups": {},
    }
    for group, indices in groups.items():
        errors, closed_predictions, details = [], [], []
        for index in indices:
            batch = collator([dataset[index]])
            with torch.inference_mode():
                prediction = (
                    runner.model(
                        batch["instructions"],
                        {"dinov3": batch["samples"]["dinov3"].to(runner.device)},
                        batch["states"].to(runner.device),
                    )[0]
                    .float()
                    .cpu()
                    .numpy()
                )
            valid = ~batch["action_is_pad"][0].numpy()
            predicted = runner.unnormalize_action(prediction)[valid]
            target = runner.unnormalize_action(batch["actions"][0].numpy())[valid]
            errors.append(np.abs(predicted - target))
            closing = target[:, 5] > args.gripper_threshold
            closed_predictions.extend((predicted[closing, 5] > args.gripper_threshold).tolist())
            details.append(
                {
                    "index": index,
                    "episode": int(episodes[index]),
                    "task": batch["instructions"][0],
                    "target_gripper": target[:, 5].tolist(),
                    "predicted_gripper": predicted[:, 5].tolist(),
                }
            )
        result = {
            "samples": len(indices),
            "raw_mae_per_joint": np.concatenate(errors).mean(axis=0).tolist() if errors else None,
            "closed_target_steps": len(closed_predictions),
            "closed_recall": float(np.mean(closed_predictions)) if closed_predictions else None,
            "details": details,
        }
        report["groups"][group] = result
        print(f"[{group}] {json.dumps({k: v for k, v in result.items() if k != 'details'})}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
