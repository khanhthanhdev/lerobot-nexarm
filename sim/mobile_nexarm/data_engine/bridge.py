#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.mobile_nexarm_artifact import write_artifacts_to_lerobot


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert mobile NexArm artifacts to LeRobotDataset v3.")
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--images", action="store_true", help="Store image files instead of videos.")
    args = parser.parse_args()
    dataset = write_artifacts_to_lerobot(
        args.episodes,
        repo_id=args.repo_id,
        root=args.root,
        accepted_only=not args.include_failed,
        use_videos=not args.images,
    )
    print(
        f"Wrote {dataset.meta.total_episodes} episodes and {dataset.meta.total_frames} frames "
        f"to {dataset.root}"
    )


if __name__ == "__main__":
    main()
