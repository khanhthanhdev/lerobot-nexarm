#!/usr/bin/env python3
"""Merge already uploaded LeRobot datasets and upload the combined dataset.

Example:
    uv run python examples/nexarm/merge_hub_datasets.py \\
        --source-repo-ids my-user/nexarm_pick_20261001_120000 my-user/nexarm_pick_20261002_150000 \\
        --output-repo-id my-user/nexarm_pick
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import uuid
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.datasets import LeRobotDataset, merge_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo-ids",
        nargs="+",
        required=True,
        help="Hub dataset repo IDs to combine, such as user/dataset_20261001 user/dataset_20261002.",
    )
    parser.add_argument(
        "--output-repo-id",
        required=True,
        help="New Hub dataset repo ID for the combined dataset, such as user/nexarm_pick.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory for the merged dataset. Defaults to the standard LeRobot cache location.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the output Hub dataset as private.",
    )
    parser.add_argument(
        "--keep-sources",
        action="store_true",
        help="Keep the source Hub repos after the merged dataset has uploaded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_repo_id in args.source_repo_ids:
        raise SystemExit("Output repo ID must be different from every source repo ID.")
    if len(set(args.source_repo_ids)) != len(args.source_repo_ids):
        raise SystemExit("Source repo IDs must be unique.")

    print(f"Loading {len(args.source_repo_ids)} source datasets from the Hub...")
    datasets = [LeRobotDataset(repo_id) for repo_id in args.source_repo_ids]
    for dataset in datasets:
        print(f"  {dataset.repo_id}: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")

    output_dir = args.output_dir
    if output_dir is None:
        # The standard cache path may already contain a downloaded dataset with
        # the output repo ID. Build in a fresh temporary directory instead.
        output_dir = Path(tempfile.gettempdir()) / f"lerobot-merge-{uuid.uuid4().hex}"

    print(f"Merging into {args.output_repo_id} at {output_dir}...")
    merged = merge_datasets(
        datasets,
        output_repo_id=args.output_repo_id,
        output_dir=output_dir,
    )
    print(f"Uploading {merged.meta.total_episodes} episodes to the Hub...")
    merged.push_to_hub(
        private=True if args.private else None,
        upload_large_folder=False,
        tags=["nexarm", "merged"],
    )
    print(f"Done: https://huggingface.co/datasets/{args.output_repo_id}")

    if args.output_dir is None:
        shutil.rmtree(output_dir)

    if not args.keep_sources:
        api = HfApi()
        for repo_id in args.source_repo_ids:
            api.delete_repo(repo_id=repo_id, repo_type="dataset")
            print(f"Deleted source repo: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
