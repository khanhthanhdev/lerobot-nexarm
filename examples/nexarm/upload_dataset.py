#!/usr/bin/env python

"""Re-upload a locally recorded LeRobot dataset to the Hugging Face Hub.

Useful when dataset recording completed locally but the upload failed due to
internet connection issues, timeout, or if --dataset.push_to_hub=false was used.

Usage:
    # Upload an existing local dataset by repo_id:
    uv run python examples/nexarm/upload_dataset.py --repo-id <username>/<dataset_name>

    # Upload from a specific local directory to a target repo_id:
    uv run python examples/nexarm/upload_dataset.py \
        --repo-id <username>/<dataset_name> \
        --root ~/.cache/huggingface/lerobot/<username>/<local_folder>

    # Use chunked resumable upload for flaky or slow connections:
    uv run python examples/nexarm/upload_dataset.py \
        --repo-id <username>/<dataset_name> \
        --large-folder
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-upload a locally saved LeRobot dataset to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face repo ID (e.g. 'thanhkt/nexarm_stack_bowls').",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Path to local dataset directory. If omitted, defaults to "
            "$HF_LEROBOT_HOME/<repo-id> (~/.cache/huggingface/lerobot/<repo-id>)."
        ),
    )
    parser.add_argument(
        "--large-folder",
        action="store_true",
        default=True,
        help="Use chunked resumable upload (HfApi.upload_large_folder). Recommended for videos.",
    )
    parser.add_argument(
        "--no-large-folder",
        dest="large_folder",
        action="store_false",
        help="Use standard single-commit upload_folder instead of upload_large_folder.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Set the Hugging Face dataset repository as private.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Optional branch name to upload to.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify Hugging Face authentication
    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"Authenticated as Hugging Face user: {username}")
    except Exception as e:
        print(f"Error: Not authenticated with Hugging Face ({e}).")
        print("Please run 'hf auth login' or export HF_TOKEN=<token> first.")
        sys.exit(1)

    # Determine root directory
    if args.root:
        dataset_root = Path(args.root).expanduser().resolve()
    else:
        dataset_root = (HF_LEROBOT_HOME / args.repo_id).resolve()

    if not dataset_root.exists():
        print(f"Error: Local dataset directory not found at: {dataset_root}")
        print(f"Available local datasets in {HF_LEROBOT_HOME}:")
        if HF_LEROBOT_HOME.exists():
            for p in HF_LEROBOT_HOME.glob("*/*"):
                if (p / "meta" / "info.json").exists():
                    rel = p.relative_to(HF_LEROBOT_HOME)
                    print(f"  - {rel}  ({p})")
        sys.exit(1)

    print(f"Loading local dataset from: {dataset_root}")
    try:
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=dataset_root,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print("Dataset successfully loaded:")
    print(f"  - Episodes: {dataset.num_episodes}")
    print(f"  - Frames:   {dataset.num_frames}")
    print(f"  - Robot:    {dataset.meta.robot_type}")
    print(f"  - Target:   https://huggingface.co/datasets/{args.repo_id}")

    print("\nStarting upload to Hugging Face Hub...")
    try:
        dataset.push_to_hub(
            branch=args.branch,
            private=args.private,
            upload_large_folder=args.large_folder,
        )
        print("\nUpload complete!")
        print(f"Dataset URL: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"\nUpload failed: {e}")
        print("Tips:")
        print("  - If connection was interrupted, running this script again will resume upload.")
        print("  - Ensure your HF token has write permission.")
        sys.exit(1)


if __name__ == "__main__":
    main()
