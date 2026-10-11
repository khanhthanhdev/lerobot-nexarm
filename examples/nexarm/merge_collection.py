#!/usr/bin/env python
"""Merge a finalized recording session into the local root dataset without uploading."""

import argparse
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.collection_merge import merge_collection_session
from lerobot.utils.constants import HF_LEROBOT_HOME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Exact timestamped local collection ID")
    parser.add_argument("--root", type=Path, help="Local session directory, if outside the normal cache")
    parser.add_argument(
        "--root-repo-id",
        default="thanhkt/nexarm_stack_bowls_top",
        help="Merged root dataset (same default as prepare_collection.py; created if it does not exist)",
    )
    parser.add_argument("--merge-root", type=Path, help="Local merged root directory")
    args = parser.parse_args()
    session = LeRobotDataset(args.repo_id, root=args.root or HF_LEROBOT_HOME / args.repo_id)
    added = merge_collection_session(session, args.root_repo_id, args.merge_root)
    print(f"Merged {added} new episodes locally: {args.merge_root or HF_LEROBOT_HOME / args.root_repo_id}")


if __name__ == "__main__":
    main()
