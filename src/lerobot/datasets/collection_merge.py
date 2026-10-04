"""Merge finalized collection sessions into a local root dataset without Hub uploads."""

import json
import shutil
import tempfile
from pathlib import Path

from filelock import FileLock

from lerobot.datasets.dataset_tools import merge_datasets, split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import create_lerobot_dataset_card
from lerobot.utils.constants import HF_LEROBOT_HOME

SESSION_MANIFEST = "meta/collection_sessions.json"


def merge_collection_session(session: LeRobotDataset, root_repo_id: str, root_dir: Path | None = None) -> int:
    """Append only new episodes locally, retaining the previous root as a backup.

    If no local root exists, download the Hub root once as a read-only starting point.
    After that, the local root is authoritative and merging works offline. The merged
    dataset and import ledger are built separately before switching the root directory.
    """
    if session.num_episodes == 0:
        return 0
    session = LeRobotDataset(session.repo_id, root=session.root)
    root = (Path(root_dir) if root_dir is not None else HF_LEROBOT_HOME / root_repo_id).resolve()
    if session.root.resolve() == root or session.repo_id == root_repo_id:
        raise ValueError("The recording session must be separate from the local root dataset")
    root.parent.mkdir(parents=True, exist_ok=True)
    backup = root.with_name(root.name + ".previous")
    with FileLock(str(root.with_name(root.name + ".merge.lock"))):
        # Recover if a process stopped between the two directory renames.
        if not root.exists() and backup.exists():
            backup.rename(root)
        if root.exists():
            existing = LeRobotDataset(root_repo_id, root=root)
        else:
            existing = LeRobotDataset(root_repo_id, revision="main")
        manifest_path = existing.root / SESSION_MANIFEST
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sessions": {}}
        previous = manifest["sessions"].get(session.repo_id, 0)
        if previous > session.num_episodes:
            raise ValueError(
                "Local session has fewer episodes than already merged; refusing to duplicate data"
            )
        if previous == session.num_episodes:
            return 0

        with tempfile.TemporaryDirectory(prefix=".nexarm-merge-", dir=root.parent) as directory:
            staging = Path(directory)
            new_session = session
            if previous:
                new_session = split_dataset(
                    session,
                    splits={"new": list(range(previous, session.num_episodes))},
                    output_dir=staging / "new-episodes",
                )["new"]
            merged = merge_datasets(
                [existing, new_session],
                output_repo_id=root_repo_id,
                output_dir=staging / "merged",
                concatenate_videos=False,
                concatenate_data=False,
            )
            manifest["sessions"][session.repo_id] = session.num_episodes
            (merged.root / SESSION_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
            card = create_lerobot_dataset_card(
                tags=["nexarm", "merged"], dataset_info=merged.meta.info, repo_id=root_repo_id
            )
            card.save(merged.root / "README.md")
            if root.exists():
                if backup.exists():
                    shutil.rmtree(backup)
                root.rename(backup)
            try:
                merged.root.rename(root)
            except BaseException:
                if not root.exists() and backup.exists():
                    backup.rename(root)
                raise
    return session.num_episodes - previous
