"""Load a LeRobot dataset from a local root and fail on missing files instead of downloading them.

Given a ``root``, LeRobot only contacts the Hub when a file it needs is missing there. These helpers
check those files first, so an incomplete local copy raises ``FileNotFoundError`` with the missing paths.
"""

import json
from pathlib import Path
from typing import Any

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import INFO_PATH

HUB_DISABLED = "Hub downloads are disabled for this dataset."


def _raise_missing(root: Path, what: str, missing: list[Path]) -> None:
    if missing:
        listed = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Local dataset {what} is incomplete at {root}. Missing (up to 5): {listed}. {HUB_DISABLED}"
        )


def load_local_metadata(repo_id: str, root: str | Path, **kwargs: Any) -> LeRobotDatasetMetadata:
    """``LeRobotDatasetMetadata`` for ``root``, after checking that its ``meta/`` files are present."""
    root = Path(root)
    info_path = root / INFO_PATH
    _raise_missing(root, "metadata", [] if info_path.is_file() else [info_path])
    info = json.loads(info_path.read_text(encoding="utf-8"))
    missing = []
    if info.get("total_tasks", 0) > 0 and not (root / "meta" / "tasks.parquet").is_file():
        missing.append(root / "meta" / "tasks.parquet")
    if info.get("total_episodes", 0) > 0 and not any((root / "meta" / "episodes").glob("**/*.parquet")):
        missing.append(root / "meta" / "episodes")
    _raise_missing(root, "metadata", missing)
    return LeRobotDatasetMetadata(repo_id, root=root, **kwargs)


def load_local_dataset(
    repo_id: str, root: str | Path, *, episodes: list[int] | None = None, **kwargs: Any
) -> LeRobotDataset:
    """``LeRobotDataset`` for ``root``, after checking the data and video files of ``episodes``."""
    root = Path(root)
    meta = load_local_metadata(repo_id, root, revision=kwargs.get("revision"))
    selected = episodes if episodes is not None else range(meta.total_episodes)
    paths = {meta.get_data_file_path(index) for index in selected}
    paths |= {meta.get_video_file_path(index, key) for index in selected for key in meta.video_keys}
    _raise_missing(root, "episode data", sorted(root / path for path in paths if not (root / path).is_file()))
    return LeRobotDataset(repo_id, root=root, episodes=episodes, **kwargs)
