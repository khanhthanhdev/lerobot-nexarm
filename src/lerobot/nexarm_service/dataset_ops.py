# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Write side of the dataset API: edits wrap ``lerobot.datasets.dataset_tools``; Hub transfers wrap
``huggingface_hub``. Every edit writes a NEW dataset and leaves its source untouched.
"""

import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.nexarm_service.dataset_store import DatasetNotFoundError, DatasetStore, valid_repo_id


class DatasetEditError(ValueError):
    pass


class DatasetOps:
    def __init__(self, store: DatasetStore, is_busy: Any = lambda repo_id: False) -> None:
        self._store = store
        self._is_busy = is_busy

    # -- guards --------------------------------------------------------------------------------
    def _require_source(self, repo_id: str) -> None:
        self._store.path(repo_id)  # raises DatasetNotFoundError
        if self._is_busy(repo_id):
            raise DatasetEditError(f"{repo_id} is being recorded; stop the session first")

    def _require_new(self, repo_id: str) -> Path:
        if not valid_repo_id(repo_id):
            raise DatasetEditError(f"{repo_id!r} is not a valid repo id (expected owner/name)")
        target = self._store.root / repo_id
        if target.exists():
            raise DatasetEditError(f"{repo_id} already exists; pick a new name")
        return target

    def _open(self, repo_id: str) -> Any:
        from lerobot.datasets import LeRobotDataset

        return LeRobotDataset(repo_id, root=self._store.path(repo_id))

    # -- edits ---------------------------------------------------------------------------------
    def delete_episodes(self, repo_id: str, episodes: list[int], output_repo_id: str) -> dict[str, Any]:
        from lerobot.datasets.dataset_tools import delete_episodes

        self._require_source(repo_id)
        target = self._require_new(output_repo_id)
        out = delete_episodes(self._open(repo_id), episodes, output_dir=target, repo_id=output_repo_id)
        return {"repo_id": output_repo_id, "total_episodes": out.meta.total_episodes}

    def drop_excluded(self, repo_id: str, output_repo_id: str) -> dict[str, Any]:
        excluded = self._store.excluded(repo_id)
        if not excluded:
            raise DatasetEditError("no episodes are excluded in this dataset")
        return self.delete_episodes(repo_id, excluded, output_repo_id)

    def split(self, repo_id: str, splits: dict[str, float | list[int]]) -> dict[str, Any]:
        """Each split becomes ``<repo_id>_<split name>``."""
        from lerobot.datasets.dataset_tools import split_dataset

        self._require_source(repo_id)
        targets = {name: self._require_new(f"{repo_id}_{name}") for name in splits}
        staging = self._store.root / ".nexarm_staging" / uuid.uuid4().hex[:8]
        try:
            split_dataset(self._open(repo_id), splits, output_dir=staging)
            made = {}
            for name, target in targets.items():
                if not (staging / name).exists():  # the library skips a split that gets no episodes
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staging / name), str(target))
                made[name] = f"{repo_id}_{name}"
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return {"splits": made}

    def merge(self, repo_ids: list[str], output_repo_id: str) -> dict[str, Any]:
        from lerobot.datasets.dataset_tools import merge_datasets

        if len(repo_ids) < 2:
            raise DatasetEditError("merge needs at least two datasets")
        for repo_id in repo_ids:
            self._require_source(repo_id)
        infos = [self._store.info(r) for r in repo_ids]
        if len({(i["fps"], i.get("robot_type")) for i in infos}) != 1:
            raise DatasetEditError("datasets must share fps and robot_type to be merged")
        target = self._require_new(output_repo_id)
        out = merge_datasets([self._open(r) for r in repo_ids], output_repo_id, output_dir=target)
        return {"repo_id": output_repo_id, "total_episodes": out.meta.total_episodes}

    def modify_features(
        self,
        repo_id: str,
        output_repo_id: str,
        remove: list[str] | None = None,
        add: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """``add`` maps a name to ``{"dtype", "shape", "names", "values"}``; values has one row per frame."""
        from lerobot.datasets.dataset_tools import modify_features

        self._require_source(repo_id)
        target = self._require_new(output_repo_id)
        dataset = self._open(repo_id)
        added: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        for name, spec in (add or {}).items():
            values = np.asarray(spec["values"], dtype=spec["dtype"])
            if len(values) != dataset.meta.total_frames:
                raise DatasetEditError(
                    f"feature {name!r} needs {dataset.meta.total_frames} values, got {len(values)}"
                )
            added[name] = (
                values,
                {"dtype": spec["dtype"], "shape": spec["shape"], "names": spec.get("names")},
            )
        if not added and not remove:
            raise DatasetEditError("nothing to change: pass features to add or remove")
        out = modify_features(
            dataset,
            add_features=added or None,
            remove_features=remove or None,
            output_dir=target,
            repo_id=output_repo_id,
        )
        return {"repo_id": output_repo_id, "features": sorted(out.meta.features)}

    def rename(self, repo_id: str, new_repo_id: str) -> dict[str, Any]:
        self._require_source(repo_id)
        target = self._require_new(new_repo_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self._store.path(repo_id)), str(target))
        return {"repo_id": new_repo_id}

    # -- Hub -----------------------------------------------------------------------------------
    def upload(self, repo_id: str, private: bool = True) -> dict[str, Any]:
        self._require_source(repo_id)
        self._open(repo_id).push_to_hub(private=private)
        return {"repo_id": repo_id, "url": f"https://huggingface.co/datasets/{repo_id}"}

    def import_from_hub(self, repo_id: str, revision: str | None = None) -> dict[str, Any]:
        from huggingface_hub import snapshot_download

        if not valid_repo_id(repo_id):
            raise DatasetEditError(f"{repo_id!r} is not a valid repo id (expected owner/name)")
        target = self._store.root / repo_id
        if target.exists():
            raise DatasetEditError(f"{repo_id} already exists locally")
        snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=target)
        try:
            self._store.path(repo_id)
        except DatasetNotFoundError as exc:
            shutil.rmtree(target, ignore_errors=True)
            raise DatasetEditError(f"{repo_id} is not a LeRobot dataset") from exc
        return {"repo_id": repo_id}


def list_hub_datasets(author: str, limit: int = 100) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    return [
        {
            "repo_id": d.id,
            "private": d.private,
            "downloads": d.downloads,
            "last_modified": str(d.last_modified),
        }
        for d in HfApi().list_datasets(author=author, filter="LeRobot", limit=limit)
    ]
