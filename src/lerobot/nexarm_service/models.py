# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Policy checkpoints: discover local and Hub models and guard deployment by robot type."""

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lerobot.nexarm_service.dataset_store import DatasetNotFoundError, DatasetStore

REQUIRED_ROBOT_TYPE = "nexarm_follower"
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


class ModelInfo(BaseModel):
    id: str  # a local path or a Hub repo id
    source: str  # job | imported | hub
    policy_type: str | None = None
    step: int | None = None
    dataset_repo_id: str | None = None
    job_id: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def describe_checkpoint(path: Path, source: str, job_id: str | None = None) -> ModelInfo:
    config = _read_json(path / "config.json")
    train = _read_json(path / "train_config.json")
    step = int(path.parent.name) if path.parent.name.isdigit() else None
    return ModelInfo(
        id=str(path),
        source=source,
        policy_type=config.get("type"),
        step=step,
        dataset_repo_id=(train.get("dataset") or {}).get("repo_id"),
        job_id=job_id,
    )


class ModelStore:
    def __init__(self, home: Path, jobs: Any, dataset_store: DatasetStore) -> None:
        self._imported = home / "models"
        self._jobs = jobs
        self._datasets = dataset_store

    def list_local(self) -> list[ModelInfo]:
        found = []
        for job in self._jobs.list():
            for ckpt in self._jobs.checkpoints(job.id):
                found.append(describe_checkpoint(Path(ckpt["path"]), "job", job.id))
        if self._imported.is_dir():
            for config in sorted(self._imported.glob("*/*/config.json")):
                found.append(describe_checkpoint(config.parent, "imported"))
        return found

    def list_hub(self, author: str, limit: int = 100) -> list[ModelInfo]:
        from huggingface_hub import HfApi

        return [
            ModelInfo(id=m.id, source="hub")
            for m in HfApi().list_models(author=author, filter="lerobot", limit=limit)
        ]

    def import_from_hub(self, repo_id: str, revision: str | None = None) -> dict[str, Any]:
        from huggingface_hub import snapshot_download

        if not _MODEL_ID_RE.match(repo_id) or ".." in repo_id:
            raise ValueError(f"{repo_id!r} is not a valid repo id (expected owner/name)")
        target = self._imported / repo_id.replace("/", "__") / "pretrained_model"
        snapshot_download(repo_id, repo_type="model", revision=revision, local_dir=target)
        return {"path": str(target)}

    # -- deploy guard --------------------------------------------------------------------------
    def check_robot_type(self, checkpoint: str, allow_unverified: bool = False) -> str:
        """Refuse a checkpoint trained on another robot type; returns the verified robot type.

        The training dataset is found through the checkpoint's ``train_config.json`` and read from the
        local dataset store or, for Hub datasets, from the Hub. A checkpoint with no readable provenance is
        refused too unless the caller explicitly allows it.
        """
        train = self._train_config(checkpoint)
        repo_id = (train.get("dataset") or {}).get("repo_id")
        robot_type = self._dataset_robot_type(repo_id) if repo_id else None
        if robot_type is None:
            if allow_unverified:
                return "unverified"
            raise ValueError(
                "cannot verify which robot this checkpoint was trained for (no readable training dataset); "
                "pass allow_unverified to deploy it anyway"
            )
        if robot_type != REQUIRED_ROBOT_TYPE:
            raise ValueError(
                f"checkpoint was trained on robot type {robot_type!r} (dataset {repo_id}); "
                f"this engine only deploys {REQUIRED_ROBOT_TYPE!r} policies"
            )
        return robot_type

    def _train_config(self, checkpoint: str) -> dict[str, Any]:
        path = Path(checkpoint)
        if path.is_dir():
            return _read_json(path / "train_config.json")
        try:
            from huggingface_hub import hf_hub_download

            return _read_json(Path(hf_hub_download(checkpoint, "train_config.json")))
        except Exception:
            return {}

    def _dataset_robot_type(self, repo_id: str) -> str | None:
        try:
            return self._datasets.info(repo_id).get("robot_type")
        except DatasetNotFoundError:
            pass
        try:
            from huggingface_hub import hf_hub_download

            return _read_json(Path(hf_hub_download(repo_id, "meta/info.json", repo_type="dataset"))).get(
                "robot_type"
            )
        except Exception:
            return None
