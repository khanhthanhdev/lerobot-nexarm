"""Merge finalized collection sessions into a local root dataset without Hub uploads."""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock
from huggingface_hub import get_token, hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from lerobot.datasets.dataset_tools import merge_datasets, split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import INFO_PATH, create_lerobot_dataset_card
from lerobot.utils.constants import DEFAULT_FEATURES, HF_LEROBOT_HOME

SESSION_MANIFEST = "meta/collection_sessions.json"


def local_root_dir(root_repo_id: str, root_dir: Path | None = None) -> Path:
    return (Path(root_dir) if root_dir is not None else HF_LEROBOT_HOME / root_repo_id).resolve()


def hub_root_info(root_repo_id: str) -> dict[str, Any] | None:
    """The root's Hub ``meta/info.json``, or None when the Hub has no such dataset.

    Network, offline and gated-repo errors are raised as RuntimeError: a new root must never be created
    by accident because the Hub was unreachable.
    """
    try:
        path = hf_hub_download(root_repo_id, INFO_PATH, repo_type="dataset", revision="main")
    except GatedRepoError as error:
        raise RuntimeError(
            f"Root dataset {root_repo_id} is gated on the Hub; accept its terms or log in (`hf auth login`)."
        ) from error
    except RepositoryNotFoundError as error:
        if get_token() is None:
            # The Hub answers "not found" for private repos when logged out.
            raise RuntimeError(
                f"Root dataset {root_repo_id} is not on the Hub for an anonymous client, and it may be private, so "
                "it cannot be told apart from a new root. Run `hf auth login` (also needed to create a new root on "
                "the first session), point --merge-root at the existing local root, or record without merging "
                "(--no-merge)."
            ) from error
        return None
    except Exception as error:
        raise RuntimeError(
            f"Could not read root dataset {root_repo_id} from the Hub ({error}). Connect to the network, "
            "point --merge-root at a local copy of the root, or record without merging (--no-merge)."
        ) from error
    return json.loads(Path(path).read_text())


def root_info(root_repo_id: str, root_dir: Path | None = None) -> dict[str, Any] | None:
    """``info.json`` of the local root (or its backup), else of the Hub root; None for a new root."""
    root = local_root_dir(root_repo_id, root_dir)
    for candidate in (root, root.with_name(root.name + ".previous")):
        info_path = candidate / INFO_PATH
        if info_path.is_file():
            return json.loads(info_path.read_text())
    return hub_root_info(root_repo_id)


def _comparable(features: dict[str, dict]) -> dict[str, tuple]:
    """Feature contract that merging requires to match; encoder ``info`` is ignored."""
    return {
        key: (
            feature.get("dtype"),
            tuple(feature.get("shape") or ()),
            tuple(feature["names"])
            if isinstance(feature.get("names"), list | tuple)
            else feature.get("names"),
        )
        for key, feature in features.items()
    }


def merge_incompatibilities(
    info: dict[str, Any], fps: int, robot_type: str | None, features: dict[str, dict]
) -> list[str]:
    """Differences between a planned session and a root ``info.json`` that would block merging."""
    problems = []
    if info.get("fps") != fps:
        problems.append(f"fps: root {info.get('fps')}, session {fps}")
    if info.get("robot_type") != robot_type:
        problems.append(f"robot_type: root {info.get('robot_type')!r}, session {robot_type!r}")
    root_features = _comparable(info.get("features", {}))
    session_features = _comparable({**features, **DEFAULT_FEATURES})
    if set(root_features) != set(session_features):
        only_root = sorted(set(root_features) - set(session_features))
        only_session = sorted(set(session_features) - set(root_features))
        problems.append(f"features: only in root {only_root}, only in session {only_session}")
    for key in sorted(set(root_features) & set(session_features)):
        if root_features[key] != session_features[key]:
            root_dtype, root_shape, _ = root_features[key]
            dtype, shape, _ = session_features[key]
            problems.append(
                f"{key}: root {root_dtype} {list(root_shape)}, session {dtype} {list(shape)}"
                + (" (names differ)" if (root_dtype, root_shape) == (dtype, shape) else "")
            )
    return problems


def check_session_mergeable(
    root_repo_id: str,
    root_dir: Path | None,
    *,
    fps: int,
    robot_type: str | None,
    features: dict[str, dict],
) -> bool:
    """Fail before recording when the session could not be merged into the root dataset.

    Returns False when the root does not exist yet (the first session will create it).
    """
    info = root_info(root_repo_id, root_dir)
    if info is None:
        message = (
            f"Root dataset {root_repo_id} was not found locally or on the Hub (check the id and "
            f"`hf auth login` for private roots); this session will create it at "
            f"{local_root_dir(root_repo_id, root_dir)}."
        )
        logging.warning(message)
        print(message, flush=True)
        return False
    problems = merge_incompatibilities(info, fps, robot_type, features)
    if problems:
        raise ValueError(
            f"This session cannot be merged into root dataset {root_repo_id}:\n  - "
            + "\n  - ".join(problems)
            + "\nRecord with the root's cameras, resolution and fps (prepare_collection.py: add or remove "
            "--no-top-cam; lerobot-record: edit --robot.cameras), choose another root (--root-repo-id / "
            "--root_repo_id), or record without merging (--no-merge / leave --root_repo_id unset)."
        )
    return True


def merge_collection_session(session: LeRobotDataset, root_repo_id: str, root_dir: Path | None = None) -> int:
    """Append only new episodes locally, retaining the previous root as a backup.

    If no local root exists, download the Hub root once as a read-only starting point; if the Hub
    has no such dataset either, the session becomes the first content of a new root. After that,
    the local root is authoritative and merging works offline. The merged dataset and import
    ledger are built separately before switching the root directory.
    """
    if session.num_episodes == 0:
        return 0
    session = LeRobotDataset(session.repo_id, root=session.root)
    root = local_root_dir(root_repo_id, root_dir)
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
        elif hub_root_info(root_repo_id) is not None:
            existing = LeRobotDataset(root_repo_id, revision="main")
        else:
            existing = None
        manifest_path = existing.root / SESSION_MANIFEST if existing is not None else None
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path is not None and manifest_path.exists()
            else {"sessions": {}}
        )
        previous = manifest["sessions"].get(session.repo_id, 0)
        if previous > session.num_episodes:
            raise ValueError(
                "Local session has fewer episodes than already merged; refusing to duplicate data"
            )
        if previous == session.num_episodes:
            return 0
        if existing is not None:
            root_contract = {
                "fps": existing.meta.fps,
                "robot_type": existing.meta.robot_type,
                "features": existing.meta.features,
            }
            problems = merge_incompatibilities(
                root_contract, session.meta.fps, session.meta.robot_type, session.meta.features
            )
            if problems:
                raise ValueError(
                    f"Cannot merge {session.repo_id} into {root_repo_id}: " + "; ".join(problems)
                )

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
                [*([existing] if existing is not None else []), new_session],
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
