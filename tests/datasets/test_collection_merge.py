"""Verify local merging preserves episodes without uploading to the Hub."""

from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import HfApi

from lerobot.datasets import LeRobotDataset, collection_merge as merge

FEATURES = {
    "action": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
    "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
}


def add_episode(dataset):
    dataset.add_frame(
        {
            "action": np.array([1], dtype=np.float32),
            "observation.state": np.array([2], dtype=np.float32),
            "task": "test collection",
        }
    )
    dataset.save_episode()


def make_dataset(repo_id, root, episodes):
    dataset = LeRobotDataset.create(repo_id, 30, root=root, features=FEATURES, use_videos=False)
    for _ in range(episodes):
        add_episode(dataset)
    dataset.finalize()
    return dataset


@pytest.fixture(autouse=True)
def forbid_uploads(monkeypatch):
    for method in ("upload_folder", "upload_large_folder", "create_commit", "delete_tag", "create_tag"):
        monkeypatch.setattr(HfApi, method, lambda *_, **__: pytest.fail("Unexpected Hub write"))
    monkeypatch.setattr(
        LeRobotDataset, "push_to_hub", lambda *_, **__: pytest.fail("Unexpected dataset upload")
    )


def test_merge_retry_resume_and_previous_root_backup(tmp_path):
    root_dir = tmp_path / "root"
    make_dataset("test/root", root_dir, 2)
    session = make_dataset("test/session_1", tmp_path / "session", 1)
    assert merge.merge_collection_session(session, "test/root", root_dir) == 1
    root = LeRobotDataset("test/root", root=root_dir)
    assert root.num_episodes == root.num_frames == 3
    assert LeRobotDataset("test/root", root=root_dir.with_name("root.previous")).num_episodes == 2
    assert merge.merge_collection_session(session, "test/root", root_dir) == 0

    resumed = LeRobotDataset.resume(session.repo_id, root=session.root)
    add_episode(resumed)
    resumed.finalize()
    assert merge.merge_collection_session(resumed, "test/root", root_dir) == 1
    root = LeRobotDataset("test/root", root=root_dir)
    assert root.num_episodes == root.num_frames == 4
    assert LeRobotDataset("test/root", root=root_dir.with_name("root.previous")).num_episodes == 3
    assert merge.merge_collection_session(resumed, "test/root", root_dir) == 0


def test_failed_install_rolls_back_existing_root(monkeypatch, tmp_path):
    root_dir = tmp_path / "root"
    make_dataset("test/root", root_dir, 2)
    session = make_dataset("test/session_1", tmp_path / "session", 1)
    rename = Path.rename

    def fail_install(path, target):
        if path.name == "merged":
            raise OSError("Unable to install merged data")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_install)
    with pytest.raises(OSError, match="Unable to install"):
        merge.merge_collection_session(session, "test/root", root_dir)
    assert LeRobotDataset("test/root", root=root_dir).num_episodes == 2
    assert LeRobotDataset(session.repo_id, root=session.root).num_episodes == 1
    assert not (root_dir / merge.SESSION_MANIFEST).exists()
    monkeypatch.setattr(Path, "rename", rename)
    assert merge.merge_collection_session(session, "test/root", root_dir) == 1


def test_missing_local_root_downloads_seed_once(monkeypatch, tmp_path):
    remote = make_dataset("test/root", tmp_path / "remote", 2)
    session = make_dataset("test/session_1", tmp_path / "session", 1)
    downloads = []

    def load(repo_id, root=None, **kwargs):
        if root is None:
            assert kwargs["revision"] == "main"
            downloads.append(repo_id)
            return remote
        return LeRobotDataset(repo_id, root=root, **kwargs)

    monkeypatch.setattr(merge, "LeRobotDataset", load)
    root_dir = tmp_path / "root"
    assert merge.merge_collection_session(session, "test/root", root_dir) == 1
    assert merge.merge_collection_session(session, "test/root", root_dir) == 0
    assert downloads == ["test/root"]
    assert LeRobotDataset("test/root", root=remote.root).num_episodes == 2
    assert LeRobotDataset("test/root", root=root_dir).num_episodes == 3
