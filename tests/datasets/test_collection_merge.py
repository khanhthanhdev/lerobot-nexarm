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
    monkeypatch.setattr(merge, "hub_root_info", lambda repo_id: {"repo_id": repo_id})
    root_dir = tmp_path / "root"
    assert merge.merge_collection_session(session, "test/root", root_dir) == 1
    assert merge.merge_collection_session(session, "test/root", root_dir) == 0
    assert downloads == ["test/root"]
    assert LeRobotDataset("test/root", root=remote.root).num_episodes == 2
    assert LeRobotDataset("test/root", root=root_dir).num_episodes == 3


def test_first_session_creates_a_new_root(monkeypatch, tmp_path):
    session = make_dataset("test/session_1", tmp_path / "session", 2)
    monkeypatch.setattr(merge, "hub_root_info", lambda _: None)
    root_dir = tmp_path / "new_root"

    assert merge.merge_collection_session(session, "test/new_root", root_dir) == 2
    root = LeRobotDataset("test/new_root", root=root_dir)
    assert root.num_episodes == root.num_frames == 2
    assert merge.merge_collection_session(session, "test/new_root", root_dir) == 0


def test_hub_not_found_means_new_root_but_other_errors_propagate(monkeypatch):
    from unittest.mock import Mock

    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    def missing(*_, **__):
        raise RepositoryNotFoundError("missing", response=Mock())

    monkeypatch.setattr(merge, "hf_hub_download", missing)
    monkeypatch.setattr(merge, "get_token", lambda: "token")
    assert merge.hub_root_info("test/missing") is None

    # Logged out, the Hub also answers "not found" for private repos, so it must not read as a new root.
    monkeypatch.setattr(merge, "get_token", lambda: None)
    with pytest.raises(RuntimeError, match="hf auth login"):
        merge.hub_root_info("test/private")

    def offline(*_, **__):
        raise OSError("network down")

    monkeypatch.setattr(merge, "hf_hub_download", offline)
    with pytest.raises(RuntimeError, match="network down.*--no-merge"):
        merge.hub_root_info("test/offline")

    def gated(*_, **__):
        raise GatedRepoError("gated", response=Mock())

    monkeypatch.setattr(merge, "hf_hub_download", gated)
    with pytest.raises(RuntimeError, match="gated"):
        merge.hub_root_info("test/gated")


def test_precheck_accepts_matching_session_and_rejects_camera_or_fps_changes(tmp_path):
    root_dir = tmp_path / "root"
    root = make_dataset("test/root", root_dir, 1)
    planned = {key: dict(value) for key, value in FEATURES.items()}
    kwargs = {"fps": 30, "robot_type": root.meta.robot_type}

    assert merge.check_session_mergeable("test/root", root_dir, features=planned, **kwargs)

    with_camera = {
        **planned,
        "observation.images.top": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
    }
    with pytest.raises(ValueError, match=r"only in session \['observation.images.top'\]"):
        merge.check_session_mergeable("test/root", root_dir, features=with_camera, **kwargs)
    with pytest.raises(ValueError, match="fps: root 30, session 15"):
        merge.check_session_mergeable("test/root", root_dir, features=planned, fps=15, robot_type=None)


def test_precheck_compares_camera_resolution():
    camera = {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"], "info": {}}
    info = {"fps": 30, "robot_type": "nexarm_follower", "features": {"observation.images.front": camera}}
    smaller = {**camera, "shape": (240, 320, 3)}
    del smaller["info"]
    problems = merge.merge_incompatibilities(
        info, 30, "nexarm_follower", {"observation.images.front": smaller}
    )
    assert "observation.images.front: root video [480, 640, 3], session video [240, 320, 3]" in problems


def test_precheck_new_root_is_allowed(monkeypatch, tmp_path):
    monkeypatch.setattr(merge, "hub_root_info", lambda _: None)
    assert not merge.check_session_mergeable(
        "test/new_root", tmp_path / "new_root", fps=30, robot_type="nexarm_follower", features=FEATURES
    )


def record_config(tmp_path, root_dir):
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.scripts.lerobot_record import RecordConfig
    from tests.mocks.mock_robot import MockRobotConfig
    from tests.mocks.mock_teleop import MockTeleopConfig

    dataset_cfg = DatasetRecordConfig(
        repo_id="test/session",
        single_task="Dummy task",
        root=tmp_path / "session",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    return RecordConfig(
        robot=MockRobotConfig(),
        dataset=dataset_cfg,
        teleop=MockTeleopConfig(),
        play_sounds=False,
        root_repo_id="test/root",
        merge_root=str(root_dir),
    )


def test_record_fails_before_recording_when_root_is_incompatible(tmp_path):
    pytest.importorskip("deepdiff")
    from lerobot.scripts.lerobot_record import record

    root_dir = tmp_path / "root"
    make_dataset("test/root", root_dir, 1)
    with pytest.raises(ValueError, match="cannot be merged into root dataset test/root"):
        record(record_config(tmp_path, root_dir))
    assert not (tmp_path / "session").exists()


def test_record_first_session_creates_root(monkeypatch, tmp_path):
    pytest.importorskip("deepdiff")
    from lerobot.scripts.lerobot_record import record

    monkeypatch.setattr(merge, "hub_root_info", lambda _: None)
    root_dir = tmp_path / "root"
    dataset = record(record_config(tmp_path, root_dir))
    root = LeRobotDataset("test/root", root=root_dir)
    assert root.num_episodes == dataset.num_episodes == 1
