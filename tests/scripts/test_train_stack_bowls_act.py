# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.nexarm import train_stack_bowls_act as launcher


def test_local_dry_run_preserves_dataset_and_output_paths(monkeypatch, tmp_path, capsys):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--dataset-root",
            "dataset",
            "--repo-id",
            "owner/bowls",
            "--dry-run",
            "--output_dir=result",
        ],
    )
    meta = SimpleNamespace(
        revision="main", total_episodes=4, total_frames=100, total_tasks=1, fps=30, features={}
    )
    metadata = Mock(return_value=meta)
    monkeypatch.setattr(launcher, "LeRobotDatasetMetadata", metadata)
    monkeypatch.setattr(launcher, "validate_metadata", Mock())
    monkeypatch.setattr(launcher, "resolve_policy", Mock(return_value=SimpleNamespace(chunk_size=100)))
    hub = Mock(side_effect=AssertionError("Local dry run must not query Hub"))
    monkeypatch.setattr(launcher, "HfApi", hub)
    run = Mock()
    monkeypatch.setattr(launcher.subprocess, "run", run)
    launcher.main()
    metadata.assert_called_once_with("owner/bowls", root=root, revision="main", local_files_only=True)
    command = capsys.readouterr().out
    assert "--dataset.repo_id=owner/bowls" in command
    assert f"--output_dir={tmp_path / 'result'}" in command
    run.assert_not_called()


def test_cpu_fp16_rejected_before_dataset_access(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train", "--device", "cpu", "--mixed-precision", "fp16"])
    hub = Mock()
    monkeypatch.setattr(launcher, "HfApi", hub)
    with pytest.raises(SystemExit):
        launcher.main()
    hub.assert_not_called()


def stack_bowls_meta(fps, shapes):
    features = {
        key: {"dtype": "float32", "shape": [6], "names": launcher.JOINT_NAMES}
        for key in ("observation.state", "action")
    }
    for name, shape in shapes.items():
        features[f"observation.images.{name}"] = {"dtype": "video", "shape": list(shape)}
    return SimpleNamespace(
        info=SimpleNamespace(robot_type="nexarm_follower"),
        fps=fps,
        total_episodes=4,
        features=features,
        stats=dict.fromkeys(features, {}),
    )


def test_metadata_fps_and_resolution_come_from_the_dataset():
    meta = stack_bowls_meta(15, {"front": (240, 320, 3), "wrist": (240, 320, 3), "top": (240, 320, 3)})
    launcher.validate_metadata(meta)
    assert launcher.camera_shape(meta) == (240, 320, 3)
    assert launcher.dataset_cameras(meta)[-1] == "observation.images.top"

    mixed = stack_bowls_meta(30, {"front": (480, 640, 3), "wrist": (240, 320, 3)})
    with pytest.raises(ValueError, match="share one resolution"):
        launcher.validate_metadata(mixed)
