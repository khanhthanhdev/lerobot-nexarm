# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.mobile_nexarm_artifact import (
    EpisodeArtifactWriter,
    EpisodeMetadata,
    split_for_seed,
    validate_episode_artifact,
    write_artifacts_to_lerobot,
)
from lerobot.robots.mobile_bi_nexarm_sim import load_contract


def _artifact(path: Path) -> Path:
    contract = load_contract()
    writer = EpisodeArtifactWriter(
        path, EpisodeMetadata(task="place the red cube in the green zone", seed=8, backend="mujoco")
    )
    for index in range(3):
        writer.add_frame(
            timestamp=index / 30,
            state=contract.home_action,
            action=contract.home_action,
            cameras={
                "front": np.full((24, 32, 3), index * 10, dtype=np.uint8),
                "left_wrist": np.full((24, 32, 3), 40 + index * 10, dtype=np.uint8),
            },
        )
    return writer.finalize(success=True, success_reason="success")


def test_artifact_round_trip_and_seed_splits(tmp_path: Path) -> None:
    episode = _artifact(tmp_path / "episode")
    metadata = validate_episode_artifact(episode)
    assert metadata["success"]
    assert split_for_seed(7, "maniskill") == "train"
    assert split_for_seed(8, "maniskill") == "validation"
    assert split_for_seed(9, "isaac") == "test"


def test_artifact_rejects_missing_frame_and_corrupt_contract(tmp_path: Path) -> None:
    episode = _artifact(tmp_path / "episode")
    (episode / "front.mp4").unlink()
    with pytest.raises(ValueError, match="missing camera"):
        validate_episode_artifact(episode, decode_videos=False)

    episode = _artifact(tmp_path / "second")
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_names"][0] = "wrong"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="feature order"):
        validate_episode_artifact(episode, decode_videos=False)


def test_writer_rejects_timestamp_drift_and_non_finite_values(tmp_path: Path) -> None:
    contract = load_contract()
    writer = EpisodeArtifactWriter(tmp_path / "drift", EpisodeMetadata(task="task", seed=1, backend="mujoco"))
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    writer.add_frame(
        timestamp=0, state=contract.home_action, action=contract.home_action, cameras={"front": frame}
    )
    bad = contract.home_action | {"left_shoulder_pan.pos": float("nan")}
    with pytest.raises(ValueError, match="non-finite"):
        writer.add_frame(timestamp=1 / 30, state=bad, action=contract.home_action, cameras={"front": frame})


def test_bridge_reloads_as_lerobot_dataset_v3(tmp_path: Path) -> None:
    episode = _artifact(tmp_path / "episode")
    dataset = write_artifacts_to_lerobot(
        [episode],
        repo_id="local/mobile_nexarm_test",
        root=tmp_path / "dataset",
        use_videos=False,
    )
    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 3
    assert dataset.fps == 30
    frame = dataset[0]
    assert frame["observation.state"].shape == (16,)
    assert frame["action"].shape == (16,)
    assert np.isfinite(frame["observation.state"].numpy()).all()
