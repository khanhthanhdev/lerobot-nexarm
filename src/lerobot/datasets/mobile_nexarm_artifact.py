# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Neutral mobile NexArm episode artifacts and LeRobotDataset v3 bridge."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from lerobot.robots.mobile_bi_nexarm_sim.contract import STATE_ACTION_NAMES

SCHEMA_VERSION = 1
TRAJECTORY_FILE = "trajectory.npz"
METADATA_FILE = "episode.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class EpisodeMetadata:
    task: str
    seed: int
    backend: str
    fps: int = 30
    domain_parameters: dict[str, Any] = field(default_factory=dict)
    stage_traces: list[dict[str, Any]] = field(default_factory=list)
    repair_attempts: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    success_reason: str | None = None
    source_asset_hashes: dict[str, str] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=lambda: list(STATE_ACTION_NAMES))
    cameras: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION


class EpisodeArtifactWriter:
    """Collect synchronized frames and write NPZ + MP4 + JSON atomically per file."""

    def __init__(
        self,
        output_dir: str | Path,
        metadata: EpisodeMetadata,
        *,
        overwrite: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.metadata = metadata
        self.overwrite = overwrite
        self._timestamps: list[float] = []
        self._states: list[npt.NDArray[np.float32]] = []
        self._actions: list[npt.NDArray[np.float32]] = []
        self._frames: dict[str, list[npt.NDArray[np.uint8]]] = {}

    def add_frame(
        self,
        *,
        timestamp: float,
        state: Mapping[str, float] | Sequence[float] | npt.NDArray[np.floating[Any]],
        action: Mapping[str, float] | Sequence[float] | npt.NDArray[np.floating[Any]],
        cameras: Mapping[str, npt.NDArray[np.uint8]],
    ) -> None:
        if self._timestamps and timestamp <= self._timestamps[-1]:
            raise ValueError("episode timestamps must be strictly increasing")
        state_array = _canonical_array(state, "state")
        action_array = _canonical_array(action, "action")
        expected_cameras = set(self._frames) if self._frames else set(cameras)
        if set(cameras) != expected_cameras:
            raise ValueError(
                f"camera set changed: expected {sorted(expected_cameras)}, got {sorted(cameras)}"
            )
        for name, image in cameras.items():
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8 [H,W,3], got {image.dtype} {image.shape}")
            prior = self._frames.setdefault(name, [])
            if prior and image.shape != prior[0].shape:
                raise ValueError(f"{name} frame shape changed from {prior[0].shape} to {image.shape}")
            prior.append(image.copy())
        self._timestamps.append(float(timestamp))
        self._states.append(state_array)
        self._actions.append(action_array)

    def finalize(
        self,
        *,
        success: bool,
        success_reason: str | None,
        source_assets: Iterable[str | Path] = (),
    ) -> Path:
        if not self._timestamps:
            raise ValueError("cannot finalize an empty episode")
        if self.output_dir.exists() and any(self.output_dir.iterdir()) and not self.overwrite:
            raise FileExistsError(f"episode directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamps = np.asarray(self._timestamps, dtype=np.float64)
        state = np.stack(self._states).astype(np.float32, copy=False)
        action = np.stack(self._actions).astype(np.float32, copy=False)
        trajectory = self.output_dir / TRAJECTORY_FILE
        np.savez_compressed(
            trajectory,
            timestamp=timestamps,
            state=state,
            action=action,
            feature_names=np.asarray(STATE_ACTION_NAMES),
        )
        for name, frames in self._frames.items():
            _write_mp4(self.output_dir / f"{name}.mp4", frames, self.metadata.fps)
        self.metadata.success = success
        self.metadata.success_reason = success_reason
        self.metadata.cameras = list(self._frames)
        self.metadata.source_asset_hashes = {str(Path(path)): sha256_file(path) for path in source_assets}
        metadata_path = self.output_dir / METADATA_FILE
        metadata_path.write_text(
            json.dumps(asdict(self.metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_episode_artifact(self.output_dir)
        return self.output_dir


def _canonical_array(
    values: Mapping[str, float] | Sequence[float] | npt.NDArray[np.floating[Any]], label: str
) -> npt.NDArray[np.float32]:
    if isinstance(values, Mapping):
        missing = [name for name in STATE_ACTION_NAMES if name not in values]
        extra = sorted(set(values) - set(STATE_ACTION_NAMES))
        if missing or extra:
            raise ValueError(f"{label} contract mismatch; missing={missing}, extra={extra}")
        array = np.asarray([values[name] for name in STATE_ACTION_NAMES], dtype=np.float32)
    else:
        array = np.asarray(values, dtype=np.float32)
    if array.shape != (len(STATE_ACTION_NAMES),):
        raise ValueError(f"{label} must have shape ({len(STATE_ACTION_NAMES)},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def _write_mp4(path: Path, frames: Sequence[npt.NDArray[np.uint8]], fps: int) -> None:
    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"video encoder produced no data for {path}")


def _read_video(path: Path) -> list[npt.NDArray[np.uint8]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode camera video {path}")
    frames: list[npt.NDArray[np.uint8]] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def load_episode_artifact(
    episode_dir: str | Path, *, decode_videos: bool = False
) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]], dict[str, list[npt.NDArray[np.uint8]]]]:
    path = Path(episode_dir)
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    with np.load(path / TRAJECTORY_FILE) as payload:
        trajectory = {name: payload[name].copy() for name in payload.files}
    videos = (
        {name: _read_video(path / f"{name}.mp4") for name in metadata["cameras"]} if decode_videos else {}
    )
    return metadata, trajectory, videos


def validate_episode_artifact(
    episode_dir: str | Path,
    *,
    timestamp_tolerance_s: float = 0.002,
    decode_videos: bool = True,
) -> dict[str, Any]:
    metadata, trajectory, videos = load_episode_artifact(episode_dir, decode_videos=decode_videos)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported episode schema {metadata.get('schema_version')}")
    if tuple(metadata.get("feature_names", ())) != STATE_ACTION_NAMES:
        raise ValueError("episode feature order differs from the physical 16-D contract")
    timestamp = trajectory.get("timestamp")
    state = trajectory.get("state")
    action = trajectory.get("action")
    if timestamp is None or state is None or action is None:
        raise ValueError("trajectory.npz must contain timestamp, state, and action")
    expected_shape = (len(timestamp), len(STATE_ACTION_NAMES))
    if state.shape != expected_shape or action.shape != expected_shape:
        raise ValueError(
            f"trajectory shape mismatch: state={state.shape}, action={action.shape}, expected={expected_shape}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("trajectory contains non-finite values")
    if len(timestamp) > 1:
        expected_dt = 1 / int(metadata["fps"])
        drift = np.max(np.abs(np.diff(timestamp) - expected_dt))
        if drift > timestamp_tolerance_s:
            raise ValueError(f"timestamps drift by {drift:.6f}s from {metadata['fps']} FPS")
    for camera in metadata.get("cameras", []):
        video_path = Path(episode_dir) / f"{camera}.mp4"
        if not video_path.is_file():
            raise ValueError(f"missing camera stream {video_path.name}")
        if decode_videos and len(videos[camera]) != len(timestamp):
            raise ValueError(
                f"{camera} has {len(videos[camera])} frames for {len(timestamp)} trajectory steps"
            )
    return metadata


def split_for_seed(seed: int, source: str) -> str:
    """Stable 80/10/10 split using disjoint source-local seed residues."""
    if source not in {"maniskill", "isaac"}:
        raise ValueError("source must be 'maniskill' or 'isaac'")
    bucket = seed % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def write_artifacts_to_lerobot(
    episode_dirs: Sequence[str | Path],
    *,
    repo_id: str,
    root: str | Path,
    accepted_only: bool = True,
    use_videos: bool = True,
) -> Any:
    """Bridge accepted artifacts without importing ManiSkill or Isaac."""
    if not episode_dirs:
        raise ValueError("no episode artifacts supplied")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    loaded = [(Path(path), *load_episode_artifact(path, decode_videos=True)) for path in episode_dirs]
    if accepted_only:
        loaded = [item for item in loaded if item[1]["success"]]
    if not loaded:
        raise ValueError("no accepted episodes to bridge")
    first_metadata, first_trajectory, first_videos = loaded[0][1:]
    del first_trajectory
    camera_shapes = {name: tuple(frames[0].shape) for name, frames in first_videos.items()}
    action_hw = dict.fromkeys(STATE_ACTION_NAMES, float)
    observation_hw: dict[str, type | tuple[int, ...]] = dict.fromkeys(STATE_ACTION_NAMES, float)
    observation_hw.update(camera_shapes)
    features = {
        **hw_to_dataset_features(action_hw, ACTION, use_videos),
        **hw_to_dataset_features(observation_hw, OBS_STR, use_videos),
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(first_metadata["fps"]),
        features=features,
        root=root,
        robot_type="mobile_bi_nexarm_sim",
        use_videos=use_videos,
    )
    for _, metadata, trajectory, videos in loaded:
        if metadata["fps"] != first_metadata["fps"] or list(videos) != list(first_videos):
            raise ValueError("all artifacts must share FPS and ordered camera names")
        num_frames = len(trajectory["timestamp"])
        if any(len(frames) != num_frames for frames in videos.values()):
            raise ValueError("camera frame count differs from trajectory length")
        for index in range(num_frames):
            frame: dict[str, Any] = {
                "observation.state": trajectory["state"][index],
                "action": trajectory["action"][index],
                "task": metadata["task"],
            }
            frame.update({f"{OBS_STR}.images.{name}": frames[index] for name, frames in videos.items()})
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return dataset
