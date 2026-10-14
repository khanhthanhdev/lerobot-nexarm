# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Read side of the dataset API: list, inspect, episodes, joint arrays, video frames, curation.

Works directly on the LeRobot v3.0 files (``meta/info.json``, ``meta/episodes``, ``data``, ``videos``)
so browsing a dataset never loads torch or opens a training pipeline. Video frames are decoded on
demand: the phase 0 spike measured about 5 ms per random seek on a short-GOP AV1 file, so no cache is kept
beyond a few open containers.
"""

import collections
import json
import re
import threading
from pathlib import Path
from typing import Any

import av
import cv2
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from pydantic import BaseModel

REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_OPEN_CONTAINERS = 4


class DatasetNotFoundError(KeyError):
    pass


class DatasetSummary(BaseModel):
    repo_id: str
    fps: int
    robot_type: str | None
    total_episodes: int
    total_frames: int
    cameras: list[str]
    excluded_episodes: int = 0


class EpisodeSummary(BaseModel):
    episode_index: int
    length: int
    tasks: list[str]
    excluded: bool


class JointsData(BaseModel):
    names: list[str]
    timestamps: list[float]
    action: list[list[float]]
    state: list[list[float]]


def valid_repo_id(repo_id: str) -> bool:
    return REPO_ID_RE.match(repo_id) is not None and ".." not in repo_id


class DatasetStore:
    def __init__(self, root: Path, curation_dir: Path) -> None:
        self.root = root
        self._curation_dir = curation_dir
        self._containers: collections.OrderedDict[Path, Any] = collections.OrderedDict()
        self._decode_lock = threading.Lock()

    # -- locating ------------------------------------------------------------------------------
    def path(self, repo_id: str) -> Path:
        if not valid_repo_id(repo_id):
            raise DatasetNotFoundError(repo_id)
        path = (self.root / repo_id).resolve()
        if self.root.resolve() not in path.parents or not (path / "meta" / "info.json").is_file():
            raise DatasetNotFoundError(repo_id)
        return path

    def _info(self, repo_id: str) -> dict[str, Any]:
        return json.loads((self.path(repo_id) / "meta" / "info.json").read_text(encoding="utf-8"))

    # -- listing -------------------------------------------------------------------------------
    def list_local(self) -> list[DatasetSummary]:
        found = []
        if self.root.is_dir():
            for info in sorted(self.root.glob("*/*/meta/info.json")):
                repo_id = f"{info.parents[2].name}/{info.parents[1].name}"
                try:
                    found.append(self.summary(repo_id))
                except (DatasetNotFoundError, ValueError, KeyError):
                    continue
        return found

    def summary(self, repo_id: str) -> DatasetSummary:
        info = self._info(repo_id)
        return DatasetSummary(
            repo_id=repo_id,
            fps=info["fps"],
            robot_type=info.get("robot_type"),
            total_episodes=info["total_episodes"],
            total_frames=info["total_frames"],
            cameras=[k for k, f in info["features"].items() if f["dtype"] in ("video", "image")],
            excluded_episodes=len(self.excluded(repo_id)),
        )

    def info(self, repo_id: str) -> dict[str, Any]:
        return self._info(repo_id)

    # -- episodes ------------------------------------------------------------------------------
    def _episode_table(self, repo_id: str) -> list[dict[str, Any]]:
        files = sorted((self.path(repo_id) / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        rows: list[dict[str, Any]] = []
        for f in files:
            table = pq.read_table(f)
            rows.extend(
                table.select([c for c in table.column_names if not c.startswith("stats/")]).to_pylist()
            )
        return sorted(rows, key=lambda r: r["episode_index"])

    def episodes(self, repo_id: str) -> list[EpisodeSummary]:
        excluded = set(self.excluded(repo_id))
        return [
            EpisodeSummary(
                episode_index=r["episode_index"],
                length=r["length"],
                tasks=list(r["tasks"]),
                excluded=r["episode_index"] in excluded,
            )
            for r in self._episode_table(repo_id)
        ]

    def _episode_row(self, repo_id: str, episode: int) -> dict[str, Any]:
        for row in self._episode_table(repo_id):
            if row["episode_index"] == episode:
                return row
        raise DatasetNotFoundError(f"{repo_id} episode {episode}")

    def joints(self, repo_id: str, episode: int) -> JointsData:
        info = self._info(repo_id)
        row = self._episode_row(repo_id, episode)
        data_file = self.path(repo_id) / info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
        )
        table = pds.dataset(data_file).to_table(
            columns=["timestamp", "action", "observation.state"],
            filter=pc.field("episode_index") == episode,
        )
        return JointsData(
            names=list(info["features"]["action"]["names"]),
            timestamps=table["timestamp"].to_pylist(),
            action=table["action"].to_pylist(),
            state=table["observation.state"].to_pylist(),
        )

    # -- video frames --------------------------------------------------------------------------
    def frame_jpeg(self, repo_id: str, episode: int, camera: str, frame: int, max_width: int = 640) -> bytes:
        info = self._info(repo_id)
        feature = info["features"].get(camera)
        if feature is None or feature["dtype"] != "video":
            raise DatasetNotFoundError(f"{repo_id} camera {camera}")
        row = self._episode_row(repo_id, episode)
        if not 0 <= frame < row["length"]:
            raise IndexError(f"frame {frame} outside 0..{row['length'] - 1}")
        video = self.path(repo_id) / info["video_path"].format(
            video_key=camera,
            chunk_index=row[f"videos/{camera}/chunk_index"],
            file_index=row[f"videos/{camera}/file_index"],
        )
        timestamp = row[f"videos/{camera}/from_timestamp"] + frame / info["fps"]
        rgb = self._decode(video, timestamp, info["fps"])
        if rgb.shape[1] > max_width:
            scale = max_width / rgb.shape[1]
            rgb = cv2.resize(rgb, (max_width, round(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("jpeg encoding failed")
        return jpeg.tobytes()

    def _decode(self, video: Path, timestamp: float, fps: int) -> Any:
        with self._decode_lock:
            container = self._containers.pop(video, None)
            if container is None:
                container = av.open(str(video))
                container.streams.video[0].thread_type = "AUTO"
            self._containers[video] = container
            while len(self._containers) > _OPEN_CONTAINERS:
                _, old = self._containers.popitem(last=False)
                old.close()
            stream = container.streams.video[0]
            container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
            half_frame = 0.5 / fps
            last = None
            for decoded in container.decode(stream):
                last = decoded
                if decoded.time is not None and decoded.time >= timestamp - half_frame:
                    break
            if last is None:
                raise RuntimeError(f"no frame decoded from {video}")
            return last.to_ndarray(format="rgb24")

    def close(self) -> None:
        with self._decode_lock:
            for container in self._containers.values():
                container.close()
            self._containers.clear()

    # -- curation (nothing is ever deleted) ----------------------------------------------------
    def _curation_file(self, repo_id: str) -> Path:
        return self._curation_dir / f"{repo_id.replace('/', '__')}.json"

    def excluded(self, repo_id: str) -> list[int]:
        f = self._curation_file(repo_id)
        return sorted(json.loads(f.read_text(encoding="utf-8"))) if f.is_file() else []

    def set_excluded(self, repo_id: str, episodes: list[int]) -> list[int]:
        total = self._info(repo_id)["total_episodes"]
        if any(not 0 <= e < total for e in episodes):
            raise ValueError(f"episode indices must be within 0..{total - 1}")
        unique = sorted(set(episodes))
        self._curation_dir.mkdir(parents=True, exist_ok=True)
        self._curation_file(repo_id).write_text(json.dumps(unique), encoding="utf-8")
        return unique
