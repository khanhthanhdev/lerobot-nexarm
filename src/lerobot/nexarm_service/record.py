# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Recording session: teleoperate and save LeRobot v3.0 episodes, steered by web events.

The control loop follows ``lerobot.scripts.lerobot_record.record_loop`` (same processors, same dataset
frame building, so output matches ``lerobot-record``). The event model (exit early, re-record, pause,
resume, task text) follows MakerMods Lab ``makermodslab/record.py`` (Apache-2.0, Copyright 2025 The
HuggingFace Inc. team), reimplemented around a thread-safe flag set instead of keyboard listeners.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.nexarm_service.arm_worker import JOINT_PUBLISH_EVERY, ArmWorker, joint_update_payload
from lerobot.nexarm_service.events import EventHub
from lerobot.nexarm_service.profiles import RobotProfile
from lerobot.nexarm_service.safety import clamp_action
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


class RecordingWorker(ArmWorker):
    with_cameras = True
    thread_name = "nexarm-record"

    def __init__(
        self,
        profile: RobotProfile,
        hardware: Any,
        events: EventHub,
        datasets_root: Path,
        params: dict[str, Any],
    ) -> None:
        super().__init__(profile, hardware, events)
        if not params.get("repo_id") or not params.get("task"):
            raise ValueError("recording needs params.repo_id and params.task")
        if not profile.cameras:
            raise ValueError(f"profile {profile.name!r} has no cameras; configure them before recording")
        self._cfg = DatasetRecordConfig(
            repo_id=params["repo_id"],
            single_task=params["task"],
            fps=int(params.get("fps", 30)),
            episode_time_s=float(params.get("episode_time_s", 20)),
            reset_time_s=float(params.get("reset_time_s", 10)),
            num_episodes=int(params.get("num_episodes", 50)),
            video=bool(params.get("video", True)),
            push_to_hub=False,
        )
        if self._cfg.repo_id.split("/", 1)[-1].startswith("eval_"):
            raise ValueError("dataset names starting with 'eval_' are reserved for policy evaluation")
        self._resume = bool(params.get("resume", False))
        self._root = datasets_root / self._cfg.repo_id
        if self._resume != self._root.exists():
            raise ValueError(
                f"{self._cfg.repo_id} {'does not exist' if self._resume else 'already exists'}; "
                f"{'drop resume' if not self._resume else 'check the name'} or choose another name"
            )
        self._task = self._cfg.single_task
        self._dataset: LeRobotDataset | None = None
        self._encoding: VideoEncodingManager | None = None
        self._processors = make_default_processors()
        self._lock = threading.Lock()
        self._flags = {"exit_early": False, "rerecord": False, "finish": False}
        self._paused = threading.Event()
        self._phase = "preparing"
        self._episodes_recorded = 0
        self._elapsed_s = 0.0
        self._frames = 0

    # -- controls ------------------------------------------------------------------------------
    def control(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if action in self._flags:
            self._flags[action] = True
        elif action == "pause":
            self._paused.set()
        elif action == "resume":
            self._paused.clear()
        elif action == "set_task":
            task = str(payload.get("task", "")).strip()
            if not task:
                raise ValueError("set_task needs a non-empty task")
            self._task = task
        else:
            return super().control(action, payload)
        return self.extra_status()

    def extra_status(self) -> dict[str, Any]:
        return {
            "repo_id": self._cfg.repo_id,
            "phase": "paused" if self._paused.is_set() and self._phase == "recording" else self._phase,
            "episodes_recorded": self._episodes_recorded,
            "episodes_target": self._cfg.num_episodes,
            "episode_index": self._dataset.num_episodes if self._dataset else 0,
            "frames": self._frames,
            "elapsed_s": round(self._elapsed_s, 1),
            "duration_s": self._cfg.episode_time_s,
            "task": self._task,
        }

    # -- lifecycle -----------------------------------------------------------------------------
    def _ready(self) -> None:
        follower = self._follower
        teleop_proc, _, obs_proc = self._processors
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_proc,
                initial_features=create_initial_features(action=follower.action_features),
                use_videos=self._cfg.video,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=obs_proc,
                initial_features=create_initial_features(observation=follower.observation_features),
                use_videos=self._cfg.video,
            ),
        )
        cameras = len(follower.cameras)
        writer_kwargs = {
            "image_writer_processes": self._cfg.num_image_writer_processes,
            "image_writer_threads": self._cfg.num_image_writer_threads_per_camera * cameras,
            "batch_encoding_size": self._cfg.video_encoding_batch_size,
            "rgb_encoder": self._cfg.rgb_encoder,
            "depth_encoder": self._cfg.depth_encoder,
            "encoder_threads": self._cfg.encoder_threads,
            "streaming_encoding": self._cfg.streaming_encoding,
            "encoder_queue_maxsize": self._cfg.encoder_queue_maxsize,
        }
        if self._resume:
            self._dataset = LeRobotDataset.resume(self._cfg.repo_id, root=self._root, **writer_kwargs)
            self._check_resume_compatible(features)
        else:
            self._dataset = LeRobotDataset.create(
                self._cfg.repo_id,
                self._cfg.fps,
                root=self._root,
                robot_type=follower.robot_type,
                features=features,
                use_videos=self._cfg.video,
                **writer_kwargs,
            )
        self._encoding = VideoEncodingManager(self._dataset)
        self._encoding.__enter__()

    def _check_resume_compatible(self, features: dict[str, dict]) -> None:
        dataset = self._dataset
        assert dataset is not None
        if dataset.fps != self._cfg.fps:
            raise ValueError(f"dataset fps is {dataset.fps}, requested {self._cfg.fps}")
        if dataset.meta.robot_type != self._follower.robot_type:
            raise ValueError(
                f"dataset robot_type is {dataset.meta.robot_type!r}, this robot is {self._follower.robot_type!r}"
            )
        recorded = {k: v["shape"] for k, v in dataset.features.items() if k in features}
        wanted = {k: list(v["shape"]) for k, v in features.items()}
        if {k: list(v) for k, v in recorded.items()} != wanted:
            raise ValueError("dataset features differ from the current robot and camera setup")

    def _finalize(self) -> None:
        if self._encoding is not None:
            encoding, self._encoding = self._encoding, None
            # An unsaved partial episode is dropped; everything saved so far is flushed to disk.
            if self._dataset is not None and self._dataset.has_pending_frames():
                self._dataset.clear_episode_buffer()
            encoding.__exit__(None, None, None)

    # -- loop ----------------------------------------------------------------------------------
    def _run(self) -> None:
        dataset = self._dataset
        assert dataset is not None
        while self._episodes_recorded < self._cfg.num_episodes and not self._stop.is_set():
            self._phase = "recording"
            outcome = self._episode(record=True, duration=self._cfg.episode_time_s)
            if self._stop.is_set() or outcome == "rerecord":
                dataset.clear_episode_buffer()
                if outcome == "rerecord":
                    self._flags["rerecord"] = False
                    continue
                return
            self._phase = "saving"
            dataset.save_episode()
            self._episodes_recorded += 1
            self._events.publish(
                "session_changed", {"state": "running", "episodes_recorded": self._episodes_recorded}
            )
            if self._flags["finish"] or self._episodes_recorded >= self._cfg.num_episodes:
                break
            self._phase = "resetting"
            self._episode(record=False, duration=self._cfg.reset_time_s)
        self._phase = "finished"

    def _episode(self, record: bool, duration: float) -> str:
        """Run one episode (or one reset period). Returns ``done``, ``exit_early``, ``rerecord`` or ``stopped``."""
        teleop_proc, robot_proc, obs_proc = self._processors
        dataset = self._dataset
        assert dataset is not None
        period = 1.0 / self._cfg.fps
        elapsed = 0.0
        if record:
            self._frames = 0
        self._elapsed_s = 0.0
        self._flags["exit_early"] = False
        while elapsed < duration:
            began = time.perf_counter()
            if self._stop.is_set():
                return "stopped"
            if self._flags["rerecord"] and record:
                return "rerecord"
            if self._flags["exit_early"] or (self._flags["finish"] and record):
                self._flags["exit_early"] = False
                return "exit_early"

            obs = self._follower.get_observation()
            obs_processed = obs_proc(obs)
            teleop_action = teleop_proc((self._leader.get_action(), obs))
            sent = self._follower.send_action(clamp_action(robot_proc((teleop_action, obs))))
            self._ticks += 1
            if self._ticks % JOINT_PUBLISH_EVERY == 0:
                self._events.publish("joint_update", joint_update_payload(obs))

            if record and not self._paused.is_set():
                frame = {
                    **build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR),
                    **build_dataset_frame(dataset.features, sent, prefix=ACTION),
                    "task": self._task,
                }
                dataset.add_frame(frame)
                self._frames += 1
            precise_sleep(max(period - (time.perf_counter() - began), 0.0))
            if not self._paused.is_set():
                elapsed += time.perf_counter() - began
                self._elapsed_s = elapsed
        return "done"
