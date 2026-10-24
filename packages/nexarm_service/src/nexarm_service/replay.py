# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Replay session: play one recorded episode's actions on the follower.

Follows ``lerobot.scripts.lerobot_replay.replay`` with two safety additions: the arm is first ramped to
the episode's opening pose (never jumped), and every action is clamped to the soft limits.
"""

import time
from typing import Any

from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES
from nexarm_service.arm_worker import JOINT_PUBLISH_EVERY, ArmWorker, joint_update_payload
from nexarm_service.dataset_store import DatasetStore
from nexarm_service.events import EventHub
from nexarm_service.profiles import RobotProfile
from nexarm_service.safety import clamp_action, ramp_to


class ReplayWorker(ArmWorker):
    needs_leader = False
    thread_name = "nexarm-replay"

    def __init__(
        self,
        profile: RobotProfile,
        hardware: Any,
        events: EventHub,
        store: DatasetStore,
        params: dict[str, Any],
    ) -> None:
        super().__init__(profile, hardware, events)
        if "repo_id" not in params or "episode" not in params:
            raise ValueError("replay needs params.repo_id and params.episode")
        self._repo_id = params["repo_id"]
        self._episode = int(params["episode"])
        self._store = store
        self._actions: list[dict[str, float]] = []
        self._fps = int(store.info(self._repo_id)["fps"])

    def _prepare(self) -> None:
        data = self._store.joints(self._repo_id, self._episode)
        index = {name: i for i, name in enumerate(data.names)}
        missing = [f"{n}.pos" for n in JOINT_NAMES if f"{n}.pos" not in index]
        if missing:
            raise ValueError(f"dataset actions lack NexArm joints: {missing}")
        self._actions = [{k: row[i] for k, i in index.items()} for row in data.action]
        if not self._actions:
            raise ValueError(f"episode {self._episode} has no frames")

    def _ready(self) -> None:
        first = clamp_action(self._actions[0])
        bus = self._follower.bus
        ramp_to(bus.read_positions, bus.write_positions, [round(first[f"{n}.pos"]) for n in JOINT_NAMES])

    def extra_status(self) -> dict[str, Any]:
        return {
            "repo_id": self._repo_id,
            "episode": self._episode,
            "frame": self._ticks,
            "frames": len(self._actions),
        }

    def _run(self) -> None:
        period = 1.0 / self._fps
        for action in self._actions:
            began = time.perf_counter()
            if self._stop.is_set():
                return
            observation = self._follower.get_observation()
            self._follower.send_action(clamp_action(action))
            self._ticks += 1
            if self._ticks % JOINT_PUBLISH_EVERY == 0:
                self._events.publish("joint_update", joint_update_payload(observation))
            self._stop.wait(max(0.0, period - (time.perf_counter() - began)))
