# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Teleoperation worker: the leader arm drives the follower at a fixed rate."""

import time
from typing import Any

from lerobot.nexarm_service.arm_worker import JOINT_PUBLISH_EVERY, ArmWorker, joint_update_payload
from lerobot.nexarm_service.events import EventHub
from lerobot.nexarm_service.profiles import RobotProfile
from lerobot.nexarm_service.safety import clamp_action


class TeleopWorker(ArmWorker):
    thread_name = "nexarm-teleop"

    def __init__(self, profile: RobotProfile, hardware: Any, events: EventHub, fps: int = 30) -> None:
        super().__init__(profile, hardware, events)
        self._period = 1.0 / fps
        self._last_loop_ms = 0.0

    def extra_status(self) -> dict[str, Any]:
        return {"loop_ms": round(self._last_loop_ms, 2)}

    def _run(self) -> None:
        while not self._stop.is_set():
            began = time.perf_counter()
            action = clamp_action(self._leader.get_action())
            observation = self._follower.get_observation()
            self._follower.send_action(action)
            self._ticks += 1
            if self._ticks % JOINT_PUBLISH_EVERY == 0:
                self._events.publish("joint_update", joint_update_payload(observation))
            self._last_loop_ms = (time.perf_counter() - began) * 1e3
            self._stop.wait(max(0.0, self._period - (time.perf_counter() - began)))
