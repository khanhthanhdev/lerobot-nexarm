# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Teleoperation worker: the leader arm drives the follower at a fixed rate.

Safety contract: the rest pose is captured from validated reads before torque-on, every action is
clamped to the soft limits, and stopping ramps the follower back to rest before cutting torque.
"""

import logging
import threading
import time
from typing import Any

from lerobot.motors.nexarm.nexarm import JOINT_NAMES, raw_to_radians
from lerobot.nexarm_service.events import EventHub
from lerobot.nexarm_service.profiles import RobotProfile
from lerobot.nexarm_service.safety import capture_rest, clamp_action, ramp_to

logger = logging.getLogger(__name__)

JOINT_PUBLISH_EVERY = 2  # publish one joint_update per N control ticks


def joint_update_payload(positions: dict[str, float]) -> dict[str, Any]:
    ticks = {name: positions[f"{name}.pos"] for name in JOINT_NAMES}
    # Gripper ticks are not an angle, so radians cover the five arm joints only.
    radians = {name: raw_to_radians(ticks[name]) for name in JOINT_NAMES[:5]}
    return {"ticks": ticks, "radians": radians}


class FollowerGuard:
    """Owns a connected follower: rest capture on entry, rest return plus torque-off on exit."""

    def __init__(self, follower: Any) -> None:
        self.follower = follower
        self.rest: list[int] | None = None
        self.error: str | None = None

    def engage(self) -> None:
        bus = self.follower.bus
        self.rest = capture_rest(bus.read_positions)

    def release(self) -> None:
        """Best effort and always ends with torque off and the bridge closed."""
        bus = self.follower.bus
        try:
            if self.rest is not None:
                ramp_to(bus.read_positions, bus.write_positions, self.rest)
        except Exception as exc:
            self.error = f"rest return failed: {exc}"
            logger.error("rest return failed: %s", exc)
        try:
            bus.set_torque(False)
        except Exception as exc:
            self.error = self.error or f"torque off failed: {exc}"
            logger.error("torque off failed: %s", exc)


class TeleopWorker:
    def __init__(
        self,
        profile: RobotProfile,
        hardware: Any,
        events: EventHub,
        fps: int = 30,
    ) -> None:
        self.error: str | None = None
        self._profile = profile
        self._hardware = hardware
        self._events = events
        self._period = 1.0 / fps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._leader: Any = None
        self._follower: Any = None
        self._guard: FollowerGuard | None = None
        self._ticks = 0
        self._last_loop_ms = 0.0

    def start(self) -> None:
        self._leader = self._hardware.make_leader(self._profile)
        self._follower = self._hardware.make_follower(self._profile)
        try:
            self._follower.connect(calibrate=False)
            self._leader.connect(calibrate=False)
            self._guard = FollowerGuard(self._follower)
            self._guard.engage()
        except Exception:
            self._shutdown_hardware()
            raise
        self._thread = threading.Thread(target=self._run, name="nexarm-teleop", daemon=True)
        self._thread.start()

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {"ticks": self._ticks, "loop_ms": round(self._last_loop_ms, 2)}

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._shutdown_hardware()

    def _run(self) -> None:
        try:
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
        except Exception as exc:
            logger.exception("teleop loop failed")
            self.error = f"teleop loop failed: {exc}"

    def _shutdown_hardware(self) -> None:
        if self._guard is not None:
            self._guard.release()
            self.error = self.error or self._guard.error
        for device in (self._follower, self._leader):
            if device is not None and device.is_connected:
                try:
                    device.disconnect()
                except Exception as exc:
                    logger.error("disconnect failed: %s", exc)
