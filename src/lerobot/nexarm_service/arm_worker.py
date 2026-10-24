# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Shared lifecycle for every session that drives the follower arm.

Safety contract, identical for all kinds: the rest pose is captured from validated reads before the
control loop starts, every command is clamped to the soft limits by the subclass, and stopping ramps the
follower back to rest before torque is cut.
"""

import logging
import threading
from typing import Any

from lerobot.nexarm_service.events import EventHub
from lerobot.nexarm_service.profiles import RobotProfile
from lerobot.nexarm_service.safety import capture_rest, ramp_to
from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES, raw_to_radians

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
        self.rest = capture_rest(self.follower.bus.read_positions)

    def release(self) -> None:
        """Best effort; always ends with torque off."""
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


class ArmWorker:
    """Base for session workers. Subclasses implement ``_run`` and may override the hooks."""

    needs_leader = True
    with_cameras = False
    thread_name = "nexarm-session"

    def __init__(self, profile: RobotProfile, hardware: Any, events: EventHub) -> None:
        self.error: str | None = None
        self._profile = profile
        self._hardware = hardware
        self._events = events
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._leader: Any = None
        self._follower: Any = None
        self._guard: FollowerGuard | None = None
        self._ticks = 0

    # -- hooks ---------------------------------------------------------------------------------
    def _prepare(self) -> None:
        """Runs before any hardware is touched (open datasets, load models, validate params)."""

    def _ready(self) -> None:
        """Runs after the arm is connected and the rest pose captured, before the loop starts."""

    def _run(self) -> None:
        raise NotImplementedError

    def _finalize(self) -> None:
        """Runs after the arm is released (flush datasets, write reports). Must not touch the arm."""

    def control(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise ValueError(f"this session does not support the {action!r} control")

    def extra_status(self) -> dict[str, Any]:
        return {}

    # -- Worker protocol -----------------------------------------------------------------------
    def start(self) -> None:
        self._prepare()
        try:
            self._follower = self._hardware.make_follower(self._profile, with_cameras=self.with_cameras)
            if self.needs_leader:
                self._leader = self._hardware.make_leader(self._profile)
            self._follower.connect(calibrate=False)
            if self._leader is not None:
                self._leader.connect(calibrate=False)
            self._guard = FollowerGuard(self._follower)
            self._guard.engage()
            self._ready()
        except Exception:
            self._shutdown_hardware()
            self._finalize_quietly()
            raise
        self._thread = threading.Thread(target=self._guarded_run, name=self.thread_name, daemon=True)
        self._thread.start()

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {"ticks": self._ticks, **self.extra_status()}

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60.0)
        self._shutdown_hardware()
        self._finalize_quietly()

    # -- internals -----------------------------------------------------------------------------
    def _guarded_run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            logger.exception("%s loop failed", self.thread_name)
            self.error = f"{self.thread_name} loop failed: {exc}"

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

    def _finalize_quietly(self) -> None:
        try:
            self._finalize()
        except Exception as exc:
            logger.exception("finalize failed")
            self.error = self.error or f"finalize failed: {exc}"
