# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Coaching (DAgger) session: a policy drives the follower, the operator takes over on command, and the
corrections are recorded into a dataset.

It runs the library's ``dagger`` rollout strategy; the phase machine and recording are the library's, and the
controls come from the desktop instead of keys or a pedal. The control names (takeover, handback, hold,
resume, reset, drop-last) follow the vocabulary of MakerMods Lab's ``makermodslab/dagger_protocol.py``
(Apache-2.0, Copyright 2026 MakerMods); no code is taken from it and the behaviour is simpler:

* ``takeover`` is one command that goes through "paused" to "correcting" with no separate settle step;
* ``reset`` holds the follower at its rest pose and brings the leader to match; it does not save or discard
  a correction;
* ``drop_last`` discards the correction that is still being recorded. MakerMods Lab keeps a finished
  correction "held" until the next attempt begins so it can still be dropped; here a saved correction is
  already in the dataset (the library saves it when the phase ends), so it is kept.

Leader arm: while the policy drives, the leader follows it (torque on); during a correction its torque is off
so the operator can drag it. The leader's start pose is captured and it is returned there, then released,
when the session ends, so it does not drop when its torque is cut.
"""

import contextlib
import logging
import threading
import time
from typing import Any

from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.nexarm_service.rollout import RolloutWorker
from lerobot.nexarm_service.safety import capture_rest, ramp_to

logger = logging.getLogger(__name__)

HANDOVER_TIMEOUT_S = 10.0

# control -> the (event, phase it must reach) steps that carry it out, by the current phase.
_PLANS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "takeover": {
        "autonomous": [("pause_resume", "paused"), ("correction", "correcting")],
        "paused": [("correction", "correcting")],
    },
    "handback": {
        "correcting": [("correction", "paused"), ("pause_resume", "autonomous")],
        "paused": [("pause_resume", "autonomous")],
    },
    "hold": {"autonomous": [("pause_resume", "paused")]},
    "resume": {"paused": [("pause_resume", "autonomous")]},
    "drop_last": {"correcting": [("correction", "paused")]},
}


class CoachingWorker(RolloutWorker):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("coaching", *args, **kwargs)
        self._hold_action: dict[str, float] | None = None
        self._leader_rest: list[int] | None = None
        self._leader_torque = False
        self._corrections = 0
        self._discarded = 0
        self._discard_next = False
        self._chain = threading.Lock()
        self._monitor: threading.Thread | None = None
        self._last_published: tuple[Any, ...] = ()

    # -- hooks ---------------------------------------------------------------------------------
    def _install_safety(self, robot: Any) -> None:
        super()._install_safety(robot)
        clamped_send = robot.send_action

        def holding_send(action: dict[str, float]) -> Any:
            # After a reset the follower is held at rest whatever the strategy sends, until a handover clears it.
            return clamped_send(self._hold_action if self._hold_action is not None else action)

        robot.send_action = holding_send

    def _after_context(self) -> None:
        teleop = self._ctx.hardware.teleop
        if teleop is None:
            raise ValueError("coaching needs the leader arm: set leader_port on the robot profile")
        self._leader_rest = capture_rest(teleop.bus.read_positions)
        enable, disable = teleop.enable_torque, teleop.disable_torque

        def tracked_enable() -> None:
            enable()
            self._leader_torque = True

        def tracked_disable() -> None:
            disable()
            self._leader_torque = False

        teleop.enable_torque, teleop.disable_torque = tracked_enable, tracked_disable

    def _after_setup(self) -> None:
        dataset = self._ctx.data.dataset
        original = dataset.save_episode

        def save_episode(*args: Any, **kwargs: Any) -> Any:
            if self._discard_next:
                self._discard_next = False
                self._discarded += 1
                dataset.clear_episode_buffer()
                return None
            result = original(*args, **kwargs)
            self._corrections += 1
            return result

        dataset.save_episode = save_episode
        self._monitor = threading.Thread(target=self._watch_phase, name="nexarm-coaching-state", daemon=True)
        self._monitor.start()

    def _before_release(self) -> None:
        self._hold_action = None
        monitor, self._monitor = self._monitor, None
        if monitor is not None:
            monitor.join(timeout=2.0)
        self._release_leader()

    # -- state ---------------------------------------------------------------------------------
    @property
    def _dagger_events(self) -> Any:
        return getattr(self._strategy, "_events", None)

    def phase(self) -> str:
        events = self._dagger_events
        return events.phase.value if events is not None else "autonomous"

    def status(self) -> dict[str, Any]:
        return {
            **super().status(),
            "phase": self.phase(),
            "intervening": self.phase() == "correcting",
            "corrections": self._corrections,
            "discarded": self._discarded,
            "hold": self._hold_action is not None,
        }

    def _watch_phase(self) -> None:
        while not self._shutdown.wait(0.1):
            snapshot = (self.phase(), self._corrections, self._discarded)
            if snapshot != self._last_published:
                self._last_published = snapshot
                self._events.publish(
                    "coaching_state",
                    {
                        "phase": snapshot[0],
                        "intervening": snapshot[0] == "correcting",
                        "corrections": snapshot[1],
                    },
                )

    # -- controls ------------------------------------------------------------------------------
    def control(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        events = self._dagger_events
        if events is None:
            raise ValueError("the coaching strategy is not running")
        if action == "finish":
            events.stop_recording.set()
            return self.status()
        if action == "reset":
            plan: list[Any] = []
            phase = self.phase()
            if phase == "correcting":
                raise ValueError("hand the arm back before resetting")
            if phase == "autonomous":
                plan.append(("pause_resume", "paused"))
            self._run_chain(plan, then=self._reset_to_rest)
            return self.status()
        if action not in _PLANS:
            raise ValueError(f"this session does not support the {action!r} control")
        phase = self.phase()
        chain = _PLANS[action].get(phase)
        if chain is None:
            raise ValueError(f"{action} is not possible while the policy is {phase}")
        if action == "drop_last":
            self._discard_next = True
            # The library counts the discarded take toward its target, so give that slot back.
            self._strategy.config.num_episodes += 1
        if action in ("takeover", "handback", "resume"):
            self._hold_action = None
        self._run_chain(chain)
        return self.status()

    def _run_chain(self, steps: list[tuple[str, str]], then: Any = None) -> None:
        """Carry out phase transitions in order on a helper thread (a handover takes seconds)."""
        if not self._chain.acquire(blocking=False):
            raise ValueError("the previous handover is still in progress")

        def run() -> None:
            try:
                for event, target in steps:
                    self._dagger_events.request_transition(event)
                    self._wait_for_phase(target)
                if then is not None:
                    then()
            except Exception as exc:
                logger.exception("coaching control failed")
                self.error = self.error or f"coaching control failed: {exc}"
            finally:
                self._chain.release()

        threading.Thread(target=run, name="nexarm-coaching-handover", daemon=True).start()

    def _wait_for_phase(self, target: str) -> None:
        deadline = time.monotonic() + HANDOVER_TIMEOUT_S
        while self.phase() != target:
            if self._shutdown.is_set() or time.monotonic() > deadline:
                raise TimeoutError(f"the policy did not reach {target!r}")
            time.sleep(0.02)

    def _reset_to_rest(self) -> None:
        """Hold the follower at its rest pose and bring the leader to the matching pose."""
        from lerobot.common.control_utils import teleop_smooth_move_to

        rest = self._guard.rest if self._guard is not None else None
        if rest is None:
            raise RuntimeError("no rest pose was captured")
        rest_action = {f"{name}.pos": float(rest[i]) for i, name in enumerate(JOINT_NAMES)}
        self._hold_action = rest_action
        teleop_smooth_move_to(self._ctx.hardware.teleop, rest_action)

    # -- leader --------------------------------------------------------------------------------
    def _release_leader(self) -> None:
        """Return a torque-enabled leader to where it started, then free it."""
        teleop = self._ctx.hardware.teleop if self._ctx is not None else None
        if teleop is None or not self._leader_torque:
            return
        bus = teleop.bus
        try:
            if self._leader_rest is not None:
                ramp_to(bus.read_positions, bus.write_positions, self._leader_rest)
        except Exception as exc:
            logger.error("leader rest return failed: %s", exc)
            self.error = self.error or f"leader rest return failed: {exc}"
        with contextlib.suppress(Exception):
            bus.set_torque(False)
            self._leader_torque = False
