# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Session manager: one live hardware session at a time, kept alive by a client lease.

The desktop sends a heartbeat every ``HEARTBEAT_INTERVAL_S``; after ``LEASE_TIMEOUT_S`` of silence the
watchdog safety-stops the session (rest return, then torque off) so a crashed client cannot leave the
arm energised.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel

from lerobot.nexarm_service.events import EventHub

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 20.0
LEASE_TIMEOUT_S = 60.0


class SessionKind(StrEnum):
    teleoperation = "teleoperation"
    recording = "recording"
    replay = "replay"
    rollout = "rollout"
    eval = "eval"
    coaching = "coaching"
    hosting = "hosting"
    remote_teleoperation = "remote_teleoperation"


class SessionState(StrEnum):
    starting = "starting"
    running = "running"
    stopping = "stopping"
    stopped = "stopped"
    failed = "failed"


class SessionInfo(BaseModel):
    id: str
    kind: SessionKind
    profile: str
    state: SessionState
    started_at: float
    params: dict[str, Any] = {}
    stop_reason: str | None = None
    error: str | None = None
    status: dict[str, Any] = {}


class Worker(Protocol):
    """What a session kind implements. ``stop`` must return the arm to rest before releasing torque."""

    error: str | None

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def alive(self) -> bool: ...
    def status(self) -> dict[str, Any]: ...
    def control(self, action: str, payload: dict[str, Any]) -> dict[str, Any]: ...


WorkerFactory = Callable[[str, dict[str, Any]], Worker]  # (profile name, params) -> worker


class SessionBusyError(RuntimeError):
    pass


class SessionNotFoundError(KeyError):
    pass


class UnsupportedKindError(ValueError):
    pass


class _Live:
    def __init__(self, info: SessionInfo, worker: Worker, now: float) -> None:
        self.info = info
        self.worker = worker
        self.last_heartbeat = now
        # Held for the whole of worker.start(); stop() waits on it so the two never overlap.
        self.start_lock = threading.Lock()


class SessionManager:
    def __init__(
        self,
        events: EventHub,
        factories: dict[SessionKind, WorkerFactory],
        lease_timeout_s: float = LEASE_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._events = events
        self._factories = factories
        self._lease_timeout_s = lease_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._live: _Live | None = None
        self._last: SessionInfo | None = None

    # -- queries -----------------------------------------------------------------------------
    def current(self) -> SessionInfo | None:
        with self._lock:
            if self._live is None:
                return None
            info = self._live.info.model_copy()
            if info.state == SessionState.running:
                info.status = self._live.worker.status()
            return info

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._live is not None

    def current_worker(self) -> Worker | None:
        """The live session's worker (for the WebSocket handlers that talk to it directly)."""
        with self._lock:
            return self._live.worker if self._live is not None else None

    def last(self) -> SessionInfo | None:
        return self._last

    # -- transitions -------------------------------------------------------------------------
    def start(self, kind: SessionKind, profile: str, params: dict[str, Any] | None = None) -> SessionInfo:
        if kind not in self._factories:
            raise UnsupportedKindError(f"session kind {kind.value!r} is not available in this engine")
        worker = self._factories[kind](profile, params or {})
        info = SessionInfo(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            profile=profile,
            state=SessionState.starting,
            started_at=time.time(),
            params=params or {},
        )
        with self._lock:
            if self._live is not None:
                raise SessionBusyError(
                    f"a {self._live.info.kind.value} session is already {self._live.info.state.value}"
                )
            live = self._live = _Live(info, worker, self._clock())
        self._publish(info)
        with live.start_lock:
            try:
                worker.start()
            except Exception as exc:
                logger.exception("session %s failed to start", info.id)
                self._finish(info.id, SessionState.failed, reason="start_failed", error=str(exc))
                raise
            with self._lock:
                # A stop requested while starting has already marked the session stopping; leave it so.
                if live.info.state == SessionState.starting:
                    live.info.state = SessionState.running
                    live.last_heartbeat = self._clock()
                info = live.info.model_copy()
        self._publish(info)
        return info

    def heartbeat(self, session_id: str) -> SessionInfo:
        with self._lock:
            live = self._require(session_id)
            live.last_heartbeat = self._clock()
            return live.info.model_copy()

    def stop(self, session_id: str, reason: str = "user") -> SessionInfo:
        with self._lock:
            live = self._require(session_id)
            if live.info.state == SessionState.stopping:
                return live.info.model_copy()
            live.info.state = SessionState.stopping
            info = live.info.model_copy()
        self._publish(info)
        # If the worker is still starting (connecting hardware, loading a policy), wait for start() to
        # return before stopping it: stop() must never overlap start(), or the arm is left energised.
        with live.start_lock:
            with self._lock:
                if self._live is not live:  # start() failed and already released everything
                    return self._last or info
            return self._stop_worker(session_id, live.worker, reason)

    def control(self, session_id: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Forward a live control (pause, takeover, ...) to the running worker; ValueError if unsupported."""
        with self._lock:
            live = self._require(session_id)
            if live.info.state != SessionState.running:
                raise ValueError(f"session is {live.info.state.value}, not running")
            worker = live.worker
        return worker.control(action, payload)

    def stop_current(self, reason: str) -> None:
        info = self.current()
        if info is not None:
            self.stop(info.id, reason)

    def tick(self) -> None:
        """Watchdog step: expire silent leases and reap workers that died on their own."""
        with self._lock:
            live = self._live
            if live is None or live.info.state != SessionState.running:
                return
            silent = self._clock() - live.last_heartbeat > self._lease_timeout_s
            dead = not live.worker.alive()
            session_id = live.info.id
        if silent:
            logger.warning("session %s lease expired; safety stop", session_id)
            self.stop(session_id, reason="lease_expired")
        elif dead:
            self._stop_worker(session_id, live.worker, "worker_exited")

    # -- internals ---------------------------------------------------------------------------
    def _require(self, session_id: str) -> _Live:
        if self._live is None or self._live.info.id != session_id:
            raise SessionNotFoundError(session_id)
        return self._live

    def _stop_worker(self, session_id: str, worker: Worker, reason: str) -> SessionInfo:
        error = None
        try:
            worker.stop()
        except Exception as exc:
            logger.exception("session %s did not stop cleanly", session_id)
            error = str(exc)
        error = error or worker.error
        return self._finish(
            session_id, SessionState.failed if error else SessionState.stopped, reason=reason, error=error
        )

    def _finish(self, session_id: str, state: SessionState, reason: str, error: str | None) -> SessionInfo:
        with self._lock:
            live = self._live
            if live is None or live.info.id != session_id:
                return self._last or SessionInfo(
                    id=session_id,
                    kind=SessionKind.teleoperation,
                    profile="",
                    state=state,
                    started_at=0,
                )
            live.info.state = state
            live.info.stop_reason = reason
            live.info.error = error
            info = live.info.model_copy()
            self._last = info
            self._live = None
        self._publish(info)
        # The desktop reclaims its serial/TCP links on this: the arm is released and the slot is free.
        self._events.publish("session_changed", {"id": info.id, "state": "idle"})
        return info

    def _publish(self, info: SessionInfo) -> None:
        self._events.publish(
            "session_changed",
            {
                "id": info.id,
                "kind": info.kind.value,
                "state": info.state.value,
                "stop_reason": info.stop_reason,
                "error": info.error,
            },
        )
