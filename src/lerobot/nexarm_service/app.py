# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""FastAPI application factory for the NexArm engine."""

import contextlib
import logging
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from lerobot.nexarm_service import API_VERSION
from lerobot.nexarm_service.calibration import reset_non_identity_files
from lerobot.nexarm_service.dataset_ops import DatasetOps
from lerobot.nexarm_service.dataset_store import DatasetStore
from lerobot.nexarm_service.events import EventHub
from lerobot.nexarm_service.hardware_factory import Hardware
from lerobot.nexarm_service.operations import OperationRunner
from lerobot.nexarm_service.profiles import ProfileStore, default_home
from lerobot.nexarm_service.record import RecordingWorker
from lerobot.nexarm_service.replay import ReplayWorker
from lerobot.nexarm_service.routes import datasets, sessions, system
from lerobot.nexarm_service.session import (
    LEASE_TIMEOUT_S,
    SessionKind,
    SessionManager,
    Worker,
)
from lerobot.nexarm_service.teleop import TeleopWorker
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
WATCHDOG_PERIOD_S = 2.0


def create_app(
    home: Path | None = None,
    hardware: Any | None = None,
    calibration_root: Path | None = None,
    datasets_root: Path | None = None,
    lease_timeout_s: float = LEASE_TIMEOUT_S,
) -> FastAPI:
    state = _State(home, hardware, calibration_root, datasets_root, lease_timeout_s)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state.reset_calibration()
        stop_watchdog = threading.Event()
        watchdog = threading.Thread(
            target=_watchdog, args=(state.sessions, stop_watchdog), name="nexarm-watchdog", daemon=True
        )
        watchdog.start()
        try:
            yield
        finally:
            stop_watchdog.set()
            watchdog.join(timeout=WATCHDOG_PERIOD_S + 1)
            # Engine shutdown must never leave the arm energised and unattended.
            state.sessions.stop_current("engine_shutdown")
            state.dataset_store.close()

    app = FastAPI(
        title="NexArm engine",
        version=API_VERSION,
        description="Local engine for the NexArm desktop app.",
        lifespan=lifespan,
    )
    for name in ("events", "profiles", "hardware", "sessions", "dataset_store", "dataset_ops", "operations"):
        setattr(app.state, name, getattr(state, name))
    app.state.reset_calibration = state.reset_calibration
    for module in (system, sessions, datasets):
        app.include_router(module.router, prefix=API_PREFIX)
    return app


class _State:
    def __init__(
        self,
        home: Path | None,
        hardware: Any | None,
        calibration_root: Path | None,
        datasets_root: Path | None,
        lease_timeout_s: float,
    ) -> None:
        home = home or default_home()
        self.events = EventHub()
        self.profiles = ProfileStore(home)
        self.hardware = hardware or Hardware()
        self._calibration_root = calibration_root
        self.datasets_root = datasets_root or HF_LEROBOT_HOME
        self.dataset_store = DatasetStore(self.datasets_root, home / "curation")
        self.operations = OperationRunner(self.events)
        factories = {
            SessionKind.teleoperation: self._teleop,
            SessionKind.recording: self._recording,
            SessionKind.replay: self._replay,
        }
        self.sessions = SessionManager(self.events, factories, lease_timeout_s=lease_timeout_s)
        self.dataset_ops = DatasetOps(self.dataset_store, self._recording_into)

    def reset_calibration(self) -> list[Path]:
        """Keep engine calibration files identity; zero offsets belong in the servos."""
        if self._calibration_root is None:
            return reset_non_identity_files()
        return reset_non_identity_files(self._calibration_root)

    def _recording_into(self, repo_id: str) -> bool:
        current = self.sessions.current()
        return (
            current is not None
            and current.kind == SessionKind.recording
            and current.params.get("repo_id") == repo_id
        )

    def _profile(self, name: str) -> Any:
        profile = self.profiles.get(name)
        if profile is None:
            raise ValueError(f"no profile named {name!r}")
        return profile

    def _teleop(self, profile_name: str, params: dict[str, Any]) -> Worker:
        return TeleopWorker(
            self._profile(profile_name), self.hardware, self.events, fps=int(params.get("fps", 30))
        )

    def _recording(self, profile_name: str, params: dict[str, Any]) -> Worker:
        return RecordingWorker(
            self._profile(profile_name), self.hardware, self.events, self.datasets_root, params
        )

    def _replay(self, profile_name: str, params: dict[str, Any]) -> Worker:
        return ReplayWorker(
            self._profile(profile_name), self.hardware, self.events, self.dataset_store, params
        )


def _watchdog(sessions: SessionManager, stop: threading.Event) -> None:
    while not stop.wait(WATCHDOG_PERIOD_S):
        try:
            sessions.tick()
        except Exception:
            logger.exception("watchdog tick failed")


def openapi_schema() -> dict[str, Any]:
    """The contract snapshot in docs/service/openapi.json; independent of the lerobot release."""
    return create_app(Path("unused")).openapi()
