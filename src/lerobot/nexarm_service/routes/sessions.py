# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""sessions and calibration routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from lerobot.nexarm_service.calibration import CalibrationReport, build_report
from lerobot.nexarm_service.session import (
    SessionBusyError,
    SessionInfo,
    SessionKind,
    SessionNotFoundError,
    UnsupportedKindError,
)

router = APIRouter()


class StartSession(BaseModel):
    kind: SessionKind
    profile: str
    params: dict[str, Any] = {}


class CalibrationCheck(BaseModel):
    profile: str


@router.post("/sessions", status_code=201, tags=["sessions"])
def start_session(body: StartSession, request: Request) -> SessionInfo:
    state = request.app.state
    if state.profiles.get(body.profile) is None:
        raise HTTPException(404, f"no profile named {body.profile!r}")
    try:
        state.reset_calibration()
        return state.sessions.start(body.kind, body.profile, body.params)
    except SessionBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (UnsupportedKindError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except KeyError as exc:  # a referenced dataset or episode does not exist
        raise HTTPException(404, f"not found: {exc.args[0]}") from exc
    except (RuntimeError, OSError, TimeoutError) as exc:
        raise HTTPException(502, f"could not start the session: {exc}") from exc


@router.get("/sessions/current", tags=["sessions"])
def current_session(request: Request) -> SessionInfo | None:
    return request.app.state.sessions.current()


@router.post("/sessions/{session_id}/heartbeat", tags=["sessions"])
def heartbeat(session_id: str, request: Request) -> SessionInfo:
    try:
        return request.app.state.sessions.heartbeat(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(404, f"no live session {session_id!r}") from exc


class SessionControl(BaseModel):
    action: str
    payload: dict[str, Any] = {}


@router.post("/sessions/{session_id}/control", tags=["sessions"])
def control_session(session_id: str, body: SessionControl, request: Request) -> dict[str, Any]:
    """Live controls: recording (exit_early, rerecord, pause, resume, set_task, finish); coaching later."""
    try:
        return request.app.state.sessions.control(session_id, body.action, body.payload)
    except SessionNotFoundError as exc:
        raise HTTPException(404, f"no live session {session_id!r}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.delete("/sessions/{session_id}", tags=["sessions"])
def stop_session(session_id: str, request: Request) -> SessionInfo:
    try:
        return request.app.state.sessions.stop(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(404, f"no live session {session_id!r}") from exc


@router.post("/calibration/check", tags=["calibration"])
def calibration_check(body: CalibrationCheck, request: Request) -> CalibrationReport:
    """Write identity calibration files and report leader/follower deltas at the vendor Home pose.

    Both arms must be placed at Home by hand first; this only reads positions and never energises
    the follower.
    """
    state = request.app.state
    profile = state.profiles.get(body.profile)
    if profile is None:
        raise HTTPException(404, f"no profile named {body.profile!r}")
    if profile.leader_port is None:
        raise HTTPException(422, f"profile {body.profile!r} has no leader_port")
    if state.sessions.busy:
        raise HTTPException(409, "stop the running session before checking calibration")
    reset = state.reset_calibration()
    leader = state.hardware.make_bus(profile.leader_port, profile.baudrate)
    follower = state.hardware.make_bus(profile.follower_port, profile.baudrate)
    try:
        leader.connect()
        follower.connect()
        follower.enter_lerobot_mode()
        try:
            leader_raw = leader.read_positions()
            follower_raw = follower.read_positions()
        finally:
            follower.exit_lerobot_mode()
    except (OSError, TimeoutError, ConnectionError) as exc:
        raise HTTPException(502, f"could not read the arms: {exc}") from exc
    finally:
        leader.disconnect()
        follower.disconnect()
    return build_report(leader_raw, follower_raw, reset)
