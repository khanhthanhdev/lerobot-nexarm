# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""``WS /api/v1/ws/remote``: the station end of remote teleoperation (see remote_teleop.py)."""

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from nexarm_service import API_VERSION
from nexarm_service.remote_teleop import (
    DEFAULT_FPS,
    JOINT_KEYS,
    HostingWorker,
    ProtocolError,
    parse_action,
)
from nexarm_service.session import SessionKind, SessionState

router = APIRouter()

STATE_PERIOD_S = 1 / 15


def _hosting_worker(ws: WebSocket) -> HostingWorker | None:
    sessions = ws.app.state.sessions
    info = sessions.current()
    if info is None or info.kind != SessionKind.hosting or info.state != SessionState.running:
        return None
    worker = sessions.current_worker()
    return worker if isinstance(worker, HostingWorker) else None


async def _send_states(ws: WebSocket, host: HostingWorker) -> None:
    while True:
        await asyncio.sleep(STATE_PERIOD_S)
        await ws.send_json({"type": "state", **host.snapshot()})


@router.websocket("/ws/remote")
async def remote_operator(ws: WebSocket) -> None:
    host = _hosting_worker(ws)
    await ws.accept()
    if host is None:
        await ws.send_json({"type": "refused", "reason": "this engine is not hosting a station"})
        await ws.close(code=4409)
        return
    try:
        hello = await asyncio.wait_for(ws.receive_json(), timeout=10)
        if (
            hello.get("type") != "hello"
            or hello.get("api_version") != API_VERSION
            or hello.get("joints") != JOINT_KEYS
        ):
            await ws.send_json(
                {"type": "refused", "reason": "incompatible operator (api version or joint set)"}
            )
            await ws.close(code=4400)
            return
    except (TimeoutError, WebSocketDisconnect, ValueError):
        with contextlib.suppress(Exception):
            await ws.close(code=4400)
        return
    reason = host.try_attach(str(hello.get("operator", "operator")))
    if reason is not None:
        await ws.send_json({"type": "refused", "reason": reason})
        await ws.close(code=4423)
        return
    sender: asyncio.Task | None = None
    try:
        await ws.send_json(
            {
                "type": "ready",
                "api_version": API_VERSION,
                "fps": host.fps or DEFAULT_FPS,
                "rate_cap_ticks_s": host.rate_cap,
                "rest": host.rest_action(),
            }
        )
        sender = asyncio.create_task(_send_states(ws, host))
        while True:
            message = await ws.receive_json()
            kind = message.get("type")
            if kind == "action":
                try:
                    seq, ticks = parse_action(message)
                except ProtocolError as exc:
                    await ws.send_json({"type": "refused", "reason": str(exc)})
                    await ws.close(code=4400)
                    return
                host.on_action(seq, ticks)
            elif kind == "bye":
                return
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if sender is not None:
            sender.cancel()
        host.detach("operator disconnected")
