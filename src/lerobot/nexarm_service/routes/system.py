# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""system: health, robot profiles, ports, cameras, events WebSocket."""

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from lerobot import __version__
from lerobot.nexarm_service import API_VERSION, CAPABILITIES
from lerobot.nexarm_service.hardware import PortInfo, list_cameras, list_ports
from lerobot.nexarm_service.profiles import RobotProfile, valid_name

router = APIRouter()


class Health(BaseModel):
    status: str
    version: str
    api_version: str
    capabilities: list[str]
    pid: int


@router.get("/health", tags=["system"])
def health() -> Health:
    return Health(
        status="ok",
        version=__version__,
        api_version=API_VERSION,
        capabilities=list(CAPABILITIES),
        pid=os.getpid(),
    )


@router.get("/profiles", tags=["system"])
def list_profiles(request: Request) -> list[RobotProfile]:
    return request.app.state.profiles.list()


@router.put("/profiles/{name}", tags=["system"])
def put_profile(name: str, profile: RobotProfile, request: Request) -> RobotProfile:
    if profile.name != name:
        raise HTTPException(422, "profile name in the body must match the URL")
    request.app.state.profiles.put(profile)
    return profile


@router.delete("/profiles/{name}", status_code=204, tags=["system"])
def delete_profile(name: str, request: Request) -> None:
    if not valid_name(name) or not request.app.state.profiles.delete(name):
        raise HTTPException(404, f"no profile named {name!r}")


@router.get("/ports", tags=["system"])
def ports() -> list[PortInfo]:
    return list_ports()


@router.get("/cameras", tags=["system"])
def cameras() -> list[dict[str, Any]]:
    return list_cameras()


@router.websocket("/ws/events")
async def events(ws: WebSocket) -> None:
    await ws.accept()
    hub = ws.app.state.events
    queue = hub.subscribe()
    try:
        await ws.send_json({"type": "hello", "data": {"api_version": API_VERSION}})
        while True:
            await ws.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(queue)
