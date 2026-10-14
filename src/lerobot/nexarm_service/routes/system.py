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

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from lerobot import __version__
from lerobot.nexarm_service import API_VERSION, CAPABILITIES
from lerobot.nexarm_service.hardware import PortInfo, list_cameras, list_ports
from lerobot.nexarm_service.profiles import RobotProfile, valid_name
from lerobot.nexarm_service.servers import RemoteServer

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


class ServerTest(BaseModel):
    ssh_ok: bool
    engine: Health | None = None
    detail: str = ""


@router.post("/servers/test", tags=["system"])
def test_server(server: RemoteServer) -> ServerTest:
    """Open the SSH tunnel to a GPU server and ask its engine for its version; starts nothing remotely."""
    from lerobot.nexarm_service import servers
    from lerobot.nexarm_service.remote import RemoteClient, RemoteError

    try:
        with servers.SshTunnel(server) as url:
            client = RemoteClient(url)
            try:
                info = client.health()
            except (RemoteError, httpx.HTTPError) as exc:
                return ServerTest(
                    ssh_ok=True,
                    detail=f"SSH works but no engine answers yet ({exc}). It is started automatically on the first remote job, or run ~/.nexarm/start-engine.sh on the server.",
                )
            finally:
                client.close()
    except (ConnectionError, OSError) as exc:
        return ServerTest(ssh_ok=False, detail=str(exc))
    return ServerTest(ssh_ok=True, engine=Health(**info))


@router.post("/shutdown", status_code=202, tags=["system"])
def shutdown(request: Request) -> dict[str, str]:
    """Ask the engine to exit gracefully (the lifespan hook stops any session and releases the arm).

    Loopback callers only: a remote-engine role must not be stoppable from the network.
    """
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "shutdown is only accepted from the local machine")
    request.app.state.request_shutdown()
    return {"status": "shutting_down"}


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
