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

"""FastAPI application factory for the NexArm engine."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI

from lerobot.nexarm_service import API_VERSION
from lerobot.nexarm_service.events import EventHub
from lerobot.nexarm_service.profiles import ProfileStore, default_home
from lerobot.nexarm_service.routes import system

API_PREFIX = "/api/v1"


def create_app(home: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="NexArm engine",
        version=API_VERSION,
        description="Local engine for the NexArm desktop app.",
    )
    app.state.events = EventHub()
    app.state.profiles = ProfileStore(home or default_home())
    app.include_router(system.router, prefix=API_PREFIX)
    return app


def openapi_schema() -> dict[str, Any]:
    """The contract snapshot in docs/service/openapi.json; independent of the lerobot release."""
    return create_app(Path("unused")).openapi()
