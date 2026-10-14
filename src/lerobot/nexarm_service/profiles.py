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

"""Robot profiles: which ports and cameras belong to a NexArm setup, stored as one JSON file."""

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROFILE_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


class CameraProfile(BaseModel):
    type: Literal["opencv"] = "opencv"
    # Device index ("0") or path ("/dev/video0"); cameras are configured per robot, never defaulted.
    source: str
    width: int = Field(640, gt=0)
    height: int = Field(480, gt=0)
    fps: int = Field(30, gt=0)


class RobotProfile(BaseModel):
    name: str = Field(pattern=PROFILE_NAME_RE)
    follower_port: str
    leader_port: str | None = None
    baudrate: int = Field(1_000_000, gt=0)
    cameras: dict[str, CameraProfile] = Field(default_factory=dict)


def default_home() -> Path:
    override = os.environ.get("LEROBOT_NEXARM_HOME")
    return Path(override) if override else Path.home() / ".cache" / "lerobot" / "nexarm_service"


def valid_name(name: str) -> bool:
    return re.match(PROFILE_NAME_RE, name) is not None


class ProfileStore:
    def __init__(self, home: Path) -> None:
        self._path = home / "profiles.json"
        self._lock = threading.Lock()

    def _load(self) -> dict[str, RobotProfile]:
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return {name: RobotProfile.model_validate(p) for name, p in raw.items()}

    def _save(self, profiles: dict[str, RobotProfile]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: p.model_dump() for name, p in profiles.items()}
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self._path)

    def list(self) -> list[RobotProfile]:
        with self._lock:
            return sorted(self._load().values(), key=lambda p: p.name)

    def get(self, name: str) -> RobotProfile | None:
        with self._lock:
            return self._load().get(name)

    def put(self, profile: RobotProfile) -> None:
        with self._lock:
            profiles = self._load()
            profiles[profile.name] = profile
            self._save(profiles)

    def delete(self, name: str) -> bool:
        with self._lock:
            profiles = self._load()
            if name not in profiles:
                return False
            del profiles[name]
            self._save(profiles)
            return True
