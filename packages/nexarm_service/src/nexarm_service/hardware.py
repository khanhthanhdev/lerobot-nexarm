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

"""Port and camera discovery. Read-only: never opens a robot port."""

import platform
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class PortInfo(BaseModel):
    device: str
    description: str = ""
    vid: int | None = None
    pid: int | None = None


def list_ports() -> list[PortInfo]:
    if platform.system() in ("Windows", "Darwin"):
        from serial.tools import list_ports as serial_ports

        return [
            PortInfo(device=p.device, description=p.description or "", vid=p.vid, pid=p.pid)
            for p in serial_ports.comports()
        ]
    # /dev/tty* is mostly virtual consoles; USB serial adapters are what a robot uses.
    devices = sorted(str(p) for pattern in ("ttyUSB*", "ttyACM*") for p in Path("/dev").glob(pattern))
    return [PortInfo(device=d) for d in devices]


def list_cameras() -> list[dict[str, Any]]:
    from lerobot.cameras.opencv import OpenCVCamera

    return OpenCVCamera.find_cameras()
