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

"""Local HTTP/WebSocket engine that the NexArm desktop app drives.

Install with ``pip install 'lerobot[service]'`` and start with ``lerobot-nexarm-service``.
"""

# Bumped on any breaking change to the HTTP/WS contract in docs/service/openapi.json.
API_VERSION = "1"

# Feature flags reported by /health; later phases append to this list as they land.
CAPABILITIES: tuple[str, ...] = (
    "profiles",
    "ports",
    "cameras",
    "events",
    "sessions",
    "teleoperation",
    "calibration",
)
