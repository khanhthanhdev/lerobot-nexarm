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

"""In-process event hub behind ``WS /api/v1/ws/events``.

Events are refetch hints for the desktop, never the source of truth. ``publish`` is safe to call
from any thread (session workers publish from background threads).
"""

import asyncio
import threading
from typing import Any

EVENT_TYPES = (
    "joint_update",
    "session_changed",
    "jobs_changed",
    "job_progress",
    "coaching_state",
)

_QUEUE_SIZE = 256


class EventHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]] = []

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register the calling event loop; must be called from inside it."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_SIZE)
        with self._lock:
            self._subscribers.append((asyncio.get_running_loop(), queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s[1] is not queue]

    def publish(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event_type!r}")
        message = {"type": event_type, "data": data or {}}
        with self._lock:
            subscribers = list(self._subscribers)
        for loop, queue in subscribers:
            loop.call_soon_threadsafe(_put_drop_oldest, queue, message)


def _put_drop_oldest(queue: asyncio.Queue[dict[str, Any]], message: dict[str, Any]) -> None:
    # A slow client must not block publishers or grow memory: drop its oldest event instead.
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(message)
