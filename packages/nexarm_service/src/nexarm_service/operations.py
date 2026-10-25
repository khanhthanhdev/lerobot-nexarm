# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Background operations for work that outlives an HTTP request (dataset edits, Hub transfers).

One operation runs at a time per kind of resource would be over-engineering for a single-user local
engine; operations run on their own threads and report progress through ``jobs_changed`` events.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from nexarm_service.events import EventHub

logger = logging.getLogger(__name__)


class OperationState(StrEnum):
    running = "running"
    done = "done"
    failed = "failed"


class Operation(BaseModel):
    id: str
    label: str
    state: OperationState
    started_at: float
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class OperationRunner:
    def __init__(self, events: EventHub) -> None:
        self._events = events
        self._lock = threading.Lock()
        self._ops: dict[str, Operation] = {}
        self._threads: dict[str, threading.Thread] = {}

    def submit(self, label: str, fn: Callable[[], dict[str, Any] | None]) -> Operation:
        op = Operation(
            id=uuid.uuid4().hex[:12], label=label, state=OperationState.running, started_at=time.time()
        )
        with self._lock:
            self._ops[op.id] = op
        thread = threading.Thread(target=self._run, args=(op.id, fn), name=f"nexarm-op-{op.id}", daemon=True)
        self._threads[op.id] = thread
        self._notify(op)
        thread.start()
        return op.model_copy()

    def get(self, op_id: str) -> Operation:
        with self._lock:
            return self._ops[op_id].model_copy()

    def list(self) -> list[Operation]:
        with self._lock:
            return sorted(
                (o.model_copy() for o in self._ops.values()), key=lambda o: o.started_at, reverse=True
            )

    def wait(self, op_id: str, timeout: float = 60.0) -> Operation:
        self._threads[op_id].join(timeout)
        return self.get(op_id)

    def _run(self, op_id: str, fn: Callable[[], dict[str, Any] | None]) -> None:
        try:
            result = fn()
            error = None
        except Exception as exc:
            logger.exception("operation %s failed", op_id)
            result, error = None, f"{type(exc).__name__}: {exc}"
        with self._lock:
            op = self._ops[op_id]
            op.state = OperationState.failed if error else OperationState.done
            op.result, op.error, op.finished_at = result, error, time.time()
            snapshot = op.model_copy()
        self._notify(snapshot)

    def _notify(self, op: Operation) -> None:
        self._events.publish("jobs_changed", {"kind": "operation", "id": op.id, "state": op.state.value})
