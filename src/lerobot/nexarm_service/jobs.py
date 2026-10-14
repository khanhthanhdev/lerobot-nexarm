# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Training jobs: a queue, persisted history and a runner interface.

A ``JobRunner`` knows how to execute one training job somewhere (``local`` runs ``lerobot-train`` as a
subprocess; ``remote_engine`` drives an engine on a GPU server). The manager owns everything else: the
queue, log and metric persistence, state transitions and ``jobs_changed`` / ``job_progress`` events.
Future runners (a managed cloud GPU service) only implement ``JobRunner``.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from lerobot.nexarm_service.events import EventHub
from lerobot.nexarm_service.servers import RemoteServer

logger = logging.getLogger(__name__)

_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
_ARG_KEY_RE = re.compile(r"^[A-Za-z0-9_.]+$")
_METRIC_RE = re.compile(r"(?<!\S)([A-Za-z_/]+):(\S+)")


class JobState(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


ACTIVE_STATES = (JobState.queued, JobState.running)


class TrainRequest(BaseModel):
    dataset_repo_id: str
    policy_type: str = Field(pattern=r"^[A-Za-z0-9_]+$")
    steps: int = Field(gt=0)
    batch_size: int = Field(8, gt=0)
    log_freq: int = Field(200, gt=0)
    save_freq: int | None = Field(None, gt=0)
    job_name: str | None = Field(None, pattern=_NAME_RE)
    use_excluded: bool = True  # leave out episodes the curation list excludes
    extra_args: dict[str, str] = {}  # additional lerobot-train flags, e.g. {"policy.chunk_size": "50"}
    runner: str = "local"
    server: RemoteServer | None = None  # remote_engine runner only: the GPU server to train on
    # Explicit episode list; the remote runner sends the locally curated list because the server has no
    # curation file. When set it replaces use_excluded.
    episodes: list[int] | None = None

    def validate_extra_args(self) -> None:
        for key, value in self.extra_args.items():
            if not _ARG_KEY_RE.match(key) or "\n" in value or key.split(".")[0] in {"output_dir", "job_name"}:
                raise ValueError(f"extra_args key {key!r} is not allowed")


class MetricPoint(BaseModel):
    step: int
    values: dict[str, float]


class Job(BaseModel):
    id: str
    kind: str = "train"
    runner: str
    state: JobState
    request: TrainRequest
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    step: int = 0
    total_steps: int
    loss: float | None = None
    lr: float | None = None
    output_dir: str | None = None


def parse_metric_line(line: str) -> dict[str, float] | None:
    """Return the metrics of a ``lerobot-train`` log line (``step:200 ... loss:0.123 lr:1.0e-05 ...``)."""
    if "loss:" not in line or "step:" not in line:
        return None
    values: dict[str, float] = {}
    for key, raw in _METRIC_RE.findall(line):
        if key in ("step", "smpl", "ep"):  # abbreviated counters ("1K"); the step is counted instead
            continue
        try:
            values[key] = float(raw)
        except ValueError:
            continue
    return values if "loss" in values else None


class JobSink:
    """What a runner reports back to the manager."""

    def __init__(self, manager: JobManager, job_id: str) -> None:
        self._manager, self._job_id = manager, job_id

    def line(self, text: str, parse: bool = True) -> None:
        self._manager._on_line(self._job_id, text, parse)

    def metric(self, values: dict[str, float], step: int | None = None) -> None:
        self._manager._on_metric(self._job_id, values, step)

    def exit(self, error: str | None) -> None:
        self._manager._on_exit(self._job_id, error)


class JobRunner(ABC):
    name: str

    @abstractmethod
    def start(self, job: Job, sink: JobSink, output_dir: Path) -> None:
        """Begin the job without blocking; report through ``sink`` and finish with ``sink.exit``."""

    @abstractmethod
    def cancel(self, job_id: str) -> None: ...

    def checkpoint_dirs(self, job: Job, output_dir: Path) -> list[Path]:
        """Local checkpoint directories (``<step>/pretrained_model``) the job produced."""
        root = output_dir / "checkpoints"
        return (
            sorted(p for p in root.glob("*/pretrained_model") if p.parent.name != "last")
            if root.is_dir()
            else []
        )

    def shutdown(self) -> None:  # noqa: B027  (optional hook)
        """Stop everything this runner started."""


class JobManager:
    def __init__(self, home: Path, events: EventHub, runners: dict[str, JobRunner]) -> None:
        self._dir = home / "jobs"
        self._events = events
        self._runners = runners
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._steps_logged: dict[str, int] = {}
        self._load()

    # -- persistence ---------------------------------------------------------------------------
    def _job_dir(self, job_id: str) -> Path:
        return self._dir / job_id

    def _save(self, job: Job) -> None:
        d = self._job_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._dir.is_dir():
            return
        for f in self._dir.glob("*/job.json"):
            job = Job.model_validate_json(f.read_text(encoding="utf-8"))
            if job.state in ACTIVE_STATES:
                job.state, job.error = JobState.failed, "engine stopped while the job was active"
                job.finished_at = time.time()
                self._save(job)
            self._jobs[job.id] = job

    def output_dir(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "output"

    # -- queries -------------------------------------------------------------------------------
    def list(self) -> list[Job]:
        with self._lock:
            return sorted(
                (j.model_copy() for j in self._jobs.values()), key=lambda j: j.created_at, reverse=True
            )

    def get(self, job_id: str) -> Job:
        with self._lock:
            return self._jobs[job_id].model_copy()

    def logs(self, job_id: str, tail: int = 200, offset: int | None = None) -> list[str]:
        """The last ``tail`` lines, or every line from ``offset`` on (how a remote runner follows the log)."""
        self.get(job_id)
        f = self._job_dir(job_id) / "train.log"
        if not f.is_file():
            return []
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[offset:] if offset is not None else lines[-tail:]

    def metrics(self, job_id: str) -> list[MetricPoint]:
        self.get(job_id)
        f = self._job_dir(job_id) / "metrics.jsonl"
        if not f.is_file():
            return []
        return [
            MetricPoint.model_validate_json(line)
            for line in f.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        job = self.get(job_id)
        runner = self._runners[job.runner]
        out = []
        for path in runner.checkpoint_dirs(job, self.output_dir(job_id)):
            out.append(
                {"step": int(path.parent.name) if path.parent.name.isdigit() else None, "path": str(path)}
            )
        return out

    # -- commands ------------------------------------------------------------------------------
    def submit(self, request: TrainRequest) -> Job:
        request.validate_extra_args()
        if request.runner not in self._runners:
            raise ValueError(f"unknown runner {request.runner!r}; available: {sorted(self._runners)}")
        job = Job(
            id=uuid.uuid4().hex[:10],
            runner=request.runner,
            state=JobState.queued,
            request=request,
            created_at=time.time(),
            total_steps=request.steps,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._save(job)
        self._publish(job)
        self._schedule()
        return self.get(job.id)

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            if job.state == JobState.queued:
                self._finish_locked(job, JobState.cancelled, None)
                snapshot = job.model_copy()
            elif job.state == JobState.running:
                snapshot = None
            else:
                return job.model_copy()
        if snapshot is not None:
            self._publish(snapshot)
            return snapshot
        self._runners[job.runner].cancel(job_id)
        return self.get(job_id)

    def shutdown(self) -> None:
        for runner in self._runners.values():
            runner.shutdown()

    # -- scheduling and runner callbacks -------------------------------------------------------
    def _schedule(self) -> None:
        with self._lock:
            starting = []
            for runner_name in self._runners:
                if any(j.state == JobState.running and j.runner == runner_name for j in self._jobs.values()):
                    continue
                queued = sorted(
                    (
                        j
                        for j in self._jobs.values()
                        if j.state == JobState.queued and j.runner == runner_name
                    ),
                    key=lambda j: j.created_at,
                )
                if queued:
                    job = queued[0]
                    job.state, job.started_at = JobState.running, time.time()
                    job.output_dir = str(self.output_dir(job.id))
                    self._steps_logged[job.id] = 0
                    self._save(job)
                    starting.append(job.model_copy())
        for job in starting:
            self._publish(job)
            try:
                self._runners[job.runner].start(job, JobSink(self, job.id), self.output_dir(job.id))
            except Exception as exc:
                logger.exception("job %s failed to start", job.id)
                self._on_exit(job.id, f"could not start: {exc}")

    def _on_line(self, job_id: str, text: str, parse: bool = True) -> None:
        with open(self._job_dir(job_id) / "train.log", "a", encoding="utf-8") as f:
            f.write(text + "\n")
        values = parse_metric_line(text) if parse else None
        if values is not None:
            self._on_metric(job_id, values, None)

    def _on_metric(self, job_id: str, values: dict[str, float], step: int | None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if step is None:
                self._steps_logged[job_id] = self._steps_logged.get(job_id, 0) + 1
                step = min(self._steps_logged[job_id] * job.request.log_freq, job.total_steps)
            job.step = step
            job.loss = values.get("loss", job.loss)
            job.lr = values.get("lr", job.lr)
            self._save(job)
            point = MetricPoint(step=step, values=values)
        with open(self._job_dir(job_id) / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(point.model_dump_json() + "\n")
        self._events.publish(
            "job_progress", {"id": job_id, "step": step, "total_steps": job.total_steps, **values}
        )

    def _on_exit(self, job_id: str, error: str | None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.state != JobState.running:
                return
            cancelled = error == "cancelled"
            self._finish_locked(
                job,
                JobState.cancelled if cancelled else JobState.failed if error else JobState.done,
                None if cancelled else error,
            )
            snapshot = job.model_copy()
        self._publish(snapshot)
        self._schedule()

    def _finish_locked(self, job: Job, state: JobState, error: str | None) -> None:
        job.state, job.error, job.finished_at = state, error, time.time()
        self._save(job)

    def _publish(self, job: Job) -> None:
        self._events.publish("jobs_changed", {"kind": "train", "id": job.id, "state": job.state.value})
