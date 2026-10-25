# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""The ``local`` job runner: ``lerobot-train`` as a child process of the engine."""

import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from nexarm_service.dataset_store import DatasetStore
from nexarm_service.jobs import Job, JobRunner, JobSink

logger = logging.getLogger(__name__)

_TERMINATE_GRACE_S = 15.0
_LINE_SPLIT = re.compile(r"[\r\n]+")


def kept_episodes(request: Any, store: DatasetStore) -> list[int] | None:
    """The episodes to train on: an explicit list, else everything not excluded; ``None`` means all."""
    if request.episodes is not None:
        return list(request.episodes)
    if not request.use_excluded:
        return None
    excluded = set(store.excluded(request.dataset_repo_id))
    if not excluded:
        return None
    total = store.summary(request.dataset_repo_id).total_episodes
    kept = [e for e in range(total) if e not in excluded]
    if not kept:
        raise ValueError("every episode is excluded; nothing to train on")
    return kept


def build_train_command(job: Job, store: DatasetStore, output_dir: Path) -> list[str]:
    request = job.request
    dataset_root = store.path(request.dataset_repo_id)
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={request.dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--policy.type={request.policy_type}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        f"--output_dir={output_dir}",
        f"--job_name={request.job_name or f'{request.policy_type}_{job.id}'}",
        f"--steps={request.steps}",
        f"--batch_size={request.batch_size}",
        f"--log_freq={min(request.log_freq, request.steps)}",
    ]
    if request.save_freq:
        command.append(f"--save_freq={request.save_freq}")
    kept = kept_episodes(request, store)
    if kept is not None:
        command.append(f"--dataset.episodes=[{','.join(map(str, kept))}]")
    command.extend(f"--{key}={value}" for key, value in request.extra_args.items())
    return command


class LocalRunner(JobRunner):
    name = "local"

    def __init__(self, store: DatasetStore) -> None:
        self._store = store
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    def start(self, job: Job, sink: JobSink, output_dir: Path) -> None:
        command = build_train_command(job, self._store, output_dir)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(  # noqa: S603  (argument list, no shell; values validated by the request model)
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, creationflags=flags
        )
        with self._lock:
            self._procs[job.id] = proc
        threading.Thread(
            target=self._pump, args=(job.id, proc, sink), name=f"nexarm-train-{job.id}", daemon=True
        ).start()

    def cancel(self, job_id: str) -> None:
        with self._lock:
            proc = self._procs.get(job_id)
            if proc is None:
                return
            self._cancelled.add(job_id)
        self._terminate(proc)

    def shutdown(self) -> None:
        with self._lock:
            procs = list(self._procs.items())
            self._cancelled.update(job_id for job_id, _ in procs)
        for _, proc in procs:
            self._terminate(proc)

    @staticmethod
    def _terminate(proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=_TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _pump(self, job_id: str, proc: subprocess.Popen[bytes], sink: JobSink) -> None:
        assert proc.stdout is not None
        pending = ""
        for chunk in iter(lambda: proc.stdout.read1(4096), b""):  # type: ignore[union-attr,attr-defined]
            parts = _LINE_SPLIT.split(pending + chunk.decode("utf-8", errors="replace"))
            pending = parts.pop()
            for text in parts:
                if text.strip():
                    sink.line(text)
        if pending.strip():
            sink.line(pending)
        code = proc.wait()
        with self._lock:
            self._procs.pop(job_id, None)
            cancelled = job_id in self._cancelled
            self._cancelled.discard(job_id)
        sink.exit(
            "cancelled" if cancelled else None if code == 0 else f"lerobot-train exited with code {code}"
        )
