# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""The ``remote_engine`` job runner: train on a GPU server by driving the engine running there.

Flow: open an SSH tunnel to the server's loopback engine, start that engine if it is not running, upload
the dataset through the engine's archive endpoint, submit the same training request to the remote jobs
API, mirror its log and metrics into the local job, then download the checkpoints. This replaces the manual
rsync steps in SERVER_TRAINING_GUIDE.md.
"""

import logging
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path, PurePath
from typing import Any

import httpx

from lerobot.nexarm_service import API_VERSION
from lerobot.nexarm_service.archive import extract, pack
from lerobot.nexarm_service.dataset_store import DatasetStore
from lerobot.nexarm_service.jobs import Job, JobRunner, JobSink
from lerobot.nexarm_service.local_runner import kept_episodes
from lerobot.nexarm_service.servers import RemoteServer, SshTunnel, start_engine_command

logger = logging.getLogger(__name__)

TERMINAL = ("done", "failed", "cancelled")
MAX_POLL_FAILURES = 5


class RemoteError(RuntimeError):
    pass


class RemoteClient:
    """The slice of the engine API a remote runner needs, over the tunnel."""

    def __init__(self, base_url: str) -> None:
        self._http = httpx.Client(base_url=f"{base_url}/api/v1", timeout=httpx.Timeout(30.0, read=900.0))

    def close(self) -> None:
        self._http.close()

    def _ok(self, response: httpx.Response) -> Any:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RemoteError(f"{detail} (HTTP {response.status_code})")
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._ok(self._http.get("/health"))

    def dataset(self, repo_id: str) -> dict[str, Any] | None:
        response = self._http.get(f"/datasets/{repo_id}")
        if response.status_code == 404:
            return None
        return self._ok(response)["summary"]

    def upload_dataset(self, repo_id: str, archive: Path) -> None:
        def body() -> Any:
            with open(archive, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk

        self._ok(self._http.put(f"/datasets/{repo_id}/archive", content=body()))

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._ok(self._http.post("/jobs", json=request))

    def job(self, job_id: str) -> dict[str, Any]:
        return self._ok(self._http.get(f"/jobs/{job_id}"))

    def logs(self, job_id: str, offset: int) -> list[str]:
        return self._ok(self._http.get(f"/jobs/{job_id}/logs", params={"offset": offset}))

    def metrics(self, job_id: str) -> list[dict[str, Any]]:
        return self._ok(self._http.get(f"/jobs/{job_id}/metrics"))

    def checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        return self._ok(self._http.get(f"/jobs/{job_id}/checkpoints"))

    def cancel(self, job_id: str) -> None:
        self._ok(self._http.delete(f"/jobs/{job_id}"))

    def download_checkpoint(self, job_id: str, step: str, destination: Path) -> None:
        with self._http.stream("GET", f"/jobs/{job_id}/checkpoints/{step}/archive") as response:
            if response.status_code >= 400:
                response.read()
                raise RemoteError(f"could not download checkpoint {step}: HTTP {response.status_code}")
            with open(destination, "wb") as f:
                for chunk in response.iter_bytes(1024 * 1024):
                    f.write(chunk)


def ensure_engine(
    server: RemoteServer, client: RemoteClient, run: Any = subprocess.run, wait_s: float = 90.0
) -> None:
    """Make sure the engine answers through the tunnel, starting it over ssh when needed."""
    try:
        client.health()
        return
    except (httpx.HTTPError, RemoteError):
        pass
    result = run(start_engine_command(server), capture_output=True, timeout=120)
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()
        raise RemoteError(f"could not start the engine on {server.host}: {detail or 'ssh failed'}")
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            client.health()
            return
        except (httpx.HTTPError, RemoteError):
            time.sleep(1.0)
    raise RemoteError(f"the engine on {server.host} did not answer after it was started")


@contextmanager
def open_remote(server: RemoteServer) -> Any:
    """SSH tunnel plus a running remote engine; yields the tunnel's base URL."""
    with SshTunnel(server) as url:
        probe = RemoteClient(url)
        try:
            ensure_engine(server, probe)
        finally:
            probe.close()
        yield url


Connect = Callable[[RemoteServer], AbstractContextManager[str]]


class RemoteRunner(JobRunner):
    name = "remote_engine"

    def __init__(
        self, store: DatasetStore, scratch: Path, connect: Connect = open_remote, poll_s: float = 2.0
    ) -> None:
        self._store = store
        self._scratch = scratch
        self._connect = connect
        self._poll_s = poll_s
        self._cancel: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def start(self, job: Job, sink: JobSink, output_dir: Path) -> None:
        if job.request.server is None:
            raise ValueError("a remote_engine job needs a server")
        event = threading.Event()
        thread = threading.Thread(
            target=self._run, args=(job, sink, output_dir, event), name=f"nexarm-remote-{job.id}", daemon=True
        )
        with self._lock:
            self._cancel[job.id] = event
            self._threads[job.id] = thread
        thread.start()

    def cancel(self, job_id: str) -> None:
        with self._lock:
            event = self._cancel.get(job_id)
        if event is not None:
            event.set()

    def shutdown(self) -> None:
        with self._lock:
            events, threads = list(self._cancel.values()), list(self._threads.values())
        for event in events:
            event.set()
        for thread in threads:
            thread.join(timeout=20)

    def wait(self, job_id: str, timeout: float = 60.0) -> None:
        self._threads[job_id].join(timeout)

    # -- the job ---------------------------------------------------------------------------------
    def _run(self, job: Job, sink: JobSink, output_dir: Path, cancel: threading.Event) -> None:
        error: str | None = None
        try:
            error = self._drive(job, sink, output_dir, cancel)
        except (RemoteError, ConnectionError, httpx.HTTPError, OSError, ValueError) as exc:
            logger.exception("remote job %s failed", job.id)
            error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._cancel.pop(job.id, None)
        sink.exit(error)

    def _drive(self, job: Job, sink: JobSink, output_dir: Path, cancel: threading.Event) -> str | None:
        request = job.request
        assert request.server is not None
        server = request.server
        with self._connect(server) as base_url:
            client = RemoteClient(base_url)
            try:
                health = client.health()
                if health["api_version"] != API_VERSION:
                    raise RemoteError(
                        f"the server's engine speaks API {health['api_version']}, this one speaks {API_VERSION}"
                    )
                sink.line(f"[remote] connected to {server.host} (lerobot {health['version']})", parse=False)
                self._ensure_dataset(client, request.dataset_repo_id, sink)
                kept = kept_episodes(request, self._store)
                remote_request = request.model_copy(
                    update={"runner": "local", "server": None, "episodes": kept, "use_excluded": False}
                )
                remote = client.submit(remote_request.model_dump(mode="json"))
                sink.line(f"[remote] job {remote['id']} queued on {server.name}", parse=False)
                return self._follow(client, remote["id"], sink, output_dir, cancel)
            finally:
                client.close()

    def _ensure_dataset(self, client: RemoteClient, repo_id: str, sink: JobSink) -> None:
        local = self._store.summary(repo_id)
        remote = client.dataset(repo_id)
        if remote is not None:
            if (remote["total_episodes"], remote["total_frames"]) != (
                local.total_episodes,
                local.total_frames,
            ):
                raise RemoteError(
                    f"a different dataset named {repo_id} already exists on the server "
                    f"({remote['total_episodes']} episodes, {remote['total_frames']} frames); "
                    "rename or remove it there"
                )
            sink.line(f"[remote] {repo_id} is already on the server; reusing it", parse=False)
            return
        sink.line(f"[remote] uploading {repo_id} ({local.total_episodes} episodes) …", parse=False)
        archive = pack(self._store.path(repo_id), self._scratch)
        try:
            client.upload_dataset(repo_id, archive)
        finally:
            archive.unlink(missing_ok=True)
        sink.line(f"[remote] uploaded {repo_id}", parse=False)

    def _follow(
        self, client: RemoteClient, remote_id: str, sink: JobSink, output_dir: Path, cancel: threading.Event
    ) -> str | None:
        seen_lines = seen_points = failures = 0
        cancel_sent = False
        while True:
            if cancel.is_set() and not cancel_sent:
                client.cancel(remote_id)
                cancel_sent = True
            try:
                job = client.job(remote_id)
                for line in client.logs(remote_id, seen_lines):
                    sink.line(line, parse=False)
                    seen_lines += 1
                points = client.metrics(remote_id)
                for point in points[seen_points:]:
                    sink.metric(point["values"], point["step"])
                seen_points = len(points)
                failures = 0
            except (httpx.HTTPError, RemoteError) as exc:
                failures += 1
                if failures >= MAX_POLL_FAILURES:
                    raise RemoteError(
                        f"lost the connection to the server ({exc}); remote job {remote_id} may still be running"
                    ) from exc
                time.sleep(self._poll_s)
                continue
            if job["state"] in TERMINAL:
                break
            time.sleep(self._poll_s)
        if job["state"] == "cancelled":
            return "cancelled"
        if job["state"] == "failed":
            return f"the remote job failed: {job.get('error') or 'see the log'}"
        self._fetch_checkpoints(client, remote_id, output_dir, sink)
        return None

    def _fetch_checkpoints(
        self, client: RemoteClient, remote_id: str, output_dir: Path, sink: JobSink
    ) -> None:
        self._scratch.mkdir(parents=True, exist_ok=True)
        for checkpoint in client.checkpoints(remote_id):
            name = PurePath(checkpoint["path"].replace("\\", "/")).parent.name
            destination = output_dir / "checkpoints" / name / "pretrained_model"
            archive = self._scratch / f"checkpoint-{remote_id}-{name}.tar.gz"
            try:
                client.download_checkpoint(remote_id, name, archive)
                extract(archive, destination)
            finally:
                archive.unlink(missing_ok=True)
            sink.line(f"[remote] downloaded checkpoint {name}", parse=False)
