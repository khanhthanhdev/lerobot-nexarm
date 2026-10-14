# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Remote training: SSH commands, archive transfer, and a full job against a second (real) engine."""

import contextlib
import io
import socket
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="install lerobot[service]")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

import uvicorn  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from lerobot.nexarm_service import archive, local_runner, servers  # noqa: E402
from lerobot.nexarm_service.app import API_PREFIX, create_app  # noqa: E402
from lerobot.nexarm_service.jobs import TrainRequest  # noqa: E402
from lerobot.nexarm_service.models import describe_checkpoint  # noqa: E402
from lerobot.nexarm_service.remote import RemoteClient, RemoteError, RemoteRunner, ensure_engine  # noqa: E402
from lerobot.nexarm_service.servers import RemoteServer  # noqa: E402
from tests.nexarm_service.dataset_factory import make_dataset  # noqa: E402

SERVER = RemoteServer(name="gpu1", host="gpu.example.com", user="ubuntu", key_path="/keys/id_ed25519")


def wait_for(predicate, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# -- ssh commands ---------------------------------------------------------------------------
def test_tunnel_command_forwards_loopback_to_loopback_and_never_prompts():
    cmd = servers.tunnel_command(SERVER, 50123)
    assert cmd[:4] == ["ssh", "-N", "-L", "127.0.0.1:50123:127.0.0.1:8765"]
    assert "ExitOnForwardFailure=yes" in cmd and "BatchMode=yes" in cmd
    assert cmd[cmd.index("-i") + 1] == "/keys/id_ed25519"
    assert cmd[-2:] == ["--", "ubuntu@gpu.example.com"]


def test_a_profile_without_a_key_leaves_it_to_the_ssh_agent():
    cmd = servers.start_engine_command(RemoteServer(name="g", host="10.0.0.5", user="root", ssh_port=2222))
    assert "-i" not in cmd and cmd[cmd.index("-p") + 1] == "2222"
    assert cmd[-1] == servers.REMOTE_START_COMMAND


@pytest.mark.parametrize(
    "bad",
    [
        {"host": "-oProxyCommand=evil"},
        {"host": "host name"},
        {"host": "a;b"},
        {"user": "root; rm -rf /"},
        {"user": "-x"},
        {"key_path": "-oProxyCommand=x"},
    ],
)
def test_option_injection_is_rejected_at_the_profile(bad):
    with pytest.raises(ValidationError):
        RemoteServer(**{"name": "g", "host": "h.example.com", "user": "u", **bad})


def test_ssh_tunnel_comes_up_when_the_forwarded_port_accepts_connections():
    listeners = []

    class FakeProc:
        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        port = int(cmd[cmd.index("-L") + 1].split(":")[1])
        listener = socket.socket()
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        listeners.append(listener)
        return FakeProc()

    tunnel = servers.SshTunnel(SERVER, popen=fake_popen, wait_s=5)
    with tunnel as url:
        assert url.startswith("http://127.0.0.1:")
    for listener in listeners:
        listener.close()


def test_ssh_tunnel_reports_why_ssh_exited():
    class DeadProc:
        stderr = io.BytesIO(b"Permission denied (publickey).")

        def poll(self):
            return 255

    with pytest.raises(ConnectionError, match="Permission denied"):
        servers.SshTunnel(SERVER, popen=lambda *a, **k: DeadProc(), wait_s=5).__enter__()


# -- archives -------------------------------------------------------------------------------
def make_tree(root: Path) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    (root / "data.bin").write_bytes(b"x" * 1000)
    return root


def test_pack_and_extract_round_trip(tmp_path):
    src = make_tree(tmp_path / "src")
    packed = archive.pack(src, tmp_path / "scratch")
    out = tmp_path / "out" / "copy"
    assert archive.extract(packed, out) == 1000 + 2
    assert (out / "meta" / "info.json").read_text() == "{}"
    assert (out / "data.bin").stat().st_size == 1000
    with pytest.raises(archive.ArchiveError, match="already exists"):
        archive.extract(packed, out)


def _tar_with(tmp_path, build):
    path = tmp_path / "evil.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        build(tar)
    return path


def test_extract_refuses_traversal_links_and_oversize_and_leaves_nothing_behind(tmp_path):
    def traversal(tar):
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))

    def symlink(tar):
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    for name, build in (("traversal", traversal), ("symlink", symlink)):
        dest = tmp_path / f"{name}-out"
        with pytest.raises(archive.ArchiveError):
            archive.extract(_tar_with(tmp_path, build), dest)
        assert not dest.exists()
        assert not any(p.name.startswith(".extract-") for p in tmp_path.iterdir())
    assert not (tmp_path.parent / "escape.txt").exists()

    packed = archive.pack(make_tree(tmp_path / "big"), tmp_path / "scratch")
    with pytest.raises(archive.ArchiveError, match="larger"):
        archive.extract(packed, tmp_path / "big-out", max_bytes=10)


# -- archive endpoints ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def template_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("remote-datasets")
    make_dataset(root, "me/demo")
    return root


def test_dataset_archive_endpoints_round_trip_and_refuse_overwrite(tmp_path, template_root):
    app = create_app(tmp_path / "home", datasets_root=template_root)
    other = create_app(tmp_path / "other", datasets_root=tmp_path / "other-datasets")
    with TestClient(app) as a, TestClient(other) as b:
        blob = a.get(f"{API_PREFIX}/datasets/me/demo/archive").content
        assert blob[:2] == b"\x1f\x8b"
        up = b.put(f"{API_PREFIX}/datasets/me/demo/archive", content=blob)
        assert up.status_code == 201 and up.json()["total_episodes"] == 3
        assert b.put(f"{API_PREFIX}/datasets/me/demo/archive", content=blob).status_code == 409
        assert b.put(f"{API_PREFIX}/datasets/me/junk/archive", content=b"not a tarball").status_code == 422
        assert not (tmp_path / "other-datasets" / "me" / "junk").exists()
        assert b.put(f"{API_PREFIX}/datasets/me/..%2Fx/archive", content=blob).status_code in (404, 422)
        assert a.get(f"{API_PREFIX}/datasets/me/missing/archive").status_code == 404


# -- a remote engine for real ---------------------------------------------------------------
class LiveEngine:
    def __init__(self, app):
        self.app = app
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.url = f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.thread.start()
        assert wait_for(lambda: self.server.started, 20)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(20)


FAKE_TRAINER = """
import json, pathlib, sys, time
out = pathlib.Path(sys.argv[1]); sleep_s = float(sys.argv[2])
line = "step:{n} smpl:2K ep:8 epch:0.50 loss:{loss} grdn:11.2 lr:1.0e-05 updt_s:0.1 data_s:0.0 smp/s:79"
for i, loss in enumerate((0.9, 0.5, 0.3), 1):
    print(line.format(n=i, loss=loss), flush=True)
    time.sleep(sleep_s)
ckpt = out / "checkpoints" / "000003" / "pretrained_model"
ckpt.mkdir(parents=True)
(ckpt / "config.json").write_text(json.dumps({"type": "act"}))
(ckpt / "train_config.json").write_text(json.dumps({"dataset": {"repo_id": "me/demo"}}))
(ckpt / "model.safetensors").write_bytes(b"weights")
"""


@pytest.fixture
def remote_engine(tmp_path, template_root, monkeypatch):
    """A second engine whose trainer is a script; records the request it was asked to train."""
    seen = []
    sleep_s = {"value": 0.05}

    def fake_command(job, store, output_dir):
        seen.append(job.request)
        return [sys.executable, "-u", "-c", FAKE_TRAINER, str(output_dir), str(sleep_s["value"])]

    monkeypatch.setattr(local_runner, "build_train_command", fake_command)
    app = create_app(tmp_path / "remote-home", datasets_root=tmp_path / "remote-datasets")
    with LiveEngine(app) as live:
        live.seen, live.sleep_s = seen, sleep_s
        yield live


def make_local(tmp_path, template_root, remote_url, excluded=()):
    import shutil

    datasets = tmp_path / "local-datasets"
    shutil.copytree(template_root / "me", datasets / "me")
    app = create_app(tmp_path / "local-home", datasets_root=datasets)
    if excluded:
        app.state.dataset_store.set_excluded("me/demo", list(excluded))
    runner = RemoteRunner(
        app.state.dataset_store,
        app.state.scratch,
        connect=lambda server: contextlib.nullcontext(remote_url),
        poll_s=0.05,
    )
    app.state.jobs._runners["remote_engine"] = runner
    return app, runner


def remote_request(**kw):
    return TrainRequest(
        dataset_repo_id="me/demo",
        policy_type="act",
        steps=3,
        log_freq=1,
        runner="remote_engine",
        server=SERVER,
        **kw,
    )


def test_a_remote_job_uploads_trains_mirrors_progress_and_returns_the_checkpoint(
    tmp_path, template_root, remote_engine
):
    app, runner = make_local(tmp_path, template_root, remote_engine.url, excluded=[1])
    jobs = app.state.jobs
    job = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(job.id).state not in ("queued", "running")), jobs.get(job.id)
    final = jobs.get(job.id)
    assert final.state == "done", final.error

    # the dataset went up, and the server was told the curated episode list explicitly
    assert (tmp_path / "remote-datasets" / "me" / "demo" / "meta" / "info.json").is_file()
    assert remote_engine.seen[0].episodes == [0, 2] and remote_engine.seen[0].server is None
    assert remote_engine.seen[0].runner == "local"

    # progress and logs from the server are mirrored without double counting
    assert [m.step for m in jobs.metrics(job.id)] == [1, 2, 3]
    assert (final.step, final.loss) == (3, 0.3)
    assert any("[remote] uploaded me/demo" in line for line in jobs.logs(job.id))

    # the checkpoint is back on this machine in the normal place
    checkpoints = jobs.checkpoints(job.id)
    assert [c["step"] for c in checkpoints] == [3]
    info = describe_checkpoint(Path(checkpoints[0]["path"]), "job", job.id)
    assert info.policy_type == "act" and info.dataset_repo_id == "me/demo"
    assert (Path(checkpoints[0]["path"]) / "model.safetensors").read_bytes() == b"weights"


def test_a_second_job_reuses_the_dataset_already_on_the_server(tmp_path, template_root, remote_engine):
    app, runner = make_local(tmp_path, template_root, remote_engine.url)
    jobs = app.state.jobs
    first = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(first.id).state == "done")
    second = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(second.id).state == "done")
    assert any("already on the server" in line for line in jobs.logs(second.id))


def test_a_different_dataset_with_the_same_name_on_the_server_is_never_overwritten(
    tmp_path, template_root, remote_engine
):
    make_dataset(tmp_path / "remote-datasets", "me/demo", episodes=2)
    app, runner = make_local(tmp_path, template_root, remote_engine.url)
    jobs = app.state.jobs
    job = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(job.id).state == "failed")
    assert "different dataset named me/demo" in jobs.get(job.id).error
    assert remote_engine.seen == []


def test_cancelling_a_remote_job_cancels_it_on_the_server_too(tmp_path, template_root, remote_engine):
    remote_engine.sleep_s["value"] = 30
    app, runner = make_local(tmp_path, template_root, remote_engine.url)
    jobs = app.state.jobs
    job = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(job.id).step >= 1)
    jobs.cancel(job.id)
    assert wait_for(lambda: jobs.get(job.id).state == "cancelled")
    remote_jobs = remote_engine.app.state.jobs.list()
    assert wait_for(lambda: remote_engine.app.state.jobs.get(remote_jobs[0].id).state == "cancelled")
    remote_engine.app.state.jobs.shutdown()


def test_losing_the_server_mid_job_fails_the_local_job_with_a_clear_message(
    tmp_path, template_root, remote_engine, monkeypatch
):
    import httpx

    remote_engine.sleep_s["value"] = 30
    app, runner = make_local(tmp_path, template_root, remote_engine.url)
    jobs = app.state.jobs
    job = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(job.id).step >= 1)

    def dead(self, job_id):  # the tunnel drops: every later poll fails to connect
        raise httpx.ConnectError("tunnel closed")

    monkeypatch.setattr(RemoteClient, "job", dead)
    assert wait_for(lambda: jobs.get(job.id).state == "failed", 30)
    error = jobs.get(job.id).error
    assert "lost the connection" in error and "may still be running" in error
    remote_engine.app.state.jobs.shutdown()


def test_a_remote_job_needs_a_server_and_a_known_runner(tmp_path, template_root):
    app = create_app(tmp_path / "home", datasets_root=template_root)
    with pytest.raises(ValueError):
        # the runner refuses to start without a server; the manager reports it as a failed start
        RemoteRunner(app.state.dataset_store, app.state.scratch).start(
            type("J", (), {"request": TrainRequest(dataset_repo_id="me/demo", policy_type="act", steps=1)})(),
            None,
            tmp_path,
        )


# -- engine start over ssh ------------------------------------------------------------------
class FakeClient:
    def __init__(self, healthy_after):
        self.calls = 0
        self.healthy_after = healthy_after

    def health(self):
        self.calls += 1
        if self.calls <= self.healthy_after:
            raise RemoteError("down")
        return {}


def test_ensure_engine_starts_the_remote_engine_only_when_it_is_down():
    ran = []

    def run(cmd, **kwargs):
        ran.append(cmd)
        return type("R", (), {"returncode": 0, "stderr": b""})()

    ensure_engine(SERVER, FakeClient(0), run=run)
    assert ran == []  # already up
    ensure_engine(SERVER, FakeClient(1), run=run, wait_s=5)
    assert ran and ran[0][-1] == servers.REMOTE_START_COMMAND


def test_ensure_engine_reports_an_ssh_failure():
    def run(cmd, **kwargs):
        return type("R", (), {"returncode": 255, "stderr": b"Connection refused"})()

    with pytest.raises(RemoteError, match="Connection refused"):
        ensure_engine(SERVER, FakeClient(99), run=run)


# -- the server test route ------------------------------------------------------------------
def test_server_test_route_reports_ssh_and_engine_state(tmp_path, remote_engine, monkeypatch):
    body = SERVER.model_dump()

    class OkTunnel:
        def __init__(self, server, **kw):
            pass

        def __enter__(self):
            return remote_engine.url

        def __exit__(self, *exc):
            pass

    class DeadTunnel(OkTunnel):
        def __enter__(self):
            raise ConnectionError("ssh to gpu.example.com failed: Permission denied")

    app = create_app(tmp_path / "home")
    with TestClient(app) as client:
        monkeypatch.setattr(servers, "SshTunnel", OkTunnel)
        ok = client.post(f"{API_PREFIX}/servers/test", json=body).json()
        assert ok["ssh_ok"] and ok["engine"]["api_version"] == "1"
        monkeypatch.setattr(servers, "SshTunnel", DeadTunnel)
        bad = client.post(f"{API_PREFIX}/servers/test", json=body).json()
        assert not bad["ssh_ok"] and "Permission denied" in bad["detail"]
        assert (
            client.post(f"{API_PREFIX}/servers/test", json={**body, "host": "-oProxyCommand=x"}).status_code
            == 422
        )


def test_remote_client_surfaces_engine_errors(remote_engine):
    client = RemoteClient(remote_engine.url)
    with pytest.raises(RemoteError, match="HTTP 404"):
        client.job("nope")
    assert client.dataset("me/none") is None
    client.close()
