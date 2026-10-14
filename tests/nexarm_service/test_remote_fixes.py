# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Remote training edge cases found in review: stuck jobs, stale server copies, shutdown auth."""

import contextlib
import json
import shutil

import pytest

pytest.importorskip("fastapi", reason="install lerobot[service]")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from fastapi.testclient import TestClient  # noqa: E402

from lerobot.nexarm_service.app import API_PREFIX, create_app  # noqa: E402
from lerobot.nexarm_service.jobs import JobState, TrainRequest  # noqa: E402
from tests.nexarm_service.live import wait_for  # noqa: E402
from tests.nexarm_service.remote_support import make_local, remote_request  # noqa: E402


def test_any_failure_in_a_remote_job_ends_it_and_the_next_queued_job_starts(
    tmp_path, template_root, remote_engine
):
    app, runner = make_local(tmp_path, template_root, remote_engine.url)
    jobs = app.state.jobs
    calls = {"n": 0}

    @contextlib.contextmanager
    def connect(server):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyError("a surprise that is not a RemoteError")
        yield remote_engine.url

    runner._connect = connect
    first = jobs.submit(remote_request())
    second = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(first.id).state == "failed", 20), jobs.get(first.id)
    assert "KeyError" in jobs.get(first.id).error
    assert wait_for(lambda: jobs.get(second.id).state == "done", 60), jobs.get(second.id)


def test_a_dataset_that_vanished_after_submit_fails_the_job_instead_of_wedging_the_queue(
    tmp_path, template_root, remote_engine
):
    app, runner = make_local(tmp_path, template_root, remote_engine.url)
    jobs = app.state.jobs
    shutil.rmtree(tmp_path / "local-datasets" / "me" / "demo")
    first = jobs.submit(remote_request())
    second = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(first.id).state == "failed", 20)
    assert wait_for(lambda: jobs.get(second.id).state == "failed", 20)  # it ran too: the queue kept moving


def test_the_fingerprint_changes_with_the_data_not_just_the_counts(tmp_path, template_root):
    datasets = tmp_path / "datasets"
    shutil.copytree(template_root / "me", datasets / "me")
    store = create_app(tmp_path / "home", datasets_root=datasets).state.dataset_store
    before = store.summary("me/demo")
    assert len(before.fingerprint) == 64 and before.transferred is False
    info_path = datasets / "me" / "demo" / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["description"] = "re-recorded"  # same counts, different content
    info_path.write_text(json.dumps(info))
    assert store.summary("me/demo").fingerprint != before.fingerprint


def test_a_changed_dataset_replaces_the_servers_own_copy_but_never_one_the_app_did_not_upload(
    tmp_path, template_root, remote_engine
):
    app, runner = make_local(tmp_path, template_root, remote_engine.url)
    jobs = app.state.jobs
    first = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(first.id).state == "done", 60)
    server_copy = tmp_path / "remote-datasets" / "me" / "demo"
    assert (server_copy / "meta" / "nexarm_transfer.json").is_file()

    # re-recorded under the same name with identical counts: the content differs, so the copy is replaced
    info_path = tmp_path / "local-datasets" / "me" / "demo" / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["description"] = "second take"
    info_path.write_text(json.dumps(info))
    second = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(second.id).state == "done", 60), jobs.get(second.id)
    assert json.loads((server_copy / "meta" / "info.json").read_text())["description"] == "second take"
    assert any("out of date" in line for line in jobs.logs(second.id))
    assert not any(p.name.startswith((".old-", ".incoming-")) for p in server_copy.parent.iterdir())

    # a dataset the user keeps on the server (no transfer marker) is never touched
    (server_copy / "meta" / "nexarm_transfer.json").unlink()
    info["description"] = "third take"
    info_path.write_text(json.dumps(info))
    third = jobs.submit(remote_request())
    assert wait_for(lambda: jobs.get(third.id).state == "failed", 30)
    assert "not uploaded by this app" in jobs.get(third.id).error
    assert json.loads((server_copy / "meta" / "info.json").read_text())["description"] == "second take"


def test_replacing_a_server_copy_is_refused_while_a_job_uses_it(tmp_path, template_root):
    app = create_app(tmp_path / "home", datasets_root=template_root)
    other = create_app(tmp_path / "other", datasets_root=tmp_path / "other-datasets")
    with TestClient(app) as a, TestClient(other) as b:
        blob = a.get(f"{API_PREFIX}/datasets/me/demo/archive").content
        assert b.put(f"{API_PREFIX}/datasets/me/demo/archive", content=blob).status_code == 201
        assert b.put(f"{API_PREFIX}/datasets/me/demo/archive?replace=true", content=blob).status_code == 201
        busy = other.state.jobs.submit(TrainRequest(dataset_repo_id="me/demo", policy_type="act", steps=5))
        other.state.jobs._jobs[busy.id].state = JobState.running  # a training job is using the copy
        refused = b.put(f"{API_PREFIX}/datasets/me/demo/archive?replace=true", content=blob)
        assert refused.status_code == 409 and "training job" in refused.json()["detail"]
        other.state.jobs._jobs[busy.id].state = JobState.cancelled


def test_shutdown_is_open_to_a_token_holder_on_the_network_and_closed_to_strangers(tmp_path):
    calls = []
    app = create_app(tmp_path / "a", token="tok")
    app.state.request_shutdown = lambda: calls.append("token")
    with TestClient(app, client=("100.64.0.9", 50000)) as client:
        assert client.post(f"{API_PREFIX}/shutdown").status_code == 401
        assert (
            client.post(f"{API_PREFIX}/shutdown", headers={"Authorization": "Bearer tok"}).status_code == 202
        )
    assert calls == ["token"]

    open_app = create_app(tmp_path / "b")  # no token: only loopback may stop it
    open_app.state.request_shutdown = lambda: calls.append("open")
    with TestClient(open_app, client=("100.64.0.9", 50000)) as client:
        assert client.post(f"{API_PREFIX}/shutdown").status_code == 403
    assert calls == ["token"]
