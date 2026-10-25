# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Helpers for the remote-training tests: a local engine wired to a remote runner."""

import contextlib
import shutil

from nexarm_service.app import create_app
from nexarm_service.jobs import TrainRequest
from nexarm_service.remote import RemoteRunner
from nexarm_service.servers import RemoteServer

SERVER = RemoteServer(name="gpu1", host="gpu.example.com", user="ubuntu", key_path="/keys/id_ed25519")


def make_local(tmp_path, template_root, remote_url, excluded=()):
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
