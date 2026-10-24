# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Fixtures shared by the remote-training tests: a template dataset and a second, real engine."""

import sys

import pytest

pytest.importorskip("fastapi", reason="install lerobot[service]")

from nexarm_service import local_runner  # noqa: E402
from nexarm_service.app import create_app  # noqa: E402
from tests.nexarm_service.dataset_factory import make_dataset  # noqa: E402
from tests.nexarm_service.live import LiveEngine  # noqa: E402


@pytest.fixture(scope="module")
def template_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("remote-datasets")
    make_dataset(root, "me/demo")
    return root


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
