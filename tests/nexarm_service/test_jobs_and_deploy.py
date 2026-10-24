# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Tests for training jobs, model provenance and the rollout and eval sessions."""

import json
import sys
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="install lerobot[service]")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from fastapi.testclient import TestClient  # noqa: E402

from lerobot_robot_nexarm.motors.nexarm import GRIPPER_CLOSED_POS, JOINT_NAMES  # noqa: E402
from nexarm_service import local_runner  # noqa: E402
from nexarm_service.app import API_PREFIX, create_app  # noqa: E402
from nexarm_service.events import EventHub  # noqa: E402
from nexarm_service.jobs import (  # noqa: E402
    JobManager,
    JobRunner,
    JobState,
    TrainRequest,
    parse_metric_line,
)
from tests.nexarm_service.fakes import FakeHardware  # noqa: E402

REST = [2000, 2100, 1900, 2050, 2048, 2000]
PROFILE = {
    "name": "bench",
    "follower_port": "FOLLOWER",
    "leader_port": "LEADER",
    "cameras": {"front": {"source": "0", "width": 64, "height": 48, "fps": 10}},
}
LOG_LINE = (
    "INFO 2026-10-14 train.py:630 step:200 smpl:2K ep:8 epch:0.50 loss:0.842 grdn:11.2 lr:1.0e-05 "
    "updt_s:0.101 data_s:0.004 smp/s:79"
)


def wait_for(predicate, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def write_dataset_info(root: Path, repo_id: str, robot_type: str) -> None:
    meta = root / repo_id / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "robot_type": robot_type,
        "total_episodes": 4,
        "total_frames": 40,
        "features": {},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes" / "chunk-000").mkdir(parents=True)


def write_checkpoint(parent: Path, step: str, dataset_repo_id: str | None) -> Path:
    ckpt = parent / "checkpoints" / step / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"type": "act"}), encoding="utf-8")
    if dataset_repo_id:
        (ckpt / "train_config.json").write_text(
            json.dumps({"dataset": {"repo_id": dataset_repo_id}}), encoding="utf-8"
        )
    return ckpt


# -- metric parsing -------------------------------------------------------------------------
def test_parse_metric_line_reads_loss_and_lr_and_ignores_abbreviated_counters():
    values = parse_metric_line(LOG_LINE)
    assert values["loss"] == pytest.approx(0.842)
    assert values["lr"] == pytest.approx(1.0e-05)
    assert values["grdn"] == pytest.approx(11.2)
    assert "step" not in values and "smpl" not in values
    assert parse_metric_line("INFO Saving checkpoint") is None
    assert parse_metric_line("step:200 no loss here") is None


# -- manager with a scripted runner ---------------------------------------------------------
class ScriptedRunner(JobRunner):
    name = "local"

    def __init__(self):
        self.started = []
        self.sinks = {}
        self.cancelled = []

    def start(self, job, sink, output_dir):
        self.started.append(job.id)
        self.sinks[job.id] = sink

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        self.sinks[job_id].exit("cancelled")


def make_manager(tmp_path, runner=None):
    runner = runner or ScriptedRunner()
    return JobManager(tmp_path, EventHub(), {"local": runner}), runner


def request(**kw):
    return TrainRequest(dataset_repo_id="me/demo", policy_type="act", steps=1000, log_freq=200, **kw)


def test_queue_runs_one_job_at_a_time_in_order(tmp_path):
    manager, runner = make_manager(tmp_path)
    first, second = manager.submit(request()), manager.submit(request())
    assert manager.get(first.id).state == JobState.running
    assert manager.get(second.id).state == JobState.queued
    runner.sinks[first.id].exit(None)
    assert manager.get(first.id).state == JobState.done
    assert manager.get(second.id).state == JobState.running
    assert runner.started == [first.id, second.id]


def test_progress_counts_log_windows_and_persists_history(tmp_path):
    manager, runner = make_manager(tmp_path)
    job = manager.submit(request())
    sink = runner.sinks[job.id]
    for loss in (0.9, 0.7, 0.5):
        sink.line(LOG_LINE.replace("0.842", str(loss)))
    sink.line("INFO unrelated line")
    current = manager.get(job.id)
    assert (current.step, current.loss) == (600, 0.5)
    metrics = manager.metrics(job.id)
    assert [m.step for m in metrics] == [200, 400, 600]
    assert [m.values["loss"] for m in metrics] == [0.9, 0.7, 0.5]
    assert manager.logs(job.id)[-1] == "INFO unrelated line"

    reloaded, _ = make_manager(tmp_path)  # history survives an engine restart
    assert len(reloaded.metrics(job.id)) == 3


def test_a_job_active_at_restart_is_marked_failed(tmp_path):
    manager, runner = make_manager(tmp_path)
    job = manager.submit(request())
    reloaded, _ = make_manager(tmp_path)
    again = reloaded.get(job.id)
    assert again.state == JobState.failed and "engine stopped" in again.error


def test_cancel_queued_and_running(tmp_path):
    manager, runner = make_manager(tmp_path)
    running, queued = manager.submit(request()), manager.submit(request())
    assert manager.cancel(queued.id).state == JobState.cancelled
    assert manager.cancel(running.id).state == JobState.cancelled
    assert runner.cancelled == [running.id]
    assert manager.cancel(running.id).state == JobState.cancelled  # idempotent


def test_failed_exit_keeps_the_error_and_next_job_starts(tmp_path):
    manager, runner = make_manager(tmp_path)
    a, b = manager.submit(request()), manager.submit(request())
    runner.sinks[a.id].exit("lerobot-train exited with code 1")
    assert manager.get(a.id).state == JobState.failed
    assert manager.get(b.id).state == JobState.running


def test_extra_args_cannot_override_output_or_inject(tmp_path):
    manager, _ = make_manager(tmp_path)
    with pytest.raises(ValueError):
        manager.submit(request(extra_args={"output_dir": "/etc"}))
    with pytest.raises(ValueError):
        manager.submit(request(extra_args={"policy.x --evil": "1"}))
    with pytest.raises(ValueError):
        manager.submit(TrainRequest(dataset_repo_id="me/demo", policy_type="act", steps=10, runner="nope"))


# -- local runner ---------------------------------------------------------------------------
@pytest.fixture
def datasets_root(tmp_path):
    root = tmp_path / "datasets"
    write_dataset_info(root, "me/demo", "nexarm_follower")
    return root


def test_train_command_uses_the_curated_episode_list(datasets_root, tmp_path):
    app = create_app(tmp_path / "home", datasets_root=datasets_root)
    store = app.state.dataset_store
    store.set_excluded("me/demo", [1, 3])
    manager, _ = make_manager(tmp_path / "m")
    job = manager.submit(request(extra_args={"policy.chunk_size": "50"}))
    command = local_runner.build_train_command(job, store, tmp_path / "out")
    assert "--dataset.episodes=[0,2]" in command
    assert "--policy.chunk_size=50" in command
    assert "--policy.push_to_hub=false" in command and "--steps=1000" in command
    assert f"--dataset.root={datasets_root / 'me' / 'demo'}" in command
    store.set_excluded("me/demo", [0, 1, 2, 3])
    with pytest.raises(ValueError, match="every episode is excluded"):
        local_runner.build_train_command(job, store, tmp_path / "out")


def run_fake_trainer(monkeypatch, tmp_path, script):
    app = create_app(tmp_path / "home", datasets_root=tmp_path / "datasets")
    monkeypatch.setattr(
        local_runner, "build_train_command", lambda job, store, out: [sys.executable, "-u", "-c", script]
    )
    manager = JobManager(
        tmp_path / "jm", EventHub(), {"local": local_runner.LocalRunner(app.state.dataset_store)}
    )
    return manager


def test_local_runner_streams_logs_and_completes(monkeypatch, tmp_path):
    script = f"import sys\nfor i in range(3):\n    print({LOG_LINE!r}.replace('0.842', str(i)), flush=True)\n"
    manager = run_fake_trainer(monkeypatch, tmp_path, script)
    job = manager.submit(request())
    assert wait_for(lambda: manager.get(job.id).state == JobState.done), manager.get(job.id)
    assert manager.get(job.id).step == 600
    assert [m.values["loss"] for m in manager.metrics(job.id)] == [0.0, 1.0, 2.0]


def test_local_runner_reports_a_crash(monkeypatch, tmp_path):
    manager = run_fake_trainer(monkeypatch, tmp_path, "import sys\nprint('boom')\nsys.exit(3)")
    job = manager.submit(request())
    assert wait_for(lambda: manager.get(job.id).state == JobState.failed)
    assert "code 3" in manager.get(job.id).error
    assert "boom" in manager.logs(job.id)


def test_local_runner_cancel_terminates_the_process(monkeypatch, tmp_path):
    manager = run_fake_trainer(
        monkeypatch, tmp_path, "import time\nprint('start', flush=True)\ntime.sleep(60)"
    )
    job = manager.submit(request())
    assert wait_for(lambda: "start" in manager.logs(job.id))
    manager.cancel(job.id)
    assert wait_for(lambda: manager.get(job.id).state == JobState.cancelled)
    manager.shutdown()


# -- HTTP surface ---------------------------------------------------------------------------
def test_jobs_api_submit_list_and_not_found(datasets_root, tmp_path, monkeypatch):
    app = create_app(tmp_path / "home", datasets_root=datasets_root)
    runner = ScriptedRunner()
    app.state.jobs._runners["local"] = runner
    with TestClient(app) as client:
        body = {"dataset_repo_id": "me/demo", "policy_type": "act", "steps": 400, "log_freq": 200}
        job = client.post(f"{API_PREFIX}/jobs", json=body)
        assert job.status_code == 201
        job_id = job.json()["id"]
        runner.sinks[job_id].line(LOG_LINE)
        assert client.get(f"{API_PREFIX}/jobs/{job_id}").json()["step"] == 200
        assert client.get(f"{API_PREFIX}/jobs/{job_id}/metrics").json()[0]["values"]["loss"] == 0.842
        assert client.get(f"{API_PREFIX}/jobs/{job_id}/logs").json() == [LOG_LINE]
        assert client.get(f"{API_PREFIX}/jobs/{job_id}/checkpoints").json() == []
        assert [j["id"] for j in client.get(f"{API_PREFIX}/jobs").json()] == [job_id]
        assert client.delete(f"{API_PREFIX}/jobs/{job_id}").json()["state"] == "cancelled"
        assert client.get(f"{API_PREFIX}/jobs/nope").status_code == 404
        missing = client.post(f"{API_PREFIX}/jobs", json={**body, "dataset_repo_id": "me/ghost"})
        assert missing.status_code == 404
        assert client.post(f"{API_PREFIX}/jobs", json={**body, "runner": "x"}).status_code == 422


def test_job_checkpoints_and_models_listing(datasets_root, tmp_path):
    app = create_app(tmp_path / "home", datasets_root=datasets_root)
    app.state.jobs._runners["local"] = ScriptedRunner()
    with TestClient(app) as client:
        job_id = client.post(
            f"{API_PREFIX}/jobs",
            json={"dataset_repo_id": "me/demo", "policy_type": "act", "steps": 400},
        ).json()["id"]
        out = app.state.jobs.output_dir(job_id)
        write_checkpoint(out, "000200", "me/demo")
        write_checkpoint(out, "000400", "me/demo")
        checkpoints = client.get(f"{API_PREFIX}/jobs/{job_id}/checkpoints").json()
        assert [c["step"] for c in checkpoints] == [200, 400]
        models = client.get(f"{API_PREFIX}/models").json()
        assert [m["step"] for m in models] == [200, 400]
        assert models[0]["policy_type"] == "act" and models[0]["dataset_repo_id"] == "me/demo"


# -- checkpoint provenance ------------------------------------------------------------------
def test_checkpoint_guard_refuses_other_robot_types_and_unknown_provenance(datasets_root, tmp_path):
    write_dataset_info(datasets_root, "me/arm100", "so100_follower")
    app = create_app(tmp_path / "home", datasets_root=datasets_root)
    models = app.state.models
    ours = write_checkpoint(tmp_path / "a", "000100", "me/demo")
    other = write_checkpoint(tmp_path / "b", "000100", "me/arm100")
    unknown = write_checkpoint(tmp_path / "c", "000100", None)

    assert models.check_robot_type(str(ours)) == "nexarm_follower"
    with pytest.raises(ValueError, match="so100_follower"):
        models.check_robot_type(str(other))
    with pytest.raises(ValueError, match="cannot verify"):
        models.check_robot_type(str(unknown))
    assert models.check_robot_type(str(unknown), allow_unverified=True) == "unverified"
    with pytest.raises(ValueError):
        models.check_robot_type(str(other), allow_unverified=True)


# -- rollout and eval sessions --------------------------------------------------------------
class FakeStrategy:
    def __init__(self, api):
        self.api = api

    def setup(self, ctx):
        self.api.calls.append("setup")

    def run(self, ctx):
        robot = ctx.hardware.robot_wrapper
        while not ctx.shutdown.is_set():
            robot.get_observation()
            robot.send_action({f"{n}.pos": 99999.0 for n in JOINT_NAMES})
            time.sleep(0.01)

    def teardown(self, ctx):
        # the follower must already be back at rest with torque off when the library tears down
        bus = ctx.hardware.robot_wrapper.inner.bus
        self.api.calls.append(("teardown", list(bus.positions), list(bus.torque_calls)))
        bus.exit_lerobot_mode()


class FakeRolloutApi:
    def __init__(self, hardware):
        self.hardware = hardware
        self.calls = []

    def make_config(self, kind, profile, checkpoint, params, datasets_root):
        self.calls.append(("make_config", kind, checkpoint))
        return types.SimpleNamespace(kind=kind)

    def build_context(self, cfg, shutdown):
        self.calls.append("build_context")
        self.hardware.follower_bus.connect()
        self.hardware.follower_bus.set_torque(True)
        # mimic ThreadSafeRobot: it calls through to the (patched) inner robot
        wrapper = types.SimpleNamespace(
            inner=self.hardware.follower,
            send_action=lambda a: self.hardware.follower.send_action(a),
            get_observation=lambda: self.hardware.follower.get_observation(),
        )
        return types.SimpleNamespace(
            hardware=types.SimpleNamespace(robot_wrapper=wrapper, teleop=None), shutdown=shutdown
        )

    def create_strategy(self, cfg):
        return FakeStrategy(self)


@pytest.fixture
def deploy_env(datasets_root, tmp_path):
    hw = FakeHardware(follower_pos=REST)
    api = FakeRolloutApi(hw)
    app = create_app(tmp_path / "home", hardware=hw, datasets_root=datasets_root, rollout_api=api)
    ckpt = write_checkpoint(tmp_path / "job", "000100", "me/demo")
    with TestClient(app) as client:
        client.put(f"{API_PREFIX}/profiles/bench", json=PROFILE)
        yield app, client, hw, api, ckpt


def test_rollout_clamps_streams_and_returns_to_rest_before_teardown(deploy_env):
    app, client, hw, api, ckpt = deploy_env
    resp = client.post(
        f"{API_PREFIX}/sessions",
        json={"kind": "rollout", "profile": "bench", "params": {"checkpoint": str(ckpt)}},
    )
    assert resp.status_code == 201, resp.text
    sid = resp.json()["id"]
    assert wait_for(lambda: client.get(f"{API_PREFIX}/sessions/current").json()["status"]["ticks"] >= 3)
    assert max(hw.follower.sent[-1][f"{n}.pos"] for n in JOINT_NAMES) <= 4095
    assert hw.follower.sent[-1]["gripper.pos"] == GRIPPER_CLOSED_POS  # clamped at the soft limit

    assert client.delete(f"{API_PREFIX}/sessions/{sid}").json()["state"] == "stopped"
    teardown = next(c for c in api.calls if isinstance(c, tuple) and c[0] == "teardown")
    assert teardown[1] == REST  # back at rest first
    assert teardown[2][-1] is False  # torque already off
    assert api.calls.index("setup") < api.calls.index(teardown)


def test_rollout_refuses_a_foreign_checkpoint_before_touching_hardware(deploy_env, tmp_path, datasets_root):
    app, client, hw, api, ckpt = deploy_env
    write_dataset_info(datasets_root, "me/arm100", "so100_follower")
    foreign = write_checkpoint(tmp_path / "foreign", "000100", "me/arm100")
    resp = client.post(
        f"{API_PREFIX}/sessions",
        json={"kind": "rollout", "profile": "bench", "params": {"checkpoint": str(foreign)}},
    )
    assert resp.status_code == 422 and "so100_follower" in resp.json()["detail"]
    assert "build_context" not in api.calls
    assert hw.follower_bus.torque_calls == []


def test_eval_and_rollout_need_a_checkpoint_and_cameras(deploy_env):
    app, client, hw, api, ckpt = deploy_env
    none = client.post(f"{API_PREFIX}/sessions", json={"kind": "eval", "profile": "bench", "params": {}})
    assert none.status_code == 422
    client.put(f"{API_PREFIX}/profiles/nocam", json={**PROFILE, "name": "nocam", "cameras": {}})
    nocam = client.post(
        f"{API_PREFIX}/sessions",
        json={"kind": "rollout", "profile": "nocam", "params": {"checkpoint": str(ckpt)}},
    )
    assert nocam.status_code == 422 and "cameras" in nocam.json()["detail"]


def test_a_failed_strategy_ends_the_session_and_still_releases_the_arm(deploy_env, monkeypatch):
    app, client, hw, api, ckpt = deploy_env

    class Crashing(FakeStrategy):
        def run(self, ctx):
            raise RuntimeError("policy exploded")

    monkeypatch.setattr(api, "create_strategy", lambda cfg: Crashing(api))
    client.post(
        f"{API_PREFIX}/sessions",
        json={"kind": "rollout", "profile": "bench", "params": {"checkpoint": str(ckpt)}},
    )
    assert wait_for(lambda: not app.state.sessions._live.worker.alive())
    app.state.sessions.tick()
    last = app.state.sessions.last()
    assert last.state == "failed" and "policy exploded" in last.error
    assert hw.follower_bus.positions == REST
    assert hw.follower_bus.torque_calls[-1] is False


def test_eval_controls_set_the_strategy_event_flags(deploy_env, monkeypatch):
    app, client, hw, api, ckpt = deploy_env

    class Eventful(FakeStrategy):
        def __init__(self, api):
            super().__init__(api)
            self._events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    strategy = Eventful(api)
    monkeypatch.setattr(api, "create_strategy", lambda cfg: strategy)
    sid = client.post(
        f"{API_PREFIX}/sessions",
        json={"kind": "eval", "profile": "bench", "params": {"checkpoint": str(ckpt)}},
    ).json()["id"]

    def control(action):
        return client.post(f"{API_PREFIX}/sessions/{sid}/control", json={"action": action})

    first = control("exit_early")
    assert first.status_code == 200, first.text
    assert strategy._events["exit_early"]
    assert control("rerecord").status_code == 200 and strategy._events["rerecord_episode"]
    assert control("finish").status_code == 200 and strategy._events["stop_recording"]
    assert control("takeover").status_code == 422
    client.delete(f"{API_PREFIX}/sessions/{sid}")


def test_a_rollout_without_strategy_events_rejects_controls(deploy_env):
    app, client, hw, api, ckpt = deploy_env
    sid = client.post(
        f"{API_PREFIX}/sessions",
        json={"kind": "rollout", "profile": "bench", "params": {"checkpoint": str(ckpt)}},
    ).json()["id"]
    resp = client.post(f"{API_PREFIX}/sessions/{sid}/control", json={"action": "exit_early"})
    assert resp.status_code == 422
    client.delete(f"{API_PREFIX}/sessions/{sid}")


def test_the_engine_never_installs_global_input_listeners():
    from lerobot.rollout.strategies import dagger, episodic
    from nexarm_service.rollout import disable_input_listeners

    disable_input_listeners()
    listener, events = episodic.init_keyboard_listener()
    assert listener is None and set(events) == {"exit_early", "rerecord_episode", "stop_recording"}
    assert dagger._init_dagger_keyboard(None, None) is None
    assert dagger._init_dagger_pedal(None, None) is None
