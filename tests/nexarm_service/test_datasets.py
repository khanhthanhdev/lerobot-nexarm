# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Tests for the dataset API (view, curation, edits) and the recording and replay sessions."""

import json
import shutil
import time

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="install lerobot[service]")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

import cv2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from nexarm_service.app import API_PREFIX, create_app  # noqa: E402
from tests.nexarm_service.dataset_factory import make_dataset  # noqa: E402
from tests.nexarm_service.fakes import FakeHardware  # noqa: E402

REPO = "me/demo"
PROFILE = {
    "name": "bench",
    "follower_port": "FOLLOWER",
    "leader_port": "LEADER",
    "cameras": {"front": {"source": "0", "width": 64, "height": 48, "fps": 10}},
}
REST = [2000, 2100, 1900, 2050, 2048, 2000]


@pytest.fixture(scope="session")
def template_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("datasets-template")
    make_dataset(root, REPO)
    return root


@pytest.fixture
def env(tmp_path, template_root):
    root = tmp_path / "datasets"
    shutil.copytree(template_root / "me", root / "me")
    hw = FakeHardware(follower_pos=REST)
    app = create_app(tmp_path / "home", hardware=hw, calibration_root=tmp_path / "cal", datasets_root=root)
    with TestClient(app) as client:
        client.put(f"{API_PREFIX}/profiles/bench", json=PROFILE)
        yield app, client, hw, root


def wait_for(predicate, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def run_operation(app, client, response):
    assert response.status_code == 202, response.text
    return app.state.operations.wait(response.json()["id"], timeout=120)


# -- view -----------------------------------------------------------------------------------
def test_list_info_and_episodes(env):
    app, client, hw, root = env
    listed = client.get(f"{API_PREFIX}/datasets").json()
    assert [d["repo_id"] for d in listed] == [REPO]
    assert listed[0]["total_episodes"] == 3 and listed[0]["robot_type"] == "nexarm_follower"
    assert listed[0]["cameras"] == ["observation.images.front"]

    body = client.get(f"{API_PREFIX}/datasets/{REPO}").json()
    assert body["info"]["codebase_version"] == "v3.0"
    episodes = client.get(f"{API_PREFIX}/datasets/{REPO}/episodes").json()
    assert [e["length"] for e in episodes] == [12, 12, 12]
    assert episodes[1]["tasks"] == ["task 1"]


def test_joints_for_one_episode(env):
    app, client, hw, root = env
    joints = client.get(f"{API_PREFIX}/datasets/{REPO}/episodes/1/joints").json()
    assert joints["names"][0] == "shoulder_pan.pos"
    assert len(joints["action"]) == 12 and len(joints["timestamps"]) == 12
    assert joints["action"][0] == [2058.0] * 6  # episode 1, frame 0
    assert joints["state"][3] == [2061.0] * 6


def test_frames_decode_to_jpeg_at_the_right_position(env):
    app, client, hw, root = env
    brightness = []
    for frame in (0, 6, 11):
        resp = client.get(f"{API_PREFIX}/datasets/{REPO}/episodes/1/frames/observation.images.front/{frame}")
        assert resp.status_code == 200 and resp.headers["content-type"] == "image/jpeg"
        image = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (48, 64, 3)
        brightness.append(float(image.mean()))
    # episode 1 frames brighten as (60 + 5 * i); lossy video keeps the ordering
    assert brightness[0] < brightness[1] < brightness[2]
    assert (
        client.get(f"{API_PREFIX}/datasets/{REPO}/episodes/1/frames/observation.images.front/99").status_code
        == 404
    )
    assert (
        client.get(f"{API_PREFIX}/datasets/{REPO}/episodes/1/frames/observation.state/0").status_code == 404
    )


def test_unknown_and_unsafe_dataset_names_are_404(env):
    app, client, hw, root = env
    assert client.get(f"{API_PREFIX}/datasets/me/nope").status_code == 404
    assert client.get(f"{API_PREFIX}/datasets/..%2Fme/demo").status_code in (404, 422)


# -- curation and edits ---------------------------------------------------------------------
def test_curation_excludes_without_deleting(env):
    app, client, hw, root = env
    assert client.put(f"{API_PREFIX}/datasets/{REPO}/excluded", json={"episodes": [2, 0, 2]}).json() == [0, 2]
    assert client.get(f"{API_PREFIX}/datasets/{REPO}/excluded").json() == [0, 2]
    episodes = client.get(f"{API_PREFIX}/datasets/{REPO}/episodes").json()
    assert [e["excluded"] for e in episodes] == [True, False, True]
    assert client.get(f"{API_PREFIX}/datasets").json()[0]["excluded_episodes"] == 2
    assert json.loads((root / REPO / "meta" / "info.json").read_text())["total_episodes"] == 3
    assert client.put(f"{API_PREFIX}/datasets/{REPO}/excluded", json={"episodes": [9]}).status_code == 422


def test_drop_excluded_writes_a_new_dataset_and_keeps_the_source(env):
    app, client, hw, root = env
    client.put(f"{API_PREFIX}/datasets/{REPO}/excluded", json={"episodes": [1]})
    op = run_operation(
        app,
        client,
        client.post(f"{API_PREFIX}/datasets/{REPO}/drop-excluded", json={"output_repo_id": "me/clean"}),
    )
    assert op.state == "done", op.error
    clean = client.get(f"{API_PREFIX}/datasets/me/clean").json()["summary"]
    assert clean["total_episodes"] == 2 and clean["total_frames"] == 24
    assert client.get(f"{API_PREFIX}/datasets/{REPO}").json()["summary"]["total_episodes"] == 3


def test_edit_refuses_an_existing_output_name(env):
    app, client, hw, root = env
    op = run_operation(
        app,
        client,
        client.post(
            f"{API_PREFIX}/datasets/{REPO}/delete-episodes", json={"episodes": [0], "output_repo_id": REPO}
        ),
    )
    assert op.state == "failed" and "already exists" in op.error


def test_split_by_fraction_merge_and_rename(env):
    app, client, hw, root = env
    op = run_operation(
        app,
        client,
        client.post(f"{API_PREFIX}/datasets/{REPO}/split", json={"splits": {"train": [0, 1], "val": [2]}}),
    )
    assert op.state == "done", op.error
    names = {d["repo_id"]: d["total_episodes"] for d in client.get(f"{API_PREFIX}/datasets").json()}
    assert names["me/demo_train"] + names["me/demo_val"] == 3

    merged = run_operation(
        app,
        client,
        client.post(
            f"{API_PREFIX}/datasets/merge",
            json={"repo_ids": ["me/demo_train", "me/demo_val"], "output_repo_id": "me/again"},
        ),
    )
    assert merged.state == "done", merged.error
    assert client.get(f"{API_PREFIX}/datasets/me/again").json()["summary"]["total_episodes"] == 3

    renamed = client.post(f"{API_PREFIX}/datasets/me/again/rename", json={"output_repo_id": "me/renamed"})
    assert renamed.status_code == 200
    assert client.get(f"{API_PREFIX}/datasets/me/again").status_code == 404
    assert (
        client.post(f"{API_PREFIX}/datasets/me/renamed/rename", json={"output_repo_id": REPO}).status_code
        == 409
    )


def test_add_and_remove_features(env):
    app, client, hw, root = env
    spec = {"dtype": "float32", "shape": [1], "names": None, "values": [[float(i)] for i in range(36)]}
    op = run_operation(
        app,
        client,
        client.post(
            f"{API_PREFIX}/datasets/{REPO}/modify-features",
            json={"output_repo_id": "me/rewarded", "add": {"reward": spec}},
        ),
    )
    assert op.state == "done", op.error
    assert "reward" in client.get(f"{API_PREFIX}/datasets/me/rewarded").json()["info"]["features"]
    short = {**spec, "values": [[0.0]]}
    bad = run_operation(
        app,
        client,
        client.post(
            f"{API_PREFIX}/datasets/{REPO}/modify-features",
            json={"output_repo_id": "me/bad", "add": {"reward": short}},
        ),
    )
    assert bad.state == "failed" and "needs 36 values" in bad.error


# -- recording ------------------------------------------------------------------------------
def start_recording(client, **params):
    body = {
        "repo_id": "me/new",
        "task": "stack the bowls",
        "fps": 10,
        "episode_time_s": 1,
        "reset_time_s": 0.2,
    }
    return client.post(
        f"{API_PREFIX}/sessions", json={"kind": "recording", "profile": "bench", "params": {**body, **params}}
    )


def test_recording_saves_episodes_that_match_lerobot_record_schema(env):
    app, client, hw, root = env
    resp = start_recording(client, num_episodes=2)
    assert resp.status_code == 201, resp.text
    assert wait_for(lambda: not app.state.sessions._live.worker.alive())
    app.state.sessions.tick()
    assert app.state.sessions.current() is None
    assert app.state.sessions.last().state == "stopped", app.state.sessions.last().error

    summary = client.get(f"{API_PREFIX}/datasets/me/new").json()
    assert summary["summary"]["total_episodes"] == 2
    assert summary["summary"]["robot_type"] == "nexarm_follower"
    features = summary["info"]["features"]
    assert features["action"]["names"] == features["observation.state"]["names"]
    assert features["action"]["names"][0] == "shoulder_pan.pos"
    assert features["observation.images.front"]["dtype"] == "video"
    assert hw.follower_bus.positions == REST
    assert hw.follower_bus.torque_calls[-1] is False


def test_recording_controls_pause_exit_early_and_task(env):
    app, client, hw, root = env
    sid = start_recording(client, num_episodes=1, episode_time_s=30).json()["id"]
    control = lambda action, **payload: client.post(  # noqa: E731
        f"{API_PREFIX}/sessions/{sid}/control", json={"action": action, "payload": payload}
    )
    status = lambda: client.get(f"{API_PREFIX}/sessions/current").json()["status"]  # noqa: E731
    assert wait_for(lambda: status()["frames"] >= 3)

    assert control("pause").status_code == 200
    assert wait_for(lambda: status()["phase"] == "paused")
    frozen = status()["frames"]
    time.sleep(0.4)
    assert status()["frames"] == frozen
    assert control("resume").status_code == 200
    assert wait_for(lambda: status()["frames"] > frozen)

    assert control("set_task", task="stack blue first").json()["task"] == "stack blue first"
    assert control("set_task", task="  ").status_code == 422
    assert control("nonsense").status_code == 422
    assert control("exit_early").status_code == 200
    assert wait_for(lambda: not app.state.sessions._live.worker.alive())
    app.state.sessions.tick()
    episodes = client.get(f"{API_PREFIX}/datasets/me/new/episodes").json()
    assert len(episodes) == 1 and 0 < episodes[0]["length"] < 300


def test_rerecord_discards_the_take(env):
    app, client, hw, root = env
    sid = start_recording(client, num_episodes=1, episode_time_s=30).json()["id"]
    status = lambda: client.get(f"{API_PREFIX}/sessions/current").json()["status"]  # noqa: E731
    assert wait_for(lambda: status()["frames"] >= 3)
    client.post(f"{API_PREFIX}/sessions/{sid}/control", json={"action": "rerecord"})
    assert wait_for(lambda: status()["episodes_recorded"] == 0 and status()["frames"] < 3)
    client.post(f"{API_PREFIX}/sessions/{sid}/control", json={"action": "finish"})
    assert wait_for(lambda: not app.state.sessions._live.worker.alive())
    app.state.sessions.tick()
    assert client.get(f"{API_PREFIX}/datasets/me/new").json()["summary"]["total_episodes"] == 1


def test_recording_stop_mid_episode_keeps_saved_episodes_only(env):
    app, client, hw, root = env
    sid = start_recording(client, num_episodes=3).json()["id"]
    status = lambda: client.get(f"{API_PREFIX}/sessions/current").json()["status"]  # noqa: E731
    assert wait_for(lambda: status()["episodes_recorded"] >= 1)
    assert client.delete(f"{API_PREFIX}/sessions/{sid}").json()["state"] == "stopped"
    total = client.get(f"{API_PREFIX}/datasets/me/new").json()["summary"]["total_episodes"]
    assert 1 <= total < 3
    assert hw.follower_bus.positions == REST


def test_recording_parameter_errors(env):
    app, client, hw, root = env
    assert start_recording(client, repo_id="me/eval_run").status_code == 422
    assert start_recording(client, repo_id=REPO).status_code == 422  # exists, resume not requested
    assert start_recording(client, repo_id="me/ghost", resume=True).status_code == 422
    no_cams = {**PROFILE, "name": "nocam", "cameras": {}}
    client.put(f"{API_PREFIX}/profiles/nocam", json=no_cams)
    resp = client.post(
        f"{API_PREFIX}/sessions",
        json={"kind": "recording", "profile": "nocam", "params": {"repo_id": "me/x", "task": "t"}},
    )
    assert resp.status_code == 422 and "cameras" in resp.json()["detail"]
    assert hw.follower_bus.writes == []


def test_recording_dataset_cannot_be_edited_while_recording(env):
    app, client, hw, root = env
    start_recording(client, num_episodes=3, episode_time_s=30)
    status = lambda: client.get(f"{API_PREFIX}/sessions/current").json()["status"]  # noqa: E731
    assert wait_for(lambda: status()["frames"] >= 1)
    assert client.post(
        f"{API_PREFIX}/datasets/me/new/rename", json={"output_repo_id": "me/moved"}
    ).status_code in (404, 409)


# -- replay ---------------------------------------------------------------------------------
def test_replay_ramps_to_the_opening_pose_then_plays_and_returns_to_rest(env):
    app, client, hw, root = env
    resp = client.post(
        f"{API_PREFIX}/sessions",
        json={"kind": "replay", "profile": "bench", "params": {"repo_id": REPO, "episode": 2}},
    )
    assert resp.status_code == 201, resp.text
    assert wait_for(lambda: not app.state.sessions._live.worker.alive())
    app.state.sessions.tick()
    assert app.state.sessions.last().state == "stopped", app.state.sessions.last().error
    sent = hw.follower.sent
    assert len(sent) == 12
    assert sent[0]["shoulder_pan.pos"] == 2068.0 and sent[-1]["shoulder_pan.pos"] == 2079.0
    # the first write before playback is a ramp step toward the opening pose, never a jump
    first_write = hw.follower_bus.writes[0]
    assert max(abs(a - b) for a, b in zip(first_write, REST, strict=True)) <= 40
    assert hw.follower_bus.positions == REST
    assert hw.follower_bus.torque_calls == [True, False]


def test_replay_rejects_a_missing_episode(env):
    app, client, hw, root = env
    resp = client.post(
        f"{API_PREFIX}/sessions",
        json={"kind": "replay", "profile": "bench", "params": {"repo_id": REPO, "episode": 99}},
    )
    assert resp.status_code in (404, 422, 502)
    assert hw.follower_bus.writes == []
