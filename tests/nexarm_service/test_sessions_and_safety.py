# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Hardware-free tests for sessions, the lease watchdog, rest return, soft limits and calibration policy."""

import json
import time

import pytest

pytest.importorskip("fastapi", reason="install lerobot[service]")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from fastapi.testclient import TestClient  # noqa: E402

from lerobot.motors.nexarm.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS, JOINT_NAMES  # noqa: E402
from lerobot.nexarm_service import calibration, safety  # noqa: E402
from lerobot.nexarm_service.app import API_PREFIX, create_app  # noqa: E402
from tests.nexarm_service.fakes import FakeBus, FakeHardware  # noqa: E402

PROFILE = {"name": "bench", "follower_port": "FOLLOWER", "leader_port": "LEADER"}
REST = [2000, 2100, 1900, 2050, 2048, 2000]


def make_client(tmp_path, hw, **kw):
    app = create_app(tmp_path / "home", hardware=hw, calibration_root=tmp_path / "cal", **kw)
    return app, TestClient(app)


def wait_for(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def start_teleop(client):
    client.put(f"{API_PREFIX}/profiles/bench", json=PROFILE)
    return client.post(f"{API_PREFIX}/sessions", json={"kind": "teleoperation", "profile": "bench"})


# -- safety helpers -------------------------------------------------------------------------
def test_soft_limits_come_from_urdf_and_gripper_range():
    assert safety.SOFT_LIMITS["gripper"] == (GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS)
    assert safety.SOFT_LIMITS["wrist_flex"] == (911, 3185)  # +-1.74533 rad, rounded inward
    assert safety.SOFT_LIMITS["wrist_roll"] == (1, 4095)


def test_clamp_action_limits_every_joint():
    out = safety.clamp_action({f"{n}.pos": 99999.0 for n in JOINT_NAMES})
    assert out["gripper.pos"] == GRIPPER_CLOSED_POS
    assert out["shoulder_pan.pos"] == safety.SOFT_LIMITS["shoulder_pan"][1]
    low = safety.clamp_action({f"{n}.pos": -5.0 for n in JOINT_NAMES})
    assert low["gripper.pos"] == GRIPPER_OPEN_POS


def test_capture_rest_skips_corrupt_reads_and_averages_a_steady_run():
    reads = iter([[0, 2048, 2048, 2048, 2048, 2048]] + [REST] * 10)
    assert safety.capture_rest(lambda: next(reads), sleep=lambda _: None) == REST


def test_capture_rest_refuses_a_noisy_arm():
    reads = iter([[2048 + 100 * (i % 2)] * 6 for i in range(100)])
    with pytest.raises(RuntimeError):
        safety.capture_rest(lambda: next(reads), attempts=20, sleep=lambda _: None)


def test_ramp_to_moves_in_bounded_steps():
    bus = FakeBus([2048] * 6)
    safety.ramp_to(bus.read_positions, bus.write_positions, [2548] * 6, max_step=100, sleep=lambda _: None)
    assert bus.positions == [2548] * 6
    assert len(bus.writes) >= 5
    steps = [
        max(abs(a - b) for a, b in zip(w0, w1, strict=True))
        for w0, w1 in zip(bus.writes, bus.writes[1:], strict=False)
    ]
    assert max(steps) <= 100


# -- calibration policy ---------------------------------------------------------------------
def test_non_identity_calibration_is_backed_up_then_reset(tmp_path):
    d = tmp_path / "teleoperators" / "nexarm_leader"
    d.mkdir(parents=True)
    offset = calibration.identity_calibration()
    offset["elbow_flex"]["homing_offset"] = 2100
    f = d / "leader.json"
    f.write_text(json.dumps(offset))

    assert calibration.reset_non_identity_files(tmp_path) == [f]
    assert calibration.is_identity(json.loads(f.read_text()))
    backups = list(d.glob("leader.json.bak-*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text())["elbow_flex"]["homing_offset"] == 2100
    assert calibration.reset_non_identity_files(tmp_path) == []  # idempotent


def test_calibration_check_reports_deltas_and_never_energises(tmp_path):
    hw = FakeHardware(leader_raw=[2048] * 6)
    # a Home leader gripper maps to 2833 (remapped), the follower elbow is 80 ticks off
    hw.buses["FOLLOWER"].positions = [2048, 2048, 2048 + 80, 2048, 2048, 2833]
    app, client = make_client(tmp_path, hw)
    client.put(f"{API_PREFIX}/profiles/bench", json=PROFILE)
    report = client.post(f"{API_PREFIX}/calibration/check", json={"profile": "bench"}).json()
    by_joint = {j["joint"]: j for j in report["joints"]}
    assert by_joint["elbow_flex"]["delta"] == 80
    assert not by_joint["elbow_flex"]["ok"]
    assert by_joint["shoulder_pan"]["ok"]
    assert report["ok"] is False
    assert "Servo Advanced" in report["hint"]
    assert hw.follower_bus.torque_calls == []
    assert hw.follower_bus.bridge_calls == [True, False]


# -- sessions -------------------------------------------------------------------------------
def test_teleop_clamps_actions_and_stop_returns_to_rest_before_torque_off(tmp_path):
    hw = FakeHardware(follower_pos=REST, leader_action={f"{n}.pos": 99999.0 for n in JOINT_NAMES})
    app, client = make_client(tmp_path, hw)
    with client:
        resp = start_teleop(client)
        assert resp.status_code == 201
        sid = resp.json()["id"]
        assert wait_for(lambda: len(hw.follower.sent) >= 3)
        assert all(a["gripper.pos"] <= GRIPPER_CLOSED_POS for a in hw.follower.sent)
        assert hw.follower_bus.positions != REST  # the arm moved during the session

        assert client.delete(f"{API_PREFIX}/sessions/{sid}").json()["state"] == "stopped"
    assert hw.follower_bus.positions == REST
    assert hw.follower_bus.torque_calls == [True, False]  # on at connect, off only after the return
    assert hw.follower_bus.bridge_calls[-1] is False
    assert client.get(f"{API_PREFIX}/sessions/current").json() is None


def test_second_session_is_rejected_while_one_is_live(tmp_path):
    app, client = make_client(tmp_path, FakeHardware(follower_pos=REST))
    with client:
        assert start_teleop(client).status_code == 201
        again = client.post(f"{API_PREFIX}/sessions", json={"kind": "teleoperation", "profile": "bench"})
        assert again.status_code == 409
        assert client.post(f"{API_PREFIX}/calibration/check", json={"profile": "bench"}).status_code == 409


def test_lease_expiry_safety_stops_the_session(tmp_path):
    hw = FakeHardware(follower_pos=REST, leader_action={f"{n}.pos": 2300.0 for n in JOINT_NAMES})
    app, client = make_client(tmp_path, hw, lease_timeout_s=0.2)
    with client:
        sid = start_teleop(client).json()["id"]
        time.sleep(0.4)
        app.state.sessions.tick()
        assert app.state.sessions.current() is None
        assert app.state.sessions.last().stop_reason == "lease_expired"
        assert hw.follower_bus.positions == REST
        assert client.post(f"{API_PREFIX}/sessions/{sid}/heartbeat").status_code == 404


def test_heartbeat_keeps_the_lease_alive(tmp_path):
    app, client = make_client(tmp_path, FakeHardware(follower_pos=REST), lease_timeout_s=0.5)
    with client:
        sid = start_teleop(client).json()["id"]
        for _ in range(4):
            time.sleep(0.2)
            assert client.post(f"{API_PREFIX}/sessions/{sid}/heartbeat").status_code == 200
            app.state.sessions.tick()
        assert app.state.sessions.current() is not None


def test_unsettled_arm_fails_the_start_and_releases_the_slot(tmp_path):
    hw = FakeHardware(follower_pos=[0] * 6)  # every read looks corrupt
    app, client = make_client(tmp_path, hw)
    with client:
        assert start_teleop(client).status_code == 502
        assert app.state.sessions.current() is None
    assert hw.follower_bus.writes == []  # never commanded a move


def test_session_events_end_with_idle(tmp_path):
    app, client = make_client(tmp_path, FakeHardware(follower_pos=REST))
    with client, client.websocket_connect(f"{API_PREFIX}/ws/events") as ws:
        ws.receive_json()  # hello
        sid = start_teleop(client).json()["id"]
        client.delete(f"{API_PREFIX}/sessions/{sid}")
        states = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "session_changed":
                states.append(msg["data"]["state"])
                if msg["data"]["state"] == "idle":
                    break
        assert states[0] == "starting"
        assert "running" in states
        assert states[-2:] == ["stopped", "idle"]


def test_unsupported_kind_is_422_and_unknown_profile_404(tmp_path):
    app, client = make_client(tmp_path, FakeHardware())
    with client:
        client.put(f"{API_PREFIX}/profiles/bench", json=PROFILE)
        recording = client.post(f"{API_PREFIX}/sessions", json={"kind": "recording", "profile": "bench"})
        assert recording.status_code == 422
        missing = client.post(f"{API_PREFIX}/sessions", json={"kind": "teleoperation", "profile": "nope"})
        assert missing.status_code == 404
