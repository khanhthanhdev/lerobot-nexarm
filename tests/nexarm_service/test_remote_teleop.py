# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Remote teleoperation: token auth, the station's safety behaviour, and operator-to-station end to end."""

import time

import httpx
import pytest

pytest.importorskip("fastapi", reason="install lerobot[service]")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from lerobot.motors.nexarm.nexarm import JOINT_NAMES  # noqa: E402
from lerobot.nexarm_service import auth, remote_teleop  # noqa: E402
from lerobot.nexarm_service.app import API_PREFIX, create_app  # noqa: E402
from lerobot.nexarm_service.safety import SOFT_LIMITS  # noqa: E402
from tests.nexarm_service.fakes import FakeHardware  # noqa: E402
from tests.nexarm_service.live import LiveEngine, wait_for  # noqa: E402

REST = [2000, 2100, 1900, 2050, 2048, 2000]
PROFILE = {"name": "bench", "follower_port": "FOLLOWER", "leader_port": "LEADER"}
HELLO = {"type": "hello", "api_version": "1", "joints": remote_teleop.JOINT_KEYS, "operator": "alice"}
WS = f"{API_PREFIX}/ws/remote"


def action(offset=0.0, **over):
    values = {f"{n}.pos": float(REST[i] + offset) for i, n in enumerate(JOINT_NAMES)}
    values.update(over)
    return values


# -- auth -----------------------------------------------------------------------------------
def test_loopback_binds_need_no_token_and_other_binds_refuse_to_start_without_one():
    assert auth.is_loopback("127.0.0.1") and auth.is_loopback("::1") and auth.is_loopback("localhost")
    assert not auth.is_loopback("0.0.0.0") and not auth.is_loopback("100.64.0.7")
    auth.require_token_for_bind("127.0.0.1", None)
    auth.require_token_for_bind("0.0.0.0", "secret")
    with pytest.raises(SystemExit, match="--token"):
        auth.require_token_for_bind("0.0.0.0", None)


def test_token_protects_http_and_websocket_routes(tmp_path):
    app = create_app(tmp_path, token="s3cret")
    with TestClient(app) as client:
        assert client.get(f"{API_PREFIX}/health").status_code == 401
        assert client.get(f"{API_PREFIX}/health", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert (
            client.get(f"{API_PREFIX}/health", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        )
        # a token in the query string is accepted for WebSockets only
        assert client.get(f"{API_PREFIX}/health?token=s3cret").status_code == 401
        with (
            pytest.raises(WebSocketDisconnect) as refused,
            client.websocket_connect(f"{API_PREFIX}/ws/events"),
        ):
            pass
        assert refused.value.code == 4401
        with client.websocket_connect(f"{API_PREFIX}/ws/events?token=s3cret") as ws:
            assert ws.receive_json()["type"] == "hello"


def test_an_engine_without_a_token_stays_open_on_loopback(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(f"{API_PREFIX}/health").status_code == 200


# -- protocol -------------------------------------------------------------------------------
def test_station_urls_are_built_from_host_port_pairs():
    assert remote_teleop.ws_url("100.64.0.7:8765") == "ws://100.64.0.7:8765/api/v1/ws/remote"
    assert remote_teleop.ws_url("http://gpu:9000") == "ws://gpu:9000/api/v1/ws/remote"
    assert remote_teleop.ws_url("https://x.example").startswith("wss://x.example/")
    with pytest.raises(ValueError):
        remote_teleop.ws_url("ftp://nope")


@pytest.mark.parametrize(
    "bad",
    [
        {"seq": "1", "action": action()},
        {"seq": 1},
        {"seq": 1, "action": {**action(), "gripper.pos": float("nan")}},
        {"seq": 1, "action": {**action(), "gripper.pos": True}},
        {"seq": 1, "action": {k: v for k, v in action().items() if k != "wrist_roll.pos"}},
        {"seq": 1, "action": {**action(), "elbow_flex.pos": "2048"}},
    ],
)
def test_malformed_actions_are_rejected(bad):
    with pytest.raises(remote_teleop.ProtocolError):
        remote_teleop.parse_action({"type": "action", **bad})


def test_step_toward_caps_each_joint():
    assert remote_teleop.step_toward([0, 100, 500], [10, 0, 900], 20) == [10, 80, 520]
    assert remote_teleop.step_toward([5, 5], [5, 5], 0) == [5, 5]


# -- the station ----------------------------------------------------------------------------
@pytest.fixture
def station(tmp_path):
    hw = FakeHardware(follower_pos=REST)
    app = create_app(tmp_path / "home", hardware=hw, calibration_root=tmp_path / "cal")
    with TestClient(app) as client:
        client.put(f"{API_PREFIX}/profiles/bench", json=PROFILE)
        started = client.post(
            f"{API_PREFIX}/sessions",
            json={
                "kind": "hosting",
                "profile": "bench",
                "params": {"fps": 60, "rate_cap_ticks_s": 1200, "silence_s": 0.4, "soft_start_s": 0.2},
            },
        )
        assert started.status_code == 201, started.text
        yield app, client, hw, started.json()["id"]


def pos(hw):
    return list(hw.follower_bus.positions)


def handshake(ws, hello=HELLO):
    ws.send_json(hello)
    return ws.receive_json()


def test_the_follower_is_parked_at_rest_until_an_operator_engages(station):
    app, client, hw, sid = station
    time.sleep(0.2)
    assert pos(hw) == REST
    status = client.get(f"{API_PREFIX}/sessions/current").json()["status"]
    assert status["phase"] == "parked" and status["operator"] is None and status["accepting"] is True


def test_an_operator_drives_the_follower_through_a_rate_cap_and_the_soft_limits(station):
    app, client, hw, sid = station
    cap = 1200 / 60
    with client.websocket_connect(WS) as ws:
        ready = handshake(ws)
        assert ready["type"] == "ready" and ready["rest"]["shoulder_pan.pos"] == REST[0]
        target = action(300, **{"gripper.pos": 99999.0})  # far away, and a gripper beyond its limit
        for seq in range(1, 200):
            ws.send_json({"type": "action", "seq": seq, "action": target})
            time.sleep(0.01)
            if pos(hw)[0] >= REST[0] + 295:
                break
        writes = hw.follower_bus.writes
        steps = [
            max(abs(a - b) for a, b in zip(w1, w0, strict=True))
            for w0, w1 in zip(writes, writes[1:], strict=False)
        ]
        assert max(steps) <= cap + 1.01, max(steps)  # +1: positions are rounded to whole ticks
        assert pos(hw)[0] >= REST[0] + 290
        assert pos(hw)[5] <= SOFT_LIMITS["gripper"][1]  # clamped at the soft limit
        status = client.get(f"{API_PREFIX}/sessions/current").json()["status"]
        assert status["operator"] == "alice" and status["phase"] in ("engaging", "engaged")
        state = ws.receive_json()
        while state["type"] != "state":
            state = ws.receive_json()
        assert set(state["ticks"]) == set(remote_teleop.JOINT_KEYS)
    # the operator left: the follower parks back at rest
    assert wait_for(lambda: pos(hw) == REST, 10)
    assert wait_for(
        lambda: client.get(f"{API_PREFIX}/sessions/current").json()["status"]["phase"] == "parked", 10
    )


def test_silence_parks_the_follower_and_a_new_operator_is_welcome_after(station):
    app, client, hw, sid = station
    with client.websocket_connect(WS) as ws:
        handshake(ws)
        for seq in range(1, 60):
            ws.send_json({"type": "action", "seq": seq, "action": action(200)})
            time.sleep(0.01)
        assert pos(hw) != REST
        # no more frames: after silence_s the station parks without being told to
        assert wait_for(lambda: pos(hw) == REST, 10)
        status = client.get(f"{API_PREFIX}/sessions/current").json()["status"]
        assert status["operator"] is None


def test_a_second_operator_is_refused_while_one_is_driving(station):
    app, client, hw, sid = station
    with client.websocket_connect(WS) as first:
        assert handshake(first)["type"] == "ready"
        with client.websocket_connect(WS) as second:
            refusal = handshake(second, {**HELLO, "operator": "bob"})
            assert refusal["type"] == "refused" and "alice" in refusal["reason"]


def test_incompatible_operators_and_bad_frames_are_refused(station):
    app, client, hw, sid = station
    with client.websocket_connect(WS) as ws:
        assert handshake(ws, {**HELLO, "api_version": "9"})["type"] == "refused"
    with client.websocket_connect(WS) as ws:
        assert handshake(ws, {**HELLO, "joints": ["a", "b"]})["type"] == "refused"
    with client.websocket_connect(WS) as ws:
        handshake(ws)
        ws.send_json({"type": "action", "seq": 1, "action": {"shoulder_pan.pos": 1}})
        assert ws.receive_json()["type"] in ("refused", "state")
    assert wait_for(
        lambda: client.get(f"{API_PREFIX}/sessions/current").json()["status"]["operator"] is None, 5
    )


def test_block_allow_and_park_controls(station):
    app, client, hw, sid = station

    def control(name):
        return client.post(f"{API_PREFIX}/sessions/{sid}/control", json={"action": name})

    assert control("block").json()["accepting"] is False
    with client.websocket_connect(WS) as ws:
        refusal = handshake(ws)
        assert refusal["type"] == "refused" and "not accepting" in refusal["reason"]
    assert control("allow").json()["accepting"] is True
    with client.websocket_connect(WS) as ws:
        assert handshake(ws)["type"] == "ready"
        for seq in range(1, 30):
            ws.send_json({"type": "action", "seq": seq, "action": action(150)})
            time.sleep(0.01)
        assert control("park").json()["operator"] is None
        assert wait_for(lambda: pos(hw) == REST, 10)


def test_stopping_the_station_releases_the_arm_at_rest(station):
    app, client, hw, sid = station
    with client.websocket_connect(WS) as ws:
        handshake(ws)
        for seq in range(1, 30):
            ws.send_json({"type": "action", "seq": seq, "action": action(250)})
            time.sleep(0.01)
        assert client.delete(f"{API_PREFIX}/sessions/{sid}").json()["state"] == "stopped"
    assert pos(hw) == REST
    assert hw.follower_bus.torque_calls[-1] is False


def test_an_engine_that_is_not_hosting_refuses_operators(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client, client.websocket_connect(WS) as ws:
        assert ws.receive_json()["type"] == "refused"


# -- operator and station, end to end over real sockets -------------------------------------
@pytest.fixture
def pair(tmp_path):
    station_hw = FakeHardware(follower_pos=REST)
    station_app = create_app(
        tmp_path / "s-home", hardware=station_hw, calibration_root=tmp_path / "s-cal", token="tok"
    )
    with LiveEngine(station_app) as station:
        headers = {"Authorization": "Bearer tok"}
        with httpx.Client(base_url=station.url + API_PREFIX, headers=headers) as http:
            http.put("/profiles/bench", json=PROFILE)
            started = http.post(
                "/sessions",
                json={
                    "kind": "hosting",
                    "profile": "bench",
                    "params": {"fps": 60, "rate_cap_ticks_s": 1800, "soft_start_s": 0.2},
                },
            )
            assert started.status_code == 201, started.text
            leader_action = action(180)
            operator_hw = FakeHardware(leader_action=leader_action)
            operator_app = create_app(
                tmp_path / "o-home", hardware=operator_hw, calibration_root=tmp_path / "o-cal"
            )
            with TestClient(operator_app) as operator:
                operator.put(f"{API_PREFIX}/profiles/bench", json=PROFILE)
                yield station, station_hw, operator, operator_hw, http


def start_operator(operator, station, token="tok"):
    return operator.post(
        f"{API_PREFIX}/sessions",
        json={
            "kind": "remote_teleoperation",
            "profile": "bench",
            "params": {"station": station.url.replace("http://", ""), "token": token, "fps": 60},
        },
    )


def test_the_operators_leader_drives_the_station_follower_and_parks_it_on_stop(pair):
    station, station_hw, operator, operator_hw, http = pair
    started = start_operator(operator, station)
    assert started.status_code == 201, started.text
    sid = started.json()["id"]
    assert wait_for(lambda: pos(station_hw)[0] >= REST[0] + 175, 20), pos(station_hw)

    def status():
        return operator.get(f"{API_PREFIX}/sessions/current").json()["status"]

    assert wait_for(lambda: status()["rtt_ms"] is not None and status()["station_phase"] == "engaged", 20), (
        status()
    )
    assert 0 <= status()["rtt_ms"] < 2000
    assert set(status()["remote_ticks"]) == set(remote_teleop.JOINT_KEYS)
    assert operator_hw.leader.connected is True

    assert operator.delete(f"{API_PREFIX}/sessions/{sid}").json()["state"] == "stopped"
    assert operator_hw.leader.connected is False
    # the operator said bye, so the station parks right away
    assert wait_for(lambda: pos(station_hw) == REST, 20)
    assert http.get("/sessions/current").json()["status"]["operator"] is None


def test_a_wrong_token_or_a_dead_station_fails_the_start_cleanly(pair):
    station, station_hw, operator, operator_hw, http = pair
    wrong = start_operator(operator, station, token="nope")
    assert wrong.status_code == 502
    assert operator_hw.leader.connected is False  # the leader was released again
    dead = operator.post(
        f"{API_PREFIX}/sessions",
        json={
            "kind": "remote_teleoperation",
            "profile": "bench",
            "params": {"station": "127.0.0.1:1", "token": "tok"},
        },
    )
    assert dead.status_code == 502
    assert operator.get(f"{API_PREFIX}/sessions/current").json() is None


def test_the_operator_needs_a_station_address(tmp_path):
    app = create_app(tmp_path / "home", hardware=FakeHardware())
    with TestClient(app) as client:
        client.put(f"{API_PREFIX}/profiles/bench", json=PROFILE)
        resp = client.post(
            f"{API_PREFIX}/sessions", json={"kind": "remote_teleoperation", "profile": "bench", "params": {}}
        )
        assert resp.status_code == 422 and "station" in resp.json()["detail"]
