# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Coaching (DAgger) session: handover controls, leader follow and release, discard, reset and finish.

The strategy loop is a small fake that uses the library's real ``DAggerEvents`` and
``DAggerStrategy._apply_transition`` (on a real, never-started strategy), so the phase machine and its leader side effects are the library's.
"""

import time
import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="install lerobot[service]")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from fastapi.testclient import TestClient  # noqa: E402

from lerobot.rollout.configs import DAggerStrategyConfig  # noqa: E402
from lerobot.rollout.strategies.dagger import DAggerEvents, DAggerPhase, DAggerStrategy  # noqa: E402
from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES, map_follower_to_leader  # noqa: E402
from nexarm_service.app import API_PREFIX, create_app  # noqa: E402
from tests.nexarm_service.fakes import FakeBus, FakeFollower  # noqa: E402

REST = [2000, 2100, 1900, 2050, 2048, 2000]
LEADER_REST = [1990, 2050, 2010, 2040, 2048, 2040]
PROFILE = {
    "name": "bench",
    "follower_port": "FOLLOWER",
    "leader_port": "LEADER",
    "cameras": {"front": {"source": "0", "width": 64, "height": 48, "fps": 10}},
}


def wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class FakeLeaderArm:
    """The slice of NexArmLeader that DAgger and the worker use."""

    feedback_features = {f"{n}.pos": float for n in JOINT_NAMES}

    def __init__(self, bus):
        self.bus = bus
        self.is_connected = True
        self.feedback = []

    def enable_torque(self):
        self.bus.set_torque(True)

    def disable_torque(self):
        self.bus.set_torque(False)

    def get_action(self):
        return {f"{n}.pos": float(2300 + i) for i, n in enumerate(JOINT_NAMES)}

    def send_feedback(self, action):
        self.feedback.append(dict(action))
        self.bus.write_positions(map_follower_to_leader([action[f"{n}.pos"] for n in JOINT_NAMES]))

    def disconnect(self):
        self.is_connected = False
        self.bus.disconnect()


class FakeDataset:
    def __init__(self):
        self.frames = 0
        self.saved = 0
        self.cleared = 0

    def add_frame(self, frame):
        self.frames += 1

    def save_episode(self):
        self.saved += 1

    def clear_episode_buffer(self):
        self.cleared += 1


class FakeDagger:
    """A control loop shaped like the library's corrections-only mode, around the real phase machine."""

    def __init__(self, api):
        self.api = api
        self._events = DAggerEvents()
        self.config = types.SimpleNamespace(num_episodes=3)
        self.engine, self.interpolator = MagicMock(), MagicMock()
        # Owns only the library's transition side effects; its own loop never runs.
        self._library = DAggerStrategy(DAggerStrategyConfig(num_episodes=3))
        self.sent_log = []

    def setup(self, ctx):
        pass

    def run(self, ctx):
        robot, teleop, dataset = ctx.hardware.robot_wrapper, ctx.hardware.teleop, ctx.data.dataset
        events, last_action = self._events, None
        while not ctx.shutdown.is_set() and not events.stop_recording.is_set():
            transition = events.consume_transition()
            if transition is not None:
                old, new = transition
                self._library._apply_transition(
                    old, new, self.engine, self.interpolator, teleop, ctx, last_action
                )
                if new == DAggerPhase.AUTONOMOUS:
                    last_action = None
                if old == DAggerPhase.CORRECTING and new == DAggerPhase.PAUSED:
                    dataset.save_episode()
            phase = events.phase
            if phase == DAggerPhase.CORRECTING:
                last_action = teleop.get_action()
                robot.send_action(last_action)
                dataset.add_frame({})
            elif phase == DAggerPhase.PAUSED:
                if last_action:
                    robot.send_action(last_action)
            else:
                last_action = {f"{n}.pos": float(2100 + i) for i, n in enumerate(JOINT_NAMES)}
                robot.send_action(last_action)
            time.sleep(0.005)

    def teardown(self, ctx):
        self.api.calls.append("teardown")


class FakeCoachApi:
    def __init__(self, hardware_bus, with_leader=True):
        self.follower_bus = hardware_bus
        self.follower = FakeFollower(hardware_bus, with_camera=True)
        self.leader_bus = FakeBus(LEADER_REST)
        self.leader = FakeLeaderArm(self.leader_bus) if with_leader else None
        self.dataset = FakeDataset()
        self.strategy = None
        self.calls = []

    def make_config(self, kind, profile, checkpoint, params, datasets_root):
        self.calls.append(("make_config", kind))
        return types.SimpleNamespace(kind=kind)

    def build_context(self, cfg, shutdown):
        self.follower_bus.connect()
        self.follower_bus.set_torque(True)
        self.leader_bus.connect()
        wrapper = types.SimpleNamespace(
            inner=self.follower,
            send_action=lambda a: self.follower.send_action(a),
            get_observation=lambda: self.follower.get_observation(),
            supports_hold=False,
        )
        return types.SimpleNamespace(
            hardware=types.SimpleNamespace(robot_wrapper=wrapper, teleop=self.leader),
            data=types.SimpleNamespace(dataset=self.dataset),
            runtime=types.SimpleNamespace(cfg=types.SimpleNamespace(play_sounds=False)),
            shutdown=shutdown,
        )

    def create_strategy(self, cfg):
        self.strategy = FakeDagger(self)
        return self.strategy


@pytest.fixture(autouse=True)
def fast_handover(monkeypatch):
    """The library's smooth handover sleeps 2 s; make it instant."""
    monkeypatch.setattr("lerobot.common.control_utils.time.sleep", lambda _: None)


@pytest.fixture
def coach(tmp_path):
    bus = FakeBus(REST)
    api = FakeCoachApi(bus)
    from tests.nexarm_service.fakes import FakeHardware

    hw = FakeHardware(follower_pos=REST)
    hw.follower_bus = bus
    app = create_app(tmp_path / "home", hardware=hw, datasets_root=tmp_path / "datasets", rollout_api=api)
    app.state.models.check_robot_type = lambda *a, **k: "nexarm_follower"
    with TestClient(app) as client:
        client.put(f"{API_PREFIX}/profiles/bench", json=PROFILE)
        started = client.post(
            f"{API_PREFIX}/sessions",
            json={
                "kind": "coaching",
                "profile": "bench",
                "params": {"checkpoint": "ckpt", "repo_id": "me/coach", "task": "t"},
            },
        )
        assert started.status_code == 201, started.text
        sid = started.json()["id"]

        def control(action, **payload):
            return client.post(
                f"{API_PREFIX}/sessions/{sid}/control", json={"action": action, "payload": payload}
            )

        def status():
            return client.get(f"{API_PREFIX}/sessions/current").json()["status"]

        yield types.SimpleNamespace(
            app=app, client=client, api=api, sid=sid, control=control, status=status, bus=bus
        )


def test_takeover_hands_the_arm_to_the_operator_and_records_the_correction(coach):
    assert coach.status()["phase"] == "autonomous"
    assert coach.control("takeover").status_code == 200
    assert wait_for(lambda: coach.status()["phase"] == "correcting")
    assert coach.status()["intervening"] is True
    assert wait_for(lambda: coach.api.dataset.frames >= 3)
    assert coach.api.leader_bus.torque_calls[-1] is False  # free for the operator to drag
    # the leader was driven to the follower's pose before the handover (no jerk)
    assert coach.api.leader.feedback, "the leader followed the policy before pausing"


def test_handback_returns_to_the_policy_and_saves_the_correction(coach):
    coach.control("takeover")
    assert wait_for(lambda: coach.status()["phase"] == "correcting")
    assert coach.control("handback").status_code == 200
    assert wait_for(lambda: coach.status()["phase"] == "autonomous")
    assert wait_for(lambda: coach.status()["corrections"] == 1)
    assert coach.api.dataset.saved == 1


def test_hold_and_resume(coach):
    assert coach.control("hold").status_code == 200
    assert wait_for(lambda: coach.status()["phase"] == "paused")
    assert coach.control("resume").status_code == 200
    assert wait_for(lambda: coach.status()["phase"] == "autonomous")


def test_impossible_controls_are_refused_with_a_reason(coach):
    resp = coach.control("handback")
    assert resp.status_code == 422 and "autonomous" in resp.json()["detail"]
    assert coach.control("resume").status_code == 422
    assert coach.control("drop_last").status_code == 422
    assert coach.control("nonsense").status_code == 422
    coach.control("takeover")
    assert wait_for(lambda: coach.status()["phase"] == "correcting")
    assert coach.control("takeover").status_code == 422
    assert coach.control("reset").status_code == 422


def test_drop_last_discards_the_correction_in_progress_and_gives_the_slot_back(coach):
    coach.control("takeover")
    assert wait_for(lambda: coach.api.dataset.frames >= 2)
    assert coach.control("drop_last").status_code == 200
    assert wait_for(lambda: coach.status()["discarded"] == 1)
    assert coach.api.dataset.cleared == 1 and coach.api.dataset.saved == 0
    assert coach.status()["corrections"] == 0
    assert coach.api.strategy.config.num_episodes == 4
    assert wait_for(lambda: coach.status()["phase"] == "paused")


def test_reset_holds_the_follower_at_rest_until_the_next_handover(coach):
    coach.control("hold")
    assert wait_for(lambda: coach.status()["phase"] == "paused")
    assert coach.control("reset").status_code == 200
    assert wait_for(lambda: coach.status()["hold"] is True)
    rest = {f"{n}.pos": float(REST[i]) for i, n in enumerate(JOINT_NAMES)}
    assert wait_for(lambda: coach.api.follower.sent and coach.api.follower.sent[-1] == rest)
    assert coach.api.leader.feedback[-1] == rest  # the leader moved to match
    coach.control("resume")
    assert wait_for(lambda: coach.status()["phase"] == "autonomous" and coach.status()["hold"] is False)


def test_finish_ends_the_loop(coach):
    assert coach.control("finish").status_code == 200
    assert wait_for(lambda: not coach.app.state.sessions._live.worker.alive())
    coach.app.state.sessions.tick()
    assert coach.app.state.sessions.last().state == "stopped"


def test_stopping_with_the_leader_holding_returns_it_to_where_it_started_then_frees_it(coach):
    coach.control("hold")  # the library keeps leader torque on while paused
    assert wait_for(lambda: coach.status()["phase"] == "paused")
    assert coach.api.leader_bus.torque_calls[-1] is True
    assert coach.client.delete(f"{API_PREFIX}/sessions/{coach.sid}").json()["state"] == "stopped"
    assert coach.api.leader_bus.positions == LEADER_REST
    assert coach.api.leader_bus.torque_calls[-1] is False
    assert coach.bus.positions == REST and coach.bus.torque_calls[-1] is False


def test_coaching_state_events_reach_the_desktop(coach):
    with coach.client.websocket_connect(f"{API_PREFIX}/ws/events") as ws:
        ws.receive_json()
        coach.control("takeover")
        seen = []
        for _ in range(400):
            msg = ws.receive_json()
            if msg["type"] == "coaching_state":
                seen.append(msg["data"])
                if msg["data"]["intervening"]:
                    break
        assert seen[-1]["phase"] == "correcting" and seen[-1]["intervening"] is True


def test_coaching_needs_a_leader_and_releases_the_arm_when_it_has_none(tmp_path):
    bus = FakeBus(REST)
    api = FakeCoachApi(bus, with_leader=False)
    from tests.nexarm_service.fakes import FakeHardware

    hw = FakeHardware(follower_pos=REST)
    app = create_app(tmp_path / "home", hardware=hw, datasets_root=tmp_path / "d", rollout_api=api)
    app.state.models.check_robot_type = lambda *a, **k: "nexarm_follower"
    with TestClient(app) as client:
        client.put(f"{API_PREFIX}/profiles/bench", json=PROFILE)
        resp = client.post(
            f"{API_PREFIX}/sessions",
            json={
                "kind": "coaching",
                "profile": "bench",
                "params": {"checkpoint": "c", "repo_id": "me/x", "task": "t"},
            },
        )
        assert resp.status_code == 422 and "leader" in resp.json()["detail"]
        assert not bus.is_connected or bus.torque_calls[-1] is False
        assert client.get(f"{API_PREFIX}/sessions/current").json() is None
