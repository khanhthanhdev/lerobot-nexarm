# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hardware-free tests for the NexArm engine skeleton (health, profiles, ports, events, contract)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="install lerobot[service]")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from fastapi.testclient import TestClient  # noqa: E402

from lerobot.nexarm_service import API_VERSION  # noqa: E402
from lerobot.nexarm_service.app import API_PREFIX, create_app, openapi_schema  # noqa: E402

OPENAPI_SNAPSHOT = Path(__file__).parents[2] / "docs" / "service" / "openapi.json"

PROFILE = {
    "name": "bench",
    "follower_port": "COM19",
    "leader_port": "COM20",
    "cameras": {"front": {"source": "0"}},
}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        yield c


def test_health_reports_versions(client):
    body = client.get(f"{API_PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["api_version"] == API_VERSION
    assert "events" in body["capabilities"]


def test_profile_round_trip_persists_across_apps(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert c.get(f"{API_PREFIX}/profiles").json() == []
        assert c.put(f"{API_PREFIX}/profiles/bench", json=PROFILE).status_code == 200
    with TestClient(create_app(tmp_path)) as c:
        profiles = c.get(f"{API_PREFIX}/profiles").json()
        assert [p["name"] for p in profiles] == ["bench"]
        assert profiles[0]["cameras"]["front"]["width"] == 640
        assert c.delete(f"{API_PREFIX}/profiles/bench").status_code == 204
        assert c.delete(f"{API_PREFIX}/profiles/bench").status_code == 404


def test_profile_rejects_mismatched_or_unsafe_names(client):
    assert client.put(f"{API_PREFIX}/profiles/other", json=PROFILE).status_code == 422
    bad = {**PROFILE, "name": "../evil"}
    assert client.put(f"{API_PREFIX}/profiles/x", json=bad).status_code == 422


def test_ports_and_cameras_are_lists(client, monkeypatch):
    monkeypatch.setattr("lerobot.nexarm_service.routes.system.list_cameras", lambda: [{"id": 0}])
    assert isinstance(client.get(f"{API_PREFIX}/ports").json(), list)
    assert client.get(f"{API_PREFIX}/cameras").json() == [{"id": 0}]


def test_events_ws_greets_and_forwards_published_events(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as c, c.websocket_connect(f"{API_PREFIX}/ws/events") as ws:
        assert ws.receive_json() == {"type": "hello", "data": {"api_version": API_VERSION}}
        app.state.events.publish("session_changed", {"state": "idle"})
        assert ws.receive_json() == {"type": "session_changed", "data": {"state": "idle"}}


def test_publish_rejects_unknown_event_type(tmp_path):
    with pytest.raises(ValueError):
        create_app(tmp_path).state.events.publish("nope")


def test_openapi_snapshot_is_fresh():
    committed = json.loads(OPENAPI_SNAPSHOT.read_text(encoding="utf-8"))
    assert committed == openapi_schema(), (
        "docs/service/openapi.json is stale; regenerate with "
        "`lerobot-nexarm-service --dump-openapi > docs/service/openapi.json`"
    )


EVENTS_SNAPSHOT = Path(__file__).parents[2] / "docs" / "service" / "events.json"


def test_events_snapshot_is_fresh_and_covers_every_event_type():
    from lerobot.nexarm_service.events import EVENT_SCHEMAS, EVENT_TYPES

    committed = json.loads(EVENTS_SNAPSHOT.read_text(encoding="utf-8"))
    assert committed == EVENT_SCHEMAS, (
        "docs/service/events.json is stale; regenerate with "
        "`lerobot-nexarm-service --dump-events > docs/service/events.json`"
    )
    assert set(EVENT_TYPES) | {"hello"} == set(EVENT_SCHEMAS)


def test_shutdown_invokes_the_exit_hook(tmp_path):
    app = create_app(tmp_path)
    calls = []
    app.state.request_shutdown = lambda: calls.append(1)
    with TestClient(app) as c:
        assert c.post(f"{API_PREFIX}/shutdown").status_code == 202
    assert calls == [1]


def test_every_event_the_engine_publishes_is_in_the_committed_schema():
    import json
    import re
    from pathlib import Path

    import lerobot.nexarm_service as pkg

    schema = json.loads((Path(__file__).parents[2] / "docs/service/events.json").read_text())
    published = set()
    for path in Path(pkg.__file__).parent.rglob("*.py"):
        published |= set(re.findall(r'\.publish\(\s*"([a-z_]+)"', path.read_text(encoding="utf-8")))
    assert published, "no publish calls found; the scan pattern is stale"
    assert published <= set(schema), sorted(published - set(schema))
