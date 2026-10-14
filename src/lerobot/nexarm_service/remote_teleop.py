# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Remote teleoperation: a leader on one machine drives a follower hosted by another engine.

Two session kinds share one wire protocol (JSON over a WebSocket, joints only):

* ``hosting`` (the **station**, where the follower is): opens the follower, captures its rest pose, and waits
  for one operator. The follower is *parked* at rest until an operator engages; it then follows the
  operator's actions through a soft start and a velocity cap, and parks again when the operator leaves, says
  bye, or goes silent for a second.
* ``remote_teleoperation`` (the **operator**, where the leader is): reads the leader and streams its
  actions, shows the station's joints, and measures the round trip.

The station never trusts the stream: every action is validated, clamped to the soft limits and
rate-limited relative to its own last command. Transport is a direct WebSocket (see
plans/reports/spike-261014-0950-remote-teleop-transport.md); the protocol is kept apart from the sockets so
another transport can carry it.
"""

import contextlib
import json
import logging
import math
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.nexarm_service import API_VERSION
from lerobot.nexarm_service.arm_worker import JOINT_PUBLISH_EVERY, ArmWorker, joint_update_payload
from lerobot.nexarm_service.events import EventHub
from lerobot.nexarm_service.profiles import RobotProfile
from lerobot.nexarm_service.safety import clamp_action

logger = logging.getLogger(__name__)

DEFAULT_FPS = 30
DEFAULT_RATE_CAP_TICKS_S = 1500.0
SILENCE_S = 1.0
SOFT_START_S = 1.0
JOINT_KEYS = [f"{n}.pos" for n in JOINT_NAMES]
WS_PATH = "/api/v1/ws/remote"


class ProtocolError(ValueError):
    pass


# -- protocol ---------------------------------------------------------------------------------
def parse_action(message: dict[str, Any]) -> tuple[int, list[float]]:
    """Validate an ``action`` frame and return ``(seq, ticks in joint order)``."""
    action = message.get("action")
    seq = message.get("seq")
    if not isinstance(action, dict) or not isinstance(seq, int):
        raise ProtocolError("an action frame needs an integer seq and an action object")
    ticks = []
    for key in JOINT_KEYS:
        value = action.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
            raise ProtocolError(f"action.{key} must be a finite number")
        ticks.append(float(value))
    return seq, ticks


def step_toward(current: list[float], target: list[float], cap: float) -> list[float]:
    """Move each joint toward ``target`` by at most ``cap`` ticks."""
    return [c + max(-cap, min(cap, t - c)) for c, t in zip(current, target, strict=True)]


def to_action(ticks: list[float]) -> dict[str, float]:
    return dict(zip(JOINT_KEYS, ticks, strict=True))


def ws_url(station: str) -> str:
    """``host:port``, ``http://host:port`` or ``ws://host:port`` to the operator WebSocket URL."""
    text = station.strip()
    if "://" not in text:
        text = f"ws://{text}"
    parts = urlsplit(text)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parts.scheme)
    if scheme is None or not parts.hostname:
        raise ValueError(f"{station!r} is not a station address (use host:port)")
    return urlunsplit((scheme, parts.netloc, WS_PATH, "", ""))


# -- station ----------------------------------------------------------------------------------
class HostingWorker(ArmWorker):
    """The station: parks the follower at rest and lets one operator drive it, safely."""

    needs_leader = False
    thread_name = "nexarm-hosting"

    def __init__(
        self, profile: RobotProfile, hardware: Any, events: EventHub, params: dict[str, Any]
    ) -> None:
        super().__init__(profile, hardware, events)
        self.fps = int(params.get("fps", DEFAULT_FPS))
        self.rate_cap = float(params.get("rate_cap_ticks_s", DEFAULT_RATE_CAP_TICKS_S))
        self.silence_s = float(params.get("silence_s", SILENCE_S))
        self.soft_start_s = float(params.get("soft_start_s", SOFT_START_S))
        self._lock = threading.Lock()
        self._operator: str | None = None
        self._accepting = True
        self._target: list[float] | None = None
        self._last_frame = 0.0
        self._echo_seq = 0
        self._engaged_at = 0.0
        self._commanded: list[float] = []
        self._phase = "parked"
        self._observation: dict[str, float] = {}

    # -- called from the WebSocket handler (event loop thread) ---------------------------------
    def rest_action(self) -> dict[str, float]:
        rest = self._guard.rest if self._guard is not None else None
        return to_action([float(p) for p in rest]) if rest else {}

    def try_attach(self, operator: str) -> str | None:
        """Claim the station for an operator; returns a refusal reason, or ``None`` when attached."""
        with self._lock:
            if not self._accepting:
                return "this station is not accepting operators"
            if self._operator is not None:
                return f"already driven by {self._operator}"
            self._operator = operator
            self._target = None
            self._engaged_at = self._last_frame = time.monotonic()  # grace: the new operator has not sent yet
            self._phase = "engaging"
        return None

    def detach(self, reason: str) -> None:
        with self._lock:
            self._operator = None
            self._target = None
            if self._phase != "parked":
                self._phase = "parking"
        logger.info("operator left the station: %s", reason)

    def on_action(self, seq: int, ticks: list[float]) -> None:
        clamped = clamp_action(to_action(ticks))
        with self._lock:
            if self._operator is None or seq < self._echo_seq:
                return  # stale or from nobody: drop it, never queue
            self._target = [clamped[k] for k in JOINT_KEYS]
            self._last_frame = time.monotonic()
            self._echo_seq = seq

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"phase": self._phase, "seq": self._echo_seq, "ticks": dict(self._observation)}

    # -- controls -----------------------------------------------------------------------------
    def control(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if action == "park":
            self.detach("parked from the station")
        elif action == "block":
            self._accepting = False
            self.detach("operators blocked")
        elif action == "allow":
            self._accepting = True
        else:
            return super().control(action, payload)
        return self.extra_status()

    def extra_status(self) -> dict[str, Any]:
        with self._lock:
            age = (time.monotonic() - self._last_frame) * 1000 if self._last_frame else None
            return {
                "phase": self._phase,
                "operator": self._operator,
                "accepting": self._accepting,
                "frame_age_ms": round(age) if age is not None else None,
                "rate_cap_ticks_s": self.rate_cap,
            }

    # -- control loop -------------------------------------------------------------------------
    def _ready(self) -> None:
        rest = self._guard.rest if self._guard is not None else None
        self._commanded = [float(p) for p in (rest or [])]

    def _run(self) -> None:
        period = 1.0 / self.fps
        rest = list(self._commanded)
        while not self._stop.is_set():
            began = time.monotonic()
            with self._lock:
                operator, target = self._operator, self._target
                silent = operator is not None and (began - self._last_frame) > self.silence_s
                engaged_for = began - self._engaged_at
            if silent:
                self.detach(f"no actions for {self.silence_s:.1f}s")
                operator, target = None, None
            if operator is not None and target is not None:
                factor = min(1.0, 0.2 + 0.8 * engaged_for / max(self.soft_start_s, 1e-6))
                cap = self.rate_cap / self.fps * factor
                desired, phase = target, "engaged" if factor >= 1.0 else "engaging"
            elif operator is not None:
                desired, cap, phase = self._commanded, 0.0, "engaging"  # attached, no frame yet: hold
            else:
                desired, cap = rest, self.rate_cap / self.fps
                phase = (
                    "parked"
                    if all(abs(a - b) <= 1.0 for a, b in zip(self._commanded, rest, strict=True))
                    else "parking"
                )
            self._commanded = step_toward(self._commanded, desired, cap)
            self._follower.send_action(clamp_action(to_action(self._commanded)))
            observation = self._follower.get_observation()
            with self._lock:
                self._phase = phase
                self._observation = {k: float(observation[k]) for k in JOINT_KEYS}
            self._ticks += 1
            if self._ticks % JOINT_PUBLISH_EVERY == 0:
                self._events.publish("joint_update", joint_update_payload(observation))
            self._stop.wait(max(0.0, period - (time.monotonic() - began)))


# -- operator ---------------------------------------------------------------------------------
class RemoteOperatorWorker:
    """The operator: streams the local leader to a station and shows the station's follower."""

    def __init__(
        self, profile: RobotProfile, hardware: Any, events: EventHub, params: dict[str, Any]
    ) -> None:
        station = params.get("station")
        if not station:
            raise ValueError("remote teleoperation needs params.station (host:port of the station's engine)")
        self.error: str | None = None
        self._url = ws_url(str(station))
        # In a header, not the URL: query strings end up in access logs.
        token = params.get("token")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._profile = profile
        self._hardware = hardware
        self._events = events
        self._fps = int(params.get("fps", DEFAULT_FPS))
        self._operator = str(params.get("operator", profile.name))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._leader: Any = None
        self._ws: Any = None
        self._stack = contextlib.ExitStack()
        self._sent_at: dict[int, float] = {}
        self._seq = 0
        self._rtt_ms: float | None = None
        self._station_phase = "connecting"
        self._remote_ticks: dict[str, float] = {}
        self._ready: dict[str, Any] = {}

    def start(self) -> None:
        from websockets.exceptions import InvalidStatus, WebSocketException
        from websockets.sync.client import connect

        self._leader = self._hardware.make_leader(self._profile)
        try:
            self._leader.connect(calibrate=False)
            try:
                self._ws = self._stack.enter_context(
                    connect(self._url, additional_headers=self._headers, open_timeout=10, max_size=1 << 20)
                )
            except InvalidStatus as exc:
                raise ConnectionError(
                    f"the station rejected the connection (HTTP {exc.response.status_code}); check the token and "
                    "that the station is hosting"
                ) from exc
            except (WebSocketException, TimeoutError) as exc:
                raise ConnectionError(f"could not connect to the station: {exc}") from exc
            hello = {
                "type": "hello",
                "api_version": API_VERSION,
                "joints": JOINT_KEYS,
                "operator": self._operator,
            }
            self._ws.send(json.dumps(hello))
            reply = json.loads(self._ws.recv(timeout=10))
            if reply.get("type") != "ready":
                raise RuntimeError(f"the station refused: {reply.get('reason', reply.get('type'))}")
            self._ready = reply
            self._station_phase = "engaging"
        except Exception:
            self._shutdown()
            raise
        self._thread = threading.Thread(target=self._run, name="nexarm-remote-operator", daemon=True)
        self._thread.start()

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {
            "station_phase": self._station_phase,
            "rtt_ms": round(self._rtt_ms, 1) if self._rtt_ms is not None else None,
            "frames_sent": self._seq,
            "remote_ticks": dict(self._remote_ticks),
        }

    def control(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise ValueError(f"this session does not support the {action!r} control")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._shutdown()

    def _shutdown(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.send(json.dumps({"type": "bye"}))
            except Exception:
                logger.debug("bye not delivered", exc_info=True)
        with contextlib.suppress(Exception):
            self._stack.close()
        leader, self._leader = self._leader, None
        if leader is not None and leader.is_connected:
            try:
                leader.disconnect()
            except Exception as exc:
                logger.error("leader disconnect failed: %s", exc)

    def _run(self) -> None:
        period = 1.0 / self._fps
        try:
            while not self._stop.is_set():
                began = time.monotonic()
                action = self._leader.get_action()
                self._seq += 1
                self._sent_at[self._seq] = began
                self._ws.send(json.dumps({"type": "action", "seq": self._seq, "t": began, "action": action}))
                self._drain()
                if len(self._sent_at) > 256:
                    for old in sorted(self._sent_at)[:-128]:
                        del self._sent_at[old]
                self._stop.wait(max(0.0, period - (time.monotonic() - began)))
        except Exception as exc:
            logger.exception("remote operator loop failed")
            self.error = f"the connection to the station failed: {exc}"

    def _drain(self) -> None:
        """Read every state frame that has arrived, without waiting."""
        while True:
            try:
                raw = self._ws.recv(timeout=0)
            except TimeoutError:
                return
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "state":
                self._station_phase = message.get("phase", self._station_phase)
                self._remote_ticks = message.get("ticks", self._remote_ticks)
                sent = self._sent_at.pop(message.get("seq", -1), None)
                if sent is not None:
                    rtt = (time.monotonic() - sent) * 1000
                    self._rtt_ms = rtt if self._rtt_ms is None else 0.8 * self._rtt_ms + 0.2 * rtt
                if len(self._remote_ticks) == len(JOINT_KEYS):
                    self._events.publish("joint_update", joint_update_payload(self._remote_ticks))
            elif kind == "parked":
                self._station_phase = "parked"
                raise RuntimeError(f"the station parked the arm: {message.get('reason', '')}")
