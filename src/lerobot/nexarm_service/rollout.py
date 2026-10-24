# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Policy deployment sessions (rollout, eval, coaching) on top of ``lerobot.rollout``.

The library's strategies own the control loop. This worker wraps them with the engine's safety
contract: the checkpoint's training robot type is verified first, every action is clamped to the soft
limits, joints are streamed to the desktop, and stopping returns the follower to a rest pose captured from
validated reads before torque is cut (the library's own return-to-initial-position is disabled so there is
exactly one return path).
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any

from lerobot.nexarm_service.arm_worker import JOINT_PUBLISH_EVERY, FollowerGuard, joint_update_payload
from lerobot.nexarm_service.events import EventHub
from lerobot.nexarm_service.hardware_factory import follower_config, leader_config
from lerobot.nexarm_service.models import ModelStore
from lerobot.nexarm_service.profiles import RobotProfile
from lerobot.nexarm_service.safety import clamp_action

logger = logging.getLogger(__name__)


def disable_input_listeners() -> None:
    """Never install a global keyboard hook or pedal thread from the engine.

    The library strategies start pynput / terminal / pedal listeners by default; on Windows a global hook
    would turn keys pressed in any other window (the desktop app included) into episode controls. Controls
    reach the strategies through ``RolloutWorker.control`` instead.
    """
    from lerobot.rollout.strategies import dagger, episodic

    episodic.init_keyboard_listener = lambda **_: (None, new_episodic_events())  # type: ignore[attr-defined]
    dagger._init_dagger_keyboard = lambda events, cfg: None
    dagger._init_dagger_pedal = lambda events, cfg: None


def new_episodic_events() -> dict[str, bool]:
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


class RolloutApi:
    """Thin seam over ``lerobot.rollout`` so tests can run the worker without a policy or a robot."""

    def make_config(
        self, kind: str, profile: RobotProfile, checkpoint: str, params: dict[str, Any], datasets_root: Path
    ) -> Any:
        from lerobot.configs.dataset import DatasetRecordConfig
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.rollout import RolloutConfig
        from lerobot.rollout.configs import BaseStrategyConfig, EpisodicStrategyConfig

        policy = PreTrainedConfig.from_pretrained(checkpoint)
        # A Hub id must stay a str: Path() would turn "user/model" into "user\model" on Windows.
        policy.pretrained_path = checkpoint  # type: ignore[assignment]
        task = str(params.get("task", ""))
        dataset = None
        strategy: Any = BaseStrategyConfig()
        teleop = None
        if kind == "eval":
            repo_id = params.get("repo_id", "")
            if not repo_id.split("/", 1)[-1].startswith("eval_"):
                raise ValueError("eval sessions record into a dataset whose name starts with 'eval_'")
            if (datasets_root / repo_id).exists():
                raise ValueError(f"{repo_id} already exists; pick a new eval dataset name")
            dataset = DatasetRecordConfig(
                repo_id=repo_id,
                single_task=task,
                root=datasets_root / repo_id,
                fps=int(params.get("fps", 30)),
                episode_time_s=float(params.get("episode_time_s", 30)),
                reset_time_s=float(params.get("reset_time_s", 10)),
                num_episodes=int(params.get("num_episodes", 5)),
                push_to_hub=False,
            )
            strategy = EpisodicStrategyConfig()
        if kind == "coaching":
            from lerobot.rollout.configs import DAggerStrategyConfig

            repo_id = params.get("repo_id", "")
            if "/" not in repo_id or repo_id.split("/", 1)[-1].startswith("eval_"):
                raise ValueError("coaching records corrections into a dataset named owner/name (not eval_*)")
            if (datasets_root / repo_id).exists():
                raise ValueError(f"{repo_id} already exists; pick a new dataset name")
            episodes = int(params.get("num_episodes", 10))
            dataset = DatasetRecordConfig(
                repo_id=repo_id,
                single_task=task,
                root=datasets_root / repo_id,
                fps=int(params.get("fps", 30)),
                num_episodes=episodes,
                push_to_hub=False,
            )
            strategy = DAggerStrategyConfig(
                num_episodes=episodes, record_autonomous=bool(params.get("record_autonomous", False))
            )
            teleop = leader_config(profile)
        return RolloutConfig(
            robot=follower_config(profile, with_cameras=True),
            teleop=teleop,
            policy=policy,
            strategy=strategy,
            dataset=dataset,
            fps=float(params.get("fps", 30)),
            duration=float(params.get("duration_s", 30 if kind == "rollout" else 0)),
            task=task,
            play_sounds=False,
            return_to_initial_position=False,
        )

    def build_context(self, cfg: Any, shutdown: threading.Event) -> Any:
        from lerobot.rollout import build_rollout_context

        return build_rollout_context(cfg, shutdown)

    def create_strategy(self, cfg: Any) -> Any:
        from lerobot.rollout import create_strategy

        disable_input_listeners()
        return create_strategy(cfg.strategy)


class RolloutWorker:
    def __init__(
        self,
        kind: str,
        profile: RobotProfile,
        events: EventHub,
        models: ModelStore,
        datasets_root: Path,
        params: dict[str, Any],
        api: RolloutApi | None = None,
    ) -> None:
        if not params.get("checkpoint"):
            raise ValueError(f"{kind} needs params.checkpoint (a checkpoint path or a Hub repo id)")
        if not profile.cameras:
            raise ValueError(
                f"profile {profile.name!r} has no cameras; configure them before deploying a policy"
            )
        self.error: str | None = None
        self._kind = kind
        self._profile = profile
        self._events = events
        self._models = models
        self._datasets_root = datasets_root
        self._params = params
        self._api = api or RolloutApi()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._ctx: Any = None
        self._strategy: Any = None
        self._guard: FollowerGuard | None = None
        self._ticks = 0
        self._started = 0.0

    # -- Worker protocol -----------------------------------------------------------------------
    def start(self) -> None:
        checkpoint = str(self._params["checkpoint"])
        self._models.check_robot_type(checkpoint, bool(self._params.get("allow_unverified", False)))
        cfg = self._api.make_config(self._kind, self._profile, checkpoint, self._params, self._datasets_root)
        self._ctx = self._api.build_context(cfg, self._shutdown)
        try:
            robot = self._ctx.hardware.robot_wrapper.inner
            self._install_safety(robot)
            self._guard = FollowerGuard(robot)
            self._guard.engage()
            self._after_context()
            self._strategy = self._api.create_strategy(cfg)
            self._strategy.setup(self._ctx)
            self._after_setup()
        except Exception:
            self._release()
            raise
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"nexarm-{self._kind}", daemon=True)
        self._thread.start()

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {
            "checkpoint": self._params["checkpoint"],
            "elapsed_s": round(time.monotonic() - self._started, 1) if self._started else 0.0,
            "ticks": self._ticks,
        }

    # Desktop control name -> the strategy event flag it sets (episodic strategy, i.e. eval sessions).
    _EPISODE_CONTROLS = {
        "exit_early": "exit_early",
        "rerecord": "rerecord_episode",
        "finish": "stop_recording",
    }

    def control(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        events = getattr(self._strategy, "_events", None)
        flag = self._EPISODE_CONTROLS.get(action)
        if flag is None or not isinstance(events, dict) or flag not in events:
            raise ValueError(f"this session does not support the {action!r} control")
        events[flag] = True
        return self.status()

    # -- hooks for session kinds that need more than the base strategy (coaching) ---------------
    def _after_context(self) -> None:
        """Hardware is connected and the rest pose captured; the strategy is not created yet."""

    def _after_setup(self) -> None:
        """The strategy is set up; its loop has not started."""

    def _before_release(self) -> None:
        """The loop has stopped; the arm is about to be returned to rest."""

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=60.0)
        self._release()

    # -- internals -----------------------------------------------------------------------------
    def _install_safety(self, robot: Any) -> None:
        send, observe = robot.send_action, robot.get_observation

        def clamped_send(action: dict[str, float]) -> Any:
            return send(clamp_action(action))

        def streaming_observe() -> Any:
            obs = observe()
            self._ticks += 1
            if self._ticks % JOINT_PUBLISH_EVERY == 0:
                self._events.publish("joint_update", joint_update_payload(obs))
            return obs

        robot.send_action = clamped_send
        robot.get_observation = streaming_observe

    def _run(self) -> None:
        try:
            self._strategy.run(self._ctx)
        except Exception as exc:
            logger.exception("%s loop failed", self._kind)
            self.error = f"{self._kind} loop failed: {exc}"

    def _release(self) -> None:
        """Rest return and torque-off first, then the strategy's own teardown (disconnect, finalize)."""
        self._before_release()
        if self._guard is not None:
            self._guard.release()
            self.error = self.error or self._guard.error
            self._guard = None
        if self._strategy is not None:
            try:
                self._strategy.teardown(self._ctx)
            except Exception as exc:
                logger.exception("rollout teardown failed")
                self.error = self.error or f"teardown failed: {exc}"
            self._strategy = None
        elif self._ctx is not None:
            # setup never completed: still release the connected hardware.
            robot = self._ctx.hardware.robot_wrapper.inner
            teleop = self._ctx.hardware.teleop
            for device in (robot, teleop):
                if device is not None and device.is_connected:
                    try:
                        device.disconnect()
                    except Exception as exc:
                        logger.error("disconnect failed: %s", exc)
        self._ctx = None
