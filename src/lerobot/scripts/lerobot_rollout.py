#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy deployment engine with pluggable rollout strategies.

``lerobot-rollout`` is the single CLI for running trained policies on
real robots.

Strategies
----------
    --strategy.type=base       Autonomous rollout, no recording
    --strategy.type=sentry     Continuous recording with auto-upload
    --strategy.type=highlight  Ring buffer + keystroke save
    --strategy.type=dagger     Human-in-the-loop (DAgger / RaC)
    --strategy.type=episodic   Episode-oriented recording with reset phases

Inference backends
------------------
    --inference.type=sync      One policy call per control tick (default)
    --inference.type=rtc       Real-Time Chunking for slow VLA models

Usage examples
--------------
::

    # Base mode — quick evaluation with sync inference
    lerobot-rollout \
        --strategy.type=base \
        --policy.path=user/act_nexarm_real \
        --robot.type=nexarm_follower \
        --robot.port=/dev/ttyUSB0 \
        --task="pick up cube" --duration=30

    # Base mode — RTC inference for slow VLAs (Pi0, Pi0.5, SmolVLA)
    lerobot-rollout \
        --strategy.type=base \
        --policy.path=lerobot/smolvla_base \
        --inference.type=rtc \
        --inference.rtc.execution_horizon=10 \
        --inference.rtc.max_guidance_weight=10.0 \
        --robot.type=nexarm_follower \
        --robot.port=/dev/ttyUSB0 \
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="pick up cube" --duration=60

    # Sentry mode — continuous recording with periodic upload
    lerobot-rollout \
        --strategy.type=sentry \
        --strategy.upload_every_n_episodes=5 \
        --policy.path=lerobot/smolvla_base \
        --inference.type=rtc \
        --robot.type=nexarm_follower \
        --robot.port=/dev/ttyUSB0 \
        --dataset.repo_id=user/rollout_sentry_data \
        --dataset.single_task="patrol" --duration=3600

    # Highlight mode — ring buffer, press 's' to save, 'h' to push
    lerobot-rollout \
        --strategy.type=highlight \
        --strategy.buffer_size_s=15 \
        --policy.path=user/my_policy \
        --robot.type=nexarm_follower \
        --robot.port=/dev/ttyUSB0 \
        --dataset.repo_id=user/rollout_highlights

    # DAgger mode — human intervention with teleoperator
    lerobot-rollout \
        --strategy.type=dagger \
        --policy.path=user/my_policy \
        --robot.type=nexarm_follower \
        --robot.port=/dev/ttyUSB0 \
        --teleop.type=nexarm_leader \
        --teleop.port=/dev/ttyUSB1 \
        --dataset.repo_id=user/rollout_dagger_data \
        --dataset.single_task="Grasp the block"

    # Episodic mode — episode-oriented recording with reset phases
    lerobot-rollout \
        --strategy.type=episodic \
        --policy.path=user/my_policy \
        --robot.type=nexarm_follower \
        --robot.port=/dev/ttyUSB0 \
        --teleop.type=nexarm_leader \
        --teleop.port=/dev/ttyUSB1 \
        --dataset.repo_id=user/rollout_episodic_data \
        --dataset.num_episodes=20 \
        --dataset.single_task="Grab the cube"

    # Stream to Foxglove instead of Rerun:
    # add --display_mode=foxglove, then connect the Foxglove app to ws://127.0.0.1:8765.
"""

import logging

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    mobile_bi_nexarm_sim,
    nexarm_follower,
    nexarm_sim,
)
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    gamepad,
    keyboard,
    make_teleoperator_from_config,
    nexarm_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

logger = logging.getLogger(__name__)


@parser.wrap()
def rollout(cfg: RolloutConfig):
    """Main entry point for policy deployment."""
    init_logging()

    if cfg.display_data:
        logger.info(
            "Initializing %s visualization (ip=%s, port=%s)",
            cfg.display_mode,
            cfg.display_ip,
            cfg.display_port,
        )
        init_visualization(
            cfg.display_mode,
            session_name="rollout",
            ip=cfg.display_ip,
            port=cfg.display_port,
            rerun_save_path=cfg.rerun_save_path,
        )
    elif cfg.rerun_save_path is not None:
        raise ValueError("--rerun_save_path requires --display_data=true.")

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    logger.info("Building rollout context...")
    ctx = build_rollout_context(cfg, shutdown_event)

    strategy = create_strategy(cfg.strategy)
    logger.info("Rollout strategy: %s", cfg.strategy.type)
    logger.info(
        "Robot: %s | FPS: %.0f | Duration: %s",
        cfg.robot.type if cfg.robot else "?",
        cfg.fps,
        f"{cfg.duration}s" if cfg.duration > 0 else "infinite",
    )

    try:
        strategy.setup(ctx)
        logger.info("Rollout setup complete, starting rollout...")
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

    logger.info("Rollout finished")


def main():
    """CLI entry point for ``lerobot-rollout``."""
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
