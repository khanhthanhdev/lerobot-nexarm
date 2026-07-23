#!/usr/bin/env python

"""Benchmark trained policies on the single-arm NexArm MuJoCo task.

Example:
    MUJOCO_GL=egl uv run python examples/nexarm/benchmark_sim.py \
        --policy act=outputs/train/act/checkpoints/last/pretrained_model \
        --policy pi0=outputs/train/pi0/checkpoints/last/pretrained_model \
        --policy smolvla=outputs/train/smolvla/checkpoints/last/pretrained_model
"""

from __future__ import annotations

from lerobot.scripts.lerobot_nexarm_sim_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
