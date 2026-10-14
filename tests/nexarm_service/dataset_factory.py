# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Builds a tiny real LeRobot v3.0 NexArm dataset (with an encoded video) for the dataset tests."""

from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.motors.nexarm.nexarm import JOINT_NAMES

JOINT_FEATURE_NAMES = [f"{n}.pos" for n in JOINT_NAMES]
FPS = 10
HEIGHT, WIDTH = 48, 64


def make_dataset(root: Path, repo_id: str, episodes: int = 3, frames: int = 12) -> Path:
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_FEATURE_NAMES},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_FEATURE_NAMES},
        "observation.images.front": {
            "dtype": "video",
            "shape": (HEIGHT, WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
    }
    path = root / repo_id
    dataset = LeRobotDataset.create(
        repo_id, FPS, root=path, robot_type="nexarm_follower", features=features, use_videos=True
    )
    for ep in range(episodes):
        for i in range(frames):
            pos = np.full(6, 2048.0 + 10 * ep + i, dtype=np.float32)
            image = np.full((HEIGHT, WIDTH, 3), (ep * 60 + i * 5) % 255, dtype=np.uint8)
            dataset.add_frame(
                {
                    "action": pos,
                    "observation.state": pos,
                    "observation.images.front": image,
                    "task": f"task {ep}",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return path
