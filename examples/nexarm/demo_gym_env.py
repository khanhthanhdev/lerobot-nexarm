#!/usr/bin/env python

"""Demonstrate running the NexArmPickPlace-v0 Gymnasium environment.

Usage:
    uv run python examples/nexarm/demo_gym_env.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

# Ensure NexArm environment registration
import lerobot.envs.nexarm  # noqa: F401


def main() -> None:
    print("Initializing NexArmPickPlace-v0 Gymnasium Environment...")
    env = gym.make(
        "NexArmPickPlace-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=50,
        observation_width=320,
        observation_height=240,
    )

    print(f"Action space:      {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    obs, info = env.reset(seed=42)
    print("\nInitial Observation:")
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            print(f"  - {key}: shape={val.shape}, dtype={val.dtype}")

    print(f"Initial Info: {info}")

    print("\nExecuting 10 steps with small exploratory actions around initial pose...")
    for step_idx in range(1, 11):
        # Sample small continuous action around current agent position
        action = obs["agent_pos"] + np.random.uniform(-15.0, 15.0, size=env.action_space.shape).astype(
            np.float32
        )
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  Step {step_idx:02d}: reward={reward:.4f}, "
            f"grasped={info['is_grasped']}, success={info['success']}"
        )
        if terminated or truncated:
            print(f"Episode ended at step {step_idx} (term={terminated}, trunc={truncated})")
            break

    frame = env.render()
    if frame is not None:
        print(f"\nRendered front camera frame shape: {frame.shape}, dtype={frame.dtype}")

    env.close()
    print("\nEnvironment closed successfully. Demo completed!")


if __name__ == "__main__":
    main()
