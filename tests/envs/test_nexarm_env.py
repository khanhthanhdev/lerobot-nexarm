# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import lerobot.envs.nexarm  # noqa: F401
from lerobot.envs.configs import NexArmEnv
from lerobot.envs.factory import make_env


def test_nexarm_env_registration():
    env = gym.make("NexArmPickPlace-v0")
    assert env is not None
    env.close()

    env_pkg = gym.make("gym_nexarm/NexArmPickPlace-v0")
    assert env_pkg is not None
    env_pkg.close()


def test_nexarm_env_gym_checker():
    env = gym.make("NexArmPickPlace-v0", obs_type="pixels_agent_pos")
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("obs_type", ["pixels_agent_pos", "pixels", "state"])
def test_nexarm_env_obs_types(obs_type):
    env = gym.make(
        "NexArmPickPlace-v0",
        obs_type=obs_type,
        observation_width=160,
        observation_height=120,
    )
    obs, info = env.reset(seed=42)

    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    assert "is_grasped" in info
    assert "success" in info

    if obs_type == "pixels_agent_pos":
        assert "agent_pos" in obs
        assert obs["agent_pos"].shape == (6,)
        assert "pixels/front" in obs
        assert obs["pixels/front"].shape == (120, 160, 3)
        assert "pixels/wrist" in obs
        assert obs["pixels/wrist"].shape == (120, 160, 3)
    elif obs_type == "pixels":
        assert "pixels/front" in obs
        assert "pixels/wrist" in obs
        assert "agent_pos" not in obs
    elif obs_type == "state":
        assert "agent_pos" in obs
        assert "environment_state" in obs
        assert obs["environment_state"].shape == (6,)

    env.close()


def test_nexarm_env_step_and_reward():
    env = gym.make(
        "NexArmPickPlace-v0",
        obs_type="state",
        reward_type="dense",
        max_episode_steps=5,
    )
    obs, info = env.reset(seed=123)

    action = np.zeros(6, dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(step_info, dict)
    assert "agent_pos" in next_obs

    # Run remaining steps to test truncation
    for _ in range(4):
        _, _, term, trunc, _ = env.step(action)
        if term:
            break

    assert trunc or term
    env.close()


def test_nexarm_env_sparse_reward():
    env = gym.make(
        "NexArmPickPlace-v0",
        obs_type="state",
        reward_type="sparse",
    )
    env.reset(seed=1)
    action = np.zeros(6, dtype=np.float32)
    _, reward, _, _, _ = env.step(action)
    assert reward in (0.0, 1.0)
    env.close()


def test_nexarm_env_seeded_reproducibility():
    env1 = gym.make("NexArmPickPlace-v0", obs_type="state")
    env2 = gym.make("NexArmPickPlace-v0", obs_type="state")

    obs1, _ = env1.reset(seed=999)
    obs2, _ = env2.reset(seed=999)

    np.testing.assert_allclose(obs1["environment_state"], obs2["environment_state"])
    np.testing.assert_allclose(obs1["agent_pos"], obs2["agent_pos"])

    env1.close()
    env2.close()


def test_nexarm_env_render_rgb():
    env = gym.make(
        "NexArmPickPlace-v0",
        render_mode="rgb_array",
        observation_width=160,
        observation_height=120,
    )
    env.reset(seed=42)
    img = env.render()
    assert img is not None
    assert isinstance(img, np.ndarray)
    assert img.shape == (120, 160, 3)
    assert img.dtype == np.uint8
    env.close()


def test_nexarm_env_lerobot_factory():
    cfg = NexArmEnv(
        task="NexArmPickPlace-v0",
        obs_type="state",
        episode_length=50,
    )
    env_dict = make_env(cfg, n_envs=1)
    assert "nexarm" in env_dict
    assert 0 in env_dict["nexarm"]

    vec_env = env_dict["nexarm"][0]
    obs, info = vec_env.reset()
    assert "agent_pos" in obs
    assert obs["agent_pos"].shape == (1, 6)

    action = np.zeros((1, 6), dtype=np.float32)
    next_obs, reward, term, trunc, step_info = vec_env.step(action)
    assert "agent_pos" in next_obs
    assert len(reward) == 1

    vec_env.close()
