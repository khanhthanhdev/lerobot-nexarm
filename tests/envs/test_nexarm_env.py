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

    vec_env.close()


def test_nexarm_env_control_modes():
    from lerobot.motors.nexarm.nexarm import JOINT_NAMES
    from lerobot.robots.nexarm_sim.mujoco_backend import RAW_RANGES

    # 1. Raw mode (default)
    env_raw = gym.make("NexArmPickPlace-v0", obs_type="state", control_mode="raw")
    for i, name in enumerate(JOINT_NAMES):
        low, high = RAW_RANGES[name]
        assert env_raw.action_space.low[i] == float(min(low, high))
        assert env_raw.action_space.high[i] == float(max(low, high))

    obs_raw, _ = env_raw.reset(seed=42)
    # Raw agent_pos should be in raw servo units (> 1.0, e.g. around 2048)
    assert np.all(obs_raw["agent_pos"] >= 0.0)
    assert np.any(obs_raw["agent_pos"] > 1.0)

    # Step with raw home action (2048)
    home_action = np.array([2048.0] * 6, dtype=np.float32)
    next_obs_raw, _, _, _, _ = env_raw.step(home_action)
    assert np.all(next_obs_raw["agent_pos"] >= 0.0)
    assert np.any(next_obs_raw["agent_pos"] > 1.0)
    env_raw.close()

    # 2. Normalized mode
    env_norm = gym.make("NexArmPickPlace-v0", obs_type="state", control_mode="normalized")
    assert np.all(env_norm.action_space.low == -1.0)
    assert np.all(env_norm.action_space.high == 1.0)

    obs_norm, _ = env_norm.reset(seed=42)
    assert np.all(obs_norm["agent_pos"] >= -1.0)
    assert np.all(obs_norm["agent_pos"] <= 1.0)

    # Step with 0.0 (normalized center)
    zero_action = np.zeros(6, dtype=np.float32)
    next_obs_norm, _, _, _, _ = env_norm.step(zero_action)
    assert np.all(next_obs_norm["agent_pos"] >= -1.0)
    assert np.all(next_obs_norm["agent_pos"] <= 1.0)
    env_norm.close()


def test_nexarm_env_config_forwards_calibration_and_domain_randomization():
    cfg = NexArmEnv(
        obs_type="state",
        calibration_path="calibration.json",
        enable_domain_randomization=True,
        action_delay_steps=2,
    )
    kwargs = cfg.gym_kwargs
    assert kwargs["calibration_path"] == "calibration.json"
    assert kwargs["enable_domain_randomization"] is True
    assert kwargs["action_delay_steps"] == 2


def test_nexarm_env_applies_calibration_latency_and_domain_randomization(tmp_path):
    from lerobot.motors.nexarm.nexarm import JOINT_NAMES
    from lerobot.robots.nexarm_sim import SimCalibration
    from lerobot.robots.nexarm_sim.mujoco_backend import HOME_POSITIONS

    path = tmp_path / "calibration.json"
    SimCalibration(action_delay_steps=2, fps=30).save(path)
    env = gym.make(
        "NexArmPickPlace-v0",
        obs_type="state",
        calibration_path=path,
        enable_domain_randomization=True,
    )
    try:
        backend = env.unwrapped.backend
        assert backend.action_delay_steps == 2
        env.reset(seed=3)
        assert "cube" in backend.last_domain_params["objects"]
        home_pan = backend.data.ctrl[backend._actuator_ids["shoulder_pan"]]
        # The delayed command reaches the actuators only after the calibrated latency.
        action = np.array([HOME_POSITIONS[name] for name in JOINT_NAMES], dtype=np.float32)
        action[0] = 3000.0
        env.step(action)
        assert backend.data.ctrl[backend._actuator_ids["shoulder_pan"]] == pytest.approx(home_pan)
    finally:
        env.close()

    with pytest.raises(ValueError, match="fps"):
        gym.make("NexArmPickPlace-v0", obs_type="state", fps=15, calibration_path=path)


def test_nexarm_stack_bowls_env_contract():
    env = gym.make("NexArmStackBowls-v0", obs_type="state")
    check_env(env.unwrapped, skip_render_check=True)
    obs, info = env.reset(seed=4)
    task = env.unwrapped.task
    assert obs["environment_state"].shape == (9,)
    bottom = task.bowl_position(task.current_order[0])
    np.testing.assert_allclose(obs["environment_state"][:3], bottom, atol=1e-6)
    assert info["task"] == env.unwrapped.task_description
    assert all(color in info["task"] for color in ("red", "blue", "black"))
    env.reset(seed=5)
    assert env.unwrapped.task_description != info["task"]  # the order follows the seed
    env.close()


def test_nexarm_stack_bowls_env_rewards_a_released_stack():
    import mujoco

    from lerobot.motors.nexarm.nexarm import GRIPPER_OPEN_POS

    env = gym.make("NexArmStackBowls-v0", obs_type="state")
    obs, _ = env.reset(seed=0)
    unwrapped = env.unwrapped
    backend, task = unwrapped.backend, unwrapped.task
    for i, color in enumerate(task.current_order):
        qadr = backend.model.jnt_qposadr[task._joint_ids[color]]
        backend.data.qpos[qadr : qadr + 7] = [0, -0.2, 0.02 + 0.03 * i, 1, 0, 0, 0]
    mujoco.mj_forward(backend.model, backend.data)
    action = obs["agent_pos"].copy()
    action[-1] = GRIPPER_OPEN_POS
    rewards, terminated, info = [], False, {}
    for _ in range(40):
        _, reward, terminated, _, info = env.step(action)
        rewards.append(reward)
        if terminated:
            break
    assert terminated and info["success"] and info["reason"] == "success"
    assert rewards[0] >= 4.0  # both nesting stages
    assert rewards[-1] >= 9.0  # plus the success bonus
    env.close()


def test_nexarm_env_config_sizes_environment_state_by_task():
    assert NexArmEnv(obs_type="state").features["environment_state"].shape == (6,)
    stack = NexArmEnv(task="NexArmStackBowls-v0", obs_type="state")
    assert stack.features["environment_state"].shape == (9,)
    env = make_env(stack, n_envs=1)["nexarm"][0]
    obs, _ = env.reset(seed=0)
    assert obs["environment_state"].shape == (1, 9)
    assert len(env.call("task_description")[0]) > 0
    env.close()


ENV_IDS = ["NexArmPickPlace-v0", "NexArmStackBowls-v0"]


def _rollout(env_id: str, seed: int, steps: int = 20, **kwargs):
    env = gym.make(env_id, obs_type="state", **kwargs)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    trace = [np.concatenate([obs["agent_pos"], obs["environment_state"]])]
    rewards = []
    for _ in range(steps):
        obs, reward, *_ = env.step(env.action_space.sample())
        trace.append(np.concatenate([obs["agent_pos"], obs["environment_state"]]))
        rewards.append(reward)
    env.close()
    return np.array(trace), np.array(rewards)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_nexarm_env_same_seed_rollouts_are_identical(env_id):
    """Catches hidden global RNG use and nondeterministic physics, bit for bit."""
    kwargs = {"enable_domain_randomization": True, "init_qpos_noise_rad": 0.02}
    first, second = _rollout(env_id, 7, **kwargs), _rollout(env_id, 7, **kwargs)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], _rollout(env_id, 8, **kwargs)[0])


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("reward_type", ["dense", "sparse"])
def test_nexarm_env_invariants_under_random_actions(env_id, reward_type):
    env = gym.make(env_id, obs_type="state", reward_type=reward_type, enable_domain_randomization=True)
    env.action_space.seed(0)
    for seed in range(2):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        for _ in range(15):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward) and np.all(np.isfinite(obs["environment_state"]))
            components = info["reward_components"]
            assert reward == pytest.approx(sum(components.values()))
            assert all(np.isfinite(value) for value in components.values())
            if reward_type == "sparse":
                assert set(components) == {"success"}
            if terminated or truncated:
                break
    env.close()


def test_nexarm_env_reward_components_name_each_term():
    env = gym.make("NexArmPickPlace-v0", obs_type="state")
    obs, _ = env.reset(seed=0)
    _, reward, *_, info = env.step(obs["agent_pos"])
    assert set(info["reward_components"]) == {"reach", "grasp", "lift", "target", "place", "success"}
    assert info["reward_components"]["reach"] > 0
    env.close()
    env = gym.make("NexArmStackBowls-v0", obs_type="state")
    obs, _ = env.reset(seed=0)
    *_, info = env.step(obs["agent_pos"])
    assert set(info["reward_components"]) == {"reach", "grasp", "carry", "stage", "success"}
    env.close()


def test_nexarm_env_initial_pose_noise_is_seeded_and_bounded():
    from lerobot.motors.nexarm.nexarm import JOINT_NAMES
    from lerobot.robots.nexarm_sim.mujoco_backend import HOME_POSITIONS

    def start(noise: float, seed: int) -> np.ndarray:
        env = gym.make("NexArmPickPlace-v0", obs_type="state", init_qpos_noise_rad=noise)
        backend = env.unwrapped.backend
        env.reset(seed=seed)
        joints = np.array(
            [backend.data.qpos[backend.model.jnt_qposadr[backend._joint_ids[n]]] for n in JOINT_NAMES]
        )
        home = np.array([backend.raw_to_control(n, HOME_POSITIONS[n]) for n in JOINT_NAMES])
        env.close()
        return joints - home

    home_start = start(0.0, 1)  # the 10 settle steps let gravity sag it slightly
    offset = start(0.05, 1) - home_start
    np.testing.assert_array_equal(offset, start(0.05, 1) - home_start)
    assert np.abs(offset[:-1]).max() > 0.01
    assert np.abs(offset[:-1]).max() < 0.05 + 0.01
    assert abs(offset[-1]) < 1e-3  # the gripper starts at home


def test_nexarm_env_config_takes_the_task_episode_length():
    assert "max_episode_steps" not in NexArmEnv().gym_kwargs
    assert NexArmEnv(episode_length=50).gym_kwargs["max_episode_steps"] == 50
    assert NexArmEnv(init_qpos_noise_rad=0.02).gym_kwargs["init_qpos_noise_rad"] == 0.02
    env = gym.make("NexArmStackBowls-v0", **NexArmEnv(obs_type="state").gym_kwargs)
    assert env.unwrapped.max_episode_steps == 900
    env.close()
