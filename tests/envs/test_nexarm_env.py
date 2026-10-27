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

import dataclasses
import functools
import pickle

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import lerobot_env_nexarm.nexarm  # noqa: F401
from lerobot.envs.factory import make_env
from lerobot_env_nexarm import NexArmEnv


def test_nexarm_env_registration():
    env = gym.make("NexArmPickPlace-v0")
    assert env is not None
    env.close()

    env_pkg = gym.make("gym_nexarm/NexArmPickPlace-v0")
    assert env_pkg is not None
    env_pkg.close()


@pytest.mark.parametrize("obs_type", ["state", "pixels_agent_pos"])
def test_nexarm_env_gym_checker(obs_type):
    env = gym.make("NexArmPickPlace-v0", obs_type=obs_type)
    unwrapped = env.unwrapped
    if obs_type != "state":
        # GPU (EGL) rasterization can differ by a few pixels between identical renders, so camera
        # observations are checked for API conformance only; "state" covers seeded-reset determinism.
        unwrapped.spec = dataclasses.replace(unwrapped.spec, nondeterministic=True)
    check_env(unwrapped, skip_render_check=True)
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


@pytest.mark.parametrize(
    ("task", "episode_length", "expected"),
    [("NexArmPickPlace-v0", None, 400), ("NexArmStackBowls-v0", None, 900), ("NexArmStackBowls-v0", 50, 50)],
)
def test_nexarm_env_exposes_max_episode_steps_to_lerobot_eval(task, episode_length, expected):
    """lerobot-eval reads `_max_episode_steps` through the vector env, with or without --env.episode_length."""
    cfg = NexArmEnv(task=task, obs_type="state", episode_length=episode_length)
    vec_env = make_env(cfg, n_envs=1)["nexarm"][0]
    assert vec_env.call("_max_episode_steps")[0] == expected
    vec_env.close()


def test_nexarm_env_control_modes():
    from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES
    from lerobot_robot_nexarm.nexarm_sim.mujoco_backend import RAW_RANGES

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
    from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES
    from lerobot_robot_nexarm.nexarm_sim import SimCalibration
    from lerobot_robot_nexarm.nexarm_sim.mujoco_backend import HOME_POSITIONS

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
    task = env.unwrapped.sim_task
    assert obs["environment_state"].shape == (9,)
    bottom = task.bowl_position(task.current_order[0])
    np.testing.assert_allclose(obs["environment_state"][:3], bottom, atol=1e-6)
    assert info["task"] == env.unwrapped.task_description
    assert all(color in info["task"] for color in ("red", "blue", "black"))
    env.reset(seed=5)
    assert env.unwrapped.task_description != info["task"]  # the order follows the seed
    env.close()


def test_nexarm_env_task_is_the_picklable_instruction():
    """LeRobot reads `task` from each sub-env; async vector envs pickle it."""
    env = gym.make("NexArmStackBowls-v0", obs_type="state")
    env.reset(seed=3)
    task = env.get_wrapper_attr("task")
    assert task == env.unwrapped.task_description and pickle.loads(pickle.dumps(task)) == task
    env.close()


def test_nexarm_stack_bowls_env_fixes_the_order():
    kwargs = NexArmEnv(task="NexArmStackBowls-v0", obs_type="state", stack_order="black,red,blue").gym_kwargs
    env = gym.make("NexArmStackBowls-v0", **kwargs)
    for seed in range(4):
        env.reset(seed=seed)
        assert env.unwrapped.sim_task.current_order == ("black", "red", "blue")
    env.close()
    assert "order" not in NexArmEnv().gym_kwargs
    with pytest.raises(ValueError, match="order"):
        gym.make("NexArmStackBowls-v0", obs_type="state", order=("red", "red", "blue"))


def test_nexarm_stack_bowls_env_rewards_a_released_stack():
    import mujoco

    from lerobot_robot_nexarm.motors.nexarm import GRIPPER_OPEN_POS

    env = gym.make("NexArmStackBowls-v0", obs_type="state")
    obs, _ = env.reset(seed=0)
    unwrapped = env.unwrapped
    backend, task = unwrapped.backend, unwrapped.sim_task
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
    assert rewards[0] >= 9.0  # placing both bowls raises the potential from near 0 to 10
    assert rewards[-1] >= 10.0  # the success bonus
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
    from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES
    from lerobot_robot_nexarm.nexarm_sim.mujoco_backend import HOME_POSITIONS

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


class _ScriptedEnvRobot:
    """Lets the stack-bowls dataset generator's scripted expert drive the gym env, one env step per action.

    While a bowl is held, the gripper never opens below ``min_open_while_held``: a real operator's partial
    opening. With ``freeze_at_top_release`` the expert's final release is replaced by holding the nested top
    bowl until the episode times out. Actions sent after the episode ends are ignored.
    """

    def __init__(self, env, *, min_open_while_held: float, freeze_at_top_release: bool = False):
        from types import SimpleNamespace

        self.env = env
        self.backend = env.backend
        self.config = SimpleNamespace(fps=env.fps)
        self.min_open_while_held = min_open_while_held
        self.freeze_at_top_release = freeze_at_top_release
        self.frozen: np.ndarray | None = None
        self._last_raw_gripper: float | None = None
        self.rewards: list[float] = []
        self.infos: list[dict] = []
        self.done = False

    def get_observation(self) -> dict:
        return {}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES

        task = self.env.sim_task
        command = np.array([action[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
        raw_gripper, previous = float(command[-1]), self._last_raw_gripper
        self._last_raw_gripper = raw_gripper
        if task.held_bowl is not None:
            command[-1] = max(command[-1], self.min_open_while_held)
        _, middle, top = task.current_order
        if (
            self.freeze_at_top_release
            and self.frozen is None
            and task.held_bowl == top
            and task.is_nested(middle, top)
            and previous is not None
            and raw_gripper < previous
        ):
            self.frozen = self._last_command
        if self.frozen is not None:
            command = self.frozen
        self._last_command = command
        if not self.done:
            _, reward, terminated, truncated, info = self.env.step(command)
            self.rewards.append(reward)
            self.infos.append(info)
            self.done = terminated or truncated
        return dict(action)


# A seed whose layout the scripted expert stacks; these tests check rewards and stages, not the expert.
_SOLVED_SEED = 1


@functools.cache
def _scripted_stack_bowls(seed: int, **kwargs) -> _ScriptedEnvRobot:
    from examples.nexarm.generate_stack_bowls_dataset import generate_episode
    from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES

    env_kwargs = {key: kwargs.pop(key) for key in ("assist_release_pos", "grasp_mode") if key in kwargs}
    env = gym.make("NexArmStackBowls-v0", obs_type="state", **env_kwargs).unwrapped
    env.reset(seed=seed)
    robot = _ScriptedEnvRobot(env, **kwargs)
    try:
        generate_episode(robot, env.sim_task, seed=seed)
        if robot.frozen is not None:
            hold = dict(zip((f"{name}.pos" for name in JOINT_NAMES), robot.frozen.tolist(), strict=True))
            while not robot.done:
                robot.send_action(hold)
    finally:
        env.close()
    return robot


def test_nexarm_stack_bowls_dense_reward_pays_release_and_success_over_holding():
    """Holding the nested top bowl until timeout must earn less than letting go, and opening never costs."""
    released = _scripted_stack_bowls(_SOLVED_SEED, min_open_while_held=1990)
    assert released.infos[-1]["is_success"]
    held = [info["held_bowl"] for info in released.infos]
    release_steps = [i for i in range(1, len(held)) if held[i - 1] and not held[i]]
    assert len(release_steps) == 2
    for step in release_steps:
        # Placing a bowl raises the potential by 1, so letting go pays more than the step before it.
        assert released.rewards[step] > 0.5
        assert released.rewards[step] >= released.rewards[step - 1]

    holding = _scripted_stack_bowls(_SOLVED_SEED, min_open_while_held=1990, freeze_at_top_release=True)
    assert holding.frozen is not None and not holding.infos[-1]["is_success"]
    assert holding.infos[-1]["held_bowl"]  # still holding the nested top bowl at timeout
    assert sum(holding.rewards) < sum(released.rewards) - 5.0


def test_nexarm_stack_bowls_env_releases_at_real_partial_openings():
    """Real operators open only to about 1990; the env's default threshold drops the bowl there."""
    released = _scripted_stack_bowls(_SOLVED_SEED, min_open_while_held=1990)
    assert released.env.sim_task.grasp_mode == "physical"
    assert released.env.sim_task.assist_release_pos == 2300
    assert released.infos[-1]["is_success"]

    assisted = _scripted_stack_bowls(_SOLVED_SEED, min_open_while_held=1990, grasp_mode="contact_assisted")
    assert assisted.infos[-1]["is_success"]
    # The rigid assist only lets go below its threshold, so 1600 never drops the bowl at real openings.
    never_released = _scripted_stack_bowls(
        _SOLVED_SEED, min_open_while_held=1990, grasp_mode="contact_assisted", assist_release_pos=1600
    )
    assert not never_released.infos[-1]["is_success"]
    assert never_released.infos[-1]["failed_stage"] == "released_middle"


def test_nexarm_stack_bowls_physical_grasp_holds_by_friction_alone(monkeypatch):
    """With no assist the bowl is a free body: it rises with the closed jaws and is never teleported."""
    import mujoco

    from examples.nexarm.generate_stack_bowls_dataset import generate_episode

    teleports = []  # the assist sets a carried bowl's orientation with mju_mat2Quat
    original = mujoco.mju_mat2Quat
    monkeypatch.setattr(mujoco, "mju_mat2Quat", lambda *args: teleports.append(args) or original(*args))
    env = gym.make("NexArmStackBowls-v0", obs_type="state").unwrapped
    try:
        env.reset(seed=_SOLVED_SEED)
        assert env.sim_task.grasp_mode == "physical"
        robot = _ScriptedEnvRobot(env, min_open_while_held=0.0)
        generate_episode(robot, env.sim_task, seed=_SOLVED_SEED)
    finally:
        env.close()
    assert not teleports
    assert robot.infos[-1]["is_success"]
    assert robot.infos[-1]["stage/picked_middle"] and robot.infos[-1]["stage/picked_top"]


def test_nexarm_stack_bowls_info_reports_stages():
    from lerobot_robot_nexarm.nexarm_sim.stack_bowls_task import STAGES

    released = _scripted_stack_bowls(_SOLVED_SEED, min_open_while_held=1990)
    first, last = released.infos[0], released.infos[-1]
    assert not any(first[f"stage/{stage}"] for stage in STAGES)
    assert first["failed_stage"] == "picked_middle"
    assert all(last[f"stage/{stage}"] for stage in STAGES) and last["failed_stage"] == ""

    holding = _scripted_stack_bowls(_SOLVED_SEED, min_open_while_held=1990, freeze_at_top_release=True)
    assert holding.infos[-1]["stage/nested_top"] and holding.infos[-1]["failed_stage"] == "released_top"


def test_nexarm_env_config_forwards_assist_thresholds_to_stack_bowls_only():
    cfg = NexArmEnv(task="NexArmStackBowls-v0", obs_type="state", assist_release_pos=2000)
    assert cfg.gym_kwargs["assist_release_pos"] == 2000
    assert "assist_engage_pos" not in cfg.gym_kwargs
    env = gym.make("NexArmStackBowls-v0", **cfg.gym_kwargs)
    assert env.unwrapped.sim_task.assist_release_pos == 2000
    env.close()
    assert "assist_release_pos" not in NexArmEnv().gym_kwargs
    with pytest.raises(ValueError, match="NexArmStackBowls"):
        NexArmEnv(assist_release_pos=2000)
    cfg = NexArmEnv(task="NexArmStackBowls-v0", obs_type="state", grasp_mode="contact_assisted")
    env = gym.make("NexArmStackBowls-v0", **cfg.gym_kwargs)
    assert env.unwrapped.sim_task.grasp_mode == "contact_assisted"
    env.close()
    with pytest.raises(ValueError, match="NexArmStackBowls"):
        NexArmEnv(grasp_mode="physical")


def test_stage_log_keeps_only_seeded_episodes(tmp_path):
    import json

    env = gym.make("NexArmStackBowls-v0", obs_type="state", max_episode_steps=2)
    env.unwrapped.log_stages_to(str(tmp_path / "stages"))
    env.reset(seed=7)
    hold = env.unwrapped._get_agent_pos()
    env.step(hold)
    env.step(hold)  # truncated: the seeded episode is written
    env.reset()  # an autoreset-style episode without a seed is not logged
    env.step(hold)
    env.close()

    (log,) = tmp_path.glob("stages.*.jsonl")
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [record["seed"] for record in records] == [7]
    assert records[0]["success"] is False and not any(records[0]["stages"].values())


def test_lerobot_eval_reports_stack_bowls_success_and_stage_rates():
    import mujoco
    from torch import nn

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import PreTrainedPolicy
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot_env_nexarm.stage_eval import eval_policy_all_with_stages

    cfg = NexArmEnv(task="NexArmStackBowls-v0", obs_type="state", episode_length=30)
    envs = make_env(cfg, n_envs=1)
    env = envs["nexarm"][0].envs[0].unwrapped

    class StackThenHold(PreTrainedPolicy):
        """Puts the bowls in a released stack without picking them, then holds the arm still."""

        config_class = PreTrainedConfig
        name = "stack_then_hold"

        def __init__(self):
            nn.Module.__init__(self)  # no config: this stub has no weights

        def reset(self):
            self.stacked = False

        forward = get_optim_params = predict_action_chunk = None

        def select_action(self, observation):
            backend, task = env.backend, env.sim_task
            for i, color in enumerate(task.current_order if not self.stacked else ()):
                qadr = backend.model.jnt_qposadr[task._joint_ids[color]]
                backend.data.qpos[qadr : qadr + 7] = [0, -0.2, 0.02 + 0.03 * i, 1, 0, 0, 0]
            mujoco.mj_forward(backend.model, backend.data)
            self.stacked = True
            return observation["observation.state"].clone()

    class Identity:
        """A pass-through processor; eval resets processors before each batch of episodes."""

        def __call__(self, x):
            return x

        def reset(self):
            pass

    identity = Identity()

    info = eval_policy_all_with_stages(
        eval_policy_all,
        envs=envs,
        policy=StackThenHold(),
        env_preprocessor=identity,
        env_postprocessor=identity,
        preprocessor=identity,
        postprocessor=identity,
        n_episodes=1,
        start_seed=0,
    )
    overall = info["overall"]
    assert overall["pc_success"] == 100.0  # needs info["is_success"], which eval reads
    assert overall["pc_picked_middle"] == 0.0 and overall["pc_picked_top"] == 0.0
    assert overall["pc_nested_middle"] == 100.0 and overall["pc_released_top"] == 100.0
    stages = info["per_task"][0]["metrics"]["stages"]
    assert stages == [{**dict.fromkeys(stages[0], True), "picked_middle": False, "picked_top": False}]
