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

from pathlib import Path

import numpy as np
import pytest

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim.mujoco_backend import HOME_POSITIONS
from lerobot.utils.constants import ACTION, OBS_STR


def create_synthetic_nexarm_dataset(root_dir: Path, repo_id: str, num_frames: int = 20) -> LeRobotDataset:
    """Create a minimal synthetic LeRobotDataset matching NexArmSim features with raw positions."""
    features = {
        ACTION: {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in JOINT_NAMES]},
        f"{OBS_STR}.state": {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in JOINT_NAMES]},
        f"{OBS_STR}.images.front": {
            "dtype": "video",
            "shape": (120, 160, 3),
            "names": ["height", "width", "channel"],
        },
        f"{OBS_STR}.images.wrist": {
            "dtype": "video",
            "shape": (120, 160, 3),
            "names": ["height", "width", "channel"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=root_dir,
        robot_type="nexarm_sim",
        features=features,
        use_videos=True,
        streaming_encoding=False,
    )

    home_vals = np.array([float(HOME_POSITIONS[j]) for j in JOINT_NAMES], dtype=np.float32)

    for i in range(num_frames):
        # Small jitter around home position in raw servo units (~2048)
        jitter = float(np.sin(i * 0.1) * 20.0)
        state = (home_vals + jitter).astype(np.float32)
        action = (home_vals + jitter).astype(np.float32)

        frame = {
            ACTION: action,
            f"{OBS_STR}.state": state,
            f"{OBS_STR}.images.front": np.zeros((120, 160, 3), dtype=np.uint8),
            f"{OBS_STR}.images.wrist": np.zeros((120, 160, 3), dtype=np.uint8),
            "task": "Pick up the red cube",
        }
        dataset.add_frame(frame)

    dataset.save_episode()
    return dataset


def test_gradient_accumulation_config_validation():
    """Verify gradient_accumulation_steps configuration and validation in TrainPipelineConfig."""
    from lerobot.configs.default import DatasetConfig

    dataset_cfg = DatasetConfig(repo_id="dummy/repo", root="outputs/dummy")
    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        gradient_accumulation_steps=2,
    )
    assert train_cfg.gradient_accumulation_steps == 2

    with pytest.raises(ValueError, match="gradient_accumulation_steps must be >= 1"):
        invalid_cfg = TrainPipelineConfig(
            dataset=dataset_cfg,
            gradient_accumulation_steps=0,
        )
        invalid_cfg.validate()


def test_nexarm_sim_gym_raw_position_contract(tmp_path: Path):
    """Verify that a policy trained on raw positions evaluates in Gym without unit mismatch (e.g. jumping to 4095)."""
    from lerobot.envs.configs import NexArmEnv
    from lerobot.envs.factory import make_env

    # 1. Verify NexArmEnv defaults to control_mode="raw"
    env_cfg = NexArmEnv(
        task="NexArmPickPlace-v0",
        obs_type="pixels_agent_pos",
        observation_height=120,
        observation_width=160,
    )
    assert env_cfg.control_mode == "raw"
    assert env_cfg.gym_kwargs["control_mode"] == "raw"

    env_dict = make_env(env_cfg, n_envs=1)
    vec_env = env_dict["nexarm"][0]
    obs, info = vec_env.reset()

    # Raw positions should be around 2048 (not normalized [-1, 1])
    raw_pos = obs["agent_pos"][0]
    assert np.all(raw_pos > 100.0), f"Expected raw servo positions, got {raw_pos}"

    # Step with a raw home action (e.g. 2048)
    home_action = np.full((1, 6), 2048.0, dtype=np.float32)
    next_obs, reward, term, trunc, info = vec_env.step(home_action)

    # In raw mode, the robot shouldn't have jumped to the 4095 limit
    stepped_pos = next_obs["agent_pos"][0]
    for j_idx, name in enumerate(JOINT_NAMES[:-1]):
        # Joints should stay around 2048, NOT saturate to 4095
        assert stepped_pos[j_idx] < 3500.0, (
            f"Joint {name} saturated to {stepped_pos[j_idx]}! "
            f"Expected raw action ~2048 to be applied directly."
        )

    vec_env.close()


def test_nexarm_train_eval_integration(tmp_path: Path):
    """End-to-end integration test: dataset -> train ACT -> Gym eval."""
    from lerobot.configs.default import DatasetConfig, EvalConfig
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import NexArmEnv
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts.lerobot_eval import eval_main
    from lerobot.scripts.lerobot_train import train

    dataset_dir = tmp_path / "dataset"
    train_dir = tmp_path / "train_output"
    eval_dir = tmp_path / "eval_output"
    repo_id = "local/nexarm_integration_test"

    # 1. Create small synthetic dataset with raw servo positions
    create_synthetic_nexarm_dataset(dataset_dir, repo_id=repo_id, num_frames=10)

    # 2. Train a tiny ACT policy for 2 steps with gradient accumulation
    policy_cfg = ACTConfig(
        dim_model=64,
        n_action_steps=4,
        chunk_size=4,
        push_to_hub=False,
        device="cpu",
    )
    dataset_cfg = DatasetConfig(
        repo_id=repo_id,
        root=dataset_dir,
    )
    env_cfg = NexArmEnv(
        task="NexArmPickPlace-v0",
        episode_length=5,
        obs_type="pixels_agent_pos",
        observation_height=120,
        observation_width=160,
    )

    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=train_dir,
        job_name="nexarm_ete_test",
        batch_size=2,
        gradient_accumulation_steps=2,
        steps=2,
        save_freq=2,
        log_freq=1,
        env_eval_freq=0,
        save_checkpoint=True,
    )
    train_cfg.validate()
    train(train_cfg)

    # 3. Verify checkpoint exists
    checkpoint_dir = train_dir / "checkpoints" / "000002" / "pretrained_model"
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "config.json").exists()

    # 4. Evaluate the trained policy with lerobot-eval in NexArmPickPlaceEnv
    eval_cfg = EvalPipelineConfig(
        env=env_cfg,
        eval=EvalConfig(n_episodes=1, batch_size=1),
        policy=PreTrainedConfig.from_pretrained(checkpoint_dir),
        output_dir=eval_dir,
    )
    eval_cfg.policy.pretrained_path = checkpoint_dir
    eval_cfg.policy.device = "cpu"

    eval_main(eval_cfg)

    eval_info_path = eval_dir / "eval_info.json"
    assert eval_info_path.exists()
    import json

    with open(eval_info_path) as f:
        eval_info = json.load(f)
    assert "overall" in eval_info
    assert eval_info["overall"]["n_episodes"] == 1
