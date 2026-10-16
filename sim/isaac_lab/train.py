"""Train a state-based PPO policy on ``Isaac-NexArm-StackBowls-Direct-v0`` with rsl_rl.

Isaac Lab's own ``scripts/reinforcement_learning/rsl_rl/train.py`` only knows the tasks in ``isaaclab_tasks``;
this script registers the NexArm task first and otherwise does the same.

    ./isaaclab.sh -p sim/isaac_lab/train.py --headless --num_envs 1024
    ./isaaclab.sh -p sim/isaac_lab/train.py --headless --resume logs/rsl_rl/nexarm_stack_bowls/<run>/model_1000.pt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-NexArm-StackBowls-Direct-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--reward_type", choices=("dense", "sparse"), default="dense")
parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to continue from.")
parser.add_argument("--log_dir", type=Path, default=Path("logs/rsl_rl"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import nexarm_isaac  # noqa: E402, F401  (registers the task)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from nexarm_isaac.agents.rsl_rl_ppo_cfg import NexArmStackBowlsPPORunnerCfg  # noqa: E402
from nexarm_isaac.stack_bowls_env import NexArmStackBowlsEnvCfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def main() -> None:
    env_cfg = NexArmStackBowlsEnvCfg()
    env_cfg.seed = args.seed
    env_cfg.reward_type = args.reward_type
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        env_cfg.sim.device = args.device
    agent_cfg = NexArmStackBowlsPPORunnerCfg()
    agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.device is not None:
        agent_cfg.device = args.device

    log_dir = args.log_dir / agent_cfg.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    if args.resume is not None:
        runner.load(str(args.resume))
    print(f"Logging to {log_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
