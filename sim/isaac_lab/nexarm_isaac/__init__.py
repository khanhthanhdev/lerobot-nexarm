"""Isaac Lab tasks for the NexArm, built from the same design as the MuJoCo sim.

Importing this package registers the Gymnasium ids; it does not import Isaac Lab until an environment is made.
"""

try:
    import gymnasium as gym
except ImportError:  # building the bowl USDs with only usd-core installed
    gym = None

if gym is not None:
    gym.register(
        id="Isaac-NexArm-StackBowls-Direct-v0",
        entry_point=f"{__name__}.stack_bowls_env:NexArmStackBowlsEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.stack_bowls_env:NexArmStackBowlsEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NexArmStackBowlsPPORunnerCfg",
        },
    )
