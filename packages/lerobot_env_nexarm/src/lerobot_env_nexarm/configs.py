"""Config for the NexArm MuJoCo gym environments (`--env.type=nexarm`)."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE


@EnvConfig.register_subclass("nexarm")
@dataclass
class NexArmEnv(EnvConfig):
    # NexArmPickPlace-v0 (cube into the target zone) or NexArmStackBowls-v0 (the real-robot task).
    task: str | None = "NexArmPickPlace-v0"
    fps: int = 30
    # None takes the task's own length: 400 steps for pick-place, 900 for stack-bowls.
    episode_length: int | None = None
    obs_type: str = "pixels_agent_pos"
    control_mode: str = "raw"
    observation_height: int = 480
    observation_width: int = 640
    reward_type: str = "dense"
    render_mode: str = "rgb_array"
    # SimCalibration JSON from tools/nexarm/calibrate_sim.py; its fps must equal ``fps``.
    calibration_path: str | None = None
    # Real-to-sim joint mapping JSON from tools/nexarm/probe_joint_mapping.py; None is the identity.
    joint_mapping_path: str | None = None
    enable_domain_randomization: bool = False
    # None takes the calibrated latency (or 0); an explicit value overrides the calibration.
    action_delay_steps: int | None = None
    # Uniform noise (rad) on the five arm joints around home at every reset; 0 starts exactly at home.
    init_qpos_noise_rad: float = 0.0
    # Stack-bowls grasp assist: raw gripper commands above engage may attach a bowl, below release drop it.
    # None keeps the env defaults (2400 and 2300; 2300 lets go at real operators' partial openings).
    assist_engage_pos: float | None = None
    assist_release_pos: float | None = None
    # Stack-bowls grasp: None keeps the env default ("physical", friction only); "contact_assisted" carries
    # a bowl rigidly once both jaws touch it.
    grasp_mode: str | None = None
    # Stack-bowls order as bottom,middle,top (e.g. "red,blue,black") for policies trained on a single-order
    # dataset (generate_stack_bowls_dataset.py --order); None cycles all six orders by seed.
    stack_order: str | None = None
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "front": f"{OBS_IMAGE}.front",
            "wrist": f"{OBS_IMAGE}.wrist",
            "top": f"{OBS_IMAGE}.top",
            "pixels/front": f"{OBS_IMAGES}.front",
            "pixels/wrist": f"{OBS_IMAGES}.wrist",
            "pixels/top": f"{OBS_IMAGES}.top",
            "environment_state": OBS_ENV_STATE,
        }
    )

    def __post_init__(self):
        assist_set = any(
            value is not None
            for value in (self.assist_engage_pos, self.assist_release_pos, self.grasp_mode, self.stack_order)
        )
        if assist_set and "StackBowls" not in self.gym_id:
            raise ValueError(
                "assist_engage_pos, assist_release_pos, grasp_mode and stack_order apply only to "
                "NexArmStackBowls-v0"
            )
        if self.obs_type in ("pixels_agent_pos", "state"):
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
        if self.obs_type in ("pixels_agent_pos", "pixels"):
            self.features["pixels/front"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features["pixels/wrist"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features["pixels/top"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        if self.obs_type == "state":
            # Cube and target positions, or the bottom, middle and top bowl positions.
            state_dim = 9 if "StackBowls" in self.gym_id else 6
            self.features["environment_state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))

    @property
    def package_name(self) -> str:
        return "lerobot_env_nexarm.nexarm"

    @property
    def gym_id(self) -> str:
        return self.task or "NexArmPickPlace-v0"

    @property
    def gym_kwargs(self) -> dict:
        kwargs = {
            "obs_type": self.obs_type,
            "control_mode": self.control_mode,
            "render_mode": self.render_mode,
            "reward_type": self.reward_type,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
            "fps": self.fps,
            "calibration_path": self.calibration_path,
            "joint_mapping_path": self.joint_mapping_path,
            "enable_domain_randomization": self.enable_domain_randomization,
            "action_delay_steps": self.action_delay_steps,
            "init_qpos_noise_rad": self.init_qpos_noise_rad,
        }
        if self.episode_length is not None:
            kwargs["max_episode_steps"] = self.episode_length
        for key in ("assist_engage_pos", "assist_release_pos", "grasp_mode", "stack_order"):
            if getattr(self, key) is not None:
                kwargs[key] = getattr(self, key)
        return kwargs
