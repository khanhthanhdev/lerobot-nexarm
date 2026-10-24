"""LeRobot configuration for TurboVLA policies."""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

# Raw NexArm gripper position separating open from closed (as `rollout_turbovla.py --gripper-threshold`).
DEFAULT_GRIPPER_THRESHOLD = 1800.0


@PreTrainedConfig.register_subclass("turbovla")
@dataclass
class TurboVLAConfig(PreTrainedConfig):
    """TurboVLA (BERT text encoder, DINOv3 vision encoder, ACT-style action head) as a LeRobot policy.

    The defaults reproduce `examples/nexarm/rollout_turbovla.py`. Images are resized by the vision
    backbone's Hugging Face image processor, state and action are min-max scaled to [-1, 1], and the
    model tokenizes the task prompt itself.

    Args:
        chunk_size: Actions predicted per forward pass (TurboVLA's `action.horizon`).
        n_action_steps: Actions executed from each chunk before predicting again
            (`rollout_turbovla.py --open-loop-steps`).
        camera_keys: Image features in the order the model was trained on. Empty means the order of
            `input_features`.
        image_size: Side of the square images fed to the vision backbone.
        text_model_name_or_path: Hugging Face repo or local folder of the BERT text encoder and tokenizer.
        vision_model_name_or_path: Hugging Face repo or local folder of the vision backbone and its image
            processor.
        local_files_only: Load both backbones from the local Hugging Face cache only.
        vision_compute_precision: TurboVLA `vision.compute_precision` (`bf16_autocast` only applies on CUDA).
        interaction_compute_precision: TurboVLA `interaction.compute_precision`.
        interaction_num_layers: Vision-language interaction layers.
        action_num_layers: Action head layers.
        default_task: Prompt used when a batch has no task (the training prompt of the checkpoint).
        binarize_gripper: Snap the gripper action to the dataset's open or closed extreme.
        gripper_index: Index of the gripper in the action vector.
        gripper_threshold: Raw gripper position separating open from closed when snapping.
    """

    chunk_size: int = 16
    n_action_steps: int = 8
    camera_keys: list[str] = field(default_factory=list)
    image_size: int = 224

    text_model_name_or_path: str = "google-bert/bert-base-uncased"
    vision_model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    local_files_only: bool = False
    vision_compute_precision: str = "bf16_autocast"
    interaction_compute_precision: str = "fp32"
    interaction_num_layers: int = 6
    action_num_layers: int = 3

    default_task: str | None = None
    binarize_gripper: bool = True
    gripper_index: int = 5
    gripper_threshold: float = DEFAULT_GRIPPER_THRESHOLD

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Same optimizer settings as examples/nexarm/train_turbovla.py.
    optimizer_lr: float = 5e-5
    optimizer_weight_decay: float = 1e-10

    def __post_init__(self) -> None:
        """Validate the chunking options."""
        super().__post_init__()
        if self.chunk_size < 1:
            raise ValueError(f"`chunk_size` must be positive. Got {self.chunk_size}.")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError(
                f"`n_action_steps` must be in [1, chunk_size={self.chunk_size}]. Got {self.n_action_steps}."
            )

    @property
    def ordered_camera_keys(self) -> list[str]:
        """Image feature keys in the order the model expects its views."""
        return list(self.camera_keys) or list(self.image_features)

    def validate_features(self) -> None:
        """Require a state, an action and at least one camera; `camera_keys` must be image features."""
        if self.robot_state_feature is None:
            raise ValueError("TurboVLA needs an `observation.state` input feature.")
        if self.action_feature is None:
            raise ValueError("TurboVLA needs an `action` output feature.")
        if not self.image_features:
            raise ValueError("TurboVLA needs at least one image input feature.")
        unknown = [key for key in self.camera_keys if key not in self.image_features]
        if unknown:
            raise ValueError(f"`camera_keys` {unknown} are not image features: {list(self.image_features)}")

    @property
    def observation_delta_indices(self) -> None:
        """Only the current observation is used."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """The current action and the next `chunk_size - 1`."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """Rewards are not used."""
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        """AdamW with the TurboVLA NexArm training defaults."""
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        """No scheduler preset."""
        return None
