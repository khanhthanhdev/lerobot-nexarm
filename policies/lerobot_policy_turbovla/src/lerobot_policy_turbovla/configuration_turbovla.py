"""LeRobot configuration for TurboVLA policies."""

import math
from dataclasses import dataclass, field

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, LRSchedulerConfig

# Raw NexArm gripper position separating open from closed (as `rollout_turbovla.py --gripper-threshold`).
DEFAULT_GRIPPER_THRESHOLD = 1800.0
LOSS_TYPES = ("l1", "mse", "smooth_l1")


@LRSchedulerConfig.register_subclass("turbovla_cosine_with_warmup")
@dataclass
class TurboVLASchedulerConfig(LRSchedulerConfig):
    """Linear warmup to the peak learning rate, then cosine decay to `min_lr_ratio` of it.

    The schedule of `examples/nexarm/train_turbovla.py`: the factor is `(step + 1) / num_warmup_steps`
    during warmup, then decays over the remaining `num_training_steps - num_warmup_steps` steps and
    stays at `min_lr_ratio`. Every parameter group keeps its own peak learning rate. lerobot-train
    steps the scheduler once per batch, so with gradient accumulation the schedule counts batches
    where the script counts optimizer updates.
    """

    num_warmup_steps: int = 1000
    min_lr_ratio: float = 0.05

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        """A `LambdaLR` over `num_training_steps` steps."""
        warmup_steps, floor = self.num_warmup_steps, self.min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return floor + (1 - floor) * (1 + math.cos(math.pi * min(progress, 1.0))) / 2

        return LambdaLR(optimizer, lr_lambda)


@PreTrainedConfig.register_subclass("turbovla")
@dataclass
class TurboVLAConfig(PreTrainedConfig):
    """TurboVLA (BERT text encoder, DINOv3 vision encoder, ACT-style action head) as a LeRobot policy.

    The inference defaults reproduce `examples/nexarm/rollout_turbovla.py` and the training defaults
    (loss, frozen modules, optimizer and schedule) reproduce `examples/nexarm/train_turbovla.py`.
    Images are resized by the vision backbone's Hugging Face image processor, state and action are
    min-max scaled to [-1, 1], and the model tokenizes the task prompt itself.

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
        freeze_text: Freeze the BERT encoder (its projection still trains), as `--freeze-text`.
        freeze_vision: Freeze the vision backbone, as `--freeze-vision`.
        freeze_interaction: Freeze the vision-language interaction layers, as `--freeze-interaction`.
        loss_type: Action loss, `l1`, `mse` or `smooth_l1`.
        gripper_loss_weight: Loss multiplier for the action at `gripper_index`.
        optimizer_lr: Learning rate of the action head and of every module except the vision backbone.
        optimizer_vision_lr: Learning rate of the vision backbone (`--dinov3-lr`).
        optimizer_weight_decay: AdamW weight decay.
        optimizer_grad_clip_norm: Gradient norm clipping (`--max-grad-norm`).
        scheduler_warmup_steps: Linear warmup steps before the cosine decay.
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

    freeze_text: bool = True
    freeze_vision: bool = False
    freeze_interaction: bool = False
    loss_type: str = "l1"
    gripper_loss_weight: float = 3.0

    optimizer_lr: float = 5e-5
    optimizer_vision_lr: float = 5e-5
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000

    def __post_init__(self) -> None:
        """Validate the chunking and training options."""
        super().__post_init__()
        if self.chunk_size < 1:
            raise ValueError(f"`chunk_size` must be positive. Got {self.chunk_size}.")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError(
                f"`n_action_steps` must be in [1, chunk_size={self.chunk_size}]. Got {self.n_action_steps}."
            )
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"`loss_type` must be one of {LOSS_TYPES}. Got {self.loss_type!r}.")
        if not math.isfinite(self.gripper_loss_weight) or self.gripper_loss_weight <= 0:
            raise ValueError(f"`gripper_loss_weight` must be positive. Got {self.gripper_loss_weight}.")
        if self.scheduler_warmup_steps < 0:
            raise ValueError(f"`scheduler_warmup_steps` must be >= 0. Got {self.scheduler_warmup_steps}.")

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
        """The current action and the next `chunk_size - 1`: lerobot-train loads `chunk_size`-step chunks."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """Rewards are not used."""
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        """AdamW as in `train_turbovla.py`; the per-group learning rates come from `get_optim_params`."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> TurboVLASchedulerConfig:
        """Warmup then cosine decay to 5%, as in `train_turbovla.py`."""
        return TurboVLASchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)
