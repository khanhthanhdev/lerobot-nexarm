"""LeRobot configuration for DEM policies."""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig

VISION_ENCODERS = ("dinov3_convnext_base", "dinov3_vit_base")
HEAD_SIZES = ("small", "new100m", "large")


@PreTrainedConfig.register_subclass("dem")
@dataclass
class DEMConfig(PreTrainedConfig):
    """DEM (DINOv3 ConvNeXt-B vision, frozen NeoBERT language, MeanFlow action head) as a LeRobot policy.

    The model defaults are the paper's final configuration (the `dem` package README): stage-3 ConvNeXt
    tokens at 256 px fine-tuned with the head, 32 NeoBERT tokens, the 107M `new100m` head, one-step
    MeanFlow sampling, dispersive weight 0.25 and `ratio_r_eq_t = 0.5`, with the vision encoder under bf16
    autocast and the head and loss in fp32. Two defaults differ, because a run with the paper's recipe
    ignored the instruction and the state on NexArm stack-bowls and its MeanFlow error stayed unstable
    until the learning rate had decayed (the plugin README has the numbers):

    - `global_conditioning` feeds the instruction and the state into every head block through AdaLN
      (`head_dem.GlobalCondMeanFlowHead`), on top of the paper's cross-attention. `false` is the paper.
    - The learning rates are half the paper's (5e-5 head, 2.5e-5 vision) with a 2000-step warmup. The
      paper's are `--policy.optimizer_lr=1e-4 --policy.optimizer_vision_lr=5e-5 --policy.scheduler_warmup_steps=0`.

    State and action are z-scored by LeRobot's normalizer, so the `dem` model's own normalizers stay the
    identity. The paper's two-stage scheme (q01/q99 scaling, then z-scoring) is an affine map per
    dimension, which makes it the same as z-scoring the raw values.

    Args:
        chunk_size: Actions predicted per forward pass (DEM's `chunk`).
        n_action_steps: Actions executed from each chunk before predicting again (open-loop steps).
        camera_keys: Image features in the order the model sees them. Empty means the order of
            `input_features`.
        image_size: Side of the square frames fed to the vision encoder.
        vision_encoder: `dinov3_convnext_base` (paper) or `dinov3_vit_base` (the vision ablation, needs timm).
        vision_model_name: Hub id or timm name of the encoder; None is the encoder's default.
        vision_feature_stage: ConvNeXt tokens from `stage3` (16 x 16 x 512 at 256 px) or `final`.
        freeze_vision: Keep the vision encoder frozen instead of fine-tuning it with the head.
        vision_bf16_autocast: Run the vision encoder under bf16 autocast on CUDA.
        language_model: Hub id or folder of the language encoder.
        language_backend: `neobert` (the in-package NeoBERT port) or `hf` (any Hugging Face encoder).
        max_language_tokens: Instruction tokens kept, special tokens included.
        head_size: MeanFlow head configuration, `small`, `new100m` (paper) or `large`.
        ratio_r_eq_t: Fraction of training samples with `r = t` (plain flow matching).
        dispersive_weight: Weight of the dispersive regularizer on the mid-head hidden state.
        dispersive_tau: Temperature of the dispersive regularizer.
        num_inference_steps: MeanFlow steps per action chunk (the paper uses 1).
        global_conditioning: Also condition every head block on the instruction and the state through
            AdaLN. Checkpoints trained without it need `false`.
        num_language_queries: Learned queries pooling the instruction tokens for `global_conditioning`.
        default_task: Instruction used when a batch has no task.
        optimizer_lr: Learning rate of the action head.
        optimizer_vision_lr: Learning rate of the vision encoder (and of the language encoder if trained).
        optimizer_betas: AdamW betas.
        optimizer_weight_decay: AdamW weight decay.
        optimizer_grad_clip_norm: Gradient norm clipping.
        scheduler_warmup_steps: Linear warmup steps before the cosine decay.
        scheduler_decay_steps: Length of the cosine decay; shortened to `--steps` when that is smaller.
        scheduler_decay_lr: Head learning rate at the end of the decay; every group decays by the same factor.
    """

    chunk_size: int = 16
    n_action_steps: int = 8
    camera_keys: list[str] = field(default_factory=list)
    image_size: int = 256

    vision_encoder: str = "dinov3_convnext_base"
    vision_model_name: str | None = None
    vision_feature_stage: str = "stage3"
    freeze_vision: bool = False
    vision_bf16_autocast: bool = True

    language_model: str = "chandar-lab/NeoBERT"
    language_backend: str = "neobert"
    max_language_tokens: int = 32

    head_size: str = "new100m"
    ratio_r_eq_t: float = 0.5
    dispersive_weight: float = 0.25
    dispersive_tau: float = 0.5
    num_inference_steps: int = 1
    global_conditioning: bool = True
    num_language_queries: int = 4

    default_task: str | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 5e-5
    optimizer_vision_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 2000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self) -> None:
        """Validate the chunking, model and training options."""
        super().__post_init__()
        if self.chunk_size < 1:
            raise ValueError(f"`chunk_size` must be positive. Got {self.chunk_size}.")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError(
                f"`n_action_steps` must be in [1, chunk_size={self.chunk_size}]. Got {self.n_action_steps}."
            )
        if self.vision_encoder not in VISION_ENCODERS:
            raise ValueError(
                f"`vision_encoder` must be one of {VISION_ENCODERS}. Got {self.vision_encoder!r}."
            )
        if self.head_size not in HEAD_SIZES:
            raise ValueError(f"`head_size` must be one of {HEAD_SIZES}. Got {self.head_size!r}.")
        if self.num_inference_steps < 1:
            raise ValueError(f"`num_inference_steps` must be positive. Got {self.num_inference_steps}.")
        if self.num_language_queries < 1:
            raise ValueError(f"`num_language_queries` must be positive. Got {self.num_language_queries}.")
        if not 0.0 <= self.ratio_r_eq_t <= 1.0:
            raise ValueError(f"`ratio_r_eq_t` must be in [0, 1]. Got {self.ratio_r_eq_t}.")
        if self.scheduler_warmup_steps < 0 or self.scheduler_decay_steps < 1:
            raise ValueError("`scheduler_warmup_steps` must be >= 0 and `scheduler_decay_steps` >= 1.")

    @property
    def ordered_camera_keys(self) -> list[str]:
        """Image feature keys in the order the model expects its views."""
        return list(self.camera_keys) or list(self.image_features)

    def validate_features(self) -> None:
        """Require a state, an action and at least one camera; `camera_keys` must be image features."""
        if self.robot_state_feature is None:
            raise ValueError("DEM needs an `observation.state` input feature.")
        if self.action_feature is None:
            raise ValueError("DEM needs an `action` output feature.")
        if not self.image_features:
            raise ValueError("DEM needs at least one image input feature.")
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
        """AdamW of the paper's runs; `get_optim_params` sets the per-group rates."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Cosine decay (after optional warmup); each group keeps its own peak rate."""
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )
