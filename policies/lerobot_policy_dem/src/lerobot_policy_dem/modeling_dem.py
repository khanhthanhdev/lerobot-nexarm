"""DEM policy: wraps the `dem` model for LeRobot's training, evaluation and rollout loops."""

import builtins
import contextlib
import contextvars
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeVar

import torch
from dem import DEM, DEMConfig as DEMModelConfig, preprocess_images
from dem.head import MeanFlowTokenHead
from dem.language.neobert import load_tokenizer
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_dem import DEMConfig
from .head_dem import GlobalCondMeanFlowHead

T = TypeVar("T", bound="DEMPolicy")

# False while `from_pretrained` builds the policy: the checkpoint replaces every weight, so the towers are
# built randomly instead of downloading the (gated) Hub weights first.
_LOAD_BACKBONE_WEIGHTS = contextvars.ContextVar("dem_load_backbone_weights", default=True)


def make_model_config(config: DEMConfig, load_backbone_weights: bool = True) -> DEMModelConfig:
    """`dem` model config for a policy config."""
    state, action = config.robot_state_feature, config.action_feature
    if state is None or action is None:
        raise ValueError("DEM needs `observation.state` and `action` features.")
    return DEMModelConfig(
        vision_encoder=config.vision_encoder,
        vision_model_name=config.vision_model_name,
        vision_feature_stage=config.vision_feature_stage,
        vision_pretrained=load_backbone_weights,
        freeze_vision=config.freeze_vision,
        image_size=config.image_size,
        num_cameras=len(config.ordered_camera_keys),
        language_model=config.language_model,
        language_backend=config.language_backend,
        language_pretrained=load_backbone_weights,
        max_language_tokens=config.max_language_tokens,
        head_size=config.head_size,
        proprio_dim=state.shape[0],
        action_dim=action.shape[0],
        chunk=config.chunk_size,
        ratio_r_eq_t=config.ratio_r_eq_t,
        dispersive_weight=config.dispersive_weight,
        dispersive_tau=config.dispersive_tau,
    )


class DEMPolicy(PreTrainedPolicy):
    """DEM through LeRobot.

    Inference: one MeanFlow head pass per action chunk; chunks are queued and executed `n_action_steps`
    at a time. Training: `forward` returns the MeanFlow loss (with the dispersive regularizer) and
    `get_optim_params` the head and vision parameter groups. State and action arrive z-scored from the
    pre-processor, so the model's own normalizers stay the identity.
    """

    config_class = DEMConfig
    name = "dem"

    def __init__(self, config: DEMConfig, **kwargs: Any) -> None:
        """Build the DEM model for `config`, loading the pretrained towers unless a checkpoint follows."""
        super().__init__(config)
        config.validate_features()
        self.config: DEMConfig = config
        load_weights = _LOAD_BACKBONE_WEIGHTS.get()
        self.model = DEM(make_model_config(config, load_backbone_weights=load_weights))
        if not load_weights and config.language_backend == "neobert":
            self.model.language.tokenizer = load_tokenizer(config.language_model)
        if config.global_conditioning:
            head = self.model.head
            dim, depth, heads = MeanFlowTokenHead.SIZES[config.head_size]
            self.model.head = GlobalCondMeanFlowHead(
                head.vis_dim,
                head.lang_dim,
                head.proprio_dim,
                head.action_dim,
                head.chunk,
                dim=dim,
                depth=depth,
                heads=heads,
                num_language_queries=config.num_language_queries,
            )
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T], pretrained_name_or_path: str | Path, *, strict: bool = True, **kwargs: Any
    ) -> T:
        """Load a checkpoint strictly, without downloading the pretrained towers it overwrites."""
        token = _LOAD_BACKBONE_WEIGHTS.set(False)
        try:
            return super().from_pretrained(pretrained_name_or_path, strict=strict, **kwargs)
        finally:
            _LOAD_BACKBONE_WEIGHTS.reset(token)

    def get_optim_params(self) -> list[dict[str, Any]]:
        """The head at `optimizer_lr`, the vision encoder (unless frozen) at `optimizer_vision_lr`."""
        return self.model.parameter_groups(
            lr_head=self.config.optimizer_lr, lr_vision=self.config.optimizer_vision_lr
        )

    def reset(self) -> None:
        """Clear the action queue; call when the environment resets."""
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def _instructions(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        tasks = batch.get("task")
        if tasks is None:
            tasks = [None] * batch_size
        elif isinstance(tasks, str):
            tasks = [tasks] * batch_size
        return [task or self.config.default_task or "" for task in tasks]

    def _states(self, batch: dict[str, Any]) -> Tensor:
        states = batch[OBS_STATE]
        return states[:, -1] if states.ndim == 3 else states  # (batch, n_obs_steps=1, state_dim)

    @contextlib.contextmanager
    def _vision_autocast(self) -> Iterator[None]:
        device_type = self.model.device.type
        with torch.autocast(
            device_type,
            dtype=torch.bfloat16,
            enabled=self.config.vision_bf16_autocast and device_type == "cuda",
        ):
            yield

    def _vision_tokens(self, batch: dict[str, Any]) -> Tensor:
        """Camera frames (batch, 3, H, W) in [0, 1], in `ordered_camera_keys` order, to vision tokens."""
        keys = self.config.ordered_camera_keys
        missing = [key for key in keys if key not in batch]
        if missing:
            raise ValueError(f"Batch is missing camera(s) {missing}; DEM expects {keys}.")
        frames = torch.stack([batch[key] for key in keys], dim=1)
        images = preprocess_images(
            frames.clamp(0.0, 1.0), size=self.config.image_size, device=self.model.device
        )
        with self._vision_autocast():
            return self.model.encode_vision(images)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Normalized action chunk of shape (batch, chunk_size, action_dim)."""
        self.eval()
        states = self._states(batch)
        vis_tokens = self._vision_tokens(batch)
        with torch.autocast(self.model.device.type, enabled=False):
            return self.model.act(
                None,
                states,
                texts=self._instructions(batch, states.shape[0]),
                vis_tokens=vis_tokens,
                nfe=self.config.num_inference_steps,
            )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Next normalized action of shape (batch, action_dim), predicting a new chunk when the queue is empty."""
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        """MeanFlow loss on a preprocessed batch with normalized `action` chunks of `chunk_size` steps.

        The vision encoder runs under bf16 autocast; the head, its Jacobian-vector product and the loss
        run in fp32 even when lerobot-train autocasts the whole step. Like the `dem` objective, the loss
        covers every step of the chunk: steps past the episode end repeat its last action.
        """
        actions = batch[ACTION]
        expected = (self.config.chunk_size, self.model.config.action_dim)
        if actions.ndim != 3 or tuple(actions.shape[1:]) != expected:
            raise ValueError(
                f"Expected `{ACTION}` chunks of shape (batch, {expected[0]}, {expected[1]}), got "
                f"{tuple(actions.shape)}; the dataset must be loaded with the policy's `action_delta_indices`."
            )
        states = self._states(batch)
        vis_tokens = self._vision_tokens(batch)
        with torch.autocast(self.model.device.type, enabled=False):
            loss, aux = self.model.loss(
                None,
                states,
                actions,
                texts=self._instructions(batch, states.shape[0]),
                vis_tokens=vis_tokens,
            )
        return loss, {
            "meanflow_mse": aux["raw_mse"],
            **({"dispersive": aux["disp"]} if "disp" in aux else {}),
        }
