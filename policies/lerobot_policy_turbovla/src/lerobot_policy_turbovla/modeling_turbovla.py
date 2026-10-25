"""TurboVLA policy: wraps the `turbovla` model for LeRobot's training, evaluation and rollout loops."""

import builtins
from collections import deque
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from turbovla.models import TurboVLAConfig as TurboVLAModelConfig, build_turbovla

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_turbovla import TurboVLAConfig
from .processor_turbovla import PIXEL_VALUES_KEY

T = TypeVar("T", bound="TurboVLAPolicy")

ACTION_IS_PAD = f"{ACTION}_is_pad"


def make_model_config(config: TurboVLAConfig) -> TurboVLAModelConfig:
    """TurboVLA model config for a policy config, built the way the NexArm TurboVLA scripts build it."""
    state, action = config.robot_state_feature, config.action_feature
    if state is None or action is None:
        raise ValueError("TurboVLA needs `observation.state` and `action` features.")
    model_config = TurboVLAModelConfig()
    model_config.text.model_name_or_path = config.text_model_name_or_path
    model_config.text.local_files_only = config.local_files_only
    model_config.text.frozen = config.freeze_text
    model_config.vision.model_name_or_path = config.vision_model_name_or_path
    model_config.vision.local_files_only = config.local_files_only
    model_config.vision.frozen = config.freeze_vision
    model_config.vision.num_views = len(config.ordered_camera_keys)
    model_config.vision.image_size = config.image_size
    model_config.vision.compute_precision = config.vision_compute_precision
    model_config.interaction.compute_precision = config.interaction_compute_precision
    model_config.interaction.num_layers = config.interaction_num_layers
    model_config.action.num_layers = config.action_num_layers
    model_config.action.action_dim = action.shape[0]
    model_config.action.state_dim = state.shape[0]
    model_config.action.horizon = config.chunk_size
    # Re-run the model config's own validation on the edited fields.
    return TurboVLAModelConfig.from_mapping(model_config.to_dict())


class TurboVLAPolicy(PreTrainedPolicy):
    """TurboVLA through LeRobot.

    Inference: action chunks are queued and executed `n_action_steps` at a time. Training: `forward`
    returns the masked action loss of the former `train_turbovla.py`, and `get_optim_params` its
    parameter groups.
    Building the policy loads both pretrained backbones (`text_model_name_or_path`,
    `vision_model_name_or_path`); `from_pretrained` then replaces every weight with the checkpoint's.
    """

    config_class = TurboVLAConfig
    name = "turbovla"

    def __init__(self, config: TurboVLAConfig, **kwargs: Any) -> None:
        """Build the TurboVLA model for `config` and freeze the configured modules."""
        super().__init__(config)
        config.validate_features()
        self.config: TurboVLAConfig = config
        self.model = build_turbovla(make_model_config(config))
        for module in self._frozen_modules():
            if module is not self.model.text_encoder:  # the model freezes BERT itself, not its projection
                module.requires_grad_(False)
        self.train(self.training)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T], pretrained_name_or_path: str | Path, *, strict: bool = True, **kwargs: Any
    ) -> T:
        """Load a checkpoint strictly: every weight must match, so trained and untrained weights never mix."""
        return super().from_pretrained(pretrained_name_or_path, strict=strict, **kwargs)

    def _frozen_modules(self) -> list[torch.nn.Module]:
        frozen = (
            (self.config.freeze_text, self.model.text_encoder),
            (self.config.freeze_vision, self.model.vision_encoder),
            (self.config.freeze_interaction, self.model.vision_language_interaction),
        )
        return [module for is_frozen, module in frozen if is_frozen]

    def train(self, mode: bool = True) -> "TurboVLAPolicy":
        """Set the training mode; frozen modules always stay in eval mode.

        lerobot-train calls `train()` before every update, so the frozen modules are switched back here.
        """
        super().train(mode)
        if mode:
            for module in self._frozen_modules():
                module.eval()
        return self

    def get_optim_params(self) -> list[dict[str, Any]]:
        """Action head, everything else except the vision backbone, then the vision backbone.

        Each group carries its own learning rate; frozen parameters are left out.
        """
        model, lr = self.model, self.config.optimizer_lr
        groups = [
            {"params": [p for p in model.action_head.parameters() if p.requires_grad], "lr": lr},
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "action_head" not in n and "vision_encoder" not in n and p.requires_grad
                ],
                "lr": lr,
            },
        ]
        if not self.config.freeze_vision:
            groups.append(
                {
                    "params": [p for p in model.vision_encoder.parameters() if p.requires_grad],
                    "lr": self.config.optimizer_vision_lr,
                }
            )
        return groups

    def reset(self) -> None:
        """Clear the action queue; call when the environment resets."""
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def _instructions(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        tasks = batch.get("task")
        if tasks is None:
            tasks = [None] * batch_size
        elif isinstance(tasks, str):
            tasks = [tasks]
        return [task or self.config.default_task or "" for task in tasks]

    def _predict(self, batch: dict[str, Any]) -> Tensor:
        states = batch[OBS_STATE]
        if states.ndim == 3:  # (batch, n_obs_steps=1, state_dim)
            states = states[:, -1]
        instructions = self._instructions(batch, states.shape[0])
        return self.model(instructions, {"dinov3": batch[PIXEL_VALUES_KEY]}, states)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Normalized action chunk of shape (batch, chunk_size, action_dim)."""
        self.eval()
        return self._predict(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Next normalized action of shape (batch, action_dim), predicting a new chunk when the queue is empty."""
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        """Action loss on a preprocessed batch with normalized `action` chunks of `chunk_size` steps.

        The loss is computed per element (`loss_type`), the gripper action is weighted by
        `gripper_loss_weight`, padded steps past the episode end (`action_is_pad`) are excluded, and the
        sum is divided by the number of valid steps times the action dimension.
        """
        predictions = self._predict(batch).float()
        targets = batch[ACTION]
        if targets.shape != predictions.shape:
            raise ValueError(
                f"Expected `{ACTION}` chunks of shape {tuple(predictions.shape)}, got {tuple(targets.shape)}; "
                "the dataset must be loaded with the policy's `action_delta_indices`."
            )
        if self.config.loss_type == "mse":
            raw_diff = (predictions - targets) ** 2
        elif self.config.loss_type == "smooth_l1":
            raw_diff = F.smooth_l1_loss(predictions, targets, reduction="none")
        else:
            raw_diff = torch.abs(predictions - targets)

        action_dim = targets.shape[-1]
        if self.config.gripper_loss_weight != 1.0 and action_dim > self.config.gripper_index:
            weights = torch.ones(action_dim, device=targets.device, dtype=targets.dtype)
            weights[self.config.gripper_index] = self.config.gripper_loss_weight
            raw_diff = raw_diff * weights.view(1, 1, -1)

        is_pad = batch.get(ACTION_IS_PAD)
        if is_pad is None:
            is_pad = torch.zeros(targets.shape[:2], dtype=torch.bool, device=targets.device)
        valid = ~is_pad.to(device=targets.device, dtype=torch.bool)
        loss = raw_diff.masked_fill(~valid.unsqueeze(-1), 0).sum() / (valid.sum().clamp_min(1) * action_dim)
        return loss, {"action_loss": loss.item()}
