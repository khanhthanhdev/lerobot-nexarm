"""TurboVLA policy: wraps the `turbovla` model for LeRobot's evaluation and rollout loops."""

import builtins
from collections import deque
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import Tensor
from turbovla.models import TurboVLAConfig as TurboVLAModelConfig, build_turbovla

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE

from .configuration_turbovla import TurboVLAConfig
from .processor_turbovla import PIXEL_VALUES_KEY

T = TypeVar("T", bound="TurboVLAPolicy")


def make_model_config(config: TurboVLAConfig) -> TurboVLAModelConfig:
    """TurboVLA model config for a policy config, built the way `rollout_turbovla.py` builds it."""
    state, action = config.robot_state_feature, config.action_feature
    if state is None or action is None:
        raise ValueError("TurboVLA needs `observation.state` and `action` features.")
    model_config = TurboVLAModelConfig()
    model_config.text.model_name_or_path = config.text_model_name_or_path
    model_config.text.local_files_only = config.local_files_only
    model_config.vision.model_name_or_path = config.vision_model_name_or_path
    model_config.vision.local_files_only = config.local_files_only
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
    """TurboVLA inference through LeRobot: action chunks are queued and executed `n_action_steps` at a time.

    Building the policy loads both pretrained backbones (`text_model_name_or_path`,
    `vision_model_name_or_path`); `from_pretrained` then replaces every weight with the checkpoint's.
    """

    config_class = TurboVLAConfig
    name = "turbovla"

    def __init__(self, config: TurboVLAConfig, **kwargs: Any) -> None:
        """Build the TurboVLA model for `config`."""
        super().__init__(config)
        config.validate_features()
        self.config: TurboVLAConfig = config
        self.model = build_turbovla(make_model_config(config))
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T], pretrained_name_or_path: str | Path, *, strict: bool = True, **kwargs: Any
    ) -> T:
        """Load a converted checkpoint; every weight must match, as in `rollout_turbovla.py`."""
        return super().from_pretrained(pretrained_name_or_path, strict=strict, **kwargs)

    def get_optim_params(self) -> list[torch.nn.Parameter]:
        """Trainable parameters."""
        return [param for param in self.parameters() if param.requires_grad]

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

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Normalized action chunk of shape (batch, chunk_size, action_dim)."""
        self.eval()
        states = batch[OBS_STATE]
        if states.ndim == 3:  # (batch, n_obs_steps=1, state_dim)
            states = states[:, -1]
        instructions = self._instructions(batch, states.shape[0])
        return self.model(instructions, {"dinov3": batch[PIXEL_VALUES_KEY]}, states)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Next normalized action of shape (batch, action_dim), predicting a new chunk when the queue is empty."""
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Training is not available through LeRobot yet."""
        raise NotImplementedError(
            "Training TurboVLA with lerobot-train is not supported yet. Train with "
            "examples/nexarm/train_turbovla.py, then convert the checkpoint with "
            "`python -m lerobot_policy_turbovla.convert_checkpoint`."
        )
