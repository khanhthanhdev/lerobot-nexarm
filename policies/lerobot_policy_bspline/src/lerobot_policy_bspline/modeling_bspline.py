"""B-spline Policy (arXiv 2607.09648): LeRobot's diffusion policy denoising B-spline action chunks.

Upstream (`bspline_policy/policy/diffusion_unet_bspline_image_policy.py`) subclasses the diffusion
UNet image policy, adds one model channel for the knot column and returns the whole denoised
parameter chunk. This module does the same on LeRobot's `DiffusionPolicy`, and also moves the
action-to-spline fit (upstream does it in its dataset) into `forward`, because LeRobot datasets
yield raw action windows.
"""

import copy
from collections import deque

import torch
from torch import Tensor

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import require_package

from .bspline import decode_param_batch, encode_action_batch
from .configuration_bspline import BsplineConfig


def _model_config(config: BsplineConfig) -> BsplineConfig:
    """Config seen by the diffusion model: the action feature gains one channel for the knots."""
    action_feature = config.action_feature
    if action_feature is None:
        raise ValueError("B-spline Policy requires an action output feature (FeatureType.ACTION).")
    model_config = copy.copy(config)
    model_config.output_features = {
        **config.output_features,
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_feature.shape[0] + 1,)),
    }
    return model_config


class BsplineModel(DiffusionModel):
    """Diffusion model that returns the full denoised parameter chunk (no `n_action_steps` slice)."""

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Denoise a ``(B, horizon, 1 + action_dim)`` parameter chunk conditioned on the observations."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps
        global_cond = self._prepare_global_conditioning(batch)
        return self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)


class BsplinePolicy(DiffusionPolicy):
    """Diffusion policy over B-spline parameter chunks ``(horizon, 1 + action_dim)``."""

    config_class = BsplineConfig
    name = "bspline"

    def __init__(self, config: BsplineConfig, **kwargs):
        """Build the diffusion model with one extra action channel for the knot column."""
        require_package("diffusers", extra="diffusion")
        # Skip DiffusionPolicy.__init__: it would build a model with the raw action width.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues: dict[str, deque[Tensor]] = {}
        self.diffusion = BsplineModel(_model_config(config))
        self.reset()

    def encode_actions(self, actions: Tensor, action_is_pad: Tensor | None = None) -> Tensor:
        """Fit normalized raw action windows ``(B, T, D)`` into parameter chunks."""
        return encode_action_batch(
            actions,
            action_is_pad,
            chunk_size=self.config.bspline_chunk_size,
            degree=self.config.bspline_degree,
            max_error=self.config.bspline_max_error,
            time_scale=self.config.bspline_time_scale,
            num_workers=self.config.bspline_fit_workers,
        )

    def decode_params(self, params: Tensor) -> Tensor:
        """Sample parameter chunks ``(B, horizon, 1 + D)`` into actions ``(B, n, D)``."""
        return decode_param_batch(
            params,
            degree=self.config.bspline_degree,
            time_scale=self.config.bspline_time_scale,
            n_actions=self.config.n_action_steps,
            speed_up=self.config.bspline_speed_up,
        )

    @torch.no_grad()
    def predict_action_params(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        """Denoised B-spline parameter chunk ``(B, horizon, 1 + action_dim)``."""
        return super().predict_action_chunk(batch, noise=noise)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        """Actions ``(B, n, action_dim)`` sampled from the predicted spline, starting at the current frame."""
        return self.decode_params(self.predict_action_params(batch, noise=noise))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Fit the raw action window into B-spline parameters, then compute the diffusion loss."""
        batch = dict(batch)
        params = self.encode_actions(batch[ACTION], batch.get("action_is_pad"))
        batch[ACTION] = params
        batch["action_is_pad"] = torch.zeros(params.shape[:2], dtype=torch.bool, device=params.device)
        return super().forward(batch)
