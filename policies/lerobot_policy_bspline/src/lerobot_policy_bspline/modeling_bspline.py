"""B-spline Policy (arXiv 2607.09648): LeRobot's diffusion policy denoising B-spline action chunks.

Upstream (`bspline_policy/policy/diffusion_unet_bspline_image_policy.py`) subclasses the diffusion
UNet image policy, adds one model channel for the knot column and returns the whole denoised
parameter chunk. This module does the same on LeRobot's `DiffusionPolicy`, and also moves the
action-to-spline fit (upstream does it in its dataset) into `forward`, because LeRobot datasets
yield raw action windows.

With `bspline_mode` set to `onestep` or `planbind` the policy instead regresses control points on a fixed
knot grid (`planbind.py`) and runs them through `SplineRuntime`, which simulates inference latency.
"""

import copy
from collections import deque

import torch
from torch import Tensor

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

from .bspline import decode_param_batch, encode_action_batch
from .configuration_bspline import BsplineConfig
from .planbind import PlanBindModel, SplineRuntime


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
        self.runtime: SplineRuntime | None = None
        if config.bspline_mode == "diffusion":
            self.diffusion = BsplineModel(_model_config(config))
        else:
            self.diffusion = PlanBindModel(config)
            self.runtime = SplineRuntime(self.diffusion, config)
        self.reset()

    def reset(self):
        """Clear the observation and action queues and the spline runtime state."""
        super().reset()
        if getattr(self, "runtime", None) is not None:
            self.runtime.reset()

    def _global_cond(self, batch: dict[str, Tensor]) -> Tensor:
        """Observation features of a dataloader-style batch (one or `n_obs_steps` frames per key)."""
        batch = dict(batch)
        if self.config.image_features:
            for key in self.config.image_features:
                if batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        if batch[OBS_STATE].ndim == 2:
            batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)
        return self.diffusion._prepare_global_conditioning(batch)

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
        """Actions ``(B, n, action_dim)`` sampled from the predicted spline, starting at the current frame.

        In `onestep` and `planbind` modes this is one stateless call (a fresh plan for `planbind`), decoded
        for `n_action_steps` frames.
        """
        if self.runtime is None:
            return self.decode_params(self.predict_action_params(batch, noise=noise))
        model = self.diffusion
        assert isinstance(model, PlanBindModel)
        global_cond = self._global_cond(batch)
        if self.config.bspline_mode == "onestep":
            ctrl = model.onestep(global_cond)
        else:
            zero = torch.zeros(global_cond.shape[0], dtype=torch.long, device=global_cond.device)
            lifted = model.lifted_plan(model.plan(global_cond), zero)
            ctrl = lifted + model.bind(lifted, zero, global_cond)
        n = min(self.config.n_action_steps, self.config.bspline_window)
        return torch.einsum("tn,bnd->btd", model.fine_basis[:n], ctrl)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        """Next action; `onestep` and `planbind` run through `SplineRuntime` (latency, commit, projection)."""
        if self.runtime is None:
            return super().select_action(batch, noise=noise)
        batch = dict(batch)
        batch.pop(ACTION, None)
        if self.config.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        def global_cond() -> Tensor:
            stacked = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            return self.diffusion._prepare_global_conditioning(stacked)

        return self.runtime.step(global_cond)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Fit the raw action window into B-spline parameters, then compute the diffusion loss.

        `onestep` and `planbind` regress the raw action window directly (`PlanBindModel.compute_loss`).
        """
        if self.runtime is not None:
            return DiffusionPolicy.forward(self, batch)
        batch = dict(batch)
        params = self.encode_actions(batch[ACTION], batch.get("action_is_pad"))
        batch[ACTION] = params
        batch["action_is_pad"] = torch.zeros(params.shape[:2], dtype=torch.bool, device=params.device)
        return super().forward(batch)
