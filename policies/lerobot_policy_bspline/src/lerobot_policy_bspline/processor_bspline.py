"""Pre/post-processors for B-spline Policy.

Normalization is the same as LeRobot's diffusion policy. The action <-> B-spline conversion runs
inside `BsplinePolicy` on normalized actions: `forward` fits raw action windows into parameter
chunks and `select_action` returns actions sampled from the predicted spline, so the standard
action unnormalizer applies unchanged.
"""

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_bspline import BsplineConfig


def make_bspline_pre_post_processors(
    config: BsplineConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Rename, normalize, batch and move to device; on output, move to CPU and unnormalize."""
    return make_default_pre_post_processors(config, dataset_stats)
