"""Pre/post-processors for DEM.

Pre-processing: rename, batch, z-score the state, move to device. Camera frames stay float in [0, 1]
and are resized inside the policy with `dem.preprocess_images`; the task passes through as text and
DEM's language encoder tokenizes it. Post-processing: undo the action z-scoring.
"""

from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_dem import DEMConfig


def make_dem_pre_post_processors(
    config: DEMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Rename, batch, normalize, move to device; then unnormalize and move to CPU."""
    steps = make_default_policy_processor_steps(config, dataset_stats)
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.normalize,
        steps.to_device,
    ]
    output_steps: list[ProcessorStep] = [steps.unnormalize, steps.to_cpu]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
