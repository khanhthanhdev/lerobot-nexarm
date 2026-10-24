"""Pre/post-processors for TurboVLA, matching `examples/nexarm/rollout_turbovla.py`.

Pre-processing: batch, min-max scale the state to [-1, 1], then run the camera frames, in the trained
camera order, through the vision backbone's Hugging Face image processor (as uint8 PIL images,
resized to `image_size`). The task prompt passes through: TurboVLA tokenizes it inside the model.
Post-processing: undo the action min-max scaling, then optionally snap the gripper to the dataset's
open or closed extreme.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ObservationProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION
from lerobot_robot_nexarm.motors.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS

from .configuration_turbovla import DEFAULT_GRIPPER_THRESHOLD, TurboVLAConfig

# Observation key holding the image processor output, shape (batch, views, 3, image_size, image_size).
PIXEL_VALUES_KEY = "observation.turbovla_pixel_values"


def to_uint8_frame(image: torch.Tensor) -> np.ndarray:
    """(3, H, W) float image in [0, 1] (or uint8) to the (H, W, 3) uint8 frame the camera produced."""
    if image.dtype != torch.uint8:
        image = (image.detach().float() * 255.0).round().clamp(0, 255).to(torch.uint8)
    return image.permute(1, 2, 0).cpu().numpy()


@ProcessorStepRegistry.register(name="turbovla_image_processor")
@dataclass
class TurboVLAImageProcessorStep(ObservationProcessorStep):
    """Turn the camera images into TurboVLA pixel values with the backbone's Hugging Face image processor."""

    camera_keys: list[str] = field(default_factory=list)
    vision_model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    image_size: int = 224
    local_files_only: bool = False

    def __post_init__(self) -> None:
        """Defer loading the image processor until the first call."""
        self._image_processor: Any = None

    @property
    def image_processor(self) -> Any:
        """The backbone's image processor with the explicit square resize used in training."""
        if self._image_processor is None:
            from transformers import AutoImageProcessor

            processor = AutoImageProcessor.from_pretrained(
                self.vision_model_name_or_path, local_files_only=self.local_files_only
            )
            if hasattr(processor, "size"):
                processor.size = {"height": self.image_size, "width": self.image_size}
            self._image_processor = processor
        return self._image_processor

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Replace the camera images with `PIXEL_VALUES_KEY`."""
        missing = [key for key in self.camera_keys if key not in observation]
        if missing:
            raise ValueError(
                f"Observation is missing camera(s) {missing}; TurboVLA expects {self.camera_keys}."
            )
        images = [observation.pop(key) for key in self.camera_keys]
        batch_size = images[0].shape[0]
        frames = [
            Image.fromarray(to_uint8_frame(images[view][index]))
            for index in range(batch_size)
            for view in range(len(images))
        ]
        pixel_values = self.image_processor(images=frames, return_tensors="pt")["pixel_values"]
        observation[PIXEL_VALUES_KEY] = pixel_values.view(batch_size, len(images), *pixel_values.shape[1:])
        return observation

    def get_config(self) -> dict[str, Any]:
        """Serializable step options."""
        return {
            "camera_keys": list(self.camera_keys),
            "vision_model_name_or_path": self.vision_model_name_or_path,
            "image_size": self.image_size,
            "local_files_only": self.local_files_only,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """The pixel values are internal to the policy; declared features are unchanged."""
        return features


@ProcessorStepRegistry.register(name="turbovla_gripper_snap")
@dataclass
class TurboVLAGripperSnapStep(PolicyActionProcessorStep):
    """Snap the unnormalized gripper action to the closed or open extreme on each side of `threshold`.

    Partial gripper targets make grasps fail on NexArm, so `rollout_turbovla.py` snaps by default.
    """

    enabled: bool = True
    gripper_index: int = 5
    threshold: float = DEFAULT_GRIPPER_THRESHOLD
    closed_value: float | None = None
    open_value: float | None = None
    # Whether larger raw positions are more closed (true when GRIPPER_CLOSED_POS > GRIPPER_OPEN_POS).
    closed_is_high: bool = GRIPPER_CLOSED_POS > GRIPPER_OPEN_POS

    def action(self, action: PolicyAction) -> PolicyAction:
        """Snapped copy of `action` (..., action_dim)."""
        if not self.enabled or action.shape[-1] <= self.gripper_index:
            return action
        if self.closed_value is None or self.open_value is None:
            raise ValueError("Gripper snapping needs the action min/max statistics; none were given.")
        gripper = action[..., self.gripper_index]
        is_closed = gripper > self.threshold if self.closed_is_high else gripper < self.threshold
        action = action.clone()
        action[..., self.gripper_index] = torch.where(
            is_closed,
            torch.tensor(self.closed_value, dtype=action.dtype, device=action.device),
            torch.tensor(self.open_value, dtype=action.dtype, device=action.device),
        )
        return action

    def get_config(self) -> dict[str, Any]:
        """Serializable step options."""
        return {
            "enabled": self.enabled,
            "gripper_index": self.gripper_index,
            "threshold": self.threshold,
            "closed_value": self.closed_value,
            "open_value": self.open_value,
            "closed_is_high": self.closed_is_high,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Shapes are unchanged."""
        return features


def gripper_extremes(
    dataset_stats: dict[str, dict[str, Any]] | None, gripper_index: int, closed_is_high: bool
) -> tuple[float | None, float | None]:
    """(closed, open) raw gripper values from the action min/max statistics, or (None, None)."""
    stats = (dataset_stats or {}).get(ACTION, {})
    if "min" not in stats or "max" not in stats:
        return None, None
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    if low.shape[-1] <= gripper_index:
        return None, None
    low_value, high_value = float(low[..., gripper_index]), float(high[..., gripper_index])
    return (high_value, low_value) if closed_is_high else (low_value, high_value)


def make_turbovla_pre_post_processors(
    config: TurboVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Rename, batch, normalize the state, build pixel values, move to device; then unnormalize and snap."""
    steps = make_default_policy_processor_steps(config, dataset_stats)
    gripper_step = TurboVLAGripperSnapStep(
        enabled=config.binarize_gripper,
        gripper_index=config.gripper_index,
        threshold=config.gripper_threshold,
    )
    gripper_step.closed_value, gripper_step.open_value = gripper_extremes(
        dataset_stats, config.gripper_index, gripper_step.closed_is_high
    )
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.normalize,
        TurboVLAImageProcessorStep(
            camera_keys=config.ordered_camera_keys,
            vision_model_name_or_path=config.vision_model_name_or_path,
            image_size=config.image_size,
            local_files_only=config.local_files_only,
        ),
        steps.to_device,
    ]
    output_steps: list[ProcessorStep] = [steps.unnormalize, gripper_step, steps.to_cpu]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
