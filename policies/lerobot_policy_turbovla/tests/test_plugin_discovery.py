import importlib.metadata

import pytest

pytest.importorskip("turbovla", reason="install the `turbovla` extra")
pytest.importorskip("lerobot_policy_turbovla", reason="install the `turbovla` extra")

import torch  # noqa: E402
from lerobot_policy_turbovla import TurboVLAConfig  # noqa: E402
from lerobot_policy_turbovla.modeling_turbovla import TurboVLAPolicy  # noqa: E402
from lerobot_policy_turbovla.processor_turbovla import (  # noqa: E402
    TurboVLAGripperSnapStep,
    TurboVLAImageProcessorStep,
    to_uint8_frame,
)

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig  # noqa: E402
from lerobot.motors.nexarm.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

CAMERAS = ["observation.images.wrist", "observation.images.front"]


def make_config(**overrides) -> TurboVLAConfig:
    kwargs = {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        "camera_keys": CAMERAS,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return TurboVLAConfig(**kwargs)


def make_stats() -> dict[str, dict[str, torch.Tensor]]:
    low = torch.tensor([1323.0, 1164.0, 942.0, 873.0, 571.0, 1195.0])
    high = torch.tensor([2650.0, 2679.0, 3119.0, 3182.0, 2243.0, 2833.0])
    return {OBS_STATE: {"min": low, "max": high}, ACTION: {"min": low, "max": high}}


def test_plugin_is_discovered():
    assert importlib.metadata.metadata("lerobot_policy_turbovla")["Name"] == "lerobot_policy_turbovla"
    register_third_party_plugins()
    assert "turbovla" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("turbovla") is TurboVLAConfig
    assert get_policy_class("turbovla") is TurboVLAPolicy


def test_config_validation_and_camera_order():
    config = make_config()
    config.validate_features()
    assert config.ordered_camera_keys == CAMERAS
    assert make_config(camera_keys=[]).ordered_camera_keys == list(reversed(CAMERAS))
    assert config.action_delta_indices == list(range(16))
    with pytest.raises(ValueError, match="n_action_steps"):
        make_config(n_action_steps=17)
    with pytest.raises(ValueError, match="not image features"):
        make_config(camera_keys=["observation.images.top"]).validate_features()


def test_factory_builds_processors_that_round_trip(tmp_path):
    config = make_config()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=make_stats())
    names = [type(step).__name__ for step in preprocessor.steps + postprocessor.steps]
    assert "TurboVLAImageProcessorStep" in names and "TurboVLAGripperSnapStep" in names

    postprocessor.save_pretrained(tmp_path)
    reloaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename="policy_postprocessor.json", to_transition=None, to_output=None
    )
    snap = next(step for step in reloaded.steps if isinstance(step, TurboVLAGripperSnapStep))
    assert (snap.closed_value, snap.open_value) == (
        (2833.0, 1195.0) if GRIPPER_CLOSED_POS > GRIPPER_OPEN_POS else (1195.0, 2833.0)
    )


def test_gripper_snap_matches_threshold_direction():
    step = TurboVLAGripperSnapStep(
        threshold=1800.0, closed_value=2833.0, open_value=1195.0, closed_is_high=True
    )
    action = torch.tensor([[0.0] * 5 + [1700.0], [0.0] * 5 + [1900.0]])
    assert step.action(action)[:, 5].tolist() == [1195.0, 2833.0]
    assert action[:, 5].tolist() == [1700.0, 1900.0]  # input untouched
    assert torch.equal(TurboVLAGripperSnapStep(enabled=False).action(action), action)
    with pytest.raises(ValueError, match="statistics"):
        TurboVLAGripperSnapStep().action(action)


def test_float_frames_map_back_to_camera_bytes():
    frame = torch.randint(0, 256, (3, 4, 5), dtype=torch.uint8)
    as_float = frame.to(torch.float32) / 255
    assert (to_uint8_frame(as_float) == frame.permute(1, 2, 0).numpy()).all()


def test_image_step_requires_every_trained_camera():
    step = TurboVLAImageProcessorStep(camera_keys=CAMERAS)
    with pytest.raises(ValueError, match="missing camera"):
        step.observation({"observation.images.front": torch.zeros(1, 3, 4, 4)})
