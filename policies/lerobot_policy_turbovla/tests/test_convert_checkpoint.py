"""`convert_checkpoint` turns a checkpoint in the former `train_turbovla.py` format into a LeRobot policy.

That format is a folder with `final_model.pt` (or `final_ema_pytorch_model.pt`) holding the model's
`state_dict`, `stats_turbovla.json` with min/max statistics, and `config.json` with the training
contract (camera keys, fps, prompt, horizon, image size). A seeded random TurboVLA (BERT-tiny,
DINOv2-small, fp32, CPU) is saved that way; the converted folder must load through LeRobot's factory with
identical weights, the saved contract and the saved statistics, and predict what the original model does.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("turbovla", reason="install the `turbovla` extra")
pytest.importorskip("lerobot_policy_turbovla", reason="install the `turbovla` extra")

import torch  # noqa: E402
from lerobot_policy_turbovla.convert_checkpoint import convert_checkpoint  # noqa: E402
from lerobot_policy_turbovla.modeling_turbovla import TurboVLAPolicy  # noqa: E402
from lerobot_policy_turbovla.processor_turbovla import PIXEL_VALUES_KEY  # noqa: E402

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CAMERA_KEYS = ["observation.images.front", "observation.images.wrist"]
TASK = "Stack the bowls with red on bottom, blue in middle, and black on top."
STATS = {
    ACTION: {"min": [-1.5, -1.0, -2.0, -1.2, -0.5, 1195.0], "max": [1.5, 1.0, 0.5, 1.2, 0.5, 2833.0]},
    OBS_STATE: {"min": [-1.6, -1.1, -2.1, -1.3, -0.6, 1100.0], "max": [1.6, 1.1, 0.6, 1.3, 0.6, 2900.0]},
}


def build_model(tiny_backbones):
    from turbovla.models import TurboVLAConfig as TurboVLAModelConfig, build_turbovla

    text_model, vision_model = tiny_backbones
    config = TurboVLAModelConfig()
    config.text.model_name_or_path = text_model
    config.vision.model_name_or_path = vision_model
    config.vision.num_views = len(CAMERA_KEYS)
    config.vision.image_size = 224
    config.vision.compute_precision = "fp32"
    config.interaction.compute_precision = "fp32"
    config.action.action_dim = 6
    config.action.state_dim = 6
    config.action.horizon = 16
    torch.manual_seed(0)
    return build_turbovla(config).eval()


def save_script_checkpoint(model, folder: Path, *, contract: dict, prefix: str = "") -> Path:
    """Write `model` the way `train_turbovla.py` wrote its final checkpoint."""
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "final_model.pt"
    torch.save({f"{prefix}{key}": value for key, value in model.state_dict().items()}, checkpoint)
    (folder / "stats_turbovla.json").write_text(json.dumps(STATS))
    config = {
        "model_name": "TurboVLA",
        **contract,
        "action": {"action_dim": 6, "state_dim": 6, "horizon": 16},
        "vision": {"num_views": len(CAMERA_KEYS), "image_size": 224},
    }
    (folder / "config.json").write_text(json.dumps(config))
    return checkpoint


@pytest.fixture(scope="module")
def model(tiny_backbones):
    return build_model(tiny_backbones)


@pytest.fixture(scope="module")
def converted(model, tiny_backbones, tmp_path_factory):
    folder = tmp_path_factory.mktemp("script_checkpoint")
    contract = {"task_type": "stack_bowls", "camera_keys": CAMERA_KEYS, "fps": 30, "task": TASK}
    checkpoint = save_script_checkpoint(model, folder, contract=contract)
    text_model, vision_model = tiny_backbones
    return convert_checkpoint(
        checkpoint,
        tmp_path_factory.mktemp("lerobot_policy"),
        text_model=text_model,
        vision_model=vision_model,
        local_files_only=True,
        device="cpu",
    )


def test_converted_policy_has_the_checkpoint_weights_and_contract(model, converted):
    policy = TurboVLAPolicy.from_pretrained(converted)
    assert policy.config.camera_keys == CAMERA_KEYS
    assert policy.config.chunk_size == 16 and policy.config.n_action_steps == 8
    assert policy.config.image_size == 224
    assert policy.config.default_task == TASK
    expected = model.state_dict()
    actual = policy.model.state_dict()
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0, msg=key)


def test_converted_processors_use_the_checkpoint_statistics(converted):
    policy = TurboVLAPolicy.from_pretrained(converted)
    _, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=converted,
        postprocessor_overrides={"turbovla_gripper_snap": {"enabled": False}},
    )
    # -1 and 1 are the ends of the min-max range.
    for normalized, stat in ((-1.0, "min"), (1.0, "max")):
        action = postprocessor(torch.full((1, 6), normalized))[0].numpy()
        np.testing.assert_allclose(action, STATS[ACTION][stat], rtol=0, atol=1e-3)


def test_converted_policy_predicts_what_the_checkpoint_model_predicts(model, converted):
    policy = TurboVLAPolicy.from_pretrained(converted)
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=converted)
    rng = np.random.default_rng(0)
    observation = {key: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for key in CAMERA_KEYS}
    observation[OBS_STATE] = np.array([0.1, -0.2, -0.5, 0.3, 0.0, 2000.0], dtype=np.float32)
    batch = preprocessor(prepare_observation_for_inference(observation, torch.device("cpu"), task=TASK))

    chunk = policy.predict_action_chunk(batch)
    with torch.no_grad():
        expected = model([TASK], {"dinov3": batch[PIXEL_VALUES_KEY]}, batch[OBS_STATE])
    assert chunk.shape == (1, 16, 6)
    torch.testing.assert_close(chunk, expected, rtol=0, atol=1e-6)


def test_legacy_checkpoint_without_camera_keys_and_with_a_wrapper_prefix(model, tiny_backbones, tmp_path):
    checkpoint = save_script_checkpoint(model, tmp_path / "legacy", contract={}, prefix="module.")
    text_model, vision_model = tiny_backbones
    converted = convert_checkpoint(
        checkpoint,
        tmp_path / "lerobot_policy",
        text_model=text_model,
        vision_model=vision_model,
        local_files_only=True,
        device="cpu",
    )
    policy = TurboVLAPolicy.from_pretrained(converted)
    assert policy.config.camera_keys == CAMERA_KEYS  # front/wrist by view count
    assert policy.config.default_task is None
    torch.testing.assert_close(
        policy.model.action_head.state_dict(), model.action_head.state_dict(), rtol=0, atol=0
    )
