"""The plugin must give the same actions as `examples/nexarm/rollout_turbovla.py` on recorded NexArm frames.

A small randomly initialised TurboVLA (BERT-tiny text encoder, DINOv2-small vision encoder) is saved the
way `train_turbovla.py` saves checkpoints, with normalization statistics from the real dataset. The
rollout script's `TurboVLAPolicyRunner` loads that checkpoint; `convert_checkpoint` turns it into a
LeRobot policy that is loaded back through LeRobot's factory. Both run in eval mode, in fp32, on CPU.

Set `TURBOVLA_PARITY_CHECKPOINT` to a trained `train_turbovla.py` checkpoint to check it as well
(with `TURBOVLA_PARITY_TEXT_MODEL` / `TURBOVLA_PARITY_VISION_MODEL` when it used other backbones).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("turbovla", reason="install the `turbovla` extra")
pytest.importorskip("lerobot_policy_turbovla", reason="install the `turbovla` extra")

import torch  # noqa: E402
from lerobot_policy_turbovla.convert_checkpoint import convert_checkpoint  # noqa: E402
from lerobot_policy_turbovla.modeling_turbovla import TurboVLAPolicy  # noqa: E402

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_STATE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # the rollout and training scripts are imported as `examples.nexarm.*`
    sys.path.insert(0, str(REPO_ROOT))

from examples.nexarm import train_turbovla  # noqa: E402
from examples.nexarm.rollout_turbovla import TurboVLAPolicyRunner  # noqa: E402

DATASET_REPO_ID = "thanhkt/nexarm_stack_bowls"
EPISODE = 143
NUM_FRAMES = 20
TOLERANCE = 1e-4
TEXT_WEIGHTS = "prajjwal1/bert-tiny"  # has no tokenizer files; bert-base-uncased's tokenizer is added
TOKENIZER = "google-bert/bert-base-uncased"
VISION_MODEL = "facebook/dinov2-with-registers-small"
CAMERA_NAMES = ("front", "wrist")


def dataset_root() -> Path:
    if root := os.environ.get("TURBOVLA_PARITY_DATASET_ROOT"):
        return Path(root)
    snapshots = sorted(
        (HF_LEROBOT_HOME / "hub" / "datasets--thanhkt--nexarm_stack_bowls" / "snapshots").glob("*")
    )
    for snapshot in snapshots:
        if (snapshot / "meta" / "info.json").is_file():
            return snapshot
    pytest.skip(f"{DATASET_REPO_ID} is not in the local cache; set TURBOVLA_PARITY_DATASET_ROOT")


@pytest.fixture(scope="module")
def dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        return LeRobotDataset(DATASET_REPO_ID, root=dataset_root(), episodes=[EPISODE], local_files_only=True)
    except FileNotFoundError as error:
        pytest.skip(f"Episode {EPISODE} of {DATASET_REPO_ID} is not available locally: {error}")


@pytest.fixture(scope="module")
def frames(dataset):
    """20 recorded frames as the robot provides them: uint8 HWC camera images and float32 joints."""
    result = []
    for index in np.linspace(0, len(dataset) - 1, NUM_FRAMES, dtype=int):
        item = dataset[int(index)]
        images = {
            name: (item[f"observation.images.{name}"] * 255).round().clamp(0, 255).to(torch.uint8)
            for name in CAMERA_NAMES
        }
        result.append(
            {
                "images": {name: image.permute(1, 2, 0).numpy() for name, image in images.items()},
                "state": item[OBS_STATE].numpy().astype(np.float32),
                "task": item["task"],
            }
        )
    return result


@pytest.fixture(scope="module")
def tiny_backbones(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    text_dir = tmp_path_factory.mktemp("bert_tiny")
    try:
        transformers.BertModel.from_pretrained(TEXT_WEIGHTS).save_pretrained(text_dir)
        transformers.AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(text_dir)
        transformers.AutoImageProcessor.from_pretrained(VISION_MODEL)
        transformers.AutoModel.from_pretrained(VISION_MODEL)
    except OSError as error:
        pytest.skip(f"Backbones are neither cached nor downloadable: {error}")
    return str(text_dir), VISION_MODEL


@pytest.fixture(scope="module")
def random_checkpoint(tmp_path_factory, dataset, tiny_backbones):
    """A seeded random TurboVLA saved exactly like a `train_turbovla.py` checkpoint."""
    from turbovla.models import TurboVLAConfig as TurboVLAModelConfig, build_turbovla

    text_model, vision_model = tiny_backbones
    camera_keys = train_turbovla.resolve_camera_keys(dataset.features)
    assert camera_keys == [f"observation.images.{name}" for name in CAMERA_NAMES]
    config = TurboVLAModelConfig()
    config.text.model_name_or_path = text_model
    config.vision.model_name_or_path = vision_model
    config.vision.num_views = len(camera_keys)
    config.vision.image_size = 224
    config.vision.compute_precision = "fp32"
    config.interaction.compute_precision = "fp32"
    config.action.action_dim = 6
    config.action.state_dim = 6
    config.action.horizon = 16
    torch.manual_seed(0)
    model = build_turbovla(config)
    stats = train_turbovla.compute_or_load_normalization_stats(dataset)
    metadata = train_turbovla.checkpoint_metadata(["real"], [dataset.meta], camera_keys, "auto")
    output_dir = tmp_path_factory.mktemp("turbovla_checkpoint")
    train_turbovla.save_checkpoint(
        model, None, config, stats, output_dir, 0, is_final=True, metadata=metadata
    )
    return output_dir / "final_model.pt", text_model, vision_model


def checkpoints(request) -> tuple[Path, str, str]:
    if request.param == "random":
        return request.getfixturevalue("random_checkpoint")
    path = os.environ.get("TURBOVLA_PARITY_CHECKPOINT")
    if not path:
        pytest.skip("TURBOVLA_PARITY_CHECKPOINT is not set")
    text_model = os.environ.get("TURBOVLA_PARITY_TEXT_MODEL", "google-bert/bert-base-uncased")
    vision_model = os.environ.get("TURBOVLA_PARITY_VISION_MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    return Path(path), text_model, vision_model


@pytest.fixture(scope="module", params=["random", "trained"])
def both_paths(request, tmp_path_factory):
    """The rollout script's runner and the converted LeRobot policy with its processors."""
    checkpoint, text_model, vision_model = checkpoints(request)
    runner = TurboVLAPolicyRunner(
        checkpoint, None, device="cpu", bert_path=text_model, vision_path=vision_model, binarize_gripper=False
    )
    policy_dir = convert_checkpoint(
        checkpoint,
        tmp_path_factory.mktemp("turbovla_lerobot"),
        text_model=text_model,
        vision_model=vision_model,
        local_files_only=True,
        device="cpu",
    )
    policy = TurboVLAPolicy.from_pretrained(policy_dir)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=policy_dir)
    _, continuous_postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_dir,
        postprocessor_overrides={"turbovla_gripper_snap": {"enabled": False}},
    )
    assert not runner.model.training and not policy.training
    return runner, policy, preprocessor, postprocessor, continuous_postprocessor


def policy_batch(frame: dict) -> dict:
    observation = {f"observation.images.{name}": image.copy() for name, image in frame["images"].items()}
    observation[OBS_STATE] = frame["state"].copy()
    return prepare_observation_for_inference(observation, torch.device("cpu"), task=frame["task"])


def test_action_chunks_match_rollout_script(both_paths, frames):
    runner, policy, preprocessor, postprocessor, continuous_postprocessor = both_paths
    max_diff = 0.0
    first_actions = []
    for frame in frames:
        runner.binarize_gripper = False
        expected = runner.predict_chunk(frame["images"], frame["state"], task=frame["task"])
        runner.binarize_gripper = True
        expected_snapped = runner.predict_chunk(frame["images"], frame["state"], task=frame["task"])

        chunk = policy.predict_action_chunk(preprocessor(policy_batch(frame)))
        actual = continuous_postprocessor(chunk)[0].numpy()
        actual_snapped = postprocessor(chunk)[0].numpy()

        assert actual.shape == expected.shape == (16, 6)
        first_actions.append(expected[0])
        max_diff = max(max_diff, float(np.abs(actual - expected).max()))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=TOLERANCE)
        np.testing.assert_array_equal(actual_snapped[:, 5], expected_snapped[:, 5])
        np.testing.assert_allclose(actual_snapped[:, :5], expected_snapped[:, :5], rtol=0, atol=TOLERANCE)
    # The frames must drive the model: identical outputs would make this comparison vacuous.
    assert np.ptp(np.stack(first_actions), axis=0).max() > 1.0
    print(f"max abs action difference over {len(frames)} frames: {max_diff:.3e}")


def test_select_action_executes_the_first_n_action_steps(both_paths, frames):
    runner, policy, preprocessor, postprocessor, _ = both_paths
    frame = frames[len(frames) // 2]
    runner.binarize_gripper = True
    expected = runner.predict_chunk(frame["images"], frame["state"], task=frame["task"])
    policy.reset()
    batch = preprocessor(policy_batch(frame))
    actions = [
        postprocessor(policy.select_action(batch))[0].numpy() for _ in range(policy.config.n_action_steps)
    ]
    np.testing.assert_allclose(
        np.stack(actions), expected[: policy.config.n_action_steps], rtol=0, atol=TOLERANCE
    )
    assert policy.count_queued_actions() == 0
