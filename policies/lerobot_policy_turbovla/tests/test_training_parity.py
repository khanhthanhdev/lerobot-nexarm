"""Training through lerobot-train must match `examples/nexarm/train_turbovla.py` on recorded NexArm data.

Both pipelines get the same randomly initialised TurboVLA (BERT-tiny text encoder, DINOv2-small vision
encoder, fp32, CPU) and the same frames of `thanhkt/nexarm_stack_bowls`, including frames near the end
of the episode whose action chunks are padded. The script side runs the script's own collator,
normalization statistics, loss, parameter groups and learning-rate schedule; the LeRobot side runs
`make_policy`, `make_pre_post_processors` with the dataset statistics, the policy's optimizer and
scheduler presets and lerobot-train's `update_policy`. Dropout and drop-path are active in training
mode, so the torch RNG is seeded identically before each forward pass.

Images: the script truncates `frame * 255` while the plugin rounds. LeRobot decodes video to
`uint8 / 255` in float32, and for all 256 levels both give back the original byte, so the two agree on
dataset frames; the plugin keeps rounding so training and inference share one image processor.
"""

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("turbovla", reason="install the `turbovla` extra")
pytest.importorskip("lerobot_policy_turbovla", reason="install the `turbovla` extra")

import torch  # noqa: E402
from lerobot_policy_turbovla import TurboVLAConfig  # noqa: E402
from lerobot_policy_turbovla.processor_turbovla import PIXEL_VALUES_KEY  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # the training script is imported as `examples.nexarm.*`
    sys.path.insert(0, str(REPO_ROOT))

from examples.nexarm import train_turbovla  # noqa: E402

DATASET_REPO_ID = "thanhkt/nexarm_stack_bowls"
EPISODE = 143
HORIZON = 16
BATCH_SIZE = 6
LOSS_TOLERANCE = 1e-5
PARAM_TOLERANCE = 1e-6
STEP_LR = 1e-3
WARMUP_STEPS = 2
TOTAL_STEPS = 10


@pytest.fixture(scope="module")
def dataset(nexarm_dataset_root):
    """Episode 143 with `HORIZON`-step action chunks, loaded the way the script loads it."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    try:
        meta = LeRobotDatasetMetadata(DATASET_REPO_ID, nexarm_dataset_root, local_files_only=True)
        delta_timestamps = {ACTION: [i / max(1, meta.fps) for i in range(HORIZON)]}
        return LeRobotDataset(
            DATASET_REPO_ID,
            root=nexarm_dataset_root,
            episodes=[EPISODE],
            delta_timestamps=delta_timestamps,
            local_files_only=True,
        )
    except FileNotFoundError as error:
        pytest.skip(f"Episode {EPISODE} of {DATASET_REPO_ID} is not available locally: {error}")


@pytest.fixture(scope="module")
def items(dataset):
    """Frames spread over the episode, the last two close enough to the end to be padded."""
    indices = [
        *np.linspace(0, len(dataset) - 20, BATCH_SIZE - 2, dtype=int),
        len(dataset) - 6,
        len(dataset) - 1,
    ]
    result = [dataset[int(index)] for index in indices]
    assert sum(bool(item["action_is_pad"].any()) for item in result) == 2
    return result


@pytest.fixture(scope="module")
def camera_keys(dataset):
    return train_turbovla.resolve_camera_keys(dataset.features)


def make_config(tiny_backbones, camera_keys, **overrides) -> TurboVLAConfig:
    text_model, vision_model = tiny_backbones
    kwargs = {
        "text_model_name_or_path": text_model,
        "vision_model_name_or_path": vision_model,
        "vision_compute_precision": "fp32",
        "interaction_compute_precision": "fp32",
        "camera_keys": camera_keys,
        "chunk_size": HORIZON,
        "device": "cpu",
        "push_to_hub": False,
    }
    return TurboVLAConfig(**{**kwargs, **overrides})


@pytest.fixture(scope="module")
def policy(dataset, tiny_backbones, camera_keys):
    """A seeded random TurboVLA built by LeRobot's factory for the dataset's features."""
    config = make_config(
        tiny_backbones,
        camera_keys,
        optimizer_lr=STEP_LR,
        optimizer_vision_lr=STEP_LR / 2,
        scheduler_warmup_steps=WARMUP_STEPS,
    )
    torch.manual_seed(0)
    return make_policy(config, ds_meta=dataset.meta)


@pytest.fixture(scope="module")
def script_batch(dataset, items, camera_keys, tiny_backbones):
    """The script's collated batch: its own statistics, image processor setup and collator."""
    from transformers import AutoImageProcessor

    stats = train_turbovla.compute_or_load_normalization_stats(dataset)
    for key in (ACTION, OBS_STATE):  # the LeRobot normalizer reads the same statistics
        for stat in ("min", "max"):
            np.testing.assert_array_equal(
                np.float32(stats[key][stat]), np.float32(dataset.meta.stats[key][stat])
            )
    image_processor = AutoImageProcessor.from_pretrained(tiny_backbones[1])
    image_processor.size = {"height": 224, "width": 224}
    collator = train_turbovla.TurboVLANexArmCollator(
        camera_keys, stats, image_processor, horizon=HORIZON, dataset_meta=dataset.meta
    )
    return collator(copy.deepcopy(items))


@pytest.fixture(scope="module")
def lerobot_batch(policy, dataset, items):
    """The DataLoader batch of lerobot-train after the policy's preprocessor."""
    preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    return preprocessor(default_collate(copy.deepcopy(items)))


def script_train_mode(model: torch.nn.Module) -> torch.nn.Module:
    """The script's training-loop setup with its default `--freeze-text`."""
    model.train()
    model.text_encoder.eval()
    return model


def script_forward_loss(model, batch, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    predictions = model(batch["instructions"], batch["samples"], batch["states"]).float()
    return train_turbovla.action_loss(predictions, batch["actions"], batch["action_is_pad"], "l1", 3.0)


def test_delta_timestamps_load_the_script_action_chunks(policy, dataset):
    assert resolve_delta_timestamps(policy.config, dataset.meta) == {
        ACTION: [i / dataset.meta.fps for i in range(HORIZON)]
    }


def test_preprocessed_batch_matches_script_collator(lerobot_batch, script_batch):
    assert lerobot_batch["task"] == script_batch["instructions"]
    torch.testing.assert_close(
        lerobot_batch[PIXEL_VALUES_KEY], script_batch["samples"]["dinov3"], rtol=0, atol=0
    )
    torch.testing.assert_close(lerobot_batch[OBS_STATE], script_batch["states"], rtol=0, atol=1e-6)
    torch.testing.assert_close(lerobot_batch[ACTION], script_batch["actions"], rtol=0, atol=1e-6)
    torch.testing.assert_close(lerobot_batch[f"{ACTION}_is_pad"], script_batch["action_is_pad"])


def test_forward_loss_matches_script(policy, lerobot_batch, script_batch):
    script_model = script_train_mode(copy.deepcopy(policy.model))
    policy.train()
    modes = {name: module.training for name, module in policy.model.named_modules()}
    assert modes == {name: module.training for name, module in script_model.named_modules()}

    for seed in (1, 2):
        expected = script_forward_loss(script_model, script_batch, seed)
        torch.manual_seed(seed)
        loss, logs = policy.forward(lerobot_batch)
        diff = abs(loss.item() - expected.item())
        print(
            f"seed {seed}: lerobot loss {loss.item():.8f}, script loss {expected.item():.8f}, diff {diff:.2e}"
        )
        assert diff <= LOSS_TOLERANCE
        assert logs["action_loss"] == pytest.approx(loss.item())

    # The padded steps must be excluded: the same batch without padding gives another loss.
    unpadded = {**lerobot_batch, f"{ACTION}_is_pad": torch.zeros_like(lerobot_batch[f"{ACTION}_is_pad"])}
    torch.manual_seed(1)
    assert (
        abs(policy.forward(unpadded)[0].item() - script_forward_loss(script_model, script_batch, 1).item())
        > 1e-4
    )


def group_names(model: torch.nn.Module, groups: list[dict]) -> list[tuple[float, list[str]]]:
    names = {id(param): name for name, param in model.named_parameters()}
    return [(group["lr"], sorted(names[id(param)] for param in group["params"])) for group in groups]


def test_param_groups_and_frozen_modules_match_script(policy, tiny_backbones, camera_keys, dataset):
    expected = train_turbovla.build_param_groups(policy.model, STEP_LR, STEP_LR / 2, freeze_vision=False)
    actual = policy.get_optim_params()
    assert group_names(policy.model, actual) == group_names(policy.model, expected)
    assert [len(group["params"]) for group in actual] and all(group["params"] for group in actual)
    frozen = {name for name, param in policy.model.named_parameters() if not param.requires_grad}
    assert any(name.startswith("text_encoder.bert.") for name in frozen)
    assert not any(name.startswith("text_encoder.text_projection.") for name in frozen)

    frozen_policy = make_policy(
        make_config(tiny_backbones, camera_keys, freeze_vision=True, freeze_interaction=True),
        ds_meta=dataset.meta,
    )
    model = frozen_policy.model
    for module in (model.vision_encoder, model.vision_language_interaction):
        assert not any(param.requires_grad for param in module.parameters())
    frozen_policy.train()
    assert not model.vision_encoder.training and not model.vision_language_interaction.training
    assert not model.text_encoder.training and model.action_head.training
    groups = frozen_policy.get_optim_params()
    assert len(groups) == 2
    assert group_names(model, groups) == group_names(
        model, train_turbovla.build_param_groups(model, 5e-5, 5e-5, freeze_vision=True)
    )


def test_optimizer_and_schedule_match_script(policy):
    optimizer_config = policy.config.get_optimizer_preset()
    assert optimizer_config.grad_clip_norm == 1.0
    lerobot_optimizer = optimizer_config.build(policy.get_optim_params())
    script_optimizer = AdamW(
        train_turbovla.build_param_groups(policy.model, STEP_LR, STEP_LR / 2, freeze_vision=False),
        weight_decay=1e-10,
    )
    for ours, theirs in zip(lerobot_optimizer.param_groups, script_optimizer.param_groups, strict=True):
        assert {k: v for k, v in ours.items() if k != "params"} == {
            k: v for k, v in theirs.items() if k != "params"
        }

    scheduler = policy.config.get_scheduler_preset().build(lerobot_optimizer, TOTAL_STEPS)
    script_scheduler = LambdaLR(
        script_optimizer, lambda update: train_turbovla.lr_multiplier(update, WARMUP_STEPS, TOTAL_STEPS)
    )
    for _ in range(TOTAL_STEPS + 2):
        for ours, theirs in zip(scheduler.get_last_lr(), script_scheduler.get_last_lr(), strict=True):
            assert math.isclose(ours, theirs, rel_tol=1e-12)
        lerobot_optimizer.step()
        script_optimizer.step()
        scheduler.step()
        script_scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.05 * STEP_LR)


def test_optimizer_steps_match_script(policy, lerobot_batch, script_batch):
    from accelerate import Accelerator

    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    lerobot_policy = copy.deepcopy(policy)
    script_model = script_train_mode(copy.deepcopy(policy.model))
    initial = {name: param.detach().clone() for name, param in script_model.named_parameters()}

    optimizer = lerobot_policy.config.get_optimizer_preset().build(lerobot_policy.get_optim_params())
    scheduler = lerobot_policy.config.get_scheduler_preset().build(optimizer, TOTAL_STEPS)
    accelerator = Accelerator(cpu=True)
    metrics = {name: AverageMeter(name) for name in ("loss", "grad_norm", "lr", "update_s")}
    tracker = MetricsTracker(BATCH_SIZE, 1000, 1, metrics)

    script_optimizer = AdamW(
        train_turbovla.build_param_groups(script_model, STEP_LR, STEP_LR / 2, freeze_vision=False),
        weight_decay=1e-10,
    )
    script_scheduler = LambdaLR(
        script_optimizer, lambda update: train_turbovla.lr_multiplier(update, WARMUP_STEPS, TOTAL_STEPS)
    )

    for step, seed in enumerate((3, 4)):
        torch.manual_seed(seed)
        tracker, _ = update_policy(
            tracker, lerobot_policy, lerobot_batch, optimizer, 1.0, accelerator, lr_scheduler=scheduler
        )
        expected = script_forward_loss(script_model, script_batch, seed)
        expected.backward()
        torch.nn.utils.clip_grad_norm_(script_model.parameters(), 1.0)
        script_optimizer.step()
        script_optimizer.zero_grad(set_to_none=True)
        script_scheduler.step()
        assert abs(tracker.loss.val - expected.item()) <= LOSS_TOLERANCE, step

    max_diff, max_change = 0.0, 0.0
    lerobot_params = dict(lerobot_policy.model.named_parameters())
    for name, param in script_model.named_parameters():
        max_diff = max(max_diff, float((lerobot_params[name] - param).detach().abs().max()))
        max_change = max(max_change, float((param.detach() - initial[name]).abs().max()))
        if not param.requires_grad:
            torch.testing.assert_close(param, initial[name], rtol=0, atol=0)
    print(f"after 2 updates: max param diff {max_diff:.2e}, max param change {max_change:.2e}")
    assert max_change > 10 * PARAM_TOLERANCE  # the updates moved the weights
    assert max_diff <= PARAM_TOLERANCE
