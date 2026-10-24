"""Training TurboVLA through lerobot-train on recorded NexArm data.

A seeded random TurboVLA (BERT-tiny text encoder, DINOv2-small vision encoder, fp32, CPU) is built by
`make_policy` for frames of `thanhkt/nexarm_stack_bowls`, including frames near the end of the episode
whose action chunks are padded. The checks pin the training contract that the 2k-step GPU run matched
against the former `train_turbovla.py` (see `experiments/261024-turbovla-nexarm-parity/`): 16-step
action chunks, the masked L1 loss with the gripper weighted 3x, three parameter groups with frozen BERT,
AdamW and the warmup-cosine schedule, and lerobot-train's `update_policy` moving only trainable weights.
Dropout and drop-path are active in training mode, so the torch RNG is seeded before each forward pass.
"""

import copy
import math

import numpy as np
import pytest

pytest.importorskip("turbovla", reason="install the `turbovla` extra")
pytest.importorskip("lerobot_policy_turbovla", reason="install the `turbovla` extra")

import torch  # noqa: E402
from lerobot_policy_turbovla import TurboVLAConfig  # noqa: E402
from lerobot_policy_turbovla.processor_turbovla import PIXEL_VALUES_KEY  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

DATASET_REPO_ID = "thanhkt/nexarm_stack_bowls"
EPISODE = 143
CAMERA_KEYS = ["observation.images.front", "observation.images.wrist"]
HORIZON = 16
BATCH_SIZE = 6
GRIPPER_WEIGHT = 3.0
STEP_LR = 1e-3
WARMUP_STEPS = 2
TOTAL_STEPS = 10


@pytest.fixture(scope="module")
def dataset(nexarm_dataset_root):
    """Episode 143 with `HORIZON`-step action chunks."""
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


def make_config(tiny_backbones, **overrides) -> TurboVLAConfig:
    text_model, vision_model = tiny_backbones
    kwargs = {
        "text_model_name_or_path": text_model,
        "vision_model_name_or_path": vision_model,
        "vision_compute_precision": "fp32",
        "interaction_compute_precision": "fp32",
        "camera_keys": CAMERA_KEYS,
        "chunk_size": HORIZON,
        "device": "cpu",
        "push_to_hub": False,
    }
    return TurboVLAConfig(**{**kwargs, **overrides})


@pytest.fixture(scope="module")
def policy(dataset, tiny_backbones):
    """A seeded random TurboVLA built by LeRobot's factory for the dataset's features."""
    config = make_config(
        tiny_backbones,
        optimizer_lr=STEP_LR,
        optimizer_vision_lr=STEP_LR / 2,
        scheduler_warmup_steps=WARMUP_STEPS,
    )
    torch.manual_seed(0)
    return make_policy(config, ds_meta=dataset.meta)


@pytest.fixture(scope="module")
def batch(policy, dataset, items):
    """The DataLoader batch of lerobot-train after the policy's preprocessor."""
    preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    return preprocessor(default_collate(copy.deepcopy(items)))


def masked_l1(predictions: torch.Tensor, targets: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    """L1 per element, gripper (index 5) weighted, summed over valid steps / (valid steps * action dim)."""
    weights = torch.ones(targets.shape[-1])
    weights[5] = GRIPPER_WEIGHT
    valid = (~is_pad).unsqueeze(-1).float()
    return ((predictions - targets).abs() * weights * valid).sum() / (valid.sum() * targets.shape[-1])


def test_delta_timestamps_load_action_chunks(policy, dataset):
    assert resolve_delta_timestamps(policy.config, dataset.meta) == {
        ACTION: [i / dataset.meta.fps for i in range(HORIZON)]
    }


def test_preprocessed_batch_shapes(batch):
    assert batch[PIXEL_VALUES_KEY].shape == (BATCH_SIZE, len(CAMERA_KEYS), 3, 224, 224)
    assert batch[ACTION].shape == (BATCH_SIZE, HORIZON, 6)
    assert batch[f"{ACTION}_is_pad"].shape == (BATCH_SIZE, HORIZON)
    # Min-max scaled to [-1, 1] with the dataset statistics.
    assert batch[OBS_STATE].abs().max() <= 1 + 1e-5
    assert batch[ACTION][~batch[f"{ACTION}_is_pad"]].abs().max() <= 1 + 1e-5


def test_forward_loss_masks_padding_and_weights_the_gripper(policy, batch):
    policy.train()
    for seed in (1, 2):
        torch.manual_seed(seed)
        predictions = policy._predict(batch).float()
        expected = masked_l1(predictions, batch[ACTION], batch[f"{ACTION}_is_pad"])
        torch.manual_seed(seed)
        loss, logs = policy.forward(batch)
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
        assert logs["action_loss"] == pytest.approx(loss.item())

    # The padded steps must be excluded: the same batch without padding gives another loss.
    unpadded = {**batch, f"{ACTION}_is_pad": torch.zeros_like(batch[f"{ACTION}_is_pad"])}
    torch.manual_seed(1)
    padded_loss = policy.forward(batch)[0].item()
    torch.manual_seed(1)
    assert abs(policy.forward(unpadded)[0].item() - padded_loss) > 1e-4


def group_names(model: torch.nn.Module, groups: list[dict]) -> list[tuple[float, list[str]]]:
    names = {id(param): name for name, param in model.named_parameters()}
    return [(group["lr"], sorted(names[id(param)] for param in group["params"])) for group in groups]


def test_param_groups_and_frozen_modules(policy, tiny_backbones, dataset):
    model = policy.model
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    head = sorted(name for name in trainable if name.startswith("action_head."))
    vision = sorted(name for name in trainable if name.startswith("vision_encoder."))
    rest = sorted(trainable - set(head) - set(vision))
    assert head and vision and rest
    assert group_names(model, policy.get_optim_params()) == [
        (STEP_LR, head),
        (STEP_LR, rest),
        (STEP_LR / 2, vision),
    ]
    frozen = {name for name, param in model.named_parameters() if not param.requires_grad}
    assert any(name.startswith("text_encoder.bert.") for name in frozen)
    assert not any(name.startswith("text_encoder.text_projection.") for name in frozen)
    policy.train()
    assert not model.text_encoder.training and model.action_head.training

    frozen_policy = make_policy(
        make_config(tiny_backbones, freeze_vision=True, freeze_interaction=True), ds_meta=dataset.meta
    )
    model = frozen_policy.model
    for module in (model.vision_encoder, model.vision_language_interaction):
        assert not any(param.requires_grad for param in module.parameters())
    frozen_policy.train()
    assert not model.vision_encoder.training and not model.vision_language_interaction.training
    assert not model.text_encoder.training and model.action_head.training
    groups = frozen_policy.get_optim_params()
    assert len(groups) == 2 and all(group["params"] for group in groups)


def test_optimizer_and_schedule_presets(policy):
    optimizer_config = policy.config.get_optimizer_preset()
    assert optimizer_config.grad_clip_norm == 1.0
    optimizer = optimizer_config.build(policy.get_optim_params())
    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["lr"] for group in optimizer.param_groups] == [STEP_LR, STEP_LR, STEP_LR / 2]
    assert all(group["weight_decay"] == 1e-10 for group in optimizer.param_groups)

    scheduler = policy.config.get_scheduler_preset().build(optimizer, TOTAL_STEPS)
    factors = []
    for _ in range(TOTAL_STEPS + 2):
        factors.append(scheduler.get_last_lr()[0] / STEP_LR)
        optimizer.step()
        scheduler.step()
    # Warmup (step + 1) / warmup, then cosine over the remaining steps down to a 5% floor.
    expected = [0.5, 1.0]
    for step in range(WARMUP_STEPS, TOTAL_STEPS + 2):
        progress = min((step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS), 1.0)
        expected.append(0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2)
    assert factors == pytest.approx(expected, rel=1e-12)
    assert factors[-1] == pytest.approx(0.05)


def test_lerobot_train_updates_move_only_trainable_weights(policy, batch):
    from accelerate import Accelerator

    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    trained = copy.deepcopy(policy)
    initial = {name: param.detach().clone() for name, param in trained.model.named_parameters()}
    optimizer = trained.config.get_optimizer_preset().build(trained.get_optim_params())
    scheduler = trained.config.get_scheduler_preset().build(optimizer, TOTAL_STEPS)
    metrics = {name: AverageMeter(name) for name in ("loss", "grad_norm", "lr", "update_s")}
    tracker = MetricsTracker(BATCH_SIZE, 1000, 1, metrics)

    for seed in (3, 4):
        torch.manual_seed(seed)
        tracker, _ = update_policy(
            tracker, trained, batch, optimizer, 1.0, Accelerator(cpu=True), lr_scheduler=scheduler
        )
        assert math.isfinite(tracker.loss.val)

    max_change = 0.0
    for name, param in trained.model.named_parameters():
        change = float((param.detach() - initial[name]).abs().max())
        if param.requires_grad:
            max_change = max(max_change, change)
        else:
            assert change == 0.0, name
    assert max_change > 1e-5
