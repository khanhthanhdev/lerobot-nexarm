import importlib.metadata

import pytest

pytest.importorskip("dem", reason="install the `dem` extra")
pytest.importorskip("lerobot_policy_dem", reason="install the `dem` extra")

import torch  # noqa: E402
from dem.head import MeanFlowTokenHead  # noqa: E402
from lerobot_policy_dem import DEMConfig  # noqa: E402
from lerobot_policy_dem.head_dem import GlobalCondMeanFlowHead  # noqa: E402
from lerobot_policy_dem.modeling_dem import _LOAD_BACKBONE_WEIGHTS, DEMPolicy  # noqa: E402

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

CAMERAS = ["observation.images.front", "observation.images.wrist"]
TASK = "Stack the bowls with red on bottom, blue in middle, and black on top."


def make_config(**overrides) -> DEMConfig:
    kwargs = {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        "camera_keys": CAMERAS,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return DEMConfig(**kwargs)


def make_batch(batch_size: int = 2, chunk: int = 4) -> dict:
    batch = {key: torch.rand(batch_size, 3, 48, 64) for key in CAMERAS}
    batch[OBS_STATE] = torch.randn(batch_size, 6)
    batch[ACTION] = torch.randn(batch_size, chunk, 6)
    batch["task"] = [TASK] * batch_size
    return batch


@pytest.fixture(scope="module")
def small_policy() -> DEMPolicy:
    """A DEM policy with randomly initialised towers, a small head and 64 px frames (no weight download)."""
    token = _LOAD_BACKBONE_WEIGHTS.set(False)
    try:
        return DEMPolicy(make_config(head_size="small", image_size=64, chunk_size=4, n_action_steps=3))
    except OSError as error:
        pytest.skip(f"NeoBERT tokenizer is neither cached nor downloadable: {error}")
    finally:
        _LOAD_BACKBONE_WEIGHTS.reset(token)


def test_plugin_is_discovered():
    assert importlib.metadata.metadata("lerobot_policy_dem")["Name"] == "lerobot_policy_dem"
    register_third_party_plugins()
    assert "dem" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("dem") is DEMConfig
    assert get_policy_class("dem") is DEMPolicy


def test_config_defaults():
    config = make_config()
    config.validate_features()
    assert config.ordered_camera_keys == CAMERAS
    assert make_config(camera_keys=[]).ordered_camera_keys == [*CAMERAS, "observation.images.top"]
    assert config.action_delta_indices == list(range(16))
    assert (config.image_size, config.head_size, config.max_language_tokens) == (256, "new100m", 32)
    assert (config.ratio_r_eq_t, config.dispersive_weight, config.num_inference_steps) == (0.5, 0.25, 1)
    assert config.global_conditioning and config.num_language_queries == 4
    optimizer = config.get_optimizer_preset()
    assert (optimizer.lr, optimizer.betas, optimizer.weight_decay, optimizer.grad_clip_norm) == (
        5e-5,
        (0.9, 0.95),
        1e-4,
        1.0,
    )
    scheduler = config.get_scheduler_preset()
    assert (scheduler.num_warmup_steps, scheduler.peak_lr, scheduler.decay_lr) == (2000, 5e-5, 0.0)
    with pytest.raises(ValueError, match="n_action_steps"):
        make_config(n_action_steps=17)
    with pytest.raises(ValueError, match="head_size"):
        make_config(head_size="huge")
    with pytest.raises(ValueError, match="not image features"):
        make_config(camera_keys=["observation.images.side"]).validate_features()


def test_processors_z_score_state_and_action():
    stats = {
        OBS_STATE: {"mean": torch.full((6,), 2048.0), "std": torch.full((6,), 100.0)},
        ACTION: {"mean": torch.full((6,), 2048.0), "std": torch.full((6,), 100.0)},
    }
    preprocessor, postprocessor = make_pre_post_processors(make_config(), dataset_stats=stats)
    observation = {OBS_STATE: torch.full((6,), 2148.0), "observation.images.front": torch.rand(3, 48, 64)}
    processed = preprocessor(observation)
    assert torch.allclose(processed[OBS_STATE], torch.ones(1, 6))
    assert processed["observation.images.front"].shape == (1, 3, 48, 64)  # resized inside the policy
    assert torch.allclose(postprocessor(torch.zeros(1, 6)), torch.full((1, 6), 2048.0))


def test_loss_backward_reaches_head_and_vision(small_policy):
    policy = small_policy
    policy.train()
    policy.zero_grad()
    # The output layer and the cross-attention gates start at zero, so nothing would reach the vision
    # encoder on the first step; open them as a few training steps would.
    with torch.no_grad():
        torch.nn.init.normal_(policy.model.head.out.weight, std=0.02)
        for block in policy.model.head.blocks:
            block.gate.fill_(0.1)
            torch.nn.init.normal_(block.ada_block.ada[-1].weight, std=0.02)
        torch.nn.init.normal_(policy.model.head.global_mlp[-1].weight, std=0.02)
    loss, info = policy.forward(make_batch())
    assert torch.isfinite(loss) and set(info) == {"meanflow_mse", "dispersive"}
    loss.backward()
    groups = policy.get_optim_params()
    assert len(groups) == 2 and [g["lr"] for g in groups] == [5e-5, 2.5e-5]
    for group in groups:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in group["params"])
    head = policy.model.head
    for module in (head.global_lang_proj, head.global_lang_pool, head.global_prop_proj):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in policy.model.language.parameters())  # frozen NeoBERT
    with pytest.raises(ValueError, match="action_delta_indices"):
        policy.forward({**make_batch(), ACTION: torch.randn(2, 6)})


def test_select_action_queues_chunks(small_policy):
    policy = small_policy
    policy.reset()
    observation = {k: v[:1] for k, v in make_batch().items() if k != ACTION}
    observation["task"] = [TASK]
    chunk = policy.predict_action_chunk(observation)
    assert chunk.shape == (1, 4, 6) and torch.isfinite(chunk).all()
    actions = [policy.select_action(observation) for _ in range(4)]
    assert all(a.shape == (1, 6) for a in actions)
    assert len(policy._action_queue) == 2  # n_action_steps=3: the fourth call predicted a new chunk


def test_save_and_load_round_trip(small_policy, tmp_path):
    small_policy.save_pretrained(tmp_path)
    loaded = DEMPolicy.from_pretrained(tmp_path)
    assert _LOAD_BACKBONE_WEIGHTS.get() is True
    saved = small_policy.state_dict()
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, saved[name]), name
    observation = {k: v[:1] for k, v in make_batch().items() if k != ACTION}
    torch.manual_seed(0)
    expected = small_policy.predict_action_chunk(observation)
    torch.manual_seed(0)
    assert torch.allclose(loaded.predict_action_chunk(observation), expected, atol=1e-5)


def make_heads() -> tuple[MeanFlowTokenHead, GlobalCondMeanFlowHead]:
    torch.manual_seed(0)
    kwargs = {"dim": 64, "depth": 2, "heads": 4}
    base = MeanFlowTokenHead(32, 48, 6, 6, 4, **kwargs)
    with torch.no_grad():  # open the zero-initialised output, AdaLN and gate layers, as training would
        torch.nn.init.normal_(base.out.weight, std=0.1)
        for block in base.blocks:
            block.gate.fill_(0.5)
            torch.nn.init.normal_(block.ada_block.ada[-1].weight, std=0.1)
    global_head = GlobalCondMeanFlowHead(32, 48, 6, 6, 4, **kwargs)
    missing, unexpected = global_head.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(name.startswith("global_") for name in missing)
    return base, global_head


def make_cond(batch_size: int = 3) -> dict:
    mask = torch.ones(batch_size, 10, dtype=torch.bool)
    mask[:, 7:] = False
    return {
        "vis_tokens": torch.randn(batch_size, 20, 32),
        "lang_tokens": torch.randn(batch_size, 10, 48),
        "lang_mask": mask,
        "proprio": torch.randn(batch_size, 6),
    }


def test_global_head_starts_as_the_paper_head():
    base, global_head = make_heads()
    z, r, t, cond = torch.randn(3, 4, 6), torch.rand(3) * 0.5, 0.5 + torch.rand(3) * 0.5, make_cond()
    assert torch.allclose(global_head(z, r, t, cond), base(z, r, t, cond), atol=1e-6)


def test_global_head_reads_instruction_and_state():
    _, head = make_heads()
    with torch.no_grad():
        torch.nn.init.normal_(head.global_mlp[-1].weight, std=0.1)
        for block in head.blocks:
            block.gate.zero_()  # cut the cross-attention path: only AdaLN can carry the condition
    z, r, t, cond = torch.randn(3, 4, 6), torch.rand(3) * 0.5, 0.5 + torch.rand(3) * 0.5, make_cond()
    out = head(z, r, t, cond)
    other_lang = {**cond, "lang_tokens": cond["lang_tokens"].flip(0)}
    other_state = {**cond, "proprio": cond["proprio"].flip(0)}
    assert (head(z, r, t, other_lang) - out).abs().mean() > 1e-3
    assert (head(z, r, t, other_state) - out).abs().mean() > 1e-3
    padded = {**cond, "lang_tokens": cond["lang_tokens"].clone()}
    padded["lang_tokens"][:, 7:] = 100.0  # masked positions must not matter
    assert torch.allclose(head(z, r, t, padded), out, atol=1e-5)
    # The MeanFlow target differentiates the head with forward-mode AD.
    _, tangent = torch.func.jvp(
        lambda z_, r_, t_: head(z_, r_, t_, cond),
        (z, r, t),
        (torch.ones_like(z), torch.zeros_like(r), torch.ones_like(t)),
    )
    assert torch.isfinite(tangent).all()


def test_paper_head_without_global_conditioning():
    token = _LOAD_BACKBONE_WEIGHTS.set(False)
    try:
        policy = DEMPolicy(
            make_config(
                head_size="small", image_size=64, chunk_size=4, n_action_steps=3, global_conditioning=False
            )
        )
    except OSError as error:
        pytest.skip(f"NeoBERT tokenizer is neither cached nor downloadable: {error}")
    finally:
        _LOAD_BACKBONE_WEIGHTS.reset(token)
    assert type(policy.model.head) is MeanFlowTokenHead
