import importlib.metadata

import numpy as np
import pytest
import torch
from lerobot_policy_bspline import BsplineConfig
from lerobot_policy_bspline.modeling_bspline import BsplinePolicy

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins

IMAGE_KEY = "observation.images.top"
STATE_DIM = 6
ACTION_DIM = 6
WINDOW = 32
BATCH = 2


def make_config(**overrides) -> BsplineConfig:
    kwargs = {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        "pretrained_backbone_weights": None,
        "use_group_norm": True,
        "down_dims": (32, 64),
        "diffusion_step_embed_dim": 16,
        "spatial_softmax_num_keypoints": 8,
        "num_inference_steps": 4,
        "bspline_window": WINDOW,
        "n_action_steps": 8,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return BsplineConfig(**kwargs)


def make_batch(config: BsplineConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    n_obs = config.n_obs_steps
    t = torch.linspace(0, 1, WINDOW)[None, :, None]
    actions = 0.8 * torch.sin(2 * np.pi * t * torch.arange(1, ACTION_DIM + 1) / 4).expand(BATCH, -1, -1)
    return {
        OBS_STATE: torch.randn(BATCH, n_obs, STATE_DIM),
        IMAGE_KEY: torch.rand(BATCH, n_obs, 3, 32, 32),
        ACTION: actions.clone(),
        "action_is_pad": torch.zeros(BATCH, WINDOW, dtype=torch.bool),
    }


def test_plugin_is_discovered():
    assert importlib.metadata.metadata("lerobot_policy_bspline")["Name"] == "lerobot_policy_bspline"
    register_third_party_plugins()
    assert "bspline" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("bspline") is BsplineConfig
    assert get_policy_class("bspline") is BsplinePolicy


def test_config_validation():
    config = make_config()
    assert config.type == "bspline"
    assert config.horizon == config.bspline_chunk_size + 2 * config.bspline_degree
    assert config.action_delta_indices == list(range(WINDOW))
    with pytest.raises(ValueError, match="horizon"):
        make_config(horizon=32)
    with pytest.raises(ValueError, match="do_mask_loss_for_padding"):
        make_config(do_mask_loss_for_padding=True)
    with pytest.raises(ValueError, match="bspline_fit_workers"):
        make_config(bspline_fit_workers=-1)


def test_forward_and_param_shapes():
    config = make_config()
    policy = BsplinePolicy(config)
    batch = make_batch(config)

    params = policy.encode_actions(batch[ACTION], batch["action_is_pad"])
    assert params.shape == (BATCH, config.horizon, ACTION_DIM + 1)

    loss, output = policy.forward(batch)
    assert output is None
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert policy.diffusion.unet.final_conv[-1].weight.grad is not None

    policy.eval()
    pred = policy.predict_action_params({k: v for k, v in batch.items() if k != ACTION})
    assert pred.shape == (BATCH, config.horizon, ACTION_DIM + 1)


def test_select_action_on_cpu_batch():
    config = make_config()
    policy = BsplinePolicy(config).eval()
    policy.reset()
    batch = make_batch(config)
    obs = {OBS_STATE: batch[OBS_STATE][:, -1], IMAGE_KEY: batch[IMAGE_KEY][:, -1]}

    action = policy.select_action(dict(obs))
    assert action.shape == (BATCH, ACTION_DIM)
    assert torch.isfinite(action).all()
    queued = len(policy._queues[ACTION])
    assert 0 <= queued < config.n_action_steps
    for _ in range(queued):
        assert policy.select_action(dict(obs)).shape == (BATCH, ACTION_DIM)


def test_save_and_load_round_trip(tmp_path):
    # DDIM is deterministic for a fixed starting noise, so both policies must agree exactly.
    config = make_config(noise_scheduler_type="DDIM")
    policy = BsplinePolicy(config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=None)
    policy.save_pretrained(tmp_path)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded_config = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded_config, BsplineConfig)
    loaded = BsplinePolicy.from_pretrained(tmp_path).eval()
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)

    batch = make_batch(config)
    obs = {OBS_STATE: batch[OBS_STATE], IMAGE_KEY: batch[IMAGE_KEY]}
    noise = torch.randn(BATCH, config.horizon, ACTION_DIM + 1)
    torch.testing.assert_close(
        policy.predict_action_params(dict(obs), noise=noise),
        loaded.predict_action_params(dict(obs), noise=noise),
    )
