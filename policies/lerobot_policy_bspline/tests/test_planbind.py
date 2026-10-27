import pytest
import torch
from lerobot_policy_bspline import BsplineConfig
from lerobot_policy_bspline.modeling_bspline import BsplineModel, BsplinePolicy
from lerobot_policy_bspline.planbind import PlanBindModel
from lerobot_policy_bspline.uniform_spline import (
    basis_matrix,
    lift_matrix,
    n_ctrl_points,
    n_frozen,
    project,
    shift_ctrl,
)

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_STATE

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
        "bspline_window": WINDOW,
        "bspline_mode": "planbind",
        "planbind_max_plan_age": 8,
        "n_action_steps": 8,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return BsplineConfig(**kwargs)


def make_batch(config: BsplineConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    n_obs = config.n_obs_steps
    n_actions = len(config.action_delta_indices)
    t = torch.linspace(0, 1, n_actions)[None, :, None]
    actions = 0.8 * torch.sin(2 * torch.pi * t * torch.arange(1, ACTION_DIM + 1) / 4).expand(BATCH, -1, -1)
    is_pad = torch.zeros(BATCH, n_actions, dtype=torch.bool)
    is_pad[1, :3] = True  # an episode start inside the stale history
    return {
        OBS_STATE: torch.randn(BATCH, n_obs, STATE_DIM),
        IMAGE_KEY: torch.rand(BATCH, n_obs, 3, 32, 32),
        ACTION: actions.clone(),
        "action_is_pad": is_pad,
    }


def curve(ctrl: torch.Tensor, times: torch.Tensor, spacing: float) -> torch.Tensor:
    return torch.einsum("tn,bnd->btd", basis_matrix(times, ctrl.shape[1], spacing), ctrl.double())


def test_lift_is_exact():
    n_coarse, subdivision = 6, 4
    n_fine = 3 + (n_coarse - 3) * subdivision
    coarse = torch.randn(3, n_coarse, 2, dtype=torch.float64)
    fine = torch.einsum("fc,bcd->bfd", lift_matrix(n_coarse, n_fine, subdivision), coarse)
    times = torch.linspace(0, (n_coarse - 3) * subdivision, 101, dtype=torch.float64)
    torch.testing.assert_close(curve(fine, times, 1.0), curve(coarse, times, subdivision), atol=1e-10, rtol=0)


def test_shift_reanchors_the_curve():
    ctrl = torch.randn(2, 12, 3, dtype=torch.float64)
    spacing, spans = 4.0, torch.tensor([1, 3])
    shifted = shift_ctrl(ctrl, spans)
    times = torch.linspace(0, 20, 41, dtype=torch.float64)
    for b in range(2):
        torch.testing.assert_close(
            curve(shifted[b : b + 1], times, spacing),
            curve(ctrl[b : b + 1], times + float(spans[b]) * spacing, spacing),
        )


@pytest.mark.parametrize("latency", [0, 3, 4, 9])
def test_latency_commit_keeps_the_curve_until_the_swap(latency):
    spacing = 4
    running, fresh = torch.randn(1, 12, 2), torch.randn(1, 12, 2)
    frozen = n_frozen(latency, spacing, "latency")
    joined = torch.cat([running[:, :frozen], fresh[:, frozen:]], dim=1)
    times = torch.linspace(0, latency, 4 * latency + 1, dtype=torch.float64)
    torch.testing.assert_close(curve(joined, times, spacing), curve(running, times, spacing))
    # Value and first two derivatives match at the swap: compare one-sided finite differences.
    h = 1e-3
    near = torch.tensor([latency - 2 * h, latency - h, float(latency)], dtype=torch.float64)
    torch.testing.assert_close(curve(joined, near, spacing), curve(running, near, spacing))
    assert n_frozen(latency, spacing, "none") == 0 and n_frozen(latency, spacing, "now") == 3
    with pytest.raises(ValueError):
        n_frozen(latency, spacing, "sometimes")


def test_projection_bounds_position_and_velocity():
    torch.manual_seed(1)
    spacing, pos, vel = 4.0, 0.8, 0.05
    ctrl = 2 * torch.randn(4, 16, 3)
    ctrl[:, :3] = 0.0  # feasible frozen prefix
    out = project(ctrl, 3, spacing, pos, vel)
    torch.testing.assert_close(out[:, :3], ctrl[:, :3])
    times = torch.linspace(0, (16 - 3) * spacing, 2001, dtype=torch.float64)
    values = curve(out, times, spacing)
    assert values.abs().max() <= pos + 1e-6
    speed = (values[:, 1:] - values[:, :-1]).abs() / (times[1] - times[0])
    assert speed.max() <= vel + 1e-6
    assert project(ctrl, 3, spacing, 0.0, 0.0) is ctrl


def test_default_mode_is_unchanged():
    config = make_config(bspline_mode="diffusion")
    assert config.action_delta_indices == list(range(WINDOW))
    policy = BsplinePolicy(config)
    assert isinstance(policy.diffusion, BsplineModel) and policy.runtime is None
    assert make_config(bspline_mode="onestep").action_delta_indices == list(range(WINDOW))
    assert make_config().action_delta_indices == list(range(-8, WINDOW))


def test_config_validation():
    with pytest.raises(ValueError, match="bspline_mode"):
        make_config(bspline_mode="twostep")
    with pytest.raises(ValueError, match="planbind_max_plan_age"):
        make_config(planbind_max_plan_age=6)
    with pytest.raises(ValueError, match="planbind_commit"):
        make_config(planbind_commit="later")
    with pytest.raises(ValueError, match="planbind_latency"):
        make_config(planbind_latency=WINDOW)
    with pytest.raises(ValueError, match="cubic"):
        make_config(bspline_degree=2, bspline_chunk_size=12)


@pytest.mark.parametrize("mode", ["onestep", "planbind"])
def test_forward_backward_and_chunk(mode):
    config = make_config(bspline_mode=mode)
    policy = BsplinePolicy(config)
    assert isinstance(policy.diffusion, PlanBindModel)
    batch = make_batch(config)
    loss, output = policy.forward(batch)
    assert output is None and loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert policy.diffusion.unet.final_conv[-1].weight.grad is not None
    assert not any(k.endswith(("fine_basis", "lift", "coarse_fit")) for k in policy.state_dict())

    policy.eval()
    chunk = policy.predict_action_chunk({k: v for k, v in batch.items() if k != ACTION})
    assert chunk.shape == (BATCH, config.n_action_steps, ACTION_DIM)


def _run(policy: BsplinePolicy, steps: int) -> list[dict]:
    """Step the policy; record, at each swap, the running and the new command's value at that frame."""
    torch.manual_seed(2)
    policy.reset()
    obs_state = torch.randn(BATCH, STATE_DIM)
    image = torch.rand(BATCH, 3, 32, 32)
    runtime = policy.runtime
    basis = policy.diffusion.fine_basis
    swaps = []
    for _ in range(steps):
        pending = runtime.pending
        if pending is not None and runtime.frame >= pending[0]:
            frame = runtime.frame
            before = torch.einsum("n,bnd->bd", basis[frame - runtime.cmd_anchor], runtime.cmd)
            action = policy.select_action({OBS_STATE: obs_state, IMAGE_KEY: image})
            swaps.append({"frame": frame, "before": before, "after": action})
        else:
            action = policy.select_action({OBS_STATE: obs_state, IMAGE_KEY: image})
        assert action.shape == (BATCH, ACTION_DIM) and torch.isfinite(action).all()
        # Vary the observation so consecutive calls disagree.
        obs_state = obs_state + 0.5 * torch.randn_like(obs_state)
    return swaps


@pytest.mark.parametrize("mode", ["onestep", "planbind"])
def test_select_action_with_latency_swaps_without_jumps(mode):
    config = make_config(bspline_mode=mode, planbind_latency=6, planbind_replan_threshold=0.01)
    swaps = _run(BsplinePolicy(config).eval(), 60)
    assert len(swaps) >= 4
    for swap in swaps:
        torch.testing.assert_close(swap["after"], swap["before"], atol=1e-5, rtol=0)

    naive = _run(
        BsplinePolicy(make_config(bspline_mode=mode, planbind_latency=6, planbind_commit="none")).eval(), 60
    )
    jumps = [float((s["after"] - s["before"]).abs().max()) for s in naive]
    assert max(jumps) > 1e-3


def test_plan_expires_at_the_trained_age():
    config = make_config(planbind_max_plan_age=8, planbind_fine_spacing=4)
    policy = BsplinePolicy(config).eval()
    _run(policy, 40)
    runtime = policy.runtime
    assert (
        runtime.frame - 1 - runtime.plan_anchor <= config.planbind_max_plan_age + config.planbind_fine_spacing
    ).all()


def test_save_and_load_round_trip(tmp_path):
    config = make_config(planbind_latency=3, planbind_commit="now")
    policy = BsplinePolicy(config).eval()
    policy.save_pretrained(tmp_path)
    loaded_config = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded_config, BsplineConfig)
    assert loaded_config.bspline_mode == "planbind" and loaded_config.planbind_commit == "now"
    loaded = BsplinePolicy.from_pretrained(tmp_path).eval()
    batch = make_batch(config)
    obs = {OBS_STATE: batch[OBS_STATE], IMAGE_KEY: batch[IMAGE_KEY]}
    torch.testing.assert_close(policy.predict_action_chunk(dict(obs)), loaded.predict_action_chunk(dict(obs)))
    assert n_ctrl_points(WINDOW, 4) <= loaded.diffusion.n_fine
