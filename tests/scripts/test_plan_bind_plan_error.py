import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lerobot_policy_bspline import BsplineConfig
from lerobot_policy_bspline.modeling_bspline import BsplinePolicy

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "261028-plan-bind-d2-latency" / "plan_error.py"
IMAGE_KEY = "observation.images.top"


def _load_script():
    spec = importlib.util.spec_from_file_location("plan_error", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan_error = _load_script()


def _policy() -> BsplinePolicy:
    config = BsplineConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        pretrained_backbone_weights=None,
        use_group_norm=True,
        down_dims=(32, 64),
        diffusion_step_embed_dim=16,
        spatial_softmax_num_keypoints=8,
        bspline_window=32,
        bspline_mode="planbind",
        planbind_max_plan_age=8,
        n_action_steps=8,
        device="cpu",
    )
    return BsplinePolicy(config).eval()


def _batches(policy: BsplinePolicy) -> dict[int, dict]:
    torch.manual_seed(0)
    n_actions = len(policy.config.action_delta_indices)
    return {
        age: {
            OBS_STATE: torch.randn(1, policy.config.n_obs_steps, 6),
            IMAGE_KEY: torch.rand(1, policy.config.n_obs_steps, 3, 32, 32),
            ACTION: torch.randn(1, n_actions, 6),
        }
        for age in (0, 4, 8)
    }


def test_trigger_matches_the_runtime_residual():
    policy = _policy()
    batches = _batches(policy)
    record = plan_error.measure_frame(policy, batches.__getitem__, [0, 4, 8])
    assert set(record) == {0, 4, 8}

    model, runtime = policy.diffusion, policy.runtime
    cond_now = policy._global_cond(batches[0])
    for age in (0, 4, 8):
        with torch.no_grad():
            runtime.plan = model.plan(policy._global_cond(batches[age]))
        runtime.plan_anchor = torch.tensor([100 - age])
        _, residual = runtime._bind(cond_now, torch.tensor([100]))
        assert record[age]["trigger"] == pytest.approx(residual.item(), rel=1e-5)
        assert record[age]["plan_error_rms"] <= record[age]["plan_error_max"]


def test_aggregate_and_table():
    records = [
        {0: {"plan_error_max": v, "plan_error_rms": v / 2, "trigger": v}} for v in (0.1, 0.4, 0.6, 1.2)
    ]
    summary = plan_error.aggregate(records, [0])
    assert summary["0"]["n_frames"] == 4
    assert summary["0"]["trigger"]["max"] == 1.2
    assert summary["0"]["trigger_above"] == {"0.3": 0.75, "0.5": 0.5, "1.0": 0.25}
    assert "| 0 | 4 |" in plan_error.table(summary, 0.02)


def test_sample_frames_keep_history_and_future_inside_the_episode():
    episodes = [
        {"dataset_from_index": 0, "dataset_to_index": 100},
        {"dataset_from_index": 100, "dataset_to_index": 130},
    ]
    dataset = SimpleNamespace(meta=SimpleNamespace(total_episodes=2, episodes=episodes))
    frames = plan_error.sample_frames(dataset, episodes=2, per_episode=5, before=20, after=32, seed=0)
    assert len(frames) == 5 and all(20 <= f < 68 for f in frames)
