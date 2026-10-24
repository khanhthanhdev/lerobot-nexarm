import numpy as np
import pytest
import torch
from lerobot_policy_bspline.bspline import (
    decode_param_batch,
    decode_params,
    encode_action_batch,
    encode_actions,
    n_param_rows,
    normalize_knots,
    safe_knots,
    spline_time_range,
    unnormalize_knots,
)

DEGREE = 3
CHUNK = 10


def smooth_actions(n_frames: int, action_dim: int = 6, seed: int = 0) -> np.ndarray:
    """Smooth joint-like trajectory in normalized units: a few sinusoids per dimension."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_frames)[:, None]
    freq = rng.uniform(0.3, 1.5, size=(1, action_dim))
    phase = rng.uniform(0, 2 * np.pi, size=(1, action_dim))
    return 0.8 * np.sin(2 * np.pi * freq * t + phase)


@pytest.mark.parametrize("max_error", [0.01, 0.002])
def test_encode_decode_reconstructs_window(max_error):
    actions = smooth_actions(64)
    params = encode_actions(actions, CHUNK, DEGREE, max_error)
    assert params.shape == (n_param_rows(CHUNK, DEGREE), 1 + actions.shape[1])

    t_min, t_max = spline_time_range(params, DEGREE)
    assert t_min == 0.0
    frames = np.arange(int(np.floor(t_max)) + 1)
    decoded = decode_params(params, frames, DEGREE)
    np.testing.assert_allclose(decoded, actions[frames], atol=max_error)


def test_simple_window_is_covered_by_one_chunk():
    # A smooth 64-frame window needs fewer knots than one chunk holds, so one chunk spans all of it.
    actions = smooth_actions(64, seed=1)
    params = encode_actions(actions, CHUNK, DEGREE, max_error=0.01)
    assert spline_time_range(params, DEGREE) == (0.0, 63.0)


def test_complex_window_covers_a_prefix_within_tolerance():
    rng = np.random.default_rng(2)
    actions = np.cumsum(rng.normal(scale=0.05, size=(64, 3)), axis=0)
    params = encode_actions(actions, CHUNK, DEGREE, max_error=0.01)
    _, t_max = spline_time_range(params, DEGREE)
    assert 0.0 < t_max < 63.0
    frames = np.arange(int(np.floor(t_max)) + 1)
    np.testing.assert_allclose(decode_params(params, frames, DEGREE), actions[frames], atol=0.01)


def test_window_shorter_than_spline_order_holds_last_action():
    actions = np.array([[0.1, -0.2], [0.3, 0.0]])
    params = encode_actions(actions, CHUNK, DEGREE, max_error=0.01)
    decoded = decode_params(params, np.arange(4), DEGREE)
    np.testing.assert_allclose(decoded[:2], actions, atol=0.01)
    np.testing.assert_allclose(decoded[2:], np.repeat(actions[-1:], 2, axis=0), atol=0.01)


def test_safe_knots_is_non_decreasing():
    knots = safe_knots(np.array([0.0, 0.0, 2.0, 1.0, 3.0]))
    assert np.all(np.diff(knots) >= 0)
    assert knots[3] == pytest.approx(2.0, abs=1e-5)


def test_knot_normalization_round_trip():
    knots = np.array([0.0, 10.0, 63.0])
    normalized = normalize_knots(knots, 63.0)
    np.testing.assert_allclose(normalized, [-1.0, 10 / 31.5 - 1.0, 1.0])
    np.testing.assert_allclose(unnormalize_knots(normalized, 63.0), knots)


def test_batch_round_trip_with_episode_padding():
    window = 64
    actions = np.stack([smooth_actions(window, seed=s) for s in range(3)])
    # Sample 2 sits 20 frames before the episode end: the dataset copy-pads the rest.
    actions[2, 20:] = actions[2, 19]
    is_pad = np.zeros((3, window), dtype=bool)
    is_pad[2, 20:] = True

    params = encode_action_batch(
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(is_pad),
        chunk_size=CHUNK,
        degree=DEGREE,
        max_error=0.01,
        time_scale=window - 1,
    )
    assert params.shape == (3, n_param_rows(CHUNK, DEGREE), 1 + actions.shape[-1])
    assert params[..., 0].min() >= -1.0 and params[..., 0].max() <= 1.0

    decoded = decode_param_batch(params, DEGREE, time_scale=window - 1, n_actions=32)
    # The padded sample's spline ends at frame 19, so the batch chunk is cut to 20 actions.
    assert decoded.shape == (3, 20, actions.shape[-1])
    np.testing.assert_allclose(decoded.numpy(), actions[:, :20], atol=0.011)


def test_decode_speed_up_samples_spline_faster():
    window = 64
    actions = smooth_actions(window, seed=3)
    params = encode_action_batch(
        torch.tensor(actions[None], dtype=torch.float32), None, CHUNK, DEGREE, 0.01, window - 1
    )
    decoded = decode_param_batch(params, DEGREE, time_scale=window - 1, n_actions=16, speed_up=2.0)
    assert decoded.shape == (1, 16, actions.shape[-1])
    np.testing.assert_allclose(decoded[0].numpy(), actions[0:32:2], atol=0.011)
