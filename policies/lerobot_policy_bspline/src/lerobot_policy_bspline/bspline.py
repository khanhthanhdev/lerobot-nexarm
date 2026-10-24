# MIT License
#
# Copyright (c) 2026 Haoyu Xiong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted from bspline_policy/common/bspline_action.py and
# bspline_policy/scripts/policy_local_bspline.py of
# https://github.com/khanhthanhdev/bspline-policy at commit 61ed5f42.
"""B-spline action chunks: fit, pack into a parameter matrix, and evaluate.

A chunk is a dense parameter matrix of shape ``(chunk_size + 2 * degree, 1 + action_dim)``.
Column 0 holds the knot vector (time in dataset frames, relative to the current frame) and the
other columns hold the control points. The last ``degree + 1`` control-point rows are padding
that the decoder ignores, exactly as in the upstream representation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline

type TensorOrArray = torch.Tensor | np.ndarray

# Knot spacing below this (in frames) counts as a repeated knot.
_KNOT_EPS = 1e-6


def n_param_rows(chunk_size: int, degree: int) -> int:
    """Number of rows of one B-spline parameter matrix."""
    return chunk_size + 2 * degree


def fit_bspline(actions: np.ndarray, degree: int = 3, max_error: float = 0.01, s: float = 1e-12) -> BSpline:
    """Least-squares B-spline fit with the fewest knots whose max abs error is below `max_error`.

    `actions` has shape ``(T, action_dim)`` sampled at ``t = 0..T-1``. Knots are added greedily by
    `scipy.interpolate.generate_knots`; if no knot set reaches `max_error`, the finest one is used
    (upstream behaviour).
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions of shape (T, action_dim), got {actions.shape}")
    if len(actions) < degree + 1:
        # A degree-k spline needs at least k + 1 samples: hold the last action.
        pad = np.repeat(actions[-1:], degree + 1 - len(actions), axis=0)
        actions = np.concatenate([actions, pad], axis=0)

    t: np.ndarray = np.arange(len(actions), dtype=np.float64)
    spline = None
    for knots in generate_knots(t, actions, k=degree, s=s):
        spline = make_lsq_spline(t, actions, knots, k=degree)
        if np.abs(spline(t) - actions).max() < max_error:
            break
    assert spline is not None
    return spline


def spline_to_params(spline: BSpline, chunk_size: int) -> np.ndarray:
    """Pack the first chunk of a fitted spline into a ``(chunk_size + 2 * degree, 1 + D)`` matrix.

    Splines with fewer knots than the chunk holds are padded by repeating the last knot and the
    last control point, which keeps the decoded range at the end of the fitted data.
    """
    degree = int(spline.k)
    n_rows = n_param_rows(chunk_size, degree)
    knots = np.asarray(spline.t[:n_rows], dtype=np.float64)
    coeffs = np.asarray(spline.c[:n_rows], dtype=np.float64)
    if len(knots) < n_rows:
        knots = np.concatenate([knots, np.full(n_rows - len(knots), knots[-1])])
    if len(coeffs) < n_rows:
        coeffs = np.concatenate([coeffs, np.repeat(coeffs[-1:], n_rows - len(coeffs), axis=0)])
    return np.concatenate([knots[:, None], coeffs], axis=1)


def encode_actions(
    actions: np.ndarray, chunk_size: int, degree: int = 3, max_error: float = 0.01
) -> np.ndarray:
    """Encode an action window ``(T, D)`` into one B-spline parameter matrix."""
    return spline_to_params(fit_bspline(actions, degree=degree, max_error=max_error), chunk_size)


def safe_knots(knots: np.ndarray) -> np.ndarray:
    """Make a predicted knot vector non-decreasing (upstream `safer_knots`)."""
    knots = np.asarray(knots, dtype=np.float64).copy()
    for idx in range(1, len(knots)):
        if knots[idx] < knots[idx - 1]:
            knots[idx] = knots[idx - 1] + _KNOT_EPS
    return knots


def params_to_spline(params: np.ndarray, degree: int = 3) -> BSpline:
    """Build the B-spline described by one parameter matrix.

    Padding repeats the last knot more than ``degree + 1`` times; those extra knots add only empty
    intervals (and make the right end evaluate to zero), so they are dropped with their control rows.
    """
    params = np.asarray(params, dtype=np.float64)
    knots = safe_knots(params[:, 0])
    while len(knots) > 2 * (degree + 1) and knots[-1] - knots[-(degree + 2)] <= _KNOT_EPS:
        knots = knots[:-1]
    return BSpline(knots, params[: len(knots) - degree - 1, 1:], degree, extrapolate=False)


def spline_time_range(params: np.ndarray, degree: int = 3) -> tuple[float, float]:
    """Valid evaluation interval ``[t_min, t_max]`` of one parameter matrix."""
    knots = safe_knots(np.asarray(params, dtype=np.float64)[:, 0])
    return float(knots[degree]), float(knots[-(degree + 1)])


def decode_params(params: np.ndarray, times: np.ndarray, degree: int = 3) -> np.ndarray:
    """Evaluate one parameter matrix at `times` (clipped to the valid interval)."""
    t_min, t_max = spline_time_range(params, degree)
    times = np.clip(np.asarray(times, dtype=np.float64), t_min, t_max)
    return params_to_spline(params, degree)(times)


def normalize_knots(knots: TensorOrArray, time_scale: float):
    """Map knot times in ``[0, time_scale]`` frames to ``[-1, 1]``."""
    return knots * (2.0 / time_scale) - 1.0


def unnormalize_knots(knots: TensorOrArray, time_scale: float):
    """Inverse of `normalize_knots`."""
    return (knots + 1.0) * (time_scale / 2.0)


def encode_action_batch(
    actions: torch.Tensor,
    action_is_pad: torch.Tensor | None,
    chunk_size: int,
    degree: int,
    max_error: float,
    time_scale: float,
) -> torch.Tensor:
    """Encode ``(B, T, D)`` action windows into ``(B, chunk_size + 2 * degree, 1 + D)`` parameters.

    Copy-padded frames past the episode end are dropped before fitting, so the spline ends where the
    episode ends. The knot column is normalized with `normalize_knots`.
    """
    actions_np = actions.detach().to("cpu", torch.float64).numpy()
    pad_np = None if action_is_pad is None else action_is_pad.detach().cpu().numpy()
    out = []
    for i, window in enumerate(actions_np):
        n_valid = len(window) if pad_np is None else int((~pad_np[i]).sum())
        params = encode_actions(window[: max(n_valid, 1)], chunk_size, degree, max_error)
        params[:, 0] = normalize_knots(params[:, 0], time_scale)
        out.append(params)
    return torch.as_tensor(np.stack(out), dtype=actions.dtype, device=actions.device)


def decode_param_batch(
    params: torch.Tensor,
    degree: int,
    time_scale: float,
    n_actions: int,
    speed_up: float = 1.0,
) -> torch.Tensor:
    """Decode ``(B, rows, 1 + D)`` parameters into ``(B, n, D)`` actions, one per control step.

    Action ``i`` is the spline at ``t = i * speed_up`` frames from the current frame. The chunk is cut
    at the shortest valid spline in the batch, so it never extrapolates (``1 <= n <= n_actions``).
    """
    params_np = params.detach().to("cpu", torch.float64).numpy().copy()
    params_np[..., 0] = unnormalize_knots(np.clip(params_np[..., 0], -1.0, 1.0), time_scale)
    times = np.arange(n_actions, dtype=np.float64) * speed_up
    n = n_actions
    for p in params_np:
        _, t_max = spline_time_range(p, degree)
        # The tolerance absorbs float32 round-off of the normalized knot column.
        n = min(n, max(1, int(np.searchsorted(times, t_max + 1e-4, side="right"))))
    decoded = np.stack([decode_params(p, times[:n], degree) for p in params_np])
    return torch.as_tensor(decoded, dtype=params.dtype, device=params.device)
