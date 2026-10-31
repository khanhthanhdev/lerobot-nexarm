"""Uniform cubic B-splines on a fixed knot grid, for the `onestep` and `planbind` modes.

A chunk is ``n_ctrl`` control points on knots ``(i - 3) * spacing``; the curve is defined on
``[0, (n_ctrl - 3) * spacing]`` frames from the chunk's anchor frame. On the span ``[j, j + 1) * spacing``
it depends only on control points ``j .. j + 3``, so:

- re-anchoring a chunk ``s`` spans later is a shift of the control points by ``s``;
- freezing the first ``3 + ceil(latency / spacing)`` control points keeps the curve unchanged up to
  ``latency`` frames, and the swap is C2 (the value and two derivatives at a knot depend on three points);
- a coarse grid whose spacing is a multiple of the fine one is lifted into the fine grid exactly.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

DEGREE = 3


def n_ctrl_points(window: int, spacing: int) -> int:
    """Control points whose domain covers frames ``0 .. window - 1``."""
    return DEGREE + math.ceil((window - 1) / spacing)


def basis_matrix(times: Tensor, n_ctrl: int, spacing: float) -> Tensor:
    """``(len(times), n_ctrl)`` matrix whose row ``i`` evaluates the spline at ``times[i]`` frames.

    Times are clamped to the domain ``[0, (n_ctrl - 3) * spacing]``.
    """
    x = (times.to(torch.float64) / spacing).clamp(0.0, float(n_ctrl - DEGREE))
    span = x.floor().clamp(max=n_ctrl - DEGREE - 1)
    u = x - span
    weights = torch.stack(
        [
            (1 - u) ** 3 / 6,
            (3 * u**3 - 6 * u**2 + 4) / 6,
            (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6,
            u**3 / 6,
        ],
        dim=-1,
    )
    out = torch.zeros(len(times), n_ctrl, dtype=torch.float64)
    cols = span.long()[:, None] + torch.arange(DEGREE + 1)
    out.scatter_(1, cols, weights)
    return out


def lift_matrix(n_coarse: int, n_fine: int, subdivision: int) -> Tensor:
    """``(n_fine, n_coarse)`` matrix mapping coarse control points to fine ones (fine spacing = 1).

    Both splines cover ``[0, (n_coarse - 3) * subdivision]``; the fine grid contains every coarse knot, so
    the coarse curve lies in the fine spline space and the least-squares lift on dense samples is exact.
    """
    end = (n_coarse - DEGREE) * subdivision
    if end > n_fine - DEGREE:
        raise ValueError(f"The fine grid ({n_fine} points) does not cover the coarse domain [0, {end}].")
    times = torch.linspace(0.0, float(end), 8 * end + 1, dtype=torch.float64)
    fine = basis_matrix(times, n_fine, 1.0)[:, : DEGREE + end]
    coarse = basis_matrix(times, n_coarse, float(subdivision))
    lift = torch.linalg.lstsq(fine, coarse).solution
    out = torch.zeros(n_fine, n_coarse, dtype=torch.float64)
    out[: DEGREE + end] = lift
    # Fine points past the coarse domain hold the last lifted point (constant extension).
    out[DEGREE + end :] = lift[-1]
    return out


def shift_ctrl(ctrl: Tensor, spans: Tensor) -> Tensor:
    """Re-anchor ``(B, N, D)`` chunks ``spans[b]`` knot spans later; missing tail points repeat the last one."""
    n = ctrl.shape[1]
    idx = (torch.arange(n, device=ctrl.device)[None, :] + spans.to(ctrl.device).long()[:, None]).clamp(
        max=n - 1
    )
    return torch.gather(ctrl, 1, idx[..., None].expand(-1, -1, ctrl.shape[2]))


def n_frozen(latency: int, spacing: int, commit: str) -> int:
    """Control points a call keeps from the running chunk.

    ``latency`` freezes the curve up to the swap (exact C2 join at any latency), ``now`` only matches
    value and two derivatives at the call frame, ``none`` replaces the whole chunk (naive swap).
    """
    if commit == "latency":
        return DEGREE + math.ceil(latency / spacing)
    if commit == "now":
        return DEGREE
    if commit == "none":
        return 0
    raise ValueError(f"Unknown commit mode {commit!r}; expected 'latency', 'now' or 'none'.")


def project(
    ctrl: Tensor,
    frozen: int,
    spacing: float,
    position_limit: float,
    velocity_limit: float,
    acceleration_limit: float = 0.0,
) -> Tensor:
    """Clip the free control points so the curve respects the limits (0 disables a limit).

    A B-spline lies in the convex hull of its control points; its first derivative is a B-spline with
    control points ``(c[i+1] - c[i]) / spacing`` and its second one a B-spline with control points
    ``(c[i+1] - 2 c[i] + c[i-1]) / spacing**2``. So bounding the points and their first and second
    differences bounds position, velocity and acceleration everywhere on the curve. Each free point is
    clamped, in order, to the intersection of the position box, the velocity interval around the previous
    point and the acceleration interval around its linear extrapolation. Frozen points are left unchanged
    (they are assumed feasible, as the output of an earlier projection).

    The velocity and acceleration intervals always meet, and so do the position box and the velocity
    interval around a point inside the box; but all three can be disjoint when the curve runs into the
    position limit faster than it can brake. Position and velocity then win: the point is clamped to their
    intersection and the acceleration bound is exceeded there. The bounds hold exactly only away from the
    position limit.
    """
    if position_limit <= 0 and velocity_limit <= 0 and acceleration_limit <= 0:
        return ctrl
    out = ctrl.clone()
    max_step = velocity_limit * spacing
    max_bend = acceleration_limit * spacing**2
    for i in range(frozen, out.shape[1]):
        point = out[:, i]
        if i > 0 and velocity_limit > 0:
            point = out[:, i - 1] + (point - out[:, i - 1]).clamp(-max_step, max_step)
        if position_limit > 0:
            # Clipping towards the bound only shrinks the step from a feasible previous point.
            point = point.clamp(-position_limit, position_limit)
        if i > 1 and acceleration_limit > 0:
            centre = 2 * out[:, i - 1] - out[:, i - 2]
            lo, hi = centre - max_bend, centre + max_bend
            if velocity_limit > 0:
                lo = torch.maximum(lo, out[:, i - 1] - max_step)
                hi = torch.minimum(hi, out[:, i - 1] + max_step)
            if position_limit > 0:
                lo, hi = lo.clamp(min=-position_limit), hi.clamp(max=position_limit)
            # `clamp` with lo > hi returns hi, so an empty intersection keeps the position-velocity point.
            point = torch.where(lo <= hi, out[:, i].clamp(lo, hi), point)
        out[:, i] = point
    return out


def derivative_bounds(ctrl: Tensor, spacing: float) -> tuple[Tensor, Tensor]:
    """Per chunk, the largest absolute velocity and acceleration of the curve's control-point bounds.

    Returns ``(B,)`` tensors: the max of ``|c[i+1] - c[i]| / spacing`` and of
    ``|c[i+1] - 2 c[i] + c[i-1]| / spacing**2`` over points and channels, upper bounds of the curve's
    speed and acceleration per channel.
    """
    first = ctrl.diff(dim=1)
    second = first.diff(dim=1)
    return (
        first.abs().amax(dim=(1, 2)) / spacing,
        second.abs().amax(dim=(1, 2)) / spacing**2,
    )
