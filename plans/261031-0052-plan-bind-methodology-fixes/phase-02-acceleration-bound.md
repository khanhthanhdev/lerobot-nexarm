---
title: "Phase 2: Acceleration bound in project"
status: done
priority: P3
effort: 2h
dependencies: []
---

# Phase 2: Acceleration bound in `project`

## Context

- Review §1 (gaps against the proposal): [`review-261031-0736-plan-bind-methodology.md`](../reports/review-261031-0736-plan-bind-methodology.md)
- Proposal §3.3 claims position, velocity and acceleration limits (the Plan-Bind proposal, `proposal-261027-0633-two-step-knot-policy.md`, kept out of the repository)
- Code: `policies/lerobot_policy_bspline/src/lerobot_policy_bspline/uniform_spline.py` (`project`, lines 95–117), `planbind.py` (`SplineRuntime.step`, the `project(...)` call), `configuration_bspline.py`
- README: `policies/lerobot_policy_bspline/README.md`, the "Limits" bullet
- Tests: `test_planbind.py::test_projection_bounds_position_and_velocity`

## Overview

`project` clamps the free control points to a position box and bounds first differences by the velocity limit. RQ2 claims acceleration as well. For a uniform cubic B-spline with knot spacing `s`, the second derivative is a B-spline whose control points are `(c[i+1] − 2c[i] + c[i−1]) / s²`, so bounding second differences bounds acceleration everywhere on the curve. No RQ2 run is scheduled, so this phase is P3; it closes the gap between the proposal's claim and the code.

## Requirements

- [ ] `project(ctrl, frozen, spacing, position_limit, velocity_limit, acceleration_limit=0.0)`; `0` disables a limit, as now.
- [ ] Each free point `i` is clamped to the intersection of three intervals: position `[−P, P]`, velocity `c[i−1] ± V·s` (when `i ≥ 1`), acceleration `2c[i−1] − c[i−2] ± A·s²` (when `i ≥ 2`).
- [ ] The velocity and acceleration intervals always intersect when the previous step satisfied the velocity bound (their centres differ by `|c[i−1] − c[i−2]| ≤ V·s`). The position box can make the intersection empty when the curve approaches a joint limit at speed. Rule: **position and velocity win**: the point is clamped to the box intersected with the velocity interval (never empty when the previous point is in the box), and only the acceleration bound is violated at that point. This is stated in the docstring and the README, because the proposal's "exactly by construction" holds only away from the position wall.
- [ ] A helper `derivative_bounds(ctrl, spacing) -> tuple[Tensor, Tensor]` returns the max absolute first and second differences scaled by `s` and `s²` per batch element, for tests and later for the violation metric of RQ2.
- [ ] New config field `planbind_acceleration_limit: float = 0.0`, validated non-negative in the existing loop of `_validate_planbind`, passed by `SplineRuntime.step`.

## Implementation steps

1. `uniform_spline.py`
   - Rewrite the loop in `project`: compute `lo, hi` as tensors over `(B, D)` from the active constraints, intersect with `torch.maximum`/`torch.minimum`, then `point = point.clamp(lo, hi)` where `lo ≤ hi`, and `point.clamp(-P, P)` where the intersection is empty (`lo > hi`). Keep the early return when every limit is 0.
   - Add `derivative_bounds`.
   - Update the docstring with the second-difference identity and the position-wins rule.
2. `configuration_bspline.py`: add the field, its docstring line and the validation.
3. `planbind.py`: pass `cfg.planbind_acceleration_limit` in `SplineRuntime.step`.
4. `test_planbind.py`
   - Extend or add next to `test_projection_bounds_position_and_velocity`: random chunks with `position_limit=0`, assert `derivative_bounds` ≤ the limits after projection, and that frozen points are unchanged.
   - Add a position-wall case: a chunk heading into the box faster than it can brake; assert the position bound holds and document the velocity or acceleration excess in the assertion message.
5. `README.md` of the plugin: extend the "Limits" bullet with `planbind_acceleration_limit` and the position-wins caveat.

## Todo

- [x] `project` with three intervals and the helper
- [x] Config field, validation and runtime call
- [x] Tests
- [x] README bullet

## Verification

Windows: `uv run pytest policies/lerobot_policy_bspline/tests/test_planbind.py -k "projection or config" -svv`; `uv run ruff check policies/lerobot_policy_bspline --no-fix`.

No server step: the projection is pure tensor code. An RQ2 evaluation that sets the limits is out of scope.

## Success criteria

- With `position_limit=0`, the projected curve's velocity and acceleration never exceed the limits (checked by `derivative_bounds`).
- D1's checkpoints and configs load unchanged (the field defaults to 0).

## Risks

- A per-point Python loop over `n_fine` points runs once per call; it is already the shape of the current code and negligible next to the UNet.
