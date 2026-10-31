---
title: "Phase 1: Position-bearing one-step control"
status: in-progress
priority: P1
effort: 3h
dependencies: []
---

# Phase 1: Position-bearing one-step control

## Context

- Review §3: [`review-261031-0736-plan-bind-methodology.md`](../reports/review-261031-0736-plan-bind-methodology.md)
- Code: `policies/lerobot_policy_bspline/src/lerobot_policy_bspline/planbind.py` (`PlanBindModel.onestep`, lines 87–91), `configuration_bspline.py` (`planbind_*` fields, `_validate_planbind`)
- Experiment: `experiments/261029-plan-bind-d1b-seeds/run_all.sh` (`job_list`, `variant_args`, `base_config`)
- Tests: `policies/lerobot_policy_bspline/tests/test_planbind.py`

## Overview

`onestep` maps an all-zeros `(B, n_fine, D)` sequence through LeRobot's `DiffusionConditionalUnet1d`. That UNet has no positional encoding along the sequence; for a constant input its interior output is exactly periodic with period 4, and every other position signal comes from zero padding at the ends. Bind receives the lifted plan, which varies along the sequence. This phase adds a one-step variant whose input carries position but no plan, so D1b can separate "two-step structure" from "position-bearing input".

The hypothesis is about unequal information, not inability: the trained `onestep` reached 36 to 84 % success, and `plan` also maps zeros to 7 control points and works. The probe that motivated this used random weights.

## Requirements

- [ ] A config option selects the one-step input: `zeros` (current behaviour, default) or `learned`.
- [ ] With `learned`, the model owns a persistent `nn.Parameter` of shape `(n_fine, action_dim)`, initialised to a linear ramp from −1 to 1 on every channel, so position is present at step 0 and the parameter can adapt.
- [ ] The parameter exists only when the option is on, so D1's checkpoints (reused by D2) load with the default and no missing-key error.
- [ ] `predict_action_chunk` and `SplineRuntime` need no change: they call `model.onestep(global_cond)`.
- [ ] D1b trains and evaluates the variant as `onestep_pos`, scheduled early in `job_list` so its latency-0 cell is among the first results.
- [ ] Parameter delta is stated for fairness accounting: `n_fine × action_dim` = 19 × 6 = 114 numbers on D1's grid, against a UNet of tens of millions.

## Implementation steps

1. `configuration_bspline.py`
   - Add `planbind_onestep_input: str = "zeros"` next to the other `planbind_*` fields, with a docstring line: "`zeros` feeds the UNet a constant sequence (D1's control); `learned` feeds a trained `(n_fine, action_dim)` input initialised to a linear ramp, so the input carries sequence position."
   - In `_validate_planbind`, raise `ValueError` unless the value is `"zeros"` or `"learned"`, and raise when it is `"learned"` while `bspline_mode` is not `"onestep"`: the option has no meaning for `planbind` or `diffusion`, and a silent no-op would hide a mistyped experiment config.
2. `planbind.py`
   - Update the module docstring's `onestep` bullet to name both inputs.
   - In `PlanBindModel.__init__`, after `self.n_coarse, self.n_fine = knot_grid(config)`:
     ```python
     if config.bspline_mode == "onestep" and config.planbind_onestep_input == "learned":
         ramp = torch.linspace(-1.0, 1.0, self.n_fine)[:, None].expand(-1, config.action_feature.shape[0])
         self.onestep_input = torch.nn.Parameter(ramp.clone())
     else:
         self.onestep_input = None
     ```
   - In `onestep()`, build the input as `self.onestep_input.expand(B, -1, -1)` when the parameter exists, else the current zeros. The role stays timestep 0.
3. `test_planbind.py`
   - Extend `test_config_validation` with an invalid `planbind_onestep_input`.
   - Add `test_onestep_learned_input_trains_and_round_trips`: build `make_config(bspline_mode="onestep", planbind_onestep_input="learned")`, run `forward` and `backward`, assert `onestep_input.grad` is not `None`, save with `save_pretrained`, load, and assert the parameter is restored and `predict_action_chunk` matches before and after the round trip. Assert the zeros config has no `onestep_input` key in `state_dict()`.
   - `test_default_mode_is_unchanged` and `test_save_and_load_round_trip` must still pass unchanged.
4. `experiments/261029-plan-bind-d1b-seeds/run_all.sh`
   - `job_list`: emit `onestep_pos_s$seed` for every seed right after `onestep_long_s1000`, so the decisive latency-0 number arrives early.
   - `variant_args`: `onestep_pos) echo --policy.planbind_onestep_input=learned ;;`. `base_config` already maps `onestep_*` to `train_onestep.yaml`.
   - Header comment: list `onestep_pos` among the controls and its purpose.
   - `eval_matrix` needs no change: every non-`bsp10` job gets the latency 0, 200, 300 ms cells with commit `latency` and the naive cell at 0 ms.
5. Record the seed trade-off for the user before the run (phase 5 writes it into the README): 3 seeds of `onestep_pos` add about 12 GPU-hours on a 3090-class card (D1 measured 3.7 h per `onestep` training); 1 seed adds 4 h. The default in this plan is 3 seeds, because this control decides how RQ1 is framed; the alternative is to drop `onestep_k2` (which lengthens the sequence and weakens the padding pathway further) to pay for it.

## Deferred

A one-step generative head from noise (flow or consistency), the proposal's original RQ1 control, would also carry position through its noise input and would answer a reviewer who rejects regression altogether. It needs a new head and training loop; not in this plan.

## Todo

- [x] Config field and validation
- [x] Parameter and `onestep()` input
- [x] Tests
- [x] `run_all.sh` job and flags
- [x] Ruff and the plugin test suite clean on Windows
- [ ] Server: smoke run and D1 `onestep` checkpoint load on `vishc-server-2`

## Verification

Windows (this box):

- `uv run pytest policies/lerobot_policy_bspline/tests/test_planbind.py -svv`
- `uv run pytest policies/lerobot_policy_bspline/tests -svv` (the default mode and the diffusion path are untouched)
- `uv run ruff check policies/lerobot_policy_bspline --no-fix && uv run ruff format policies/lerobot_policy_bspline --check`
- `bash -n experiments/261029-plan-bind-d1b-seeds/run_all.sh` (syntax only; run it through PowerShell's `bash`)

Server (`vishc-server-2`, Linux with EGL):

- `experiments/261029-plan-bind-d1b-seeds/run_all.sh smoke` with `GPUS=6`; the smoke run must train `onestep_pos_s1000` for 200 steps and write its `final0` cell.
- Load a D1 `onestep` checkpoint from `thanhkt/plan-bind-d1` with the new code and confirm `BsplinePolicy.from_pretrained` succeeds with no missing or unexpected keys.

## Success criteria

- The learned input is trained, saved and restored; the default config produces a model with the same `state_dict` keys as before.
- The smoke run shows `onestep_pos` cells next to `onestep` cells.
- The RQ1 decision rule is written (phase 5) before the full run: if `onestep_pos` closes most of the gap to Plan-Bind at latency 0 across seeds, RQ1 is reframed to the interface (RQ2, RQ3); if the gap holds, RQ1 stands with a matched-input control.

## Risks

- The ramp initialisation could itself be a poor prior for joint trajectories; the parameter is trained, so this only affects early steps.
- With 11 trainings plus 3 more, D1b's wall time on three A5000s grows to roughly 21 h; GPUs 0 to 5 on `vishc-server-2` were busy with other users' jobs on 2026-10-31.
