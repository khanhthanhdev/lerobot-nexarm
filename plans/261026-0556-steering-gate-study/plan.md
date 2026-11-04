# Test Plan: Gated Activation Steering on NexArm Policies

Status: code ready, not yet run. The code lives in the local repo `C:\Users\Administrator\code\policy-steering` (package `psteer`). Its `scripts/run_all.sh` runs the whole study unattended: a smoke test and throughput benchmark, training, then phases 0–4 with the decision rules below applied automatically. Evaluation seeds start at 2,000,000 so that they never overlap the dataset generator's training seeds (below 1,000,000). Decisions on 2026-10-26:

- ACT is the first policy.
- Everything runs in sim first.
- Training uses the existing 1000-episode sim dataset.

TurboVLA's hook sites are in `plans/reports/research-261026-0601-turbovla-architecture-hook-sites.md`.

Builds on `plans/reports/research-261025-2308-steering-research-gaps-and-ideas.md` (gaps G1–G8, Ideas 1–7).

## Bottom line

The study can't start yet: there is no trained NexArm checkpoint and no baseline success rate. `experiments/261024-bspline-vs-diffusion-stack-bowls` is still pending, and its sim eval needs a Linux machine with EGL. Phase 0 removes that blocker, and Phase 1 decides whether the study continues at all.

The repo has two things that make it a better test bed than the published setups:

- `NexArmStackBowls-v0` records which stages each episode reached (`stage_eval.py`), so each failure has a stage, not just a fail flag.
- The B-spline and Diffusion policies accept a fixed `noise` in `predict_action_chunk`, so each episode can be run with and without steering on the same env seed and the same sampled noise. None of the steering papers evaluates in pairs like this.

## Outcome, constraints, non-goals

- **Outcome:** decide, with tests strong enough to trust, whether (a) a gate that works with any edit operator makes steering help more often and hurt less, (b) a gate trained only on successes is as good as one that needs failures, and (c) handling chunks differently (replanning early, reweighting the ensemble) changes how much steering helps.
- **Constraints:**
  - Policies stay frozen.
  - All code lives in plugins or experiment folders, and LeRobot is not modified.
  - One 8 GB GPU (marinelab) plus a Linux EGL machine for the sim.
  - Each method is reimplemented from its paper, because SteerAct's repo has no licence.
- **Non-goals:**
  - Large VLAs beyond TurboVLA.
  - Training SAEs on more than one layer.
  - Updating the reference banks online (Idea 7).
  - Steering on the real robot before the sim results hold up.

## Phases

| #   | Phase                                 | Depends on | Exit (go/no-go)                                                               |
| --- | ------------------------------------- | ---------- | ----------------------------------------------------------------------------- |
| 0   | Baselines and the 20–80% success band | —          | One policy–task pair has a base success rate between 20 and 80% on 100 seeds. |
| 1   | Logging and feasibility probes        | 0          | Held-out probe AUC ≥ 0.75 for success, or progress R² ≥ 0.5. Otherwise stop.  |
| 2   | Screening grid: operators × gates     | 1          | The top 3 cells beat the base policy in the paired screen.                    |
| 3   | Confirmation runs                     | 2          | Pre-registered paired test on fresh seeds.                                    |
| 4   | Chunk handling and the site question  | 3          | —                                                                             |
| 5   | Real arm (optional)                   | 4          | Only if Phase 3 shows a gain of at least 15 points.                           |

### Phase 0 — Baselines

- Train ACT on the 1000-episode sim dataset (`scripts/train_act.sh`). Keep `chunk_size=100`, and save checkpoints along the way so that an earlier one can serve if the final checkpoint succeeds too often. B-spline and Diffusion come later.
- Evaluate each with `lerobot-nexarm-eval` on 100 seeds, recording success and the per-stage rates.
- If every policy is outside the 20–80% band:
  - Use an earlier checkpoint, as COAST does, and record the training step.
  - Or add a controlled disturbance: object pose jitter or action noise. Keep it fixed and document it, because it changes what the result means.
- Pick a task whose episodes are long enough. SteerAct's descriptor needs at least 5 calls.
  - B-spline and Diffusion both default to `n_action_steps=32`. That gives about 28 policy calls on stack-bowls (900 steps) and about 12 on pick-place (400 steps).
  - ACT defaults to `n_action_steps=100`, which would give only 9 calls on stack-bowls. `psteer` calls `predict_action_chunk` itself, with an execution stride set at run time (default 25, which gives 36 calls), so ACT needs no retraining for this.

### Phase 1 — Logging and feasibility

- **Hook site.** Log one vector per policy call:
  - For B-spline and Diffusion, log the global conditioning vector (the observation encoding fed into the UNet's FiLM layers). It is computed once per call and does not depend on the denoising step, which avoids RoboIRS's "which denoise step" question.
  - For ACT, log the encoder output, mean-pooled.
  - Also log the action chunk and the env state.
- **Seed splits, fixed before any fitting:**
  - _Fit_ (60 episodes): the reference bank, probes, conceptors and SAE.
  - _Select_ (60 episodes): choosing directions and strengths.
  - _Test_ (≥ 200 episodes per arm): reported numbers only.
- **Probes:**
  - Logistic regression for success.
  - Ridge regression for progress, with label `t/T`, trained on successes only.
  - A stage classifier, trained on the stage flags.
  - Report each on the select split.
- **No-go rule:** the literature shows steering fails on models whose outcomes are not linearly separable (WA-LQR, RoboIRS). If both probes are below the thresholds, write that up as the result, a negative finding for small policies, and stop.

### Phase 2 — Screening grid

- **Operators:**
  - Probe direction (RoboIRS).
  - Contrastive conceptor (COAST).
  - SAE feature (SteerAct). The SAE is 8×d with k scaled down to d; it is cheap because d is small.
  - Demonstration-variance filter (DHS, linear version).
- **Gates:**
  - Always on.
  - SteerAct's ordinal-pattern gate.
  - Progress-lag gate, trained on successes only.
  - Distance-to-reference (OOD) gate.
  - A random gate that fires as often as the best gate. This is the control the papers lack: it shows whether the gate's _timing_ matters or only how often it fires.
- **Size:** 4 × 5 = 20 cells × 50 paired episodes, plus the base run. Strength is tuned per operator on the select split only.
- **Keep:** the 3 cells with the best paired difference that also have a harm rate below 10%.

### Phase 3 — Confirmation

- Run the base policy and the 3 kept cells on 200 fresh test seeds, with matched noise.
- **Primary test:** McNemar's test on paired success, with Holm correction across the 3 cells.
- **Secondary results:**
  - Stage-reach rates.
  - Harm rate.
  - Gate precision and recall against eventual failure.
  - Time from the gate opening to the stage where the episode failed.
- **Selection-bias check (Idea 3):** repeat with the direction chosen by causal score and held-out separation, versus SteerAct's top-1 choice on the fit split.

### Phase 4 — Chunk handling and site

- **Chunk handling (Idea 4):** with the best cell, compare three settings:
  - Steer on the next call (as all papers do now).
  - Steer and replan immediately when the gate opens.
  - Steer and down-weight older chunks in the temporal ensemble.

  Run each with temporal ensembling on and off. In LeRobot only ACT has temporal ensembling built in. For B-spline or Diffusion, this step depends on the ensembler from the state-feedback-ensembling work (`plans/261025-2045-state-feedback-ensembling/`).

- **Site (Idea 6):** on TurboVLA, apply the probe operator at the VLM output and at the action expert, using the same seeds.
- **Action space (Idea 5):** as a baseline, apply the same operator to the B-spline control points instead of the hidden state.

## Metrics, for every cell

| Metric                              | Definition                                                                                                 | Why                                                                    |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Paired Δ success                    | Steered minus base success, on the same seed and noise.                                                    | Removes sampling noise, so fewer episodes are needed.                  |
| Harm rate                           | Episodes the base policy solved that steering fails, divided by the base policy's successes.               | Shows the "steering hurts" effect, which papers only see in aggregate. |
| Rescue rate                         | Episodes the base policy failed that steering solves, divided by the base policy's failures.               | The benefit, measured in the same way.                                 |
| Stage-reach rates                   | From `stage_eval.py`.                                                                                      | Shows which part of the task improves.                                 |
| Gate firing rate, precision, recall | Precision and recall of the gate's firing, scored against the base policy's outcome on the paired episode. | Evaluates the gate as a failure detector, which no paper does.         |
| Mean ‖Δa‖ per call                  | How far steering moves the action chunk.                                                                   | Separates "edit had no effect" from "edit had the wrong effect".       |
| Added latency                       | Milliseconds per call.                                                                                     | Small policies run at higher control rates than VLAs.                  |

## Statistical power

Unpaired, 200 episodes per arm detects a difference of about 14 points around a 50% base rate (α = 0.05, power 0.8). Detecting 10 points would need about 390 per arm. Pairing helps only as much as the outcomes are correlated across arms. Phase 1 measures that correlation by running the base policy twice with different noise. Set the Phase 3 sample size from that measurement, not from these defaults.

## Problems found while planning

These are problems in the literature that affect testing, in addition to G1–G8 in the earlier report.

1. **No paper reports harm and rescue separately.** The only number given is net Δ success. A +9 point net gain could be +20 rescued and −11 harmed. This is the core question for a gate, and nobody has measured it.
2. **Gates are never scored as detectors.** SteerAct reports only end success. It never checks whether its failure gate fires on episodes that actually fail, or how early.
3. **No control for how often the gate fires.** Without a random gate at the same firing rate, "gating helps" may simply mean "steering less often helps".
4. **Evaluation ignores the policy's own randomness.** Diffusion and flow policies sample. All papers compare independent runs, so part of each reported difference is sampling noise. Using matched noise fixes this for B-spline and Diffusion. TurboVLA needs its sampler seeded at each call.
5. **Directions are chosen and tested on the same few rollouts.** SteerAct picks its feature on the same 24 rollouts its gate uses, with no held-out data. The select split above prevents this.
6. **Outcomes are only success or failure.** Stage labels allow finer descriptors and gates, such as "steer only during grasp", which no paper tries.
7. **Getting failures changes the setup.** Using early checkpoints or added disturbances to reach the 20–80% band changes the policy or the task. Results must name which was used, since COAST's early checkpoints are not the deployed policy.

## Risks

- **Probes are too weak** (Phase 1 no-go). Mitigation: report it as a finding and switch the effort to the B-spline or state-feedback work.
- **Sim throughput** is unknown, and the grid is about 1,200 screening plus 800 confirmation episodes per task. Mitigation: screen with 50 episodes and use parallel envs.
- **Stage-flag reliability:** stage latches may misfire. Mitigation: audit 20 episodes by video before using stage labels.

## Unresolved questions

- How many episodes per hour does stack-bowls run on the EGL machine? This sets whether 2 tasks are feasible.
- Is ACT on pick-place in the band, or does that task need a disturbance?
- Does TurboVLA's sampler accept a fixed noise tensor, as the B-spline and Diffusion policies do?
- Should Phase 5 (real arm) be included at all, given the venue choice still open from the earlier report?
