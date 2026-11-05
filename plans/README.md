# Plans

One folder per plan, named `YYMMDD-HHMM-<slug>`, with a `plan.md` (outcome, decisions, phases, acceptance criteria) and phase files when the work needs them. Reviews and research reports are in [`reports/`](reports/). Plans are dated records: they keep the paths and facts of their time, and a closed plan is not updated when the code moves on. The current user documentation is in [`docs/`](../docs/README.md).

## Open work

| Plan                                                                                      | Status      | What is left                                                                                                                                                                               | Needs              |
| ----------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| [NexArm CAD zero pose = real Home pose](261019-1440-nexarm-cad-home-pose/plan.md)         | backlog     | Measure the joint signs on the real follower (open problem 6), a follower-only recorder for sim fitting (7), regenerate sim datasets and calibrations (8), the Isaac USD and mobile assets | Robot session      |
| [NexArm single-arm sim fidelity](261014-1423-nexarm-sim-fidelity/plan.md)                 | in-progress | Phase 0 on the robot (shoulder-pan sign, mount height), the D435i fy read and `top` pose check, a viewer review of the collisions. Phases 1–4 and the kp choice are done                   | Robot session      |
| [Plan-Bind methodology fixes before D1b](261031-0052-plan-bind-methodology-fixes/plan.md) | active      | All five phases coded, and D1b ran with `onestep_pos` (3 seeds, 2026-10-29 to 2026-11-03); left: the plan-error run (phase 4) on D1's `planbind` checkpoint                                | GPU server         |
| [Plan-Bind on LIBERO (D4)](261103-1000-plan-bind-libero/plan.md)                          | proposed    | Fit-error check, a task-text condition for the policy, then the four-suite `lerobot-eval --env.type=libero` run. Starts after the D1b write-up and the D3 smoke run                        | Linux GPU with EGL |

Research runs that the reorganization made possible, tracked in their experiment folders rather than in a plan:

| Experiment                                                                                             | What is left                                                               | Needs              |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------------ |
| [TurboVLA NexArm parity](../experiments/261024-turbovla-nexarm-parity/README.md)                       | Full 50k-step runs, 50-episode `NexArmStackBowls-v0` eval, the 2-GPU check | GPU                |
| [TurboVLA on LIBERO](../experiments/261024-turbovla-libero/README.md)                                  | Training and the four-suite eval                                           | Linux GPU with EGL |
| [B-spline vs diffusion, stack bowls](../experiments/261024-bspline-vs-diffusion-stack-bowls/README.md) | 50-episode sim comparison and 10 real-arm episodes                         | GPU, real arm      |
| [Plan-Bind D3: Push-T](../experiments/261028-plan-bind-d3-pusht/README.md)                             | Four policies, two seeds, 100k steps each, and 66 eval cells               | GPU                |
| TurboVLA and B-spline on the real arm                                                                  | 10 `lerobot-rollout` episodes and a desktop-app eval session per policy    | Real arm           |

Plan-Bind D1 ([two-step](../experiments/261027-plan-bind-two-step/README.md)), D1b ([seeds](../experiments/261029-plan-bind-d1b-seeds/README.md)) and D2 ([latency](../experiments/261028-plan-bind-d2-latency/README.md)) have run; their results are in the experiment READMEs.

Plans whose code lives in a separate local repository (the plan records the design decisions; the repository's README covers usage):

| Plan                                                                       | Status                                  | Code                                   |
| -------------------------------------------------------------------------- | --------------------------------------- | -------------------------------------- |
| [State-feedback ensembling](261025-2045-state-feedback-ensembling/plan.md) | v0 done in sim; arm rollout not started | `../state-feedback-ensembling` (`sfe`) |
| [Gated activation steering study](261026-0556-steering-gate-study/plan.md) | Code ready, not run yet                 | `../policy-steering` (`psteer`)        |

## Closed

| Plan                                                                             | Status    | Result                                                                                                     |
| -------------------------------------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------- |
| [Centralize the NexArm documentation](261025-1331-docs-centralization/plan.md)   | completed | All user guides in `docs/`, one owner per topic, start-here index                                          |
| [Research repo reorganization](261024-0016-research-repo-reorganization/plan.md) | completed | Upstream LeRobot pinned, NexArm and policy plugins, TurboVLA and B-spline through the LeRobot CLIs (PR #3) |
| [NexArm consistency fixes](261011-0836-nexarm-consistency-fixes/plan.md)         | completed | One set of conventions for ports, gripper, joint mapping and cameras (`8e216eb28`, `d15b42d34`)            |
