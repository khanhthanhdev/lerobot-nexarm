# Plans

One folder per plan, named `YYMMDD-HHMM-<slug>`, with a `plan.md` (outcome, decisions, phases, acceptance criteria) and phase files when the work needs them. Reviews and research reports are in [`reports/`](reports/). Plans are dated records: they keep the paths and facts of their time, and a closed plan is not updated when the code moves on. The current user documentation is in [`docs/`](../docs/README.md).

## Open work

| Plan                                                                                      | Status  | What is left                                                                                                                                                                               | Needs                   |
| ----------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------- |
| [NexArm CAD zero pose = real Home pose](261019-1440-nexarm-cad-home-pose/plan.md)         | backlog | Measure the joint signs on the real follower (open problem 6), a follower-only recorder for sim fitting (7), regenerate sim datasets and calibrations (8), the Isaac USD and mobile assets | Robot session           |
| [NexArm single-arm sim fidelity](261014-1423-nexarm-sim-fidelity/plan.md)                 | backlog | All five phases: real-to-sim joint mapping, canonical asset tree, scene and object fixes, servo model, collision refit. Starts after the CAD plan, which fixes the zero pose it builds on  | Robot session, CAD plan |
| [Plan-Bind methodology fixes before D1b](261031-0052-plan-bind-methodology-fixes/plan.md) | active  | All five phases coded; left: the D1b smoke run with `onestep_pos`, a D1 checkpoint load and the plan-error runs on the server                                                              | GPU server              |

Research runs that the reorganization made possible, tracked in their experiment folders rather than in a plan:

| Experiment                                                                                             | What is left                                                               | Needs              |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------------ |
| [TurboVLA NexArm parity](../experiments/261024-turbovla-nexarm-parity/README.md)                       | Full 50k-step runs, 50-episode `NexArmStackBowls-v0` eval, the 2-GPU check | GPU                |
| [TurboVLA on LIBERO](../experiments/261024-turbovla-libero/README.md)                                  | Training and the four-suite eval                                           | Linux GPU with EGL |
| [B-spline vs diffusion, stack bowls](../experiments/261024-bspline-vs-diffusion-stack-bowls/README.md) | 50-episode sim comparison and 10 real-arm episodes                         | GPU, real arm      |
| [Plan-Bind D2: measured latency](../experiments/261028-plan-bind-d2-latency/README.md)                 | Inference timings and about 40 eval cells on D1's checkpoints              | GPU                |
| [Plan-Bind D3: Push-T](../experiments/261028-plan-bind-d3-pusht/README.md)                             | Four policies, two seeds, 100k steps each, and 66 eval cells               | GPU                |
| TurboVLA and B-spline on the real arm                                                                  | 10 `lerobot-rollout` episodes and a desktop-app eval session per policy    | Real arm           |

## Closed

| Plan                                                                             | Status    | Result                                                                                                     |
| -------------------------------------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------- |
| [Centralize the NexArm documentation](261025-1331-docs-centralization/plan.md)   | completed | All user guides in `docs/`, one owner per topic, start-here index                                          |
| [Research repo reorganization](261024-0016-research-repo-reorganization/plan.md) | completed | Upstream LeRobot pinned, NexArm and policy plugins, TurboVLA and B-spline through the LeRobot CLIs (PR #3) |
| [NexArm consistency fixes](261011-0836-nexarm-consistency-fixes/plan.md)         | completed | One set of conventions for ports, gripper, joint mapping and cameras (`8e216eb28`, `d15b42d34`)            |
