---
title: "Research repo reorganization"
description: "Turn lerobot-nexarm into a research repo: custom policies as LeRobot plugins, TurboVLA and B-spline Policy running through lerobot-train/eval/rollout, NexArm hardware as plugins, and upstream LeRobot as a dependency."
status: in-progress
priority: P1
effort: "3-5 weeks (phases 06-07 need GPU time)"
branch: main
tags: [refactor, policies, turbovla, bspline, upstream, docs]
blockedBy: []
blocks: []
created: 2026-10-24
---

# Research repo reorganization

## Outcome

Every policy you research (ACT, SmolVLA, pi05 and the other upstream baselines, plus TurboVLA, B-spline Policy and future VLAs) trains with `lerobot-train`, evaluates with `lerobot-eval` (NexArm sim and benchmarks such as LIBERO) and runs on the real arm with `lerobot-rollout` and the desktop app, using the same data loading and normalization. NexArm hardware, sim and the desktop service sit on top of an unmodified upstream LeRobot, so new LeRobot releases are a version bump.

Research behind this plan: [research-261024-0706-research-repo-layout-options.md](../reports/research-261024-0706-research-repo-layout-options.md).

## Decisions so far

| Decision                       | Choice                                                                                                                                              | Source                                                |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Edits to upstream train/record | Not intended; you use LeRobot's abstractions. The existing edits get retired (phase 02)                                                             | User, 2026-10-24                                      |
| What TurboVLA trains on        | NexArm datasets now; other datasets for benchmarks later                                                                                            | User, 2026-10-24                                      |
| B-spline Policy                | Active research, port it                                                                                                                            | User, 2026-10-24                                      |
| TurboVLA source                | Stays its own repo, installed as a pinned dependency (no submodule)                                                                                 | Recommended, research report                          |
| B-spline source                | Ported into a LeRobot policy plugin; the `bspline-policy` fork (of `B-spline-policy/bspline-policy`, built on `diffusion_policy`) is reference only | Recommended; it is third-party code, not LeRobot code |
| TurboVLA wrap                  | Inference first (phase 05), then training (phase 06)                                                                                                | Recommended                                           |
| Upstream                       | Add `upstream` remote, merge, retire core edits, then depend on upstream `lerobot`                                                                  | Recommended                                           |
| Old TurboVLA scripts           | Retired after the 2k-step GPU parity; the full-run and LIBERO comparisons no longer gate it (phase 06)                                              | User, 2026-10-24                                      |
| TurboVLA image-step slowdown   | Follow-up, not part of this plan                                                                                                                    | User, 2026-10-24                                      |
| Long training runs             | The user trains on their own servers; this repo carries the code and experiment configs                                                             | User, 2026-10-24                                      |
| Removing `src/lerobot/`        | NexArm-side fallback for each carried patch; upstream PRs only for plain bug fixes (phase 08 table)                                                 | Maintainer, delegated by user 2026-10-24              |

## Target layout

```
lerobot-nexarm/
├── packages/                       # NexArm side, LeRobot plugins (phase 08)
│   ├── lerobot_robot_nexarm/       # motors/nexarm + nexarm_follower, nexarm_sim, mobile_bi_nexarm_sim
│   ├── lerobot_teleoperator_nexarm/
│   ├── lerobot_env_nexarm/         # NexArmEnv config + gym envs
│   └── nexarm_service/             # desktop-app engine, keeps lerobot-nexarm-service
├── policies/                       # research policies, LeRobot plugins
│   ├── lerobot_policy_turbovla/    # adapter; model code comes from the turbovla package
│   └── lerobot_policy_bspline/     # port of B-spline Policy
├── experiments/<date>-<slug>/      # config + launch + README (hypothesis, data, ckpt, result)
├── tools/nexarm/                   # calibration, probes, dataset audits
├── examples/nexarm/                # tutorials only: teleoperate, record, rollout
├── sim/  firmware/  hardware/  docs/  plans/  scripts/
└── pyproject.toml                  # uv workspace; depends on upstream lerobot (pinned)
```

## Phases

| #   | Phase                                                                                    | Priority | Effort     | Depends on | Status                       |
| --- | ---------------------------------------------------------------------------------------- | -------- | ---------- | ---------- | ---------------------------- |
| 01  | [Housekeeping and baseline](./phase-01-housekeeping-and-baseline.md)                     | P1       | 2h         | none       | completed                    |
| 02  | [Upstream sync and retire core edits](./phase-02-upstream-sync-and-retire-core-edits.md) | P1       | 2-3d       | 01         | completed (real arm skipped) |
| 03  | [Workspace and experiments layout](./phase-03-workspace-and-experiments-layout.md)       | P1       | 3h         | 02         | completed                    |
| 04  | [TurboVLA as a pinned dependency](./phase-04-turbovla-pinned-dependency.md)              | P1       | 0.5-1d     | 03         | completed                    |
| 05  | [TurboVLA policy plugin, inference](./phase-05-turbovla-policy-inference.md)             | P1       | 2-3d       | 04         | in-progress                  |
| 06  | [TurboVLA training and LIBERO benchmark](./phase-06-turbovla-training-and-benchmark.md)  | P2       | 3-5d + GPU | 05         | in-progress                  |
| 07  | [B-spline Policy port](./phase-07-bspline-policy-port.md)                                | P2       | 3-5d + GPU | 03         | in-progress                  |
| 08  | [NexArm plugins on upstream LeRobot](./phase-08-nexarm-plugins-on-upstream-lerobot.md)   | P2       | 3-5d       | 02, 05     | completed                    |
| 09  | [Tools/examples split and docs sweep](./phase-09-tools-examples-split-and-docs.md)       | P2       | 1d         | 06, 07, 08 | completed                    |

Phase 07 can run alongside 04-06. Phase 08 can start once phase 05 has shown the plugin pattern works.

## Acceptance criteria

- On a fresh clone with no `git submodule` step, `uv sync` installs LeRobot, the NexArm plugins and the B-spline plugin, and `uv sync --extra turbovla` adds TurboVLA and its plugin.
- `lerobot-train --policy.type=turbovla` and `--policy.type=bspline` train on a NexArm dataset; `lerobot-eval` scores them on the NexArm stack-bowls sim and on LIBERO; `lerobot-rollout` and the desktop app run their checkpoints on the real arm.
- TurboVLA through LeRobot matches the current scripts: same actions on the same inputs (phase 05) and comparable sim success after training (phase 06).
- `src/lerobot/` no longer exists in this repo; LeRobot comes from a pinned upstream version (phase 08). Deferred by validation decision 5, then restored by decision 6 once `src/lerobot/` matched upstream; done 2026-10-24 (pinned to `d3793b7e36`).
- No doc, script or test points at a moved or deleted path (phase 09).
- NexArm tests pass on a Linux box at least as well as the phase-01 baseline.

## Non-goals

- New research results or tuning. This plan only changes where code lives and how it runs.
- Changing TurboVLA's own LIBERO/RoboTwin RLDS training inside the TurboVLA repo.
- Fixing the Windows-only test failures (no OpenGL, CRLF on `nexarm_scene.json`) beyond recording them.
- Changing the desktop app's HTTP/WebSocket contract.

## Risks

- **Dependency conflict.** TurboVLA pins `numpy<2` and `transformers<5`; this repo needs `numpy>=2` and `transformers>=5.4`. TurboVLA also installs a top-level `experiments` package. Phase 04 fixes both in the TurboVLA repo before anything else in the TurboVLA track.
- **Upstream PRs not merged.** Phase 02 gives each carried patch a fallback that removes it from `src/lerobot/`, so phase 08 never waits on upstream review.
- **Silent normalization mismatch.** A wrong processor trains and evaluates without errors but ruins results (LeRobot's own warning). Phases 05-07 gate on parity tests, not on "it runs".
- **Large first merge.** Upstream is 332 commits ahead and the fork modifies or deletes 25 core files. Phase 02 runs on a branch and includes a real-arm smoke test before merging.
- **Import path changes in phase 08.** Registered type names (`nexarm_follower`, `nexarm_leader`, `nexarm`) stay identical so datasets, checkpoints and the desktop app keep working; Python import paths change.

## Rollback

Phase 01 tags `pre-reorg`. Each phase lands as its own branch and merge, so any phase can be reverted on its own.

## Validation decisions (user, 2026-10-24)

1. `lerobot-record --merge_root` and `--rerun_save_path` move to a NexArm record wrapper with the same flags; `lerobot_record.py` returns to upstream (phase 02).
2. Stop pruning: accept upstream's non-NexArm robots and motors during the merge (phase 02).
3. Phase 08 removes `src/lerobot/` and depends on a pinned upstream `lerobot`.
4. The TurboVLA adapter lives in this repo at `policies/lerobot_policy_turbovla/`; TurboVLA stays a plain model package.
5. (2026-10-24) Phase 08 moves NexArm into `packages/` plugins now. `src/lerobot/` stays, carrying only generic patches, until each patch has a merged upstream PR or a NexArm-side fallback. This supersedes the removal in item 3.
6. (2026-10-24, after every patch got a NexArm-side fallback and `src/lerobot/` matched upstream) Remove `src/lerobot/` and pin `lerobot` to an upstream git commit. This restores item 3. The NexArm tests stay in the root `tests/` (only upstream's tests were deleted, so nothing had to move).
7. (2026-10-24) Phase 09 runs next, after the pin.
8. (2026-10-24) `lerobot-find-joint-limits` and the HIL-SERL commands stay without plugin loading; documented, no wrapper and no upstream PR.
9. (2026-10-24) `NexArmSim.name` stays `nexarm_follower`, so plain `lerobot-record`/`lerobot-rollout` stamp sim datasets with the physical type.
10. (2026-10-25) Merge `chore/upstream-sync` into `main` once CI passes, without the phase-02 real-arm smoke test (PR #3).

<!-- slug: research-repo-reorganization -->
