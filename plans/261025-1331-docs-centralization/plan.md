---
title: "Centralize the NexArm documentation"
description: "Move every user guide into docs/, one owner per topic, a start-here index by role, and close or backlog the open plans."
status: completed
priority: P2
effort: "1 day"
branch: docs/centralize-docs
tags: [docs]
blockedBy: []
blocks: []
created: 2026-10-25
---

# Centralize the NexArm documentation

## Outcome

A new contributor opens `docs/README.md`, picks their role, and reaches every guide from there. Each topic is written in exactly one file; the others link to it.

## User decisions (fixed)

1. All guides live in `docs/`, grouped into `getting-started/`, `guides/`, `reference/` and `hiwonder/`. The root keeps `README.md`, `AGENTS.md`/`CLAUDE.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md` and `AI_POLICY.md`.
2. One owner per topic. `README.md` shrinks to an overview, a 5-minute quickstart and links.
3. Plans: 261011 completed (its branch is merged), 261014 and 261019 backlog, the reorganization plan completed. `plans/README.md` lists the open work.

## Non-goals

- No change to code behaviour, commands or flags. Commands are moved, not rewritten.
- The Hiwonder vendor tutorials move unchanged.
- `docs/service/` stays where it is: `openapi.json` and `events.json` are a machine contract that the engine, its tests and the desktop app's sync script read by path. The index links its README.
- READMEs that sit next to what they describe stay there: `sim/`, `firmware/`, `hardware/`, `docker/`, `policies/*`, `examples/nexarm/stack_bowls_act.md`, `experiments/`.
- Dated records in `plans/` and `plans/reports/` keep their old links.

## Topic owners and file moves

| New file                             | Owns                                                                                      | Built from                                                            |
| ------------------------------------ | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `docs/README.md`                     | Start here: paths by role, map of every doc                                               | new; AGENT_GUIDE §3, §11                                              |
| `docs/getting-started/install.md`    | uv, dependency profiles, Git LFS, HF login, serial permissions, platforms, verify install | run.md §1, README install                                             |
| `docs/getting-started/simulation.md` | Sim quickstart: viewer, keyboard teleop, pick-and-place, gym, Rerun                       | run.md §2, README path A                                              |
| `docs/getting-started/hardware.md`   | Wiring, ports, cameras, leader-follower and Cartesian teleop, vision grasp, camera align  | run.md §3, README path B steps 1-5                                    |
| `docs/guides/pipeline.md`            | Record, inspect, replay, train, rollout, benchmark                                        | `docs/source/nexarm/pipeline.md` + run.md §4, README path B steps 6-8 |
| `docs/guides/data-collection.md`     | The stack-bowls collection rig (`collect.sh`), keys, merge                                | `DATA_COLLECTION_GUIDE.md`                                            |
| `docs/guides/training-tips.md`       | Data tips, which policy, how long to train, evaluation targets                            | AGENT_GUIDE §5-8, §10                                                 |
| `docs/guides/sim-training.md`        | Multi-GPU sim generation and training                                                     | `docs/source/nexarm/sim_training.md`                                  |
| `docs/guides/server-training.md`     | TurboVLA on a remote server or Slurm                                                      | `SERVER_TRAINING_GUIDE.md`                                            |
| `docs/guides/turbovla.md`            | TurboVLA install, local dev, training, conversion                                         | `docs/source/nexarm/turbovla_guide.md`                                |
| `docs/guides/sim-calibration.md`     | Sim system identification                                                                 | `docs/source/nexarm/sim_calibration.md`                               |
| `docs/reference/architecture.md`     | LeRobot in 60 s, package layout, conventions                                              | AGENT_GUIDE §2, README architecture, AGENTS.md conventions            |
| `docs/reference/python-api.md`       | Kinematics/dynamics, motor bus, idle torque timeout                                       | run.md §5, README kinematics                                          |
| `docs/reference/troubleshooting.md`  | General FAQs (permissions, timeouts, cameras, slow loops)                                 | run.md §7, README troubleshooting                                     |
| `docs/hiwonder/1_...md` .. `8_...md` | Vendor tutorials                                                                          | `docs/[1-8]_*.md`, unchanged                                          |
| `CONTRIBUTING.md`                    | Tests and quality checks (already)                                                        | run.md §6 and README testing become links                             |
| `AGENTS.md`                          | Agent rules, including AGENT_GUIDE §1 (ask the user first)                                | AGENT_GUIDE §1                                                        |

Removed: `run.md`, `AGENT_GUIDE.md`, `DATA_COLLECTION_GUIDE.md`, `SERVER_TRAINING_GUIDE.md`, `docs/source/`. Moves use `git mv` so history follows the file.

## Steps

1. Branch `docs/centralize-docs` from `main`.
2. `git mv` the files with a direct successor; split `run.md`, `AGENT_GUIDE.md` and README into the new files; drop the duplicates in favour of a link to the owner.
3. Write `docs/README.md` and the slim `README.md` (remove the broken `README_cn.md` link).
4. Update every link outside `plans/`: AGENTS.md, CONTRIBUTING, `sim/README.md`, `examples/`, `policies/`, `pyproject.toml`, `packages/nexarm_service/src/nexarm_service/remote.py`.
5. Close the plans and write `plans/README.md`.
6. Check: a relative-link checker over every tracked `.md` outside `plans/`, a grep for the old paths, `pre-commit run --all-files`.
7. PR on `khanhthanhdev/lerobot-nexarm`.

## Acceptance criteria

- Every tracked `.md` outside `plans/` has no broken relative link or anchor.
- No reference to the removed paths outside `plans/`.
- Each topic in the table above has one full description; other files link to it.
- `docs/README.md` reaches every guide in at most one click.
- Pre-commit and CI pass.

## Risk and rollback

Docs and comments only; revert the merge commit to roll back. External links to the old root guides on GitHub break; the PR description lists the new locations.
