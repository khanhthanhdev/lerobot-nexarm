# Research repo layout: options for policies, TurboVLA and upstream LeRobot

Date: 2026-10-24. Scope: how to organize `lerobot-nexarm` so custom policies and VLAs can be developed and compared, and how comparable projects do it.

## Facts measured in this repo

- The fork split from upstream `huggingface/lerobot` at `3e538352` (2026-07-08). It has 91 commits of its own, and upstream has 332 newer commits touching 488 files under `src/lerobot` (+65k / −20k lines).
- Outside the NexArm hardware folders, the fork modifies 23 upstream files, deletes 2 (`lerobot_setup_can.py`, `lerobot_setup_motors.py`) and adds 5 new ones. Only the modified and deleted files conflict on a sync. The modified ones include `scripts/lerobot_train.py` (140 lines changed), `lerobot_record.py` (222), `lerobot_rollout.py` (186), `lerobot_eval.py`, `datasets/lerobot_dataset.py` and `envs/configs.py`. These are the files that will conflict on every upstream sync.
- There is no `upstream` git remote.
- TurboVLA runs outside the LeRobot policy API. `examples/nexarm/train_turbovla.py` (1,128 lines) is its own PyTorch loop with torchrun support that reads `LeRobotDataset` directly. `rollout_turbovla.py` (638 lines) is its own rollout. The TurboVLA repo also has LIBERO and RoboTwin experiments, a mixed-suite trainer, and a NexArm experiment folder (`TurboVLA/experiments/nexarm`).
- `bspline-policy` is a submodule that has never been initialized here.
- LeRobot discovers plugins by installed distribution name (`lerobot_policy_*`, `lerobot_robot_*`, `lerobot_teleoperator_*`, `lerobot_env_*`, `lerobot_camera_*`) and imports that name as a module (`src/lerobot/utils/import_utils.py:227`).

## How other projects do it

- **LeRobot's own guidance.** Ship a new policy as an out-of-tree `lerobot_policy_<name>` package first. Move it into `src/lerobot/policies/` only once it is stable. A policy is three files: `configuration_<name>.py`, `modeling_<name>.py` and `processor_<name>.py`, with a factory called `make_<name>_pre_post_processors`. The docs say a normalization mismatch in the processor "will train and eval without erroring, yet silently wreck results".
- **Community plugin example (DiTFlow).** The policy lives in its own repo (`lerobot_policy_ditflow`) and is used from a separate example repo. Upstream LeRobot is just a dependency.
- **Trossen Robotics (hardware vendor, closest to your situation).** They dropped their LeRobot fork; it is now listed as "LeRobot Fork (Deprecated)". Their current repo is a uv workspace with `lerobot-robot-trossen` and `lerobot-teleoperator-trossen` plugins, and upstream `lerobot>=0.5.1` is a normal dependency. Users choose policy extras (`pi`, `smolvla`) in their own project.
- **Physical Intelligence openpi.** The model, its training loop and its configs live in one repo. LeRobot is used only as the dataset format, and a policy server handles inference. Trossen's openpi tutorial reports needing three different LeRobot versions across collection, training and inference, which is the version friction that comes with a separate trainer.
- **General tooling advice.** Submodules pin an exact commit but leave folders empty after a plain clone and need care in CI. Subtrees copy the code in but keep a sync path. A uv workspace is the lightest option when you own all the packages.

## Decision 1: how to hold TurboVLA and bspline-policy

| Option                       | What it means                                                                                                                                             | Good                                                                                                                             | Bad                                                                                                                                   |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| A. Submodule + thin adapter  | Keep the git submodule. Add `policies/lerobot_policy_turbovla/` in this repo that imports `turbovla`                                                      | TurboVLA keeps its own history, its LIBERO/RoboTwin work, and can be released with a paper                                       | Pointer drift (already happened: `5082ece6` vs `98ee1be3`), empty after clone, every model change is two commits in two repos         |
| B. Copy into this repo       | Move the TurboVLA source into `policies/`, delete the submodule                                                                                           | One repo, one commit per change, no drift                                                                                        | Loses the link to the TurboVLA repo; its non-NexArm experiments either come along or are dropped                                      |
| C. Pinned package dependency | Remove the submodule. Declare `turbovla` in `pyproject.toml` from git at a commit, or as a local editable path under `[tool.uv.sources]` while developing | `uv.lock` records the exact commit, so there is no separate pointer to forget. This matches how Trossen and DiTFlow consume code | Changing TurboVLA still means a commit there and a lock bump here; a local path setup only works on machines that have both checkouts |

Recommendation: TurboVLA is an active model you also run outside NexArm, so keep it a separate repo and use **C** (or **A** if you prefer git-native pinning). Choose **B** only if you will never use TurboVLA outside NexArm. For `bspline-policy`, decide whether it is still alive; if not, remove the submodule.

## Decision 2: wrap TurboVLA as a LeRobot policy

| Option                    | What it means                                                                                        | Good                                                                                                                                                                                                                                                                                      | Bad                                                                                                                                                                      |
| ------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A. Full wrap              | Config, policy and processor classes so `lerobot-train`, `lerobot-eval` and `lerobot-rollout` run it | Same data loading, normalization and eval as ACT, SmolVLA and pi05, so comparisons are fair. Everything in B, plus checkpoints push to the Hub with a model card. Retires about 1,800 lines of custom train and rollout scripts                                                           | Largest effort. `lerobot-train` may not cover TurboVLA's distributed and mixed-suite training as-is, and the processor has to reproduce TurboVLA's normalization exactly |
| B. Inference-only wrap    | Wrap loading and `select_action` only, keep TurboVLA's own trainer                                   | Fair eval and real-robot rollout through LeRobot tools at a fraction of the cost. The desktop app's rollout, eval and DAgger coaching also work, because `nexarm_service/rollout.py:59` loads any registered policy through `PreTrainedConfig.from_pretrained` and runs `lerobot.rollout` | Two training paths remain, so training-side comparisons are still apples to oranges                                                                                      |
| C. No wrap (openpi style) | Keep the custom scripts                                                                              | No work now                                                                                                                                                                                                                                                                               | Every comparison needs custom glue, and the scripts drift from LeRobot as it changes                                                                                     |

Recommendation: **B first, then A.** The inference wrap gives fair evaluation quickly and proves the processor matches. Do the training wrap once the eval numbers agree with the current scripts.

## Decision 3: relationship with upstream LeRobot

| Option                    | What it means                                                                                                                                                                                                                | Good                                                         | Bad                                                                                                                                       |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| A. Track upstream         | Add an `upstream` remote and merge `upstream/main` regularly. Merge rather than rebase, because this branch is already pushed                                                                                                | New policies, benchmarks and fixes keep arriving             | The 25 modified or deleted core files conflict on every sync; the first sync covers 332 commits                                           |
| B. Freeze                 | Treat the fork as your own codebase and port upstream pieces by hand when needed                                                                                                                                             | Nothing breaks unexpectedly                                  | The gap grows; every new upstream VLA becomes a manual port                                                                               |
| C. Invert (Trossen model) | Depend on upstream `lerobot` as a package. This repo keeps only plugins (`lerobot_robot_nexarm`, `lerobot_teleoperator_nexarm`, `lerobot_env_nexarm`, `lerobot_policy_*`), `nexarm_service`, `sim`, firmware and experiments | Upgrading LeRobot becomes a version bump; cleanest long term | Biggest move. The core edits to train, record, rollout, eval and datasets must be removed, upstreamed as PRs, or rebuilt as plugins first |

Recommendation: aim for **C**, reached through **A**. Add the remote now. Each time you sync, move one core edit out of `src/lerobot/` into a plugin or an upstream PR, so the conflict surface shrinks instead of growing.

## Experiment organization (applies to every option)

Keep one folder per experiment with its config, launch command and a short README (hypothesis, dataset revision, checkpoint, result). Keep outputs out of git and log runs to a tracker. Report real-robot success over at least 10 episodes and sim success over at least 50, as the LeRobot policy guide asks.

## Unresolved questions

- Are the edits to `lerobot_train.py`, `lerobot_record.py` and `lerobot_rollout.py` NexArm-specific, or generic fixes that could go upstream as PRs? This decides how hard option 3C is.
- Does TurboVLA need its mixed-suite and RLDS training inside this repo, or only NexArm LeRobot datasets?
- Is `bspline-policy` still being developed?

## Sources

- [LeRobot: Adding a Policy (bring your own policies)](https://huggingface.co/docs/lerobot/bring_your_own_policies)
- [DiTFlow LeRobot policy plugin](https://github.com/danielsanjosepro/lerobot_policy_ditflow)
- [Trossen: LeRobot plugin, use in your own project](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot_plugin/use_in_your_project.html)
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- [Trossen openpi tutorial](https://docs.trossenrobotics.com/trossen_arm/v1.9/tutorials/openpi.html)
- [Git submodules vs subtrees vs monorepos](https://slicker.me/git/submodules-vs-subtrees-vs-monorepos.html)
- [Modern Python monorepo with uv workspaces](https://pydevtools.com/blog/fosdem-talk-modern-python-monorepo.md)
- [Git mailing list: maintaining a fork workflows](https://lore.kernel.ime.usp.br/git/87k4uid3zo.fsf@troilus.org/t/)
