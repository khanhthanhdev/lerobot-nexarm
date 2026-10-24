# Phase 07 report: B-spline Policy port (2026-10-24)

Branch `chore/upstream-sync`. Nothing was pushed.

## Commits

- `3e79d105d` chore: remove the uninitialized bspline-policy submodule (`.gitmodules`, gitlink, `AGENTS.md` / `CLAUDE.md` submodule note)
- `01dc76f58` feat(policies): add the B-spline Policy plugin as a uv workspace member (package, root `pyproject.toml`, `uv.lock`)
- a third commit with the experiment folder, the phase-07 execution notes and this report (see `git log`)

## Design decision: subclass

The upstream reference is `khanhthanhdev/bspline-policy` at `61ed5f42`, MIT, Copyright (c) 2026 Haoyu Xiong. Its policy classes are about 20-line subclasses of the diffusion UNet and transformer policies. They add one action channel for the knot column and return the whole denoised chunk. The real work sits in the dataset sampler: a per-episode adaptive-knot least-squares fit (`generate_knots` + `make_lsq_spline`), cut into `(chunk_size + 2 * degree, 1 + action_dim)` parameter chunks. Loss, network and scheduler are unchanged, so the port subclasses LeRobot's diffusion policy:

- `BsplineConfig(DiffusionConfig)` is registered as `"bspline"`, with `horizon = 16` parameter rows (10 + 2 \* 3) and `action_delta_indices = range(bspline_window)` (64 raw future frames).
- `BsplinePolicy(DiffusionPolicy)` builds a `BsplineModel(DiffusionModel)` from a config copy whose action feature has `action_dim + 1` channels (the upstream `shape_meta` trick). `forward` fits the normalized raw window with the upstream adaptive-knot algorithm. A plugin cannot change LeRobot's dataset, so the fit cannot run there. `predict_action_chunk` decodes the predicted spline at `t = i * bspline_speed_up` frames.
- `make_bspline_pre_post_processors` uses the default pipeline, so normalization is the same as for diffusion.
- `bspline.py` holds the fit, the packing, `safe_knots` and the decode, with the MIT header.

The full analysis (action representation, decoder, loss, sampling, real-env timing) and the deliberate differences are in `policies/lerobot_policy_bspline/README.md`. The main differences:

- the fit runs per window instead of per episode, with `t = 0` at the current frame;
- `max_error` is in normalized units;
- upstream's asynchronous controller (background inference, `predict_before_end` replanning, time alignment, derivative feed-forward, gripper slowdown, CUDA graphs) is not ported, and only the speed-up is kept;
- the transformer variant and relative knots are not ported.

One fix beyond upstream: knots that padding repeats more than `degree + 1` times are dropped before evaluation. Without that, the spline's right end evaluated to 0. A unit test caught it.

## Packaging

`[tool.uv.workspace]` gets its first member. The root installs the plugin through `[dependency-groups] policies = ["lerobot_policy_bspline"]`, with `[tool.uv] default-groups = ["policies"]` and `[tool.uv.sources] lerobot_policy_bspline = { workspace = true }`. The plugin depends on `lerobot[diffusion]` (workspace source) and `scipy>=1.15`.

I chose a dependency group over `project.dependencies` on purpose: it keeps an unpublished package out of the `lerobot` wheel metadata, and plain `uv sync` still installs it. `uv lock` resolved the cycle (346 packages, `lerobot-policy-bspline v0.1.0` added). `uv sync --locked --extra test --extra dev --inexact` installed it. The metadata name is `lerobot_policy_bspline`.

## Verification (real output)

- Plugin tests: `uv run pytest policies/lerobot_policy_bspline/tests -q` gave **14 passed**. They cover encode then decode within `max_error` (simple and complex windows, episode-end padding, a short window, speed-up), knot normalization, `safe_knots`, plugin discovery, config validation, forward and param shapes with backward, `select_action` on a CPU batch, and save/load with DDIM-identical outputs.
- Plugin discovery: after `register_third_party_plugins()`, `'bspline' in PreTrainedConfig.get_known_choices()` is True. `make_policy_config('bspline')` returns `BsplineConfig`, and `get_policy_class` returns `BsplinePolicy`.
- `lerobot-train --policy.type=bspline` on **`thanhkt/nexarm_stack_bowls`** (a real NexArm dataset; only episodes 143-152 were downloaded, about 80 MB, AV1 decoded with pyav), with `resize_shape=[96,128]`, `crop_ratio=0.9`, batch 8, CPU. It ran 200 steps in 9:44, loss went from 1.059 (step 20) to 0.356 (step 200), and it saved `checkpoints/000200/pretrained_model`.
- Reload: `make_policy(cfg, ds_meta)` with `pretrained_path` gave `BsplinePolicy`, and `select_action` on episode 143 frames returned action shape (1, 6) with 31 queued. The first call took 2.16 s and later calls 5 ms. All values were finite, but far from the data after only 200 steps, which is expected.
- `lerobot-eval`: NexArm sim needs OpenGL, which this machine lacks. Instead, a 200-step bspline checkpoint trained on `lerobot/pusht` ran `lerobot-eval --env.type=pusht --eval.n_episodes=2`: exit 0, pc_success 0.0, avg_sum_reward 1.49. That is expected for 200 steps; the run only checks the wiring.
- `uv run pytest tests -k nexarm -q --ignore=tests/utils/test_process.py` gave **338 passed / 10 failed**, the same as the baseline (MuJoCo/OpenGL env failures).
- `pre-commit run --files <phase files>`: every hook passes except mypy. Mypy fails only on the existing `src/lerobot/utils/keyboard_input.py` termios errors on Windows; the hook does not check the plugin. `mypy policies/lerobot_policy_bspline/src` is clean on its own. To make plugin tests follow the repo convention, the root `pyproject.toml` adds a ruff per-file ignore `"policies/*/tests/**" = ["D"]`.

## Open items

- 50-episode bspline vs diffusion in NexArm sim, the speed-up variant, and 10 real-arm episodes. The commands are in `experiments/261024-bspline-vs-diffusion-stack-bowls/README.md`; results are empty and pending a Linux GPU machine.
- The porting-fidelity check against an upstream number.
- The per-sample CPU spline fit costs a few ms per window. Add caching if it bottlenecks GPU training.
- Upstream's time-aligned asynchronous replanning is not ported. Revisit it if high `bspline_speed_up` stutters on the real arm.

Status: DONE_WITH_CONCERNS
Summary: The B-spline Policy plugin (`--policy.type=bspline`, a subclass of diffusion) is ported, tested, installed by `uv sync`, trains and reloads on a NexArm dataset on CPU, and `lerobot-eval` runs on PushT. The submodule is removed.
Concerns/Blockers: The NexArm sim eval, the bspline vs diffusion comparison and the real-arm rollout are pending GPU/Linux. The plugin is wired through a default dependency group, not `project.dependencies`, to keep the wheel installable. The port fits splines per window, not per episode as upstream does.
