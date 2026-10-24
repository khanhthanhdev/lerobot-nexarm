# lerobot_policy_bspline

[B-spline Policy](https://arxiv.org/abs/2607.09648) (Han, Xiong et al., 2026) as a LeRobot policy plugin. It registers the policy type `bspline` and builds on LeRobot's diffusion policy.

|                         |                                                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| Paper                   | _B-spline Policy: Accelerating Manipulation Policies via B-spline Action Representations_, arXiv 2607.09648     |
| Upstream repo           | <https://github.com/khanhthanhdev/bspline-policy> (fork of <https://github.com/B-spline-policy/bspline-policy>) |
| Pinned reference commit | `61ed5f42` ("hello bspline policy", 2026-07-22)                                                                 |
| License of ported code  | MIT, Copyright (c) 2026 Haoyu Xiong (see [`LICENSE`](LICENSE))                                                  |

## Install and use

Plain `uv sync` at the repository root installs this package (it is a workspace member in the default `policies` dependency group). LeRobot discovers it by its distribution name, so every `lerobot-*` CLI accepts `--policy.type=bspline`:

```bash
uv run lerobot-train --policy.type=bspline --dataset.repo_id=<repo_id> --policy.push_to_hub=false
uv run lerobot-eval --policy.path=<checkpoint>/pretrained_model --env.type=<env>
```

Options specific to this policy (all others are the `diffusion` options):

| Flag                          | Default | Meaning                                                                                  |
| ----------------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `--policy.bspline_degree`     | 3       | Spline degree (cubic)                                                                    |
| `--policy.bspline_chunk_size` | 10      | Knot intervals in one parameter chunk                                                    |
| `--policy.horizon`            | 16      | Parameter rows; must equal `bspline_chunk_size + 2 * bspline_degree`                     |
| `--policy.bspline_window`     | 64      | Future actions (frames) loaded per training sample; the longest time one chunk can cover |
| `--policy.bspline_max_error`  | 0.01    | Max fitting error in normalized action units (actions are min-max scaled to [-1, 1])     |
| `--policy.bspline_speed_up`   | 1.0     | Spline frames advanced per control step at inference; above 1 executes faster            |
| `--policy.n_action_steps`     | 32      | Max actions queued per prediction (fewer if the predicted spline ends sooner)            |

## What B-spline Policy changes compared with diffusion policy

Upstream is built on the original `diffusion_policy` codebase. Its policy classes (`bspline_policy/policy/diffusion_{unet,transformer}_bspline_image_policy.py`) are small subclasses of the diffusion UNet and transformer hybrid image policies. The changes are:

1. **Action representation.** A training target is no longer `horizon` raw actions. Each demonstration episode is fitted once with a least-squares B-spline whose knots are chosen adaptively (`scipy.interpolate.generate_knots` adds knots until the max abs error is below `max_error`, 0.002 in the configs; `ScipyBSplineCompression`). The fitted spline is cut into fixed-size chunks (`chunk_bspline_trajectory`). A chunk is a dense matrix of shape `(chunk_size + 2 * degree, 1 + action_dim)`: column 0 is the knot vector and the other columns are control points. The last `degree + 1` control-point rows are padding. Knots are in dataset frames and are shifted so that the sample's current frame is `t = 0`. Each timestep uses the first chunk whose first valid knot is at or after it. Because knots are adaptive, a chunk with a fixed number of knots covers a long time span where the motion is simple and a short one where it is complex. Chunks at the episode end are padded by repeating the last knot and control point. An optional relative-knot encoding (first knot plus differences) exists, off by default.
2. **Decoder / network.** The denoising network is unchanged (conditional 1D UNet with FiLM, or the transformer). The only change is one more action channel for the knot column (`shape_meta.action.shape + 1`), and the policy returns the whole denoised chunk instead of slicing `n_action_steps` actions.
3. **Loss.** Unchanged: epsilon-prediction MSE of the diffusion model, now over the whole parameter matrix (knot column included), without a padding mask. Normalization is a per-channel range normalizer fitted on all parameter chunks of the dataset.
4. **Inference-time sampling.** DDIM with 16 (configs) or 10 (rollout docs) denoising steps produces a parameter chunk. The knot column is made non-decreasing (`safer_knots`) and the chunk becomes a `scipy.interpolate.BSpline`.
5. **Real-env timing** (`scripts/policy_local_bspline.py`). The spline is a continuous-time plan. A controller loop (`--control-freq 200`) evaluates it at `t = elapsed_wall_time * speed_up_times * origin_time_scale`, where `origin_time_scale` is the data frequency (10 Hz), so `--speed-up-times 2` executes demonstrations twice as fast. Inference runs in a background thread. A new plan is requested when less than `predict_before_end` seconds of the current plan remain. A new plan is time-aligned to the old one by finding the spline time closest to the action being executed (`minimize_scalar`), with an error threshold and an optional restart at `t = 0`. Optional extras are velocity and acceleration feed-forward from spline derivatives, and a gripper slowdown that ramps the speed-up after a gripper change. CUDA graphs can capture the DDIM loop.

## How this port maps it onto LeRobot

`BsplineConfig(DiffusionConfig)` and `BsplinePolicy(DiffusionPolicy)` subclass LeRobot's diffusion policy. That fits because, as upstream, the change is limited to the action representation, the model's action width and the output handling. The network, observation encoders, noise scheduler and loss are reused unchanged.

- **Action width.** The diffusion model is built from a copy of the config whose action feature has `action_dim + 1` channels, the same trick upstream uses on `shape_meta`. `horizon` is the number of parameter rows (default 16 = 10 + 2 \* 3).
- **Fit in `forward`.** LeRobot datasets give each sample a raw action window (`action_delta_indices = range(bspline_window)`, starting at the current frame), and a plugin cannot change the dataset. So `BsplinePolicy.forward` fits each normalized window with the upstream adaptive-knot algorithm and packs the first chunk exactly as upstream does (`bspline.py`). Copy-padded frames past the episode end are dropped first, so the spline ends where the episode ends. Knot times are scaled from `[0, bspline_window - 1]` to `[-1, 1]`.
- **Normalization.** It is the same as for diffusion (`make_bspline_pre_post_processors` is the default LeRobot pipeline). Actions are min-max scaled before the fit, so control points live in about `[-1, 1]` and `max_error` is in normalized units. Upstream instead fits raw actions and range-normalizes the parameter matrix. `clip_sample_range` defaults to 1.5 because control points can overshoot the data range slightly.
- **Decode in `select_action`.** The predicted knot column is unnormalized and made non-decreasing. Action `i` of the queue is the spline at `t = i * bspline_speed_up` frames, cut where the predicted spline ends, at most `n_action_steps`. Repeated end knots from padding are dropped before evaluation, so the spline's right end evaluates correctly. The queue holds normalized actions, and the standard postprocessor unnormalizes them, so `lerobot-eval` and `lerobot-rollout` work unchanged.

### Deliberate differences from upstream

- **Per-window fit instead of a per-episode fit and slicing.** Knots are chosen for the window, not for the whole episode. The chunk can therefore cover at most `bspline_window` frames, and the first chunk always starts at the current frame. Upstream's chunk may start a few frames after it.
- **Time origin.** Upstream's `t = 0` is the first observation of the `n_obs_steps` window; here it is the current frame, matching LeRobot's convention that the executed actions start at the current step.
- **Not ported.** The asynchronous wall-clock controller: background inference, `predict_before_end` replanning, time alignment of consecutive plans, derivative feed-forward, gripper slowdown and CUDA-graph capture. LeRobot's `select_action` is step-synchronous, so only `bspline_speed_up` is kept and replanning happens when the queue is empty. The transformer variant and relative knots are also not ported.
- **No cache.** The fit runs on CPU for every training sample, a few ms per window. Upstream fits once per dataset and caches the chunks.

## Tests

```bash
uv run pytest policies/lerobot_policy_bspline/tests -q
```

`test_bspline.py` checks that encode then decode reconstructs action chunks within `max_error`, including episode-end padding and speed-up. `test_policy.py` checks plugin discovery, config validation, forward and output shapes, `select_action` on a CPU batch, and the save/load round trip.
