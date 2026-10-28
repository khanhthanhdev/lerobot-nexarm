"""Time one inference call of each B-spline policy checkpoint at batch 1, stage by stage.

usage:
  uv run python experiments/261028-plan-bind-d2-latency/bench_inference.py \
      --policy bsp10=<pretrained_model dir> --policy onestep=<dir> --policy planbind=<dir> \
      --device cuda --out bench_cuda.json [--cpu-threads 4] [--amp]

A call is what the policy runs once per update: the processor (device copy and normalization), the
observation encoders on `n_obs_steps` frames, then the head:

- `diffusion` (bsp10): DDIM denoising of the spline parameters, then the CPU spline decode;
- `onestep`: one UNet pass;
- `planbind`: Bind (one UNet pass) on every call, Plan (one more) when the plan expires or a residual
  trigger fires. The latency budget uses the slower plan + bind call.

Observations are synthetic, shaped from the checkpoint's input features: no data-dependent branch in
these networks depends on pixel values, so no simulator is needed and the script also runs on a robot's
own computer. Every stage is wrapped in `torch.cuda.synchronize()` on CUDA. The JSON gives, per policy,
the median and p95 of each stage in ms and `latency_frames`: the p95 of a full call rounded up to whole
frames at `--fps`, the value to pass as `--policy.planbind_latency`. A second loop times `select_action`
the way `lerobot-eval` calls it, frame by frame, so the per-frame compute cost is reported too.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from lerobot_policy_bspline.modeling_bspline import BsplinePolicy
from lerobot_policy_bspline.planbind import PlanBindModel

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE


def percentile(values: list[float], q: float) -> float:
    """The `q` quantile (0-1) of `values`, linearly interpolated."""
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def summary(values_s: list[float]) -> dict[str, float]:
    """Median, p95, mean and max of durations in seconds, in ms."""
    ms = [1000 * v for v in values_s]
    return {
        "p50_ms": percentile(ms, 0.5),
        "p95_ms": percentile(ms, 0.95),
        "mean_ms": statistics.fmean(ms),
        "max_ms": max(ms),
    }


class Timer:
    """Accumulates the duration of named stages, synchronizing CUDA around each one."""

    def __init__(self, device: torch.device):
        """Time stages on `device`."""
        self.device = device
        self.times: dict[str, list[float]] = {}

    def sync(self) -> None:
        """Wait for queued CUDA work, so the clock measures it."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __call__(self, name: str, fn: Callable[[], object]) -> object:
        """Run `fn`, add its duration to stage `name` and return its result."""
        self.sync()
        start = time.perf_counter()
        out = fn()
        self.sync()
        self.times.setdefault(name, []).append(time.perf_counter() - start)
        return out


def synthetic_observation(config: PreTrainedConfig, step: int) -> dict[str, torch.Tensor]:
    """One raw observation as `preprocess_observation` returns it: float images in [0, 1], batch 1."""
    gen = torch.Generator().manual_seed(step)
    obs: dict[str, torch.Tensor] = {}
    for key, feature in config.input_features.items():
        if feature.type is FeatureType.VISUAL:
            obs[key] = torch.rand((1, *feature.shape), generator=gen)
        elif key == OBS_STATE:
            obs[key] = torch.rand((1, *feature.shape), generator=gen) * 2 - 1
    return obs


def with_time(batch: dict[str, torch.Tensor], n_obs_steps: int) -> dict[str, torch.Tensor]:
    """Repeat each observation tensor over `n_obs_steps`, as the policy's observation queue does."""
    return {
        k: v.unsqueeze(1).expand(-1, n_obs_steps, *v.shape[1:]).contiguous()
        for k, v in batch.items()
        if isinstance(v, torch.Tensor)
    }


def bench_policy(name: str, path: Path, args: argparse.Namespace, device: torch.device) -> dict:
    """Load one checkpoint and time its calls; returns the JSON record for it."""
    config = PreTrainedConfig.from_pretrained(path)
    config.device = device.type
    policy = BsplinePolicy.from_pretrained(path, config=config).to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        config, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device.type}}
    )
    model = policy.diffusion
    mode = config.bspline_mode
    timer = Timer(device)
    autocast = (
        torch.autocast(device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16)
        if args.amp
        else contextlib.nullcontext()
    )
    zero = torch.zeros(1, dtype=torch.long, device=device)

    def one_call(record: bool) -> None:
        t = timer if record else (lambda _name, fn: fn())
        obs = synthetic_observation(config, step=0)
        batch = t("preprocess", lambda: preprocessor(obs))
        timed = with_time(batch, config.n_obs_steps)
        global_cond = t("encode", lambda: policy._global_cond(timed))
        if mode == "diffusion":
            params = t("denoise", lambda: model.conditional_sample(1, global_cond=global_cond))
            t("decode", lambda: policy.decode_params(params))
        elif mode == "onestep":
            t("onestep", lambda: model.onestep(global_cond))
        else:
            assert isinstance(model, PlanBindModel)
            plan = t("plan", lambda: model.plan(global_cond))
            t("bind", lambda: model.bind(model.lifted_plan(plan, zero), zero, global_cond))

    with torch.inference_mode(), autocast:
        for _ in range(args.warmup):
            one_call(record=False)
        for _ in range(args.n):
            one_call(record=True)

        # Per-frame cost of select_action as lerobot-eval drives it (runtime or action queue).
        policy.reset()
        frame_times = []
        for step in range(args.frames):
            batch = preprocessor(synthetic_observation(config, step))
            timer.sync()
            start = time.perf_counter()
            policy.select_action(batch)
            timer.sync()
            frame_times.append(time.perf_counter() - start)

    stages = {k: summary(v) for k, v in timer.times.items()}
    head = {"diffusion": ["denoise", "decode"], "onestep": ["onestep"], "planbind": ["plan", "bind"]}[mode]
    call_parts = ["preprocess", "encode", *head]
    call_s = [sum(timer.times[k][i] for k in call_parts) for i in range(args.n)]
    call = summary(call_s)
    record = {
        "path": str(path),
        "mode": mode,
        "num_inference_steps": config.num_inference_steps if mode == "diffusion" else None,
        "stages": stages,
        "call": call,
        "latency_frames": max(0, math.ceil(call["p95_ms"] * args.fps / 1000 - 1e-9)),
        "select_action": summary(frame_times[args.warmup :] or frame_times),
    }
    if mode == "planbind":
        bind_s = [sum(timer.times[k][i] for k in ("preprocess", "encode", "bind")) for i in range(args.n)]
        record["bind_only_call"] = summary(bind_s)
    return record


def main() -> None:
    """Time every `--policy` checkpoint and write the JSON report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", action="append", required=True, help="name=pretrained_model directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpu-threads", type=int, default=0, help="torch threads on CPU; 0 keeps the default"
    )
    parser.add_argument("--amp", action="store_true", help="autocast (fp16 on CUDA, bf16 on CPU)")
    parser.add_argument("--n", type=int, default=200, help="timed calls per policy")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--frames", type=int, default=240, help="select_action frames timed per policy")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    hardware = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else f"cpu x{torch.get_num_threads()}"
    )
    results = {
        "device": args.device,
        "hardware": hardware,
        "amp": args.amp,
        "fps": args.fps,
        "torch": torch.__version__,
        "policies": {},
    }
    for item in args.policy:
        name, _, path = item.partition("=")
        print(f"timing {name} on {hardware} ...", flush=True)
        results["policies"][name] = bench_policy(name, Path(path), args, device)
        rec = results["policies"][name]
        print(
            f"  call p50 {rec['call']['p50_ms']:.1f} ms, p95 {rec['call']['p95_ms']:.1f} ms -> "
            f"{rec['latency_frames']} frames; select_action mean {rec['select_action']['mean_ms']:.1f} ms",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
