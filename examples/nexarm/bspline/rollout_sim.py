"""Roll out an upstream bspline-policy checkpoint in the NexArmStackBowls-v0 MuJoCo sim.

The checkpoint is a `diffusion_policy` workspace `.ckpt` trained on the HDF5 written by
`lerobot_to_robomimic.py` (task `nexarm_joint_bspline.yaml`). Only the policy is rebuilt from the saved
config; the training workspace is not imported.

Timing follows the training sampler: the two observation frames are [k-1, k], knots are in dataset frames
(30 Hz, the sim's fps) with t = 0 at frame k-1, so the action for the current step k is the spline at t = 1.
Each prediction runs `--n-action-steps` sim steps (fewer if the spline ends sooner), then replans.

Usage:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run --with zarr python examples/nexarm/bspline/rollout_sim.py \
        --ckpt bspline-policy/bspline_policy/data/outputs/<run>/checkpoints/latest.ckpt \
        --episodes 10 --video-dir outputs/eval/bspline_stack_bowls
"""

import argparse
import json
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path

if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import dill
import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.interpolate import BSpline

import lerobot_env_nexarm  # noqa: F401  registers NexArmStackBowls-v0
from lerobot_robot_nexarm.nexarm_sim.stack_bowls_task import STAGES


def load_policy(ckpt: Path, bspline_repo: Path, device: torch.device, num_inference_steps: int | None):
    sys.path[:0] = [str(bspline_repo / "bspline_policy"), str(bspline_repo / "diffusion_policy")]
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with open(ckpt, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    use_ema = bool(cfg.training.use_ema) and "ema_model" in payload["state_dicts"]
    policy.load_state_dict(payload["state_dicts"]["ema_model" if use_ema else "model"])
    if num_inference_steps is not None:
        policy.num_inference_steps = num_inference_steps
    policy.eval().to(device)
    print(f"Loaded {ckpt} ({'ema' if use_ema else 'model'} weights, {policy.num_inference_steps} DDIM steps)")
    return policy, cfg


def safer_knots(knots: np.ndarray) -> np.ndarray:
    knots = np.asarray(knots, dtype=np.float64).copy()
    for i in range(1, len(knots)):
        if knots[i] < knots[i - 1]:
            knots[i] = knots[i - 1] + 1e-6
    return knots


class GripperSlowdown:
    """Run at 1x while the planned gripper command changes, then ramp back to ``speed_up``.

    Upstream's real-env slowdown reacts to a gripper change already under way; ``lookahead`` frames of the
    spline are checked instead, so the arm is already slow when it reaches the grasp. The ramp back to full
    speed over ``ramp_steps`` sim steps is upstream's. The state carries across replans within an episode.
    """

    def __init__(self, speed_up: float, lookahead: float, threshold: float, ramp_steps: int) -> None:
        self.speed_up = speed_up
        self.lookahead = lookahead
        self.threshold = threshold
        self.ramp_steps = ramp_steps
        self.reset()

    def reset(self) -> None:
        self._ramp_left = 0
        self.slow_steps = 0

    def rate(self, spline: BSpline, t: float, t_start: float, t_end: float) -> float:
        window = np.clip(np.linspace(t, t + self.lookahead, 9), t_start, t_end)
        gripper = spline(window)[:, -1]
        if np.ptp(gripper) > self.threshold:
            self._ramp_left = self.ramp_steps
            self.slow_steps += 1
            return 1.0
        if self._ramp_left > 0:
            self._ramp_left -= 1
            return 1.0 + (1.0 - self._ramp_left / self.ramp_steps) * (self.speed_up - 1.0)
        return self.speed_up


def decode_chunk(
    chunk: np.ndarray,
    degree: int,
    n_steps: int,
    speed_up: float,
    slowdown: GripperSlowdown | None = None,
) -> np.ndarray:
    """Evaluate a (horizon, 1 + action_dim) parameter chunk from t = 1, advancing ``speed_up`` frames a step."""
    knots = safer_knots(chunk[:, 0])
    ctrl = chunk[: -(degree + 1), 1:].astype(np.float64)
    spline = BSpline(knots, ctrl, degree, extrapolate=False)
    # The sampler's chunk can start after t = 0; as upstream, times before the first knot hold its value.
    t_start, t_end = knots[degree], knots[len(ctrl)]
    ts, t = [], 1.0
    while len(ts) < n_steps and t <= t_end:
        ts.append(max(t, t_start))
        t += speed_up if slowdown is None else slowdown.rate(spline, t, t_start, t_end)
    if not ts:
        ts = [t_end]
    return spline(np.asarray(ts)).astype(np.float32)


def to_policy_obs(obs: dict, image_size: int) -> dict[str, np.ndarray]:
    front = cv2.resize(obs["pixels"]["front"], (image_size, image_size), interpolation=cv2.INTER_AREA)
    return {"front_image": front, "joint_pos": obs["agent_pos"].astype(np.float32)}


def predict(policy, history: deque, device: torch.device) -> np.ndarray:
    images = np.stack([o["front_image"] for o in history]).astype(np.float32) / 255.0
    batch = {
        "front_image": torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).unsqueeze(0).to(device),
        "joint_pos": torch.from_numpy(np.stack([o["joint_pos"] for o in history])).unsqueeze(0).to(device),
    }
    with torch.no_grad():
        return policy.predict_action(batch)["action"][0].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--bspline-repo", type=Path, default=Path("bspline-policy"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--seed", type=int, default=100000, help="First episode seed; episode i uses seed + i"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-action-steps", type=int, default=None, help="Default: the run's n_action_steps")
    parser.add_argument("--speed-up", type=float, default=1.0, help="Spline frames advanced per sim step")
    parser.add_argument(
        "--gripper-slowdown",
        action="store_true",
        help="With --speed-up > 1, run at 1x while the planned gripper command changes",
    )
    parser.add_argument("--slowdown-lookahead", type=float, default=10.0, help="Spline frames checked ahead")
    parser.add_argument(
        "--slowdown-threshold", type=float, default=30.0, help="Raw gripper change that counts as a grasp"
    )
    parser.add_argument(
        "--slowdown-ramp-steps", type=int, default=7, help="Sim steps to ramp back to full speed"
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--grasp-mode", default="physical", choices=("physical", "contact_assisted"))
    # The dataset generator's release threshold (generation_report.json), not the env's teleop default.
    parser.add_argument("--assist-release-pos", type=float, default=1600.0)
    parser.add_argument("--video-dir", type=Path, default=None, help="Write one front-camera mp4 per episode")
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, cfg = load_policy(args.ckpt, args.bspline_repo, device, args.num_inference_steps)
    n_obs_steps = int(cfg.n_obs_steps)
    degree = int(cfg.policy.bspline_degree)
    image_size = int(cfg.shape_meta.obs.front_image.shape[-1])
    n_action_steps = args.n_action_steps or int(cfg.n_action_steps)
    slowdown = None
    if args.gripper_slowdown:
        slowdown = GripperSlowdown(
            args.speed_up, args.slowdown_lookahead, args.slowdown_threshold, args.slowdown_ramp_steps
        )

    env = gym.make(
        "NexArmStackBowls-v0",
        obs_type="pixels_agent_pos",
        control_mode="raw",
        max_episode_steps=args.max_steps,
        enable_domain_randomization=args.domain_randomization,
        grasp_mode=args.grasp_mode,
        assist_release_pos=args.assist_release_pos,
    )
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, info = env.reset(seed=seed)
        policy.reset()
        if slowdown is not None:
            slowdown.reset()
        history: deque = deque([to_policy_obs(obs, image_size)] * n_obs_steps, maxlen=n_obs_steps)
        frames = [obs["pixels"]["front"]] if args.video_dir is not None else []
        queue: deque = deque()
        step, n_predictions, infer_s = 0, 0, 0.0
        done = False
        while not done:
            if not queue:
                start = time.perf_counter()
                queue.extend(
                    decode_chunk(
                        predict(policy, history, device), degree, n_action_steps, args.speed_up, slowdown
                    )
                )
                infer_s += time.perf_counter() - start
                n_predictions += 1
            obs, _, terminated, truncated, info = env.step(queue.popleft())
            history.append(to_policy_obs(obs, image_size))
            if args.video_dir is not None:
                frames.append(obs["pixels"]["front"])
            step += 1
            done = terminated or truncated or info.get("is_success", False)

        stages = {s: bool(info.get(f"stage/{s}", False)) for s in STAGES}
        result = {
            "seed": seed,
            "success": bool(info.get("is_success", False)),
            "steps": step,
            "stages": stages,
            "slow_steps": slowdown.slow_steps if slowdown is not None else 0,
        }
        results.append(result)
        reached = sum(stages.values())
        print(
            f"episode {ep} seed={seed} success={result['success']} steps={step} stages={reached}/{len(STAGES)} "
            f"failed_stage={info.get('failed_stage', '')!r} infer={1000 * infer_s / max(n_predictions, 1):.0f}ms"
        )
        if args.video_dir is not None:
            path = args.video_dir / f"episode_{ep:03d}_seed_{seed}.mp4"
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
    env.close()

    summary = {
        "success_rate": float(np.mean([r["success"] for r in results])),
        "stage_rates": {s: float(np.mean([r["stages"][s] for r in results])) for s in STAGES},
        "episodes": results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    if args.video_dir is not None:
        (args.video_dir / "results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
