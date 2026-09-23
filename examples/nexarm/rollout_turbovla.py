#!/usr/bin/env python3
"""Run real-time inference with a trained TurboVLA policy on NexArm (Simulation or Hardware).

Examples:
  # In MuJoCo simulation:
  python examples/nexarm/rollout_turbovla.py \
      --robot sim \
      --checkpoint /path/to/steps_20000_ema_pytorch_model.pt \
      --task "Pick up the red cube and place it into the green zone"

  # On physical NexArm:
  python examples/nexarm/rollout_turbovla.py \
      --robot real \
      --follower-port /dev/ttyUSB1 \
      --front-cam 0 --wrist-cam 1 \
      --checkpoint /path/to/steps_20000_ema_pytorch_model.pt \
      --task "Pick up the red cube and place it into the green zone"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Import NexArm robot interfaces
from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.utils.robot_utils import precise_sleep

DEFAULT_TURBOVLA_DIR = Path(__file__).resolve().parents[2] / "TurboVLA"


def parse_args():
    parser = argparse.ArgumentParser(description="Run TurboVLA policy on NexArm")
    parser.add_argument("--robot", choices=["sim", "real"], default="sim", help="Robot backend")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to TurboVLA model checkpoint (.pt or .safetensors)",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="Path to stats_turbovla.json or stats_gr00t.json for unnormalization (auto-detected if in checkpoint dir)",
    )
    parser.add_argument(
        "--turbovla-repo",
        type=Path,
        default=DEFAULT_TURBOVLA_DIR if DEFAULT_TURBOVLA_DIR.exists() else Path("/home/25thanh.tk/TurboVLA"),
        help="Path to TurboVLA repository",
    )
    parser.add_argument(
        "--task",
        default="Pick up the red cube, place it in the green target zone, and release it.",
        help="Language prompt",
    )
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency")
    parser.add_argument(
        "--open-loop-steps", type=int, default=8, help="Number of steps to execute before re-inferring chunk"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Real hardware args
    parser.add_argument("--follower-port", default="/dev/ttyUSB1")
    parser.add_argument("--front-cam", type=int, default=0)
    parser.add_argument("--wrist-cam", type=int, default=1)

    # Sim args
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/scene.xml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    return parser.parse_args()


class TurboVLAPolicyRunner:
    def __init__(
        self, checkpoint_path: Path, turbovla_repo: Path, stats_path: Path | None, device: str = "cuda"
    ):
        self.device = torch.device(device)
        sys.path.insert(0, str(turbovla_repo))
        sys.path.insert(0, str(turbovla_repo / "third_party" / "starvla_runtime"))

        from transformers import AutoImageProcessor
        from turbovla.models import TurboVLAConfig, build_turbovla

        # Default TurboVLA config for NexArm
        config = TurboVLAConfig()
        config.vision.num_views = 2
        config.vision.image_size = 224
        config.action.action_dim = 6
        config.action.state_dim = 6
        config.action.horizon = 16

        # Check for saved config.json
        cfg_path = checkpoint_path.parent / "config.json"
        if cfg_path.is_file():
            try:
                with open(cfg_path) as f:
                    saved_cfg = json.load(f)
                if "action" in saved_cfg and "horizon" in saved_cfg["action"]:
                    config.action.horizon = int(saved_cfg["action"]["horizon"])
                elif "horizon" in saved_cfg:
                    config.action.horizon = int(saved_cfg["horizon"])
                print(f"[INFO] Loaded config overrides from {cfg_path} (horizon={config.action.horizon})")
            except Exception as e:
                print(f"[WARN] Failed to parse {cfg_path}: {e}")

        print(f"[INFO] Building TurboVLA model on {self.device}...")
        self.model = build_turbovla(config).to(self.device)

        # Load weights
        print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")  # nosec B614
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

        # Clean prefix if needed
        clean_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("model.", "")
            clean_state_dict[k] = v

        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.eval()

        # Image processor
        dino_cache = Path(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/3a0fa61ff39414e2d3be4a1d50c7dfbf459aa8b8"
            )
        )
        dino_path = str(dino_cache) if dino_cache.exists() else "facebook/dinov3-vitb16-pretrain-lvd1689m"
        self.image_processor = AutoImageProcessor.from_pretrained(dino_path)

        # Normalization stats
        self.stats = None
        if stats_path is None:
            # Auto-detect in checkpoint directory
            for candidate_name in ("stats_turbovla.json", "stats_gr00t.json", "stats.json"):
                cand = checkpoint_path.parent / candidate_name
                if cand.is_file():
                    stats_path = cand
                    break

        if stats_path and stats_path.is_file():
            with open(stats_path) as f:
                self.stats = json.load(f)
            print(f"[INFO] Loaded normalization stats from {stats_path}")
        else:
            print("[WARN] No normalization stats loaded! Actions/states will not be normalized.")

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.stats and "action" in self.stats:
            amin = np.array(self.stats["action"]["min"], dtype=np.float32)
            amax = np.array(self.stats["action"]["max"], dtype=np.float32)
            # TurboVLA min_max maps [min, max] -> [-1, 1]
            return 0.5 * (action + 1.0) * (amax - amin) + amin
        return action

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.stats and "observation.state" in self.stats:
            smin = np.array(self.stats["observation.state"]["min"], dtype=np.float32)
            smax = np.array(self.stats["observation.state"]["max"], dtype=np.float32)
            return 2.0 * (state - smin) / np.maximum(smax - smin, 1e-6) - 1.0
        return state

    @torch.no_grad()
    def predict_chunk(
        self, front_img: np.ndarray, wrist_img: np.ndarray, state_vector: np.ndarray, task: str
    ) -> np.ndarray:
        # Preprocess images
        img_front = Image.fromarray(front_img).resize((224, 224))
        img_wrist = Image.fromarray(wrist_img).resize((224, 224))
        pv = self.image_processor(images=[img_front, img_wrist], return_tensors="pt")["pixel_values"]
        # shape [1, 2, 3, 224, 224]
        samples = {"dinov3": pv.unsqueeze(0).to(self.device)}

        norm_state = self.normalize_state(state_vector)
        states = torch.as_tensor(norm_state, device=self.device, dtype=torch.float32).unsqueeze(0)
        instructions = [task]

        predicted = self.model(instructions, samples, states)
        pred_actions = predicted.squeeze(0).cpu().numpy()
        return self.unnormalize_action(pred_actions)


def run_sim(args, runner: TurboVLAPolicyRunner):
    import mujoco.viewer

    from lerobot.robots.nexarm_sim import NexArmPickPlaceTask, NexArmSim, NexArmSimConfig

    config = NexArmSimConfig(
        id="turbovla_rollout",
        model_path=args.model,
        fps=args.fps,
        camera_names=("front", "wrist"),
        camera_width=224,
        camera_height=224,
        settle_steps=0,
    )
    robot = NexArmSim(config)
    robot.connect()
    task = NexArmPickPlaceTask(robot.backend, timeout_s=30.0)
    task.reset(seed=args.seed, settle_steps=25)

    print(f"[INFO] Running TurboVLA in simulation for task: '{args.task}'")
    try:
        with mujoco.viewer.launch_passive(robot.backend.model, robot.backend.data) as viewer:
            step_idx = 0
            while viewer.is_running() and step_idx < args.max_steps:
                obs = robot.get_observation()
                front_img = obs["front"]
                wrist_img = obs["wrist"]
                curr_joints = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)

                # Predict chunk
                chunk = runner.predict_chunk(front_img, wrist_img, curr_joints, args.task)

                # Execute open_loop_steps from chunk
                for i in range(min(args.open_loop_steps, len(chunk))):
                    t0 = time.perf_counter()
                    act_vec = chunk[i]
                    act_dict = {f"{name}.pos": float(act_vec[j]) for j, name in enumerate(JOINT_NAMES)}
                    robot.send_action(act_dict)
                    status = task.step()
                    viewer.sync()
                    step_idx += 1
                    if status.terminated:
                        print(f"[INFO] Episode ended at step {step_idx}: {status.reason}")
                        return
                    precise_sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - t0)))
    finally:
        robot.disconnect()


def run_real(args, runner: TurboVLAPolicyRunner):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig

    cameras = {
        "front": OpenCVCameraConfig(index_or_path=args.front_cam, width=640, height=480, fps=args.fps),
        "wrist": OpenCVCameraConfig(index_or_path=args.wrist_cam, width=640, height=480, fps=args.fps),
    }
    config = NexArmFollowerConfig(port=args.follower_port, cameras=cameras)
    robot = NexArmFollower(config)
    robot.connect()

    print(f"[INFO] Connected to NexArm hardware on {args.follower_port}.")
    print(f"[INFO] Executing TurboVLA policy for task: '{args.task}'")
    try:
        while True:
            obs = robot.get_observation()
            front_img = obs["front"]
            wrist_img = obs["wrist"]
            curr_joints = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)

            chunk = runner.predict_chunk(front_img, wrist_img, curr_joints, args.task)
            for i in range(min(args.open_loop_steps, len(chunk))):
                t0 = time.perf_counter()
                act_vec = chunk[i]
                act_dict = {f"{name}.pos": float(act_vec[j]) for j, name in enumerate(JOINT_NAMES)}
                robot.send_action(act_dict)
                precise_sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("[INFO] Rollout stopped by user.")
    finally:
        robot.disconnect()


def main():
    args = parse_args()
    runner = TurboVLAPolicyRunner(
        checkpoint_path=args.checkpoint,
        turbovla_repo=args.turbovla_repo,
        stats_path=args.stats_path,
        device=args.device,
    )
    if args.robot == "sim":
        run_sim(args, runner)
    else:
        run_real(args, runner)


if __name__ == "__main__":
    main()
