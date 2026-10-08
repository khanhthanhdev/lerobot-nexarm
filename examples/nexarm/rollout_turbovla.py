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
      --front-cam 0 --wrist-cam 1 [--top-cam <RealSense serial>] \
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

# In headless environments without X11 DISPLAY, default MuJoCo to EGL hardware acceleration
if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

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
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="Log observed joints and predicted targets, including gripper values before snapping",
    )

    # Text & Foundation model args
    parser.add_argument(
        "--bert-path",
        type=str,
        default=None,
        help="Hugging Face repo or local path for BERT text encoder (defaults to auto-detected local cache or google-bert/bert-base-uncased)",
    )
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        help="Allow online downloads from Hugging Face Hub (default: False, use local cache only)",
    )

    # Real hardware args
    parser.add_argument("--follower-port", default="/dev/ttyUSB1")
    parser.add_argument("--front-cam", type=int, default=0)
    parser.add_argument("--wrist-cam", type=int, default=1)
    parser.add_argument(
        "--top-cam", help="RealSense serial number for the top camera (required for 3-view checkpoints)"
    )

    # Sim args
    parser.add_argument(
        "--task-type",
        choices=["auto", "pick_place", "stack_bowls"],
        default="auto",
        help="Simulation task environment (auto-detects from checkpoint/model)",
    )
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/scene.xml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=40.0, help="Episode timeout in seconds")
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Path to save MP4 rollout video (stacks front and wrist views side-by-side)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run simulation headless without GUI display (auto-enabled if DISPLAY is unset)",
    )
    parser.add_argument(
        "--binarize-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Snap gripper action to discrete open/closed positions (avoids partial-pinch regression failures)",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=1800.0,
        help="Raw position threshold above which the gripper snaps to closed (default: 1800.0)",
    )
    return parser.parse_args()


class TurboVLAPolicyRunner:
    def __init__(
        self,
        checkpoint_path: Path | str,
        turbovla_repo: Path | str,
        stats_path: Path | None,
        device: str = "cuda",
        bert_path: str | None = None,
        allow_hf_download: bool = False,
        binarize_gripper: bool = True,
        gripper_threshold: float = 1800.0,
        debug_actions: bool = False,
    ):
        self.device = torch.device(device)
        self.binarize_gripper = binarize_gripper
        self.gripper_threshold = gripper_threshold
        self.debug_actions = debug_actions
        checkpoint_path = Path(checkpoint_path)
        turbovla_repo = Path(turbovla_repo)
        sys.path.insert(0, str(turbovla_repo))
        sys.path.insert(0, str(turbovla_repo / "third_party" / "starvla_runtime"))

        from transformers import AutoImageProcessor
        from turbovla.models import TurboVLAConfig, build_turbovla

        # Resolve BERT path (check local cache snapshots first)
        if bert_path is None:
            bert_snapshots = [
                Path(
                    os.path.expanduser(
                        "~/.cache/huggingface/hub/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
                    )
                ),
                Path(
                    os.path.expanduser(
                        "~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
                    )
                ),
            ]
            for cand in bert_snapshots:
                if cand.exists():
                    bert_path = str(cand)
                    break
            if bert_path is None:
                bert_path = "google-bert/bert-base-uncased"

        # Resolve DINOv3 path
        dino_cache = Path(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/3a0fa61ff39414e2d3be4a1d50c7dfbf459aa8b8"
            )
        )
        dino_path = str(dino_cache) if dino_cache.exists() else "facebook/dinov3-vitb16-pretrain-lvd1689m"

        # Default TurboVLA config for NexArm
        config = TurboVLAConfig()
        config.text.model_name_or_path = str(bert_path)
        config.text.local_files_only = not allow_hf_download
        config.vision.model_name_or_path = str(dino_path)
        config.vision.local_files_only = not allow_hf_download
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
                if "vision" in saved_cfg and "num_views" in saved_cfg["vision"]:
                    config.vision.num_views = int(saved_cfg["vision"]["num_views"])
                elif "num_views" in saved_cfg:
                    config.vision.num_views = int(saved_cfg["num_views"])
                print(
                    f"[INFO] Loaded config overrides from {cfg_path} "
                    f"(horizon={config.action.horizon}, num_views={config.vision.num_views})"
                )
            except Exception as e:
                print(f"[WARN] Failed to parse {cfg_path}: {e}")

        self.num_views = config.vision.num_views

        print(f"[INFO] Building TurboVLA model on {self.device} (BERT: {bert_path})...")
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
            # Internal `.model.` names belong to DINOv3, not checkpoint wrappers.
            while k.startswith(("module.", "model.")):
                k = k.split(".", 1)[1]
            clean_state_dict[k] = v

        # Inference requires the complete trained model; partial loading can silently
        # combine a trained action head with an untrained or original vision encoder.
        self.model.load_state_dict(clean_state_dict, strict=True)
        self.model.eval()

        # Image processor
        self.image_processor = AutoImageProcessor.from_pretrained(
            dino_path, local_files_only=not allow_hf_download
        )

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
            unnorm = 0.5 * (action + 1.0) * (amax - amin) + amin
            if self.debug_actions:
                print(
                    f"[ACTION] Predicted gripper before snapping: "
                    f"min={unnorm[..., 5].min():.1f}, max={unnorm[..., 5].max():.1f}; "
                    f"snapping={self.binarize_gripper}, threshold={self.gripper_threshold:.1f}"
                )
            if self.binarize_gripper and unnorm.shape[-1] >= 6:
                unnorm[..., 5] = np.where(unnorm[..., 5] > self.gripper_threshold, amax[5], amin[5])
            return unnorm
        return action

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.stats and "observation.state" in self.stats:
            smin = np.array(self.stats["observation.state"]["min"], dtype=np.float32)
            smax = np.array(self.stats["observation.state"]["max"], dtype=np.float32)
            return 2.0 * (state - smin) / np.maximum(smax - smin, 1e-6) - 1.0
        return state

    @torch.no_grad()
    def predict_chunk(
        self,
        images: list[np.ndarray] | dict[str, np.ndarray] | np.ndarray,
        wrist_img: np.ndarray | None = None,
        state_vector: np.ndarray | None = None,
        task: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        # Handle backward compatibility: predict_chunk(front_img, wrist_img, state_vector, task)
        if isinstance(images, np.ndarray) and wrist_img is not None:
            if self.num_views == 3 and "top_img" in kwargs:
                img_list = [images, wrist_img, kwargs["top_img"]]
            else:
                img_list = [images, wrist_img]
        elif isinstance(images, dict):
            cam_order = ["front", "wrist", "top"] if self.num_views == 3 else ["front", "wrist"]
            img_list = [images[k] for k in cam_order if k in images]
        elif isinstance(images, list):
            img_list = images
        else:
            img_list = [images]

        if len(img_list) < self.num_views:
            img_list.extend([img_list[-1]] * (self.num_views - len(img_list)))
        elif len(img_list) > self.num_views:
            img_list = img_list[: self.num_views]

        pil_imgs = [Image.fromarray(x) if isinstance(x, np.ndarray) else x for x in img_list]
        pv = self.image_processor(images=pil_imgs, return_tensors="pt")["pixel_values"]
        # shape [1, num_views, 3, 224, 224]
        samples = {"dinov3": pv.unsqueeze(0).to(self.device)}

        if state_vector is None:
            state_vector = kwargs.get("curr_joints", np.zeros(6, dtype=np.float32))
        norm_state = self.normalize_state(state_vector)
        states = torch.as_tensor(norm_state, device=self.device, dtype=torch.float32).unsqueeze(0)
        instructions = [task or ""]

        predicted = self.model(instructions, samples, states)
        pred_actions = predicted.squeeze(0).cpu().numpy()
        chunk = self.unnormalize_action(pred_actions)
        if self.debug_actions:
            print(f"[ACTION] Observed joints: {np.round(state_vector, 1).tolist()}")
            print(f"[ACTION] First target:    {np.round(chunk[0], 1).tolist()}")
            print(f"[ACTION] Last target:     {np.round(chunk[-1], 1).tolist()}")
            print(
                f"[ACTION] Max arm target offset across chunk: "
                f"{np.max(np.abs(chunk[:, :5] - state_vector[:5])):.1f} raw units; "
                f"normalized state range=[{norm_state.min():.3f}, {norm_state.max():.3f}]"
            )
        return chunk


def run_sim(args, runner: TurboVLAPolicyRunner):
    from lerobot.robots.nexarm_sim import (
        NexArmPickPlaceTask,
        NexArmSim,
        NexArmSimConfig,
        NexArmStackBowlsTask,
    )

    task_type = args.task_type
    if task_type == "auto":
        is_bowl = (
            runner.num_views == 3
            or any("bowl" in str(x).lower() for x in (args.checkpoint, args.model, args.task))
            or any(kw in str(args.checkpoint).lower() for kw in ("paper_finetuned", "turbovla_ddp"))
        )
        task_type = "stack_bowls" if is_bowl else "pick_place"

    if task_type == "stack_bowls" and (
        "bowl" not in str(args.model).lower() or args.model.name == "scene.xml"
    ):
        args.model = Path("sim/fusion_export/bowl_stack_scene.xml")

    # Match 640x480 native aspect ratio of the dataset cameras
    desired_cams = ("front", "wrist", "top") if runner.num_views == 3 else ("front", "wrist")

    # Inspect cameras available in the MuJoCo model to avoid hard crash on missing cameras
    import mujoco

    from lerobot.robots.nexarm_sim.mujoco_backend import resolve_model_path

    resolved_model_path = resolve_model_path(args.model)
    _temp_model = mujoco.MjModel.from_xml_path(str(resolved_model_path))
    available_cams = {
        mujoco.mj_id2name(_temp_model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(_temp_model.ncam)
    }

    cam_names = tuple(c for c in desired_cams if c in available_cams)
    if not cam_names:
        cam_names = ("front", "wrist")

    missing_cams = [c for c in desired_cams if c not in available_cams]
    if missing_cams:
        print(
            f"[WARN] Cameras {missing_cams} not found in {args.model}. Available: {sorted(available_cams)}. Falling back to {cam_names}."
        )

    config = NexArmSimConfig(
        id="turbovla_rollout",
        model_path=args.model,
        fps=args.fps,
        camera_names=cam_names,
        camera_width=640,
        camera_height=480,
        settle_steps=0,
    )
    robot = NexArmSim(config)
    robot.connect()

    if task_type == "stack_bowls":
        task = NexArmStackBowlsTask(robot.backend, timeout_s=args.timeout)
        task.reset(seed=args.seed, settle_steps=25)
        if "Pick up the red cube" in args.task or "Stack the bowls" in args.task:
            from lerobot.robots.nexarm_sim.stack_bowls_task import get_task_instruction

            args.task = get_task_instruction(*task.current_order)
    else:
        task = NexArmPickPlaceTask(robot.backend, timeout_s=args.timeout)
        task.reset(seed=args.seed, settle_steps=25)

    print(f"[INFO] Running TurboVLA in simulation ({task_type}) for task: '{args.task}'")

    has_display = bool(os.environ.get("DISPLAY"))
    use_viewer = not args.headless and has_display
    viewer = None
    if use_viewer:
        try:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(robot.backend.model, robot.backend.data)
        except Exception as e:
            print(f"[WARN] Failed to launch GUI viewer ({e}), continuing in headless mode.")
    else:
        print("[INFO] Running in headless mode (no GUI display).")

    recorded_frames: list[np.ndarray] | None = [] if args.save_video else None

    try:
        step_idx = 0
        while step_idx < args.max_steps:
            if viewer is not None and not viewer.is_running():
                break

            obs = robot.get_observation()
            curr_joints = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)

            # Predict chunk
            chunk = runner.predict_chunk(obs, state_vector=curr_joints, task=args.task)

            # Execute open_loop_steps from chunk
            for i in range(min(args.open_loop_steps, len(chunk))):
                t0 = time.perf_counter()
                act_vec = chunk[i]
                act_dict = {f"{name}.pos": float(act_vec[j]) for j, name in enumerate(JOINT_NAMES)}
                robot.send_action(act_dict)
                status = task.observe()
                if viewer is not None:
                    viewer.sync()
                if recorded_frames is not None:
                    step_obs = robot.get_observation()
                    cam_frames = [step_obs[c] for c in cam_names if c in step_obs]
                    recorded_frames.append(np.hstack(cam_frames))
                step_idx += 1
                if step_idx % 25 == 0 or status.terminated:
                    print(f"[INFO] Step {step_idx}/{args.max_steps}")
                if status.terminated:
                    result_str = "SUCCESS" if status.success else f"FAILED ({status.reason})"
                    print(f"[INFO] Episode ended at step {step_idx}: {result_str}")
                    return
                if viewer is not None:
                    precise_sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - t0)))
    finally:
        if viewer is not None:
            viewer.close()
        robot.disconnect()

        if recorded_frames and args.save_video:
            from lerobot.utils.io_utils import write_video

            args.save_video.parent.mkdir(parents=True, exist_ok=True)
            write_video(args.save_video, recorded_frames, args.fps)
            print(f"[INFO] Saved rollout video ({len(recorded_frames)} frames) to {args.save_video}")


def run_real(args, runner: TurboVLAPolicyRunner):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig

    cameras = {
        "front": OpenCVCameraConfig(index_or_path=args.front_cam, width=640, height=480, fps=args.fps),
        "wrist": OpenCVCameraConfig(index_or_path=args.wrist_cam, width=640, height=480, fps=args.fps),
    }
    if runner.num_views == 3:
        if not args.top_cam:
            raise SystemExit("This checkpoint uses 3 views; pass --top-cam <RealSense serial>.")
        from lerobot.cameras.realsense import RealSenseCameraConfig

        cameras["top"] = RealSenseCameraConfig(
            serial_number_or_name=args.top_cam, width=640, height=480, fps=args.fps
        )
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

            extra = {"top_img": obs["top"]} if "top" in cameras else {}
            chunk = runner.predict_chunk(front_img, wrist_img, curr_joints, args.task, **extra)
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
        bert_path=args.bert_path,
        allow_hf_download=args.allow_hf_download,
        binarize_gripper=args.binarize_gripper,
        gripper_threshold=args.gripper_threshold,
        debug_actions=args.debug_actions,
    )
    if args.robot == "sim":
        run_sim(args, runner)
    else:
        run_real(args, runner)


if __name__ == "__main__":
    main()
