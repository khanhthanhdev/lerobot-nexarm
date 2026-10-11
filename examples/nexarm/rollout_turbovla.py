#!/usr/bin/env python3
"""Run real-time inference with a trained TurboVLA policy on NexArm (Simulation or Hardware).

The checkpoint directory's config.json (written by train_turbovla.py) supplies the task type,
camera keys and order, image size, fps and default prompt. Older checkpoints without those fields
fall back to front/wrist[/top] by view count, 224 px images, 30 fps and a task-type heuristic.

Examples:
  # In MuJoCo simulation (stack-bowls prompts follow the sampled bowl order):
  python examples/nexarm/rollout_turbovla.py \
      --robot sim \
      --checkpoint /path/to/steps_20000_ema_pytorch_model.pt

  # Pick-and-place in simulation with an explicit prompt:
  python examples/nexarm/rollout_turbovla.py \
      --robot sim --task-type pick_place \
      --checkpoint /path/to/steps_20000_ema_pytorch_model.pt \
      --task "Pick up the red cube, place it in the green target zone, and release it."

  # On physical NexArm (front, wrist and auto-detected RealSense top camera by default):
  python examples/nexarm/rollout_turbovla.py \
      --robot real \
      --follower-port /dev/ttyUSB1 \
      --front-cam 0 --wrist-cam 1 [--top-cam <RealSense serial> | --no-top-cam] \
      --checkpoint /path/to/steps_20000_ema_pytorch_model.pt \
      --task "Stack the bowls with red on bottom, blue in middle, and black on top."
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Headless Linux hosts normally need EGL; other platforms keep MuJoCo's normal choice.
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from PIL import Image

# Import NexArm robot interfaces
from lerobot.motors.nexarm.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS, JOINT_NAMES
from lerobot.utils.robot_utils import precise_sleep

try:  # imported as a package module (tests) or run as a script
    from examples.nexarm.camera_config import (
        CameraSetupError,
        add_camera_args,
        build_camera_configs,
        camera_names,
        check_policy_cameras,
    )
except ModuleNotFoundError:
    from camera_config import (  # type: ignore[no-redef]
        CameraSetupError,
        add_camera_args,
        build_camera_configs,
        camera_names,
        check_policy_cameras,
    )

DEFAULT_TURBOVLA_DIR = Path(__file__).resolve().parents[2] / "TurboVLA"
# Raw gripper position between GRIPPER_OPEN_POS and GRIPPER_CLOSED_POS that separates open from closed.
DEFAULT_GRIPPER_THRESHOLD = 1800.0
DEFAULT_IMAGE_SIZE = 224
DEFAULT_FPS = 30
TASK_TYPES = ("pick_place", "stack_bowls")
IMAGE_PREFIX = "observation.images."
# Camera order used by checkpoints that predate saved camera keys.
LEGACY_CAMERA_ORDER = ("front", "wrist", "top")
# Prompts used to generate/record the training data for each task.
PICK_PLACE_TASK = "Pick up the red cube, place it in the green target zone, and release it."
REAL_STACK_BOWLS_TASK = "Stack the bowls with red on bottom, blue in middle, and black on top."
DEFAULT_TASKS = {"pick_place": PICK_PLACE_TASK, "stack_bowls": REAL_STACK_BOWLS_TASK}


def gripper_is_closed(values: np.ndarray | float, threshold: float) -> np.ndarray:
    """Whether raw gripper positions are on the closed side of ``threshold``."""
    values = np.asarray(values)
    return values > threshold if GRIPPER_CLOSED_POS > GRIPPER_OPEN_POS else values < threshold


@dataclass(frozen=True)
class TurboVLACheckpointConfig:
    """Training-time contract saved next to a TurboVLA checkpoint (``config.json``)."""

    horizon: int = 16
    num_views: int = 2
    image_size: int = DEFAULT_IMAGE_SIZE
    camera_keys: tuple[str, ...] | None = None
    task_type: str | None = None
    fps: int | None = None
    task: str | None = None

    @property
    def camera_names(self) -> list[str]:
        """Robot/sim camera names in the order the model was trained on."""
        if self.camera_keys:
            return [key.removeprefix(IMAGE_PREFIX) for key in self.camera_keys]
        return list(LEGACY_CAMERA_ORDER[: self.num_views])


def load_checkpoint_config(checkpoint_path: Path | str) -> TurboVLACheckpointConfig:
    """Read ``config.json`` beside the checkpoint; missing fields keep backward-compatible defaults."""
    cfg_path = Path(checkpoint_path).parent / "config.json"
    if not cfg_path.is_file():
        print(f"[WARN] {cfg_path} not found; assuming a 2-view, {DEFAULT_IMAGE_SIZE}px checkpoint.")
        return TurboVLACheckpointConfig()
    saved = json.loads(cfg_path.read_text())
    action, vision = saved.get("action", {}), saved.get("vision", {})
    camera_keys = saved.get("camera_keys")
    config = TurboVLACheckpointConfig(
        horizon=int(action.get("horizon", saved.get("horizon", 16))),
        num_views=int(vision.get("num_views", saved.get("num_views", 2))),
        image_size=int(vision.get("image_size", saved.get("image_size", DEFAULT_IMAGE_SIZE))),
        camera_keys=tuple(camera_keys) if camera_keys else None,
        task_type=saved.get("task_type"),
        fps=int(saved["fps"]) if saved.get("fps") else None,
        task=saved.get("task") or None,
    )
    if config.camera_keys is not None and len(config.camera_keys) != config.num_views:
        raise ValueError(
            f"{cfg_path}: camera_keys {list(config.camera_keys)} do not match num_views={config.num_views}"
        )
    if config.task_type is not None and config.task_type not in TASK_TYPES:
        raise ValueError(f"{cfg_path}: unknown task_type {config.task_type!r}; expected one of {TASK_TYPES}")
    print(
        f"[INFO] Loaded {cfg_path}: horizon={config.horizon}, cameras={config.camera_names}, "
        f"image_size={config.image_size}, task_type={config.task_type}, fps={config.fps}"
    )
    return config


def resolve_task_type(requested: str, checkpoint: TurboVLACheckpointConfig, hints: list[str]) -> str:
    """Explicit --task-type, then the saved task type, then the legacy name/view-count heuristic."""
    if requested != "auto":
        return requested
    if checkpoint.task_type:
        return checkpoint.task_type
    names = [hint.lower() for hint in hints]
    is_bowl = checkpoint.num_views == 3 or any(
        keyword in name for name in names for keyword in ("bowl", "paper_finetuned", "turbovla_ddp")
    )
    task_type = "stack_bowls" if is_bowl else "pick_place"
    print(f"[WARN] Checkpoint has no saved task_type; guessed {task_type!r}. Pass --task-type to override.")
    return task_type


def resolve_prompt(task: str | None, checkpoint: TurboVLACheckpointConfig, task_type: str) -> str:
    """Explicit --task, then the prompt saved with the checkpoint, then the task's training prompt."""
    return task or checkpoint.task or DEFAULT_TASKS[task_type]


def parse_args(argv=None):
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
        default=None,
        help="Language prompt (default: the prompt saved with the checkpoint; in the stack-bowls sim, "
        "the instruction for the sampled bowl order)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Control loop frequency (default: the training dataset fps saved with the checkpoint, "
        f"else {DEFAULT_FPS})",
    )
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
    add_camera_args(parser)

    # Sim args
    parser.add_argument(
        "--task-type",
        choices=["auto", "pick_place", "stack_bowls"],
        default="auto",
        help="Task environment and default prompt (auto: the task_type saved with the checkpoint, "
        "else guessed from checkpoint/model names)",
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
        default=DEFAULT_GRIPPER_THRESHOLD,
        help="Raw gripper position separating open from closed when snapping "
        f"(closed side is towards {GRIPPER_CLOSED_POS}; default: {DEFAULT_GRIPPER_THRESHOLD})",
    )
    return parser.parse_args(argv)


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
        gripper_threshold: float = DEFAULT_GRIPPER_THRESHOLD,
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

        # TurboVLA config for NexArm, matching the training-time contract saved with the checkpoint.
        self.checkpoint_config = load_checkpoint_config(checkpoint_path)
        config = TurboVLAConfig()
        config.text.model_name_or_path = str(bert_path)
        config.text.local_files_only = not allow_hf_download
        config.vision.model_name_or_path = str(dino_path)
        config.vision.local_files_only = not allow_hf_download
        config.vision.num_views = self.checkpoint_config.num_views
        config.vision.image_size = self.checkpoint_config.image_size
        config.action.action_dim = 6
        config.action.state_dim = 6
        config.action.horizon = self.checkpoint_config.horizon

        self.num_views = config.vision.num_views
        self.camera_names = self.checkpoint_config.camera_names

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
        # Same explicit resize as training.
        if hasattr(self.image_processor, "size"):
            size = self.checkpoint_config.image_size
            self.image_processor.size = {"height": size, "width": size}

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
                # Snap to the dataset's extreme closed/open values on each side of the threshold.
                closed, opened = (
                    (amax[5], amin[5]) if GRIPPER_CLOSED_POS > GRIPPER_OPEN_POS else (amin[5], amax[5])
                )
                is_closed = gripper_is_closed(unnorm[..., 5], self.gripper_threshold)
                unnorm[..., 5] = np.where(is_closed, closed, opened)
            return unnorm
        return action

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.stats and "observation.state" in self.stats:
            smin = np.array(self.stats["observation.state"]["min"], dtype=np.float32)
            smax = np.array(self.stats["observation.state"]["max"], dtype=np.float32)
            return 2.0 * (state - smin) / np.maximum(smax - smin, 1e-6) - 1.0
        return state

    def ordered_views(self, images: dict[str, np.ndarray] | list[np.ndarray]) -> list[np.ndarray]:
        """Camera frames in the trained order; missing views are an error, never padded."""
        if isinstance(images, dict):
            missing = [name for name in self.camera_names if name not in images]
            if missing:
                raise ValueError(
                    f"Observation is missing camera(s) {missing}; the checkpoint expects {self.camera_names}."
                )
            return [images[name] for name in self.camera_names]
        img_list = list(images)
        if len(img_list) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} camera views {self.camera_names}, got {len(img_list)}."
            )
        return img_list

    @torch.no_grad()
    def predict_chunk(
        self,
        images: dict[str, np.ndarray] | list[np.ndarray],
        state_vector: np.ndarray,
        task: str | None = None,
    ) -> np.ndarray:
        img_list = self.ordered_views(images)
        pil_imgs = [Image.fromarray(x) if isinstance(x, np.ndarray) else x for x in img_list]
        pv = self.image_processor(images=pil_imgs, return_tensors="pt")["pixel_values"]
        # shape [1, num_views, 3, image_size, image_size]
        samples = {"dinov3": pv.unsqueeze(0).to(self.device)}

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

    task_type = resolve_task_type(
        args.task_type, runner.checkpoint_config, [str(x) for x in (args.checkpoint, args.model, args.task)]
    )

    if task_type == "stack_bowls" and (
        "bowl" not in str(args.model).lower() or args.model.name == "scene.xml"
    ):
        args.model = Path("sim/fusion_export/bowl_stack_scene.xml")

    # The trained camera views, in order; a scene without one of them cannot run this checkpoint.
    cam_names = tuple(runner.camera_names)
    import mujoco

    from lerobot.robots.nexarm_sim.mujoco_backend import resolve_model_path

    resolved_model_path = resolve_model_path(args.model)
    _temp_model = mujoco.MjModel.from_xml_path(str(resolved_model_path))
    available_cams = {
        mujoco.mj_id2name(_temp_model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(_temp_model.ncam)
    }
    missing_cams = [c for c in cam_names if c not in available_cams]
    if missing_cams:
        raise SystemExit(
            f"Cameras {missing_cams} required by the checkpoint are not in {args.model} "
            f"(available: {sorted(available_cams)}). Pass a --model scene with cameras {list(cam_names)}."
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
        # Training covers every bowl order, so the prompt must describe this episode's order.
        if args.task is None or "Stack the bowls" in args.task:
            from lerobot.robots.nexarm_sim.stack_bowls_task import get_task_instruction

            args.task = get_task_instruction(*task.current_order)
    else:
        task = NexArmPickPlaceTask(robot.backend, timeout_s=args.timeout)
        task.reset(seed=args.seed, settle_steps=25)
        args.task = resolve_prompt(args.task, runner.checkpoint_config, task_type)

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
    from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig

    check_policy_cameras(camera_names(args), runner.camera_names, str(args.checkpoint))
    task_type = resolve_task_type(args.task_type, runner.checkpoint_config, [str(args.checkpoint)])
    args.task = resolve_prompt(args.task, runner.checkpoint_config, task_type)
    config = NexArmFollowerConfig(port=args.follower_port, cameras=build_camera_configs(args, args.fps))
    robot = NexArmFollower(config)
    robot.connect()

    print(f"[INFO] Connected to NexArm hardware on {args.follower_port}.")
    print(f"[INFO] Executing TurboVLA policy for task: '{args.task}'")
    try:
        while True:
            obs = robot.get_observation()
            curr_joints = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
            chunk = runner.predict_chunk(obs, state_vector=curr_joints, task=args.task)
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
    args.fps = args.fps or runner.checkpoint_config.fps or DEFAULT_FPS
    if args.robot == "sim":
        run_sim(args, runner)
    else:
        run_real(args, runner)


if __name__ == "__main__":
    try:
        main()
    except CameraSetupError as error:
        raise SystemExit(f"Rollout setup failed: {error}") from error
