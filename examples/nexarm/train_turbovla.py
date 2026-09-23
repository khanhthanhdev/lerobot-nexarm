#!/usr/bin/env python3
"""Train TurboVLA policy on Hiwonder NexArm datasets (LeRobot format).

Supports both synthetic simulation demonstrations (e.g. 3-bowl stacking) and
real-world physical teleoperation datasets.

Examples:
  # Train on local dataset:
  uv run python examples/nexarm/train_turbovla.py \
      --dataset-root outputs/datasets/nexarm_stack_bowls \
      --output-dir outputs/train/nexarm_turbovla \
      --batch-size 16 \
      --max-steps 50000 \
      --save-steps 5000

  # Fine-tune from pretrained TurboVLA release:
  uv run python examples/nexarm/train_turbovla.py \
      --dataset-root outputs/datasets/nexarm_stack_bowls \
      --pretrained-checkpoint pretrained/TurboVLA/checkpoints/robotwin/steps_55000_ema_model.safetensors \
      --output-dir outputs/train/nexarm_turbovla_ft \
      --freeze-vision
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

DEFAULT_TURBOVLA_DIR = Path(__file__).resolve().parents[2] / "TurboVLA"


class EMAModel:
    """Exponential Moving Average (EMA) shadow weights tracker for ACT/TurboVLA."""

    def __init__(self, model: nn.Module, decay: float = 0.999, device: torch.device | None = None) -> None:
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.shadow = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.detach().clone().to(self.device).float()

    def update(self, model: nn.Module) -> None:
        if self.decay <= 0.0 or not self.shadow:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.detach().to(self.device).float(), alpha=1.0 - self.decay
                    )

    def state_dict(self, base_model: nn.Module) -> dict[str, torch.Tensor]:
        base_dict = copy.deepcopy(base_model.state_dict())
        for name, shadow_param in self.shadow.items():
            if name in base_dict:
                base_dict[name] = shadow_param.to(device="cpu", dtype=base_dict[name].dtype)
        return base_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train TurboVLA on NexArm LeRobot dataset")
    # Dataset args
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/datasets/nexarm_stack_bowls"),
        help="Path to local LeRobot dataset root",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face repo_id or local dataset identifier",
    )
    parser.add_argument(
        "--front-cam-key",
        type=str,
        default=None,
        help="Key for front camera (auto-detected if None)",
    )
    parser.add_argument(
        "--wrist-cam-key",
        type=str,
        default=None,
        help="Key for wrist camera (auto-detected if None)",
    )
    parser.add_argument("--horizon", type=int, default=16, help="Action chunk prediction horizon")
    parser.add_argument("--image-size", type=int, default=224, help="Input image dimension (224x224)")

    # Model args
    parser.add_argument(
        "--turbovla-repo",
        type=Path,
        default=DEFAULT_TURBOVLA_DIR,
        help="Path to TurboVLA repository",
    )
    parser.add_argument(
        "--dinov3-path",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face repo or local path for DINOv3 vision backbone",
    )
    parser.add_argument(
        "--bert-path",
        type=str,
        default="google-bert/bert-base-uncased",
        help="Hugging Face repo or local path for BERT text encoder",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Path to pretrained TurboVLA or GroundingDINO checkpoint (.pt or .safetensors)",
    )
    parser.add_argument("--freeze-text", action="store_true", default=True, help="Freeze BERT text encoder")
    parser.add_argument("--unfreeze-text", dest="freeze_text", action="store_false")
    parser.add_argument(
        "--freeze-vision", action="store_true", help="Freeze DINOv3 backbone for fast head tuning"
    )
    parser.add_argument(
        "--freeze-interaction", action="store_true", help="Freeze vision-language interaction layers"
    )

    # Training args
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train/nexarm_turbovla"))
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--max-steps", type=int, default=50000, help="Total training optimization steps")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Linear warmup steps")
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate for action head and interaction"
    )
    parser.add_argument("--dinov3-lr", type=float, default=5e-5, help="Learning rate for DINOv3 backbone")
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm clipping")
    parser.add_argument("--save-steps", type=int, default=5000, help="Checkpoint save interval")
    parser.add_argument("--log-steps", type=int, default=20, help="Logging interval")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA shadow decay factor")
    parser.add_argument("--loss-type", choices=["l1", "mse", "smooth_l1"], default="l1")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None, help="Precision mode")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device ('cuda' or 'cpu')",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="nexarm-turbovla")
    return parser.parse_args()


def resolve_camera_keys(
    features: dict[str, Any], front_hint: str | None, wrist_hint: str | None
) -> tuple[str, str]:
    """Identify front and wrist image keys from dataset features."""
    all_keys = list(features.keys())

    def match_key(hint: str | None, candidates: list[str]) -> str | None:
        if hint and hint in all_keys:
            return hint
        for c in candidates:
            for k in all_keys:
                if c.lower() in k.lower():
                    return k
        return None

    front_key = match_key(front_hint, ["observation.images.front", "front", "cam_high", "camera_front"])
    wrist_key = match_key(wrist_hint, ["observation.images.wrist", "wrist", "cam_wrist", "camera_wrist"])

    if not front_key or not wrist_key:
        image_keys = [k for k in all_keys if "image" in k or features[k].get("dtype") in ("image", "video")]
        if len(image_keys) >= 2:
            front_key = front_key or image_keys[0]
            wrist_key = wrist_key or image_keys[1]
        elif len(image_keys) == 1:
            front_key = front_key or image_keys[0]
            wrist_key = wrist_key or image_keys[0]
        else:
            raise ValueError(f"Could not resolve 2 camera keys from dataset features: {all_keys}")

    return front_key, wrist_key


def compute_or_load_normalization_stats(
    dataset, action_key: str = "action", state_key: str = "observation.state"
) -> dict[str, dict[str, list[float]]]:
    """Extract or compute min/max statistics for actions and states."""
    stats = {}
    meta_stats = getattr(dataset.meta, "stats", None) or {}

    # 1. Action statistics
    if action_key in meta_stats and "min" in meta_stats[action_key] and "max" in meta_stats[action_key]:
        stats["action"] = {
            "min": [float(x) for x in meta_stats[action_key]["min"]],
            "max": [float(x) for x in meta_stats[action_key]["max"]],
        }
    else:
        print("[INFO] Computing action min/max statistics from dataset...")
        actions = []
        sample_indices = np.linspace(0, len(dataset) - 1, min(len(dataset), 500), dtype=int)
        for idx in sample_indices:
            raw_act = dataset[idx][action_key]
            if isinstance(raw_act, torch.Tensor):
                raw_act = raw_act.numpy()
            if raw_act.ndim > 1:
                actions.append(raw_act[0])
            else:
                actions.append(raw_act)
        actions = np.stack(actions)
        stats["action"] = {
            "min": np.min(actions, axis=0).tolist(),
            "max": np.max(actions, axis=0).tolist(),
        }

    # 2. State statistics
    resolved_state_key = state_key if state_key in meta_stats else "observation.state"
    if resolved_state_key in meta_stats and "min" in meta_stats[resolved_state_key]:
        stats["observation.state"] = {
            "min": [float(x) for x in meta_stats[resolved_state_key]["min"]],
            "max": [float(x) for x in meta_stats[resolved_state_key]["max"]],
        }
    else:
        print("[INFO] Computing state min/max statistics from dataset...")
        states = []
        sample_indices = np.linspace(0, len(dataset) - 1, min(len(dataset), 500), dtype=int)
        for idx in sample_indices:
            raw_st = dataset[idx].get(state_key, dataset[idx].get("observation.state", None))
            if raw_st is not None:
                if isinstance(raw_st, torch.Tensor):
                    raw_st = raw_st.numpy()
                states.append(raw_st)
        if states:
            states = np.stack(states)
            stats["observation.state"] = {
                "min": np.min(states, axis=0).tolist(),
                "max": np.max(states, axis=0).tolist(),
            }
        else:
            # Fallback identity range [-1, 1]
            stats["observation.state"] = {
                "min": [-1.0] * 6,
                "max": [1.0] * 6,
            }

    # Epsilon safeguard
    for key in ("action", "observation.state"):
        smin = np.array(stats[key]["min"], dtype=np.float32)
        smax = np.array(stats[key]["max"], dtype=np.float32)
        eq_mask = np.abs(smax - smin) < 1e-4
        if np.any(eq_mask):
            smin[eq_mask] -= 0.1
            smax[eq_mask] += 0.1
            stats[key]["min"] = smin.tolist()
            stats[key]["max"] = smax.tolist()

    return stats


def load_compatible_weights(model: nn.Module, checkpoint_path: Path) -> int:
    """Load pretrained weights into TurboVLA, skipping shape-mismatched heads."""
    print(f"[INFO] Loading initialization weights from {checkpoint_path}...")
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        source_state = load_file(checkpoint_path)
    else:
        source_state = torch.load(checkpoint_path, map_location="cpu")  # nosec B614
        if isinstance(source_state, dict):
            for k in ("model", "model_state_dict", "state_dict"):
                if isinstance(source_state.get(k), dict):
                    source_state = source_state[k]
                    break

    target_state = model.state_dict()
    matched_state = {}
    skipped = []

    # Map prefixes if checkpoint comes from starVLA / GroundingDINO
    for k, v in source_state.items():
        clean_k = k.replace("module.", "").replace("model.", "")
        if clean_k in target_state:
            if target_state[clean_k].shape == v.shape:
                matched_state[clean_k] = v
            else:
                skipped.append(
                    f"{clean_k} (shape mismatch: {tuple(target_state[clean_k].shape)} vs {tuple(v.shape)})"
                )
        elif "backbone." in clean_k:
            mapped_k = clean_k.replace("backbone.", "vision_encoder.backbone.")
            if mapped_k in target_state and target_state[mapped_k].shape == v.shape:
                matched_state[mapped_k] = v

    target_state.update(matched_state)
    model.load_state_dict(target_state, strict=False)
    print(f"[INFO] Loaded {len(matched_state)} tensors from {checkpoint_path.name}")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} mismatched tensors (e.g. action head fine-tuned for 6-DOF):")
        for s in skipped[:4]:
            print(f"  - {s}")
    return len(matched_state)


def to_pil(image_val: Any) -> Image.Image:
    if isinstance(image_val, Image.Image):
        return image_val
    if isinstance(image_val, torch.Tensor):
        img_np = image_val.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):  # CHW -> HWC
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.dtype in (np.float32, np.float64):
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
    if isinstance(image_val, np.ndarray):
        if image_val.ndim == 3 and image_val.shape[0] in (1, 3):
            image_val = np.transpose(image_val, (1, 2, 0))
        if image_val.dtype in (np.float32, np.float64):
            image_val = np.clip(image_val * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(image_val)
    raise TypeError(f"Cannot convert {type(image_val)} to PIL Image")


class TurboVLANexArmCollator:
    def __init__(
        self,
        front_key: str,
        wrist_key: str,
        stats: dict[str, dict[str, list[float]]],
        image_processor,
        horizon: int = 16,
        dataset_meta=None,
    ) -> None:
        self.front_key = front_key
        self.wrist_key = wrist_key
        self.stats = stats
        self.image_processor = image_processor
        self.horizon = horizon
        self.dataset_meta = dataset_meta

        self.amin = np.array(stats["action"]["min"], dtype=np.float32)
        self.amax = np.array(stats["action"]["max"], dtype=np.float32)
        self.smin = np.array(stats["observation.state"]["min"], dtype=np.float32)
        self.smax = np.array(stats["observation.state"]["max"], dtype=np.float32)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        instructions = []
        front_imgs = []
        wrist_imgs = []
        states = []
        actions = []

        for item in batch:
            # Language prompt
            lang = item.get("task", None)
            if lang is None and self.dataset_meta is not None:
                task_idx = item.get("task_index", None)
                if task_idx is not None and hasattr(self.dataset_meta, "tasks"):
                    if isinstance(task_idx, torch.Tensor):
                        task_idx = int(task_idx.item())
                    lang = self.dataset_meta.tasks.get(task_idx, "")
            if not lang:
                lang = "Complete the robotic manipulation task."
            instructions.append(str(lang))

            # Images
            f_img = to_pil(item[self.front_key]).resize((224, 224))
            w_img = to_pil(item[self.wrist_key]).resize((224, 224))
            front_imgs.append(f_img)
            wrist_imgs.append(w_img)

            # State
            raw_state = item.get("observation.state", item.get("state", None))
            if isinstance(raw_state, torch.Tensor):
                raw_state = raw_state.detach().cpu().numpy()
            if raw_state.ndim > 1:
                raw_state = raw_state[0]
            norm_state = 2.0 * (raw_state - self.smin) / np.maximum(self.smax - self.smin, 1e-6) - 1.0
            states.append(norm_state.astype(np.float32))

            # Action chunk
            raw_action = item["action"]
            if isinstance(raw_action, torch.Tensor):
                raw_action = raw_action.detach().cpu().numpy()
            # If shape is [6] instead of [H, 6], repeat to horizon
            if raw_action.ndim == 1:
                raw_action = np.tile(raw_action, (self.horizon, 1))
            elif raw_action.shape[0] < self.horizon:
                pad_len = self.horizon - raw_action.shape[0]
                last_row = raw_action[-1:]
                raw_action = np.concatenate([raw_action, np.tile(last_row, (pad_len, 1))], axis=0)
            elif raw_action.shape[0] > self.horizon:
                raw_action = raw_action[: self.horizon]

            norm_action = 2.0 * (raw_action - self.amin) / np.maximum(self.amax - self.amin, 1e-6) - 1.0
            actions.append(norm_action.astype(np.float32))

        # Vision preprocessing
        # Stack views: [B, 2] -> flat list of 2*B PIL images
        flat_views = []
        for i in range(len(batch)):
            flat_views.extend([front_imgs[i], wrist_imgs[i]])

        pv = self.image_processor(images=flat_views, return_tensors="pt")["pixel_values"]
        # Reshape to [B, 2, 3, 224, 224]
        pv = pv.view(len(batch), 2, *pv.shape[1:])

        return {
            "instructions": instructions,
            "samples": {"dinov3": pv},
            "states": torch.as_tensor(np.stack(states), dtype=torch.float32),
            "actions": torch.as_tensor(np.stack(actions), dtype=torch.float32),
        }


def save_checkpoint(
    model: nn.Module,
    ema: EMAModel | None,
    config: Any,
    stats: dict,
    output_dir: Path,
    step: int,
    is_final: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    step_str = "final" if is_final else f"steps_{step}"

    # 1. Save standard model
    model_path = output_dir / f"{step_str}_model.pt"
    torch.save(model.state_dict(), model_path)

    # 2. Save EMA model if available
    if ema is not None:
        ema_state = ema.state_dict(model)
        ema_pt_path = output_dir / f"{step_str}_ema_pytorch_model.pt"
        torch.save(ema_state, ema_pt_path)
        try:
            from safetensors.torch import save_file

            ema_safe_path = output_dir / f"{step_str}_ema_model.safetensors"
            save_file(ema_state, ema_safe_path)
        except Exception as e:
            print(f"[WARN] Failed to write safetensors checkpoint: {e}")

    # 3. Save stats_turbovla.json
    stats_path = output_dir / "stats_turbovla.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # 4. Save config.json
    cfg_path = output_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(
            {
                "model_name": "TurboVLA",
                "action": {
                    "action_dim": config.action.action_dim,
                    "state_dim": config.action.state_dim,
                    "horizon": config.action.horizon,
                },
                "vision": {
                    "num_views": config.vision.num_views,
                    "image_size": config.vision.image_size,
                },
            },
            f,
            indent=2,
        )

    print(f"[CHECKPOINT] Saved checkpoint at step {step} -> {output_dir / step_str}*")


def main():
    args = parse_args()

    # Detect distributed training (e.g. launched via torchrun)
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.distributed.init_process_group("nccl")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device(args.device)

    is_main_process = rank == 0

    if is_main_process:
        print(f"=== Training TurboVLA on Hiwonder NexArm (device={device}, world_size={world_size}) ===")

    # Add TurboVLA submodule to path
    if args.turbovla_repo.exists():
        sys.path.insert(0, str(args.turbovla_repo))
        sys.path.insert(0, str(args.turbovla_repo / "third_party" / "starvla_runtime"))
    else:
        raise FileNotFoundError(f"TurboVLA repository not found at {args.turbovla_repo}")

    from transformers import AutoImageProcessor
    from turbovla.models import TurboVLAConfig, build_turbovla

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    # 1. Dataset loading
    repo_id = args.repo_id or "local/nexarm_dataset"
    if is_main_process:
        print(f"\n[1/5] Loading LeRobot dataset from {args.dataset_root or repo_id}...")
    meta = LeRobotDatasetMetadata(repo_id, args.dataset_root)
    delta_timestamps = {"action": [i / max(1, meta.fps) for i in range(args.horizon)]}
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
    )
    if is_main_process:
        print(
            f"  Dataset contains {len(dataset)} frames across {dataset.num_episodes} episodes (fps={meta.fps})."
        )

    # 2. Camera resolution & Normalization statistics
    front_key, wrist_key = resolve_camera_keys(dataset.features, args.front_cam_key, args.wrist_cam_key)
    if is_main_process:
        print(f"  Resolved camera views -> front: '{front_key}', wrist: '{wrist_key}'")

    stats = compute_or_load_normalization_stats(dataset)
    if is_main_process:
        print("  Normalization min/max statistics initialized:")
        print(f"    Action min: {np.round(stats['action']['min'], 3)}")
        print(f"    Action max: {np.round(stats['action']['max'], 3)}")

    # 3. Vision Processor & Model configuration
    if is_main_process:
        print("\n[2/5] Initializing vision processor & TurboVLA model...")
    image_processor = AutoImageProcessor.from_pretrained(args.dinov3_path)
    if hasattr(image_processor, "size"):
        image_processor.size = {"height": args.image_size, "width": args.image_size}

    config = TurboVLAConfig()
    config.vision.num_views = 2
    config.vision.image_size = args.image_size
    config.vision.model_name_or_path = args.dinov3_path
    config.text.model_name_or_path = args.bert_path
    config.action.action_dim = 6
    config.action.state_dim = 6
    config.action.horizon = args.horizon

    # Precision
    if args.precision is None:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            args.precision = "bf16"
        elif device.type == "cuda":
            args.precision = "fp16"
        else:
            args.precision = "fp32"

    if args.precision == "bf16":
        config.vision.compute_precision = "bf16_autocast"
        config.interaction.compute_precision = "bf16_autocast"
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16":
        config.vision.compute_precision = "fp32"
        config.interaction.compute_precision = "fp32"
        amp_dtype = torch.float16
    else:
        config.vision.compute_precision = "fp32"
        config.interaction.compute_precision = "fp32"
        amp_dtype = None

    if is_main_process:
        print(f"  Building TurboVLA architecture (precision={args.precision})...")
    model = build_turbovla(config)

    # Pretrained checkpoint loading
    if args.pretrained_checkpoint and args.pretrained_checkpoint.is_file():
        load_compatible_weights(model, args.pretrained_checkpoint)

    # Freezing configuration
    if args.freeze_text:
        if is_main_process:
            print("  [INFO] Freezing BERT text encoder.")
        for p in model.text_encoder.parameters():
            p.requires_grad = False
        model.text_encoder.eval()

    if args.freeze_vision:
        if is_main_process:
            print("  [INFO] Freezing DINOv3 vision backbone.")
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        model.vision_encoder.eval()

    if args.freeze_interaction:
        if is_main_process:
            print("  [INFO] Freezing Vision-Language interaction layers.")
        for p in model.vision_language_interaction.parameters():
            p.requires_grad = False

    model.to(device)

    # EMA Tracker (tracks un-wrapped base model)
    ema = (
        EMAModel(model, decay=args.ema_decay, device=device)
        if (is_main_process and args.ema_decay > 0.0)
        else None
    )

    # DDP wrapper
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # 4. DataLoader setup
    if is_main_process:
        print("\n[3/5] Setting up DataLoader and Optimizer...")
    collator = TurboVLANexArmCollator(
        front_key=front_key,
        wrist_key=wrist_key,
        stats=stats,
        image_processor=image_processor,
        horizon=args.horizon,
        dataset_meta=dataset.meta,
    )

    sampler = (
        torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        if is_distributed
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Optimizer with parameter groups
    raw_model = model.module if is_distributed else model
    param_groups = [
        {
            "params": [p for n, p in raw_model.action_head.named_parameters() if p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in raw_model.named_parameters()
                if "action_head" not in n and "vision_encoder" not in n and p.requires_grad
            ],
            "lr": args.lr,
        },
    ]
    if not args.freeze_vision:
        param_groups.append(
            {
                "params": [p for n, p in raw_model.vision_encoder.named_parameters() if p.requires_grad],
                "lr": args.dinov3_lr,
            }
        )

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    # Warmup + Cosine Annealing LR Schedule
    warmup_steps = min(args.warmup_steps, max(1, args.max_steps // 10))
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda s: float(s + 1) / float(max(1, warmup_steps)))
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, args.max_steps - warmup_steps), eta_min=args.lr * 0.05
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Weights & Biases
    if args.wandb and is_main_process:
        try:
            import wandb

            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            print("[WARN] WandB not installed; continuing without wandb logging.")
            args.wandb = False

    # 5. Training Loop
    if is_main_process:
        print(f"\n[4/5] Starting training loop for {args.max_steps} steps...")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pbar = tqdm(total=args.max_steps, desc="TurboVLA Training")
    else:
        pbar = None
    data_iter = iter(dataloader)

    model.train()
    running_loss = 0.0
    start_time = time.time()
    step = 0

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        instructions = batch["instructions"]
        samples = {"dinov3": batch["samples"]["dinov3"].to(device)}
        states = batch["states"].to(device)
        targets = batch["actions"].to(device)

        # Autocast context
        use_amp = amp_dtype is not None and device.type == "cuda"
        with (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else torch.autocast(device_type="cpu", enabled=False)
        ):
            predictions = model(instructions, samples, states)
            if args.loss_type == "mse":
                loss = F.mse_loss(predictions, targets)
            elif args.loss_type == "smooth_l1":
                loss = F.smooth_l1_loss(predictions, targets)
            else:
                loss = F.l1_loss(predictions, targets)

            if args.grad_accum_steps > 1:
                loss = loss / args.grad_accum_steps

        loss.backward()

        if (step + 1) % args.grad_accum_steps == 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if ema is not None and is_main_process:
                ema.update(raw_model)

        step += 1
        if pbar is not None:
            pbar.update(1)
        running_loss += loss.item() * args.grad_accum_steps

        # Logging
        if step % args.log_steps == 0 and is_main_process:
            avg_loss = running_loss / args.log_steps
            cur_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            sps = step / max(elapsed, 1e-4)
            if pbar is not None:
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{cur_lr:.2e}", "sps": f"{sps:.1f}"})
            if args.wandb:
                import wandb

                wandb.log({"train/action_loss": avg_loss, "train/lr": cur_lr, "train/step": step})
            running_loss = 0.0

        # Periodic checkpoint
        if (step % args.save_steps == 0 or step == args.max_steps) and is_main_process:
            save_checkpoint(raw_model, ema, config, stats, args.output_dir, step)

    if pbar is not None:
        pbar.close()

    # 6. Finalization
    if is_main_process:
        print("\n[5/5] Finalizing training and saving final checkpoints...")
        save_checkpoint(raw_model, ema, config, stats, args.output_dir, step, is_final=True)

        print("\n=== Training Complete ===")
        print(f"Checkpoints and normalization statistics saved in: {args.output_dir.resolve()}")
        print("\nTo evaluate this model in simulation:")
        print(
            f"  uv run python examples/nexarm/rollout_turbovla.py \\\n"
            f"      --robot sim \\\n"
            f"      --checkpoint {args.output_dir / 'final_ema_pytorch_model.pt'} \\\n"
            f"      --stats-path {args.output_dir / 'stats_turbovla.json'}"
        )

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
