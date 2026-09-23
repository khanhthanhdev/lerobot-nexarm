#!/usr/bin/env python3
"""Verify environment, dependencies, and download pretrained models for TurboVLA on NexArm.

Examples:
  # Check environment dependencies and models without downloading:
  uv run python examples/nexarm/setup_turbovla.py --check-only

  # Download required vision/text backbones and official pretrained TurboVLA weights:
  uv run python examples/nexarm/setup_turbovla.py --download-all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_TURBOVLA_DIR = Path(__file__).resolve().parents[2] / "TurboVLA"


def check_dependencies() -> bool:
    print("[1/3] Checking core Python dependencies...")
    all_ok = True
    deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Hugging Face Transformers"),
        ("timm", "PyTorch Image Models (timm)"),
        ("safetensors", "SafeTensors"),
        ("accelerate", "Hugging Face Accelerate"),
        ("lerobot", "LeRobot"),
    ]

    for module_name, display_name in deps:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {display_name}: version {ver}")
        except ImportError as e:
            print(f"  ✗ {display_name} ({module_name}) missing: {e}")
            all_ok = False

    # Check CUDA
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✓ GPU Available: {device_name} ({vram_gb:.1f} GB VRAM)")
        else:
            print(
                "  ℹ CUDA is not active (running on CPU). CPU training/inference is functional for dry-runs."
            )
    except Exception as e:
        print(f"  ℹ Torch GPU check skipped: {e}")

    # Check TurboVLA package import
    if DEFAULT_TURBOVLA_DIR.exists():
        sys.path.insert(0, str(DEFAULT_TURBOVLA_DIR))
        try:
            from turbovla.models import TurboVLAConfig

            cfg = TurboVLAConfig()
            cfg.vision.num_views = 2
            cfg.action.action_dim = 6
            cfg.action.state_dim = 6
            print(f"  ✓ TurboVLA package found at {DEFAULT_TURBOVLA_DIR}")
        except Exception as e:
            print(f"  ✗ Failed to import TurboVLA from {DEFAULT_TURBOVLA_DIR}: {e}")
            all_ok = False
    else:
        print(f"  ✗ TurboVLA submodule not found at {DEFAULT_TURBOVLA_DIR}")
        all_ok = False

    return all_ok


def check_or_download_models(
    download_backbones: bool = False,
    download_pretrained: bool = False,
    pretrained_dir: Path = Path("pretrained/TurboVLA"),
) -> bool:
    print("\n[2/3] Checking foundation models (DINOv3 and BERT)...")
    all_ok = True

    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
    except ImportError:
        print("  ✗ huggingface_hub or transformers is required to download models.")
        return False

    # 1. BERT
    bert_id = "google-bert/bert-base-uncased"
    try:
        if download_backbones:
            print(f"  Downloading/caching {bert_id}...")
            AutoTokenizer.from_pretrained(bert_id)
            AutoModel.from_pretrained(bert_id)
            print(f"  ✓ {bert_id} ready.")
        else:
            print(f"  ℹ {bert_id} (online HF access available)")
    except Exception as e:
        print(f"  ✗ Failed to load {bert_id}: {e}")
        all_ok = False

    # 2. DINOv3 ViT-B
    dino_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    try:
        if download_backbones:
            print(f"  Downloading/caching {dino_id}...")
            AutoImageProcessor.from_pretrained(dino_id)
            AutoModel.from_pretrained(dino_id)
            print(f"  ✓ {dino_id} ready.")
        else:
            print(f"  ℹ {dino_id} (online HF access available)")
    except Exception as e:
        print(f"  ✗ Failed to load {dino_id}: {e}")
        all_ok = False

    # 3. TurboVLA Pretrained Weights
    print("\n[3/3] Checking official TurboVLA weights...")
    if download_pretrained:
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading H-EmbodVis/TurboVLA checkpoints into {pretrained_dir}...")
        try:
            snapshot_download(
                repo_id="H-EmbodVis/TurboVLA",
                local_dir=str(pretrained_dir),
                allow_patterns=["checkpoints/*", "*.json", "*.yaml"],
            )
            print(f"  ✓ Pretrained weights downloaded successfully to {pretrained_dir}")
        except Exception as e:
            print(f"  ✗ Failed to download TurboVLA weights: {e}")
            all_ok = False
    else:
        existing_weights = (
            list(pretrained_dir.glob("**/*.safetensors"))
            + list(pretrained_dir.glob("**/*.pth"))
            + list(pretrained_dir.glob("**/*.pt"))
        )
        if existing_weights:
            print(f"  ✓ Found {len(existing_weights)} existing checkpoint(s) in {pretrained_dir}:")
            for w in existing_weights[:3]:
                print(f"    - {w.relative_to(pretrained_dir)}")
        else:
            print(f"  ℹ No local checkpoints found in {pretrained_dir}.")
            print("    Run with `--download-pretrained` or `--download-all` to fetch the official weights.")

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Only verify environment, do not download")
    parser.add_argument(
        "--download-backbones", action="store_true", help="Download DINOv3 and BERT models to HF cache"
    )
    parser.add_argument(
        "--download-pretrained", action="store_true", help="Download official TurboVLA release weights"
    )
    parser.add_argument(
        "--download-all", action="store_true", help="Download both backbones and pretrained weights"
    )
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=Path("pretrained/TurboVLA"),
        help="Local directory to store pretrained TurboVLA weights",
    )
    args = parser.parse_args()

    if args.download_all:
        args.download_backbones = True
        args.download_pretrained = True

    print("=== TurboVLA on NexArm: Setup & Health Check ===\n")
    deps_ok = check_dependencies()

    if args.check_only:
        check_or_download_models(
            download_backbones=False, download_pretrained=False, pretrained_dir=args.pretrained_dir
        )
        print("\n" + ("=" * 50))
        if deps_ok:
            print("Status: All essential dependencies are verified! You are ready to train.")
            return 0
        else:
            print(
                "Status: Some dependencies are missing. Run `uv sync --extra smolvla --extra training --extra groot`."
            )
            return 1

    models_ok = check_or_download_models(
        download_backbones=args.download_backbones,
        download_pretrained=args.download_pretrained,
        pretrained_dir=args.pretrained_dir,
    )

    print("\n" + ("=" * 50))
    if deps_ok and models_ok:
        print("Status: Setup complete! You are ready to train NexArm with TurboVLA.")
        print("\nTo train TurboVLA on a dataset:")
        print(
            "  uv run python examples/nexarm/train_turbovla.py --dataset-root outputs/datasets/nexarm_stack_bowls"
        )
        return 0
    else:
        print("Status: Setup encountered warnings or errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
