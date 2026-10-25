r"""Convert a checkpoint of the former `examples/nexarm/train_turbovla.py` into a LeRobot policy.

The output folder holds `config.json`, `model.safetensors` and the pre/post-processor files, so
`lerobot-eval --policy.path=<folder>`, `lerobot-rollout` and the desktop app can load it.

Example:
  uv run python -m lerobot_policy_turbovla.convert_checkpoint \
      --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt \
      --output-dir outputs/train/nexarm_turbovla/lerobot_policy

The checkpoint folder must contain `stats_turbovla.json` (or pass `--stats-path`). Its `config.json`
supplies horizon, cameras, image size and training prompt; checkpoints without camera keys get
front/wrist[/top] by view count. Pass the backbones the checkpoint was trained with if they differ
from the defaults (the script's `--dinov3-path/--bert-path`).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .configuration_turbovla import DEFAULT_GRIPPER_THRESHOLD, TurboVLAConfig
from .modeling_turbovla import TurboVLAPolicy
from .processor_turbovla import make_turbovla_pre_post_processors

STATS_FILE_NAMES = ("stats_turbovla.json", "stats_gr00t.json", "stats.json")
# Camera order of checkpoints saved before `camera_keys` was recorded.
LEGACY_CAMERA_ORDER = ("front", "wrist", "top")
# NexArm cameras record 640x480 frames.
DEFAULT_IMAGE_SHAPE = (3, 480, 640)


def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Model weights with `module.`/`model.` wrapper prefixes removed."""
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")  # nosec B614
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
    cleaned = {}
    for key, value in state_dict.items():
        # Internal `.model.` names belong to the backbone, so strip leading wrappers only.
        while key.startswith(("module.", "model.")):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def find_stats(checkpoint_path: Path, stats_path: Path | None) -> Path:
    """The normalization statistics saved next to the checkpoint, or `stats_path`."""
    if stats_path is not None:
        return stats_path
    for name in STATS_FILE_NAMES:
        candidate = checkpoint_path.parent / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No {' / '.join(STATS_FILE_NAMES)} next to {checkpoint_path}; pass --stats-path."
    )


def read_training_contract(checkpoint_path: Path) -> dict[str, Any]:
    """Horizon, camera keys, image size and prompt from the checkpoint's `config.json`."""
    config_path = checkpoint_path.parent / "config.json"
    saved = json.loads(config_path.read_text()) if config_path.is_file() else {}
    action, vision = saved.get("action", {}), saved.get("vision", {})
    num_views = int(vision.get("num_views", saved.get("num_views", 2)))
    camera_keys = saved.get("camera_keys") or [
        f"{OBS_IMAGES}.{name}" for name in LEGACY_CAMERA_ORDER[:num_views]
    ]
    if len(camera_keys) != num_views:
        raise ValueError(f"{config_path}: camera_keys {camera_keys} do not match num_views={num_views}")
    return {
        "horizon": int(action.get("horizon", saved.get("horizon", 16))),
        "action_dim": int(action.get("action_dim", 6)),
        "state_dim": int(action.get("state_dim", 6)),
        "image_size": int(vision.get("image_size", saved.get("image_size", 224))),
        "camera_keys": list(camera_keys),
        "task": saved.get("task") or None,
    }


def convert_checkpoint(
    checkpoint_path: Path | str,
    output_dir: Path | str,
    *,
    stats_path: Path | str | None = None,
    text_model: str = "google-bert/bert-base-uncased",
    vision_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    local_files_only: bool = True,
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    n_action_steps: int = 8,
    binarize_gripper: bool = True,
    gripper_threshold: float = DEFAULT_GRIPPER_THRESHOLD,
    device: str = "cpu",
) -> Path:
    """Write `checkpoint_path` as a LeRobot TurboVLA policy folder at `output_dir` and return it."""
    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    contract = read_training_contract(checkpoint_path)
    stats_file = find_stats(checkpoint_path, Path(stats_path) if stats_path else None)
    raw_stats = json.loads(stats_file.read_text())
    dataset_stats = {
        key: {name: torch.tensor(raw_stats[key][name], dtype=torch.float32) for name in ("min", "max")}
        for key in (OBS_STATE, ACTION)
    }

    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(contract["state_dim"],))}
    for key in contract["camera_keys"]:
        input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=tuple(image_shape))
    config = TurboVLAConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(contract["action_dim"],))},
        device=device,
        chunk_size=contract["horizon"],
        n_action_steps=min(n_action_steps, contract["horizon"]),
        camera_keys=contract["camera_keys"],
        image_size=contract["image_size"],
        text_model_name_or_path=text_model,
        vision_model_name_or_path=vision_model,
        local_files_only=local_files_only,
        default_task=contract["task"],
        binarize_gripper=binarize_gripper,
        gripper_threshold=gripper_threshold,
        push_to_hub=False,
    )

    policy = TurboVLAPolicy(config)
    # Strict: a partial load would silently mix trained and untrained weights.
    policy.model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    preprocessor, postprocessor = make_turbovla_pre_post_processors(config, dataset_stats=dataset_stats)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="train_turbovla.py checkpoint (.pt or .safetensors)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, default=None, help="Default: found next to the checkpoint")
    parser.add_argument("--text-model", default="google-bert/bert-base-uncased", help="BERT repo or folder")
    parser.add_argument(
        "--vision-model",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Vision backbone repo or folder",
    )
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        help="Allow downloading the backbones (default: local Hugging Face cache only, saved in the config)",
    )
    parser.add_argument(
        "--image-shape", type=int, nargs=3, default=list(DEFAULT_IMAGE_SHAPE), metavar=("C", "H", "W")
    )
    parser.add_argument("--n-action-steps", type=int, default=8, help="Actions executed per predicted chunk")
    parser.add_argument("--binarize-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-threshold", type=float, default=DEFAULT_GRIPPER_THRESHOLD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    output_dir = convert_checkpoint(
        args.checkpoint,
        args.output_dir,
        stats_path=args.stats_path,
        text_model=args.text_model,
        vision_model=args.vision_model,
        local_files_only=not args.allow_hf_download,
        image_shape=tuple(args.image_shape),
        n_action_steps=args.n_action_steps,
        binarize_gripper=args.binarize_gripper,
        gripper_threshold=args.gripper_threshold,
        device=args.device,
    )
    print(f"Wrote LeRobot TurboVLA policy to {output_dir}")


if __name__ == "__main__":
    main()
