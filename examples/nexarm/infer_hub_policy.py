#!/usr/bin/env python

# Run a NexArm policy that is already on the Hugging Face Hub: TurboVLA, B-spline Policy or Plan-Bind.
#
# The script downloads only the chosen checkpoint into a plain folder, outputs/models/stack_bowls/<preset>,
# the first time (the TurboVLA checkpoint of the former train_turbovla.py is converted into a LeRobot policy
# there), then runs it from that folder in the MuJoCo stack-bowls env
# (`--target sim`, lerobot-nexarm-eval: success rate and per-stage rates) or on the real follower
# (`--target real`, examples/nexarm/rollout.py). Every flag the script does not know is passed on to
# that command, for example `--eval.n_episodes=50` in sim or `--follower-port COM19 --no-top-cam` on the arm.
#
# Usage:
#   uv run python examples/nexarm/infer_hub_policy.py planbind --download-only   # fetch weights, run later
#   MUJOCO_GL=egl uv run python examples/nexarm/infer_hub_policy.py planbind
#   uv run python examples/nexarm/infer_hub_policy.py bspline --target real --follower-port /dev/ttyUSB1
#   uv run python examples/nexarm/infer_hub_policy.py turbovla --target real \
#       --task "Stack the bowls with red on bottom, blue in middle, and black on top."
#   uv run python examples/nexarm/infer_hub_policy.py planbind --list        # checkpoints in the repo
#   uv run python examples/nexarm/infer_hub_policy.py planbind \
#       --subfolder train_onestep/checkpoints/060000/pretrained_model         # another run of the repo
#
# TurboVLA needs `--extra turbovla` in your `uv sync` line. The presets were trained in simulation; before
# running them on the real arm, see docs/guides/calibration.md.

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

STACK_BOWLS_TASK = "NexArmStackBowls-v0"
POLICY_CONFIG = "pretrained_model/config.json"
MODELS_DIR = Path("outputs/models/stack_bowls")


@dataclass(frozen=True)
class Preset:
    repo_id: str
    # Folder of the LeRobot policy inside the repo ("" = repo root), or the former train_turbovla.py checkpoint.
    subfolder: str = ""
    turbovla_checkpoint: str | None = None
    # Bowl order of a single-order training set; the sim env then gives every episode that order's prompt.
    stack_order: str | None = None


PRESETS = {
    "turbovla": Preset(
        "thanhkt/nexarm_stack_bowls_turbovla", turbovla_checkpoint="final_ema_pytorch_model.pt"
    ),
    "bspline": Preset("thanhkt/bspline-nexarm-stack-bowls-single-order", stack_order="red,blue,black"),
    "planbind": Preset(
        "thanhkt/plan-bind-d1",
        subfolder="train_planbind/checkpoints/060000/pretrained_model",
        stack_order="red,blue,black",
    ),
}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a TurboVLA, B-spline or Plan-Bind policy from the Hugging Face Hub on NexArm",
        epilog="Unknown flags are passed to lerobot-nexarm-eval (sim) or examples/nexarm/rollout.py (real).",
    )
    parser.add_argument("policy", choices=sorted(PRESETS), help="Which preset model to run")
    parser.add_argument(
        "--target", choices=["sim", "real"], default="sim", help="MuJoCo eval or the real arm"
    )
    parser.add_argument("--repo-id", help="Another Hub repo with the same kind of model")
    parser.add_argument(
        "--subfolder", help="Policy folder inside the repo, e.g. a run or step of plan-bind-d1"
    )
    parser.add_argument(
        "--checkpoint-file", help="TurboVLA only: another .pt in the repo, e.g. a step's EMA weights"
    )
    parser.add_argument("--revision", help="Hub branch, tag or commit")
    parser.add_argument(
        "--stack-order", help="Sim only: bottom,middle,top bowl order (default: the preset's)"
    )
    parser.add_argument("--n-episodes", type=int, default=20, help="Sim only: episodes to evaluate")
    parser.add_argument("--output-dir", type=Path, help="Sim only (default: outputs/eval/hub_<policy>)")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Where the policy folders are kept, one per model (default: {MODELS_DIR})",
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Download (and convert) the model, then exit"
    )
    parser.add_argument("--list", action="store_true", help="List the policy folders in the repo and exit")
    return parser.parse_known_args(argv)


def resolve(args: argparse.Namespace) -> Preset:
    preset = PRESETS[args.policy]
    return Preset(
        repo_id=args.repo_id or preset.repo_id,
        subfolder=preset.subfolder if args.subfolder is None else args.subfolder.strip("/"),
        turbovla_checkpoint=args.checkpoint_file or preset.turbovla_checkpoint,
        stack_order=args.stack_order or preset.stack_order,
    )


def list_policies(repo_id: str, revision: str | None) -> list[str]:
    """Folders in the repo that hold a LeRobot policy, or the TurboVLA checkpoints of the former format."""
    files = HfApi().list_repo_files(repo_id, revision=revision)
    folders = sorted(f.removesuffix("/config.json") for f in files if f.endswith(POLICY_CONFIG))
    if "config.json" in files and "model.safetensors" in files:
        folders.insert(0, "(repo root)")
    return folders + sorted(f for f in files if f.endswith("_pytorch_model.pt"))


def model_name(args: argparse.Namespace, preset: Preset) -> str:
    """Folder name under --models-dir: the preset name, or a name for the repo and checkpoint chosen."""
    if args.repo_id is None and args.subfolder is None and args.checkpoint_file is None:
        return args.policy
    source = f"{preset.repo_id}/{preset.subfolder or preset.turbovla_checkpoint or 'root'}"
    return f"{args.policy}--" + re.sub(r"[^\w.-]+", "_", source).strip("_")


def download_policy(preset: Preset, revision: str | None, folder: Path) -> Path:
    """Make `folder` a LeRobot policy folder for the preset, downloading only its files the first time."""
    if (folder / "config.json").is_file():
        return folder
    # Files land in .hub/<name> first; a second run resumes an interrupted download from there.
    staging = folder.parent / ".hub" / folder.name
    if preset.turbovla_checkpoint:
        return convert_turbovla(preset, revision, staging, folder)
    pattern = f"{preset.subfolder}/*" if preset.subfolder else "*"
    snapshot_download(preset.repo_id, revision=revision, allow_patterns=[pattern], local_dir=staging)
    source = staging / preset.subfolder
    if not (source / "config.json").is_file():
        raise SystemExit(
            f"No policy at {preset.repo_id}/{preset.subfolder or '(root)'}. "
            f"Run with --list to see the folders it has."
        )
    shutil.move(source, folder)
    return folder


def convert_turbovla(preset: Preset, revision: str | None, staging: Path, folder: Path) -> Path:
    """Download a former train_turbovla.py checkpoint and convert it into a LeRobot policy folder."""
    checkpoint = preset.turbovla_checkpoint or ""
    try:
        from lerobot_policy_turbovla.convert_checkpoint import convert_checkpoint
    except ImportError as error:
        raise SystemExit(
            f"TurboVLA is not installed ({error}); add --extra turbovla to your uv sync line."
        ) from error
    # The converter reads the horizon, cameras and statistics saved next to the weights.
    snapshot_download(
        preset.repo_id,
        revision=revision,
        allow_patterns=[checkpoint, "config.json", "stats_*.json", "stats.json"],
        local_dir=staging,
    )
    print(f"Converting {preset.repo_id}/{checkpoint} into {folder}...")
    # The DINOv3 and BERT backbones are fetched from the Hub the first time.
    return convert_checkpoint(staging / checkpoint, folder, local_files_only=False)


def command(args: argparse.Namespace, preset: Preset, policy_path: Path, extra: list[str]) -> list[str]:
    if args.target == "real":
        rollout = Path(__file__).with_name("rollout.py")
        return [sys.executable, str(rollout), "--policy-path", str(policy_path), *extra]
    output_dir = args.output_dir or Path("outputs/eval") / f"hub_{args.policy}"
    cmd = [
        sys.executable,
        "-m",
        "lerobot_env_nexarm.stage_eval",  # lerobot-nexarm-eval
        f"--policy.path={policy_path}",
        "--env.type=nexarm",
        f"--env.task={STACK_BOWLS_TASK}",
        f"--eval.n_episodes={args.n_episodes}",
        "--eval.batch_size=1",
        f"--output_dir={output_dir}",
    ]
    if preset.stack_order:
        cmd.append(f"--env.stack_order={preset.stack_order}")
    if not any(arg.startswith("--policy.device") for arg in extra):
        # The Hub configs were saved on a GPU machine (device: cuda).
        import torch

        cmd.append(f"--policy.device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    return cmd + extra


def main() -> None:
    args, extra = parse_args()
    preset = resolve(args)
    if args.list:
        for folder in list_policies(preset.repo_id, args.revision):
            print(folder)
        return
    if (
        args.target == "real"
        and preset.turbovla_checkpoint
        and not any(a.startswith("--task") for a in extra)
    ):
        # In sim the env supplies the prompt; on the arm TurboVLA needs the one it was trained with.
        raise SystemExit('TurboVLA on the real arm needs its training prompt: add --task "<prompt>".')
    policy_path = download_policy(preset, args.revision, args.models_dir / model_name(args, preset))
    if args.download_only:
        print(f"Policy ready: {policy_path}")
        return
    cmd = command(args, preset, policy_path, extra)
    print("Running:", subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
