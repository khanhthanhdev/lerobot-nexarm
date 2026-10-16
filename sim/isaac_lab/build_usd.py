"""Build the USD assets the Isaac Lab NexArm tasks load, into ``sim/isaac_lab/usd/`` (not tracked).

- ``nexarm.usd``: ``sim/description/urdf/nexarm.urdf`` through Isaac Lab's URDF converter. Fixed joints are kept
  so ``cam_mount`` (wrist camera) and ``link_6_gripper_base`` (gripper frame) stay separate prims, and the mimic
  jaw becomes a normal joint that the environment drives with the same target.
- ``bowl_{red,blue,black}.usd``: rigid bowls with the MJCF contact proxies (see ``nexarm_isaac/bowl_usd.py``).

Run it with Isaac Lab's Python after every Fusion export (the URDF and scene JSON change then):

    ./isaaclab.sh -p sim/isaac_lab/build_usd.py --headless          # Linux
    isaaclab.bat -p sim\\isaac_lab\\build_usd.py --headless         # Windows
    python sim/isaac_lab/build_usd.py --bowls-only                   # bowls only, needs just `pip install usd-core`
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nexarm_isaac.bowl_usd import write_bowl_usd  # noqa: E402
from nexarm_isaac.scene_spec import USD_DIR, SceneSpec  # noqa: E402


def build_bowls(spec: SceneSpec, out_dir: Path) -> list[Path]:
    stl = spec.resolve(spec.bowls["visual"]["mesh"])
    return [
        write_bowl_usd(spec.bowls, color, stl, out_dir / f"bowl_{color}.usd") for color in spec.task["colors"]
    ]


def build_robot(spec: SceneSpec, out_dir: Path) -> Path:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    fields = {f.name for f in dataclasses.fields(UrdfConverterCfg)}
    options = {
        "asset_path": str(spec.resolve(spec.robot["urdf"])),
        "usd_dir": str(out_dir),
        "usd_file_name": "nexarm.usd",
        "force_usd_conversion": True,
        "make_instanceable": False,
        "fix_base": True,
        "merge_fixed_joints": False,
        "convert_mimic_joints_to_normal_joints": True,
        "self_collision": True,
        # Only meshes would be approximated; the URDF collisions are boxes.
        "collider_type": "convex_hull",
    }
    if "joint_drive" in fields:  # Isaac Lab 2.x; the env's actuators override these gains
        drive = UrdfConverterCfg.JointDriveCfg
        options["joint_drive"] = drive(
            gains=drive.PDGainsCfg(stiffness=100.0, damping=1.0), target_type="position"
        )
    unknown = sorted(set(options) - fields)
    for name in unknown:
        print(f"Note: this Isaac Lab's UrdfConverterCfg has no {name!r}; skipped.")
    converter = UrdfConverter(UrdfConverterCfg(**{k: v for k, v in options.items() if k in fields}))
    return Path(converter.usd_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=USD_DIR)
    parser.add_argument("--bowls-only", action="store_true", help="Skip the robot (no Isaac needed).")
    if "--bowls-only" in sys.argv:
        args = parser.parse_args()
        app = None
    else:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        app = AppLauncher(args).app
    spec = SceneSpec.load()
    try:
        for path in build_bowls(spec, args.out_dir):
            print(f"Wrote {path}")
        if app is not None:
            print(f"Wrote {build_robot(spec, args.out_dir)}")
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
