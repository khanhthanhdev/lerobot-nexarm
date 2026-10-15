#!/usr/bin/env python

"""Read the RealSense colour intrinsics and print the matching MuJoCo camera ``fovy``.

MuJoCo's ``fovy`` is the vertical field of view of an ideal pinhole camera with a centred principal point, so
it follows from the stream height and the vertical focal length: ``fovy = 2 * atan(height / 2 / fy)``. Set it
on the ``top`` camera in the scene XML, then check the camera pose with ``calibrate_camera_alignment.py``.

Example (on the robot PC, D435i connected):
    uv run python examples/nexarm/realsense_fovy.py --width 640 --height 480
Offline, from intrinsics you already have:
    uv run python examples/nexarm/realsense_fovy.py --height 480 --fy 615.3
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence


def fovy_deg(height: int, fy: float) -> float:
    """Vertical field of view in degrees of a pinhole camera."""
    if height <= 0 or fy <= 0:
        raise ValueError("height and fy must be positive")
    return math.degrees(2.0 * math.atan(height / 2.0 / fy))


def principal_point_offset_px(width: int, height: int, ppx: float, ppy: float) -> tuple[float, float]:
    """How far the principal point sits from the image centre; MuJoCo assumes it is centred."""
    return ppx - width / 2.0, ppy - height / 2.0


def read_color_intrinsics(width: int, height: int, fps: int, serial: str | None) -> dict[str, float]:
    import pyrealsense2 as rs

    pipeline, config = rs.pipeline(), rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    profile = pipeline.start(config)
    try:
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    finally:
        pipeline.stop()
    return {"fx": intrinsics.fx, "fy": intrinsics.fy, "ppx": intrinsics.ppx, "ppy": intrinsics.ppy}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--serial", default=None, help="Device serial when more than one RealSense is connected"
    )
    parser.add_argument("--fy", type=float, default=None, help="Skip the device and use this focal length")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fy is None:
        intrinsics = read_color_intrinsics(args.width, args.height, args.fps, args.serial)
        dx, dy = principal_point_offset_px(args.width, args.height, intrinsics["ppx"], intrinsics["ppy"])
        print(
            f"colour {args.width}x{args.height}: fx {intrinsics['fx']:.1f} fy {intrinsics['fy']:.1f} "
            f"principal point offset ({dx:+.1f}, {dy:+.1f}) px"
        )
        if max(abs(dx), abs(dy)) > 0.02 * args.width:
            print(
                "The principal point is off-centre; MuJoCo cannot model this, so expect a small image shift."
            )
        fy = intrinsics["fy"]
    else:
        fy = args.fy
    print(
        f'fovy = {fovy_deg(args.height, fy):.2f}  ->  <camera name="top" ... fovy="{fovy_deg(args.height, fy):.1f}" />'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
