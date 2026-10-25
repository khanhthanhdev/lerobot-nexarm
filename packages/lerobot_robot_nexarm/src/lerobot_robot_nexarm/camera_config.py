"""Shared NexArm camera flags and camera-config builders for the real-robot scripts.

Every real-robot script (collection, record, teleoperate, rollout, TurboVLA rollout) accepts the
same camera flags with the same defaults:

  --front-cam / --wrist-cam      OpenCV index (``0``) or device path (``/dev/v4l/by-id/...``)
  --front-fourcc / --wrist-fourcc  Optional 4-character pixel format, e.g. MJPG or YUYV
  --top-cam SERIAL               RealSense serial number (default: auto-detect exactly one)
  --no-top-cam                   Record/run with front and wrist cameras only

The top RealSense camera is on by default. It needs the ``intelrealsense`` extra (pyrealsense2).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_FRONT_CAM = 0
DEFAULT_WRIST_CAM = 1
TOP_CAM_AUTO = "auto"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class CameraSetupError(RuntimeError):
    """A selected camera is unavailable; the message says how to fix or disable it."""


TOP_CAM_HELP = (
    "Pass --no-top-cam to run with the front and wrist cameras only, or install the RealSense "
    "driver with `uv sync --extra intelrealsense` (pyrealsense2)."
)


def camera_source(value: str) -> int | str:
    """Parse an OpenCV camera index (``0``) or a device path (``/dev/v4l/by-id/...``)."""
    return int(value) if value.isdecimal() else str(Path(value).expanduser())


def fourcc(value: str) -> str:
    if len(value) != 4:
        raise argparse.ArgumentTypeError(f"FOURCC must have 4 characters, got {value!r}")
    return value.upper()


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared --front-cam/--wrist-cam/--*-fourcc/--top-cam/--no-top-cam flags."""
    parser.add_argument(
        "--front-cam",
        type=camera_source,
        default=DEFAULT_FRONT_CAM,
        help="Front camera: OpenCV index or device path such as /dev/v4l/by-id/... (default: 0)",
    )
    parser.add_argument(
        "--wrist-cam",
        type=camera_source,
        default=DEFAULT_WRIST_CAM,
        help="Wrist camera: OpenCV index or device path such as /dev/v4l/by-id/... (default: 1)",
    )
    parser.add_argument(
        "--front-fourcc",
        type=fourcc,
        default=None,
        help="Front camera pixel format, e.g. MJPG (default: auto)",
    )
    parser.add_argument(
        "--wrist-fourcc",
        type=fourcc,
        default=None,
        help="Wrist camera pixel format, e.g. YUYV (default: auto)",
    )
    parser.add_argument(
        "--top-cam",
        default=TOP_CAM_AUTO,
        help="RealSense D435i top camera serial number (default: auto-detect exactly one). "
        "Use --no-top-cam to disable.",
    )
    parser.add_argument(
        "--no-top-cam", action="store_true", help="Disable the top camera (front and wrist cameras only)"
    )


def camera_names(args: argparse.Namespace) -> list[str]:
    """Camera names the flags will open, without touching any hardware."""
    return ["front", "wrist"] + ([] if args.no_top_cam else ["top"])


def find_realsense_serial(requested: str = TOP_CAM_AUTO) -> str:
    """Return the requested (or the only connected) RealSense serial, with a clear error otherwise."""
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise CameraSetupError(
            f"The top camera needs a working pyrealsense2 install ({error}). {TOP_CAM_HELP}"
        ) from error

    serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
    if requested != TOP_CAM_AUTO:
        if requested not in serials:
            raise CameraSetupError(
                f"RealSense {requested} not found; connected: {serials or 'none'}. {TOP_CAM_HELP}"
            )
        return requested
    if len(serials) != 1:
        hint = "pass --top-cam <serial> to choose one" if serials else TOP_CAM_HELP
        raise CameraSetupError(f"Expected exactly one RealSense top camera, found {len(serials)}; {hint}")
    return serials[0]


def opencv_camera_dict(
    source: int | str, fps: int, width: int, height: int, pixel_format: str | None
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "type": "opencv",
        "index_or_path": source,
        "width": width,
        "height": height,
        "fps": fps,
    }
    if pixel_format:
        config["fourcc"] = pixel_format
    return config


def realsense_camera_dict(serial: str, fps: int, width: int, height: int) -> dict[str, Any]:
    # Recording and inference store RGB only.
    return {
        "type": "intelrealsense",
        "serial_number_or_name": serial,
        "width": width,
        "height": height,
        "fps": fps,
        "use_depth": False,
    }


def build_camera_dicts(
    args: argparse.Namespace,
    fps: int,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    top_serial: str | None = None,
) -> dict[str, dict[str, Any]]:
    """JSON-serializable camera configs (``--robot.cameras=...``) for the selected cameras.

    The RealSense is detected here unless ``top_serial`` was already resolved.
    """
    cameras = {
        "front": opencv_camera_dict(args.front_cam, fps, width, height, args.front_fourcc),
        "wrist": opencv_camera_dict(args.wrist_cam, fps, width, height, args.wrist_fourcc),
    }
    if not args.no_top_cam:
        serial = top_serial or find_realsense_serial(args.top_cam)
        cameras["top"] = realsense_camera_dict(serial, fps, width, height)
    return cameras


def build_camera_configs(
    args: argparse.Namespace, fps: int, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT
) -> dict[str, Any]:
    """Camera config objects for constructing a ``NexArmFollowerConfig`` directly."""
    from lerobot.cameras.opencv import OpenCVCameraConfig

    configs: dict[str, Any] = {}
    for name, config in build_camera_dicts(args, fps, width, height).items():
        kwargs = {key: value for key, value in config.items() if key != "type"}
        if config["type"] == "opencv":
            configs[name] = OpenCVCameraConfig(**kwargs)
        else:
            from lerobot.cameras.realsense import RealSenseCameraConfig

            configs[name] = RealSenseCameraConfig(**kwargs)
    return configs


def check_policy_cameras(opened: Iterable[str], expected: Iterable[str], policy: str) -> None:
    """Fail before moving the robot when the opened cameras differ from the policy's image inputs."""
    opened, expected = list(opened), list(expected)
    if sorted(opened) == sorted(expected):
        return
    missing = [name for name in expected if name not in opened]
    extra = [name for name in opened if name not in expected]
    hints = []
    if "top" in missing:
        hints.append("remove --no-top-cam")
    if "top" in extra:
        hints.append("add --no-top-cam")
    raise SystemExit(
        f"Camera mismatch: {policy} expects cameras {expected}, but this run opens {opened} "
        f"(missing {missing}, unexpected {extra})." + (f" Fix: {', '.join(hints)}." if hints else "")
    )
