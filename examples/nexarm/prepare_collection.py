#!/usr/bin/env python
"""Check USB access and cameras, then open the manual NexArm recorder."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def camera_source(value):
    return int(value) if value.isdecimal() else str(Path(value).expanduser())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-port", default="/dev/ttyUSB0")
    parser.add_argument("--leader-port", default="/dev/ttyUSB1")
    parser.add_argument(
        "--front-cam",
        type=camera_source,
        default="/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_E49C2640-video-index0",
    )
    parser.add_argument(
        "--wrist-cam", type=camera_source, default="/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0"
    )
    parser.add_argument("--repo-id", default="thanhkt/nexarm_stack_bowls")
    parser.add_argument("--root-repo-id", default="thanhkt/nexarm_stack_bowls")
    parser.add_argument("--no-merge", action="store_true", help="Keep the session separate without merging")
    parser.add_argument("--merge-root", type=Path, help="Local merged root directory")
    parser.add_argument(
        "--task", default="Stack the bowls with red on bottom, blue in middle, and black on top."
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--front-fourcc", default="MJPG")
    parser.add_argument("--wrist-fourcc", default="YUYV")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Test USB access and cameras without connecting the arms"
    )
    parser.add_argument(
        "--test-cameras", action="store_true", help="Test cameras only; no serial ports needed"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Use the exact existing --repo-id, including its timestamp"
    )
    parser.add_argument("--root", type=Path, help="Optional dataset directory")
    args = parser.parse_args()
    if min(args.num_episodes, args.fps, args.width, args.height) <= 0:
        parser.error("Episode count, fps, width, and height must be positive")
    return args


def ensure_access(devices):
    """Fix only the selected device nodes, and only when access is missing."""
    paths = [Path(device).resolve(strict=True) for device in devices]
    for path in paths:
        if not path.is_char_device():
            raise ValueError(f"Expected a USB device node: {path}")
    missing = [str(path) for path in paths if not os.access(path, os.R_OK | os.W_OK)]
    if missing:
        print("Setting read/write access for: " + ", ".join(missing), flush=True)
        sudo = shutil.which("sudo")
        chmod = shutil.which("chmod")
        if sudo is None or chmod is None:
            raise RuntimeError("sudo and chmod are required to repair USB permissions")
        subprocess.run([sudo, chmod, "666", *missing], check=True)  # noqa: S603
    if any(not os.access(path, os.R_OK | os.W_OK) for path in paths):
        raise PermissionError("USB permissions still unavailable after chmod")


def test_cameras(configs, output_dir):
    # Use the same camera implementation/settings as recording, with both streams open together.
    import cv2

    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    cameras = {}
    try:
        for name, config in configs.items():
            camera = OpenCVCamera(OpenCVCameraConfig(**{k: v for k, v in config.items() if k != "type"}))
            cameras[name] = camera
            camera.connect()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            for name, camera in cameras.items():
                frame = camera.async_read(timeout_ms=1000)
                if frame.shape != (camera.height, camera.width, 3):
                    raise RuntimeError(f"{name}: unexpected camera frame shape {frame.shape}")
            time.sleep(0.02)
        for name, camera in cameras.items():
            snapshot = output_dir / f"{name}.jpg"
            frame = camera.async_read(timeout_ms=1000)
            if not cv2.imwrite(str(snapshot), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not save {snapshot}")
            print(f"Camera OK: {name} — {camera.width}×{camera.height}, {camera.fps} fps; {snapshot}")
    finally:
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()


def main():
    args = parse_args()
    if not (args.prepare_only or args.test_cameras) and not sys.stdin.isatty():
        raise RuntimeError(
            "Run collect.sh in an interactive terminal so Enter / Right Arrow can control recording."
        )
    if not args.test_cameras:
        if Path(args.follower_port).resolve() == Path(args.leader_port).resolve():
            raise ValueError("Leader and follower must use different serial ports")
        ensure_access([args.follower_port, args.leader_port])
        print(f"USB OK: follower={args.follower_port}, leader={args.leader_port}")
    configs = {
        name: {
            "type": "opencv",
            "index_or_path": source,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "fourcc": fourcc,
        }
        for name, source, fourcc in (
            ("front", args.front_cam, args.front_fourcc),
            ("wrist", args.wrist_cam, args.wrist_fourcc),
        )
    }
    sources = [
        Path(f"/dev/video{source}" if isinstance(source, int) else source).resolve()
        for source in (args.front_cam, args.wrist_cam)
    ]
    if sources[0] == sources[1]:
        raise ValueError("Front and wrist must use different cameras")
    ensure_access(sources)
    output_dir = REPO / "outputs" / "collection" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    test_cameras(configs, output_dir)
    if args.prepare_only or args.test_cameras:
        print("Preflight passed. No arm motion or recording started.")
        return
    log_path = output_dir / "recording.log"
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_record",
        "--robot.type=nexarm_follower",
        f"--robot.port={args.follower_port}",
        f"--robot.cameras={json.dumps(configs)}",
        "--teleop.type=nexarm_leader",
        f"--teleop.port={args.leader_port}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.fps={args.fps}",
        "--dataset.episode_time_s=inf",
        "--dataset.reset_time_s=inf",
        "--dataset.streaming_encoding=true",
        "--dataset.encoder_threads=2",
        "--dataset.push_to_hub=false",
        "--display_data=false",
        "--play_sounds=false",
        "--manual_control=true",
        f"--recording_log={log_path}",
        f"--resume={'true' if args.resume else 'false'}",
    ]
    if args.root:
        command.append(f"--dataset.root={args.root.resolve()}")
    if not args.no_merge:
        command.append(f"--root_repo_id={args.root_repo_id}")
        if args.merge_root:
            command.append(f"--merge_root={args.merge_root.resolve()}")
    print(f"Preparing recorder. Logs: {log_path}", flush=True)
    with log_path.open("a") as log:
        result = subprocess.run(command, cwd=REPO, stderr=log)  # noqa: S603
    if result.returncode:
        print(log_path.read_text()[-6000:], file=sys.stderr)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Collection setup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
