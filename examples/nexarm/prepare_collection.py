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
    parser.add_argument(
        "--top-cam",
        nargs="?",
        const="auto",
        help="Enable the RealSense D435i top camera (optional serial number; auto-detected if omitted)",
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
        "--stress-test",
        type=float,
        nargs="?",
        const=30.0,
        metavar="SECONDS",
        help="After the camera test, run all cameras (D435i RGB+depth too) together for SECONDS (default 30) "
        "and report per-stream fps, drops and stalls",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Use the exact existing --repo-id, including its timestamp"
    )
    parser.add_argument("--root", type=Path, help="Optional dataset directory")
    args = parser.parse_args()
    if min(args.num_episodes, args.fps, args.width, args.height) <= 0:
        parser.error("Episode count, fps, width, and height must be positive")
    if args.stress_test is not None and args.stress_test <= 0:
        parser.error("--stress-test duration must be positive")
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
    # Test OpenCV cameras using recording resolution/fps settings.
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


def find_realsense_serial(requested):
    import pyrealsense2 as rs

    serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
    if requested != "auto":
        if requested not in serials:
            raise ValueError(f"RealSense {requested} not found; connected: {serials or 'none'}")
        return requested
    if len(serials) != 1:
        raise ValueError(f"Expected exactly one RealSense, found {len(serials)}; pass --top-cam <serial>")
    return serials[0]


def test_realsense(serial, args, output_dir):
    """Check the D435i RGB and depth streams and save an RGB snapshot, depth map and point cloud."""
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

    camera = RealSenseCamera(
        RealSenseCameraConfig(
            serial_number_or_name=serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_depth=True,
        )
    )
    camera.connect()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            camera.async_read(timeout_ms=1000)
            time.sleep(0.02)
        rgb = camera.async_read(timeout_ms=1000)
        depth = np.squeeze(camera.async_read_depth(timeout_ms=1000), axis=-1)
        if rgb.shape != (args.height, args.width, 3) or depth.shape != (args.height, args.width):
            raise RuntimeError(f"top: unexpected frame shapes rgb={rgb.shape} depth={depth.shape}")
        valid = (depth > 0) & (depth < np.iinfo(np.uint16).max)
        if not valid.any():
            raise RuntimeError("top: depth stream returned no valid pixels")
        # Depth intrinsics and scale for back-projection (depth is not aligned to color here).
        profile = camera.rs_profile
        intr = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = intr.get_intrinsics()
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
    finally:
        camera.disconnect()

    cv2.imwrite(str(output_dir / "top.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    shown = cv2.convertScaleAbs(np.clip(depth, 0, 4000), alpha=255 / 4000)
    cv2.imwrite(str(output_dir / "top_depth.png"), cv2.applyColorMap(shown, cv2.COLORMAP_JET))
    v, u = np.nonzero(valid)
    z = depth[valid].astype(np.float32) * scale
    points = np.stack([(u - intr.ppx) / intr.fx * z, (v - intr.ppy) / intr.fy * z, z], axis=1)
    ply = output_dir / "top_pointcloud.ply"
    with ply.open("w") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        np.savetxt(f, points, fmt="%.4f")
    print(
        f"Camera OK: top (RealSense {serial}) — RGB+depth {args.width}×{args.height}, {args.fps} fps; "
        f"{len(points)} point-cloud points, depth {z.min():.2f}–{z.max():.2f} m; {ply}"
    )


def stress_test_cameras(configs, serial, args, seconds):
    """Stream every camera at once (RealSense with depth) and check each keeps up with the target fps."""
    import threading

    import numpy as np

    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

    cameras = {
        name: OpenCVCamera(OpenCVCameraConfig(**{k: v for k, v in config.items() if k != "type"}))
        for name, config in configs.items()
        if config["type"] == "opencv"
    }
    if serial:
        cameras["top"] = RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=args.width,
                height=args.height,
                fps=args.fps,
                use_depth=True,
            )
        )
    # stream name -> list of frame arrival times; depth is read as its own stream.
    streams = {name: [] for name in cameras}
    if serial:
        streams["top_depth"] = []
    timeouts = dict.fromkeys(streams, 0)
    stop = threading.Event()

    def worker(stream, read):
        while not stop.is_set():
            try:
                read(timeout_ms=500)
                streams[stream].append(time.monotonic())
            except TimeoutError:
                timeouts[stream] += 1

    def realsense_worker(camera):
        while not stop.is_set():
            try:
                camera.async_read(timeout_ms=500)
                now = time.monotonic()
                streams["top"].append(now)
                try:
                    camera.read_latest_depth(max_age_ms=500)
                    streams["top_depth"].append(now)
                except Exception:
                    timeouts["top_depth"] += 1
            except TimeoutError:
                timeouts["top"] += 1
                timeouts["top_depth"] += 1

    threads = []
    try:
        for camera in cameras.values():
            camera.connect()
        if serial:
            import pyrealsense2 as rs

            usb = next(
                d.get_info(rs.camera_info.usb_type_descriptor)
                for d in rs.context().query_devices()
                if d.get_info(rs.camera_info.serial_number) == serial
            )
            print(
                f"RealSense USB link: {usb}"
                + ("" if usb.startswith("3") else "  <-- NOT USB 3, use a blue port/cable")
            )
        for name, camera in cameras.items():
            if name == "top":
                threads.append(threading.Thread(target=realsense_worker, args=(camera,), daemon=True))
            else:
                threads.append(threading.Thread(target=worker, args=(name, camera.async_read), daemon=True))
        print(f"Stress test: {len(streams)} streams together for {seconds:g}s ...", flush=True)
        start = time.monotonic()
        for thread in threads:
            thread.start()
        time.sleep(seconds)
        stop.set()
        for thread in threads:
            thread.join(timeout=2)
    finally:
        stop.set()
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()

    failed = []
    for stream, times in streams.items():
        times = [t for t in times if t >= start + 1.0]  # skip warm-up
        if len(times) < 2:
            failed.append(f"{stream}: no frames")
            print(f"  {stream:10s} NO FRAMES ({timeouts[stream]} timeouts)")
            continue
        gaps = np.diff(times)
        fps = (len(times) - 1) / (times[-1] - times[0])
        worst = gaps.max() * 1000
        stalls = int((gaps > 2.5 / args.fps).sum())
        print(
            f"  {stream:10s} {fps:5.1f} fps (target {args.fps}), worst gap {worst:5.0f} ms, "
            f"stalls {stalls}, timeouts {timeouts[stream]}"
        )
        if fps < 0.9 * args.fps or timeouts[stream] or stalls > 0.01 * len(times):
            failed.append(f"{stream}: {fps:.1f} fps, {stalls} stalls, {timeouts[stream]} timeouts")
    if failed:
        raise RuntimeError("Stress test failed — " + "; ".join(failed))
    print("Stress test passed: all streams sustained the target fps.")


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
    cameras = [
        ("front", args.front_cam, args.front_fourcc),
        ("wrist", args.wrist_cam, args.wrist_fourcc),
    ]
    configs = {
        name: {
            "type": "opencv",
            "index_or_path": source,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "fourcc": fourcc,
        }
        for name, source, fourcc in cameras
    }
    sources = [
        Path(f"/dev/video{source}" if isinstance(source, int) else source).resolve()
        for _, source, _ in cameras
    ]
    if len(set(sources)) != len(sources):
        raise ValueError("Each camera must use a different device")
    ensure_access(sources)
    output_dir = REPO / "outputs" / "collection" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    test_cameras(configs, output_dir)
    serial = None
    if args.top_cam is not None:
        serial = find_realsense_serial(args.top_cam)
        test_realsense(serial, args, output_dir)
        # Recording stores RGB only; depth and point cloud are checked in the test above.
        configs["top"] = {
            "type": "intelrealsense",
            "serial_number_or_name": serial,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "use_depth": False,
        }
    if args.stress_test:
        stress_test_cameras(configs, serial, args, args.stress_test)
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
