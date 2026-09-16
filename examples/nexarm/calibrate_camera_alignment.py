#!/usr/bin/env python

"""Pre-flight camera alignment and calibration tool for NexArm Sim-to-Real transfer.

Displays side-by-side and alpha-blended overlay of the simulated camera view
from MuJoCo against the live physical camera feed, ensuring exact alignment
of viewpoints, field-of-view, and table plane before policy rollout.

Usage:
    python examples/nexarm/calibrate_camera_alignment.py --cam-index 0 --camera-name front
    python examples/nexarm/calibrate_camera_alignment.py --mock --save-snapshot outputs/calibration/test_align.png
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.nexarm_sim.mujoco_backend import NexArmMujocoBackend, resolve_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NexArm Sim-to-Real Camera Alignment Tool")
    parser.add_argument("--cam-index", type=int, default=0, help="Physical camera OpenCV index (default: 0)")
    parser.add_argument("--camera-name", default="front", choices=["front", "wrist"], help="Sim camera name")
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/scene.xml"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--mock", action="store_true", help="Use a simulated mock feed instead of physical webcam"
    )
    parser.add_argument(
        "--save-snapshot", type=Path, default=None, help="Save snapshot image and exit immediately"
    )
    return parser.parse_args()


def draw_grid_and_crosshairs(img: np.ndarray, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Overlay alignment crosshairs and center lines on frame."""
    out = img.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    # Center crosshair
    cv2.line(out, (cx - 20, cy), (cx + 20, cy), color, 1)
    cv2.line(out, (cx, cy - 20), (cx, cy + 20), color, 1)
    # Thirds guide lines
    cv2.line(out, (w // 3, 0), (w // 3, h), (100, 100, 100), 1)
    cv2.line(out, (2 * w // 3, 0), (2 * w // 3, h), (100, 100, 100), 1)
    cv2.line(out, (0, h // 3), (w, h // 3), (100, 100, 100), 1)
    cv2.line(out, (0, 2 * h // 3), (w, 2 * h // 3), (100, 100, 100), 1)
    return out


def main() -> int:
    args = parse_args()

    # 1. Initialize MuJoCo simulation camera
    model_path = resolve_model_path(args.model)
    backend = NexArmMujocoBackend(
        model_path=model_path,
        fps=args.fps,
        camera_width=args.width,
        camera_height=args.height,
        camera_names=(args.camera_name,),
    )
    backend.reset(settle_steps=10)

    # 2. Initialize real camera feed or mock
    cap = None
    if not args.mock:
        cap = cv2.VideoCapture(args.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        if not cap.isOpened():
            print(
                f"WARNING: Unable to open physical camera at index {args.cam_index}. "
                "Falling back to mock frame. Pass --mock to silence."
            )
            cap = None

    alpha = 0.5
    mode = "overlay"  # 'overlay' or 'side_by_side'

    print("\n" + "=" * 60)
    print(" NEXARM SIM-TO-REAL CAMERA ALIGNMENT")
    print("=" * 60)
    print(f" Camera:     {args.camera_name} ({args.width}x{args.height} @ {args.fps}fps)")
    print(f" Sim Model:  {model_path}")
    print(" Controls:")
    print("   [m] - Toggle mode (Overlay / Side-by-Side)")
    print("   [+] - Increase sim overlay opacity")
    print("   [-] - Decrease sim overlay opacity")
    print("   [s] - Save alignment snapshot")
    print("   [q] - Quit")
    print("=" * 60 + "\n")

    try:
        while True:
            # Render virtual camera
            sim_rgb = backend.render(args.camera_name)
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

            # Get physical camera frame or synthetic mock
            if cap is not None and cap.isOpened():
                ret, real_bgr = cap.read()
                if not ret:
                    real_bgr = sim_bgr.copy()
                elif real_bgr.shape[:2] != (args.height, args.width):
                    real_bgr = cv2.resize(real_bgr, (args.width, args.height))
            else:
                # Mock real frame: dim simulation frame with slight tint for demonstration
                real_bgr = (sim_bgr.astype(np.float32) * 0.85).astype(np.uint8)
                cv2.putText(
                    real_bgr,
                    "MOCK REAL FEED (Connect camera or pass --cam-index)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    2,
                )

            # Draw guides
            sim_guided = draw_grid_and_crosshairs(sim_bgr, color=(0, 255, 255))
            real_guided = draw_grid_and_crosshairs(real_bgr, color=(0, 255, 0))

            if mode == "overlay":
                combined = cv2.addWeighted(sim_guided, alpha, real_guided, 1.0 - alpha, 0)
                cv2.putText(
                    combined,
                    f"OVERLAY (Sim alpha={alpha:.2f}) [m: switch mode, s: save, q: quit]",
                    (20, args.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            else:
                combined = np.hstack([sim_guided, real_guided])
                cv2.putText(
                    combined,
                    "VIRTUAL SIMULATION",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    combined,
                    "PHYSICAL CAMERA",
                    (args.width + 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            if args.save_snapshot:
                args.save_snapshot.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save_snapshot), combined)
                print(f"Snapshot saved to: {args.save_snapshot}")
                return 0

            try:
                cv2.imshow("NexArm Sim2Real Alignment", combined)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                print(
                    "Headless environment detected (no GUI display). Use --save-snapshot to record alignments."
                )
                return 0

            if key == ord("q") or key == 27:
                break
            elif key == ord("m"):
                mode = "side_by_side" if mode == "overlay" else "overlay"
            elif key in (ord("+"), ord("=")):
                alpha = min(1.0, alpha + 0.05)
            elif key in (ord("-"), ord("_")):
                alpha = max(0.0, alpha - 0.05)
            elif key == ord("s"):
                timestamp = int(time.time())
                out_path = Path(f"outputs/calibration/alignment_{timestamp}.png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), combined)
                print(f"Saved snapshot to: {out_path}")

    finally:
        if cap is not None:
            cap.release()
        with contextlib.suppress(Exception):
            cv2.destroyAllWindows()
        backend.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
