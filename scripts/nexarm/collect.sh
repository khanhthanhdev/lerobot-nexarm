#!/usr/bin/env bash
# Collection rig defaults: leader /dev/ttyUSB0, follower /dev/ttyUSB1, rig cameras by stable
# /dev/v4l/by-id paths, top RealSense auto-detected. Any flag passed here overrides these.
set -euo pipefail
collection_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$collection_repo"
exec uv run --no-sync python examples/nexarm/prepare_collection.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --front-cam /dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_E49C2640-video-index0 \
    --front-fourcc MJPG \
    --wrist-cam /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
    --wrist-fourcc YUYV \
    "$@"
