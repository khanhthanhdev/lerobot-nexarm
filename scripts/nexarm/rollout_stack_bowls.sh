#!/usr/bin/env bash
# Roll out a published stack-bowls policy on the collection rig's real arm.
#   ./scripts/nexarm/rollout_stack_bowls.sh planbind|bspline|turbovla [rollout.py flags...]
# Rig defaults, as in collect.sh: follower /dev/ttyUSB1, front and wrist cameras by stable /dev/v4l/by-id
# paths, top RealSense auto-detected. The weights come from outputs/models/stack_bowls/<policy>, downloaded
# on the first run. Any flag passed here overrides these, e.g. --follower-port /dev/ttyUSB0 or --duration 60.
# TASK sets the prompt (TurboVLA reads it; B-spline and Plan-Bind always stack red, blue, black).
set -euo pipefail
policy="${1:?usage: $0 planbind|bspline|turbovla [rollout.py flags...]}"
shift
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
exec uv run --no-sync python examples/nexarm/infer_hub_policy.py "$policy" --target real \
    --follower-port /dev/ttyUSB1 \
    --front-cam /dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_E49C2640-video-index0 \
    --front-fourcc MJPG \
    --wrist-cam /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
    --wrist-fourcc YUYV \
    --task "${TASK:-Stack the bowls with red on bottom, blue in middle, and black on top.}" \
    "$@"
