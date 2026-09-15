#!/usr/bin/env python
"""Convenience CLI entry point for NexArm Rerun multi-mode visualization.

Modes:
    --mode joint_sweep : Test each joint limit one-by-one (Joint 1..5 + Gripper)
    --mode workspace   : 3D spherical exploration demonstrating full reach & continuous roll
    --mode poses       : Smooth interpolation through 8 robotic operational keyframe poses
    --mode trajectory  : Multi-axis smooth sinusoidal Lissajous curve
    --mode pick_place  : Automated physical manipulation task with cube pick and place
    --mode all         : Comprehensive sequential showcase combining all modes

Usage:
    # 1. Test each joint independently (default mode):
    uv run python sim/visualize_rerun.py --mode joint_sweep --view save

    # 2. Explore 3D workspace reach (di chuyển xung quanh không gian làm việc):
    uv run python sim/visualize_rerun.py --mode workspace --view save

    # 3. Standard robotic keyframe poses:
    uv run python sim/visualize_rerun.py --mode poses --view save

    # 4. Run all simulations in sequence:
    uv run python sim/visualize_rerun.py --mode all --view save

    # 5. Launch in web browser viewer:
    uv run python sim/visualize_rerun.py --mode joint_sweep --view web
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples.nexarm.visualize_rerun_sim import main  # noqa: E402

if __name__ == "__main__":
    main()
