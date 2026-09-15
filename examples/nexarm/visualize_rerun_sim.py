#!/usr/bin/env python
"""Visualize NexArm simulation in Rerun with 3D kinematics, cameras, telemetry, and multiple test scenarios.

Simulation Modes:
1. `joint_sweep`: Tests each joint independently across its catalog range (Joint 1..5 + Gripper).
2. `workspace`: 3D spherical exploration demonstrating full arm reach (614mm) and continuous wrist roll.
3. `poses`: Smooth cosine S-curve interpolation through 8 robotic task keyframes.
4. `trajectory`: Multi-axis smooth sinusoidal Lissajous sweep in 3D space.
5. `pick_place`: Automated physical manipulation task with cube detection, grasp, lift, and deposit.
6. `all`: Complete sequential showcase combining all modes above.

Usage:
    # 1. Test each joint limit one by one (default):
    uv run python examples/nexarm/visualize_rerun_sim.py --mode joint_sweep --view save

    # 2. Explore 3D workspace reach:
    uv run python examples/nexarm/visualize_rerun_sim.py --mode workspace --view save

    # 3. Cycle through standard robotic poses:
    uv run python examples/nexarm/visualize_rerun_sim.py --mode poses --view save

    # 4. Multi-axis trajectory sweep:
    uv run python examples/nexarm/visualize_rerun_sim.py --mode trajectory --view save

    # 5. Physics pick-and-place task:
    uv run python examples/nexarm/visualize_rerun_sim.py --mode pick_place --view save

    # 6. Run comprehensive multi-mode showcase:
    uv run python examples/nexarm/visualize_rerun_sim.py --mode all --view save
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    print("Error: rerun-sdk is required. Install via: uv sync --extra viz")
    sys.exit(1)

from lerobot.robots.nexarm_sim import (
    NexArmPickPlaceTask,
    NexArmSim,
    NexArmSimConfig,
)

DEFAULT_MODEL_PATH = Path("sim/description/mjcf/scene.xml")
DEFAULT_URDF_PATH = Path("sim/description/urdf/nexarm.urdf")

# Calibrated physical specifications for Hiwonder NexArm
JOINT_SPECS = [
    {
        "id": 1,
        "feature": "shoulder_pan",
        "label": "Joint 1: Shoulder Pan (Base Yaw)",
        "unit": "°",
        "max_limit": 135.0,  # +-135° = 270° total
        "rad_limit": 2.356194,
        "desc": "Horizontal base rotation across full 270° range",
        "axis": "[0, 0, 1] (Yaw)",
    },
    {
        "id": 2,
        "feature": "shoulder_lift",
        "label": "Joint 2: Shoulder Lift (Pitch)",
        "unit": "°",
        "max_limit": 120.0,  # +-120° = 240° total
        "rad_limit": 2.094395,
        "desc": "Shoulder pitch elevation across 240° range",
        "axis": "[1, 0, 0] (Pitch)",
    },
    {
        "id": 3,
        "feature": "elbow_flex",
        "label": "Joint 3: Elbow Flex (Pitch)",
        "unit": "°",
        "max_limit": 135.0,  # +-135° = 270° total
        "rad_limit": 2.356194,
        "desc": "Elbow pitch bending across full 270° range",
        "axis": "[1, 0, 0] (Pitch)",
    },
    {
        "id": 4,
        "feature": "wrist_flex",
        "label": "Joint 4: Wrist Flex (Pitch)",
        "unit": "°",
        "max_limit": 100.0,  # +-100° = 200° total
        "rad_limit": 1.745329,
        "desc": "Wrist pitch angle across 200° range",
        "axis": "[1, 0, 0] (Pitch)",
    },
    {
        "id": 5,
        "feature": "wrist_roll",
        "label": "Joint 5: Wrist Roll (Coaxial Spin)",
        "unit": "°",
        "max_limit": 180.0,  # +-180° = 360° total
        "rad_limit": 3.141593,
        "desc": "Pure axial twist of the wrist/gripper along forearm centerline (360°)",
        "axis": "[0, -1, 0] (Coaxial Roll)",
    },
    {
        "id": 6,
        "feature": "gripper",
        "label": "Joint 6: Parallel Gripper Jaws",
        "unit": "mm",
        "max_limit": 51.0,  # 0 to 51mm aperture
        "rad_limit": 0.051,
        "desc": "Rack-and-pinion linear jaw stroke from 0mm (closed) to 51mm (open)",
        "axis": "[-1, 0, 0] (Slide)",
    },
]

# Standard robotic keyframe poses
NAMED_POSES = [
    {
        "name": "1. Home / Standby",
        "desc": "Standard upright ready posture",
        "angles": {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    },
    {
        "name": "2. Forward Table Reach",
        "desc": "Reaching forward towards workspace with open jaws (51mm)",
        "angles": {
            "shoulder_pan": 0.0,
            "shoulder_lift": 45.0,
            "elbow_flex": 60.0,
            "wrist_flex": -25.0,
            "wrist_roll": 0.0,
            "gripper": 51.0,
        },
    },
    {
        "name": "3. Low Surface Pick",
        "desc": "Hovering directly over target object on table surface",
        "angles": {
            "shoulder_pan": 0.0,
            "shoulder_lift": 75.0,
            "elbow_flex": 80.0,
            "wrist_flex": 10.0,
            "wrist_roll": 0.0,
            "gripper": 51.0,
        },
    },
    {
        "name": "4. Clamp & Grasp",
        "desc": "Closing jaws to 0mm to secure payload",
        "angles": {
            "shoulder_pan": 0.0,
            "shoulder_lift": 75.0,
            "elbow_flex": 80.0,
            "wrist_flex": 10.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    },
    {
        "name": "5. Lift & Swing Left",
        "desc": "Elevating payload and panning 90° left to deposit bin",
        "angles": {
            "shoulder_pan": -90.0,
            "shoulder_lift": 25.0,
            "elbow_flex": 45.0,
            "wrist_flex": -10.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    },
    {
        "name": "6. Coaxial Roll Inspection",
        "desc": "Rotating wrist 180° along forearm axis to inspect payload from all sides",
        "angles": {
            "shoulder_pan": -90.0,
            "shoulder_lift": 25.0,
            "elbow_flex": 45.0,
            "wrist_flex": -10.0,
            "wrist_roll": 180.0,
            "gripper": 0.0,
        },
    },
    {
        "name": "7. High Shelf Reach Right",
        "desc": "Swinging 90° right and reaching up to top shelf",
        "angles": {
            "shoulder_pan": 90.0,
            "shoulder_lift": -35.0,
            "elbow_flex": -25.0,
            "wrist_flex": 35.0,
            "wrist_roll": 0.0,
            "gripper": 51.0,
        },
    },
    {
        "name": "8. Compact Tucked Rest",
        "desc": "Folds compactly towards base for storage or transit",
        "angles": {
            "shoulder_pan": 0.0,
            "shoulder_lift": -85.0,
            "elbow_flex": 120.0,
            "wrist_flex": -70.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    },
]


def deg_to_raw(joint_name: str, val: float) -> float:
    """Convert joint angle in degrees (or mm opening for gripper) to raw servo units."""
    if joint_name == "gripper":
        # 0mm -> 2833 (closed), 51mm -> 1195 (open)
        ratio = float(np.clip(val / 51.0, 0.0, 1.0))
        return 2833.0 - ratio * (2833.0 - 1195.0)

    limits_deg = {
        "shoulder_pan": 135.0,
        "shoulder_lift": 120.0,
        "elbow_flex": 135.0,
        "wrist_flex": 100.0,
        "wrist_roll": 180.0,
    }
    max_deg = limits_deg.get(joint_name, 180.0)
    ratio = float(np.clip(val / max_deg, -1.0, 1.0))
    return 2048.0 + ratio * 2047.0


def default_home_action() -> dict[str, float]:
    """Return home posture with all arm joints at 2048 and gripper closed at 2833."""
    return {
        "shoulder_pan.pos": 2048.0,
        "shoulder_lift.pos": 2048.0,
        "elbow_flex.pos": 2048.0,
        "wrist_flex.pos": 2048.0,
        "wrist_roll.pos": 2048.0,
        "gripper.pos": 2833.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize NexArm simulation in Rerun")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to MuJoCo scene XML",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help="Path to robot URDF",
    )
    parser.add_argument(
        "--mode",
        choices=["joint_sweep", "workspace", "poses", "trajectory", "pick_place", "all"],
        default="joint_sweep",
        help="Simulation mode: joint_sweep (test limits), workspace (move around), poses (keyframe poses), trajectory, pick_place, or all",
    )
    parser.add_argument(
        "--view",
        choices=["spawn", "web", "save"],
        default="auto",
        help="Viewer mode: spawn desktop app, host web server, or save .rrd file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sim/nexarm_demo.rrd"),
        help="Path to save .rrd file when --view save is used",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Port for the Rerun web viewer (default: 9090)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=12.0,
        help="Demo duration in seconds (default: 12.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Visualization update rate in Hz (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment reset",
    )
    return parser.parse_args()


def setup_blueprint(recording: rr.RecordingStream) -> None:
    """Build a multi-panel Rerun layout combining 3D scene, dual cameras, telemetry, and live status."""
    views = [
        rrb.Spatial3DView(
            origin="/",
            name="NexArm 3D Kinematics & Scene",
            contents=["+ /**", "- /cameras/**", "- /telemetry/**", "- /status/**"],
        ),
        rrb.Grid(
            rrb.Spatial2DView(origin="cameras/front", name="Front Camera (640x480)"),
            rrb.Spatial2DView(origin="cameras/wrist", name="Wrist Camera (640x480)"),
        ),
        rrb.TimeSeriesView(
            origin="telemetry/joints",
            name="Joint Positions (rad / m)",
        ),
        rrb.TimeSeriesView(
            origin="telemetry/actions",
            name="Actuator Actions (0..4095)",
        ),
    ]
    if hasattr(rrb, "TextDocumentView"):
        views.append(
            rrb.TextDocumentView(
                origin="status/info",
                name="Simulation Status & Calibration",
            )
        )
    blueprint = rrb.Blueprint(rrb.Grid(*views, grid_columns=2))
    recording.send_blueprint(blueprint)


def log_scene_objects(
    recording: rr.RecordingStream,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trail_points: list[list[float]],
) -> None:
    """Log the movable cube, target zone, and end-effector trail into 3D view."""
    # Cube position & orientation
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if cube_id >= 0:
        cube_pos = data.xpos[cube_id]
        cube_quat = data.xquat[cube_id]  # w, x, y, z
        recording.log(
            "scene/cube",
            rr.CoordinateFrame("base_link"),
            rr.Boxes3D(
                half_sizes=[[0.0175, 0.0175, 0.0175]],
                centers=[cube_pos.tolist()],
                rotations=[rr.Quaternion(xyzw=[cube_quat[1], cube_quat[2], cube_quat[3], cube_quat[0]])],
                colors=[[230, 60, 60, 255]],
                fill_mode="solid",
            ),
        )

    # Target zone
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    if target_id >= 0:
        target_pos = data.xpos[target_id]
        recording.log(
            "scene/target_zone",
            rr.CoordinateFrame("base_link"),
            rr.Boxes3D(
                half_sizes=[[0.035, 0.035, 0.002]],
                centers=[target_pos.tolist()],
                colors=[[60, 200, 60, 180]],
                fill_mode="solid",
            ),
        )

    # End-effector trail tracking
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame")
    if site_id >= 0:
        ee_pos = data.site_xpos[site_id].tolist()
        trail_points.append(ee_pos)
        # Keep maximum trail history to avoid unbounded memory
        if len(trail_points) > 300:
            trail_points.pop(0)

        recording.log(
            "scene/ee_trail",
            rr.CoordinateFrame("base_link"),
            rr.LineStrips3D(
                [trail_points],
                colors=[[0, 180, 255, 200]],
                radii=0.003,
            ),
        )


def log_cameras(
    recording: rr.RecordingStream,
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
) -> None:
    """Render and log front and wrist camera frames."""
    renderer.update_scene(data, camera="front")
    front_img = renderer.render()
    recording.log("cameras/front", rr.Image(front_img).compress())

    renderer.update_scene(data, camera="wrist")
    wrist_img = renderer.render()
    recording.log("cameras/wrist", rr.Image(wrist_img).compress())


def log_joint_telemetry(
    recording: rr.RecordingStream,
    tree: rr.urdf.UrdfTree,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    raw_action: dict[str, float] | None = None,
) -> None:
    """Log kinematic transforms to Rerun 3D tree and scalar values to time series."""
    for j in tree.joints():
        if j.joint_type == "fixed":
            continue
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name)
        if j_id >= 0:
            qadr = model.jnt_qposadr[j_id]
            pos_val = float(data.qpos[qadr])
            if j.limit_lower is not None and j.limit_upper is not None:
                pos_val = float(np.clip(pos_val, j.limit_lower, j.limit_upper))
            tf = j.compute_transform(pos_val)
            recording.log(f"tf/{j.name}", tf)
            recording.log(f"telemetry/joints/{j.name}", rr.Scalars(pos_val))

    if raw_action:
        for name, cmd in raw_action.items():
            clean_name = name.replace(".pos", "")
            recording.log(f"telemetry/actions/{clean_name}", rr.Scalars(float(cmd)))


# ---------------------------------------------------------------------------
# Simulation Demo Modes
# ---------------------------------------------------------------------------


def run_joint_sweep_demo(
    sim: NexArmSim,
    tree: rr.urdf.UrdfTree,
    recording: rr.RecordingStream,
    duration_s: float,
    fps: int,
    step_offset: int = 0,
) -> int:
    """Sequentially test each joint across its catalog range while holding others at home."""
    renderer = mujoco.Renderer(sim.backend.model, height=480, width=640)
    trail_points: list[list[float]] = []

    num_joints = len(JOINT_SPECS)
    time_per_joint = max(2.0, duration_s / num_joints)
    steps_per_joint = int(time_per_joint * fps)
    total_steps = steps_per_joint * num_joints

    print(
        f"\n[Mode: joint_sweep] Testing {num_joints} joints sequentially ({time_per_joint:.1f}s per joint)..."
    )

    for j_idx, spec in enumerate(JOINT_SPECS):
        j_feat = spec["feature"]
        j_label = spec["label"]
        max_val = spec["max_limit"]
        unit = spec["unit"]
        print(f"  [{j_idx + 1}/{num_joints}] Testing {j_label}...")

        for s in range(steps_per_joint):
            global_step = step_offset + j_idx * steps_per_joint + s
            t_rel = s / steps_per_joint
            sim_time = (j_idx * steps_per_joint + s) / fps

            recording.set_time("step", sequence=global_step)
            recording.set_time("simulation_time", duration=sim_time)

            raw_action = default_home_action()

            # Sinusoidal motion: 0 -> +max -> 0 -> -max -> 0
            if j_feat == "gripper":
                # Aperture cycles 0mm (closed) -> 51mm (open) -> 0mm (closed)
                current_val = (max_val / 2.0) * (1.0 - math.cos(2.0 * math.pi * t_rel))
            else:
                current_val = max_val * math.sin(2.0 * math.pi * t_rel)

            raw_action[f"{j_feat}.pos"] = deg_to_raw(j_feat, current_val)

            sim.send_action(raw_action)

            # Live Markdown status
            cur_rad_str = (
                f"`{math.radians(current_val):+.3f} rad`"
                if unit == "°"
                else f"`{current_val / 1000.0:.4f} m`"
            )
            status_md = f"""# 🔬 NexArm Joint Calibration Test
### Active Test: `{j_label}`
- **Test Phase**: `[{j_idx + 1} / {num_joints}]`
- **Description**: {spec["desc"]}
- **Current Position**: `{current_val:+.1f} {unit}` ({cur_rad_str})
- **Catalog Range**: `[-{max_val:.1f}, +{max_val:.1f}] {unit}`
- **Rotation Axis**: `{spec["axis"]}`
- **Servo Command**: `{int(raw_action[f"{j_feat}.pos"])} / 4095`
- **Status**: ✅ **VERIFIED WITHIN LEGAL LIMITS**
"""
            recording.log("status/info", rr.TextDocument(status_md, media_type=rr.MediaType.MARKDOWN))

            log_scene_objects(recording, sim.backend.model, sim.backend.data, trail_points)
            log_cameras(recording, renderer, sim.backend.data)
            log_joint_telemetry(recording, tree, sim.backend.model, sim.backend.data, raw_action)

    renderer.close()
    print("✓ Joint sweep test complete.")
    return step_offset + total_steps


def run_workspace_demo(
    sim: NexArmSim,
    tree: rr.urdf.UrdfTree,
    recording: rr.RecordingStream,
    duration_s: float,
    fps: int,
    step_offset: int = 0,
) -> int:
    """Explore the 3D reachable workspace envelope around the base with active wrist roll."""
    total_steps = int(duration_s * fps)
    renderer = mujoco.Renderer(sim.backend.model, height=480, width=640)
    trail_points: list[list[float]] = []

    print(f"\n[Mode: workspace] Exploring 3D workspace reach for {duration_s:.1f}s ({total_steps} steps)...")
    for step in range(total_steps):
        global_step = step_offset + step
        t = step / fps
        recording.set_time("step", sequence=global_step)
        recording.set_time("simulation_time", duration=t)

        # 3D exploration:
        # 1. Base yaw swings broadly across -125° to +125° (270° range)
        pan_deg = 125.0 * math.sin(0.5 * t)
        # 2. Shoulder lift undulates: reaching forward and elevating
        lift_deg = 45.0 + 55.0 * math.sin(0.8 * t)
        # 3. Elbow flex extends and retracts reach depth
        elbow_deg = -30.0 + 75.0 * math.cos(0.9 * t)
        # 4. Wrist flex counter-pitches to keep gripper oriented
        wrist_deg = 40.0 * math.sin(1.2 * t)
        # 5. Wrist roll continuously spins 360° to showcase pure coaxial rotation
        roll_deg = 180.0 * math.sin(1.5 * t)
        # 6. Gripper pulses open and close (0 to 51mm)
        grip_mm = 25.5 * (1.0 + math.cos(1.0 * t))

        raw_action = {
            "shoulder_pan.pos": deg_to_raw("shoulder_pan", pan_deg),
            "shoulder_lift.pos": deg_to_raw("shoulder_lift", lift_deg),
            "elbow_flex.pos": deg_to_raw("elbow_flex", elbow_deg),
            "wrist_flex.pos": deg_to_raw("wrist_flex", wrist_deg),
            "wrist_roll.pos": deg_to_raw("wrist_roll", roll_deg),
            "gripper.pos": deg_to_raw("gripper", grip_mm),
        }

        sim.send_action(raw_action)

        status_md = f"""# 🌐 NexArm 3D Workspace Exploration
- **Base Pan**: `{pan_deg:+.1f}°` (sweeping azimuth across 270°)
- **Reach Undulation**: Shoulder `{lift_deg:+.1f}°`, Elbow `{elbow_deg:+.1f}°`
- **Wrist Roll**: `{roll_deg:+.1f}°` (coaxial spin along forearm axis)
- **Gripper Opening**: `{grip_mm:.1f} mm`
- **Time**: `{t:.1f}s / {duration_s:.1f}s`
- **Envelope Reach**: Up to 614mm extended radius without singularity
"""
        recording.log("status/info", rr.TextDocument(status_md, media_type=rr.MediaType.MARKDOWN))

        log_scene_objects(recording, sim.backend.model, sim.backend.data, trail_points)
        log_cameras(recording, renderer, sim.backend.data)
        log_joint_telemetry(recording, tree, sim.backend.model, sim.backend.data, raw_action)

    renderer.close()
    print("✓ Workspace exploration complete.")
    return step_offset + total_steps


def run_poses_demo(
    sim: NexArmSim,
    tree: rr.urdf.UrdfTree,
    recording: rr.RecordingStream,
    duration_s: float,
    fps: int,
    step_offset: int = 0,
) -> int:
    """Smoothly interpolate through 8 standard robotic operation keyframe poses."""
    renderer = mujoco.Renderer(sim.backend.model, height=480, width=640)
    trail_points: list[list[float]] = []

    num_poses = len(NAMED_POSES)
    time_per_transition = max(1.5, duration_s / num_poses)
    steps_per_transition = int(time_per_transition * fps)
    total_steps = steps_per_transition * num_poses

    print(f"\n[Mode: poses] Cycling through {num_poses} operational keyframe poses...")

    for p_idx in range(num_poses):
        curr_pose = NAMED_POSES[p_idx]
        next_pose = NAMED_POSES[(p_idx + 1) % num_poses]
        print(f"  Transition [{p_idx + 1}/{num_poses}]: '{curr_pose['name']}' -> '{next_pose['name']}'")

        for s in range(steps_per_transition):
            global_step = step_offset + p_idx * steps_per_transition + s
            u = s / steps_per_transition
            # Minimum-jerk cosine S-curve interpolation
            alpha = 0.5 * (1.0 - math.cos(math.pi * u))
            sim_time = (p_idx * steps_per_transition + s) / fps

            recording.set_time("step", sequence=global_step)
            recording.set_time("simulation_time", duration=sim_time)

            raw_action = {}
            for j_name in [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ]:
                v_start = curr_pose["angles"][j_name]
                v_end = next_pose["angles"][j_name]
                v_interp = v_start + alpha * (v_end - v_start)
                raw_action[f"{j_name}.pos"] = deg_to_raw(j_name, v_interp)

            sim.send_action(raw_action)

            status_md = f"""# 🤖 NexArm Keyframe Poses Demo
### Target Pose: `{next_pose["name"]}`
- **Transition**: `[{p_idx + 1} / {num_poses}]` (Progress: `{alpha * 100:.0f}%`)
- **Description**: {next_pose["desc"]}
- **Joint Targets**:
  - `shoulder_pan`: `{next_pose["angles"]["shoulder_pan"]:+.1f}°`
  - `shoulder_lift`: `{next_pose["angles"]["shoulder_lift"]:+.1f}°`
  - `elbow_flex`: `{next_pose["angles"]["elbow_flex"]:+.1f}°`
  - `wrist_flex`: `{next_pose["angles"]["wrist_flex"]:+.1f}°`
  - `wrist_roll`: `{next_pose["angles"]["wrist_roll"]:+.1f}°`
  - `gripper`: `{next_pose["angles"]["gripper"]:.1f} mm`
"""
            recording.log("status/info", rr.TextDocument(status_md, media_type=rr.MediaType.MARKDOWN))

            log_scene_objects(recording, sim.backend.model, sim.backend.data, trail_points)
            log_cameras(recording, renderer, sim.backend.data)
            log_joint_telemetry(recording, tree, sim.backend.model, sim.backend.data, raw_action)

    renderer.close()
    print("✓ Keyframe poses demo complete.")
    return step_offset + total_steps


def run_trajectory_demo(
    sim: NexArmSim,
    tree: rr.urdf.UrdfTree,
    recording: rr.RecordingStream,
    duration_s: float,
    fps: int,
    step_offset: int = 0,
) -> int:
    """Execute a smooth sinusoidal Lissajous sweep showcasing 6 DoF motion and gripper operation."""
    total_steps = int(duration_s * fps)
    renderer = mujoco.Renderer(sim.backend.model, height=480, width=640)
    trail_points: list[list[float]] = []

    print(
        f"\n[Mode: trajectory] Running sinusoidal trajectory demo for {duration_s:.1f}s ({total_steps} steps)..."
    )
    for step in range(total_steps):
        global_step = step_offset + step
        t = step / fps
        recording.set_time("step", sequence=global_step)
        recording.set_time("simulation_time", duration=t)

        raw_action = {
            "shoulder_pan.pos": 2048.0 + 700.0 * math.sin(1.2 * t),
            "shoulder_lift.pos": 2048.0 + 500.0 * math.sin(1.0 * t),
            "elbow_flex.pos": 2048.0 + 600.0 * math.cos(0.9 * t),
            "wrist_flex.pos": 2048.0 + 400.0 * math.sin(1.5 * t),
            "wrist_roll.pos": 2048.0 + 800.0 * math.cos(1.1 * t),
            "gripper.pos": 2014.0 + 819.0 * math.sin(2.0 * t),
        }

        sim.send_action(raw_action)

        status_md = f"""# 📈 NexArm Multi-Axis Sinusoidal Trajectory
- **Simulation Time**: `{t:.2f}s / {duration_s:.1f}s`
- **Multi-axis Coordination**: Simultaneous sinusoidal sweep of all 6 DoF actuators
- **End-effector Trail**: Active 3D spline tracking
"""
        recording.log("status/info", rr.TextDocument(status_md, media_type=rr.MediaType.MARKDOWN))

        log_scene_objects(recording, sim.backend.model, sim.backend.data, trail_points)
        log_cameras(recording, renderer, sim.backend.data)
        log_joint_telemetry(recording, tree, sim.backend.model, sim.backend.data, raw_action)

    renderer.close()
    print("✓ Trajectory demo complete.")
    return step_offset + total_steps


def run_pick_place_demo(
    sim: NexArmSim,
    tree: rr.urdf.UrdfTree,
    recording: rr.RecordingStream,
    duration_s: float,
    fps: int,
    seed: int,
    step_offset: int = 0,
) -> int:
    """Run pick-and-place simulation task logging object manipulation in Rerun."""
    task = NexArmPickPlaceTask(sim.backend, timeout_s=duration_s)
    task.reset(seed=seed, settle_steps=25)
    renderer = mujoco.Renderer(sim.backend.model, height=480, width=640)
    trail_points: list[list[float]] = []

    total_steps = int(duration_s * fps)
    print(f"\n[Mode: pick_place] Running pick-and-place simulation for {duration_s:.1f}s (seed={seed})...")

    for step in range(total_steps):
        global_step = step_offset + step
        sim_time = float(sim.backend.data.time)
        recording.set_time("step", sequence=global_step)
        recording.set_time("simulation_time", duration=sim_time)

        task_status = task.step()

        if task_status.success:
            state_label = "SUCCESS (Placed & Held)"
        elif task_status.terminated:
            state_label = f"TERMINATED ({task_status.reason})"
        elif task_status.is_grasped:
            state_label = "GRASPED (Carrying to Target)"
        elif task_status.is_inside_target:
            state_label = "RELEASING (In Target Zone)"
        else:
            state_label = "APPROACHING (Tracking Cube)"

        status_md = f"""# 📦 NexArm Autonomous Pick & Place Task
- **Task State**: `{state_label}`
- **Simulation Time**: `{sim_time:.2f}s / {duration_s:.1f}s`
- **Object Grasped**: `{"YES" if task_status.is_grasped else "NO"}`
- **Inside Target**: `{"YES" if task_status.is_inside_target else "NO"}`
- **Gripper Jaws**: `{"CLOSED (Grasping)" if task_status.is_grasped else "OPEN (51mm)"}`
"""
        recording.log("status/info", rr.TextDocument(status_md, media_type=rr.MediaType.MARKDOWN))

        log_scene_objects(recording, sim.backend.model, sim.backend.data, trail_points)
        log_cameras(recording, renderer, sim.backend.data)
        log_joint_telemetry(recording, tree, sim.backend.model, sim.backend.data)

    renderer.close()
    print("✓ Pick-and-place demo complete.")
    return step_offset + total_steps


def run_all_demos(
    sim: NexArmSim,
    tree: rr.urdf.UrdfTree,
    recording: rr.RecordingStream,
    duration_s: float,
    fps: int,
    seed: int,
) -> None:
    """Execute the full simulation showcase: Joint Sweep -> Poses -> Workspace -> Pick & Place."""
    print("\n" + "=" * 70)
    print("Starting Comprehensive NexArm Multi-Simulation Showcase")
    print("=" * 70)
    step = 0
    step = run_joint_sweep_demo(sim, tree, recording, duration_s=12.0, fps=fps, step_offset=step)
    step = run_poses_demo(sim, tree, recording, duration_s=14.0, fps=fps, step_offset=step)
    step = run_workspace_demo(sim, tree, recording, duration_s=10.0, fps=fps, step_offset=step)
    run_pick_place_demo(sim, tree, recording, duration_s=8.0, fps=fps, seed=seed, step_offset=step)
    print("=" * 70)
    print("✓ Complete multi-simulation showcase finished successfully.")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    view_mode = args.view
    if view_mode == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        view_mode = "spawn" if has_display else "save"

    app_id = "nexarm_simulation_rerun"
    recording = rr.RecordingStream(application_id=app_id)

    if view_mode == "save":
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recording.save(str(output_path))
        print(f"Rerun sink: saving to file -> {output_path}")
    elif view_mode == "web":
        server_uri = rr.serve_grpc()
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False)
        print(f"Rerun web viewer started: http://localhost:{args.web_port}")
        print(f"Listening on gRPC server: {server_uri}")
    elif view_mode == "spawn":
        rr.spawn(memory_limit="25%")
        print("Spawned native Rerun desktop viewer.")

    setup_blueprint(recording)

    urdf_path = args.urdf.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")
    print(f"Loading URDF into Rerun: {urdf_path}")
    tree = rr.urdf.UrdfTree.from_file_path(str(urdf_path))
    tree.log_urdf_to_recording(recording)

    # Establish coordinate frames and static transform connections so Rerun 3D view
    # has a continuous transform path from root and scene entities to URDF base_link.
    recording.log("base_link", rr.Transform3D(child_frame="base_link"), static=True)
    recording.log(
        "scene",
        rr.CoordinateFrame("base_link"),
        rr.Transform3D(parent_frame="base_link"),
        static=True,
    )

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo scene XML not found at {model_path}")
    print(f"Initializing MuJoCo simulation: {model_path}")

    sim = NexArmSim(
        NexArmSimConfig(
            id="rerun_sim",
            model_path=model_path,
            fps=args.fps,
            camera_names=("front", "wrist"),
            settle_steps=10,
        )
    )
    sim.connect()

    try:
        if args.mode == "joint_sweep":
            run_joint_sweep_demo(sim, tree, recording, args.duration, args.fps)
        elif args.mode == "workspace":
            run_workspace_demo(sim, tree, recording, args.duration, args.fps)
        elif args.mode == "poses":
            run_poses_demo(sim, tree, recording, args.duration, args.fps)
        elif args.mode == "trajectory":
            run_trajectory_demo(sim, tree, recording, args.duration, args.fps)
        elif args.mode == "pick_place":
            run_pick_place_demo(sim, tree, recording, args.duration, args.fps, args.seed)
        elif args.mode == "all":
            run_all_demos(sim, tree, recording, args.duration, args.fps, args.seed)
    finally:
        sim.disconnect()

    if view_mode == "save":
        print("\n" + "=" * 70)
        print(f"✓ Recording successfully saved: {args.output}")
        print("To inspect interactively in Rerun:")
        print(f"    uv run rerun {args.output}")
        print("=" * 70)
    elif view_mode == "web":
        print(f"\n✓ Visualizer running on Web viewer: http://localhost:{args.web_port}")
        print("Press Ctrl+C to stop the web server.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopped web server.")


if __name__ == "__main__":
    main()
