#!/usr/bin/env python
"""Comprehensive validation suite for NexArm simulation assets.

Tests:
1. URDF: XML syntax, kinematics, inertia, MuJoCo loading/stepping, and Rerun UrdfTree.
2. XACRO: Macro expansion, ROS vs standalone modes, parameter flags, and MuJoCo load.
3. MuJoCo MJCF XML: Robot model, scene XML, actuators, physics stability, and camera rendering.

Usage:
    uv run python sim/test_sim_models.py
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# Ensure rerun is importable if installed
try:
    import rerun as rr

    HAVE_RERUN = True
except ImportError:
    HAVE_RERUN = False

# Ensure xacro is importable if installed
try:
    import xacro

    HAVE_XACRO = True
except ImportError:
    HAVE_XACRO = False

ROOT_DIR = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT_DIR / "sim"
DESC_DIR = SIM_DIR / "description"
FUSION_DIR = SIM_DIR / "fusion_export"
ASSETS_DIR = SIM_DIR / "assets" / "meshes" / "visual"


@dataclass
class TestResult:
    category: str
    target: str
    passed: bool
    message: str
    elapsed_ms: float = 0.0


results: list[TestResult] = []


def run_check(category: str, target: str, check_fn: Callable[[], str | None]) -> None:
    t0 = time.perf_counter()
    try:
        msg = check_fn()
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(
            TestResult(
                category=category,
                target=target,
                passed=True,
                message=msg or "OK",
                elapsed_ms=elapsed,
            )
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(
            TestResult(
                category=category,
                target=target,
                passed=False,
                message=f"FAILED: {exc}",
                elapsed_ms=elapsed,
            )
        )


# ==============================================================================
# 1. URDF Tests
# ==============================================================================
def check_urdf_file(urdf_path: Path) -> str:
    if not urdf_path.exists():
        raise FileNotFoundError(f"File not found: {urdf_path}")

    # Check loading in MuJoCo
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)
    if model.nq != 8 or model.nv != 8:
        raise ValueError(f"Expected nq=8, nv=8, got nq={model.nq}, nv={model.nv}")

    # Verify physical stepping
    for _ in range(50):
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            raise ValueError("Simulation divergence detected (NaNs in state)")

    msg = f"MuJoCo loaded: nq={model.nq}, nv={model.nv}, nbody={model.nbody}"

    # Check loading in Rerun UrdfTree if rerun available
    if HAVE_RERUN:
        tree = rr.urdf.UrdfTree.from_file_path(str(urdf_path))
        joints = tree.joints()
        if len(joints) < 8:
            raise ValueError(f"UrdfTree found only {len(joints)} joints")
        msg += f" | Rerun UrdfTree: {len(joints)} joints"

    return msg


def check_urdf_inertia(urdf_path: Path) -> str:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    non_zero_bodies = 0
    for body_id in range(model.nbody):
        mass = model.body_mass[body_id]
        if mass > 0:
            non_zero_bodies += 1
            inertia = model.body_inertia[body_id]
            if any(i <= 0 for i in inertia):
                raise ValueError(
                    f"Body {body_id} ({mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)}) has non-positive inertia: {inertia}"
                )
    return f"{non_zero_bodies} inertial bodies verified positive-definite"


# ==============================================================================
# 2. Xacro Tests
# ==============================================================================
def check_xacro_ros_compilation(xacro_path: Path) -> str:
    if not HAVE_XACRO:
        return "SKIPPED: xacro package not installed"

    doc = xacro.process_file(str(xacro_path))
    xml_content = doc.toxml()
    if "package://nexarm_description/meshes/visual/base_link.stl" not in xml_content:
        raise ValueError("Default XACRO output did not contain expected ROS package:// mesh paths")
    if "joint_1_base_to_link_1" not in xml_content:
        raise ValueError("Missing joint_1_base_to_link_1 in compiled XACRO")
    return "Compiled to ROS URDF with package:// URIs"


def check_xacro_standalone_and_mujoco(xacro_path: Path) -> str:
    if not HAVE_XACRO:
        return "SKIPPED: xacro package not installed"

    # Compile with mesh_dir:=meshes
    doc = xacro.process_file(str(xacro_path), mappings={"mesh_dir": "meshes"})
    xml_content = doc.toxml()
    if "meshes/base_link.stl" not in xml_content:
        raise ValueError("Standalone XACRO output did not contain relative meshes/ paths")

    # MuJoCo load test from the urdf directory context
    orig_cwd = os.getcwd()
    try:
        os.chdir(xacro_path.parent)
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
        for _ in range(20):
            mujoco.mj_step(model, data)
    finally:
        os.chdir(orig_cwd)

    return f"Compiled standalone & stepped in MuJoCo (nq={model.nq}, nv={model.nv})"


def check_xacro_camera_mount_flag(xacro_path: Path) -> str:
    if not HAVE_XACRO:
        return "SKIPPED: xacro package not installed"

    # With attach_cam_mount:=false
    doc_no_cam = xacro.process_file(str(xacro_path), mappings={"attach_cam_mount": "false"})
    xml_no_cam = doc_no_cam.toxml()
    if "cam_mount" in xml_no_cam:
        raise ValueError("cam_mount link was found when attach_cam_mount=false")

    # With attach_cam_mount:=true
    doc_cam = xacro.process_file(str(xacro_path), mappings={"attach_cam_mount": "true"})
    xml_cam = doc_cam.toxml()
    if "cam_mount" not in xml_cam:
        raise ValueError("cam_mount link was missing when attach_cam_mount=true")

    return "Conditional cam_mount verified (supports optional camera bracket)"


# ==============================================================================
# 3. MuJoCo MJCF XML Tests
# ==============================================================================
def check_mjcf_robot(xml_path: Path) -> str:
    if not xml_path.exists():
        raise FileNotFoundError(f"File not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if model.nq != 8 or model.nv != 8:
        raise ValueError(f"Expected nq=8, nv=8, got nq={model.nq}, nv={model.nv}")
    if model.nu != 6:
        raise ValueError(f"Expected nu=6 (5 arm + 1 gripper), got {model.nu}")

    for _ in range(50):
        mujoco.mj_step(model, data)

    return f"Loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}, ngeom={model.ngeom}"


def check_mjcf_scene_and_cameras(scene_path: Path) -> str:
    if not scene_path.exists():
        raise FileNotFoundError(f"File not found: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Check cameras
    cam_front_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")

    if cam_front_id < 0:
        raise ValueError("Camera 'front' not found in scene")
    if cam_wrist_id < 0:
        raise ValueError("Camera 'wrist' not found in scene")

    # Step simulation
    for _ in range(20):
        mujoco.mj_step(model, data)

    # Test offscreen renderer
    renderer = mujoco.Renderer(model, height=240, width=320)
    renderer.update_scene(data, camera="front")
    img_front = renderer.render()
    if img_front.shape != (240, 320, 3) or img_front.dtype != np.uint8:
        raise ValueError(f"Unexpected front image shape/dtype: {img_front.shape}, {img_front.dtype}")

    renderer.update_scene(data, camera="wrist")
    img_wrist = renderer.render()
    if img_wrist.shape != (240, 320, 3) or img_wrist.dtype != np.uint8:
        raise ValueError(f"Unexpected wrist image shape/dtype: {img_wrist.shape}, {img_wrist.dtype}")

    renderer.close()
    return f"Verified scene, front ({img_front.shape}) & wrist ({img_wrist.shape}) cameras"


def check_joint_limits_and_axes(xml_path: Path) -> str:
    """Verify calibrated joint angular limits and axes against official NexArm specifications."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    expected_ranges = {
        "joint_1_base_to_link_1": (-2.356194, 2.356194),  # 270° (+-135°)
        "joint_2_link_1_to_link_2": (-2.094395, 2.094395),  # 240° (+-120°)
        "joint_3_link_2_to_link_3": (-2.356194, 2.356194),  # 270° (+-135°)
        "joint_4_link_3_to_link_4": (-1.745329, 1.745329),  # 200° (+-100°)
        "joint_5_link_4_to_link_5": (-3.141593, 3.141593),  # 360° (+-180°)
        "right_jaw_slide_joint": (-0.0255, 0.0),  # 51mm total aperture (stroke 25.5mm)
    }

    for j_name, (exp_low, exp_high) in expected_ranges.items():
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id < 0:
            raise ValueError(f"Joint '{j_name}' missing from model {xml_path.name}")
        r_low, r_high = model.jnt_range[j_id]
        if abs(r_low - exp_low) > 0.02 or abs(r_high - exp_high) > 0.02:
            raise ValueError(
                f"Joint '{j_name}' range mismatch: [{r_low:.4f}, {r_high:.4f}] vs expected [{exp_low:.4f}, {exp_high:.4f}]"
            )

    # Verify Joint 5 wrist roll axis points purely along Y (coaxial with forearm)
    j5_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_5_link_4_to_link_5")
    axis_j5 = model.jnt_axis[j5_id]
    if abs(axis_j5[1]) < 0.9 or abs(axis_j5[2]) > 0.1:
        raise ValueError(
            f"Joint 5 axis '{axis_j5}' is tilting (Z={axis_j5[2]:.4f}). Must be along Y ([0, -1, 0] coaxial roll)."
        )

    return "All 6 joints calibrated: 270°/240°/270°/200°/360° & 51mm gripper with coaxial roll axis"


def main() -> int:
    print("=" * 80)
    print(" NEXARM SIMULATION ASSET VALIDATION (URDF, XACRO, MUJOCO XML)")
    print("=" * 80)

    # 1. URDF
    print("\n[1/3] Testing URDF models...")
    urdf_desc = DESC_DIR / "urdf" / "nexarm.urdf"
    urdf_fusion = FUSION_DIR / "nexarm.urdf"

    run_check("URDF", "description/urdf/nexarm.urdf", lambda: check_urdf_file(urdf_desc))
    run_check("URDF", "description/urdf/nexarm.urdf (inertia)", lambda: check_urdf_inertia(urdf_desc))
    if urdf_fusion.exists():
        run_check("URDF", "fusion_export/nexarm.urdf", lambda: check_urdf_file(urdf_fusion))

    # 2. XACRO
    print("[2/3] Testing XACRO models...")
    xacro_desc = DESC_DIR / "urdf" / "nexarm.urdf.xacro"
    run_check("XACRO", "nexarm.urdf.xacro (ROS)", lambda: check_xacro_ros_compilation(xacro_desc))
    run_check(
        "XACRO",
        "nexarm.urdf.xacro (Standalone/MuJoCo)",
        lambda: check_xacro_standalone_and_mujoco(xacro_desc),
    )
    run_check(
        "XACRO", "nexarm.urdf.xacro (Cam mount param)", lambda: check_xacro_camera_mount_flag(xacro_desc)
    )

    # 3. MuJoCo MJCF XML
    print("[3/3] Testing MuJoCo MJCF XML...")
    mjcf_desc_arm = DESC_DIR / "mjcf" / "nexarm.xml"
    mjcf_desc_scene = DESC_DIR / "mjcf" / "scene.xml"
    mjcf_fusion_arm = FUSION_DIR / "NexArm-sim.xml"
    mjcf_fusion_scene = FUSION_DIR / "scene.xml"
    mjcf_root_arm = SIM_DIR / "NexArm-sim.xml"

    run_check("MJCF", "description/mjcf/nexarm.xml", lambda: check_mjcf_robot(mjcf_desc_arm))
    run_check(
        "MJCF",
        "description/mjcf/nexarm.xml (limits & axes)",
        lambda: check_joint_limits_and_axes(mjcf_desc_arm),
    )
    run_check("MJCF", "description/mjcf/scene.xml", lambda: check_mjcf_scene_and_cameras(mjcf_desc_scene))
    if mjcf_fusion_arm.exists():
        run_check("MJCF", "fusion_export/NexArm-sim.xml", lambda: check_mjcf_robot(mjcf_fusion_arm))
    if mjcf_fusion_scene.exists():
        run_check("MJCF", "fusion_export/scene.xml", lambda: check_mjcf_scene_and_cameras(mjcf_fusion_scene))
    if mjcf_root_arm.exists():
        run_check("MJCF", "sim/NexArm-sim.xml", lambda: check_mjcf_robot(mjcf_root_arm))

    # Print Summary Table
    print("\n" + "=" * 80)
    print(f"{'CATEGORY':<8} | {'TARGET':<35} | {'STATUS':<6} | {'TIME (ms)':<9} | {'DETAILS'}")
    print("-" * 80)
    all_passed = True
    for r in results:
        status_str = "\033[92mPASS\033[0m" if r.passed else "\033[91mFAIL\033[0m"
        if not r.passed:
            all_passed = False
        print(f"{r.category:<8} | {r.target:<35} | {status_str:<15} | {r.elapsed_ms:>7.1f}ms | {r.message}")

    print("=" * 80)
    if all_passed:
        print("\033[92m✓ All URDF, XACRO, and MuJoCo simulation model checks PASSED successfully!\033[0m\n")
        return 0
    else:
        print("\033[91m✗ Some simulation model checks FAILED. See details above.\033[0m\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
