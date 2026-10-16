# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""The single-arm sim loads from one asset tree: sim/description/ with meshes in sim/assets/meshes/visual/."""

import shutil
import subprocess  # nosec B404 - runs git only
import xml.etree.ElementTree as ET  # nosec B405 - parses the repo's own URDF
from pathlib import Path

import mujoco
import numpy as np
import pytest

from lerobot.envs.nexarm import DEFAULT_SCENE_PATH
from lerobot.robots.nexarm_sim.config_nexarm_sim import NexArmSimConfig
from lerobot.robots.nexarm_sim.mujoco_backend import resolve_model_path

REPO_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTION = REPO_ROOT / "sim" / "description"
MODELS = [
    DESCRIPTION / "mjcf" / "nexarm.xml",
    DESCRIPTION / "mjcf" / "scene.xml",
    DESCRIPTION / "mjcf" / "bowl_stack_scene.xml",
]
URDF = DESCRIPTION / "urdf" / "nexarm.urdf"


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.name)
def test_model_compiles_from_any_working_directory(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nmesh >= 11


def test_urdf_compiles_from_any_working_directory_and_its_meshes_resolve(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(URDF))  # MuJoCo keeps only the URDF collisions
    assert model.ngeom >= 13
    meshes = [m.get("filename") for m in ET.parse(URDF).iter("mesh")]  # nosec B314
    assert len(meshes) >= 11
    assert all((URDF.parent / mesh).is_file() for mesh in meshes)


def test_env_and_robot_defaults_compile_to_the_same_model():
    env_model = mujoco.MjModel.from_xml_path(str(resolve_model_path(DEFAULT_SCENE_PATH)))
    robot_model = mujoco.MjModel.from_xml_path(str(resolve_model_path(NexArmSimConfig().model_path)))
    for field in ("nq", "nu", "nmesh", "nbody"):
        assert getattr(env_model, field) == getattr(robot_model, field), field
    np.testing.assert_array_equal(env_model.body_mass, robot_model.body_mass)


def test_no_symlinks_under_sim():
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available")
    listing = subprocess.run(  # nosec B603
        [git, "ls-files", "-s", "sim"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    symlinks = [line.split("\t")[-1] for line in listing.splitlines() if line.startswith("120000")]
    assert not symlinks


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.name)
def test_actuator_ctrlrange_lies_inside_the_joint_range(path):
    """A ctrlrange wider than the joint range lets a command push the joint into its limit."""
    model = mujoco.MjModel.from_xml_path(str(path))
    for actuator in range(model.nu):
        joint = model.actuator_trnid[actuator, 0]
        low, high = model.actuator_ctrlrange[actuator]
        jlow, jhigh = model.jnt_range[joint]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        assert model.actuator_ctrllimited[actuator] and model.jnt_limited[joint], name
        assert jlow - 1e-9 <= low < high <= jhigh + 1e-9, name


@pytest.mark.parametrize("fps", [30, 15])
def test_equality_time_constants_survive_the_frame_aligned_timestep(fps):
    """MuJoCo silently raises a time constant below 2 timesteps; keep each written value (near) the one in effect.

    The backend shrinks or stretches the timestep slightly so whole steps span one frame.
    """
    model = mujoco.MjModel.from_xml_path(str(MODELS[0]))
    steps = max(1, round((1.0 / fps) / model.opt.timestep))
    timestep = (1.0 / fps) / steps
    for eq in range(model.neq):
        written = model.eq_solref[eq, 0]
        in_effect = max(written, 2 * timestep)
        assert in_effect / written < 1.05, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq)
