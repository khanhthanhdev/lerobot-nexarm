"""Skip MuJoCo rendering tests on hosts without a working OpenGL context."""

from __future__ import annotations

import functools
import os

import mujoco
import pytest

_PROBE_XML = "<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>"


@functools.cache
def opengl_error() -> str | None:
    """Try to create a tiny ``mujoco.Renderer`` once; return why it failed, or None when rendering works."""
    try:
        renderer = mujoco.Renderer(mujoco.MjModel.from_xml_string(_PROBE_XML), height=8, width=8)
    # PyOpenGL raises AttributeError when the EGL/OSMesa library it was pointed at is missing.
    except (mujoco.FatalError, RuntimeError, ImportError, OSError, AttributeError) as error:
        return f"{type(error).__name__}: {error}"
    renderer.close()
    return None


def require_opengl() -> None:
    """Skip the calling test when MuJoCo cannot render offscreen (set MUJOCO_GL=egl or osmesa).

    With ``NEXARM_REQUIRE_OPENGL=1`` (set in CI) the test fails instead, so a broken GL setup is not hidden.
    """
    error = opengl_error()
    if error is None:
        return
    message = f"MuJoCo offscreen rendering is unavailable ({error})"
    if os.environ.get("NEXARM_REQUIRE_OPENGL") == "1":
        pytest.fail(message)
    pytest.skip(message)
