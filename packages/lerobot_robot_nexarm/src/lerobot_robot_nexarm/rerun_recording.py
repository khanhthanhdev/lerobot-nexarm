"""Save the Rerun stream of a NexArm session to a local ``.rrd`` file while it is also shown live.

``init_visualization`` takes the arguments of LeRobot's function of the same name plus ``rerun_save_path``.
``lerobot-nexarm-rollout`` is ``lerobot-rollout`` with an extra ``--rerun_save_path=<file.rrd>`` flag.
"""

import functools
import sys
from pathlib import Path

from lerobot.utils import rerun_visualization, visualization_utils

# The port rr.spawn() starts the local viewer on.
SPAWNED_VIEWER_PORT = 9876


def init_rerun(
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
    save_path: str | Path | None = None,
) -> None:
    """LeRobot's ``init_rerun``, then, with ``save_path``, a copy of everything logged in that file."""
    rerun_visualization.init_rerun(session_name=session_name, ip=ip, port=port)
    if save_path is None:
        return
    import rerun as rr

    # A FileSink alone would replace the viewer connection, so set both sinks.
    url = (
        f"rerun+http://{ip}:{port}/proxy"
        if ip and port
        else f"rerun+http://127.0.0.1:{SPAWNED_VIEWER_PORT}/proxy"
    )
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.set_sinks(rr.GrpcSink(url), rr.FileSink(str(output_path)))


def init_visualization(
    display_mode: str = "rerun",
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
    rerun_save_path: str | Path | None = None,
) -> None:
    """LeRobot's ``init_visualization``; ``rerun_save_path`` needs ``display_mode="rerun"``."""
    if rerun_save_path is None:
        visualization_utils.init_visualization(display_mode, session_name=session_name, ip=ip, port=port)
    elif display_mode != "rerun":
        raise ValueError("rerun_save_path requires display_mode='rerun'.")
    else:
        init_rerun(session_name=session_name, ip=ip, port=port, save_path=rerun_save_path)


def split_save_path(argv: list[str]) -> tuple[list[str], str | None]:
    """Remove ``--rerun_save_path=<file>`` from LeRobot CLI arguments and return it."""
    flag = "--rerun_save_path="
    paths = [arg.removeprefix(flag) for arg in argv if arg.startswith(flag)]
    if len(paths) > 1:
        raise ValueError("--rerun_save_path was given more than once.")
    if paths and "--display_data=true" not in (arg.lower() for arg in argv):
        raise ValueError("--rerun_save_path requires --display_data=true.")
    return [arg for arg in argv if not arg.startswith(flag)], paths[0] if paths else None


def rollout_main() -> None:
    """``lerobot-rollout`` that also accepts ``--rerun_save_path``."""
    from lerobot.scripts import lerobot_rollout

    argv, save_path = split_save_path(sys.argv[1:])
    if save_path is not None:
        # lerobot-rollout calls init_visualization by its module-level name.
        setattr(  # noqa: B010
            lerobot_rollout,
            "init_visualization",
            functools.partial(init_visualization, rerun_save_path=save_path),
        )
    sys.argv = [sys.argv[0], *argv]
    lerobot_rollout.main()


if __name__ == "__main__":
    rollout_main()
