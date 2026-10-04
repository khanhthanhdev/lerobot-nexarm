"""Manual episode collection with a small terminal dashboard."""

import sys
import time
from collections.abc import Callable

_DIGITS = (
    (" ███ ", "█   █", "█   █", "█   █", " ███ "),
    ("  █  ", " ██  ", "  █  ", "  █  ", " ███ "),
    (" ███ ", "█   █", "  ██ ", " █   ", "█████"),
    ("████ ", "    █", " ███ ", "    █", "████ "),
    ("█  █ ", "█  █ ", "█████", "   █ ", "   █ "),
    ("█████", "█    ", "████ ", "    █", "████ "),
    (" ███ ", "█    ", "████ ", "█   █", " ███ "),
    ("█████", "    █", "   █ ", "  █  ", " █   "),
    (" ███ ", "█   █", " ███ ", "█   █", " ███ "),
    (" ███ ", "█   █", " ████", "    █", " ███ "),
)


def show_recording_status(count: int, status: str, location: str = "") -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    print("EPISODES RECORDED\n")
    digits = [_DIGITS[int(digit)] for digit in str(count)]
    for row in range(5):
        print("  ".join(digit[row] for digit in digits))
    print(f"\n{status}\n")
    if status not in ("SAVING...", "MERGING DATA...", "FINISHED"):
        print("Enter / →: start or stop    ←: discard current episode    Esc / q: finish")
    if location:
        print(f"\nDataset: {location}")
    sys.stdout.flush()


def run_manual_session(
    dataset, events: dict, num_episodes: int, run_phase: Callable, show=show_recording_status
):
    """Record only between deliberate key presses; reset phases never receive a dataset.

    ``run_phase(dataset_or_none)`` runs the ordinary teleoperation loop indefinitely.
    The initial wait holds the arms still until the first start key.
    """

    def phase(status, *, allow_rerecord=False):
        events["controls_enabled"] = False
        events["exit_early"] = False
        events["rerecord_episode"] = False
        events["allow_rerecord"] = allow_rerecord
        show(dataset.num_episodes, status, str(dataset.root))
        events["controls_enabled"] = True

    phase("READY — press Enter / → to start")
    while not events["exit_early"] and not events["stop_recording"]:
        time.sleep(0.02)
    events["exit_early"] = False
    recorded = 0
    while recorded < num_episodes and not events["stop_recording"]:
        phase("COLLECTING...", allow_rerecord=True)
        run_phase(dataset)
        events["controls_enabled"] = False
        # Ctrl+C propagates to the recorder's cleanup. Esc saves a nonempty current episode.
        if events["rerecord_episode"] or not dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        else:
            show(dataset.num_episodes, "SAVING...", str(dataset.root))
            dataset.save_episode()
            recorded += 1
        if events["stop_recording"] or recorded >= num_episodes:
            break
        phase("RESET THE ENVIRONMENT — Enter / → when ready")
        run_phase(None)
    events["controls_enabled"] = False
