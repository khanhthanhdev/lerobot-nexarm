"""Manual episode collection with a small terminal dashboard."""

import importlib.util
import sys
import time
from collections.abc import Callable

from lerobot.utils.keyboard_input import TerminalKeyListener, create_key_listener

# Ignore a second press this soon after the last one, so key repeat cannot skip a phase.
KEY_DEBOUNCE_S = 0.5
# The stdin key reader needs termios, which Windows lacks.
HAS_TERMIOS = importlib.util.find_spec("termios") is not None

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


def init_manual_keyboard_listener():
    """Keyboard listener for ``run_manual_session``: returns ``(listener, events)`` like LeRobot's.

    Enter or Right/n ends the current phase, Left/r discards the episode being recorded and Esc/q
    finishes. Keys are silent, ignored while ``events["controls_enabled"]`` is false (during saving)
    and debounced. A terminal reads keys from stdin when it can; otherwise LeRobot picks the backend.
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "controls_enabled": False,
    }
    last_press = 0.0

    def on_key(name: str) -> None:
        nonlocal last_press
        key = name.lower()
        if key in ("esc", "q"):
            events["stop_recording"] = True
            events["exit_early"] = True
            return
        if key not in ("enter", "right", "n", "left", "r"):
            return
        if key in ("left", "r") and not events.get("allow_rerecord", False):
            return
        now = time.monotonic()
        if not events["controls_enabled"] or now - last_press < KEY_DEBOUNCE_S:
            return
        last_press = now
        if key in ("left", "r"):
            events["rerecord_episode"] = True
        events["exit_early"] = True

    if sys.stdin.isatty() and HAS_TERMIOS:
        listener = TerminalKeyListener(on_key)
        listener.start()
        return listener, events
    controls = "Enter/Right/n: start or stop, Left/r: discard, Esc/q: finish"
    return create_key_listener(on_key, controls_help=controls), events


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
