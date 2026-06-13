#!/usr/bin/env python3
"""jog_joints.py -- Keyboard teleoperation for Buddy Jr via pure FK + Foxglove.

Moves each joint by a small step when you press a key, recomputes the
forward kinematics, and streams the new pose to Foxglove in real time.
No physics engine is required -- everything runs on pure FK, so this works
on macOS even without PyBullet.

Key bindings (interactive/TTY mode)
------------------------------------
    1 / q   -- base_yaw       +step / -step
    2 / w   -- shoulder_pitch +step / -step
    3 / e   -- elbow_pitch    +step / -step
    4 / r   -- camera_tilt    +step / -step
    0       -- reset all joints to zero
    ?       -- print help
    ESC     -- quit

Fall-back REPL (stdin is not a TTY, e.g. piped input)
------------------------------------------------------
    Enter lines of the form:  <joint_index> <value_in_radians>
    Example:  ``1 0.5``  sets shoulder_pitch to 0.5 rad.
    Type ``reset`` to zero all joints, ``quit`` (or EOF) to exit.

Usage
-----
    python scripts/jog_joints.py
    python scripts/jog_joints.py --port 8766 --step 0.05

    # or via the Makefile:
    make jog
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default step size in radians when a key is pressed.
_DEFAULT_STEP_RAD: float = 0.05

# A fixed demo goal that sits inside the arm's reachable workspace.
# x = 0.10 m forward along the world X-axis, z = 0.10 m up from base_link.
_DEMO_GOAL: np.ndarray = np.array([0.10, 0.0, 0.10])

# Joint names, indexed 0-3, for display purposes.
_JOINT_NAMES: tuple[str, ...] = ("base_yaw", "shoulder_pitch", "elbow_pitch", "camera_tilt")

# Map of lowercase keys to (joint_index, sign) tuples.
# Row keys (1-4) increase the joint; Q-R keys decrease it.
_KEY_MAP: dict[str, tuple[int, int]] = {
    "1": (0, +1),
    "q": (0, -1),
    "2": (1, +1),
    "w": (1, -1),
    "3": (2, +1),
    "e": (2, -1),
    "4": (3, +1),
    "r": (3, -1),
}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jog_joints",
        description="Keyboard teleoperation of the Buddy Jr arm via pure FK + Foxglove.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Keys (TTY mode):  1/q=base_yaw  2/w=shoulder_pitch  "
            "3/e=elbow_pitch  4/r=camera_tilt  0=reset  ESC=quit"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Foxglove WebSocket port (default: 8765).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Foxglove WebSocket host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=_DEFAULT_STEP_RAD,
        metavar="RADIANS",
        help=f"Joint step size in radians per key-press (default: {_DEFAULT_STEP_RAD}).",
    )
    return parser


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #


def _print_state(q: np.ndarray, tip: np.ndarray, dist: float) -> None:
    """Print the current joint angles and tip position to stdout."""
    names_str = "  ".join(f"{n}={q[i]:+.3f}" for i, n in enumerate(_JOINT_NAMES))
    print(f"  [{names_str}]  tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f})  dist={dist:.3f} m")


def _print_help() -> None:
    """Print the key-binding cheat sheet."""
    print()
    print("  Key bindings:")
    print("    1 / q   -- base_yaw       +step / -step")
    print("    2 / w   -- shoulder_pitch +step / -step")
    print("    3 / e   -- elbow_pitch    +step / -step")
    print("    4 / r   -- camera_tilt    +step / -step")
    print("    0       -- reset all joints to zero")
    print("    ?       -- print this help")
    print("    ESC     -- quit")
    print()


# --------------------------------------------------------------------------- #
# POSIX raw-TTY reader
# --------------------------------------------------------------------------- #


def _read_raw_key() -> str:
    """Read a single character from stdin in raw (non-canonical) mode.

    On a POSIX system this reads exactly one byte without waiting for Enter.
    Returns the character as a str; ESC is returned as ``"\x1b"``.

    This function is only called when stdin *is* a TTY (checked by the
    caller), so it is safe to use ``termios``/``tty`` here.
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        # Always restore the terminal, even if an exception is raised.
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# --------------------------------------------------------------------------- #
# Interactive (TTY) loop
# --------------------------------------------------------------------------- #


def _tty_loop(
    q: np.ndarray,
    step: float,
    streamer: object,  # FoxgloveStreamer
) -> None:
    """Run the raw-key teleoperation loop on a POSIX TTY.

    Reads single key-presses and adjusts joints until the user presses ESC.
    """
    from rl_lab.robot.buddy_jr import clamp_to_limits
    from rl_lab.robot.kinematics import forward_position
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

    streamer_: FoxgloveStreamer = streamer  # type: ignore[assignment]

    _print_help()
    print("  Jogging Buddy Jr -- press ESC to quit.")
    print()

    while True:
        # Compute and display current state.
        tip = forward_position(q)
        dist = float(np.linalg.norm(tip - _DEMO_GOAL))
        _print_state(q, tip, dist)
        streamer_.publish(joint_q=q, p_ee=tip, g=_DEMO_GOAL, dist=dist, force=True)

        # Block until one key is pressed.
        key = _read_raw_key()

        if key == "\x1b":  # ESC
            print("\n  ESC pressed -- quitting.")
            break
        elif key == "0":
            q = np.zeros(4)
            print("  >> Reset to zero")
        elif key == "?":
            _print_help()
        elif key in _KEY_MAP:
            idx, sign = _KEY_MAP[key]
            q = q.copy()
            q[idx] += sign * step
            q = clamp_to_limits(q)
            print(f"  >> {'+' if sign > 0 else '-'}{_JOINT_NAMES[idx]}")
        # Any other key is silently ignored.


# --------------------------------------------------------------------------- #
# Fall-back REPL (non-TTY / piped input)
# --------------------------------------------------------------------------- #


def _repl_loop(
    q: np.ndarray,
    step: float,  # noqa: ARG001 -- accepted for API symmetry but unused in REPL mode
    streamer: object,  # FoxgloveStreamer
) -> None:
    """Run a simple line-oriented REPL when stdin is not a TTY.

    Accepts lines of the form ``<joint_index> <value_rad>``
    (e.g. ``1 0.5``), ``reset``, or ``quit``.
    This mode is useful for scripted testing or CI.
    """
    from rl_lab.robot.buddy_jr import clamp_to_limits
    from rl_lab.robot.kinematics import forward_position
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

    streamer_: FoxgloveStreamer = streamer  # type: ignore[assignment]

    print("  stdin is not a TTY -- entering REPL mode.")
    print("  Commands: '<index> <value_rad>'  |  'reset'  |  'quit'")
    print()

    # Publish the initial state.
    tip = forward_position(q)
    dist = float(np.linalg.norm(tip - _DEMO_GOAL))
    _print_state(q, tip, dist)
    streamer_.publish(joint_q=q, p_ee=tip, g=_DEMO_GOAL, dist=dist, force=True)

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue  # skip blank / comment lines

            if line in ("quit", "exit", "q"):
                print("  Quitting.")
                break

            if line == "reset":
                q = np.zeros(4)
                print("  >> Reset to zero")
            else:
                parts = line.split()
                if len(parts) != 2:  # noqa: PLR2004
                    print(f"  Unrecognised command: {line!r}")
                    print("  Use: '<index 0-3> <value_in_rad>'  or  'reset'  or  'quit'")
                    continue

                try:
                    idx = int(parts[0])
                    val = float(parts[1])
                except ValueError:
                    print(f"  Could not parse '{line}' -- expected integer index and float value.")
                    continue

                if not (0 <= idx <= 3):  # noqa: PLR2004
                    print(f"  Joint index {idx} out of range [0, 3].")
                    continue

                q = q.copy()
                q[idx] = val
                q = clamp_to_limits(q)
                print(f"  >> Set {_JOINT_NAMES[idx]} = {q[idx]:+.4f} rad")

            # Re-publish after every command.
            tip = forward_position(q)
            dist = float(np.linalg.norm(tip - _DEMO_GOAL))
            _print_state(q, tip, dist)
            streamer_.publish(joint_q=q, p_ee=tip, g=_DEMO_GOAL, dist=dist, force=True)

    except EOFError:
        print("  EOF received -- quitting.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns an exit code (0 = clean, 1 = error)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    from rl_lab.robot.kinematics import forward_position
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

    # Start the Foxglove streamer.
    print(f"Connecting to Foxglove on ws://{args.host}:{args.port} ...")
    try:
        streamer = FoxgloveStreamer(
            render_mode="foxglove",
            host=args.host,
            port=args.port,
            render_fps=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not start Foxglove server: {exc}", file=sys.stderr)
        return 1

    # Initial joint state: all zeros.
    q: np.ndarray = np.zeros(4)

    with streamer:
        ws_url = f"ws://{args.host}:{args.port}"
        print(f"  WebSocket: {ws_url}")
        if streamer.app_url is not None:
            print(f"  Desktop : {streamer.app_url}")
        print(f"  Step size: {args.step:.4f} rad per key-press")
        print()

        # Publish initial state immediately.
        tip = forward_position(q)
        dist = float(np.linalg.norm(tip - _DEMO_GOAL))
        streamer.publish(joint_q=q, p_ee=tip, g=_DEMO_GOAL, dist=dist, force=True)

        try:
            # Choose interactive vs. REPL mode based on whether stdin is a TTY.
            if sys.stdin.isatty():
                _tty_loop(q, args.step, streamer)
            else:
                _repl_loop(q, args.step, streamer)
        except KeyboardInterrupt:
            print("\n  Ctrl-C -- shutting down.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
