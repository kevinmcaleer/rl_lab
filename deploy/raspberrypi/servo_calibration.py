#!/usr/bin/env python3
"""deploy/raspberrypi/servo_calibration.py -- Interactive on-device servo calibration.

Steps through each of the 4 Buddy Jr joints (base_yaw, shoulder_pitch,
elbow_pitch, camera_tilt), lets the user nudge each servo to find its
mechanical centre and limits via keyboard +/-, then saves a JSON calibration
file consumable by ``rl_lab.robot.servo_map.ServoMap.from_file``.

Key bindings (per servo)
------------------------
    +  /  =     nudge servo +1 deg
    -           nudge servo -1 deg
    ]           nudge servo +5 deg (coarse)
    [           nudge servo -5 deg (coarse)
    s           mark this position as the mechanical centre (zero)
    r           reset servo to 90 deg (electrical centre)
    f           flip/invert sign for this joint
    n           accept current position and move to next joint
    q / ESC     quit without saving
    ?           show help

After all joints are stepped through, the tool computes:
    offset_deg[i] = current_deg[i] - 90          (shift from electrical centre)
    sign[i]       = +1 or -1  (user-set via 'f' key)
and writes the JSON calibration file.

Dry-run mode
------------
    python servo_calibration.py --dry-run

No hardware is touched; the tool prints what it *would* command so the flow
can be tested on macOS / CI without a PCA9685 board.

Usage
-----
    python servo_calibration.py [--output PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import termios
import tty
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rl_lab.robot.buddy_jr import (
    JOINT_NAMES,
    NUM_JOINTS,
    SERVO_MAX_DEG,
    SERVO_MIN_DEG,
    SERVO_ZERO_DEG,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT: str = "servo_calibration.json"
_FINE_STEP_DEG: float = 1.0
_COARSE_STEP_DEG: float = 5.0

# PCA9685 default channels match buddy_jr.JOINTS ordering.
_DEFAULT_CHANNELS: tuple[int, ...] = (0, 1, 2, 3)

# ---------------------------------------------------------------------------
# Hardware interface (lazily imported)
# ---------------------------------------------------------------------------


def _make_kit() -> object:
    """Lazily import and construct adafruit_servokit.ServoKit.

    Raises ImportError (with a helpful message) if the adafruit library is
    absent — i.e. we are not on a Raspberry Pi.
    """
    try:
        from adafruit_servokit import ServoKit  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "adafruit-circuitpython-servokit is not installed. "
            "Run: pip install adafruit-circuitpython-servokit\n"
            "Or use --dry-run to run without hardware."
        ) from exc
    return ServoKit(channels=16)


def _set_servo(kit: object | None, channel: int, angle_deg: float, *, dry_run: bool) -> None:
    """Command one servo to *angle_deg*.

    Always clamps to [SERVO_MIN_DEG, SERVO_MAX_DEG] before sending. In
    dry-run mode the command is printed instead of sent to hardware.

    Parameters
    ----------
    kit:
        The ServoKit instance (None is accepted so dry_run callers do not
        need a kit object).
    channel:
        PCA9685 channel index [0, 15].
    angle_deg:
        Desired servo angle in degrees.
    dry_run:
        When True print the command; when False send to hardware.
    """
    clamped = float(np.clip(angle_deg, SERVO_MIN_DEG, SERVO_MAX_DEG))
    if angle_deg != clamped:
        print(
            f"  [WARN] Requested {angle_deg:.1f} deg clamped to "
            f"{clamped:.1f} deg (hardware limit [{SERVO_MIN_DEG:.0f}, {SERVO_MAX_DEG:.0f}])."
        )
    if dry_run:
        print(f"  [DRY-RUN] ch{channel} -> {clamped:.1f} deg")
    else:
        if kit is None:
            raise RuntimeError("BUG: kit is None in live mode.")
        # adafruit_servokit.Servo.angle property expects a float in [0, 180].
        kit.servo[channel].angle = clamped  # type: ignore[index]


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------


def _read_raw_key() -> str:
    """Read one character from stdin in raw (non-canonical) mode.

    Returns the character as a str; ESC is returned as ``'\\x1b'``.
    Only called when stdin is a TTY.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def _print_status(
    joint_name: str,
    channel: int,
    current_deg: float,
    centre_deg: float,
    sign: int,
    *,
    dry_run: bool,
) -> None:
    """Print current calibration state for the active joint."""
    offset = current_deg - centre_deg
    tag = "[DRY-RUN] " if dry_run else ""
    print(
        f"  {tag}"
        f"Joint: {joint_name:<18} ch={channel}  "
        f"pos={current_deg:6.1f} deg  "
        f"centre={centre_deg:6.1f} deg  "
        f"offset={offset:+6.1f} deg  "
        f"sign={sign:+d}"
    )


def _print_help() -> None:
    """Print the key-binding cheat sheet."""
    print()
    print("  Key bindings:")
    print("    + / =       nudge +1 deg (fine)")
    print("    -           nudge -1 deg (fine)")
    print("    ]           nudge +5 deg (coarse)")
    print("    [           nudge -5 deg (coarse)")
    print("    s           mark this position as the mechanical centre (zero)")
    print("    r           reset servo to 90 deg (electrical centre)")
    print("    f           flip/invert sign for this joint")
    print("    n           accept and move to next joint")
    print("    q / ESC     quit without saving")
    print("    ?           print this help")
    print()


# ---------------------------------------------------------------------------
# Per-joint calibration loop
# ---------------------------------------------------------------------------


def _calibrate_joint_tty(
    joint_name: str,
    channel: int,
    kit: object | None,
    *,
    dry_run: bool,
) -> tuple[float, int] | None:
    """Interactively calibrate one joint in raw-TTY mode.

    Returns ``(offset_deg, sign)`` where:
    * ``offset_deg = centre_deg - SERVO_ZERO_DEG``
    * ``sign`` is +1 or -1 (inverted if the user pressed 'f').

    Returns ``None`` if the user pressed ESC / 'q' (abort).
    """
    # Start at the electrical centre.
    current_deg: float = SERVO_ZERO_DEG
    centre_deg: float = SERVO_ZERO_DEG
    sign: int = 1

    print()
    print(f"  === Joint {joint_name} (channel {channel}) ===")
    print(
        "  Move the servo to its mechanical zero (the position you want the "
        "robot to treat as 0 rad).\n"
        "  Press 's' to mark the mechanical centre, then 'n' to confirm."
    )
    _print_help()

    # Command the servo to its starting position.
    _set_servo(kit, channel, current_deg, dry_run=dry_run)
    _print_status(joint_name, channel, current_deg, centre_deg, sign, dry_run=dry_run)

    while True:
        key = _read_raw_key()

        if key in ("\x1b", "q"):
            print("\n  Aborted by user.")
            return None

        elif key in ("+", "="):
            current_deg = float(np.clip(current_deg + _FINE_STEP_DEG, SERVO_MIN_DEG, SERVO_MAX_DEG))
            _set_servo(kit, channel, current_deg, dry_run=dry_run)

        elif key == "-":
            current_deg = float(np.clip(current_deg - _FINE_STEP_DEG, SERVO_MIN_DEG, SERVO_MAX_DEG))
            _set_servo(kit, channel, current_deg, dry_run=dry_run)

        elif key == "]":
            current_deg = float(
                np.clip(current_deg + _COARSE_STEP_DEG, SERVO_MIN_DEG, SERVO_MAX_DEG)
            )
            _set_servo(kit, channel, current_deg, dry_run=dry_run)

        elif key == "[":
            current_deg = float(
                np.clip(current_deg - _COARSE_STEP_DEG, SERVO_MIN_DEG, SERVO_MAX_DEG)
            )
            _set_servo(kit, channel, current_deg, dry_run=dry_run)

        elif key == "s":
            centre_deg = current_deg
            print(f"  >> Centre marked at {centre_deg:.1f} deg")

        elif key == "r":
            current_deg = SERVO_ZERO_DEG
            _set_servo(kit, channel, current_deg, dry_run=dry_run)
            print(f"  >> Reset to electrical centre ({SERVO_ZERO_DEG:.0f} deg)")

        elif key == "f":
            sign *= -1
            print(f"  >> Sign flipped to {sign:+d}")

        elif key == "n":
            offset_deg = centre_deg - SERVO_ZERO_DEG
            print(
                f"  >> Accepted: offset={offset_deg:+.1f} deg  sign={sign:+d}"
                f"  (centre_deg={centre_deg:.1f})"
            )
            return (offset_deg, sign)

        elif key == "?":
            _print_help()

        # Silently ignore any other key.
        _print_status(joint_name, channel, current_deg, centre_deg, sign, dry_run=dry_run)


def _calibrate_joint_repl(
    joint_name: str,
    channel: int,
    kit: object | None,
    joint_index: int,
    total_joints: int,
    *,
    dry_run: bool,
) -> tuple[float, int]:
    """Simulated calibration in non-TTY / dry-run REPL mode.

    In REPL mode (e.g. CI, piped stdin) we cannot read raw key-presses.
    This function prints what the interactive session would do and returns
    neutral calibration values (offset=0, sign=+1) immediately.

    Parameters
    ----------
    joint_name:
        Name of the joint being calibrated.
    channel:
        PCA9685 channel for this joint.
    kit:
        ServoKit instance (may be None in dry_run mode).
    joint_index:
        Zero-based index of the joint in the calibration sequence.
    total_joints:
        Total number of joints to calibrate.
    dry_run:
        When True hardware commands are printed only.
    """
    print()
    print(
        f"  [REPL/SIMULATED] Joint {joint_index + 1}/{total_joints}: "
        f"{joint_name} (channel {channel})"
    )
    print(
        "  stdin is not a TTY — skipping interactive calibration. "
        "Returning neutral offset=0 deg, sign=+1."
    )
    _set_servo(kit, channel, SERVO_ZERO_DEG, dry_run=dry_run)
    _print_status(joint_name, channel, SERVO_ZERO_DEG, SERVO_ZERO_DEG, 1, dry_run=dry_run)
    return (0.0, 1)


# ---------------------------------------------------------------------------
# Save / display calibration
# ---------------------------------------------------------------------------


def _save_calibration(
    output_path: str,
    signs: list[int],
    offsets_deg: list[float],
    channels: list[int],
    *,
    dry_run: bool,
) -> None:
    """Write (or print in dry-run) the JSON calibration file.

    The file format matches what ``ServoMap.from_file`` expects::

        {
            "signs":       [1, 1, 1, 1],
            "offsets_deg": [0.0, 2.5, -1.0, 0.0],
            "channels":    [0, 1, 2, 3]
        }
    """
    calibration: dict[str, list[int] | list[float]] = {
        "signs": signs,
        "offsets_deg": offsets_deg,
        "channels": channels,
    }
    payload = json.dumps(calibration, indent=2)

    if dry_run:
        print()
        print("  [DRY-RUN] Would write calibration to:", output_path)
        print(payload)
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        print()
        print(f"  Calibration saved to: {path.resolve()}")
        print(payload)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="servo_calibration",
        description=(
            "Interactive on-device servo calibration for Buddy Jr. "
            "Steps through each joint, lets you nudge each servo to its "
            "mechanical centre via keyboard +/-, then saves a JSON calibration "
            "file consumable by ServoMap.from_file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Key bindings: +/= nudge +1 deg  - nudge -1 deg  ] coarse+5  [ coarse-5\n"
            "              s mark centre  r reset to 90  f flip sign  n next  q/ESC quit"
        ),
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"Path to write the JSON calibration file (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not command hardware; print servo commands to stdout instead.",
    )
    parser.add_argument(
        "--channels",
        nargs=NUM_JOINTS,
        type=int,
        default=list(_DEFAULT_CHANNELS),
        metavar="CH",
        help=(
            f"PCA9685 channel for each joint in order {list(JOINT_NAMES)} "
            f"(default: {list(_DEFAULT_CHANNELS)})."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the interactive servo calibration wizard.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error or user abort.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    dry_run: bool = args.dry_run
    channels: list[int] = args.channels
    output_path: str = args.output

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print()
    print("  ==========================================================")
    print("   Buddy Jr — Servo Calibration Wizard")
    print("  ==========================================================")
    if dry_run:
        print("  MODE: DRY-RUN (no hardware will be commanded)")
    else:
        print("  MODE: LIVE (commanding PCA9685 via I2C)")
    print(f"  Joints   : {', '.join(JOINT_NAMES)}")
    print(f"  Channels : {channels}")
    print(f"  Output   : {output_path}")
    print("  ----------------------------------------------------------")
    print()

    # ------------------------------------------------------------------
    # Initialise hardware (live mode only)
    # ------------------------------------------------------------------
    kit: object | None = None
    if not dry_run:
        try:
            kit = _make_kit()
            print("  ServoKit (PCA9685) initialised successfully.")
        except ImportError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: could not initialise ServoKit: {exc}", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # Determine interaction mode
    # ------------------------------------------------------------------
    is_tty = sys.stdin.isatty()

    # ------------------------------------------------------------------
    # Step through each joint
    # ------------------------------------------------------------------
    signs: list[int] = []
    offsets_deg: list[float] = []

    for i, joint_name in enumerate(JOINT_NAMES):
        channel = channels[i]

        if is_tty:
            result = _calibrate_joint_tty(joint_name, channel, kit, dry_run=dry_run)
            if result is None:
                # User pressed ESC or 'q' — abort without saving.
                print("  Calibration aborted. No file was written.")
                return 1
            offset, sign = result
        else:
            offset, sign = _calibrate_joint_repl(
                joint_name,
                channel,
                kit,
                joint_index=i,
                total_joints=NUM_JOINTS,
                dry_run=dry_run,
            )

        offsets_deg.append(float(offset))
        signs.append(int(sign))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("  ===== Calibration Summary =====")
    for i, name in enumerate(JOINT_NAMES):
        print(
            f"  {name:<18}  channel={channels[i]}  "
            f"sign={signs[i]:+d}  offset={offsets_deg[i]:+.1f} deg"
        )
    print()

    # ------------------------------------------------------------------
    # Write calibration file
    # ------------------------------------------------------------------
    try:
        _save_calibration(output_path, signs, offsets_deg, channels, dry_run=dry_run)
    except OSError as exc:
        print(f"  ERROR: could not write calibration file: {exc}", file=sys.stderr)
        return 1

    print()
    print("  Calibration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
