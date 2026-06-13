"""Hardware safety layer for the Buddy Jr arm.

Three independent guards, used by both the deploy runner and any other code
that drives real servos:

* :func:`clamp_joint_limits` — enforces URDF joint limits on a raw q vector,
  regardless of what the policy produced.  This is intentionally separate from
  the policy so the constraint is always active.
* :class:`RateLimiter` — caps the per-step change in servo-degree position to
  protect SG90 plastic gears from high-speed impacts.
* :class:`EmergencyStop` — a software e-stop whose ``engaged`` flag is set by
  pressing ESC or Space.  Works with stdlib ``termios``/``tty`` (Linux/macOS),
  falls back to the optional ``keyboard`` third-party package, and is a safe
  no-op when there is no TTY (CI, SSH without a terminal).

All guards are pure stdlib + numpy — no hardware or torch imports here.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
from collections.abc import Sequence

import numpy as np

from rl_lab.robot import buddy_jr as bj

__all__ = [
    "clamp_joint_limits",
    "RateLimiter",
    "EmergencyStop",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Joint-limit clamping
# --------------------------------------------------------------------------- #


def clamp_joint_limits(q_rad: np.ndarray) -> np.ndarray:
    """Clamp ``q_rad`` to the per-joint URDF limits (pi/2 for SG90s).

    Delegates to :func:`rl_lab.robot.buddy_jr.clamp_to_limits` so there is one
    source of truth for the numeric bounds.  The clamping is *independent* of
    the policy: this function is called even when the policy output was already
    within limits, so any upstream bug can never reach the hardware.

    Parameters
    ----------
    q_rad:
        Joint angles in radians, shape ``(4,)`` or broadcastable.

    Returns
    -------
    np.ndarray
        Clamped joint vector, same dtype/shape as the input (float64).
    """
    return bj.clamp_to_limits(q_rad)


# --------------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------------- #


class RateLimiter:
    """Cap the per-step servo-degree change to protect SG90 plastic gears.

    The limiter works in *servo-degree* space (0-180) rather than radians
    because the SG90 datasheet's slew-rate concern is mechanical: the gears
    care about how many degrees the output shaft moves per control loop tick,
    not about the equivalent radian value.

    Typical usage in a deploy loop::

        rl = RateLimiter(max_delta_deg=5.0)
        current_deg = servo_map.to_servo_degrees(home_q)
        rl.reset(current_deg)
        for obs in env:
            q   = policy.predict(obs)
            deg = servo_map.to_servo_degrees(q)
            deg = rl.apply(deg, current_deg)    # slew-rate limited
            drive_servos(deg)
            current_deg = deg

    Parameters
    ----------
    max_delta_deg:
        Maximum degrees any single servo may move in one control step.
        A value around 5–10 degrees is safe for SG90s at typical loop rates
        of 10–20 Hz.
    """

    def __init__(self, max_delta_deg: float = 5.0) -> None:
        if max_delta_deg <= 0:
            raise ValueError(f"max_delta_deg must be positive, got {max_delta_deg}")
        self.max_delta_deg: float = float(max_delta_deg)
        self._current_deg: np.ndarray = np.full(bj.NUM_JOINTS, bj.SERVO_ZERO_DEG, dtype=np.float64)

    # ---------------------------------------------------------------------- #

    def reset(self, current_deg: np.ndarray | Sequence[float]) -> None:
        """Seed the limiter with the servo's *actual* current position.

        Call this once before the control loop starts (or after a pause) so
        the first ``apply`` step has a meaningful baseline to compare against.

        Parameters
        ----------
        current_deg:
            Current servo-degree positions, shape ``(4,)``.
        """
        self._current_deg = np.asarray(current_deg, dtype=np.float64).copy()

    def apply(
        self,
        target_deg: np.ndarray | Sequence[float],
        current_deg: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Return a rate-limited servo-degree command.

        The delta from *current_deg* to *target_deg* is clipped per joint to
        ``+/- max_delta_deg``, then added back to *current_deg*.

        Both *target_deg* and *current_deg* are expected to be in ``[0, 180]``
        (the raw SG90 output range); no additional clamping is performed here
        so the values remain in that range after rate-limiting.

        Parameters
        ----------
        target_deg:
            Desired servo positions (degrees), shape ``(4,)``.
        current_deg:
            Actual present servo positions (degrees), shape ``(4,)``.  Pass the
            same array you drive the hardware with — typically what
            :meth:`reset` was seeded with, updated each step.

        Returns
        -------
        np.ndarray
            Rate-limited servo commands, shape ``(4,)``, dtype float64.
        """
        t: np.ndarray = np.asarray(target_deg, dtype=np.float64)
        c: np.ndarray = np.asarray(current_deg, dtype=np.float64)
        delta = np.clip(t - c, -self.max_delta_deg, self.max_delta_deg)
        limited: np.ndarray = c + delta
        # Keep internal state in sync so callers that omit tracking can rely
        # on apply() returning a consistent result without passing current_deg.
        self._current_deg = limited.copy()
        return limited


# --------------------------------------------------------------------------- #
# Emergency stop
# --------------------------------------------------------------------------- #


class EmergencyStop:
    """Software e-stop: set ``engaged`` to ``True`` on ESC or Space.

    The listener runs in a daemon thread so it never blocks the control loop.
    Calling :meth:`start_keyboard_listener` is a no-op when stdin is not a
    TTY (headless CI, SSH without a PTY), ensuring the code is safe to import
    and use in all environments.

    Usage::

        estop = EmergencyStop()
        estop.start_keyboard_listener()

        for obs in env:
            if estop.engaged:
                safe_halt()
                break
            ...

    The stop can also be triggered programmatically via :meth:`engage`.
    """

    #: ``True`` once the e-stop has been engaged (either via keyboard or code).
    engaged: bool

    def __init__(self) -> None:
        self.engaged = False
        self._listener_thread: threading.Thread | None = None

    # ---------------------------------------------------------------------- #

    def engage(self) -> None:
        """Engage the e-stop immediately (thread-safe).

        After this call ``self.engaged`` is ``True`` and subsequent checks in
        the control loop will halt the robot.
        """
        if not self.engaged:
            log.warning("EmergencyStop engaged.")
        self.engaged = True

    # ---------------------------------------------------------------------- #

    def start_keyboard_listener(self) -> None:
        """Start a background thread that engages the stop on ESC or Space.

        Strategy (in priority order):

        1. If stdin is not a TTY — return immediately (safe no-op for CI/SSH).
        2. Try stdlib ``termios`` + ``tty`` (Linux/macOS).  A raw-mode reader
           thread watches for ESC (``\\x1b``) or Space (``' '``).
        3. Fall back to the optional third-party ``keyboard`` package if
           ``termios`` is unavailable (Windows) — imported lazily so the module
           still works without it.

        The listener thread is a daemon thread (will not block process exit) and
        is started at most once per :class:`EmergencyStop` instance.
        """
        if self._listener_thread is not None and self._listener_thread.is_alive():
            return  # already running

        if not _stdin_is_tty():
            log.debug("EmergencyStop: stdin is not a TTY — keyboard listener skipped.")
            return

        if _try_start_termios_listener(self):
            return

        _try_start_keyboard_listener(self)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _stdin_is_tty() -> bool:
    """Return ``True`` when stdin is an interactive terminal."""
    try:
        return sys.stdin.isatty()
    except Exception:  # noqa: BLE001
        return False


def _try_start_termios_listener(estop: EmergencyStop) -> bool:
    """Attempt to start a raw-mode termios reader thread.

    Returns ``True`` if the thread was started successfully, ``False`` if
    ``termios`` / ``tty`` are unavailable (e.g. on Windows).
    """
    try:
        import termios  # noqa: PLC0415 (lazy import — not available on Windows)
        import tty  # noqa: PLC0415
    except ImportError:
        return False

    def _run() -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not estop.engaged:
                # Read one byte at a time — cbreak/raw mode makes this
                # available immediately without waiting for a newline.
                try:
                    ch = sys.stdin.read(1)
                except Exception:  # noqa: BLE001
                    break
                if ch in ("\x1b", " "):  # ESC = 0x1b, Space = 0x20
                    estop.engage()
                    break
        finally:
            with contextlib.suppress(Exception):
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    t = threading.Thread(target=_run, name="estop-termios", daemon=True)
    t.start()
    estop._listener_thread = t
    log.debug("EmergencyStop: termios keyboard listener started (ESC or Space to stop).")
    return True


def _try_start_keyboard_listener(estop: EmergencyStop) -> None:
    """Fall back to the optional ``keyboard`` package (Windows / no-termios).

    Imported lazily; silently does nothing if the package is not installed.
    """
    try:
        import keyboard as _kb  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        log.debug(
            "EmergencyStop: 'keyboard' package not found — "
            "keyboard listener unavailable.  Call .engage() manually."
        )
        return

    def _on_event(event: object) -> None:  # pragma: no cover
        estop.engage()

    try:
        _kb.on_press_key("esc", _on_event)
        _kb.on_press_key("space", _on_event)
        log.debug("EmergencyStop: 'keyboard' package listener started (ESC or Space to stop).")
    except Exception as exc:  # noqa: BLE001
        log.debug("EmergencyStop: 'keyboard' listener setup failed: %s", exc)
