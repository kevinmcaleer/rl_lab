"""Sim2real servo mapping for Buddy Jr: joint radians <-> PCA9685 servo degrees.

This module is the *only* place that knows how the software joint angles (in
radians, zero-centred, symmetric about the URDF neutral pose) translate into the
hardware servo commands (degrees, 0-180, centre at 90) that the PCA9685 PWM
board sends to the SG90 servos.

**Why a dedicated class and not just :func:`~rl_lab.robot.buddy_jr.radians_to_servo_degrees`?**

The base helper in ``buddy_jr`` assumes a perfect, calibrated robot where every
joint's zero lines up exactly with 90 deg of servo travel and positive rotation
always means increasing servo angle.  Real hardware rarely satisfies either:

* A servo may be physically mounted backwards (sign flip).
* The linkage zero may not fall at servo 90 deg (centre offset).
* You may want to remap joints to non-default PCA9685 channels.

:class:`ServoMap` captures those three per-joint corrections so the rest of the
codebase stays clean: call ``to_servo_degrees`` for simulation -> robot, call
``to_radians`` for robot -> simulation (e.g. when reading back encoder-less
positions).

The calibration can be persisted to / loaded from a JSON file so that robot-
specific settings survive code updates.

Mapping formula (per joint *i*)::

    servo_deg[i] = clamp(
        sign[i] * degrees(q_rad[i]) + 90.0 + offset_deg[i],
        0.0,
        180.0,
    )

Inverse (best-effort; ignores the clamp, so only exact for angles inside limits)::

    q_rad[i] = radians( (servo_deg[i] - 90.0 - offset_deg[i]) / sign[i] )

Hardware-library imports (**adafruit-circuitpython-servokit**, **board**,
**busio**) are intentionally absent.  This file is pure Python (stdlib + numpy +
json) so it imports cleanly on macOS / CI / the sim environment without any
hardware stack installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Re-use the canonical clamping helper and the zero/limit constants so that
# this module stays consistent with the rest of the robot model.  We do NOT
# import the full PCA9685 / servo-kit stack — those are done lazily in the
# deploy layer.
from rl_lab.robot.buddy_jr import (
    NUM_JOINTS,
    SERVO_MAX_DEG,
    SERVO_MIN_DEG,
    SERVO_ZERO_DEG,
    clamp_to_limits,
)

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #
_FloatSeq = tuple[float, ...] | list[float]
_IntSeq = tuple[int, ...] | list[int]


class ServoMap:
    """Per-joint calibration table: sign, centre-offset, and channel assignment.

    Parameters
    ----------
    signs:
        Per-joint sign: +1 if positive joint rotation increases servo angle,
        -1 if it is mounted the other way around.  Must be +1 or -1.
    offsets_deg:
        Per-joint centre offset in servo degrees.  Added *after* the sign flip so
        that a value of +5 means "the servo is 5 deg off centre at joint zero".
    channels:
        PCA9685 channel index for each joint.  Defaults to (0, 1, 2, 3) which is
        the standard Buddy Jr wiring.  Must be 4 distinct ints in [0, 15].

    Joint order throughout: [base_yaw, shoulder_pitch, elbow_pitch, camera_tilt].

    Examples
    --------
    Default (calibrated) map — all joints behave as the URDF says:

    >>> sm = ServoMap()
    >>> import numpy as np
    >>> sm.to_servo_degrees(np.zeros(4))  # neutral -> all 90 deg
    array([90., 90., 90., 90.])

    Shoulder mounted backwards, 2 deg trim on elbow:

    >>> sm2 = ServoMap(signs=(1, -1, 1, 1), offsets_deg=(0, 0, 2, 0))
    >>> import numpy as np; import math
    >>> q = np.array([0.0, math.pi/4, 0.0, 0.0])
    >>> sm2.to_servo_degrees(q)  # shoulder goes down instead of up
    array([90.  , 45.  , 90.  , 90.  ])
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        signs: _FloatSeq = (1, 1, 1, 1),
        offsets_deg: _FloatSeq = (0, 0, 0, 0),
        channels: _IntSeq = (0, 1, 2, 3),
    ) -> None:
        # Validate and store signs — only +1 / -1 are physically meaningful.
        signs_arr = np.asarray(signs, dtype=np.float64)
        if signs_arr.shape != (NUM_JOINTS,):
            raise ValueError(f"signs must have length {NUM_JOINTS}, got {len(signs)}")
        if not np.all(np.abs(signs_arr) == 1.0):
            raise ValueError(f"Every element of signs must be +1 or -1, got {signs}")
        # Store as a read-only view so callers cannot mutate the calibration in
        # place (which would silently break safety guarantees).
        self._signs: np.ndarray = signs_arr
        self._signs.flags.writeable = False

        # Per-joint centre offsets in servo degrees.  Stored as float64 so
        # arithmetic with the angle arrays stays in float.
        offsets_arr = np.asarray(offsets_deg, dtype=np.float64)
        if offsets_arr.shape != (NUM_JOINTS,):
            raise ValueError(f"offsets_deg must have length {NUM_JOINTS}, got {len(offsets_deg)}")
        self._offsets_deg: np.ndarray = offsets_arr
        self._offsets_deg.flags.writeable = False

        # PCA9685 channel assignments.  Must be integers in [0, 15].
        channels_t = tuple(int(c) for c in channels)
        if len(channels_t) != NUM_JOINTS:
            raise ValueError(f"channels must have length {NUM_JOINTS}, got {len(channels)}")
        if any(c < 0 or c > 15 for c in channels_t):
            raise ValueError(f"PCA9685 channels must be in [0, 15], got {channels_t}")
        # Note: duplicate channel numbers are technically allowed in case two
        # joints are driven by the same servo (unusual but not impossible for
        # passive linkages).  We do not enforce uniqueness here.
        self._channels: tuple[int, ...] = channels_t

    # ------------------------------------------------------------------ #
    # Core mapping
    # ------------------------------------------------------------------ #

    def to_servo_degrees(self, q_rad: np.ndarray) -> np.ndarray:
        """Convert joint angles (rad) to PCA9685 servo degrees.

        Formula per joint *i*::

            servo_deg[i] = clamp(
                sign[i] * degrees(q_rad[i]) + SERVO_ZERO_DEG + offset_deg[i],
                SERVO_MIN_DEG,
                SERVO_MAX_DEG,
            )

        The joint-limit clamp is applied *before* converting to degrees so that
        any out-of-range command from the policy is caught at the limit boundary
        rather than at the servo's physical end-stop.  This protects the
        mechanism and keeps the calibration from having to cover beyond-limit
        territory.

        Parameters
        ----------
        q_rad:
            Joint angles in radians, shape ``(4,)``.
            Order: [base_yaw, shoulder_pitch, elbow_pitch, camera_tilt].

        Returns
        -------
        np.ndarray
            Servo commands in degrees, shape ``(4,)``, dtype float64,
            values in ``[0, 180]``.
        """
        # ---- 1. Ensure float64 and correct shape -------------------------
        q = np.asarray(q_rad, dtype=np.float64)
        if q.shape != (NUM_JOINTS,):
            raise ValueError(f"q_rad must have shape ({NUM_JOINTS},), got {q.shape}")

        # ---- 2. Clamp to kinematic limits before mapping ------------------
        # This is a safety gate: if the RL policy or trajectory planner
        # produces a slightly out-of-range command we clip it here rather than
        # letting the servo overshoot into a linkage hard-stop.
        q_clamped = clamp_to_limits(q)

        # ---- 3. Sim2real formula -----------------------------------------
        # Convert radians -> degrees then apply sign flip and centre offset.
        #
        #   servo_deg = sign * deg(theta) + SERVO_ZERO_DEG + offset
        #
        # SERVO_ZERO_DEG (= 90) centres the SG90 at joint angle 0 rad.
        # sign handles servos that are physically mounted in reverse.
        # offset_deg absorbs any mechanical trim (linkage not at true zero).
        servo_deg = self._signs * np.degrees(q_clamped) + SERVO_ZERO_DEG + self._offsets_deg

        # ---- 4. Clamp to the servo's physical range ----------------------
        # SG90 nominally spans 0-180 deg.  Even after joint-limit clamping a
        # large sign inversion + offset could push past the servo limits, so we
        # clamp the output too.
        return np.clip(servo_deg, SERVO_MIN_DEG, SERVO_MAX_DEG)

    def to_radians(self, servo_deg: np.ndarray) -> np.ndarray:
        """Best-effort inverse: servo degrees -> joint angles (rad).

        "Best-effort" because:

        * The forward mapping clamps to ``[0, 180]`` so any angles outside the
          kinematic limits are irreversibly lost.  This function just inverts
          the linear part and does NOT attempt to recover the clamped value.
        * Floating-point round-trips through degrees/radians accumulate small
          errors.

        Use this when reading back an estimated pose from the servo's
        commanded position (not a true encoder reading) — for example to
        initialise the state when the arm boots.

        Formula (per joint *i*)::

            q_rad[i] = radians( (servo_deg[i] - SERVO_ZERO_DEG - offset_deg[i])
                                / sign[i] )

        Parameters
        ----------
        servo_deg:
            Servo positions in degrees, shape ``(4,)``, expected in ``[0, 180]``.

        Returns
        -------
        np.ndarray
            Joint angles in radians, shape ``(4,)``, dtype float64.
        """
        deg = np.asarray(servo_deg, dtype=np.float64)
        if deg.shape != (NUM_JOINTS,):
            raise ValueError(f"servo_deg must have shape ({NUM_JOINTS},), got {deg.shape}")

        # Invert: remove the zero offset and offset_deg, undo the sign flip,
        # then convert back to radians.
        #
        # Dividing by sign is the same as multiplying (sign is +/-1) but we
        # write it as division so the formula mirrors the forward direction.
        return np.radians((deg - SERVO_ZERO_DEG - self._offsets_deg) / self._signs)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def channels(self) -> tuple[int, ...]:
        """PCA9685 channel for each joint (read-only).

        Returns
        -------
        tuple[int, ...]
            Length-4 tuple mapping joint index -> PCA9685 channel.
            E.g. ``(0, 1, 2, 3)`` means base_yaw -> channel 0, etc.
        """
        return self._channels

    @property
    def signs(self) -> np.ndarray:
        """Per-joint sign array, read-only float64 (4,)."""
        return self._signs  # already non-writeable

    @property
    def offsets_deg(self) -> np.ndarray:
        """Per-joint centre offsets in degrees, read-only float64 (4,)."""
        return self._offsets_deg  # already non-writeable

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Write calibration to a JSON file.

        The file format is intentionally simple so that users can hand-edit it
        during a calibration session::

            {
              "signs":       [1, -1, 1, 1],
              "offsets_deg": [0, 0, 2.5, 0],
              "channels":    [0, 1, 2, 3]
            }

        Parameters
        ----------
        path:
            Destination path.  Parent directories must already exist.
        """
        calibration = {
            # Store as plain Python lists so json.dumps produces compact output
            # and the file is both human- and machine-readable.
            "signs": self._signs.tolist(),
            "offsets_deg": self._offsets_deg.tolist(),
            "channels": list(self._channels),
        }
        out = Path(path)
        out.write_text(json.dumps(calibration, indent=2))

    @classmethod
    def from_file(cls, path: str | Path) -> ServoMap:
        """Load calibration from a JSON file produced by :meth:`save`.

        Parameters
        ----------
        path:
            Path to a JSON calibration file.  Missing keys fall back to
            the default values (unit signs, zero offsets, channels 0-3) so
            that older calibration files remain valid after new fields are
            added.

        Returns
        -------
        ServoMap
            Fully initialised map with the stored calibration.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        ValueError
            If any stored value violates the ServoMap invariants (e.g. a sign
            value that is not +/-1).
        """
        data = json.loads(Path(path).read_text())

        # Use .get() with sensible defaults so that a minimal JSON file such as
        # {"offsets_deg": [0, 0, 2.5, 0]} still produces a valid ServoMap.
        signs = data.get("signs", [1, 1, 1, 1])
        offsets_deg = data.get("offsets_deg", [0, 0, 0, 0])
        channels = data.get("channels", [0, 1, 2, 3])

        return cls(
            signs=tuple(signs),
            offsets_deg=tuple(offsets_deg),
            channels=tuple(channels),
        )

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"signs={self._signs.tolist()}, "
            f"offsets_deg={self._offsets_deg.tolist()}, "
            f"channels={list(self._channels)})"
        )

    def __eq__(self, other: object) -> bool:
        """Two ServoMaps are equal if all three calibration tables match."""
        if not isinstance(other, ServoMap):
            return NotImplemented
        return (
            np.array_equal(self._signs, other._signs)
            and np.array_equal(self._offsets_deg, other._offsets_deg)
            and self._channels == other._channels
        )
