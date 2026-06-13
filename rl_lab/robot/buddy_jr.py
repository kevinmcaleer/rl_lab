"""Buddy Jr hardware truth — the single source of kinematic constants.

Everything that needs to know "how big is the arm" or "which servo is which"
imports it from here, so there is exactly one place to change if the robot
changes. The numbers match ``urdf/buddy_jr.urdf`` exactly (SI units: metres,
radians) so the analytic kinematics in :mod:`rl_lab.robot.kinematics` agree with
whatever the physics backend reports.

Real hardware: 4x SG90 servos driven by a PCA9685 PWM board (channels 0-3) from
a Raspberry Pi. The sim works in radians about each joint's zero; the servos
work in degrees 0-180. The mapping is documented in :data:`SERVO_ZERO_DEG`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees, pi
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Link geometry (metres) — matches the URDF joint origins exactly.
# --------------------------------------------------------------------------- #
# Both arm segments are 80 mm (the blog's SHOLDER_LENGTH / ELBOW_LENGTH = 80).
SHOULDER_LENGTH: float = 0.08
ELBOW_LENGTH: float = 0.08

# Height of the shoulder-pitch axis above base_link: the base_yaw origin
# (z=0.025) plus the shoulder_pitch origin (z=0.025).
BASE_HEIGHT: float = 0.05

# Fixed offset from the camera_tilt axis (camera_mount frame) to the camera_link
# origin = the point we treat as the end-effector ("camera tip").
CAMERA_OFFSET: tuple[float, float, float] = (0.0145, 0.0, 0.010)

# --------------------------------------------------------------------------- #
# Joint map — ordered exactly as the action/observation vectors and the URDF.
# --------------------------------------------------------------------------- #
JOINT_LIMIT: float = pi / 2  # +/-90 deg, the SG90 usable span used in the URDF


@dataclass(frozen=True)
class Joint:
    """One actuated joint: its place in the chain and on the real robot."""

    index: int  # position in the q vector / URDF revolute order
    name: str  # URDF joint name
    axis: str  # rotation axis in its parent frame: 'x', 'y' or 'z'
    pca9685_channel: int  # servo channel on the PCA9685 board
    lower: float = -JOINT_LIMIT
    upper: float = JOINT_LIMIT


JOINTS: tuple[Joint, ...] = (
    Joint(0, "base_yaw", "z", 0),
    Joint(1, "shoulder_pitch", "y", 1),
    Joint(2, "elbow_pitch", "y", 2),
    Joint(3, "camera_tilt", "y", 3),
)
NUM_JOINTS: int = len(JOINTS)
JOINT_NAMES: tuple[str, ...] = tuple(j.name for j in JOINTS)

# (lower, upper) per joint, shape (4, 2) — handy for clamping and Gym spaces.
JOINT_LIMITS: np.ndarray = np.array([[j.lower, j.upper] for j in JOINTS], dtype=np.float64)

# Name of the URDF link we treat as the end-effector / camera tip.
END_EFFECTOR_LINK: str = "camera_link"


def urdf_path() -> Path:
    """Absolute path to the packaged Buddy Jr URDF (``urdf/buddy_jr.urdf``).

    Lives here, in the engine-agnostic robot module, so both the physics
    backends and the visualization can find the model without either importing
    the other.
    """
    return Path(__file__).resolve().parents[2] / "urdf" / "buddy_jr.urdf"


# --------------------------------------------------------------------------- #
# sim2real servo mapping.
# --------------------------------------------------------------------------- #
# A joint angle of 0 rad maps to the servo's centre (90 deg); +/-90 deg of
# servo travel covers +/-pi/2 rad. So:  servo_deg = degrees(theta) + 90, clamped
# to [0, 180].
SERVO_ZERO_DEG: float = 90.0
SERVO_MIN_DEG: float = 0.0
SERVO_MAX_DEG: float = 180.0


def clamp_to_limits(q: np.ndarray) -> np.ndarray:
    """Clamp a joint vector to the per-joint URDF limits."""
    q = np.asarray(q, dtype=np.float64)
    return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def within_limits(q: np.ndarray, tol: float = 1e-6) -> bool:
    """True if every joint angle is inside its limit (within ``tol``)."""
    q = np.asarray(q, dtype=np.float64)
    return bool(np.all(q >= JOINT_LIMITS[:, 0] - tol) and np.all(q <= JOINT_LIMITS[:, 1] + tol))


def radians_to_servo_degrees(q: np.ndarray) -> np.ndarray:
    """Map joint angles (rad) to SG90 servo commands (deg), clamped to [0, 180].

    This is the bridge to the real robot: ``servo_deg = degrees(theta) + 90``.
    """
    q = np.asarray(q, dtype=np.float64)
    deg = np.degrees(q) + SERVO_ZERO_DEG
    return np.clip(deg, SERVO_MIN_DEG, SERVO_MAX_DEG)


def servo_degrees_to_radians(deg: np.ndarray) -> np.ndarray:
    """Inverse of :func:`radians_to_servo_degrees` (deg -> rad)."""
    deg = np.asarray(deg, dtype=np.float64)
    return np.radians(deg - SERVO_ZERO_DEG)


# Convenience for the docs/tests: a quick textual summary of the mapping.
def describe_servo_mapping() -> str:
    """One-line human-readable description of the rad<->servo mapping."""
    return (
        f"servo_deg = degrees(theta) + {SERVO_ZERO_DEG:.0f}, "
        f"clamped to [{SERVO_MIN_DEG:.0f}, {SERVO_MAX_DEG:.0f}]; "
        f"e.g. theta=0 -> {SERVO_ZERO_DEG:.0f} deg, "
        f"theta=+pi/2 -> {degrees(JOINT_LIMIT) + SERVO_ZERO_DEG:.0f} deg"
    )
