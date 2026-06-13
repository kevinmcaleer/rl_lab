"""Forward and inverse kinematics for the Buddy Jr arm.

Two things live here:

* :func:`forward` — exact forward kinematics built as a product of the URDF
  joint transforms, so it agrees with whatever the physics backend reports for
  the camera-tip pose.
* :func:`inverse` — a classical **law-of-cosines** inverse kinematics solver,
  the same approach the Buddy Jr blog firmware uses to aim the camera. A lesson
  later puts this analytic solver head-to-head with a learned RL policy.

Everything is SI (metres, radians). The joint vector ``q`` is ordered
``[base_yaw, shoulder_pitch, elbow_pitch, camera_tilt]`` to match
:data:`rl_lab.robot.buddy_jr.JOINTS`.

Geometry recap (see ``urdf/buddy_jr.urdf``): the base yaws about Z; the rest is
a planar 2-link arm in the yawed vertical plane — ``L1`` shoulder->elbow,
``L2`` elbow->wrist — followed by a small fixed camera offset. With the camera
tilt held fixed, the elbow->tip segment is rigid, so the position problem is a
standard 2R arm with an offset wrist.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, sin

import numpy as np

from rl_lab.robot import buddy_jr as bj

# --------------------------------------------------------------------------- #
# Kinematic chain — (parent, child, translation, axis, q_index). Mirrors the
# URDF joint origins exactly. A None axis/index means a fixed joint.
# --------------------------------------------------------------------------- #
_CHAIN: tuple[tuple[str, str, tuple[float, float, float], str | None, int | None], ...] = (
    ("base_link", "shoulder_bracket", (0.0, 0.0, 0.025), "z", 0),
    ("shoulder_bracket", "upper_arm", (0.0, 0.0, 0.025), "y", 1),
    ("upper_arm", "forearm", (0.0, 0.0, 0.080), "y", 2),
    ("forearm", "camera_mount", (0.0, 0.0, 0.080), "y", 3),
    ("camera_mount", "camera_link", bj.CAMERA_OFFSET, None, None),
)

_TOL = 1e-9


class UnreachableError(ValueError):
    """Raised when a target cannot be reached within the arm's limits.

    ``reason`` is ``"out_of_reach"`` (geometrically too far / too close) or
    ``"joint_limits"`` (geometrically reachable but only by exceeding a limit).
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Pose:
    """A rigid pose: position (xyz, metres) and orientation (xyzw quaternion)."""

    position: np.ndarray  # shape (3,)
    orientation: np.ndarray  # shape (4,), [x, y, z, w]


# --------------------------------------------------------------------------- #
# Small SE(3) helpers (no external transform library needed).
# --------------------------------------------------------------------------- #
def _rot(axis: str | None, angle: float) -> np.ndarray:
    """3x3 rotation about a principal axis ('x'|'y'|'z'); identity if None."""
    if axis is None:
        return np.eye(3)
    c, s = cos(angle), sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"unknown axis {axis!r}")


def _homogeneous(translation: tuple[float, float, float], rotation: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a translation then a 3x3 rotation."""
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def _mat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an [x, y, z, w] quaternion."""
    m = rot
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #
def _chain_transforms(q: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """Per-joint *local* transforms (parent->child) down the chain."""
    out = []
    for parent, child, trans, axis, idx in _CHAIN:
        angle = float(q[idx]) if idx is not None else 0.0
        out.append((parent, child, _homogeneous(trans, _rot(axis, angle))))
    return out


def forward(q: np.ndarray) -> Pose:
    """Forward kinematics: joint angles -> camera-tip (``camera_link``) pose."""
    q = np.asarray(q, dtype=np.float64)
    t = np.eye(4)
    for _, _, local in _chain_transforms(q):
        t = t @ local
    return Pose(position=t[:3, 3].copy(), orientation=_mat_to_quat(t[:3, :3]))


def forward_position(q: np.ndarray) -> np.ndarray:
    """Just the camera-tip position (shape ``(3,)``) — a thin convenience."""
    return forward(q).position


def joint_transforms(q: np.ndarray) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Local (parent, child, translation, quaternion) for every frame.

    This is what the Foxglove URDF publisher streams as ``FrameTransform``s so
    the arm animates — and it is pure FK, so the visualization works even where
    a physics engine cannot be installed (e.g. macOS without PyBullet).
    """
    result = []
    for parent, child, local in _chain_transforms(q):
        result.append((parent, child, local[:3, 3].copy(), _mat_to_quat(local[:3, :3])))
    return result


def link_world_poses(q: np.ndarray) -> dict[str, Pose]:
    """World pose of every link frame (handy for tests and debugging)."""
    q = np.asarray(q, dtype=np.float64)
    poses: dict[str, Pose] = {"base_link": Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))}
    t = np.eye(4)
    for _parent, child, local in _chain_transforms(q):
        t = t @ local
        poses[child] = Pose(t[:3, 3].copy(), _mat_to_quat(t[:3, :3]))
    return poses


# --------------------------------------------------------------------------- #
# Inverse kinematics (law of cosines)
# --------------------------------------------------------------------------- #
def _wrist_segment(camera_tilt: float) -> tuple[float, float]:
    """Elbow->tip vector (in the forearm plane) for a fixed camera tilt.

    With ``camera_tilt`` held fixed, the elbow->tip segment is rigid: it is the
    80 mm elbow->wrist link plus the rotated camera offset. Returns ``(L2,
    phi)`` — its length and its angle off the forearm's +Z axis.
    """
    ox, _, oz = bj.CAMERA_OFFSET
    vx = ox * cos(camera_tilt) + oz * sin(camera_tilt)
    vz = bj.ELBOW_LENGTH - ox * sin(camera_tilt) + oz * cos(camera_tilt)
    return hypot(vx, vz), atan2(vx, vz)


def is_reachable(target: np.ndarray, camera_tilt: float = 0.0) -> bool:
    """True if :func:`inverse` would find a limit-valid solution for ``target``."""
    try:
        inverse(target, camera_tilt=camera_tilt)
        return True
    except UnreachableError:
        return False


def inverse(target: np.ndarray, camera_tilt: float = 0.0) -> np.ndarray:
    """Inverse kinematics: a target camera-tip position -> joint vector ``q``.

    Uses the classical law of cosines on the planar 2-link sub-arm. The
    ``camera_tilt`` (q3) is a free DOF and is held at the given value (default
    level); the solver returns the ``[base_yaw, shoulder, elbow, tilt]`` that
    places the camera tip at ``target``.

    Raises :class:`UnreachableError` if the target is out of reach or can only
    be reached by violating a joint limit (both elbow solutions are tried).
    """
    x, y, z = (float(v) for v in np.asarray(target, dtype=np.float64))

    l1 = bj.SHOULDER_LENGTH
    l2, phi = _wrist_segment(camera_tilt)
    r = hypot(x, y)
    dz = z - bj.BASE_HEIGHT
    half_pi = bj.JOINT_LIMIT

    # 1. Base-yaw candidates. The arm plane can point *at* the target
    #    (q0 = azimuth, in-plane reach +r) or, by folding the arm backward,
    #    point away from it (q0 = azimuth +/- pi wrapped into +/-90 deg,
    #    in-plane reach -r). The fold lets the arm reach behind itself without
    #    exceeding the +/-90 deg yaw limit.
    candidates: list[tuple[float, float]] = []
    if r <= _TOL:
        candidates.append((0.0, 0.0))  # target on the vertical axis: yaw is free
    else:
        theta = atan2(y, x)
        if abs(theta) <= half_pi + _TOL:
            candidates.append((theta, r))
        for wrapped in (theta - np.pi, theta + np.pi):
            if -half_pi - _TOL <= wrapped <= half_pi + _TOL:
                candidates.append((wrapped, -r))
                break

    # 2. Reachability of the planar 2R sub-arm (independent of dr's sign).
    dist = hypot(r, dz)
    if dist > l1 + l2 + 1e-6 or dist < abs(l1 - l2) - 1e-6:
        raise UnreachableError(
            f"target {target} is geometrically out of reach (|d|={dist:.4f}, "
            f"reach=[{abs(l1 - l2):.4f}, {l1 + l2:.4f}])",
            reason="out_of_reach",
        )

    cos_gamma = float(np.clip((dist * dist - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0))
    gamma_mag = atan2(np.sqrt(max(0.0, 1.0 - cos_gamma * cos_gamma)), cos_gamma)

    # 3. For each yaw candidate try both elbow configurations; return the first
    #    solution that respects every joint limit.
    best: np.ndarray | None = None
    for q0, dr in candidates:
        alpha = atan2(dr, dz)  # in-plane direction to the target, from +Z
        for gamma in (gamma_mag, -gamma_mag):
            beta = atan2(l2 * sin(gamma), l1 + l2 * cos(gamma))
            q = np.array([q0, alpha - beta, gamma - phi, camera_tilt], dtype=np.float64)
            if bj.within_limits(q):
                return q
            if best is None:
                best = q

    raise UnreachableError(
        f"target {target} needs joint angles outside the +/-90 deg limits "
        f"(closest solution {np.round(best, 3) if best is not None else None})",
        reason="joint_limits",
    )
