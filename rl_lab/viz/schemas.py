"""Builders for the Foxglove messages the lab publishes.

Thin, well-typed helpers around ``foxglove.messages`` so the rest of the viz
code (the bridge, the URDF publisher, the metrics streamer) reads clearly and
there is one place that knows the schema field names.

We publish three kinds of thing:

* **FrameTransform(s)** — one per joint, so Foxglove builds the TF tree and the
  URDF-shaped arm animates as the joints move.
* **SceneUpdate** — the goal sphere (green inside tolerance), the tip marker,
  and a tip->target line.
* **numeric JSON** — scalar metrics (distance, reward, ...) for plot panels.

All geometry is SI (metres, radians); quaternions are ``[x, y, z, w]``.
"""

from __future__ import annotations

import numpy as np
from foxglove.messages import (
    Color,
    CubePrimitive,
    FrameTransform,
    FrameTransforms,
    LinePrimitive,
    LinePrimitiveLineType,
    Point3,
    Pose,
    Quaternion,
    SceneEntity,
    SceneUpdate,
    SpherePrimitive,
    Vector3,
)

# --------------------------------------------------------------------------- #
# Colours (RGBA, 0-1)
# --------------------------------------------------------------------------- #
GREEN = Color(r=0.1, g=0.8, b=0.2, a=1.0)  # target reached (within tolerance)
AMBER = Color(r=0.95, g=0.6, b=0.1, a=1.0)  # target not yet reached
BLUE = Color(r=0.2, g=0.5, b=0.95, a=1.0)  # camera tip marker
GREY = Color(r=0.6, g=0.6, b=0.6, a=1.0)  # arm links
WHITE = Color(r=0.9, g=0.9, b=0.9, a=0.8)  # tip->target line

# JSON schema for the numeric metrics channel (used by the bridge / live_metrics).
METRICS_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "distance": {"type": "number"},
        "reward": {"type": "number"},
        "episode_return": {"type": "number"},
        "success_rate": {"type": "number"},
    },
}


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def frame_transform(
    parent: str, child: str, translation: np.ndarray, quaternion: np.ndarray
) -> FrameTransform:
    """Build one ``FrameTransform`` (``quaternion`` is ``[x, y, z, w]``)."""
    t = np.asarray(translation, dtype=np.float64)
    qx, qy, qz, qw = (float(v) for v in np.asarray(quaternion, dtype=np.float64))
    return FrameTransform(
        parent_frame_id=parent,
        child_frame_id=child,
        translation=Vector3(x=float(t[0]), y=float(t[1]), z=float(t[2])),
        rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )


def frame_transforms(
    transforms: list[tuple[str, str, np.ndarray, np.ndarray]],
) -> FrameTransforms:
    """Build a ``FrameTransforms`` bundle from ``(parent, child, t, quat)`` tuples.

    This is exactly the shape returned by
    :func:`rl_lab.robot.kinematics.joint_transforms`.
    """
    return FrameTransforms(transforms=[frame_transform(*item) for item in transforms])


# --------------------------------------------------------------------------- #
# Scene primitives
# --------------------------------------------------------------------------- #
def _pose(position: np.ndarray, quaternion: np.ndarray | None = None) -> Pose:
    p = np.asarray(position, dtype=np.float64)
    if quaternion is None:
        q = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        q = np.asarray(quaternion, dtype=np.float64)
    return Pose(
        position=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
        orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
    )


def sphere_entity(
    entity_id: str,
    position: np.ndarray,
    diameter: float,
    color: Color,
    frame_id: str = "world",
) -> SceneEntity:
    """A single-sphere entity (e.g. the goal marker or the tip marker)."""
    return SceneEntity(
        frame_id=frame_id,
        id=entity_id,
        spheres=[
            SpherePrimitive(
                pose=_pose(position),
                size=Vector3(x=diameter, y=diameter, z=diameter),
                color=color,
            )
        ],
    )


def cube_entity(
    entity_id: str,
    position: np.ndarray,
    size: tuple[float, float, float],
    color: Color,
    quaternion: np.ndarray | None = None,
    frame_id: str = "world",
) -> SceneEntity:
    """A single-cube entity (used for link primitives if you don't have meshes)."""
    sx, sy, sz = size
    return SceneEntity(
        frame_id=frame_id,
        id=entity_id,
        cubes=[
            CubePrimitive(
                pose=_pose(position, quaternion),
                size=Vector3(x=sx, y=sy, z=sz),
                color=color,
            )
        ],
    )


def line_entity(
    entity_id: str,
    points: list[np.ndarray],
    color: Color,
    thickness: float = 0.002,
    frame_id: str = "world",
) -> SceneEntity:
    """A poly-line entity (e.g. the tip->target line)."""
    pts = [Point3(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
    return SceneEntity(
        frame_id=frame_id,
        id=entity_id,
        lines=[
            LinePrimitive(
                type=LinePrimitiveLineType.LineStrip,
                thickness=thickness,
                scale_invariant=True,
                points=pts,
                color=color,
            )
        ],
    )


def scene_update(entities: list[SceneEntity]) -> SceneUpdate:
    """Wrap entities in a ``SceneUpdate``."""
    return SceneUpdate(entities=entities)


def target_color(within_tolerance: bool) -> Color:
    """Goal-sphere colour: green once the tip is within tolerance, else amber."""
    return GREEN if within_tolerance else AMBER


def reach_scene(
    target: np.ndarray,
    tip: np.ndarray,
    within_tolerance: bool,
    target_diameter: float = 0.02,
    tip_diameter: float = 0.012,
) -> SceneUpdate:
    """The standard reach-task scene: goal sphere, tip marker, and a joining line.

    The goal turns green once the tip is within tolerance, so a learner can *see*
    success without reading any numbers.
    """
    return scene_update(
        [
            sphere_entity("target", target, target_diameter, target_color(within_tolerance)),
            sphere_entity("tip", tip, tip_diameter, BLUE),
            line_entity("tip_to_target", [tip, target], WHITE),
        ]
    )


# --------------------------------------------------------------------------- #
# Numeric metrics
# --------------------------------------------------------------------------- #
def metrics_message(
    distance: float,
    reward: float,
    episode_return: float = 0.0,
    success_rate: float = 0.0,
) -> dict:
    """Build the JSON payload for the numeric metrics channel."""
    return {
        "distance": float(distance),
        "reward": float(reward),
        "episode_return": float(episode_return),
        "success_rate": float(success_rate),
    }
