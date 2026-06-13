"""Turn the Buddy Jr URDF into Foxglove messages: geometry once, motion per step.

Two pure builders (no channels, no I/O beyond reading the URDF once):

* :func:`robot_geometry_scene` — one :class:`SceneEntity` per link, each a
  primitive pinned to *its own link frame*. Sent once on connect.
* :func:`robot_transforms` — the per-step ``FrameTransforms`` (from forward
  kinematics) that move those link frames so the arm animates.

Because the motion comes from :func:`rl_lab.robot.kinematics.joint_transforms`
(pure FK), the visualization works even where a physics engine cannot be
installed — e.g. macOS without PyBullet.

Today every link is drawn as a primitive box (cylinders are approximated by
their bounding box). Swap in ``ModelPrimitive`` meshes once STLs land.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from foxglove.messages import FrameTransforms, SceneUpdate

from rl_lab.robot import kinematics as kin

# The canonical URDF-path helper lives in the engine-agnostic robot module;
# re-exported here under its long-standing name for callers of this module.
from rl_lab.robot.buddy_jr import urdf_path as default_urdf_path
from rl_lab.viz import schemas

WORLD_FRAME = "world"
ROOT_LINK = "base_link"


@dataclass(frozen=True)
class LinkVisual:
    """A link's primitive visual: name, box size (m), and local origin (m)."""

    name: str
    size: tuple[float, float, float]
    origin: tuple[float, float, float]


def _parse_floats(text: str | None, n: int) -> tuple[float, ...]:
    if not text:
        return (0.0,) * n
    return tuple(float(v) for v in text.split())


@lru_cache(maxsize=4)
def _load_link_visuals(urdf_path: str) -> tuple[LinkVisual, ...]:
    """Parse each link's visual geometry into an axis-aligned box approximation."""
    root = ET.parse(urdf_path).getroot()
    visuals: list[LinkVisual] = []
    for link in root.findall("link"):
        visual = link.find("visual")
        if visual is None:
            continue
        geom = visual.find("geometry")
        shape = list(geom)[0] if geom is not None and len(list(geom)) else None
        if shape is None:
            continue
        if shape.tag == "box":
            sx, sy, sz = _parse_floats(shape.get("size"), 3)
        elif shape.tag == "cylinder":
            radius = float(shape.get("radius", 0.01))
            length = float(shape.get("length", 0.01))
            sx, sy, sz = 2 * radius, 2 * radius, length
        else:  # sphere or other: fall back to a small cube
            r = float(shape.get("radius", 0.01))
            sx = sy = sz = 2 * r
        origin_el = visual.find("origin")
        ox, oy, oz = _parse_floats(origin_el.get("xyz") if origin_el is not None else None, 3)
        visuals.append(LinkVisual(link.get("name", "?"), (sx, sy, sz), (ox, oy, oz)))
    return tuple(visuals)


def robot_geometry_scene(urdf_path: str | Path | None = None) -> SceneUpdate:
    """Build the static robot geometry: one box entity per link, in its frame."""
    path = str(urdf_path) if urdf_path is not None else str(default_urdf_path())
    entities = [
        schemas.cube_entity(
            entity_id=lv.name,
            position=np.array(lv.origin),
            size=lv.size,
            color=schemas.GREY,
            frame_id=lv.name,
        )
        for lv in _load_link_visuals(path)
    ]
    return schemas.scene_update(entities)


def robot_transforms(q: np.ndarray) -> FrameTransforms:
    """Per-step ``FrameTransforms``: world->base_link plus every joint transform."""
    transforms: list[tuple[str, str, np.ndarray, np.ndarray]] = [
        (WORLD_FRAME, ROOT_LINK, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])),
        *kin.joint_transforms(q),
    ]
    return schemas.frame_transforms(transforms)
