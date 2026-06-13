"""PyBullet world setup and target-marker helpers — small, pure-ish functions.

This module is deliberately thin: every function takes the *PyBullet client
module* (``pybullet`` itself, or whatever object exposes the same API) as its
first argument, so nothing here imports PyBullet at module load time. That keeps
``import rl_lab.sim.loader`` cheap and lets the package import on macOS arm64,
where there is no PyBullet wheel (it lives in the optional ``[sim]`` extra and is
tested in CI only on Linux).

The functions cover the four jobs :class:`rl_lab.sim.pybullet_sim.PyBulletBackend`
needs from PyBullet:

#. connect a headless (``DIRECT``) or windowed (``GUI``) physics client;
#. configure gravity and the integrator timestep;
#. load ``urdf/buddy_jr.urdf`` as a fixed-base robot;
#. spawn (and later move) a small coloured sphere as the reach target.

All units are SI (metres, radians, seconds) and the world frame is REP-103
(``+X`` forward, ``+Y`` left, ``+Z`` up), matching the URDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_lab.robot.buddy_jr import urdf_path as default_urdf_path

# A pleasant amber sphere for the target marker — RGBA in 0..1.
_TARGET_RGBA: tuple[float, float, float, float] = (0.95, 0.65, 0.10, 0.9)

# Default integrator step. PyBullet is happiest near 240 Hz; the backend runs
# several of these substeps per control step (1/60 s).
DEFAULT_SIM_DT: float = 1.0 / 240.0

# Standard Earth gravity along -Z (m/s^2).
DEFAULT_GRAVITY_Z: float = -9.81


def connect(pb: Any, *, gui: bool = False) -> int:
    """Connect a PyBullet physics client and return its integer id.

    Parameters
    ----------
    pb:
        The PyBullet module (passed in so this stays import-light).
    gui:
        If ``True`` open the windowed ``GUI`` server (handy for eyeballing the
        arm); otherwise use the headless ``DIRECT`` server used in tests and
        training.

    Returns
    -------
    int
        The client id, suitable for the ``physicsClientId=`` kwarg.
    """
    mode = pb.GUI if gui else pb.DIRECT
    client_id = pb.connect(mode)
    if gui:
        # A clean teaching view: hide the debug-GUI clutter so the arm is the
        # star of the show. These calls are no-ops in DIRECT mode.
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=client_id)
    return client_id


def configure_world(
    pb: Any,
    client_id: int,
    *,
    sim_dt: float = DEFAULT_SIM_DT,
    gravity_z: float = DEFAULT_GRAVITY_Z,
) -> None:
    """Set gravity and the fixed integrator timestep for a connected client."""
    pb.setGravity(0.0, 0.0, gravity_z, physicsClientId=client_id)
    pb.setTimeStep(sim_dt, physicsClientId=client_id)


def load_robot(
    pb: Any,
    client_id: int,
    urdf_path: str | Path | None = None,
) -> int:
    """Load the Buddy Jr URDF as a *fixed-base* robot and return its body id.

    The base of a robot arm is bolted to the bench, so we load it with
    ``useFixedBase=True``: the base link will not fall under gravity and the
    only freedoms are the four revolute joints.

    ``urdf_path`` defaults to the packaged ``urdf/buddy_jr.urdf`` (resolved via
    :func:`rl_lab.viz.urdf_publisher.default_urdf_path`), but you may pass an
    explicit path.
    """
    path = Path(urdf_path) if urdf_path is not None else default_urdf_path()
    if not path.exists():
        raise FileNotFoundError(f"URDF not found: {path}")
    return pb.loadURDF(
        str(path),
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
        physicsClientId=client_id,
    )


def spawn_target_marker(
    pb: Any,
    client_id: int,
    position: Any,
    *,
    radius: float = 0.01,
    rgba: tuple[float, float, float, float] = _TARGET_RGBA,
) -> int:
    """Create a massless, collision-free coloured sphere at ``position``.

    The marker is purely visual: ``baseMass=0`` means it is static, and we give
    it no collision shape so it never perturbs the arm. It exists only so a
    human (or the Foxglove view) can see where the goal is.

    Returns the new body id, which you keep so you can later move the marker
    with :func:`move_target_marker` instead of spawning a fresh one each frame.
    """
    x, y, z = (float(v) for v in position)
    visual_shape = pb.createVisualShape(
        pb.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=list(rgba),
        physicsClientId=client_id,
    )
    return pb.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,  # -1 == no collision geometry
        baseVisualShapeIndex=visual_shape,
        basePosition=[x, y, z],
        physicsClientId=client_id,
    )


def move_target_marker(pb: Any, client_id: int, body_id: int, position: Any) -> None:
    """Teleport an existing target marker to ``position`` (identity rotation)."""
    x, y, z = (float(v) for v in position)
    pb.resetBasePositionAndOrientation(
        body_id,
        [x, y, z],
        [0.0, 0.0, 0.0, 1.0],
        physicsClientId=client_id,
    )
