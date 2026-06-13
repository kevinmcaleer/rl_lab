"""The default physics backend: Buddy Jr in PyBullet.

:class:`PyBulletBackend` implements the :class:`rl_lab.sim.base.SimBackend`
contract on top of PyBullet. It loads the URDF as a fixed-base robot, drives the
four revolute joints with position control (force/velocity capped to mimic the
quasi-static SG90 hobby servos), and reports joint states and link poses in the
SI/world conventions the rest of the lab assumes.

Why PyBullet is imported *lazily*
---------------------------------
PyBullet has no macOS arm64 wheel, so it lives in the optional ``[sim]`` extra
and is exercised in CI only on Linux. To keep ``import rl_lab`` working
everywhere (the Foxglove visualization runs on pure forward kinematics and needs
no physics engine), we import ``pybullet`` inside :meth:`__init__`, not at module
load time, and raise a friendly error pointing at the conda-forge fallback if it
is missing.

Coordinate / units recap (see :mod:`rl_lab.sim.base`): metres, radians, seconds;
joint order ``[base_yaw, shoulder_pitch, elbow_pitch, camera_tilt]``; poses are
``(position[3], quaternion_xyzw[4])`` in the world frame. PyBullet's quaternion
order is already ``[x, y, z, w]``, so no reordering is needed.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from rl_lab.robot import buddy_jr as bj
from rl_lab.sim import loader
from rl_lab.sim.base import SimBackend

# Per-joint force caps (N*m), taken straight from the URDF ``effort`` values, in
# joint order. They keep the position controller from applying super-servo
# torque, so the sim behaves like the real SG90s.
_JOINT_FORCES: tuple[float, ...] = (1.5, 1.5, 1.2, 0.8)

# Max joint speed (rad/s) for the position controller — the URDF ``velocity``.
_MAX_JOINT_VELOCITY: float = 6.0

_IMPORT_HINT = (
    "PyBullet is required for the physics backend but could not be imported.\n"
    "Install the optional sim extra:  pip install 'rl_lab[sim]'\n"
    "On macOS (Apple Silicon) there is no PyBullet wheel on PyPI; use "
    "conda-forge instead:  conda install -c conda-forge pybullet"
)


class PyBulletBackend(SimBackend):
    """A :class:`SimBackend` driven by the PyBullet physics engine.

    Parameters
    ----------
    mode:
        ``"direct"`` for a headless server (tests, training) or ``"gui"`` for a
        windowed server you can watch.
    urdf_path:
        Optional explicit path to the robot URDF; defaults to the packaged
        ``urdf/buddy_jr.urdf``.
    sim_dt:
        The physics integrator timestep (seconds). :meth:`step` runs
        ``round(ctrl_dt / sim_dt)`` substeps so one control step advances the
        world by :attr:`ctrl_dt`.
    gravity_z:
        Gravitational acceleration along world ``-Z`` (m/s^2).
    """

    def __init__(
        self,
        mode: str = "direct",
        *,
        urdf_path: str | Path | None = None,
        sim_dt: float = loader.DEFAULT_SIM_DT,
        gravity_z: float = loader.DEFAULT_GRAVITY_Z,
    ) -> None:
        try:
            import pybullet  # noqa: PLC0415  (lazy import is intentional — see module docstring)
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_IMPORT_HINT) from exc

        mode_l = mode.lower()
        if mode_l not in ("direct", "gui"):
            raise ValueError(f"mode must be 'direct' or 'gui', got {mode!r}")

        self._pb = pybullet
        self.mode = mode_l
        self.sim_dt = float(sim_dt)
        # How many physics substeps make up one control step. round() avoids an
        # off-by-one from float division (1/60 / (1/240) == 3.9999...).
        self._n_substeps = max(1, round(self.ctrl_dt / self.sim_dt))

        # --- connect & configure the world -------------------------------- #
        self._client = loader.connect(pybullet, gui=(mode_l == "gui"))
        loader.configure_world(pybullet, self._client, sim_dt=self.sim_dt, gravity_z=gravity_z)

        # --- load the robot ----------------------------------------------- #
        self._robot = loader.load_robot(pybullet, self._client, urdf_path)

        # --- build name -> index maps from the URDF ----------------------- #
        # PyBullet identifies a link by the index of the joint whose *child* it
        # is, so the joint and link maps are built from the same getJointInfo.
        self._joint_index: dict[str, int] = {}
        self._link_index: dict[str, int] = {}
        n_joints = pybullet.getNumJoints(self._robot, physicsClientId=self._client)
        for j in range(n_joints):
            info = pybullet.getJointInfo(self._robot, j, physicsClientId=self._client)
            joint_name = info[1].decode("utf-8")
            child_link = info[12].decode("utf-8")
            self._joint_index[joint_name] = j
            self._link_index[child_link] = j

        # Our 4 actuated joints, in the canonical action/observation order.
        missing = [n for n in bj.JOINT_NAMES if n not in self._joint_index]
        if missing:
            raise RuntimeError(f"URDF is missing expected joints: {missing}")
        self._actuated: list[int] = [self._joint_index[n] for n in bj.JOINT_NAMES]

        # Target-marker body id, cached so spawn_target moves it on repeat.
        self._target_id: int | None = None
        self._closed = False

        # Place the arm at rest.
        self.reset()

    # ------------------------------------------------------------------ #
    # SimBackend contract
    # ------------------------------------------------------------------ #
    def reset(self, initial_q: np.ndarray | None = None) -> None:
        """Reset the arm to ``initial_q`` (default zeros) and hold it there.

        We hard-set each joint with ``resetJointState`` (zero velocity), then
        register a position-control target at the same angle so the motors hold
        the pose against gravity once :meth:`step` runs.
        """
        if initial_q is None:
            q = np.zeros(bj.NUM_JOINTS, dtype=np.float64)
        else:
            q = bj.clamp_to_limits(initial_q)

        for joint_idx, angle in zip(self._actuated, q, strict=True):
            self._pb.resetJointState(
                self._robot,
                joint_idx,
                targetValue=float(angle),
                targetVelocity=0.0,
                physicsClientId=self._client,
            )
        # Hold the reset pose under position control.
        self.set_joint_targets(q)

    def step(self) -> None:
        """Advance the world by one control step (several physics substeps)."""
        for _ in range(self._n_substeps):
            self._pb.stepSimulation(physicsClientId=self._client)

    def set_joint_targets(self, q: np.ndarray) -> None:
        """Command clamped position targets, capping force and speed per joint."""
        q = bj.clamp_to_limits(q)
        for joint_idx, angle, force in zip(self._actuated, q, _JOINT_FORCES, strict=True):
            self._pb.setJointMotorControl2(
                self._robot,
                joint_idx,
                controlMode=self._pb.POSITION_CONTROL,
                targetPosition=float(angle),
                force=float(force),
                maxVelocity=_MAX_JOINT_VELOCITY,
                physicsClientId=self._client,
            )

    def get_joint_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(positions, velocities)`` for the 4 joints as ``(4,)`` arrays."""
        states = self._pb.getJointStates(self._robot, self._actuated, physicsClientId=self._client)
        positions = np.array([s[0] for s in states], dtype=np.float64)
        velocities = np.array([s[1] for s in states], dtype=np.float64)
        return positions, velocities

    def get_link_pose(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        """World ``(position[3], quaternion_xyzw[4])`` of ``link_name``."""
        if link_name == "base_link":
            pos, orn = self._pb.getBasePositionAndOrientation(
                self._robot, physicsClientId=self._client
            )
            return np.asarray(pos, dtype=np.float64), np.asarray(orn, dtype=np.float64)

        if link_name not in self._link_index:
            raise KeyError(f"unknown link {link_name!r}; known links: {sorted(self._link_index)}")
        state = self._pb.getLinkState(
            self._robot,
            self._link_index[link_name],
            computeForwardKinematics=1,
            physicsClientId=self._client,
        )
        # Indices 4/5 are the world *link frame* pose (URDF frame), which is what
        # our kinematics report — not the CoM pose at indices 0/1.
        pos = np.asarray(state[4], dtype=np.float64)
        orn = np.asarray(state[5], dtype=np.float64)
        return pos, orn

    def spawn_target(self, position: np.ndarray, radius: float = 0.01) -> int:
        """Spawn the goal marker, or move the existing one; return its body id."""
        if self._target_id is None:
            self._target_id = loader.spawn_target_marker(
                self._pb, self._client, position, radius=radius
            )
        else:
            loader.move_target_marker(self._pb, self._client, self._target_id, position)
        return self._target_id

    def close(self) -> None:
        """Disconnect the physics client (idempotent)."""
        if self._closed:
            return
        with contextlib.suppress(Exception):  # pragma: no cover - already torn down
            self._pb.disconnect(physicsClientId=self._client)
        self._closed = True
