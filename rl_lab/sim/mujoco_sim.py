"""Optional MuJoCo physics backend for the Buddy Jr arm.

MuJoCo is the swappable alternative to the default PyBullet backend (same
:class:`rl_lab.sim.base.SimBackend` interface), enabling the "does my policy
transfer across simulators?" experiment. It is **not** imported at package load
time — ``import mujoco`` happens lazily inside :meth:`MujocoBackend.__init__`,
so the default install never needs the ``mujoco`` wheel. Add it with::

    pip install rl-lab[mujoco]

How the model is built
----------------------
MuJoCo 3.x parses URDF natively: ``mujoco.MjModel.from_xml_path`` on the URDF
welds the root link (``base_link``) to the world (so the base is fixed, like
PyBullet's ``useFixedBase=True``) and turns the four revolute joints into hinge
joints. We do *not* try to inject MuJoCo actuators into the URDF (that needs
fragile XML surgery); instead :meth:`step` applies a simple, force-capped PD
controller through ``data.qfrc_applied`` — which mimics the quasi-static SG90
servos and keeps the loader trivial and robust across MuJoCo versions.

The reach target is purely visual and is drawn by the Foxglove bridge, so the
MuJoCo backend just remembers the target position rather than adding geometry.

Conventions match the rest of the lab: SI units, joint order
``[base_yaw, shoulder_pitch, elbow_pitch, camera_tilt]``, poses as
``(position[3], quaternion_xyzw[4])``. MuJoCo stores quaternions scalar-first
``[w, x, y, z]``; :meth:`get_link_pose` converts to our ``[x, y, z, w]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_lab.robot import buddy_jr as bj
from rl_lab.sim.base import SimBackend

# PD gains + force cap approximating the SG90 servos (small, low-torque hobby
# servos). Tuned so the arm tracks position targets and holds against gravity
# at this ~80 mm scale without oscillating.
ACTUATOR_KP: float = 0.6
ACTUATOR_KD: float = 0.02
ACTUATOR_FORCE_MAX: float = 2.0  # N*m hard cap (well above SG90 stall torque)

PHYSICS_DT: float = 0.002  # MuJoCo integrator timestep (s); substeps bridge to ctrl_dt


class MujocoBackend(SimBackend):
    """A :class:`SimBackend` driven by MuJoCo (optional ``[mujoco]`` extra)."""

    ctrl_dt: float = 1.0 / 60.0

    def __init__(self, urdf_path: str | Path | None = None, ctrl_dt: float = 1.0 / 60.0) -> None:
        try:
            import mujoco  # noqa: PLC0415  (lazy import is intentional — see module docstring)
        except ModuleNotFoundError as exc:  # pragma: no cover - only without the extra
            raise ModuleNotFoundError(
                "MuJoCo is not installed. Install the optional extra:\n"
                "    pip install 'rl_lab[mujoco]'   (or: pip install mujoco)"
            ) from exc

        self._mj = mujoco
        self.ctrl_dt = float(ctrl_dt)
        path = Path(urdf_path) if urdf_path is not None else bj.urdf_path()
        if not path.exists():
            raise FileNotFoundError(f"URDF not found: {path}")

        # MuJoCo parses the URDF directly; the root link is welded to the world.
        self._model = mujoco.MjModel.from_xml_path(str(path))
        self._model.opt.timestep = PHYSICS_DT
        self._data = mujoco.MjData(self._model)
        self._n_substeps = max(1, round(self.ctrl_dt / PHYSICS_DT))

        # Cache joint qpos/dof addresses in canonical order.
        self._qpos_adr: list[int] = []
        self._dof_adr: list[int] = []
        for name in bj.JOINT_NAMES:
            jid = self._model.joint(name).id
            self._qpos_adr.append(int(self._model.jnt_qposadr[jid]))
            self._dof_adr.append(int(self._model.jnt_dofadr[jid]))

        self._targets = np.zeros(bj.NUM_JOINTS, dtype=np.float64)
        self._target_pos = np.zeros(3, dtype=np.float64)
        self.reset()

    # --------------------------------------------------------------- helpers
    def _read(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.array([self._data.qpos[a] for a in self._qpos_adr], dtype=np.float64)
        v = np.array([self._data.qvel[a] for a in self._dof_adr], dtype=np.float64)
        return q, v

    # ----------------------------------------------------------- SimBackend
    def reset(self, initial_q: np.ndarray | None = None) -> None:
        self._mj.mj_resetData(self._model, self._data)
        q = np.zeros(bj.NUM_JOINTS) if initial_q is None else bj.clamp_to_limits(initial_q)
        self._targets = q.copy()
        for adr, angle in zip(self._qpos_adr, q, strict=True):
            self._data.qpos[adr] = float(angle)
        self._mj.mj_forward(self._model, self._data)

    def step(self) -> None:
        """Advance one control step, applying a force-capped PD torque per substep."""
        for _ in range(self._n_substeps):
            q, v = self._read()
            tau = ACTUATOR_KP * (self._targets - q) - ACTUATOR_KD * v
            tau = np.clip(tau, -ACTUATOR_FORCE_MAX, ACTUATOR_FORCE_MAX)
            for adr, t in zip(self._dof_adr, tau, strict=True):
                self._data.qfrc_applied[adr] = float(t)
            self._mj.mj_step(self._model, self._data)

    def set_joint_targets(self, q: np.ndarray) -> None:
        self._targets = bj.clamp_to_limits(q)

    def get_joint_states(self) -> tuple[np.ndarray, np.ndarray]:
        return self._read()

    @staticmethod
    def _wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
        """3x3 rotation matrix from a MuJoCo ``[w, x, y, z]`` quaternion."""
        w, x, y, z = (float(v) for v in q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def get_link_pose(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        """World ``(position[3], quaternion_xyzw[4])`` of a URDF link.

        MuJoCo welds fixed-joint child links into their parent and renames the
        welded root to ``world``. We remap ``base_link`` -> ``world`` and rebuild
        the camera-tip frames (``camera_link`` / ``camera_optical_frame``) from
        ``camera_mount`` plus the fixed camera offset, so the contract still
        exposes every URDF link.
        """
        target = "world" if link_name == "base_link" else link_name
        try:
            body = self._data.body(target)
            pos = np.array(body.xpos, dtype=np.float64)
            q_wxyz = np.array(body.xquat, dtype=np.float64)
        except KeyError:
            if link_name in ("camera_link", "camera_optical_frame"):
                mount = self._data.body("camera_mount")
                m_pos = np.array(mount.xpos, dtype=np.float64)
                q_wxyz = np.array(mount.xquat, dtype=np.float64)
                # camera_joint is a pure translation (rpy=0), so the orientation
                # is unchanged and the tip is the mount pose + rotated offset.
                pos = m_pos + self._wxyz_to_matrix(q_wxyz) @ np.array(bj.CAMERA_OFFSET)
            else:
                raise
        w, x, y, z = (float(v) for v in q_wxyz)
        return pos, np.array([x, y, z, w], dtype=np.float64)

    def spawn_target(self, position: np.ndarray, radius: float = 0.01) -> int:
        """Record the target position (the marker is drawn by Foxglove, not MuJoCo)."""
        self._target_pos = np.asarray(position, dtype=np.float64).copy()
        return 0

    def close(self) -> None:
        # MuJoCo frees its C objects on garbage-collection; drop references.
        self._data = None
        self._model = None
