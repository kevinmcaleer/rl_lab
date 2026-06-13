"""The ``SimBackend`` contract — the seam between RL code and the physics engine.

Every backend (PyBullet by default, MuJoCo optional) implements this single
interface, so the environment, the kinematics checks, and the Foxglove bridge
never import a specific engine. Swapping engines becomes a one-line change —
and the basis for a future "does my policy transfer across simulators?"
experiment.

Units & conventions (the whole lab obeys these):

* **Lengths** are metres, **angles** are radians, **time** is seconds (SI).
* **Joint order** is ``[base_yaw, shoulder_pitch, elbow_pitch, camera_tilt]``,
  matching :data:`rl_lab.robot.buddy_jr.JOINTS`.
* **Poses** are returned as ``(position, orientation)`` where ``position`` is a
  ``(3,)`` array ``[x, y, z]`` and ``orientation`` is a ``(4,)`` quaternion
  ``[x, y, z, w]`` in the world frame.
* **Joint states** are returned as ``(positions, velocities)``, each a ``(4,)``
  array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SimBackend(ABC):
    """Abstract physics backend for the Buddy Jr arm.

    Implementations load ``urdf/buddy_jr.urdf``, drive the joints toward
    position targets with a force/velocity cap that approximates the quasi-static
    SG90 servos, and report joint states and link poses.
    """

    #: Control timestep in seconds (one :meth:`step` advances the world by this).
    ctrl_dt: float = 1.0 / 60.0

    @abstractmethod
    def reset(self, initial_q: np.ndarray | None = None) -> None:
        """Reset the world and place the arm at ``initial_q`` (default: zeros)."""

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one control step (running internal substeps)."""

    @abstractmethod
    def set_joint_targets(self, q: np.ndarray) -> None:
        """Command position targets for the 4 joints (clamped to URDF limits)."""

    @abstractmethod
    def get_joint_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(positions, velocities)``, each a ``(4,)`` radian/rad-per-s array."""

    @abstractmethod
    def get_link_pose(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the world ``(position[3], quaternion_xyzw[4])`` of ``link_name``."""

    @abstractmethod
    def spawn_target(self, position: np.ndarray, radius: float = 0.01) -> int:
        """Spawn / move the goal marker to ``position``; return its integer handle."""

    @abstractmethod
    def close(self) -> None:
        """Release the physics client and any windows."""

    # Convenience shared by all backends -------------------------------------
    def get_camera_tip_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the camera tip (the end-effector link)."""
        from rl_lab.robot.buddy_jr import END_EFFECTOR_LINK

        return self.get_link_pose(END_EFFECTOR_LINK)

    def __enter__(self) -> SimBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
