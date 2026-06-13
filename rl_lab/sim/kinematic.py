"""A physics-free ``SimBackend`` driven purely by forward kinematics.

This is the **default** backend for the RL environment because it needs no
physics engine at all — just NumPy — so the lab runs out of the box on macOS
(where PyBullet has no wheel) and in CI everywhere. It is also the right tool
for the early experiments (bandit, tabular Q-learning, reward shaping): the
dynamics are deterministic and instantaneous, so a learner sees the *RL* signal
without dynamics noise muddying it.

The model: joints move toward their targets at a capped rate (no gravity, no
contacts), and every link pose comes straight from
:func:`rl_lab.robot.kinematics`. Swap in :class:`~rl_lab.sim.pybullet_sim.PyBulletBackend`
or :class:`~rl_lab.sim.mujoco_sim.MujocoBackend` for real dynamics later — the
``SimBackend`` interface is identical, which is itself a lesson in sim-to-sim
transfer.
"""

from __future__ import annotations

import numpy as np

from rl_lab.robot import buddy_jr as bj
from rl_lab.robot import kinematics as kin
from rl_lab.sim.base import SimBackend

# Max joint speed (rad/s) — the URDF ``velocity`` limit. One control step moves
# a joint at most ``MAX_JOINT_VELOCITY * ctrl_dt`` radians toward its target.
MAX_JOINT_VELOCITY: float = 6.0


class KinematicBackend(SimBackend):
    """Instantaneous, physics-free backend: joints slew toward targets via FK."""

    ctrl_dt: float = 1.0 / 60.0

    def __init__(self, ctrl_dt: float = 1.0 / 60.0, max_joint_velocity: float = MAX_JOINT_VELOCITY):
        self.ctrl_dt = float(ctrl_dt)
        self._max_step = float(max_joint_velocity) * self.ctrl_dt
        self._q = np.zeros(bj.NUM_JOINTS, dtype=np.float64)
        self._qd = np.zeros(bj.NUM_JOINTS, dtype=np.float64)
        self._targets = np.zeros(bj.NUM_JOINTS, dtype=np.float64)
        self._target_pos = np.zeros(3, dtype=np.float64)

    def reset(self, initial_q: np.ndarray | None = None) -> None:
        q = np.zeros(bj.NUM_JOINTS) if initial_q is None else bj.clamp_to_limits(initial_q)
        self._q = np.asarray(q, dtype=np.float64).copy()
        self._targets = self._q.copy()
        self._qd = np.zeros(bj.NUM_JOINTS, dtype=np.float64)

    def step(self) -> None:
        """Slew each joint toward its target by at most one velocity-limited step."""
        delta = np.clip(self._targets - self._q, -self._max_step, self._max_step)
        self._qd = delta / self.ctrl_dt
        self._q = bj.clamp_to_limits(self._q + delta)

    def set_joint_targets(self, q: np.ndarray) -> None:
        self._targets = bj.clamp_to_limits(q)

    def get_joint_states(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q.copy(), self._qd.copy()

    def get_link_pose(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        pose = kin.link_world_poses(self._q)[link_name]
        return pose.position.copy(), pose.orientation.copy()

    def spawn_target(self, position: np.ndarray, radius: float = 0.01) -> int:
        self._target_pos = np.asarray(position, dtype=np.float64).copy()
        return 0

    def close(self) -> None:  # nothing to release
        pass
