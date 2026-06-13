"""MuJoCo backend smoke test, cross-checked against the analytic kinematics.

Runs only where the optional ``[mujoco]`` extra is installed; self-skips
otherwise (CI does not install MuJoCo). Same FK cross-check as the PyBullet
backend test, proving both engines agree with :func:`rl_lab.robot.kinematics`.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_lab.robot import buddy_jr as bj
from rl_lab.robot import kinematics as kin

pytest.importorskip("mujoco")  # skip unless the [mujoco] extra is installed

from rl_lab.sim.mujoco_sim import MujocoBackend  # noqa: E402

_QS = [
    np.zeros(4),
    np.array([0.3, 0.5, -0.4, 0.1]),
    np.array([-0.6, 0.4, 0.5, -0.2]),
]


@pytest.fixture(scope="module")
def backend():
    be = MujocoBackend()
    yield be
    be.close()


@pytest.mark.parametrize("q", _QS)
def test_camera_tip_matches_analytic_fk(backend, q) -> None:
    """The MuJoCo camera-tip pose matches the analytic FK after reset(q)."""
    backend.reset(q)
    pos, _quat = backend.get_camera_tip_pose()
    np.testing.assert_allclose(pos, kin.forward(q).position, atol=1e-4)


@pytest.mark.parametrize("q", _QS)
def test_reset_sets_joint_states(backend, q) -> None:
    backend.reset(q)
    positions, _vel = backend.get_joint_states()
    np.testing.assert_allclose(positions, q, atol=1e-6)


def test_targets_clamped_to_limits(backend) -> None:
    backend.reset(np.zeros(4))
    backend.set_joint_targets(np.array([5.0, -5.0, 5.0, -5.0]))
    for _ in range(400):
        backend.step()
    positions, _ = backend.get_joint_states()
    assert np.all(np.abs(positions) <= bj.JOINT_LIMIT + 0.05)
