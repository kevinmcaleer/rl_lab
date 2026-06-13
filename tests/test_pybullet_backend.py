"""PyBullet backend smoke test, cross-checked against the analytic kinematics.

Runs only where the ``[sim]`` extra is installed (CI Linux); self-skips on
macOS / anywhere PyBullet is unavailable. The key assertion is that the
backend's reported camera-tip pose matches :func:`rl_lab.robot.kinematics.forward`
— if the URDF, the loader, and the FK ever drift apart, this fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_lab.robot import buddy_jr as bj
from rl_lab.robot import kinematics as kin

pytest.importorskip("pybullet")  # skip unless the [sim] extra is installed

from rl_lab.sim.pybullet_sim import PyBulletBackend  # noqa: E402

_QS = [
    np.zeros(4),
    np.array([0.0, 0.5, -0.5, 0.0]),
    np.array([0.6, -0.4, 0.3, 0.2]),
    np.array([-0.8, 0.7, -0.9, -0.3]),
]


@pytest.fixture(scope="module")
def backend():
    be = PyBulletBackend("direct")
    yield be
    be.close()


@pytest.mark.parametrize("q", _QS)
def test_camera_tip_matches_analytic_fk(backend, q) -> None:
    """After resetting to q, the backend's camera-tip pose matches FK (<2 mm)."""
    backend.reset(q)
    pos, _quat = backend.get_camera_tip_pose()
    np.testing.assert_allclose(pos, kin.forward(q).position, atol=2e-3)


@pytest.mark.parametrize("q", _QS)
def test_reset_sets_joint_states(backend, q) -> None:
    """reset(q) places the joints exactly at q (zero velocity)."""
    backend.reset(q)
    positions, velocities = backend.get_joint_states()
    np.testing.assert_allclose(positions, q, atol=1e-6)
    np.testing.assert_allclose(velocities, np.zeros(4), atol=1e-6)


def test_position_control_settles_toward_target(backend) -> None:
    """Stepping under position control drives the joints toward the target."""
    backend.reset(np.zeros(4))
    target = np.array([0.4, -0.3, 0.5, 0.1])
    backend.set_joint_targets(target)
    for _ in range(400):
        backend.step()
    positions, _ = backend.get_joint_states()
    np.testing.assert_allclose(positions, target, atol=0.05)


def test_targets_are_clamped_to_limits(backend) -> None:
    """Out-of-range targets are clamped to the +/-90 deg joint limits."""
    backend.reset(np.zeros(4))
    backend.set_joint_targets(np.array([5.0, -5.0, 5.0, -5.0]))
    for _ in range(400):
        backend.step()
    positions, _ = backend.get_joint_states()
    assert np.all(positions <= bj.JOINT_LIMIT + 0.05)
    assert np.all(positions >= -bj.JOINT_LIMIT - 0.05)


def test_spawn_target_reuses_handle(backend) -> None:
    """Spawning then re-spawning moves the same body instead of leaking ids."""
    h1 = backend.spawn_target(np.array([0.1, 0.0, 0.15]))
    h2 = backend.spawn_target(np.array([0.0, 0.1, 0.18]))
    assert isinstance(h1, int) and h1 == h2
