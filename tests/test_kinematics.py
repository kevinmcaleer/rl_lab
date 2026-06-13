"""FK/IK correctness for the Buddy Jr arm: round-trip, reachability, limits.

Pure NumPy — runs everywhere (no physics engine needed), so it is the
authoritative check that the analytic kinematics match the URDF geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_lab.robot import buddy_jr as bj
from rl_lab.robot import kinematics as kin


def test_forward_zero_config_matches_urdf_geometry() -> None:
    """At q=0 the tip is the straight-up arm plus the fixed camera offset."""
    pose = kin.forward(np.zeros(4))
    # z = 0.025 + 0.025 + 0.080 + 0.080 + 0.010 ; x = camera offset 0.0145
    np.testing.assert_allclose(pose.position, [0.0145, 0.0, 0.22], atol=1e-9)
    # unit quaternion
    np.testing.assert_allclose(np.linalg.norm(pose.orientation), 1.0, atol=1e-9)


def test_base_yaw_rotates_tip_about_z() -> None:
    """A pure base yaw should swing the tip around the vertical axis."""
    q = np.array([np.pi / 2, 0.0, 0.0, 0.0])
    p = kin.forward(q).position
    # x and y swap (tip that was at +x is now at +y), z unchanged.
    np.testing.assert_allclose(p, [0.0, 0.0145, 0.22], atol=1e-9)


def _reachable_targets() -> list[np.ndarray]:
    """A grid of targets generated from valid joint configs (guaranteed reachable)."""
    targets = []
    rng = np.linspace(-1.2, 1.2, 5)
    for yaw in (-1.0, 0.0, 1.0):
        for sh in rng:
            for el in rng:
                q = np.array([yaw, sh, el, 0.0])
                targets.append(kin.forward(q).position)
    return targets


@pytest.mark.parametrize("target", _reachable_targets())
def test_fk_ik_round_trip(target: np.ndarray) -> None:
    """FK(IK(target)) returns to the target for reachable points (tilt=0)."""
    q = kin.inverse(target, camera_tilt=0.0)
    assert bj.within_limits(q)
    recovered = kin.forward(q).position
    np.testing.assert_allclose(recovered, target, atol=1e-6)


def test_round_trip_with_nonzero_camera_tilt() -> None:
    """The solver honours a fixed camera tilt and still round-trips."""
    for tilt in (-0.5, 0.3, 0.8):
        target = kin.forward(np.array([0.4, 0.3, -0.2, tilt])).position
        q = kin.inverse(target, camera_tilt=tilt)
        np.testing.assert_allclose(kin.forward(q).position, target, atol=1e-6)
        assert q[3] == pytest.approx(tilt)


def test_unreachable_far_target_rejected() -> None:
    """A target well outside the arm's reach raises out_of_reach."""
    with pytest.raises(kin.UnreachableError) as exc:
        kin.inverse(np.array([1.0, 0.0, 0.2]))
    assert exc.value.reason == "out_of_reach"
    assert not kin.is_reachable(np.array([1.0, 0.0, 0.2]))


def test_unreachable_too_close_rejected() -> None:
    """A target inside the inner reach annulus is rejected."""
    # Directly at the shoulder axis: distance 0 < |L1 - L2|.
    with pytest.raises(kin.UnreachableError) as exc:
        kin.inverse(np.array([0.0, 0.0, bj.BASE_HEIGHT]))
    assert exc.value.reason == "out_of_reach"


def test_behind_base_reachable_by_folding() -> None:
    """A target behind the base is reachable by folding (yaw stays in range)."""
    target = np.array([-0.05, 0.0, 0.20])
    q = kin.inverse(target)
    assert bj.within_limits(q)
    np.testing.assert_allclose(kin.forward(q).position, target, atol=1e-6)


def test_joint_limit_violation_caught() -> None:
    """An inner-annulus target is reachable only with the elbow past 90 deg."""
    # Just outside the shoulder axis (dist ~ 0.02): geometrically inside the
    # reach annulus, but the elbow would have to bend ~170 deg, past its limit.
    with pytest.raises(kin.UnreachableError) as exc:
        kin.inverse(np.array([0.0, 0.0, bj.BASE_HEIGHT + 0.02]))
    assert exc.value.reason == "joint_limits"


def test_servo_mapping_endpoints() -> None:
    """rad<->servo mapping hits the documented endpoints."""
    np.testing.assert_allclose(
        bj.radians_to_servo_degrees(np.array([0.0, np.pi / 2, -np.pi / 2, 0.0])),
        [90.0, 180.0, 0.0, 90.0],
    )
    # round trip
    q = np.array([0.1, -0.3, 0.5, -0.2])
    np.testing.assert_allclose(
        bj.servo_degrees_to_radians(bj.radians_to_servo_degrees(q)), q, atol=1e-9
    )


def test_clamp_and_within_limits() -> None:
    over = np.array([2.0, -2.0, 0.0, 0.0])
    clamped = bj.clamp_to_limits(over)
    np.testing.assert_allclose(clamped, [bj.JOINT_LIMIT, -bj.JOINT_LIMIT, 0.0, 0.0])
    assert bj.within_limits(clamped)
    assert not bj.within_limits(over)
