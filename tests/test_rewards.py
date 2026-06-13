"""Reward tests: term behaviour, the three modes, and sane scale bounds."""

from __future__ import annotations

import numpy as np

from rl_lab.env import rewards as R


def _cfg(mode: str, **kw) -> R.RewardConfig:
    return R.RewardConfig(mode=mode, success_tol=0.02, **kw)


def test_terms_have_expected_sign() -> None:
    assert R.distance_term(0.1) == -0.1
    assert R.success_term(0.01, tol=0.02) == R.SUCCESS_BONUS
    assert R.success_term(0.5, tol=0.02) == 0.0
    assert R.control_term(np.array([1.0, 0, 0, 0])) == -1.0
    assert R.joint_limit_term(np.zeros(4)) == 0.0
    assert R.joint_limit_term(np.full(4, 1.55)) < 0.0  # near the +/-pi/2 limit


def test_sparse_reward_only_on_success() -> None:
    far, _ = R.compute_reward(
        _cfg("sparse"), distance=0.2, prev_distance=0.2, action=np.zeros(4), q=np.zeros(4)
    )
    near, comp = R.compute_reward(
        _cfg("sparse"), distance=0.01, prev_distance=0.05, action=np.zeros(4), q=np.zeros(4)
    )
    assert far == 0.0
    assert near == R.SUCCESS_BONUS
    assert comp["is_success"] == 1.0


def test_dense_reward_decreases_with_distance() -> None:
    close, _ = R.compute_reward(
        _cfg("dense"), distance=0.05, prev_distance=0.05, action=np.zeros(4), q=np.zeros(4)
    )
    far, _ = R.compute_reward(
        _cfg("dense"), distance=0.25, prev_distance=0.25, action=np.zeros(4), q=np.zeros(4)
    )
    assert close > far  # nearer the target is better


def test_shaped_reward_rewards_progress() -> None:
    progress, _ = R.compute_reward(
        _cfg("shaped"), distance=0.10, prev_distance=0.20, action=np.zeros(4), q=np.zeros(4)
    )
    regress, _ = R.compute_reward(
        _cfg("shaped"), distance=0.20, prev_distance=0.10, action=np.zeros(4), q=np.zeros(4)
    )
    assert progress > 0 > regress  # moving toward the goal pays, away costs


def test_distance_is_reported_consistently() -> None:
    _, comp = R.compute_reward(
        _cfg("dense"), distance=0.123, prev_distance=0.2, action=np.zeros(4), q=np.zeros(4)
    )
    assert comp["distance"] == 0.123  # same value feeds reward and info


def test_reward_scale_is_bounded() -> None:
    """No single step should dwarf the +10 success bonus (kept on scale)."""
    for mode in ("sparse", "dense", "shaped"):
        r, _ = R.compute_reward(
            _cfg(mode), distance=0.3, prev_distance=0.0, action=np.ones(4), q=np.zeros(4)
        )
        assert -50.0 < r < R.SUCCESS_BONUS + 1.0
