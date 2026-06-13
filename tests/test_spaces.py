"""Observation/action space builder tests: dimensionality, scaling, bounds."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl_lab.env import spaces as S
from rl_lab.robot import buddy_jr as bj


def test_observation_dimensionality() -> None:
    assert S.make_observation_space().shape == (17,)
    assert S.make_observation_space(include_velocity=True).shape == (21,)


def test_observation_is_normalised_to_unit_range() -> None:
    """Every component lands in [-1, 1] even for extreme inputs."""
    q = np.array([bj.JOINT_LIMIT, -bj.JOINT_LIMIT, 0.5, -0.5])
    qd = np.full(4, 100.0)  # absurd velocity
    ee = np.array([0.2, -0.2, 0.25])
    target = np.array([-0.2, 0.2, 0.02])
    obs = S.build_observation(q, qd, ee, target, include_velocity=True)
    assert obs.shape == (21,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


def test_sincos_encoding_is_continuous_across_pi() -> None:
    """sin/cos avoids the wraparound jump that raw angles would show."""
    a = S.build_observation(np.array([3.10, 0, 0, 0]), np.zeros(4), np.zeros(3), np.zeros(3))
    b = S.build_observation(np.array([-3.10, 0, 0, 0]), np.zeros(4), np.zeros(3), np.zeros(3))
    # angles ~pi apart in raw terms, but sin/cos are nearly identical.
    assert np.allclose(a[:8], b[:8], atol=0.1)


def test_action_spaces() -> None:
    cont = S.make_action_space(discrete=False)
    assert isinstance(cont, gym.spaces.Box) and cont.shape == (4,)
    assert np.all(cont.low == -1.0) and np.all(cont.high == 1.0)
    disc = S.make_action_space(discrete=True)
    assert isinstance(disc, gym.spaces.Discrete) and disc.n == 9


def test_discrete_to_continuous_mapping() -> None:
    assert np.allclose(S.discrete_to_continuous(0), np.zeros(4))  # no-op
    # indices 1..8 each nudge exactly one joint by +/-1
    for a in range(1, 9):
        v = S.discrete_to_continuous(a)
        assert np.count_nonzero(v) == 1 and abs(v.sum()) == 1.0


def test_goal_env_space_is_dict() -> None:
    space = S.make_observation_space(goal_env=True)
    assert isinstance(space, gym.spaces.Dict)
    assert set(space.spaces) == {"observation", "achieved_goal", "desired_goal"}
