"""Environment API tests: check_env on every registered id + the Gym contract."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import rl_lab  # noqa: F401  (registers the envs on import)
from rl_lab.env.buddy_jr_reach_env import BuddyJrReachEnv
from rl_lab.env.wrappers import DomainRandomization

REGISTERED = ["BuddyJrReach-v0", "BuddyJrReachDiscrete-v0", "BuddyJrCameraPoint-v0"]


@pytest.mark.parametrize("env_id", REGISTERED)
def test_registered_env_passes_check_env(env_id: str) -> None:
    env = gym.make(env_id)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", REGISTERED)
def test_gymnasium_make_works(env_id: str) -> None:
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()


def test_info_carries_required_keys() -> None:
    env = BuddyJrReachEnv()
    _obs, info = env.reset(seed=1)
    for key in ("distance", "is_success", "joint_q", "ee_pos", "target"):
        assert key in info
    _obs, _r, _term, _trunc, info = env.step(env.action_space.sample())
    for key in ("distance", "is_success", "joint_q", "ee_pos", "target"):
        assert key in info
    env.close()


def test_terminated_and_truncated_are_separate() -> None:
    # Truncation: a short horizon with a no-op action never succeeds -> truncates.
    env = BuddyJrReachEnv(max_steps=5)
    env.reset(seed=2)
    term = trunc = False
    for _ in range(5):
        _o, _r, term, trunc, _i = env.step(np.zeros(4))
    assert trunc is True and term is False
    env.close()

    # Termination: place the target on the current tip -> success -> terminated.
    env = BuddyJrReachEnv(max_steps=100)
    env.reset(seed=3)
    env.unwrapped._target = env.unwrapped._ee_pos().copy()
    _o, _r, term, trunc, info = env.step(np.zeros(4))
    assert term is True and trunc is False and info["is_success"] is True
    env.close()


def test_discrete_variant_shares_task() -> None:
    env = gym.make("BuddyJrReachDiscrete-v0")
    assert isinstance(env.action_space, gym.spaces.Discrete) and env.action_space.n == 9
    obs, _ = env.reset(seed=4)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert "distance" in info
    env.close()


def test_goal_env_compute_reward() -> None:
    env = BuddyJrReachEnv(goal_env=True)
    obs, _ = env.reset(seed=5)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    hit = env.compute_reward(obs["desired_goal"], obs["desired_goal"], {})
    miss = env.compute_reward(obs["desired_goal"] + 1.0, obs["desired_goal"], {})
    assert float(hit) == 0.0 and float(miss) == -1.0
    env.close()


def test_camera_point_reports_alignment() -> None:
    env = BuddyJrReachEnv(camera_point=True)
    env.reset(seed=6)
    _o, _r, _t, _tr, info = env.step(np.zeros(4))
    assert "alignment" in info and -1.0001 <= info["alignment"] <= 1.0001
    env.close()


def test_domain_randomization_composes() -> None:
    env = DomainRandomization(
        BuddyJrReachEnv(), action_noise_std=0.05, obs_noise_std=0.02, action_rate_limit=0.2
    )
    obs, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    for _ in range(10):
        obs, _r, term, trunc, _i = env.step(env.action_space.sample())
        if term or trunc:
            break
    assert env.observation_space.contains(obs)
    env.close()
