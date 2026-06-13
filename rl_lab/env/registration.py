"""Register the Buddy Jr environments with Gymnasium.

:func:`register_envs` runs on ``import rl_lab`` (idempotently), so after that
``gymnasium.make("BuddyJrReach-v0")`` and friends just work. Three ids:

* ``BuddyJrReach-v0``         — continuous reach (PPO / SAC).
* ``BuddyJrReachDiscrete-v0`` — ``Discrete(9)`` jog (DQN / tabular).
* ``BuddyJrCameraPoint-v0``   — reach + camera-pointing variant.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from rl_lab.env.buddy_jr_reach_env import BuddyJrReachEnv
from rl_lab.env.wrappers import DiscretizeBuddyJr

_DEFAULT_MAX_STEPS = 200


def make_reach(**kwargs: Any) -> gym.Env:
    """Factory for the continuous reach env."""
    return BuddyJrReachEnv(**kwargs)


def make_reach_discrete(**kwargs: Any) -> gym.Env:
    """Factory for the discrete-jog reach env (same task, Discrete(9) actions)."""
    return DiscretizeBuddyJr(BuddyJrReachEnv(**kwargs))


def make_camera_point(**kwargs: Any) -> gym.Env:
    """Factory for the camera-pointing variant (rewards aiming at the target)."""
    kwargs.setdefault("camera_point", True)
    return BuddyJrReachEnv(**kwargs)


_REGISTRY = {
    "BuddyJrReach-v0": "rl_lab.env.registration:make_reach",
    "BuddyJrReachDiscrete-v0": "rl_lab.env.registration:make_reach_discrete",
    "BuddyJrCameraPoint-v0": "rl_lab.env.registration:make_camera_point",
}


def register_envs() -> None:
    """Register all Buddy Jr env ids (safe to call more than once)."""
    for env_id, entry_point in _REGISTRY.items():
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point=entry_point,
            max_episode_steps=_DEFAULT_MAX_STEPS,
        )
