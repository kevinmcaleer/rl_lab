"""Wrappers that let the *same* reach task run under every algorithm family.

The core :class:`~rl_lab.env.buddy_jr_reach_env.BuddyJrReachEnv` is continuous.
Thin wrappers adapt it so tabular Q-learning, DQN, PPO, SAC and HER can all be
compared honestly on identical physics, reward and visualization:

* :class:`DiscretizeBuddyJr` — ``Discrete(9)`` jog actions (for DQN / tabular).
* :class:`TabularBuddyJr`   — bins the observation to a single integer state
  (and makes the curse of dimensionality visible).
* :class:`ActionRepeat`, :class:`DomainRandomization` — general-purpose, plus
  thin re-exports of Gymnasium's ``NormalizeObservation`` / ``TimeLimit``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import NormalizeObservation, TimeLimit

from rl_lab.env import spaces as S

__all__ = [
    "DiscretizeBuddyJr",
    "TabularBuddyJr",
    "ActionRepeat",
    "DomainRandomization",
    "NormalizeObservation",
    "TimeLimit",
]


class DiscretizeBuddyJr(gym.ActionWrapper):
    """Expose a ``Discrete(9)`` jog that maps onto the same joint-target update."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = S.make_action_space(discrete=True)

    def action(self, action: int) -> np.ndarray:
        return S.discrete_to_continuous(int(action))


class TabularBuddyJr(gym.ObservationWrapper):
    """Bin the observation to one integer index — for tabular Q-learning.

    With ``obs_indices=None`` the *entire* observation is binned, so the state
    count is ``bins ** obs_dim`` — astronomically large. That is the lesson:
    tabular methods do not scale, which motivates function approximation (DQN).
    For a runnable tabular experiment, pass a small ``obs_indices`` subset
    (e.g. the tip->target vector).
    """

    def __init__(self, env: gym.Env, bins: int = 5, obs_indices: list[int] | None = None) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("TabularBuddyJr requires a Box observation space")
        self.bins = int(bins)
        n_dims = int(np.prod(env.observation_space.shape))
        self.obs_indices = list(range(n_dims)) if obs_indices is None else list(obs_indices)
        self.observation_space = gym.spaces.Discrete(self.bins ** len(self.obs_indices))

    def observation(self, observation: np.ndarray) -> int:
        obs = np.asarray(observation, dtype=np.float64)[self.obs_indices]
        # obs is scaled to [-1, 1]; map to a bin in [0, bins-1] per dim.
        idx = np.clip(((obs + 1.0) / 2.0 * self.bins).astype(int), 0, self.bins - 1)
        flat = 0
        for i in idx:
            flat = flat * self.bins + int(i)
        return flat


class ActionRepeat(gym.Wrapper):
    """Repeat each action ``n`` times, accumulating reward (frame-skip)."""

    def __init__(self, env: gym.Env, n: int = 2) -> None:
        super().__init__(env)
        self.n = max(1, int(n))

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        total = 0.0
        terminated = truncated = False
        obs: Any = None
        info: dict[str, Any] = {}
        for _ in range(self.n):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info


class DomainRandomization(gym.Wrapper):
    """Randomise the task to force generalisation / bridge the sim-to-real gap.

    Each toggle is independent and off by default:

    * ``action_noise_std`` — SG90-style command jitter added to every action.
    * ``obs_noise_std``    — sensor noise added to observations.
    * ``action_rate_limit``— clamp how fast an action can change (servo slew).
    * ``randomize_radius``  — resample the target-distribution radius each reset.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        action_noise_std: float = 0.0,
        obs_noise_std: float = 0.0,
        action_rate_limit: float | None = None,
        randomize_radius: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(env)
        self.action_noise_std = float(action_noise_std)
        self.obs_noise_std = float(obs_noise_std)
        self.action_rate_limit = action_rate_limit
        self.randomize_radius = randomize_radius
        self._prev_action: np.ndarray | None = None

    def _noisy_obs(self, obs: Any) -> Any:
        if self.obs_noise_std > 0.0 and isinstance(obs, np.ndarray):
            obs = np.clip(obs + self.np_random.normal(0, self.obs_noise_std, obs.shape), -1.0, 1.0)
            return obs.astype(np.float32)
        return obs

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if self.randomize_radius is not None and hasattr(self.unwrapped, "target_radius"):
            lo, hi = self.randomize_radius
            mid = self.np_random.uniform(lo, hi)
            self.unwrapped.target_radius = (lo, max(mid, lo + 0.01))
        self._prev_action = None
        obs, info = self.env.reset(**kwargs)
        return self._noisy_obs(obs), info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float64)
        if self.action_noise_std > 0.0:
            a = a + self.np_random.normal(0, self.action_noise_std, a.shape)
        if self.action_rate_limit is not None and self._prev_action is not None:
            lim = self.action_rate_limit
            a = np.clip(a, self._prev_action - lim, self._prev_action + lim)
        a = np.clip(a, -1.0, 1.0)
        self._prev_action = a
        obs, reward, terminated, truncated, info = self.env.step(a)
        return self._noisy_obs(obs), float(reward), terminated, truncated, info
