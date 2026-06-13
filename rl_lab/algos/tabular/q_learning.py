"""Tabular Q-learning for the Buddy Jr reach task — pure NumPy, fully commented.

Q-learning is the canonical *off-policy* temporal-difference (TD) algorithm
(Watkins & Dayan, 1992).  "Off-policy" means the policy used to *collect* data
(epsilon-greedy) is different from the policy being *improved* (greedy).

The update rule is the *Bellman optimality equation* applied one step at a time::

    Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a')  −  Q(s, a)]
                                └──────── TD target ────────┘
                          └───────────── TD error (δ) ────────────────┘

Terms
-----
* Q(s, a)              — current estimate of the expected return when taking
                          action *a* in state *s* and acting optimally after.
* α   (alpha)          — learning rate: how much to shift the estimate toward
                          the new target on each update (0 < α ≤ 1).
* γ   (gamma)          — discount factor: how much future rewards are worth
                          relative to immediate ones (0 ≤ γ ≤ 1).
* r                    — reward received after taking action *a* in state *s*.
* s'                   — the next state after taking that action.
* max_{a'} Q(s', a')   — the Q-value of the *best* action in s' under the
                          *current* estimate — this is the "off-policy" part.
                          Q-learning always bootstraps from the greedy value
                          regardless of which action was actually taken next.

Because the workspace is continuous (17-D Box), we first discretise the
observation with :class:`~rl_lab.env.wrappers.TabularBuddyJr`, which bins the
tip→target error vector (obs indices 14–16) into ``bins``³ = 343 states with
default ``bins=7``.  The action space is already ``Discrete(9)`` on a
``BuddyJrReachDiscrete-v0`` env, or is left intact here so the wrapper handles
the discrete actions too.

This file is intentionally verbose — every equation appears in the code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from rl_lab.env.wrappers import TabularBuddyJr
from rl_lab.utils.seeding import set_global_seed


class QLearning:
    """Off-policy tabular Q-learning.

    Parameters
    ----------
    env:
        A Gymnasium environment.  If its ``observation_space`` is a continuous
        ``Box``, it is automatically wrapped with
        :class:`~rl_lab.env.wrappers.TabularBuddyJr` (binning obs indices
        14–16, the tip→target vector).
    seed:
        Master RNG seed for reproducibility.
    alpha:
        Learning rate α — how aggressively to shift Q toward new TD targets.
        Typical range 0.01–0.5.  Too large ⇒ oscillation; too small ⇒ slow.
    gamma:
        Discount factor γ — how much future rewards are worth.  Values close
        to 1.0 make the agent far-sighted; 0.0 makes it myopic.
    epsilon:
        Initial exploration probability for ε-greedy action selection.
    epsilon_min:
        Floor for ε — we always keep some exploration to avoid getting stuck.
    epsilon_decay:
        Per-episode multiplicative decay of ε (ε ← ε · epsilon_decay).
    bins:
        Number of bins per observation dimension fed to ``TabularBuddyJr``.
        With ``obs_indices=[14,15,16]`` the Q-table has ``bins³`` rows.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        seed: int = 0,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        bins: int = 7,
    ) -> None:
        self.seed = seed
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # ── Observation discretisation ────────────────────────────────────────
        # If the env already exposes a Discrete observation (e.g. because the
        # user pre-wrapped it), use it directly.  Otherwise wrap it so we get a
        # single integer state index.  We only bin the tip→target error vector
        # (dims 14–16) — the full 17-D obs would yield bins^17 states, which is
        # impossibly large (the curse of dimensionality in action).
        if isinstance(env.observation_space, gym.spaces.Box):
            self.env: Any = TabularBuddyJr(env, bins=bins, obs_indices=[14, 15, 16])
        else:
            self.env = env  # already discrete

        # ── Q-table dimensions ────────────────────────────────────────────────
        # n_states : number of distinct discrete state indices (bins^3 = 343)
        # n_actions: size of the action space (Discrete(9) = 9)
        self.n_states: int = int(self.env.observation_space.n)  # type: ignore[attr-defined]
        self.n_actions: int = int(self.env.action_space.n)  # type: ignore[attr-defined]

        # ── Q-table initialisation ────────────────────────────────────────────
        # Initialise to zero: we know nothing, so we assume zero expected return
        # for every (state, action) pair.  An optimistic initialisation (e.g.
        # fill with a positive constant) encourages exploration automatically,
        # but zero is the simplest and clearest starting point.
        self.q: np.ndarray = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

        # Seed every RNG source for reproducibility.
        set_global_seed(seed, self.env)
        self._rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        total_steps: int,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run Q-learning for *total_steps* environment steps.

        The agent interacts with the environment in complete episodes.  At the
        end of each episode ε is decayed so the agent explores less over time
        (exploration→exploitation trade-off).

        Parameters
        ----------
        total_steps:
            Approximate number of env ``step()`` calls.  Training stops as soon
            as the running step count meets or exceeds this value.
        callback:
            Optional function called at the end of every episode with a metrics
            dict::

                {"step": int, "episode": int, "episode_return": float,
                 "epsilon": float, "success_rate": float}

        Returns
        -------
        dict with keys:
            ``episode_returns`` — list of per-episode cumulative rewards,
            ``q_table_shape``   — shape of the final Q-table (n_states × n_actions),
            ``epsilon``         — final exploration probability.
        """
        episode_returns: list[float] = []
        # Track recent successes (rolling window) to report a smooth rate.
        recent_successes: list[float] = []
        _window = 50

        step_count = 0  # total env steps taken so far
        episode = 0  # total episodes completed so far

        while step_count < total_steps:
            # ── Episode start ─────────────────────────────────────────────────
            obs, _ = self.env.reset()
            # obs is already a discrete int because of the TabularBuddyJr wrapper.
            state: int = int(obs)
            episode_return = 0.0
            done = False

            while not done:
                # ── ε-greedy action selection ─────────────────────────────────
                # With probability ε pick a random action (explore the state
                # space); with probability 1−ε pick the greedy action
                # argmax_{a} Q(s, a) (exploit current knowledge).
                if self._rng.random() < self.epsilon:
                    action = int(self.env.action_space.sample())
                else:
                    action = int(np.argmax(self.q[state]))

                # ── Environment step ──────────────────────────────────────────
                next_obs, reward_raw, terminated, truncated, info = self.env.step(action)
                reward = float(reward_raw)  # cast SupportsFloat → float
                next_state: int = int(next_obs)
                done = terminated or truncated

                # ── Q-learning (off-policy) Bellman update ────────────────────
                #
                # TD target:   r + γ · max_{a'} Q(s', a')
                # TD error δ:  TD_target − Q(s, a)
                # Update:      Q(s, a) ← Q(s, a) + α · δ
                #
                # KEY INSIGHT: we bootstrap from the *maximum* Q-value in s',
                # NOT the Q-value of the action that was actually chosen next.
                # This is what makes Q-learning off-policy — the update always
                # assumes optimal future behaviour, independent of the
                # (possibly exploratory) action the agent will actually take.
                #
                # When the episode terminates (reached goal or time-limit) there
                # is no next state, so the target is just r (no future reward).
                if terminated:
                    # Terminal state: no future reward — target is just r.
                    td_target = reward
                else:
                    # Non-terminal: bootstrap from the greedy value in s'.
                    #   max_{a'} Q(s', a') is the best Q-value we believe we
                    #   can achieve from s' onward.
                    best_next_q = float(np.max(self.q[next_state]))
                    td_target = reward + self.gamma * best_next_q

                # TD error (also called the prediction error or δ):
                # positive δ ⇒ the outcome was better than expected → increase Q
                # negative δ ⇒ the outcome was worse than expected  → decrease Q
                td_error = td_target - self.q[state, action]

                # Apply the update — the learning rate α controls the step size.
                self.q[state, action] = self.q[state, action] + self.alpha * td_error
                # ── end Bellman update ─────────────────────────────────────────

                episode_return += reward
                state = next_state
                step_count += 1

                # Honour the step budget mid-episode if we overshoot.
                if step_count >= total_steps:
                    break

            # ── End-of-episode bookkeeping ────────────────────────────────────
            episode_returns.append(episode_return)
            success = float(info.get("is_success", False))
            recent_successes.append(success)
            if len(recent_successes) > _window:
                recent_successes.pop(0)
            success_rate = float(np.mean(recent_successes))

            # Decay ε: after each episode we are slightly more confident in Q,
            # so we reduce exploration.  The floor epsilon_min keeps diversity.
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            episode += 1

            if callback is not None:
                callback(
                    {
                        "step": step_count,
                        "episode": episode,
                        "episode_return": episode_return,
                        "epsilon": self.epsilon,
                        "success_rate": success_rate,
                    }
                )

        return {
            "episode_returns": episode_returns,
            "q_table_shape": self.q.shape,
            "epsilon": self.epsilon,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        """Return the greedy (or ε-greedy) action for an observation.

        Parameters
        ----------
        observation:
            Either an already-discretised integer state, or a raw Box obs
            array (it will be run through the wrapper's discretiser).
        deterministic:
            If ``True`` (default for evaluation), always pick argmax Q.
            If ``False``, use the current ε-greedy policy (for rollouts).

        Returns
        -------
        ``(action, None)`` — the action to take and a dummy state (to match the
        :class:`~rl_lab.algos.base.Algorithm` protocol).
        """
        # Discretise if we received a raw Box observation array.
        if isinstance(observation, (np.ndarray, list)):
            state = int(self.env.observation(np.asarray(observation, dtype=np.float64)))
        else:
            state = int(observation)

        if not deterministic and self._rng.random() < self.epsilon:
            return int(self.env.action_space.sample()), None

        return int(np.argmax(self.q[state])), None

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the Q-table to a compressed NumPy archive at *path*.npz.

        np.savez automatically appends the ``.npz`` suffix — do not add it
        yourself or you will get ``path.npz.npz``.
        """
        np.savez(path, q=self.q)

    def load(self, path: str) -> None:
        """Load the Q-table from a NumPy archive.

        Handles both ``path`` and ``path.npz`` (``np.savez`` appends the
        suffix, so callers often omit it).
        """
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path)
        self.q = data["q"]

    # ─────────────────────────────────────────────────────────────────────────
    # Extras
    # ─────────────────────────────────────────────────────────────────────────

    def dump_q_table(self) -> np.ndarray:
        """Return a copy of the Q-table for inspection / plotting.

        Shape: ``(n_states, n_actions)`` — rows are discretised states,
        columns are the 9 discrete jog actions.  The value at ``[s, a]`` is
        the estimated expected discounted return when taking action *a* in
        state *s* and then acting greedily.
        """
        return self.q.copy()
