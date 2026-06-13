"""Tabular SARSA for the Buddy Jr reach task — pure NumPy, fully commented.

SARSA is the canonical *on-policy* temporal-difference (TD) algorithm
(Rummery & Niranjan, 1994; Sutton & Barto, 2018, §6.4).  Its name is an
acronym for the five quantities used in each update step:

    S — current state
    A — action taken in that state
    R — reward received
    S'— next state
    A'— action taken in the NEXT state (the key difference from Q-learning)

The update rule is::

    Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a')  −  Q(s, a)]
                                └─── TD target ───┘
                          └────────── TD error (δ) ───────────┘

Contrast with Q-learning (see ``q_learning.py``) where the TD target uses
``max_{a'} Q(s', a')`` (the *best possible* next Q-value), ignoring which
action is actually chosen next.  SARSA instead uses ``Q(s', a')`` — the value
of the action ``a'`` that the *same* ε-greedy policy will actually pick.  This
makes SARSA on-policy: the value function it learns is the one that describes
the ε-greedy behaviour policy, including its exploratory detours.

Consequences of on-policy vs off-policy
----------------------------------------
* During training with a risky (large ε) policy, SARSA "knows" about the
  risk of exploration and tends to find safer paths.
* Q-learning optimises for the greedy policy regardless, so it can appear to
  ignore risk near dangerous states and converge to a potentially riskier path
  in cliff-walking style tasks.
* After epsilon decays to epsilon_min both algorithms converge to similar
  (near-optimal) solutions — the difference matters most early in training.
* On simple continuous-control tasks like Buddy Jr reach, the practical
  performance gap is small; the comparison is primarily educational.

This file is intentionally verbose — every equation appears in the code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from rl_lab.env.wrappers import TabularBuddyJr
from rl_lab.utils.seeding import set_global_seed


class SARSA:
    """On-policy tabular SARSA.

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
    gamma:
        Discount factor γ — how much future rewards are worth.
    epsilon:
        Initial exploration probability for ε-greedy action selection.
    epsilon_min:
        Floor for ε — we always keep some exploration.
    epsilon_decay:
        Per-episode multiplicative decay of ε (ε ← ε · epsilon_decay).
    bins:
        Number of bins per observation dimension fed to ``TabularBuddyJr``.
        With ``obs_indices=[14,15,16]`` the Q-table has ``bins³`` rows.

    Notes
    -----
    SARSA and Q-learning share identical hyper-parameters and interfaces so
    you can run them side-by-side with the same seed and compare learning
    curves directly via the experiment notebooks.
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
        # Identical wrapping logic to QLearning: bin only the tip→target error
        # vector (obs dims 14–16) to keep the table to 7³ = 343 rows.
        if isinstance(env.observation_space, gym.spaces.Box):
            self.env: Any = TabularBuddyJr(env, bins=bins, obs_indices=[14, 15, 16])
        else:
            self.env = env

        # ── Q-table dimensions ────────────────────────────────────────────────
        self.n_states: int = int(self.env.observation_space.n)  # type: ignore[attr-defined]
        self.n_actions: int = int(self.env.action_space.n)  # type: ignore[attr-defined]

        # ── Q-table initialisation ────────────────────────────────────────────
        # Zero-initialised: same as Q-learning for a fair comparison.
        self.q: np.ndarray = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

        set_global_seed(seed, self.env)
        self._rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _epsilon_greedy(self, state: int) -> int:
        """Sample an action from the ε-greedy policy for *state*.

        Factored out so we can call it in two places per step (for *a* and *a'*)
        without repeating logic.  The on-policy nature of SARSA depends on both
        *a* and *a'* being drawn from the **same** policy.
        """
        if self._rng.random() < self.epsilon:
            return int(self.env.action_space.sample())
        return int(np.argmax(self.q[state]))

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        total_steps: int,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run SARSA for *total_steps* environment steps.

        SARSA requires the next action *a'* to be chosen **before** the update,
        so the loop structure is slightly different from Q-learning: at the top
        of each step we already hold the action for the current state (chosen at
        the end of the previous step, or at episode start).

        Parameters
        ----------
        total_steps:
            Approximate number of env ``step()`` calls.
        callback:
            Optional function called at the end of every episode with::

                {"step": int, "episode": int, "episode_return": float,
                 "epsilon": float, "success_rate": float}

        Returns
        -------
        dict with keys:
            ``episode_returns`` — list of per-episode cumulative rewards,
            ``q_table_shape``   — shape of the final Q-table,
            ``epsilon``         — final exploration probability.
        """
        episode_returns: list[float] = []
        recent_successes: list[float] = []
        _window = 50

        step_count = 0
        episode = 0

        while step_count < total_steps:
            # ── Episode start ─────────────────────────────────────────────────
            obs, _ = self.env.reset()
            state: int = int(obs)
            episode_return = 0.0

            # SARSA pre-selects the first action *before* the loop.
            # This is the critical structural difference: we must know *a* (and
            # later *a'*) from the same ε-greedy policy *before* we update Q.
            action: int = self._epsilon_greedy(state)

            done = False
            while not done:
                # ── Environment step ──────────────────────────────────────────
                next_obs, reward_raw, terminated, truncated, info = self.env.step(action)
                reward = float(reward_raw)  # cast SupportsFloat → float
                next_state: int = int(next_obs)
                done = terminated or truncated

                # ── Choose a' BEFORE the update ───────────────────────────────
                # This is the heart of on-policy learning.  We choose *a'* using
                # the SAME ε-greedy policy that chose *a*.  The Q-value we use
                # in the TD target, Q(s', a'), corresponds to what we *actually
                # plan to do* in s' — not an idealised greedy action as in
                # Q-learning.
                #
                # On terminal transitions there is no next state, so a' and
                # Q(s', a') are both irrelevant (target = r only).
                if not terminated:
                    next_action: int = self._epsilon_greedy(next_state)
                else:
                    next_action = 0  # placeholder — value unused in update

                # ── SARSA (on-policy) Bellman update ──────────────────────────
                #
                # TD target:   r + γ · Q(s', a')       ← uses the ACTUAL next action
                # TD error δ:  TD_target − Q(s, a)
                # Update:      Q(s, a) ← Q(s, a) + α · δ
                #
                # Compare with Q-learning target: r + γ · max_{a''} Q(s', a'')
                # Q-learning always assumes the best next action; SARSA assumes
                # the next action the *current policy* will actually take.
                #
                # When ε → 0, both updates become equivalent (the greedy
                # action is also the argmax), so the two algorithms converge
                # to the same optimal policy in the limit.
                if terminated:
                    # Terminal state: no future reward — target is just r.
                    td_target = reward
                else:
                    # On-policy bootstrap: use Q of the *actual* next action a'.
                    td_target = reward + self.gamma * self.q[next_state, next_action]

                # TD error: signed difference between what we expected and got.
                td_error = td_target - self.q[state, action]

                # Update Q towards the TD target by a fraction α.
                self.q[state, action] = self.q[state, action] + self.alpha * td_error
                # ── end SARSA update ───────────────────────────────────────────

                episode_return += reward

                # Advance: the next action we pre-selected becomes the current
                # action in the next iteration (SARSA's defining control flow).
                state = next_state
                action = next_action  # ← carry *a'* forward as the next *a*

                step_count += 1
                if step_count >= total_steps:
                    break

            # ── End-of-episode bookkeeping ────────────────────────────────────
            episode_returns.append(episode_return)
            success = float(info.get("is_success", False))
            recent_successes.append(success)
            if len(recent_successes) > _window:
                recent_successes.pop(0)
            success_rate = float(np.mean(recent_successes))

            # Decay ε per episode — same schedule as Q-learning for fair comparison.
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
        ``(action, None)`` — to match the :class:`~rl_lab.algos.base.Algorithm`
        protocol (state is unused by tabular methods).
        """
        if isinstance(observation, (np.ndarray, list)):
            state = int(self.env.observation(np.asarray(observation, dtype=np.float64)))
        else:
            state = int(observation)

        if not deterministic:
            return self._epsilon_greedy(state), None

        return int(np.argmax(self.q[state])), None

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the Q-table to a compressed NumPy archive at *path*.npz."""
        np.savez(path, q=self.q)

    def load(self, path: str) -> None:
        """Load the Q-table from a NumPy archive (handles optional .npz suffix)."""
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
        columns are the 9 discrete jog actions.  Compare this table with the
        one from :class:`~rl_lab.algos.tabular.q_learning.QLearning` trained
        for the same number of steps: the SARSA table is typically more
        conservative near the edges of the workspace because the exploratory
        policy occasionally "falls off" and SARSA accounts for that.
        """
        return self.q.copy()
