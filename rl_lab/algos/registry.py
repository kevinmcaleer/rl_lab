"""Algorithm registry for the Buddy Jr RL Lab.

This module is the single place that maps a short algorithm *name* (e.g.
``"dqn"``) to the class that implements it.  Critically, every import is
**lazy** — the module-level ``import rl_lab.algos.registry`` never pulls in
torch, gymnasium, or stable-baselines3 unless ``make_algorithm`` is actually
called.  That keeps CLI startup fast and lets each algorithm's heavy deps be
optional.

Public API
----------
    ALGORITHMS          -- tuple of supported name strings.
    make_algorithm()    -- factory: name + env -> Algorithm instance.
    recommended_env_id()-- returns the best-fit env id for a given algo name.

Typical usage::

    import gymnasium as gym
    from rl_lab.algos.registry import make_algorithm, recommended_env_id

    env_id = recommended_env_id("dqn")
    env = gym.make(env_id)
    algo = make_algorithm("dqn", env, seed=42, lr=1e-3)
    history = algo.train(total_steps=50_000)
    action, _ = algo.predict(obs)
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------
# This tuple is the single source of truth for valid algorithm names.
# The smoke tests, the CLI, and the documentation all derive their lists from
# here — so adding a new algorithm is a one-liner in the mapping below plus
# an entry in ALGORITHMS.

ALGORITHMS: tuple[str, ...] = (
    "qlearning",  # tabular Q-learning  — pure NumPy, discrete env
    "sarsa",  # tabular SARSA        — pure NumPy, discrete env
    "dqn",  # Deep Q-Network       — PyTorch, discrete env
    "reinforce",  # REINFORCE (Monte-Carlo policy gradient) — PyTorch
    "ppo_min",  # Minimal PPO          — PyTorch, teaching implementation
    "ppo",  # Stable-Baselines3 PPO  — continuous env
    "sac",  # Stable-Baselines3 SAC  — continuous env
    "td3",  # Stable-Baselines3 TD3  — continuous env
    "ddpg",  # Stable-Baselines3 DDPG — continuous env
)

# ---------------------------------------------------------------------------
# Env-id recommendations
# ---------------------------------------------------------------------------
# Tabular and value-based methods that need a *discrete* action space work
# best on BuddyJrReachDiscrete-v0 (Discrete(9) jog actions, obs Box(17,)).
# Continuous-action methods (PPO/SAC/TD3/DDPG) use BuddyJrReach-v0
# (action Box(4,), obs Box(17,)).

_DISCRETE_ENV_ID: str = "BuddyJrReachDiscrete-v0"
_CONTINUOUS_ENV_ID: str = "BuddyJrReach-v0"

# Algorithms that need a discrete action space:
_DISCRETE_ALGOS: frozenset[str] = frozenset({"qlearning", "sarsa", "dqn", "reinforce", "ppo_min"})


def recommended_env_id(name: str) -> str:
    """Return the Gymnasium env id best suited for *name*.

    Parameters
    ----------
    name:
        Algorithm name, case-sensitive, must be in :data:`ALGORITHMS`.

    Returns
    -------
    str
        ``"BuddyJrReachDiscrete-v0"`` for tabular / discrete-action algorithms
        (qlearning, sarsa, dqn, reinforce, ppo_min); ``"BuddyJrReach-v0"``
        for continuous-action algorithms (ppo, sac, td3, ddpg).

    Raises
    ------
    ValueError
        If *name* is not in :data:`ALGORITHMS`.

    Examples
    --------
    >>> recommended_env_id("dqn")
    'BuddyJrReachDiscrete-v0'
    >>> recommended_env_id("sac")
    'BuddyJrReach-v0'
    """
    _validate_name(name)
    return _DISCRETE_ENV_ID if name in _DISCRETE_ALGOS else _CONTINUOUS_ENV_ID


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_name(name: str) -> None:
    """Raise :exc:`ValueError` with a helpful message if *name* is unknown."""
    if name not in ALGORITHMS:
        known = ", ".join(ALGORITHMS)
        raise ValueError(f"Unknown algorithm {name!r}.  " f"Valid names are: {known}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_algorithm(
    name: str,
    env: Any,
    *,
    seed: int = 0,
    **hparams: Any,
) -> Any:  # -> Algorithm (Protocol — avoids importing all heavy deps at module level)
    """Instantiate and return the algorithm called *name*.

    All imports are **lazy** so this module remains importable on any machine
    regardless of whether torch or stable-baselines3 are installed — the
    ImportError is raised only when the relevant algorithm is requested.

    Parameters
    ----------
    name:
        Algorithm name — must be one of :data:`ALGORITHMS`.
    env:
        A Gymnasium environment instance (already constructed by the caller).
        The Algorithm constructor receives it as its first positional argument.
    seed:
        Global random seed forwarded to the Algorithm constructor.
    **hparams:
        Extra keyword hyper-parameters forwarded verbatim to the constructor
        (e.g. ``lr=3e-4``, ``gamma=0.99``, ``batch_size=64``).

    Returns
    -------
    Algorithm
        A freshly constructed algorithm instance that satisfies the
        :class:`~rl_lab.algos.base.Algorithm` protocol.

    Raises
    ------
    ValueError
        If *name* is not in :data:`ALGORITHMS`.
    ImportError
        If the requested algorithm's optional dependencies (torch,
        stable-baselines3) are not installed.

    Examples
    --------
    >>> import gymnasium as gym
    >>> from rl_lab.algos.registry import make_algorithm
    >>> env = gym.make("BuddyJrReachDiscrete-v0")
    >>> algo = make_algorithm("qlearning", env, seed=0, lr=0.1, gamma=0.99)
    >>> algo.train(total_steps=5000)  # doctest: +SKIP
    """
    _validate_name(name)

    # ------------------------------------------------------------------
    # Tabular algorithms — pure NumPy, no torch required.
    # ------------------------------------------------------------------

    if name == "qlearning":
        # Tabular Q-learning:
        #   Q(s,a) <- Q(s,a) + lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        # Implemented in rl_lab.algos.tabular.q_learning.
        from rl_lab.algos.tabular.q_learning import QLearning  # noqa: PLC0415

        return QLearning(env, seed=seed, **hparams)

    if name == "sarsa":
        # Tabular SARSA (on-policy TD(0)):
        #   Q(s,a) <- Q(s,a) + lr * [r + gamma * Q(s',a') - Q(s,a)]
        # Unlike Q-learning the *next* action a' is sampled from the *same*
        # policy (on-policy), not the greedy one (off-policy).
        from rl_lab.algos.tabular.sarsa import SARSA  # noqa: PLC0415

        return SARSA(env, seed=seed, **hparams)

    # ------------------------------------------------------------------
    # Value-based deep RL — requires torch.
    # ------------------------------------------------------------------

    if name == "dqn":
        # Deep Q-Network (Mnih et al., 2015):
        #   Uses a neural network to approximate Q(s,a) for all actions
        #   simultaneously.  Adds experience replay + a target network to
        #   stabilise training.
        from rl_lab.algos.value_based.dqn import DQN  # noqa: PLC0415

        return DQN(env, seed=seed, **hparams)

    # ------------------------------------------------------------------
    # Policy-gradient deep RL — requires torch.
    # ------------------------------------------------------------------

    if name == "reinforce":
        # REINFORCE (Williams, 1992) — Monte-Carlo policy gradient:
        #   nabla J(theta) = E[nabla log pi(a|s) * G_t]
        # where G_t is the discounted return from time t.
        # Simple but high-variance; no critic, no value baseline.
        from rl_lab.algos.policy_gradient.reinforce import REINFORCE  # noqa: PLC0415

        return REINFORCE(env, seed=seed, **hparams)

    if name == "ppo_min":
        # Minimal PPO (Schulman et al., 2017) — clipped surrogate objective:
        #   L_CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
        # where r_t = pi(a|s) / pi_old(a|s) is the probability ratio and
        # A_t is the generalised advantage estimate (GAE).
        # This teaching implementation keeps the code readable over optimal.
        from rl_lab.algos.policy_gradient.ppo_min import PPOMin  # noqa: PLC0415

        return PPOMin(env, seed=seed, **hparams)

    # ------------------------------------------------------------------
    # Stable-Baselines3 wrappers — ppo / sac / td3 / ddpg.
    # ------------------------------------------------------------------
    # All four are routed through a single SB3Algorithm adapter that
    # accepts ``algo=name`` as a keyword so the registry entry point stays
    # clean.  SB3 manages its own random seeding internally.
    #
    # ppo  — Proximal Policy Optimisation (on-policy, clip objective)
    # sac  — Soft Actor-Critic (off-policy, entropy-regularised)
    # td3  — Twin Delayed DDPG (off-policy, deterministic, double critic)
    # ddpg — Deep Deterministic Policy Gradient (off-policy, deterministic)

    # name must be one of {"ppo", "sac", "td3", "ddpg"} at this point because
    # _validate_name() already accepted it and earlier branches handled the
    # remaining names.
    from rl_lab.algos.sb3_integration import SB3Algorithm  # noqa: PLC0415

    return SB3Algorithm(env, algo=name, seed=seed, **hparams)
