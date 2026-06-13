"""Stable-Baselines3 integration for the Buddy Jr RL Lab.

This module wraps four SB3 algorithms — PPO, SAC, TD3, DDPG — behind the
lab's :class:`~rl_lab.algos.base.Algorithm` protocol so that the training
CLI, the smoke-tests and the Jupyter notebooks can treat them identically to
the from-scratch implementations (Q-learning, SARSA, DQN, REINFORCE, PPOMin).

Why use SB3 at all?
--------------------
SB3 provides battle-tested, publication-quality implementations of the most
commonly used deep RL algorithms.  By exposing them through the same thin
protocol as our teaching implementations, learners can:

1. Compare their hand-coded DQN against SB3-DQN on identical environments.
2. Quickly run SAC or TD3 on the continuous BuddyJrReach-v0 task — algorithms
   whose critic and actor updates are significantly more complex to implement
   from scratch than Q-learning or REINFORCE.
3. Enable Hindsight Experience Replay (HER) with a single flag to see how
   goal-conditioned RL dramatically accelerates sparse-reward tasks.

Algorithm summary
-----------------
PPO  (Proximal Policy Optimization, on-policy, continuous or discrete)
    Clips the policy-gradient update so the new policy cannot stray too far
    from the old one.  Update equation (clipped surrogate objective)::

        L_CLIP(θ) = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]

        where  r_t(θ) = π_θ(a|s) / π_θ_old(a|s)   (probability ratio)
               A_t    = GAE advantage estimate
               ε      = clip range (typically 0.2)

    Good default for *any* environment; the recommended first algorithm to try.

SAC  (Soft Actor-Critic, off-policy, continuous only)
    Maximises a trade-off between expected return *and* policy entropy:

        J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]

        where  α  = temperature (auto-tuned to a target entropy)
               H  = differential entropy of the Gaussian policy

    The temperature α is learnt by minimising::

        J(α) = E[-α log π(a|s) - α H̄]   with  H̄ = -dim(A)  (target entropy)

    SAC is extremely sample-efficient on continuous control tasks — the
    recommended choice for BuddyJrReach-v0 if fast wall-clock learning matters.

TD3  (Twin Delayed DDPG, off-policy, continuous only)
    Extends DDPG with three tricks to address overestimation bias:
      1. *Twin critics*: take the min of two Q-networks.
         Q_target(s,a) = r + γ min(Q1(s',a'), Q2(s',a'))
      2. *Delayed policy updates*: update actor every ``policy_delay`` critic steps.
      3. *Target policy smoothing*: add clipped noise to the target action.
         a'(s') = clip(π(s') + clip(N(0,σ), -c, c), a_lo, a_hi)

DDPG (Deep Deterministic Policy Gradient, off-policy, continuous only)
    The ancestor of TD3; actor/critic trained with the deterministic PG theorem::

        ∇_θ J(θ) = E[∇_a Q^π(s,a)|_{a=π(s)} · ∇_θ π_θ(s)]

    Has known instability issues (overestimation, sensitivity to hyper-
    parameters) — included here so learners can see why TD3 was invented.

HER  (Hindsight Experience Replay, off-policy only, requires goal env)
    Re-labels past transitions with a substitute goal that *was* achieved,
    turning every failed episode into useful experience::

        For a transition (s, a, r, s', g) from a failed episode,
        replace g with g' = achieved_goal(s')  → r' = compute_reward(s', g')
        and add (s, a, r', s', g') to the replay buffer.

    Strategy "future": substitute goals are sampled from future timesteps of
    the same episode — the most effective strategy in practice.
    Requires the env to have a Dict observation space with keys
    {observation, achieved_goal, desired_goal} and to implement
    ``compute_reward(achieved, desired, info)``.
    Enable via ``gym.make("BuddyJrReach-v0", goal_env=True)``.

Env compatibility
-----------------
* PPO  : continuous (BuddyJrReach-v0) or discrete (BuddyJrReachDiscrete-v0).
* SAC  : continuous only (BuddyJrReach-v0, or goal_env=True for HER).
* TD3  : continuous only (same as SAC).
* DDPG : continuous only (same as SAC).
* HER  : requires SAC/TD3/DDPG + goal_env=True → Dict obs space.

Usage examples
--------------
::

    import gymnasium as gym
    import rl_lab  # registers the envs
    from rl_lab.algos.sb3_integration import SB3Algorithm

    # --- PPO on continuous reach ---
    env = gym.make("BuddyJrReach-v0")
    algo = SB3Algorithm(env, algo="ppo", seed=42)
    algo.train(total_steps=50_000)

    # --- SAC with HER ---
    goal_env = gym.make("BuddyJrReach-v0", goal_env=True)
    algo = SB3Algorithm(goal_env, algo="sac", seed=0, use_her=True)
    algo.train(total_steps=100_000)

    # --- Save / reload ---
    algo.save("runs/sac_her/model")
    algo.load("runs/sac_her/model")
    action, _ = algo.predict(obs)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

# ---------------------------------------------------------------------------
# Public helpers / type aliases
# ---------------------------------------------------------------------------

#: The four SB3 algorithm names the integration understands (lower-case).
SUPPORTED_ALGOS = frozenset({"ppo", "sac", "td3", "ddpg"})

#: Off-policy SB3 algos that use a replay buffer (and thus can use HER).
_OFF_POLICY = frozenset({"sac", "td3", "ddpg"})

# ---------------------------------------------------------------------------
# CPU-friendly hyperparameter defaults
# ---------------------------------------------------------------------------
# The BuddyJr task has a small observation space (17 dims) and low-dimensional
# action space (Box(4,)).  We therefore use tiny networks and modest buffer /
# rollout sizes so the lab runs fast even without a GPU.

_PPO_DEFAULTS: dict[str, Any] = {
    # --n_steps: steps collected per rollout before an SGD update.
    # 512 is small enough for fast iteration on a CPU.
    "n_steps": 512,
    # --batch_size: mini-batch size for the PPO SGD updates.
    "batch_size": 64,
    # --n_epochs: how many passes over each rollout batch.
    "n_epochs": 10,
    # --gamma: discount factor.
    "gamma": 0.99,
    # --gae_lambda: GAE λ for advantage estimation.
    "gae_lambda": 0.95,
    # --clip_range: ε in the clipped surrogate objective.
    "clip_range": 0.2,
    # --ent_coef: entropy regularisation weight (encourages exploration).
    "ent_coef": 0.01,
    # --learning_rate: Adam step size.
    "learning_rate": 3e-4,
    # --policy_kwargs: two hidden layers of 64 units — tiny but sufficient.
    "policy_kwargs": {"net_arch": [64, 64]},
}

_SAC_DEFAULTS: dict[str, Any] = {
    # --buffer_size: replay buffer capacity.
    "buffer_size": 50_000,
    # --learning_starts: steps before the first gradient update.
    "learning_starts": 1_000,
    # --batch_size: mini-batch drawn from the replay buffer each update.
    "batch_size": 256,
    # --tau: Polyak averaging coefficient for the target networks.
    # τ=0.005 → slow, stable target network updates.
    "tau": 0.005,
    # --gamma: discount factor.
    "gamma": 0.99,
    # --train_freq: update the networks every this many env steps.
    "train_freq": 1,
    # --gradient_steps: gradient steps per env step.
    "gradient_steps": 1,
    # --learning_rate: shared for actor and critic.
    "learning_rate": 3e-4,
    # --policy_kwargs: tiny networks keep memory and compute low on CPU.
    "policy_kwargs": {"net_arch": [64, 64]},
}

_TD3_DEFAULTS: dict[str, Any] = {
    # TD3 is more sensitive to buffer size than SAC; 50k is fine for this task.
    "buffer_size": 50_000,
    "learning_starts": 1_000,
    "batch_size": 256,
    # --tau: TD3 uses the same Polyak averaging as DDPG/SAC.
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    # --policy_delay: number of critic updates per actor update (TD3 trick #2).
    "policy_delay": 2,
    # --target_policy_noise: σ of the smoothing noise added to target actions
    # (TD3 trick #3 — prevents the critic from over-exploiting narrow peaks).
    "target_policy_noise": 0.2,
    # --target_noise_clip: clipping bound c for the smoothing noise.
    "target_noise_clip": 0.5,
    "learning_rate": 3e-4,
    "policy_kwargs": {"net_arch": [64, 64]},
}

_DDPG_DEFAULTS: dict[str, Any] = {
    # DDPG has the same interface as TD3 (it is TD3 without the three tricks).
    "buffer_size": 50_000,
    "learning_starts": 1_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "learning_rate": 3e-4,
    "policy_kwargs": {"net_arch": [64, 64]},
}

_ALGO_DEFAULTS: dict[str, dict[str, Any]] = {
    "ppo": _PPO_DEFAULTS,
    "sac": _SAC_DEFAULTS,
    "td3": _TD3_DEFAULTS,
    "ddpg": _DDPG_DEFAULTS,
}

# ---------------------------------------------------------------------------
# SB3 class map  (imported lazily inside _build_model to avoid hard-importing
# torch at module load time — the tabular algos don't need it)
# ---------------------------------------------------------------------------


def _sb3_cls(algo: str) -> Any:
    """Return the SB3 class for the given algo name string.

    Raises ImportError with a helpful message if stable-baselines3 is not
    installed, or ValueError for an unsupported name.
    """
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(
            f"Unknown algo {algo!r}. Supported: {sorted(SUPPORTED_ALGOS)}. "
            "SB3 integration only covers PPO, SAC, TD3, DDPG."
        )
    try:
        if algo == "ppo":
            from stable_baselines3 import PPO

            return PPO
        if algo == "sac":
            from stable_baselines3 import SAC

            return SAC
        if algo == "td3":
            from stable_baselines3 import TD3

            return TD3
        # algo == "ddpg"
        from stable_baselines3 import DDPG

        return DDPG
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required. Install it with:\n" "    pip install stable-baselines3"
        ) from exc


# ---------------------------------------------------------------------------
# The Algorithm-protocol implementation
# ---------------------------------------------------------------------------


class SB3Algorithm:
    """Adapter that exposes Stable-Baselines3 algorithms via the lab's protocol.

    Parameters
    ----------
    env:
        A Gymnasium environment.  For SAC/TD3/DDPG the action space must be
        ``Box`` (continuous).  For HER it must also have a ``Dict`` observation
        space with keys ``{observation, achieved_goal, desired_goal}``
        (use ``gym.make("BuddyJrReach-v0", goal_env=True)``).
    algo:
        One of ``"ppo"``, ``"sac"``, ``"td3"``, ``"ddpg"`` (lower-case).
    seed:
        Random seed passed both to the SB3 model constructor and to SB3's
        internal environment seeding.  Reproducible runs need the same seed.
    use_her:
        When ``True`` enable Hindsight Experience Replay.  Requires an
        off-policy algorithm (sac / td3 / ddpg) and a goal-conditioned env
        (Dict obs with achieved/desired_goal keys).
    policy:
        SB3 policy string.  Defaults to ``"MultiInputPolicy"`` when ``use_her``
        is True or the observation space is a ``Dict``; ``"MlpPolicy"``
        otherwise.  You can override this (e.g. ``"CnnPolicy"``) if needed.
    **hparams:
        Any additional keyword arguments are forwarded to the SB3 model
        constructor, overriding the cpu-friendly defaults.  For example::

            SB3Algorithm(env, algo="ppo", learning_rate=1e-3, n_steps=1024)
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        algo: str = "ppo",
        seed: int = 0,
        use_her: bool = False,
        policy: str | None = None,
        **hparams: Any,
    ) -> None:
        algo = algo.lower()
        if algo not in SUPPORTED_ALGOS:
            raise ValueError(
                f"SB3Algorithm does not support algo={algo!r}. "
                f"Choose from {sorted(SUPPORTED_ALGOS)}."
            )
        if use_her and algo not in _OFF_POLICY:
            raise ValueError(
                f"HER requires an off-policy algo (sac, td3, ddpg), got {algo!r}. "
                "PPO is on-policy and cannot use a replay buffer."
            )

        self.env: gym.Env = env
        self.algo: str = algo
        self.seed: int = int(seed)
        self.use_her: bool = use_her
        self._hparams: dict[str, Any] = hparams

        # Determine the policy class string.
        # SB3 requires "MultiInputPolicy" for Dict observation spaces
        # (which is what the goal-conditioned env produces).
        obs_space = env.observation_space
        _is_dict_obs = isinstance(obs_space, gym.spaces.Dict)
        if policy is not None:
            self._policy: str = policy
        elif use_her or _is_dict_obs:
            self._policy = "MultiInputPolicy"
        else:
            self._policy = "MlpPolicy"

        # Build the SB3 model immediately so predict/save work before train().
        self.model: Any = self._build_model()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_model(self) -> Any:
        """Construct and return the SB3 model with sensible CPU defaults.

        The keyword arguments are layered as follows (later wins)::

            _ALGO_DEFAULTS[algo]  <--  self._hparams  (user overrides)

        HER is injected via ``replay_buffer_class`` and
        ``replay_buffer_kwargs`` — SB3's standard mechanism.
        """
        SB3Cls: Any = _sb3_cls(self.algo)

        # Start from the cpu-friendly defaults for this algo.
        kwargs: dict[str, Any] = dict(_ALGO_DEFAULTS[self.algo])

        # Apply user overrides (from **hparams passed to __init__).
        kwargs.update(self._hparams)

        # Always run on CPU — the BuddyJr task is tiny and
        # Apple Silicon MPS support in older torch versions is inconsistent.
        kwargs["device"] = "cpu"
        kwargs["seed"] = self.seed
        kwargs["verbose"] = 0  # silence SB3's print statements in the lab

        if self.use_her:
            # ------------------------------------------------------------------
            # Hindsight Experience Replay configuration.
            # ------------------------------------------------------------------
            # SB3's HER is implemented as a special ReplayBuffer subclass that
            # intercepts stored transitions and re-labels some of them with
            # substitute goals sampled from later in the same episode.
            #
            # goal_selection_strategy="future":
            #   The substitute goal g' is sampled uniformly from the achieved
            #   goals at timesteps AFTER the current transition, within the
            #   same episode.  "future" is empirically the strongest strategy
            #   (Andrychowicz et al., 2017).
            #
            # n_sampled_goal=4:
            #   For every real transition stored, add 4 HER transitions with
            #   relabelled goals.  Ratio of 4:1 (HER:real) is the paper default.
            try:
                from stable_baselines3.her import HerReplayBuffer
            except ImportError as exc:
                raise ImportError("stable-baselines3 >= 1.0 with HER support is required.") from exc

            kwargs["replay_buffer_class"] = HerReplayBuffer
            kwargs["replay_buffer_kwargs"] = {
                "goal_selection_strategy": "future",
                # n_sampled_goal: how many HER transitions per real transition.
                # 4 is the default from the original HER paper.
                "n_sampled_goal": 4,
            }

        return SB3Cls(self._policy, self.env, **kwargs)

    # -----------------------------------------------------------------------
    # Algorithm protocol implementation
    # -----------------------------------------------------------------------

    def train(
        self,
        total_steps: int,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Train for ``total_steps`` env steps using ``model.learn()``.

        Parameters
        ----------
        total_steps:
            Total number of environment steps to train for.  Passed directly
            to SB3's ``model.learn(total_timesteps=total_steps)``.
        callback:
            Optional lab-style callback ``(metrics: dict) -> None``.  If
            provided it is wrapped in a :class:`~rl_lab.train.callbacks.
            SimpleCallbackBridge` and passed to SB3 so it receives periodic
            metric updates during training.

            Metrics dict keys you can expect:
            - ``"step"``           : current env step count.
            - ``"episode_return"`` : mean return over recent episodes.
            - ``"success_rate"``   : mean success rate (if logged).
            - ``"loss"``           : most recent training loss.

        Returns
        -------
        dict
            A history dict with at minimum ``{"algo": self.algo}``.  The
            SB3 logger accumulates richer data in ``runs/`` if
            ``tensorboard_log`` was passed to the constructor.
        """
        sb3_callback: Any = None
        if callback is not None:
            # Build the SB3-compatible wrapper around the user's callback.
            from rl_lab.train.callbacks import SimpleCallbackBridge

            sb3_callback = SimpleCallbackBridge(callback)

        # model.learn is the main SB3 training loop.  It handles rollout
        # collection, replay buffer management, gradient updates, logging, and
        # early stopping (via callbacks that return False from on_step).
        self.model.learn(
            total_timesteps=total_steps,
            # callback=None is fine for SB3 — it just trains with no hooks.
            callback=sb3_callback,
            # reset_num_timesteps=True means each train() call starts fresh
            # from step 0 in the logger.  Set to False to resume a run.
            reset_num_timesteps=True,
        )

        return {"algo": self.algo}

    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> tuple[Any, Any]:
        """Return ``(action, state)`` for the given observation.

        Delegates directly to ``model.predict``, which returns a
        ``(np.ndarray action, None)`` tuple for MLP policies.

        Parameters
        ----------
        observation:
            A single observation (or batched observations) matching the env's
            observation space.  Dict observations are supported when the model
            was built with ``MultiInputPolicy``.
        deterministic:
            When ``True`` (default) the policy outputs the mode of the action
            distribution (greedy for evaluation).  ``False`` samples from the
            distribution (useful for exploration or stochastic evaluation).
        """
        return self.model.predict(observation, deterministic=deterministic)  # type: ignore[no-any-return]

    def save(self, path: str) -> None:
        """Save the SB3 model to ``<path>.zip``.

        SB3 appends ``.zip`` automatically when saving, so ``path`` should
        be given *without* the extension (e.g. ``"runs/ppo/model"``).

        The checkpoint includes the full model: policy weights, optimiser
        state, hyperparameters and the observation/action space definition.
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load SB3 model weights from ``<path>.zip`` into ``self.model``.

        After loading the model is ready for ``predict()`` or for resuming
        training via another ``train()`` call.

        Parameters
        ----------
        path:
            Path to the ``.zip`` checkpoint (with or without the extension —
            SB3 handles both).
        """
        SB3Cls: Any = _sb3_cls(self.algo)
        # SB3's .load() is a classmethod that returns a new model instance.
        # We pass env=self.env so the model stays connected to the environment
        # (required for train() to work after load()).
        self.model = SB3Cls.load(path, env=self.env)
