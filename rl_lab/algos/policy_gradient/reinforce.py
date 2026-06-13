"""REINFORCE — the Monte-Carlo policy-gradient algorithm, from scratch.

This is the simplest policy-gradient method and a good first stop after the
value-based algorithms (Q-learning / DQN). Instead of learning *values* and
acting greedily, we learn the *policy* directly: a neural network maps a state
``s`` to a probability distribution over actions ``pi(a | s; theta)``, and we
nudge ``theta`` to make the actions that led to high return more likely.

The maths (the "score-function" / "log-derivative" trick)
---------------------------------------------------------
We want to maximise the expected return ``J(theta) = E[ sum_t r_t ]``. The
policy-gradient theorem says::

    grad_theta J(theta) = E[ sum_t  grad_theta log pi(a_t | s_t)  *  G_t ]

where ``G_t = sum_{k>=t} gamma^(k-t) r_k`` is the discounted return *from step
t onwards*. The key identity that gets us there is the score function (a.k.a.
log-derivative trick)::

    grad_theta pi = pi * grad_theta log pi

i.e. we can push the gradient inside an expectation by weighting
``grad log pi`` by the sampled return. Intuitively: increase the log-probability
of actions that were followed by a large return, decrease it for actions
followed by a small (or negative) one.

Because we *maximise* ``J`` but optimisers *minimise*, our loss is the negative::

    loss = - (1/N) * sum_t  log pi(a_t | s_t)  *  A_t

Baselines and variance reduction (the teaching point of this file)
------------------------------------------------------------------
Multiplying ``log pi`` by the raw return ``G_t`` is an *unbiased* but very
*noisy* gradient estimate. We can subtract any quantity that does not depend on
the action — a "baseline" ``b(s_t)`` — without introducing bias::

    A_t = G_t - b(s_t)

A good baseline is the state-value ``V(s_t)``, learned by a second network
("the critic") via regression onto the observed returns. Subtracting it leaves
the *advantage* "was this action better or worse than average from here?",
which has much smaller variance. This file logs ``advantage_var`` every update
so you can watch the variance drop when ``baseline=True`` versus ``False`` —
that drop is the whole point of a baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rl_lab.utils.seeding import set_global_seed


def _mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    """A small fully-connected ``tanh`` MLP: ``in_dim -> hidden... -> out_dim``."""
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.Tanh())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class REINFORCE:
    """Monte-Carlo policy gradient with an optional learned value baseline.

    Parameters
    ----------
    env:
        A discrete-action Gymnasium env (designed for ``BuddyJrReachDiscrete-v0``:
        obs ``Box(17,)``, action ``Discrete(9)``).
    seed:
        Seeds Python/NumPy/torch and the env for reproducible curves.
    lr:
        Learning rate shared by the policy (and value) optimiser(s).
    gamma:
        Discount factor used to compute the returns ``G_t``.
    baseline:
        If ``True`` learn a value-network baseline ``V(s)`` and use it as the
        advantage baseline; if ``False`` use the *batch mean* of returns as a
        constant baseline (still unbiased, but a much weaker variance reducer).
    hidden:
        Hidden-layer sizes for the policy (and value) MLPs.
    """

    def __init__(
        self,
        env: Any,
        *,
        seed: int = 0,
        lr: float = 1e-3,
        gamma: float = 0.99,
        baseline: bool = True,
        hidden: tuple[int, ...] = (64, 64),
    ) -> None:
        self.env = env
        self.seed = set_global_seed(seed, env)
        self.lr = lr
        self.gamma = gamma
        self.use_baseline = baseline
        self.hidden = tuple(hidden)

        # CPU-only lab: keep everything on the CPU device explicitly.
        self.device = torch.device("cpu")

        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Policy network: state -> logits over the discrete actions.
        # softmax(logits) gives pi(a | s; theta).
        self.policy: nn.Sequential = _mlp(obs_dim, n_actions, self.hidden).to(self.device)

        # Optional value baseline V(s) (the "critic"): state -> scalar value.
        self.value: nn.Sequential | None = None
        params = list(self.policy.parameters())
        if self.use_baseline:
            self.value = _mlp(obs_dim, 1, self.hidden).to(self.device)
            params = params + list(self.value.parameters())

        # A single optimiser updates the policy (and value head if present).
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Rolling training history (also forwarded to the callback per episode).
        self.history: dict[str, list[float]] = {
            "step": [],
            "episode_return": [],
            "success_rate": [],
            "loss": [],
            "advantage_var": [],
        }
        self._total_steps_done = 0

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def _policy_dist(self, obs_t: torch.Tensor) -> torch.distributions.Categorical:
        """Build the categorical action distribution ``pi(. | s)`` from logits."""
        logits = self.policy(obs_t)
        return torch.distributions.Categorical(logits=logits)

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, Any]:
        """Return ``(action, None)``.

        ``deterministic=True`` takes the arg-max action (greedy w.r.t. the
        policy); ``False`` samples from ``pi(. | s)`` (useful for exploration /
        rollouts).
        """
        obs_t = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.policy(obs_t)
            if deterministic:
                action = int(torch.argmax(logits).item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
        return action, None

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def _collect_episode(self) -> tuple[np.ndarray, list[int], list[float], bool]:
        """Play one full episode with the *stochastic* policy.

        Returns ``(observations, actions, rewards, success)``. We need whole
        episodes because REINFORCE's return ``G_t`` is a Monte-Carlo estimate:
        it sums the *actual* rewards collected until the episode ends.
        """
        obs, _info = self.env.reset()
        obs_list: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        success = False

        terminated = truncated = False
        while not (terminated or truncated):
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action = int(self._policy_dist(obs_t).sample().item())

            obs_list.append(np.asarray(obs, dtype=np.float32))
            actions.append(action)

            obs, reward, terminated, truncated, info = self.env.step(action)
            # gymnasium types reward as SupportsFloat — cast to a plain float.
            rewards.append(float(reward))
            success = bool(info.get("is_success", False))

        observations: np.ndarray = np.asarray(obs_list, dtype=np.float32)
        return observations, actions, rewards, success

    def _discounted_returns(self, rewards: list[float]) -> np.ndarray:
        """Compute ``G_t = sum_{k>=t} gamma^(k-t) r_k`` by a backward scan."""
        returns: np.ndarray = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        # Walk backwards so each G_t reuses G_{t+1}: G_t = r_t + gamma * G_{t+1}.
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return returns

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self, total_steps: int, callback: Callable[[dict], None] | None = None
    ) -> dict[str, Any]:
        """Run REINFORCE for roughly ``total_steps`` environment steps.

        We collect one full episode, form the discounted returns, compute the
        advantages (return minus baseline), and take a single gradient step on
        the (negative) log-prob * advantage loss. Repeat until the step budget
        is exhausted.
        """
        steps_done = 0
        # A small running estimate of the success rate over recent episodes.
        recent_success: list[float] = []

        while steps_done < total_steps:
            observations, actions, rewards, success = self._collect_episode()
            ep_len = len(rewards)
            steps_done += ep_len
            self._total_steps_done += ep_len

            ep_return = float(np.sum(rewards))
            recent_success.append(1.0 if success else 0.0)
            if len(recent_success) > 100:
                recent_success.pop(0)
            success_rate = float(np.mean(recent_success))

            # --- Tensors for this episode -------------------------------------
            obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            returns_np = self._discounted_returns(rewards)
            returns_t = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)

            # --- Baseline & advantage ----------------------------------------
            # The baseline b(s_t) must NOT depend on the action a_t, so it does
            # not bias the gradient; it only reduces its variance.
            if self.use_baseline and self.value is not None:
                # Critic prediction V(s_t). detach() when forming the advantage:
                # the baseline is a *constant* w.r.t. the policy gradient.
                values_t = self.value(obs_t).squeeze(-1)
                advantages_t = returns_t - values_t.detach()
            else:
                # No critic: subtract the batch mean (a constant baseline). This
                # is still unbiased but a much weaker variance reducer than V(s).
                advantages_t = returns_t - returns_t.mean()
                values_t = None

            # The variance of the per-step advantages is the diagnostic that
            # makes the baseline's effect visible — log it every update.
            advantage_var = float(advantages_t.detach().var(unbiased=False).item())

            # --- Policy (score-function) loss --------------------------------
            # log pi(a_t | s_t): the log-probability the policy assigned to the
            # action it actually took. grad of this is the "score function".
            dist = self._policy_dist(obs_t)
            log_probs = dist.log_prob(act_t)

            # REINFORCE loss = -sum_t log pi(a_t | s_t) * A_t. Maximising the
            # return == minimising the negative weighted log-prob.
            policy_loss = -(log_probs * advantages_t).sum()

            # --- Value (baseline) regression loss ----------------------------
            # Fit V(s_t) towards the observed returns G_t with an MSE loss.
            value_loss = torch.zeros((), device=self.device)
            if self.use_baseline and values_t is not None:
                value_loss = nn.functional.mse_loss(values_t, returns_t)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # --- Logging ------------------------------------------------------
            loss_val = float(loss.detach().item())
            self.history["step"].append(float(self._total_steps_done))
            self.history["episode_return"].append(ep_return)
            self.history["success_rate"].append(success_rate)
            self.history["loss"].append(loss_val)
            self.history["advantage_var"].append(advantage_var)

            if callback is not None:
                callback(
                    {
                        "step": self._total_steps_done,
                        "episode_return": ep_return,
                        "success_rate": success_rate,
                        "loss": loss_val,
                        "advantage_var": advantage_var,
                    }
                )

        return self.history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the policy (and value) ``state_dict`` plus hyperparameters."""
        payload: dict[str, Any] = {
            "policy": self.policy.state_dict(),
            "hparams": {
                "lr": self.lr,
                "gamma": self.gamma,
                "baseline": self.use_baseline,
                "hidden": self.hidden,
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
            },
        }
        if self.value is not None:
            payload["value"] = self.value.state_dict()
        torch.save(payload, path)

    def load(self, path: str) -> None:
        """Load parameters saved by :meth:`save` into this constructed algo."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(payload["policy"])
        if self.value is not None and "value" in payload:
            self.value.load_state_dict(payload["value"])
