"""PPOMin — a minimal, readable Proximal Policy Optimization, from scratch.

PPO is the workhorse of modern policy-gradient RL. It builds on REINFORCE (see
``reinforce.py``) but fixes its two biggest weaknesses:

1. **High-variance, on-episode returns.** REINFORCE waits for whole episodes
   and uses raw Monte-Carlo returns. PPO instead collects a fixed *rollout* of
   ``n_steps`` transitions (which may span several episodes) and estimates the
   advantage with **GAE(lambda)** — Generalised Advantage Estimation — which
   trades a little bias for a large reduction in variance.

2. **Destructively large updates.** A single big policy-gradient step can move
   the policy so far that the data it was estimated from is no longer
   representative, and learning collapses. PPO performs several epochs of
   minibatch updates on the *same* rollout, but keeps each update "proximal"
   with a **clipped surrogate objective** that removes the incentive to move
   the action probabilities too far from the policy that collected the data.

The two equations to read here
------------------------------
GAE(lambda) — the advantage of taking the rollout's action at step ``t``::

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)   # TD residual
    A_t     = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}  # backward scan

``lambda=0`` gives the low-variance/high-bias one-step TD advantage; ``lambda=1``
recovers the high-variance Monte-Carlo advantage. The returns target for the
critic is then ``R_t = A_t + V(s_t)``.

The clipped surrogate objective (per sample)::

    ratio_t = exp( log pi_new(a_t|s_t) - log pi_old(a_t|s_t) )
    L_clip  = - min( ratio_t * A_t,  clip(ratio_t, 1-eps, 1+eps) * A_t )

``ratio_t`` is how much *more or less* likely the new policy makes the action
than the policy that collected it (``ratio=1`` at the start of the epochs). The
``min`` + ``clip`` means: if pushing the probability further would only help
because the advantage is positive, we stop crediting it once ``ratio`` leaves
``[1-eps, 1+eps]`` — so the update stays close ("proximal") to ``pi_old``.
``eps`` is ``clip_range`` and is a constructor knob so you can run the
clip-sweep lesson and watch how clipping tames the updates.
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


class PPOMin:
    """Minimal clipped-objective PPO with GAE(lambda), separate actor & critic.

    Parameters
    ----------
    env:
        A discrete-action Gymnasium env (designed for ``BuddyJrReachDiscrete-v0``:
        obs ``Box(17,)``, action ``Discrete(9)``).
    seed:
        Seeds Python/NumPy/torch and the env for reproducible curves.
    lr:
        Adam learning rate for the combined actor+critic parameters.
    gamma:
        Discount factor used by GAE and the returns target.
    gae_lambda:
        The ``lambda`` in GAE; the bias/variance knob (0 = TD, 1 = Monte-Carlo).
    clip_range:
        The PPO clipping epsilon ``eps`` (the clip-sweep lesson's knob).
    n_steps:
        Transitions collected per rollout before each optimisation phase.
    n_epochs:
        Passes over each rollout during optimisation.
    batch_size:
        Minibatch size for the SGD passes.
    hidden:
        Hidden-layer sizes for the actor and critic MLPs.
    """

    def __init__(
        self,
        env: Any,
        *,
        seed: int = 0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        hidden: tuple[int, ...] = (64, 64),
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.env = env
        self.seed = set_global_seed(seed, env)
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_steps = int(n_steps)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.hidden = tuple(hidden)
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # CPU-only lab.
        self.device = torch.device("cpu")

        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Separate actor (logits over actions) and critic (state value) MLPs.
        self.actor: nn.Sequential = _mlp(obs_dim, n_actions, self.hidden).to(self.device)
        self.critic: nn.Sequential = _mlp(obs_dim, 1, self.hidden).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.history: dict[str, list[float]] = {
            "step": [],
            "episode_return": [],
            "success_rate": [],
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "clip_fraction": [],
        }
        self._total_steps_done = 0

        # Rollout cursor: we carry the current obs across rollouts so a rollout
        # can start mid-episode (PPO collects a fixed number of *steps*, not
        # whole episodes).
        self._obs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def _dist(self, obs_t: torch.Tensor) -> torch.distributions.Categorical:
        """Build the categorical action distribution ``pi(. | s)`` from logits."""
        return torch.distributions.Categorical(logits=self.actor(obs_t))

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, Any]:
        """Return ``(action, None)``; greedy if ``deterministic`` else sampled."""
        obs_t = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.actor(obs_t)
            if deterministic:
                action = int(torch.argmax(logits).item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
        return action, None

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def _collect_rollout(
        self,
    ) -> tuple[
        np.ndarray,  # observations  (n_steps, obs_dim)
        np.ndarray,  # actions       (n_steps,)
        np.ndarray,  # log_probs_old (n_steps,)
        np.ndarray,  # rewards       (n_steps,)
        np.ndarray,  # values        (n_steps,)
        np.ndarray,  # dones         (n_steps,)  done flag AFTER each step
        float,  # bootstrap value V(s_{n_steps}) for the final transition
        list[float],  # completed-episode returns within this rollout
        list[float],  # completed-episode successes within this rollout
    ]:
        """Roll the *current* policy forward for ``n_steps`` transitions.

        We record everything the PPO update needs: the observations, the actions
        taken, the log-probability the *behaviour* policy assigned to them
        (``log pi_old`` — frozen for the whole optimisation phase), the rewards,
        the critic's value estimates, and the done flags. We also bootstrap the
        value of the state after the last step so GAE can be computed even when
        the rollout cuts an episode in half.
        """
        obs_buf: np.ndarray = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        act_buf: np.ndarray = np.zeros(self.n_steps, dtype=np.int64)
        logp_buf: np.ndarray = np.zeros(self.n_steps, dtype=np.float32)
        rew_buf: np.ndarray = np.zeros(self.n_steps, dtype=np.float32)
        val_buf: np.ndarray = np.zeros(self.n_steps, dtype=np.float32)
        done_buf: np.ndarray = np.zeros(self.n_steps, dtype=np.float32)

        ep_returns: list[float] = []
        ep_successes: list[float] = []
        ep_return = 0.0

        if self._obs is None:
            obs, _info = self.env.reset()
            self._obs = np.asarray(obs, dtype=np.float32)
        obs = self._obs

        for t in range(self.n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                dist = self._dist(obs_t)
                action_t = dist.sample()
                logp = dist.log_prob(action_t)
                value = self.critic(obs_t).squeeze(-1)
            action = int(action_t.item())

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            reward_f = float(reward)  # gymnasium types reward as SupportsFloat
            ep_return += reward_f
            done = bool(terminated or truncated)

            obs_buf[t] = obs
            act_buf[t] = action
            logp_buf[t] = float(logp.item())
            rew_buf[t] = reward_f
            val_buf[t] = float(value.item())
            done_buf[t] = 1.0 if done else 0.0

            obs = np.asarray(next_obs, dtype=np.float32)
            if done:
                ep_returns.append(ep_return)
                ep_successes.append(1.0 if info.get("is_success", False) else 0.0)
                ep_return = 0.0
                reset_obs, _info = self.env.reset()
                obs = np.asarray(reset_obs, dtype=np.float32)

        # Persist the cursor for the next rollout and bootstrap the tail value.
        self._obs = obs
        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            last_value = float(self.critic(last_obs_t).squeeze(-1).item())

        return (
            obs_buf,
            act_buf,
            logp_buf,
            rew_buf,
            val_buf,
            done_buf,
            last_value,
            ep_returns,
            ep_successes,
        )

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation, computed by a single backward scan.

        Returns ``(advantages, returns)`` where ``returns = advantages + values``
        is the regression target for the critic.
        """
        n = len(rewards)
        advantages: np.ndarray = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        next_value = last_value
        # Walk backwards so each A_t reuses A_{t+1}:
        #   delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        #   A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        # done_t masks the bootstrap so we never leak value across an episode
        # boundary that fell inside the rollout.
        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns: np.ndarray = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self, total_steps: int, callback: Callable[[dict], None] | None = None
    ) -> dict[str, Any]:
        """Run PPO for roughly ``total_steps`` environment steps.

        Each iteration: collect a rollout, compute GAE advantages + returns,
        then do ``n_epochs`` of minibatch SGD on the clipped surrogate.
        """
        # Recent-episode success window for a smoothed success-rate signal.
        recent_success: list[float] = []

        while self._total_steps_done < total_steps:
            (
                obs_buf,
                act_buf,
                logp_buf,
                rew_buf,
                val_buf,
                done_buf,
                last_value,
                ep_returns,
                ep_successes,
            ) = self._collect_rollout()
            self._total_steps_done += self.n_steps

            advantages_np, returns_np = self._compute_gae(rew_buf, val_buf, done_buf, last_value)

            # Move the rollout to tensors once.
            obs_t = torch.as_tensor(obs_buf, dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(act_buf, dtype=torch.long, device=self.device)
            logp_old_t = torch.as_tensor(logp_buf, dtype=torch.float32, device=self.device)
            adv_t = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)

            # Advantage normalisation — a standard PPO trick that stabilises the
            # gradient scale across rollouts. Normalise over the whole batch.
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            # --- Optimisation phase ------------------------------------------
            last_policy_loss = 0.0
            last_value_loss = 0.0
            last_entropy = 0.0
            last_clip_frac = 0.0
            last_total_loss = 0.0

            n = self.n_steps
            for _epoch in range(self.n_epochs):
                # Shuffle indices each epoch and iterate over minibatches.
                perm: np.ndarray = np.random.permutation(n)
                for start in range(0, n, self.batch_size):
                    idx = perm[start : start + self.batch_size]
                    mb_idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)

                    mb_obs = obs_t[mb_idx]
                    mb_act = act_t[mb_idx]
                    mb_logp_old = logp_old_t[mb_idx]
                    mb_adv = adv_t[mb_idx]
                    mb_ret = ret_t[mb_idx]

                    # Re-evaluate the CURRENT policy on the minibatch.
                    dist = self._dist(mb_obs)
                    logp_new = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()
                    value_pred = self.critic(mb_obs).squeeze(-1)

                    # ratio = pi_new / pi_old = exp(logp_new - logp_old).
                    # Equals 1.0 on the first epoch (new == old) and drifts as
                    # the policy moves away from the behaviour policy.
                    ratio = torch.exp(logp_new - mb_logp_old)

                    # Clipped surrogate: take the *pessimistic* (min) of the
                    # unclipped and clipped objectives. We MINIMISE the negative.
                    surr1 = ratio * mb_adv
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Critic regression onto the GAE returns.
                    value_loss = nn.functional.mse_loss(value_pred, mb_ret)

                    # Entropy bonus encourages exploration (subtracted from loss).
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    # Clip the global grad-norm — keeps a single batch from
                    # blowing the update up (another "stay proximal" guard).
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

                    # Diagnostics: fraction of samples whose ratio was clipped —
                    # a direct read-out of how hard the clip is biting (great for
                    # the clip-sweep lesson).
                    with torch.no_grad():
                        clipped = (torch.abs(ratio - 1.0) > self.clip_range).float().mean()
                    last_policy_loss = float(policy_loss.detach().item())
                    last_value_loss = float(value_loss.detach().item())
                    last_entropy = float(entropy.detach().item())
                    last_clip_frac = float(clipped.item())
                    last_total_loss = float(loss.detach().item())

            # --- Logging ------------------------------------------------------
            for s in ep_successes:
                recent_success.append(s)
            while len(recent_success) > 100:
                recent_success.pop(0)
            success_rate = float(np.mean(recent_success)) if recent_success else 0.0
            mean_return = float(np.mean(ep_returns)) if ep_returns else float("nan")

            self.history["step"].append(float(self._total_steps_done))
            self.history["episode_return"].append(mean_return)
            self.history["success_rate"].append(success_rate)
            self.history["loss"].append(last_total_loss)
            self.history["policy_loss"].append(last_policy_loss)
            self.history["value_loss"].append(last_value_loss)
            self.history["entropy"].append(last_entropy)
            self.history["clip_fraction"].append(last_clip_frac)

            if callback is not None:
                callback(
                    {
                        "step": self._total_steps_done,
                        "episode_return": mean_return,
                        "success_rate": success_rate,
                        "loss": last_total_loss,
                        "policy_loss": last_policy_loss,
                        "value_loss": last_value_loss,
                        "entropy": last_entropy,
                        "clip_fraction": last_clip_frac,
                    }
                )

        return self.history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the actor & critic ``state_dict`` plus hyperparameters."""
        payload: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "hparams": {
                "lr": self.lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_range": self.clip_range,
                "n_steps": self.n_steps,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "hidden": self.hidden,
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
            },
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        """Load parameters saved by :meth:`save` into this constructed algo."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
