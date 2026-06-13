"""Deep Q-Network (DQN) — a minimal, from-scratch, heavily-commented teaching version.

DQN is the algorithm that first showed a single neural-network value function could
learn control directly from raw observations (Mnih et al., 2015, *Human-level control
through deep reinforcement learning*). It is "Q-learning with a neural network", plus
two stabilising tricks that make that combination actually work:

  1. **Experience replay** — store transitions ``(s, a, r, s', done)`` in a buffer and
     train on *random minibatches* drawn from it. This breaks the strong temporal
     correlation between consecutive samples (gradient descent assumes i.i.d. data) and
     lets each transition be reused many times.

  2. **A target network** — a *frozen copy* of the Q-network used to compute the
     bootstrap target ``y = r + γ·max_a' Q_target(s', a')``. If we instead bootstrapped
     off the network we are currently updating, the target would shift every step and
     the regression would chase a moving goal, often diverging. We sync the target to
     the online network only every ``target_sync`` steps.

The core learning rule is exactly the Q-learning TD update, written as a regression:

    target   y = r + γ · max_a' Q_target(s', a') · (1 − done)
    loss     L = ( Q_online(s, a) − y )²            # MSE over a minibatch

For ``BuddyJrReachDiscrete-v0`` the observation is ``Box(17,)`` and the action is
``Discrete(9)``, so the Q-network is a tiny MLP mapping 17 inputs to 9 Q-values
(one per discrete action). Everything runs on CPU.

Two knobs are deliberately exposed so the lab can *demonstrate the instability* that the
tricks above fix:
  * shrink ``buffer_size`` (e.g. to a few hundred) to weaken experience replay, and/or
  * set ``target_sync=1`` to sync the target every step (i.e. effectively no target net).
With either ablation the learning curve becomes visibly noisier / can diverge.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from rl_lab.utils.seeding import set_global_seed


# --------------------------------------------------------------------------------------
# Experience replay buffer
# --------------------------------------------------------------------------------------
class ReplayBuffer:
    """A fixed-capacity ring buffer of transitions stored as flat NumPy arrays.

    Storing transitions and sampling *random* minibatches from them is the
    "experience replay" stabilising trick: it decorrelates the data the network
    trains on and lets each environment step be learned from more than once.

    When the buffer is full, ``_ptr`` wraps back to 0 and we overwrite the oldest
    transitions (hence "ring" buffer). Capacity is the ``buffer_size`` knob; shrinking
    it is the experience-replay ablation.
    """

    def __init__(self, capacity: int, obs_dim: int, *, rng: np.random.Generator) -> None:
        self.capacity = int(capacity)
        self._rng = rng

        # Pre-allocate contiguous arrays — far cheaper than a Python list of tuples.
        self.obs: np.ndarray = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros(self.capacity, dtype=np.int64)
        self.rewards: np.ndarray = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs: np.ndarray = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        # ``done`` here means "terminal" (the episode ended because the task ended),
        # NOT "truncated by the time limit". Only terminal transitions zero the
        # bootstrap term, so we are careful to store the right flag (see train()).
        self.dones: np.ndarray = np.zeros(self.capacity, dtype=np.float32)

        self._ptr = 0  # index of the next slot to write
        self._size = 0  # number of valid transitions currently stored

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Insert one transition, overwriting the oldest if the buffer is full."""
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity  # wrap around (ring buffer)
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a uniformly-random minibatch of ``batch_size`` transitions."""
        idx: np.ndarray = self._rng.integers(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


# --------------------------------------------------------------------------------------
# Q-network: a tiny MLP mapping an observation to one Q-value per discrete action
# --------------------------------------------------------------------------------------
class QNetwork(nn.Module):
    """MLP ``obs_dim -> hidden... -> n_actions``.

    The output layer has no activation: Q-values are unbounded real numbers, and the
    i-th output is the estimated return of taking discrete action ``i`` in that state.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, n_actions))  # linear head -> one Q-value per action
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------------------
# DQN agent
# --------------------------------------------------------------------------------------
class DQN:
    """From-scratch DQN for discrete-action Buddy Jr environments.

    Construction follows the lab's registry convention ``DQN(env, *, seed=0, **hparams)``.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        seed: int = 0,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_sync: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        learning_starts: int = 500,
        hidden: tuple[int, ...] = (64, 64),
    ) -> None:
        self.env = env
        self.seed = seed

        # Seed Python / NumPy / torch (and the env) for reproducible learning curves.
        set_global_seed(seed, env)
        # A dedicated Generator for replay sampling keeps that stream independent and
        # reproducible regardless of what else touches the global NumPy RNG.
        self._rng = np.random.default_rng(seed)

        # DQN only handles discrete actions and a flat (vector) observation space.
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "DQN requires a Discrete action space (e.g. BuddyJrReachDiscrete-v0)."
        assert isinstance(
            env.observation_space, gym.spaces.Box
        ), "DQN requires a Box (vector) observation space."
        self.n_actions = int(env.action_space.n)
        self.obs_dim = int(np.prod(env.observation_space.shape))

        # --- hyperparameters -----------------------------------------------------
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync  # sync target net every this many learning steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.learning_starts = learning_starts  # collect this many steps before training

        # --- networks ------------------------------------------------------------
        # CPU-only by design (the lab targets Apple Silicon without CUDA).
        self.device = torch.device("cpu")
        self.q_net = QNetwork(self.obs_dim, self.n_actions, tuple(hidden)).to(self.device)
        # The target network is a frozen copy used only to compute bootstrap targets.
        self.target_net = QNetwork(self.obs_dim, self.n_actions, tuple(hidden)).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # --- replay buffer -------------------------------------------------------
        self.buffer = ReplayBuffer(buffer_size, self.obs_dim, rng=self._rng)

        # Global environment-step counter; drives epsilon decay and target syncs.
        self._step_count = 0

    # ----------------------------------------------------------------------------------
    # Epsilon-greedy exploration
    # ----------------------------------------------------------------------------------
    def _epsilon(self) -> float:
        """Linearly anneal epsilon from ``epsilon_start`` to ``epsilon_end``.

        Early on we explore (high epsilon -> mostly random actions) so the replay buffer
        sees a wide variety of states; later we exploit the learned Q-values. The decay
        is linear over ``epsilon_decay_steps`` and then held flat at ``epsilon_end``.
        """
        frac = min(1.0, self._step_count / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def _greedy_action(self, obs: np.ndarray) -> int:
        """Return ``argmax_a Q_online(obs, a)`` — the current best-guess action."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q_net(obs_t)  # shape (1, n_actions)
        return int(torch.argmax(q_values, dim=1).item())

    def _select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy: random action with prob. epsilon, else the greedy action."""
        if self._rng.random() < epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return self._greedy_action(obs)

    # ----------------------------------------------------------------------------------
    # One gradient step on a replayed minibatch — the heart of DQN
    # ----------------------------------------------------------------------------------
    def _learn(self) -> float:
        """Sample a minibatch and apply one MSE TD update; return the scalar loss."""
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # --- prediction: Q_online(s, a) for the actions actually taken ---------------
        # q_net(obs) gives all 9 Q-values; gather() picks the one for the taken action.
        q_all = self.q_net(obs_t)  # (batch, n_actions)
        q_pred = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (batch,)

        # --- target: y = r + gamma * max_a' Q_target(s', a') * (1 - done) ------------
        # Computed under no_grad and with the FROZEN target network so the regression
        # target does not move while we fit it (the target-network trick).
        with torch.no_grad():
            q_next_max = self.target_net(next_obs_t).max(dim=1).values  # max over actions
            # (1 - done) zeroes the bootstrap term for terminal transitions: there is no
            # future return after a terminal state.
            target = rewards_t + self.gamma * q_next_max * (1.0 - dones_t)

        # --- regression: minimise ( Q_pred - y )^2 -----------------------------------
        loss = self.loss_fn(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ----------------------------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------------------------
    def train(
        self, total_steps: int, callback: Callable[[dict[str, Any]], None] | None = None
    ) -> dict[str, Any]:
        """Run the standard DQN loop for roughly ``total_steps`` environment steps.

        Per step we: act epsilon-greedily, store the transition, and (once we have
        collected ``learning_starts`` steps) do one gradient update on a replayed
        minibatch. The target network is synced every ``target_sync`` learning steps.
        ``callback`` (if given) is invoked once per finished episode with a metrics dict.
        """
        episode_returns: list[float] = []
        episode_successes: list[float] = []
        losses: list[float] = []

        obs, _info = self.env.reset(seed=self.seed)
        ep_return = 0.0
        last_loss = 0.0

        for _ in range(int(total_steps)):
            self._step_count += 1

            # 1) Act epsilon-greedily and step the environment.
            epsilon = self._epsilon()
            action = self._select_action(np.asarray(obs, dtype=np.float32), epsilon)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            reward = float(reward)  # gym types reward as SupportsFloat
            ep_return += reward

            # 2) Store the transition. We bootstrap-off only on TRUE termination, not on
            #    time-limit truncation, so the stored ``done`` flag uses ``terminated``
            #    alone (a truncated episode's final state still has a real future return).
            self.buffer.add(
                np.asarray(obs, dtype=np.float32),
                action,
                reward,
                np.asarray(next_obs, dtype=np.float32),
                bool(terminated),
            )
            obs = next_obs

            # 3) Learn once enough experience has been collected.
            if len(self.buffer) >= self.learning_starts and len(self.buffer) >= self.batch_size:
                last_loss = self._learn()
                losses.append(last_loss)

                # 4) Periodically copy the online weights into the target network.
                #    target_sync=1 (sync every step) is the "no target network" ablation.
                if self._step_count % self.target_sync == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

            # 5) Episode bookkeeping + callback on episode end.
            if terminated or truncated:
                episode_returns.append(ep_return)
                episode_successes.append(float(info.get("is_success", False)))
                # Rolling success rate over the last 100 episodes (a stable, readable metric).
                window = episode_successes[-100:]
                success_rate = float(np.mean(window)) if window else 0.0

                if callback is not None:
                    callback(
                        {
                            "step": self._step_count,
                            "episode": len(episode_returns),
                            "episode_return": ep_return,
                            "success_rate": success_rate,
                            "epsilon": epsilon,
                            "loss": last_loss,
                        }
                    )

                obs, _info = self.env.reset()
                ep_return = 0.0

        return {
            "episode_returns": episode_returns,
            "episode_successes": episode_successes,
            "losses": losses,
            "steps": self._step_count,
        }

    # ----------------------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------------------
    def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, None]:
        """Return ``(action, state)``. State is ``None`` (DQN is memoryless).

        ``deterministic=True`` -> greedy ``argmax`` action (epsilon = 0).
        ``deterministic=False`` -> sample epsilon-greedily at the current epsilon, which
        is handy for evaluating exploration behaviour.
        """
        obs = np.asarray(observation, dtype=np.float32)
        if deterministic:
            return self._greedy_action(obs), None
        return self._select_action(obs, self._epsilon()), None

    # ----------------------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the online Q-network weights with ``torch.save``."""
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load weights into both the online and target networks."""
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
