"""The common ``Algorithm`` contract shared by every learner in the lab.

Both the from-scratch teaching implementations (tabular Q-learning, SARSA, DQN,
REINFORCE, minimal PPO) and the Stable-Baselines3 adapters satisfy this tiny
protocol, so the registry, the training CLI and the smoke tests treat them all
the same way::

    algo = make_algorithm("dqn", env, seed=0)
    algo.train(total_steps=1000)
    action, _ = algo.predict(obs)
    algo.save("runs/dqn/model")

It is deliberately minimal — the educational value is in each algorithm's body,
not in a heavy base class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

#: A training callback is called once per logged step/episode with a metrics
#: dict (e.g. ``{"step": 1000, "episode_return": 3.2, "success_rate": 0.4}``).
TrainCallback = Callable[[dict[str, Any]], None]


@runtime_checkable
class Algorithm(Protocol):
    """Minimal interface every algorithm implements."""

    def train(self, total_steps: int, callback: TrainCallback | None = None) -> dict[str, Any]:
        """Train for roughly ``total_steps`` env steps; return a history dict."""
        ...

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, Any]:
        """Return ``(action, state)`` for an observation (state is usually None)."""
        ...

    def save(self, path: str) -> None:
        """Persist the policy/value parameters to ``path``."""
        ...

    def load(self, path: str) -> None:
        """Load parameters from ``path`` into this (already-constructed) algo."""
        ...
