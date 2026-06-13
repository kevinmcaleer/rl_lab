"""Stream scalar training metrics to a Foxglove plot panel.

A learner watches ``distance``, ``reward``, ``episode_return`` and
``success_rate`` curves *next to* the live 3D scene. These mirror what the
TensorBoard logger records, and the training callbacks (M4) call
:meth:`LiveMetrics.publish` once per step / episode.
"""

from __future__ import annotations

import foxglove

from rl_lab.viz import schemas


class LiveMetrics:
    """A thin wrapper over a JSON Foxglove channel for numeric metrics."""

    def __init__(self, topic: str = "/metrics") -> None:
        self._channel = foxglove.Channel(topic, message_encoding="json")

    def publish(
        self,
        distance: float,
        reward: float,
        episode_return: float = 0.0,
        success_rate: float = 0.0,
    ) -> None:
        """Log one metrics sample (a JSON object) to the channel."""
        self._channel.log(
            schemas.metrics_message(
                distance=distance,
                reward=reward,
                episode_return=episode_return,
                success_rate=success_rate,
            )
        )

    def close(self) -> None:
        self._channel.close()
