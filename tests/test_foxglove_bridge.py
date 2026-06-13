"""FoxgloveStreamer behaviour: no-op when off, records all channels, throttles.

Uses ``render_mode="mcap"`` so no WebSocket server (or firewall prompt) is
needed — the publish path is identical, just written to a file we read back.
Skipped automatically if ``foxglove`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("foxglove")

from rl_lab.robot import kinematics as kin  # noqa: E402
from rl_lab.viz.foxglove_bridge import FoxgloveStreamer  # noqa: E402


def _mcap_topic_counts(path) -> dict[str, int]:
    from mcap.reader import make_reader

    counts: dict[str, int] = {}
    with open(path, "rb") as fh:
        for _schema, channel, _message in make_reader(fh).iter_messages():
            counts[channel.topic] = counts.get(channel.topic, 0) + 1
    return counts


def test_off_mode_is_a_noop() -> None:
    streamer = FoxgloveStreamer("off")
    assert not streamer.enabled
    # Must not raise and must do nothing.
    streamer.publish(np.zeros(4), np.zeros(3), np.ones(3), dist=1.0)
    streamer.close()


def test_mcap_records_all_channels(tmp_path) -> None:
    path = tmp_path / "episode.mcap"
    # render_fps=0 disables throttling so every frame is logged.
    with FoxgloveStreamer("mcap", mcap_path=str(path), render_fps=0) as streamer:
        for sh in np.linspace(-0.5, 0.5, 6):
            q = np.array([0.2, sh, -0.3, 0.0])
            tip = kin.forward(q).position
            goal = np.array([0.1, 0.05, 0.18])
            streamer.publish(q, tip, goal, dist=float(np.linalg.norm(tip - goal)), reward=-0.1)

    counts = _mcap_topic_counts(path)
    assert counts.get("/tf", 0) == 6  # one per frame
    assert counts.get("/scene", 0) == 6
    assert counts.get("/metrics", 0) == 6
    assert counts.get("/robot", 0) == 1  # geometry sent exactly once


def test_throttle_limits_publishes(tmp_path) -> None:
    path = tmp_path / "throttled.mcap"
    # A ~1000 s period means only the first publish in a tight loop gets through.
    with FoxgloveStreamer("mcap", mcap_path=str(path), render_fps=0.001) as streamer:
        for _ in range(5):
            streamer.publish(np.zeros(4), np.zeros(3), np.ones(3), dist=1.0)

    counts = _mcap_topic_counts(path)
    assert counts.get("/tf", 0) == 1
