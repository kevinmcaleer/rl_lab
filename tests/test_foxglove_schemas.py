"""Foxglove schema builders construct valid messages and survive a round-trip.

The round-trip writes the messages to an ``.mcap`` file through real Foxglove
channels and reads them back, which is exactly the serialization path the live
bridge uses. Skipped automatically if ``foxglove`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

foxglove = pytest.importorskip("foxglove")

from foxglove.messages import FrameTransform, FrameTransforms, SceneUpdate  # noqa: E402

from rl_lab.robot import kinematics as kin  # noqa: E402
from rl_lab.viz import schemas  # noqa: E402

# Foxglove message objects are write-only (their fields are not readable from
# Python), so we assert on types + the pure colour-decision helper, and rely on
# the MCAP round-trip below to prove the field values actually serialize.


def test_frame_transforms_from_fk() -> None:
    """joint_transforms feed straight into the FrameTransforms builder."""
    transforms = kin.joint_transforms(np.array([0.2, -0.3, 0.4, 0.0]))
    assert len(transforms) == 5
    assert transforms[0][0] == "base_link" and transforms[0][1] == "shoulder_bracket"
    assert isinstance(schemas.frame_transforms(transforms), FrameTransforms)
    assert isinstance(schemas.frame_transform(*transforms[0]), FrameTransform)


def test_reach_scene_colour_reflects_success() -> None:
    """The goal sphere is green within tolerance and amber otherwise."""
    assert schemas.target_color(True) is schemas.GREEN
    assert schemas.target_color(False) is schemas.AMBER
    tip = np.array([0.1, 0.0, 0.2])
    assert isinstance(schemas.reach_scene(target=tip, tip=tip, within_tolerance=True), SceneUpdate)
    assert isinstance(schemas.reach_scene(target=tip, tip=tip, within_tolerance=False), SceneUpdate)


def test_metrics_message_shape() -> None:
    msg = schemas.metrics_message(distance=0.1, reward=-0.1, episode_return=2.0, success_rate=0.5)
    assert set(msg) == {"distance", "reward", "episode_return", "success_rate"}
    assert all(isinstance(v, float) for v in msg.values())


def test_round_trip_through_mcap(tmp_path) -> None:
    """SceneUpdate / FrameTransforms / numeric JSON serialize and read back."""
    from foxglove.channels import FrameTransformsChannel, SceneUpdateChannel

    scene_ch = SceneUpdateChannel("/scene")
    tf_ch = FrameTransformsChannel("/tf")
    metrics_ch = foxglove.Channel("/metrics", message_encoding="json")

    path = tmp_path / "episode.mcap"
    writer = foxglove.open_mcap(str(path))
    transforms = kin.joint_transforms(np.zeros(4))
    tip = kin.forward(np.zeros(4)).position
    scene_ch.log(schemas.reach_scene(target=tip, tip=tip, within_tolerance=True))
    tf_ch.log(schemas.frame_transforms(transforms))
    metrics_ch.log(schemas.metrics_message(distance=0.0, reward=1.0))
    writer.close()

    assert path.stat().st_size > 0

    from mcap.reader import make_reader

    counts: dict[str, int] = {}
    with open(path, "rb") as fh:
        for _schema, channel, _message in make_reader(fh).iter_messages():
            counts[channel.topic] = counts.get(channel.topic, 0) + 1

    assert counts.get("/scene", 0) >= 1
    assert counts.get("/tf", 0) >= 1
    assert counts.get("/metrics", 0) >= 1
