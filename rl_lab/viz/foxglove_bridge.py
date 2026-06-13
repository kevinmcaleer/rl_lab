"""``FoxgloveStreamer`` — the live link between the sim and Foxglove.

It runs an in-process ``foxglove-sdk`` WebSocket server (default
``ws://localhost:8765``) and/or records the same channels to an ``.mcap`` file
you can scrub later. One call per control step:

    streamer.publish(joint_q, p_ee, g, dist)

moves the URDF-shaped arm (per-step ``FrameTransforms``), shows the goal sphere
(green within tolerance), a tip marker and a tip->target line, and streams the
numeric metrics. Publishing is throttled to ``render_fps`` and is a complete
no-op when ``render_mode`` is ``"off"`` — so training at full speed pays nothing
for visualization it isn't using.

Render modes:

* ``"foxglove"`` — live WebSocket server (optionally also record to MCAP).
* ``"mcap"``     — record to an ``.mcap`` file only, no server (headless capture).
* ``"off"``      — disabled; every :meth:`publish` returns immediately.

macOS note: the first time the server opens a localhost port, macOS may ask to
"accept incoming connections". It is localhost-only and safe to allow.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

RenderMode = Literal["foxglove", "mcap", "off"]

ROBOT_TOPIC = "/robot"
SCENE_TOPIC = "/scene"
TF_TOPIC = "/tf"

DEFAULT_PORT = 8765
DEFAULT_SUCCESS_RADIUS = 0.02  # metres; tip within this of the goal => "reached"


class FoxgloveStreamer:
    """Publish per-step robot state + metrics to Foxglove and/or an MCAP file."""

    def __init__(
        self,
        render_mode: RenderMode = "foxglove",
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        render_fps: float = 30.0,
        mcap_path: str | None = None,
        success_radius: float = DEFAULT_SUCCESS_RADIUS,
    ) -> None:
        self.render_mode: RenderMode = render_mode
        self.render_fps = render_fps
        self.success_radius = success_radius
        self._min_period = 1.0 / render_fps if render_fps > 0 else 0.0
        self._last_publish = -1e9
        self._geometry_sent = False
        self._server = None
        self._mcap = None
        self._scene_ch = None
        self._robot_ch = None
        self._tf_ch = None
        self._metrics = None

        if render_mode == "off":
            return

        # Import lazily so `import rl_lab.viz` never hard-requires foxglove-sdk.
        import foxglove
        from foxglove.channels import FrameTransformsChannel, SceneUpdateChannel

        from rl_lab.viz.live_metrics import LiveMetrics

        if render_mode == "foxglove":
            self._server = foxglove.start_server(host=host, port=port)
        if mcap_path is not None:
            self._mcap = foxglove.open_mcap(mcap_path)

        self._robot_ch = SceneUpdateChannel(ROBOT_TOPIC)
        self._scene_ch = SceneUpdateChannel(SCENE_TOPIC)
        self._tf_ch = FrameTransformsChannel(TF_TOPIC)
        self._metrics = LiveMetrics()

    # ----------------------------------------------------------------- API --
    @property
    def enabled(self) -> bool:
        return self.render_mode != "off"

    @property
    def app_url(self) -> str | None:
        """A ``foxglove://`` URL that opens the desktop app onto this server."""
        return self._server.app_url() if self._server is not None else None

    def publish(
        self,
        joint_q: np.ndarray,
        p_ee: np.ndarray,
        g: np.ndarray,
        dist: float,
        *,
        reward: float = 0.0,
        episode_return: float = 0.0,
        success_rate: float = 0.0,
        force: bool = False,
    ) -> None:
        """Stream one frame. Throttled to ``render_fps``; a no-op when disabled."""
        if not self.enabled:
            return

        now = time.monotonic()
        if not force and (now - self._last_publish) < self._min_period:
            return
        self._last_publish = now

        # Imported here to keep the module import cheap and engine-free.
        from rl_lab.viz import schemas
        from rl_lab.viz.urdf_publisher import robot_geometry_scene, robot_transforms

        if not self._geometry_sent:
            self._robot_ch.log(robot_geometry_scene())  # type: ignore[union-attr]
            self._geometry_sent = True

        self._tf_ch.log(robot_transforms(np.asarray(joint_q)))  # type: ignore[union-attr]
        within_tol = float(dist) <= self.success_radius
        self._scene_ch.log(  # type: ignore[union-attr]
            schemas.reach_scene(
                target=np.asarray(g), tip=np.asarray(p_ee), within_tolerance=within_tol
            )
        )
        self._metrics.publish(  # type: ignore[union-attr]
            distance=float(dist),
            reward=reward,
            episode_return=episode_return,
            success_rate=success_rate,
        )

    def close(self) -> None:
        """Stop the server, flush/close the MCAP file, and close channels."""
        if self._server is not None:
            self._server.stop()
            self._server = None
        if self._mcap is not None:
            self._mcap.close()
            self._mcap = None
        for ch in (self._scene_ch, self._robot_ch, self._tf_ch):
            if ch is not None:
                ch.close()
        if self._metrics is not None:
            self._metrics.close()

    def __enter__(self) -> FoxgloveStreamer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
