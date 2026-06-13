"""ROS2/rviz2 optional visualization back-end for Buddy Jr RL Lab.

This module provides ``ROS2Publisher``, a drop-in replacement for
:class:`rl_lab.viz.foxglove_bridge.FoxgloveStreamer` that publishes to ROS2
topics instead of a Foxglove WebSocket.  Lesson code (``experiments/``) can
switch visualisation back-ends by changing **one constructor call** — the rest
of each script stays byte-for-byte identical.

What this publishes
-------------------
* ``/joint_states``  :  ``sensor_msgs/msg/JointState``
  Joint names, positions (radians), and zero velocities/efforts.  Run
  ``robot_state_publisher`` *separately* (see
  ``docs/getting_started/installation_ros2.md``) and it will broadcast the full
  TF tree from the URDF + this JointState stream.  We deliberately do **not**
  republish TF here — that is robot_state_publisher's job.

* ``/target_marker`` :  ``visualization_msgs/msg/Marker``
  A sphere at the goal position.  Colour: green when the tip is within
  ``success_radius`` of the goal, amber otherwise.  The rviz2 config
  (``buddy_jr.rviz``) adds a *Marker* display subscribed to this topic.

Platform note
-------------
ROS2 (rclpy) **cannot be installed natively on macOS**.  This file must only
be used inside the Docker container or Linux VM described in
``docs/getting_started/installation_ros2.md``.  All ``rclpy`` and
``sensor_msgs`` / ``visualization_msgs`` imports are *lazy* — they happen
inside ``__init__`` so ``import rl_lab.viz.rviz.ros2_publisher`` itself
never fails on macOS; only *instantiation* raises ``ImportError``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

# rl_lab contracts — always available (no ROS2 dependency).
from rl_lab.robot.buddy_jr import JOINT_NAMES

if TYPE_CHECKING:
    # Only resolved by type-checkers, never at runtime.
    import rclpy
    import rclpy.node

# Marker.ADD constant (we replicate the value here so we don't need to import
# visualization_msgs at module level).
_MARKER_ADD: int = 0
_MARKER_SPHERE: int = 2  # visualization_msgs/msg/Marker::SPHERE

# Default ROS2 frame IDs.
_WORLD_FRAME: str = "world"
_BASE_FRAME: str = "base_link"

# Topic names (match the rviz2 config).
_JOINT_STATE_TOPIC: str = "/joint_states"
_TARGET_MARKER_TOPIC: str = "/target_marker"

# Default success radius (metres).  Matches FoxgloveStreamer's default.
_DEFAULT_SUCCESS_RADIUS: float = 0.02

# Colours: RGBA 0-1.  Match rl_lab/viz/schemas.py GREEN / AMBER.
_COLOUR_GREEN: tuple[float, float, float, float] = (0.1, 0.8, 0.2, 1.0)
_COLOUR_AMBER: tuple[float, float, float, float] = (0.95, 0.6, 0.1, 1.0)


class ROS2Publisher:
    """Drop-in replacement for :class:`~rl_lab.viz.foxglove_bridge.FoxgloveStreamer`.

    Publishes ``sensor_msgs/JointState`` and a target ``visualization_msgs/Marker``
    to a live ROS2 graph.  Use ``robot_state_publisher`` (run separately) to
    generate TF from the JointState + the URDF.

    Parameters
    ----------
    render_mode:
        ``"ros2"`` (active) or ``"off"`` (no-op, matching FoxgloveStreamer).
    node_name:
        Name of the rclpy node this publisher creates.
    render_fps:
        Maximum publish rate in Hz.  Publishing is throttled; a no-op step
        costs nothing beyond a monotonic clock read.
    success_radius:
        Tip-to-goal distance (metres) that counts as a success.  Controls
        the target marker colour (green/amber).
    """

    def __init__(
        self,
        render_mode: str = "ros2",
        *,
        node_name: str = "buddy_jr_rl_lab",
        render_fps: float = 30.0,
        success_radius: float = _DEFAULT_SUCCESS_RADIUS,
    ) -> None:
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.success_radius = success_radius
        self._min_period: float = 1.0 / render_fps if render_fps > 0 else 0.0
        self._last_publish: float = -1e9

        # These are set to None when render_mode is 'off' — same pattern as
        # FoxgloveStreamer so callers can test `if self.enabled`.
        self._node: rclpy.node.Node | None = None
        self._js_pub: rclpy.publisher.Publisher | None = None  # type: ignore[name-defined]
        self._marker_pub: rclpy.publisher.Publisher | None = None  # type: ignore[name-defined]

        if render_mode == "off":
            return

        # --- Lazy ROS2 imports -------------------------------------------
        # All ROS2 packages are imported here so that the module itself can
        # be imported on macOS without error (rclpy is Linux/Docker only).
        try:
            import rclpy as _rclpy
            from sensor_msgs.msg import JointState as _JointState
            from visualization_msgs.msg import Marker as _Marker
        except ImportError as exc:
            raise ImportError(
                "rclpy / sensor_msgs / visualization_msgs could not be imported. "
                "The ROS2 visualisation back-end must run inside the Docker container "
                "or a Linux VM — never natively on macOS.  See "
                "docs/getting_started/installation_ros2.md for setup instructions."
            ) from exc

        # Store references for use in publish() / close().
        self._rclpy = _rclpy
        self._JointState = _JointState
        self._Marker = _Marker

        # Initialise rclpy if it hasn't been already (idempotent).
        if not _rclpy.ok():
            _rclpy.init()

        self._node = _rclpy.create_node(node_name)
        self._js_pub = self._node.create_publisher(_JointState, _JOINT_STATE_TOPIC, 10)
        self._marker_pub = self._node.create_publisher(_Marker, _TARGET_MARKER_TOPIC, 10)

    # ------------------------------------------------------------------
    # Public interface — mirrors FoxgloveStreamer exactly.
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when ``render_mode`` is not ``'off'``."""
        return self.render_mode != "off"

    @property
    def app_url(self) -> str | None:
        """rviz2 does not generate a URL; always returns ``None``.

        Present so callers can do ``if streamer.app_url: webbrowser.open(...)``
        without an ``AttributeError``.
        """
        return None

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
        """Publish one frame of robot state to ROS2.

        Throttled to ``render_fps``; a no-op when ``render_mode='off'``.

        Parameters
        ----------
        joint_q:
            Joint positions in radians, shape ``(4,)``.
            Order: ``[base_yaw, shoulder_pitch, elbow_pitch, camera_tilt]``.
        p_ee:
            End-effector (camera tip) world position, shape ``(3,)``.
        g:
            Goal / target position in world frame, shape ``(3,)``.
        dist:
            Euclidean distance (metres) from tip to goal.
        reward:
            Scalar reward for this step (logged but not published to ROS2).
        episode_return:
            Cumulative episode return (logged but not published to ROS2).
        success_rate:
            Running success fraction (logged but not published to ROS2).
        force:
            If ``True``, bypass the rate limiter and publish immediately.
        """
        if not self.enabled:
            return

        now = time.monotonic()
        if not force and (now - self._last_publish) < self._min_period:
            return
        self._last_publish = now

        self._publish_joint_state(np.asarray(joint_q, dtype=np.float64))
        self._publish_target_marker(
            target=np.asarray(g, dtype=np.float64),
            within_tolerance=float(dist) <= self.success_radius,
        )

        # Spin once so subscribers receive the messages without blocking.
        # rclpy.spin_once returns immediately if there is nothing to process.
        self._rclpy.spin_once(self._node, timeout_sec=0.0)  # type: ignore[union-attr]

    def close(self) -> None:
        """Destroy the ROS2 node and shut down rclpy.

        Idempotent — safe to call multiple times or when ``render_mode='off'``.
        """
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        # Only shut down if we initialised rclpy ourselves and it is still running.
        if self.enabled and hasattr(self, "_rclpy") and self._rclpy.ok():
            self._rclpy.shutdown()

    # Context-manager support — mirrors FoxgloveStreamer.
    def __enter__(self) -> ROS2Publisher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ros_stamp(self) -> object:
        """Return a ROS2 ``Time`` message for the current wall clock."""
        from builtin_interfaces.msg import Time  # always available with rclpy

        t = time.time()
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        return Time(sec=sec, nanosec=nanosec)

    def _publish_joint_state(self, joint_q: np.ndarray) -> None:
        """Build and publish a ``sensor_msgs/JointState`` message.

        ``robot_state_publisher`` subscribes to this topic and, combined with
        the URDF loaded via the ``/robot_description`` parameter, broadcasts the
        full TF tree.  rviz2 then renders the RobotModel from that TF tree.
        """
        msg = self._JointState()
        msg.header.stamp = self._ros_stamp()  # type: ignore[assignment]
        msg.header.frame_id = ""  # convention: empty for JointState
        # JOINT_NAMES = ('base_yaw', 'shoulder_pitch', 'elbow_pitch', 'camera_tilt')
        msg.name = list(JOINT_NAMES)
        msg.position = [float(v) for v in joint_q]
        # Velocities and efforts are not tracked by the RL sim;
        # publishing empty lists is the ROS2 convention for "unknown".
        msg.velocity = []
        msg.effort = []
        self._js_pub.publish(msg)  # type: ignore[union-attr]

    def _publish_target_marker(
        self,
        target: np.ndarray,
        within_tolerance: bool,
    ) -> None:
        """Build and publish a ``visualization_msgs/Marker`` sphere at the goal.

        The marker is green when the tip is within ``success_radius`` of the
        goal (matching the Foxglove scene colours), and amber otherwise.  The
        frame_id is ``world`` — make sure rviz2 is set to the ``world`` fixed
        frame, or change it to ``base_link`` if you prefer.
        """
        msg = self._Marker()
        msg.header.stamp = self._ros_stamp()  # type: ignore[assignment]
        msg.header.frame_id = _WORLD_FRAME
        msg.ns = "rl_lab"
        msg.id = 0  # unique per namespace; only one target at a time
        msg.type = _MARKER_SPHERE
        msg.action = _MARKER_ADD

        # Position in world frame.
        msg.pose.position.x = float(target[0])
        msg.pose.position.y = float(target[1])
        msg.pose.position.z = float(target[2])
        # Identity orientation (sphere does not need rotation).
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        # Diameter 2 cm — same visual size as the Foxglove target sphere.
        diameter = 0.02
        msg.scale.x = diameter
        msg.scale.y = diameter
        msg.scale.z = diameter

        # Colour: green inside tolerance, amber outside (matches schemas.py).
        r, g, b, a = _COLOUR_GREEN if within_tolerance else _COLOUR_AMBER
        msg.color.r = r
        msg.color.g = g
        msg.color.b = b
        msg.color.a = a

        # Lifetime = 0 means the marker persists until replaced or deleted.
        msg.lifetime.sec = 0
        msg.lifetime.nanosec = 0

        self._marker_pub.publish(msg)  # type: ignore[union-attr]
