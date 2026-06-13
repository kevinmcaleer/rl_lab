#!/usr/bin/env python3
"""launch_foxglove_bridge.py -- Start a live Foxglove WebSocket bridge for Buddy Jr.

This script is the fastest way to get the arm visible in Foxglove without
running a full simulation.  It:

  1. Parses ``--host``, ``--port``, and an optional ``--mcap`` path.
  2. Starts a :class:`~rl_lab.viz.foxglove_bridge.FoxgloveStreamer` in
     ``"foxglove"`` mode (and optionally also records to an MCAP file).
  3. Publishes the arm once at the zero pose so Foxglove immediately shows
     the URDF geometry and a demo goal sphere.
  4. Idles (with a 0.5 s poll loop) until the user presses Ctrl-C.
  5. Closes the streamer cleanly on exit.

Usage
-----
    python scripts/launch_foxglove_bridge.py
    python scripts/launch_foxglove_bridge.py --port 8766
    python scripts/launch_foxglove_bridge.py --mcap recording.mcap

    # or via the Makefile:
    make foxglove

macOS note
----------
    On first launch macOS will show a firewall prompt asking whether to
    "accept incoming network connections".  The server binds to *localhost*
    only (127.0.0.1), so it is safe to click **Allow** -- no external traffic
    is involved.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# A fixed demo goal that sits comfortably inside the arm's reachable workspace
# (x = 0.10 m forward, z = 0.10 m up from base_link).
_DEMO_GOAL: np.ndarray = np.array([0.10, 0.0, 0.10])

# Zero-pose joint vector: all joints at 0 rad.
_ZERO_Q: np.ndarray = np.zeros(4)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launch_foxglove_bridge",
        description="Stream the Buddy Jr arm to a Foxglove WebSocket server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Open foxglove://open?ds=foxglove-websocket&ds.url=ws://HOST:PORT\n"
            "or paste the printed URL into Foxglove Studio (Help > Open URL)."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the WebSocket server to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="TCP port for the WebSocket server (default: 8765).",
    )
    parser.add_argument(
        "--mcap",
        metavar="PATH",
        default=None,
        help="Optional path to also record the session as an MCAP file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns an exit code (0 = clean, 1 = error)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazily import the streamer so the top-level import remains fast.
    from rl_lab.robot.kinematics import forward_position
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

    print(f"Starting Foxglove bridge on {args.host}:{args.port} ...")
    if args.mcap:
        print(f"Recording session to: {args.mcap}")

    try:
        streamer = FoxgloveStreamer(
            render_mode="foxglove",
            host=args.host,
            port=args.port,
            render_fps=30.0,
            mcap_path=args.mcap,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not start Foxglove server: {exc}", file=sys.stderr)
        return 1

    with streamer:
        # Print connection URLs so the learner can copy-paste them.
        ws_url = f"ws://{args.host}:{args.port}"
        print(f"WebSocket URL : {ws_url}")
        if streamer.app_url is not None:
            print(f"Desktop app  : {streamer.app_url}")
        print()
        print("macOS firewall note: if prompted, click 'Allow' -- the server")
        print("listens on localhost only; no external network access occurs.")
        print()
        print("Buddy Jr is at the zero pose.  Open Foxglove and add a 3D panel.")
        print("Press Ctrl-C to stop.")
        print()

        # Publish the arm once at the zero pose so Foxglove immediately shows
        # the geometry.  ``force=True`` bypasses the render-fps throttle.
        tip = forward_position(_ZERO_Q)
        dist = float(np.linalg.norm(tip - _DEMO_GOAL))
        streamer.publish(
            joint_q=_ZERO_Q,
            p_ee=tip,
            g=_DEMO_GOAL,
            dist=dist,
            force=True,
        )

        # Idle loop -- keep the server alive until Ctrl-C.
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nShutting down Foxglove bridge. Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
