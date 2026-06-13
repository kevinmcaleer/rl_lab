# Foxglove Setup for the Buddy Jr RL Lab

This guide takes you from a fresh machine to watching the Buddy Jr robot arm
move in a 3D viewer with live training metrics alongside it — in about five
minutes.

---

## 1. Install the Foxglove desktop app

Foxglove Studio is free to download and runs natively on macOS, Linux, and
Windows.

1. Go to <https://foxglove.dev/download> and download the version for your OS.
2. On macOS, drag **Foxglove Studio** into `/Applications` and open it.
   - If macOS shows _"Foxglove Studio cannot be opened because it is from an
     unidentified developer"_, go to **System Settings > Privacy & Security**
     and click **Open Anyway** in the security warning that appeared after you
     tried to launch it.
3. Accept any first-run prompts. No account is required for local use.

---

## 2. Start the WebSocket bridge

The lab exposes a WebSocket server that Foxglove connects to. You need this
running before you open a connection in the app.

### Option A — via Make (recommended)

```bash
make foxglove
```

This runs `rl-lab viz` inside the project's virtual environment, which starts a
server on `ws://127.0.0.1:8765` and prints a `foxglove://` URL you can click to
open the app directly.

### Option B — via the Python script

```bash
python scripts/launch_foxglove_bridge.py
```

Both options start the same server. The script is useful if you need to pass
extra flags such as `--port` or `--mcap-path`.

### macOS "accept incoming connections" prompt

The first time you run the server, macOS Firewall may ask:

> _"Do you want the application python3.x to accept incoming network
> connections?"_

Click **Allow**. The server binds only to `127.0.0.1` (localhost) and is never
exposed to your network. Without this permission the Foxglove app cannot reach
the server even though they are on the same machine.

---

## 3. Connect Foxglove to the running server

1. In Foxglove Studio, click **Open connection** (top-left of the home screen,
   or **File > Open connection** if you already have a layout open).
2. Select **Foxglove WebSocket** from the connection type list.
3. Set the URL to:
   ```
   ws://localhost:8765
   ```
4. Click **Open**.

Foxglove will connect and begin receiving data. If the bridge is not running yet
you will see a _"Connection refused"_ error — start the bridge first, then
retry.

---

## 4. Import the Buddy Jr layout

The repo ships a pre-built layout that arranges all the panels for you:

1. In Foxglove Studio, click the **Layouts** icon in the left sidebar (the
   grid-of-squares icon), or open **Layouts** from the top-right menu.
2. Click **Import layout from file** (or the upload icon).
3. Navigate to `rl_lab/viz/layouts/buddy_jr.json` inside the repo and select it.
4. The layout loads immediately. You can rename it to _"Buddy Jr RL Lab"_ if
   you like; Foxglove remembers it between sessions.

---

## 5. What each panel shows

The layout has four panels arranged as: **3D viewer on the left** taking 62 %
of the width, and **three plot panels stacked on the right**.

### 3D panel (left)

Subscribes to three topics:

| Topic | Content |
|-------|---------|
| `/tf` | `FrameTransforms` — one transform per joint, updated every control step. This animates the URDF arm in real time as the agent applies actions. |
| `/robot` | `SceneUpdate` — the URDF link meshes / primitive geometry (sent once at startup). |
| `/scene` | `SceneUpdate` — the goal sphere (amber while unreached, green when within 2 cm), the blue camera-tip marker, and a white line joining tip to target. |

The camera is positioned slightly above and to the side of the robot. The
display frame is **world**, so all geometry stays in the one fixed reference
frame.

### Plot: Distance to target (top right)

Subscribes to `/metrics.distance` (metres). Tracks how far the camera tip is
from the goal in real time. In early training this wobbles randomly; a good
policy drives it steadily toward zero.

### Plot: Reward & episode return (middle right)

Subscribes to:
- `/metrics.reward` — the per-step shaping reward (positive = closer, negative = farther).
- `/metrics.episode_return` — the cumulative sum of rewards in the current episode.

Watch the episode return grow over training as the agent learns to hold the tip
near the target for longer.

### Plot: Success rate (bottom right)

Subscribes to `/metrics.success_rate` (0.0 to 1.0). Shows the rolling fraction
of episodes where the tip reached the goal. The Y-axis is locked to `[0, 1]`.
Aim for > 0.8 before declaring an experiment successful.

---

## 6. Tips

- **Replay with MCAP.** Run your experiment with `render_mode='mcap'` to write a
  `.mcap` file, then open it in Foxglove via **Open local file**. Scrub through
  the entire training run at any speed.
- **Resize panels** by dragging the dividers between them.
- **Orbit the 3D view** with left-drag; zoom with scroll; pan with right-drag or
  middle-drag.
- **Topic visibility** — use the panel settings (gear icon) in the 3D panel to
  toggle `/robot`, `/scene`, or `/tf` on/off independently.
- If the arm disappears, check that the bridge is still running (`make foxglove`)
  and that `/tf` is receiving frames (the Topics list in the 3D panel settings
  shows a green dot for live topics).
- The layout is JSON and version-controlled. Feel free to add more Plot panels or
  rearrange them and re-export via **Layouts > Export layout to file**.
