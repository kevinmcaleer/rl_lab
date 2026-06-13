# ROS2 / rviz2 Installation (Advanced Track)

> **Do NOT install ROS2 natively on macOS.**  ROS2 Humble (and later) has no
> official macOS arm64 support.  Attempting a source build on Apple Silicon is
> fragile, time-consuming, and unnecessary — the Docker path below takes under
> 10 minutes and works perfectly on any Mac, Linux machine, or Windows host.

This guide covers the optional ROS2 visualisation path for Buddy Jr RL Lab.
All RL code is identical whether you use Foxglove (the default, no ROS required)
or rviz2.  Only the visualisation *sink* changes.

If you just want to run the experiments, use the default Foxglove path described
in `docs/getting_started/installation.md`.  Come here only when you already live
in ROS2 or want the full "joint-angles → robot_state_publisher → TF → rviz2"
pipeline as a learning exercise.

---

## Prerequisites

- Docker Desktop (macOS / Windows) **or** Docker Engine (Linux).  
  Download: <https://docs.docker.com/get-docker/>
- The `rl_lab` repository cloned on your host machine.
- (Optional) An X11 server on macOS (e.g. XQuartz) **or** use
  `rviz2 --display` inside the container via VNC — both options are described
  below.

---

## 1. Pull the ROS2 Humble image

```bash
docker pull ros:humble
```

Humble is the current LTS release (supported until May 2027) and the version
the lab is tested against.  If you need a newer distro, substitute `ros:iron`
or `ros:jazzy` — the commands are identical.

---

## 2. Start an interactive container

Mount the `rl_lab` repository into the container so edits you make on the host
are immediately visible inside Docker:

```bash
docker run -it --rm \
  --name rl_lab_ros2 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --network host \
  ros:humble \
  bash
```

**macOS with XQuartz:** Start XQuartz first, then run
`xhost +localhost` in a Terminal before launching the container.  Set
`-e DISPLAY=host.docker.internal:0` instead of `$DISPLAY`.

**macOS without XQuartz (VNC option):** Add a VNC server inside the container
(e.g. `tigervnc`) and connect from your Mac's built-in Screen Sharing app.  A
simpler alternative is to record and replay sessions with `ros2 bag` instead of
running rviz2 interactively.

---

## 3. Install rl_lab inside the container

Run these commands *inside* the container (`docker exec` or the shell from
step 2):

```bash
# Source the ROS2 environment.
source /opt/ros/humble/setup.bash

# Install rl_lab with its ROS2 optional dependencies.
pip install -e "/workspace[ros2]"

# Verify rclpy is importable.
python -c "import rclpy; print('rclpy OK')"
```

The `[ros2]` extra installs nothing extra at the Python level (rclpy is already
present in the ROS image); the bracket exists so the dependency is declared in
`pyproject.toml` and so tooling can detect the optional path.

---

## 4. Run robot_state_publisher

`robot_state_publisher` converts the URDF + incoming `sensor_msgs/JointState`
messages into a live TF tree that rviz2 uses to render the robot.

Open a *second* terminal in the container (or a new `docker exec` session):

```bash
source /opt/ros/humble/setup.bash

# Read the URDF from the repository.
export URDF_PATH=/workspace/urdf/buddy_jr.urdf

# Launch robot_state_publisher with the URDF loaded as a parameter.
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args \
  -p robot_description:="$(cat $URDF_PATH)"
```

You should see a log line like:

```
[robot_state_publisher]: Robot initialized
```

Leave this terminal running throughout your session.

---

## 5. Start rviz2 with the Buddy Jr config

Open a *third* terminal in the container:

```bash
source /opt/ros/humble/setup.bash

rviz2 -d /workspace/rl_lab/viz/rviz/buddy_jr.rviz
```

The rviz2 config (`rl_lab/viz/rviz/buddy_jr.rviz`) pre-loads four displays:

| Display | What you see |
|---|---|
| **Grid** | Ground-plane reference grid at z=0 |
| **RobotModel** | Buddy Jr arm driven by the URDF + TF |
| **TF** | Coordinate frames for each link |
| **TargetMarker** | Goal sphere (green = tip inside tolerance, amber = not yet) |

rviz2 will show the arm at its zero pose until an experiment starts publishing
`/joint_states`.

---

## 6. Run a lesson with the ROS2 back-end

The `ROS2Publisher` class in `rl_lab/viz/rviz/ros2_publisher.py` has the same
interface as `FoxgloveStreamer`.  To use it, replace the streamer import in any
experiment script:

```python
# Default Foxglove path (no change needed on macOS / plain Linux):
# from rl_lab.viz.foxglove_bridge import FoxgloveStreamer as Streamer

# ROS2 path (inside the Docker container):
from rl_lab.viz.rviz.ros2_publisher import ROS2Publisher as Streamer

with Streamer(render_mode="ros2") as streamer:
    # ... training loop ...
    streamer.publish(joint_q, p_ee, g, dist)
```

No other change is needed.  The `publish()` signature is identical, including
the optional keyword arguments (`reward`, `episode_return`, `success_rate`,
`force`).

---

## 7. What the pipeline looks like end-to-end

```
 experiment script
      |
      | streamer.publish(joint_q, p_ee, g, dist)
      v
 ROS2Publisher  (rl_lab/viz/rviz/ros2_publisher.py)
      |
      |-- publishes /joint_states (sensor_msgs/JointState)
      |-- publishes /target_marker (visualization_msgs/Marker)
      v
 robot_state_publisher    (terminal 2)
      |
      | reads /robot_description param (buddy_jr.urdf)
      | + /joint_states  ->  broadcasts TF tree
      v
 rviz2  (terminal 3)
      |
      |-- RobotModel display renders arm from TF + URDF meshes
      |-- TF display shows per-link coordinate frames
      |-- Marker display shows target sphere
```

This is the same conceptual flow as the Foxglove path — joint angles flow into
a transform tree which drives the 3D geometry — but using the standard ROS2
messaging stack instead of the Foxglove SDK.  Comparing the two paths
side-by-side is itself a useful exercise: the policy never changes; only the
rendering sink does.

---

## Troubleshooting

**`rclpy` not found outside Docker**

If you see:

```
ImportError: rclpy / sensor_msgs / visualization_msgs could not be imported.
```

You are trying to instantiate `ROS2Publisher` outside the ROS2 Docker
environment.  Use the Foxglove back-end instead, or start the Docker container
as described in steps 1-2 above.

**rviz2 shows no robot**

Check that `robot_state_publisher` is running (step 4) and has loaded the URDF.
Run `ros2 topic echo /joint_states` to confirm the experiment is publishing.
In rviz2, verify the *Fixed Frame* is set to `world`.

**rviz2 does not appear (macOS / no display)**

Ensure XQuartz is running and `xhost +localhost` has been run before starting
the container.  Check `echo $DISPLAY` inside the container; it should be
`host.docker.internal:0` or similar.

**TF tree looks wrong**

Confirm the URDF path in the `robot_state_publisher` command (step 4) points to
`urdf/buddy_jr.urdf` inside the container (i.e. `/workspace/urdf/buddy_jr.urdf`).
The URDF is the single source of truth: if the file path is wrong, TF will be
silent and the arm will not render.

---

## Why not install ROS2 natively on macOS?

ROS2 has no official macOS arm64 (Apple Silicon) support.  A source build
requires patching several C++ dependencies, is broken by routine Xcode and
Homebrew updates, and is not tested by the ROS2 project.  Even when it works,
the result is fragile and frequently breaks after `brew upgrade`.

The Docker path is faster to set up, reproducible across machines, and matches
the environment the lab's CI tests against.  It also teaches good habits: in
real robotics projects, the robot's software runs in a controlled Linux
environment regardless of the developer's host OS.

All rl_lab Python packages — including `rl_lab.viz.rviz.ros2_publisher` — import
cleanly on macOS without rclpy installed because all ROS2 imports are lazy
(inside `__init__`, not at module level).  Only *instantiating* `ROS2Publisher`
with `render_mode="ros2"` raises `ImportError` outside the container.
