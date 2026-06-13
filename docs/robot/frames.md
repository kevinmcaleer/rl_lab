# Buddy Jr Frame Conventions

This page defines the coordinate frames used throughout the Buddy Jr RL lab:
the world frame, every link frame, the camera optical frame, and the
radians-to-servo-degrees mapping that bridges the simulator to real hardware.

---

## 1. REP-103 world frame

The project follows [ROS REP-103](https://www.ros.org/reps/rep-0103.html)
(the standard for robots using SI units and right-handed frames):

| Axis | Direction | Notes |
|---|---|---|
| +X | Forward | "Arm pointing forward" at q[0]=0 means the camera is in the +X half-plane |
| +Y | Left | Right-hand rule from +X forward and +Z up |
| +Z | Up | Vertical, aligned with gravity opposition |

The base of the robot sits flat on the ground at z = 0.  The world frame
origin is the centre of the base puck on the ground plane.

---

## 2. Per-link frames

Each link frame has its origin at the **parent joint's rotation axis**.
At the URDF neutral pose (all joint angles = 0 rad) every link frame is
aligned with the world frame (no rotation offsets in the URDF).

| Link frame | Origin in world (q=0) | Local +Z | Notes |
|---|---|---|---|
| `base_link` | (0, 0, 0) | Up | Fixed to the world; never moves |
| `shoulder_bracket` | (0, 0, 0.025) | Up | Rotates about world Z with `base_yaw` |
| `upper_arm` | (0, 0, 0.050) | Up (extends to 0.130) | Pitches about its own Y with `shoulder_pitch` |
| `forearm` | (0, 0, 0.130) | Up (extends to 0.210) | Pitches about its own Y with `elbow_pitch` |
| `camera_mount` | (0, 0, 0.210) | Up | Tilts about its own Y with `camera_tilt` |
| `camera_link` | (0.0145, 0, 0.220) | +X (lens forward) | TCP / end-effector; see §3 |
| `camera_optical_frame` | (0.0225, 0, 0.220) | +Z forward | Optical convention; see §4 |

The z-heights above are for q = 0.  When joints move, each child frame rotates
and the subsequent frames follow.

### 2.1 Visual diagram at q = 0

```
z (up)
^
|   0.220 m  camera_link  (+x = lens axis)
|   0.210 m  camera_mount
|   0.130 m  forearm (top of upper_arm = base of forearm)
|   0.050 m  upper_arm (top of shoulder_bracket = base of upper_arm)
|   0.025 m  shoulder_bracket
|   0.000 m  base_link
+---------------------------------> x (forward)
```

---

## 3. The camera_link frame (TCP)

`camera_link` is the **tool centre point (TCP)**: the point the RL reward
function and IK solver aim at.  It corresponds to the Pi camera lens barrel.

- **Origin**: 14.5 mm in front (+X) and 10 mm above (+Z) the `camera_mount`
  origin (the URDF `camera_joint` origin is `xyz="0.0145 0 0.010"`).
- **+X axis**: the optical boresight — the direction the lens looks.
- **+Z axis**: aligned with the parent `camera_mount` +Z at q[3]=0 (upward
  when the arm is in the neutral pose).

Because the fixed `camera_joint` has no rotation (`rpy="0 0 0"`),
`camera_link` and `camera_mount` share the same orientation; only the
origin is offset.

---

## 4. The camera_optical_frame

Camera projection math (intrinsics, image coordinates) conventionally uses a
different frame from the REP-103 robot frame:

| camera_optical_frame axis | Direction in robot-frame terms |
|---|---|
| +Z | Forward (lens boresight) = robot `camera_link` +X |
| +X | Right (image right) = robot `camera_link` −Y |
| +Y | Down (image down) = robot `camera_link` −Z |

The URDF fixed joint that creates this frame is:

```xml
<joint name="camera_optical_joint" type="fixed">
  <parent link="camera_link"/>
  <child  link="camera_optical_frame"/>
  <origin xyz="0.008 0 0" rpy="-1.5708 0 -1.5708"/>
</joint>
```

The two rotations (`rpy = [−π/2, 0, −π/2]`) implement the standard
`camera_link` → optical frame rotation.

For the RL experiments the optical frame is not used directly — all reward
calculations work in `camera_link` coordinates or world coordinates.  The
optical frame is there so that if you attach a live Pi Camera feed and want to
draw reprojected target markers, the projection matrix maths works out of the box.

---

## 5. Joint-angle to servo-degree mapping

The simulator works in **radians** (zero-centred, symmetric).  The real SG90
servos accept **degrees** on a 0–180 range, with 90° as the mechanical centre.

### 5.1 The formula

```
servo_deg = degrees(theta) + 90
```

Then clamp the result to [0, 180] before sending to the PCA9685.

In code (`rl_lab.robot.buddy_jr`):

```python
from rl_lab.robot.buddy_jr import radians_to_servo_degrees
import numpy as np

q_rad = np.array([0.0, 0.5, -0.3, 0.0])
servo_deg = radians_to_servo_degrees(q_rad)
# servo_deg is clamped to [0, 180]
```

### 5.2 Lookup table

| theta (rad) | theta (deg) | servo_deg | Servo position |
|---|---|---|---|
| −π/2 = −1.5708 | −90° | 0° | Full anticlockwise |
| −π/4 = −0.7854 | −45° | 45° | Quarter back |
| 0 | 0° | 90° | Mechanical centre |
| +π/4 = +0.7854 | +45° | 135° | Quarter forward |
| +π/2 = +1.5708 | +90° | 180° | Full clockwise |

### 5.3 Per-joint channel assignment

| q index | Joint name | PCA9685 channel |
|---|---|---|
| 0 | `base_yaw` | Ch 0 |
| 1 | `shoulder_pitch` | Ch 1 |
| 2 | `elbow_pitch` | Ch 2 |
| 3 | `camera_tilt` | Ch 3 |

### 5.4 Sign and direction

The URDF defines positive rotation using the **right-hand rule** about each
joint's axis:

- `base_yaw` rotates about +Z: **positive = anticlockwise viewed from above**
  (arm swings left).
- `shoulder_pitch`, `elbow_pitch`, `camera_tilt` rotate about +Y:
  **positive = tip moves forward/down** (right-hand rule about the Y axis
  pointing left).

If a servo turns the opposite way to the simulation, either flip the `<axis>`
sign in the URDF or set `sign=-1` in `ServoMap`:

```python
from rl_lab.robot.servo_map import ServoMap

# Shoulder physically mounted backwards:
sm = ServoMap(signs=(1, -1, 1, 1))
```

### 5.5 Centre-offset calibration

Real servos are rarely exactly centred at 90°.  Use `ServoMap(offsets_deg=...)`
to add a per-joint trim in degrees:

```python
sm = ServoMap(offsets_deg=(0, 0, 2.5, 0))  # elbow 2.5° off centre
servo_commands = sm.to_servo_degrees(q_rad)
```

Calibration can be saved and reloaded:

```python
sm.save("calibration.json")
sm2 = ServoMap.from_file("calibration.json")
```

---

## 6. Quick reference: all frames in one diagram

```
World frame (+X fwd, +Y left, +Z up)
|
+-- base_link          (z=0, fixed)
    |
    +-- [base_yaw, Z]  --> shoulder_bracket     (z=0.025)
        |
        +-- [shoulder_pitch, Y] --> upper_arm   (z=0.050)
            |
            +-- [elbow_pitch, Y] --> forearm    (z=0.130)
                |
                +-- [camera_tilt, Y] --> camera_mount  (z=0.210)
                    |
                    +-- [fixed, +x=0.0145, +z=0.010] --> camera_link (TCP)
                        |
                        +-- [fixed, rpy=-pi/2,0,-pi/2] --> camera_optical_frame
                                                          (+Z fwd, +X right, +Y down)
```
