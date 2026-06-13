# Buddy Jr URDF

FILE: `urdf/buddy_jr.urdf`

Primitive-only URDF (boxes and cylinders, no external mesh files) for the
Buddy Jr 4-DOF robot arm.  Loads in PyBullet, MuJoCo, rviz, and Foxglove
without any additional dependencies.

Full documentation: see [`docs/robot/urdf.md`](../docs/robot/urdf.md),
[`docs/robot/kinematics.md`](../docs/robot/kinematics.md), and
[`docs/robot/frames.md`](../docs/robot/frames.md).

---

## Validation

```bash
python scripts/validate_urdf.py        # checks XML, joint limits, inertias, optional PyBullet load
```

Exit code 0 = all checks passed.  Run after every edit.

---

## Joint table

| # | Joint | Parent | Child | Axis | Origin (m) | Limits (rad) | Effort (N·m) |
|---|---|---|---|---|---|---|---|
| 0 | `base_yaw` | `base_link` | `shoulder_bracket` | Z | 0 0 0.025 | ±1.5708 | 1.5 |
| 1 | `shoulder_pitch` | `shoulder_bracket` | `upper_arm` | Y | 0 0 0.025 | ±1.5708 | 1.5 |
| 2 | `elbow_pitch` | `upper_arm` | `forearm` | Y | 0 0 0.080 | ±1.5708 | 1.2 |
| 3 | `camera_tilt` | `forearm` | `camera_mount` | Y | 0 0 0.080 | ±1.5708 | 0.8 |
| — | `camera_joint` (fixed) | `camera_mount` | `camera_link` | — | 0.0145 0 0.010 | — | — |
| — | `camera_optical_joint` (fixed) | `camera_link` | `camera_optical_frame` | — | 0.008 0 0, rpy −π/2 0 −π/2 | — | — |

All four revolute joints span ±π/2 rad (0–180° of SG90 travel).
`camera_link` is the TCP (end-effector) used by the RL reward and IK solver.

---

## Frames summary (REP-103: +X fwd, +Y left, +Z up)

```
World (z=0)
  └─ base_link            z=0.000  fixed
       └─ [base_yaw, Z]   z=0.025
            └─ [shoulder_pitch, Y]  z=0.050   upper_arm extends to z=0.130
                 └─ [elbow_pitch, Y]  z=0.130  forearm extends to z=0.210
                      └─ [camera_tilt, Y]  z=0.210  camera_mount
                           └─ camera_link  (TCP, x=0.0145, z=0.220)
                                └─ camera_optical_frame  (+Z fwd optical)
```

At q = [0, 0, 0, 0] the arm points straight up; the camera tip is at
(x=0.0145, y=0, z=0.220) m.

---

## Radians to servo degrees

```
servo_deg = degrees(theta) + 90    clamped to [0, 180]
```

| theta (rad) | servo_deg | Position |
|---|---|---|
| −π/2 | 0° | Full anticlockwise |
| 0 | 90° | Centre |
| +π/2 | 180° | Full clockwise |

Joint order maps directly to PCA9685 channels 0–3:
`base_yaw` → Ch 0, `shoulder_pitch` → Ch 1, `elbow_pitch` → Ch 2, `camera_tilt` → Ch 3.

API: `rl_lab.robot.buddy_jr.radians_to_servo_degrees(q)` and `servo_degrees_to_radians(deg)`.
For per-joint sign / trim calibration, use `rl_lab.robot.servo_map.ServoMap`.

---

## Loading in PyBullet

```python
import pybullet as p
from rl_lab.robot.buddy_jr import urdf_path

cid = p.connect(p.DIRECT)
p.loadURDF(str(urdf_path()), useFixedBase=True, physicsClientId=cid)
```

---

## Swapping in STL meshes later

The primitives are intentionally placed so each link's geometry grows from its
parent joint origin, mirroring how a printed part attaches at its servo horn.
To switch to real prints:

1. Export each part as STL in **millimetres** into `urdf/meshes/`.
2. In each `<visual><geometry>`, replace `<box .../>` with:
   ```xml
   <mesh filename="package://buddy_jr_description/meshes/<part>.stl"
         scale="0.001 0.001 0.001"/>
   ```
3. Adjust the `<visual><origin>` (not the `<joint><origin>`) so the mesh
   pivot aligns with the joint frame.
4. Keep the `<collision>` shapes as boxes/cylinders for simulation speed.
5. Run `python scripts/validate_urdf.py` to confirm nothing broke.

See [`docs/robot/urdf.md`](../docs/robot/urdf.md) §6 for the full procedure.

---

## Regenerating

For small edits (link lengths, masses) edit the URDF directly — it is plain XML.
For a fully parameterised rebuild, convert to Xacro:

```bash
xacro urdf/buddy_jr.urdf.xacro > urdf/buddy_jr.urdf
python scripts/validate_urdf.py
```

If you change `SHOULDER_LENGTH`, `ELBOW_LENGTH`, `BASE_HEIGHT`, or
`CAMERA_OFFSET`, update the matching constants in `rl_lab/robot/buddy_jr.py`
as well — the analytic kinematics read from those values, not from the URDF XML.
