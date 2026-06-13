# Buddy Jr URDF Reference

The robot model lives at `urdf/buddy_jr.urdf` and is the single source of geometric
truth for the simulation, the Foxglove visualiser, and the analytic kinematics.
Everything here is SI (metres, kilograms, radians).

---

## 1. Joint table

All four actuated joints are `revolute`.  The two fixed joints (`camera_joint`,
`camera_optical_joint`) attach the camera body and optical frame and carry no
actuator.

| # | Joint name | Type | Parent link | Child link | Axis (parent frame) | Origin xyz (m) | Origin rpy (rad) | Lower (rad) | Upper (rad) | Effort (N·m) | Velocity (rad/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `base_yaw` | revolute | `base_link` | `shoulder_bracket` | Z (0 0 1) | 0 0 0.025 | 0 0 0 | −1.5708 | +1.5708 | 1.5 | 6.0 |
| 1 | `shoulder_pitch` | revolute | `shoulder_bracket` | `upper_arm` | Y (0 1 0) | 0 0 0.025 | 0 0 0 | −1.5708 | +1.5708 | 1.5 | 6.0 |
| 2 | `elbow_pitch` | revolute | `upper_arm` | `forearm` | Y (0 1 0) | 0 0 0.080 | 0 0 0 | −1.5708 | +1.5708 | 1.2 | 6.0 |
| 3 | `camera_tilt` | revolute | `forearm` | `camera_mount` | Y (0 1 0) | 0 0 0.080 | 0 0 0 | −1.5708 | +1.5708 | 0.8 | 6.0 |
| — | `camera_joint` | fixed | `camera_mount` | `camera_link` | — | 0.0145 0 0.010 | 0 0 0 | — | — | — | — |
| — | `camera_optical_joint` | fixed | `camera_link` | `camera_optical_frame` | — | 0.008 0 0 | −1.5708 0 −1.5708 | — | — | — | — |

The joint limit of ±π/2 rad (±90°) spans the full 180° of an SG90 servo, with
the midpoint at 0 rad mapping to the servo's mechanical centre (90°).

---

## 2. Link table

| Link name | Geometry | Size | Mass (kg) | Notes |
|---|---|---|---|---|
| `base_link` | cylinder | r = 0.030 m, h = 0.025 m | 0.150 | Footprint puck, bolted to bench; inertia origin at z = 0.0125 m |
| `shoulder_bracket` | box | 0.040 × 0.040 × 0.025 m | 0.050 | Rotating turret carrying the shoulder servo; visual origin at z = 0.0125 m |
| `upper_arm` | box | 0.020 × 0.018 × 0.080 m | 0.040 | First 80 mm link (orange); extends along local +Z from the shoulder pivot; visual CoM at z = 0.040 m |
| `forearm` | box | 0.018 × 0.016 × 0.080 m | 0.030 | Second 80 mm link (blue); extends along local +Z from the elbow pivot; visual CoM at z = 0.040 m |
| `camera_mount` | box | 0.025 × 0.025 × 0.020 m | 0.015 | Small holder for the Pi camera module (green); visual CoM at z = 0.010 m |
| `camera_link` | cylinder | r = 0.005 m, h = 0.008 m | 0.003 | Lens barrel, oriented along +X; the TCP / end-effector used by the RL reward |
| `camera_optical_frame` | — (massless) | — | — | Standard optical frame (+Z forward, +X right, +Y down); used for image-projection maths only |

All visual geometries grow along the link's own local +Z from the parent joint
origin.  This mirrors how printed parts attach at a servo horn: the pivot is at
the link's origin, the mass hangs "above" it in local +Z.

---

## 3. Inertia notes

Inertia values use the analytic formulas for uniform solids:

- **Cylinder** (`base_link`): I_xx = I_yy = m(3r² + h²)/12, I_zz = mr²/2.
- **Box** (all arm links): I_xx = m(b² + c²)/12, cyclically.

All tensors are diagonal (off-diagonal elements are zero), positive-definite,
and satisfy the principal-moment triangle inequality required by PyBullet.

---

## 4. How to validate the URDF

The repo ships a standalone validator at `scripts/validate_urdf.py` that checks
five things without needing a full ROS or physics install:

1. Well-formed XML.
2. Single root link (`base_link`); acyclic joint tree.
3. Each revolute joint has a unit-norm axis, lower < upper, effort > 0, velocity > 0.
4. Every `<inertial>` block is positive-definite and satisfies the triangle
   inequality.
5. Optional PyBullet DIRECT-mode load (skipped gracefully if not installed).

```bash
# From the repo root:
python scripts/validate_urdf.py              # validates urdf/buddy_jr.urdf
python scripts/validate_urdf.py path/to.urdf # validate a custom file
```

Exit code is 0 on a full pass.  Run this any time you edit the URDF.

---

## 5. Loading in PyBullet

```python
import pybullet as p

cid = p.connect(p.DIRECT)                          # headless, or p.GUI for a window
robot_id = p.loadURDF("urdf/buddy_jr.urdf", useFixedBase=True, physicsClientId=cid)
```

The programmatic path is available via the `rl_lab` package:

```python
from rl_lab.robot.buddy_jr import urdf_path
# urdf_path() returns a pathlib.Path pointing to the packaged file
p.loadURDF(str(urdf_path()), useFixedBase=True)
```

---

## 6. Swapping in real STL meshes

The URDF currently uses only `<box>` and `<cylinder>` primitives.  They load
without any external files and render correctly in Foxglove and rviz today.
When you have 3D-printed parts and want to show the real geometry, follow
these steps:

### 6.1 Export from your CAD tool

- Export each part as an STL **in millimetres** (the Fusion 360 / Tinkercad
  default).
- Place the files in `urdf/meshes/` using the naming convention
  `<link_name>.stl`, for example `upper_arm.stl`.

### 6.2 Replace `<geometry>` in the visual element

For each link, swap the `<box .../>` inside `<visual>` with:

```xml
<geometry>
  <mesh filename="package://buddy_jr_description/meshes/upper_arm.stl"
        scale="0.001 0.001 0.001"/>
</geometry>
```

The `scale="0.001 0.001 0.001"` converts from millimetres to metres.

### 6.3 Adjust the visual origin

STL mesh origins are rarely at the joint rotation axis.  Keep the `<joint>`
`<origin>` values exactly as they are (they are the kinematic ground truth)
and adjust only the `<visual><origin>` to slide the mesh so the part's
physical pivot aligns with the joint's coordinate frame.

### 6.4 Keep primitive collision shapes

Leave the `<collision>` elements as boxes/cylinders.  Full-resolution mesh
collision is slow in PyBullet and can destabilise contact detection during RL
training.  If you need mesh-quality collision, use a convex-hull-decomposed
version (e.g. from VHACD).

### 6.5 Mass and inertia

The current mass estimates (base 150 g, upper_arm 40 g, forearm 30 g, etc.)
are close to typical PLA print weights.  For higher dynamic fidelity, replace
them with the mass reported by your slicer and the inertia tensor from CAD.

### 6.6 Validate again

```bash
python scripts/validate_urdf.py
```

This confirms the new mesh references do not break the tree structure, limits,
or inertia correctness.

---

## 7. Regenerating the URDF from scratch

The current URDF was written by hand to match the Buddy Jr hardware description
at [kevsrobots.com/blog/buddy_jr.html](https://www.kevsrobots.com/blog/buddy_jr.html).
If you redesign the arm (different link lengths, additional joints, etc.) the
recommended regeneration approach is:

1. **URDF directly** — for small changes (link length, mass), edit `urdf/buddy_jr.urdf`
   in a text editor.  The file is intentionally readable.
2. **Xacro** — for a parameterised design, create `urdf/buddy_jr.urdf.xacro` and
   generate the URDF with `xacro buddy_jr.urdf.xacro > buddy_jr.urdf` (requires
   the `xacro` package from `ros-{distro}-xacro` or `pip install xacro`).
3. **SOLIDWORKS / Fusion 360 exporter** — the SW2URDF plugin or the Fusion
   URDF exporter can export a full assembly.  Always run `validate_urdf.py`
   on the result: exporters often produce invalid inertia tensors or missing limits.
4. **Update `rl_lab/robot/buddy_jr.py`** — the constants `SHOULDER_LENGTH`,
   `ELBOW_LENGTH`, `BASE_HEIGHT`, and `CAMERA_OFFSET` must stay in sync with the
   URDF joint origins.  The kinematics module reads from those constants.
