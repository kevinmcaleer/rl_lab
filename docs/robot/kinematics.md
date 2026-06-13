# Buddy Jr Kinematics

This page covers the arm's geometry, the forward-kinematics (FK) transform chain,
the law-of-cosines inverse-kinematics (IK) solver, reachability limits, and the
`rl_lab.robot.kinematics` API that the RL environments use internally.

All lengths are in metres, all angles in radians.

---

## 1. Arm geometry

```
                    camera_link  (end-effector / TCP)
                       |
              [camera_tilt joint, Y-axis]
                       |
                    forearm      80 mm
                       |
              [elbow_pitch joint, Y-axis]
                       |
                   upper_arm    80 mm
                       |
              [shoulder_pitch joint, Y-axis]
                       |
                shoulder_bracket  (height 25 mm)
                       |
              [base_yaw joint, Z-axis]
                       |
                   base_link    (height 25 mm)
                       |
                ground plane (z = 0)
```

Key dimensions (from `rl_lab.robot.buddy_jr`):

| Constant | Value | Meaning |
|---|---|---|
| `SHOULDER_LENGTH` | 0.080 m | Upper-arm link length |
| `ELBOW_LENGTH` | 0.080 m | Forearm link length |
| `BASE_HEIGHT` | 0.050 m | Height of the shoulder-pitch axis above ground (0.025 + 0.025) |
| `CAMERA_OFFSET` | (0.0145, 0, 0.010) m | Fixed translation from camera_mount frame to camera_link |
| `JOINT_LIMIT` | π/2 rad | Each revolute joint spans [−π/2, +π/2] |

---

## 2. Forward kinematics (FK)

FK computes the world position and orientation of the camera tip (`camera_link`)
given a joint angle vector `q = [base_yaw, shoulder_pitch, elbow_pitch, camera_tilt]`.

### 2.1 Transform chain

Each step in the chain is a 4 × 4 homogeneous transform:

```
T_world_camera = T0 · T1 · T2 · T3 · T4
```

| Step | Parent | Child | Translation (m) | Rotation axis | q index |
|---|---|---|---|---|---|
| T0 | `base_link` | `shoulder_bracket` | (0, 0, 0.025) | Z | q[0] base_yaw |
| T1 | `shoulder_bracket` | `upper_arm` | (0, 0, 0.025) | Y | q[1] shoulder_pitch |
| T2 | `upper_arm` | `forearm` | (0, 0, 0.080) | Y | q[2] elbow_pitch |
| T3 | `forearm` | `camera_mount` | (0, 0, 0.080) | Y | q[3] camera_tilt |
| T4 | `camera_mount` | `camera_link` | (0.0145, 0, 0.010) | — (fixed) | — |

Each transform Ti = Trans(d) · Rot(axis, q[i]), where Trans(d) is the URDF
`<joint><origin>` translation and Rot is a rotation about the stated axis.

### 2.2 Zero-configuration pose

At q = [0, 0, 0, 0] the arm points straight up.  Substituting into the chain:

```
z = 0.025 + 0.025 + 0.080 + 0.080 + 0.010 = 0.220 m
x = CAMERA_OFFSET[0]                        = 0.0145 m
y = 0
```

This is verified by `tests/test_kinematics.py::test_forward_zero_config_matches_urdf_geometry`.

### 2.3 Python usage

```python
import numpy as np
from rl_lab.robot import kinematics as kin

q = np.array([0.0, 0.5, -0.3, 0.0])   # radians
pose = kin.forward(q)
print(pose.position)       # (x, y, z) in metres
print(pose.orientation)    # quaternion [x, y, z, w]

# Position only (slightly faster)
xyz = kin.forward_position(q)
```

---

## 3. Inverse kinematics (IK)

The IK solver finds joint angles that place the camera tip at a desired world
position.  It uses the classical law-of-cosines on the planar 2-link sub-arm.

### 3.1 Decomposition

The problem decomposes into two independent parts:

1. **Azimuth (base_yaw)**: the target (x, y, z) implies an azimuth angle
   `θ₀ = atan2(y, x)`.  The horizontal reach from the base is `r = hypot(x, y)`.
2. **Planar 2R arm (shoulder + elbow)**: once the arm plane is chosen, the
   shoulder and elbow solve the standard 2-link problem in that vertical plane.

### 3.2 Camera-tilt handling

`camera_tilt` (q[3]) is a free DOF for aiming.  The solver holds it fixed at a
caller-supplied value (default 0 rad, i.e. horizontal).  With tilt fixed, the
elbow→tip segment is rigid.  Its effective length `L2` and angle offset `φ`
are computed from `ELBOW_LENGTH` and `CAMERA_OFFSET`:

```python
vx = ox * cos(tilt) + oz * sin(tilt)
vz = ELBOW_LENGTH - ox * sin(tilt) + oz * cos(tilt)
L2, phi = hypot(vx, vz), atan2(vx, vz)
```

where `(ox, _, oz)` = `CAMERA_OFFSET`.

### 3.3 Law-of-cosines solution

With shoulder–elbow link `L1 = SHOULDER_LENGTH` and the effective wrist segment
`L2` (as above), the distance from shoulder axis to target in the arm plane is:

```
d  = hypot(r_horizontal, z - BASE_HEIGHT)
```

The law of cosines gives the elbow bend angle γ:

```
cos γ = (d² − L1² − L2²) / (2 · L1 · L2)
γ     = ±acos(cos γ)   ← two elbow configurations
```

The shoulder angle is then:

```
β  = atan2(L2 · sin γ,  L1 + L2 · cos γ)
α  = atan2(r_horizontal, z − BASE_HEIGHT)    ← in-plane angle to target
q1 = α − β
q2 = γ − φ                                   ← subtract the tilt offset
```

Both elbow configurations (+γ and −γ) are tried.  The solver also attempts a
"folded" yaw (arm facing away from the target, yaw ± π) to reach points that
would otherwise require yaw beyond ±π/2.  The first solution that satisfies all
joint limits is returned.

### 3.4 Reachable workspace

The arm spans a **spherical shell** centred on the shoulder-pitch axis
(x=0, y=0, z=0.05 m):

| Quantity | Value |
|---|---|
| Maximum reach (both links fully extended) | L1 + L2 = 160 mm |
| Minimum reach (links folded back on themselves) | |L1 − L2| = 0 mm |
| Horizontal radius at shoulder height | up to 160 mm |
| Maximum height above ground | 0.050 + 0.160 = 210 mm |
| Base yaw range | ±π/2 rad (±90°) |

Because `L1 = L2`, the inner-reach minimum is theoretically 0 mm, but in
practice the elbow joint cannot reach exactly −π/2 to fold the arm fully back
on itself without hitting the joint limit; the solver catches this and raises
`UnreachableError(reason="joint_limits")`.

### 3.5 Python usage

```python
import numpy as np
from rl_lab.robot import kinematics as kin

target = np.array([0.08, 0.0, 0.15])   # metres, world frame

# Default: camera held horizontal (tilt = 0)
q = kin.inverse(target)

# With a specific camera tilt angle
q = kin.inverse(target, camera_tilt=0.3)

# Check without raising
if kin.is_reachable(target):
    q = kin.inverse(target)
```

If the target cannot be reached, `kin.inverse()` raises `UnreachableError`
with a `reason` attribute:

```python
from rl_lab.robot.kinematics import UnreachableError

try:
    q = kin.inverse(far_away)
except UnreachableError as e:
    print(e.reason)   # "out_of_reach" or "joint_limits"
```

---

## 4. `rl_lab.robot.kinematics` API

```python
from rl_lab.robot import kinematics as kin
```

### 4.1 Return types

#### `Pose` (dataclass, frozen)

| Field | Type | Description |
|---|---|---|
| `position` | `np.ndarray` shape (3,) | World position (x, y, z) in metres |
| `orientation` | `np.ndarray` shape (4,) | Quaternion [x, y, z, w] |

#### `UnreachableError` (ValueError subclass)

| Attribute | Type | Values |
|---|---|---|
| `reason` | `str` | `"out_of_reach"` or `"joint_limits"` |

### 4.2 Functions

| Function | Signature | Returns | Notes |
|---|---|---|---|
| `forward` | `(q: ndarray) -> Pose` | `Pose` | Full FK: joint angles → camera-tip pose |
| `forward_position` | `(q: ndarray) -> ndarray` | shape (3,) | Position-only convenience wrapper |
| `inverse` | `(target: ndarray, camera_tilt: float = 0.0) -> ndarray` | shape (4,) | Law-of-cosines IK; raises `UnreachableError` |
| `is_reachable` | `(target: ndarray, camera_tilt: float = 0.0) -> bool` | `bool` | Returns `True` if `inverse` would succeed |
| `joint_transforms` | `(q: ndarray) -> list[tuple]` | list of (parent, child, translation, quaternion) | Local frame transforms; used by the Foxglove publisher |
| `link_world_poses` | `(q: ndarray) -> dict[str, Pose]` | dict | World pose of every link; handy for debugging |

---

## 5. Analytic solver vs. learned policy

Experiment 10 (`experiments/10_sac_aim.py`) puts this analytic IK solver
head-to-head with a SAC policy trained to reach the same targets.  That
comparison is a core learning objective: understanding when a closed-form
mathematical solution is better than a learned one, and when it is not (e.g.
obstacle avoidance, contact-rich tasks).
