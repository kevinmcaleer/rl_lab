# Experiment 02 — Build the World

**Concept:** The environment is a first-class object.

Before any agent can learn anything you must be able to:

1. Define what actions do (move each joint by a commanded delta).
2. Observe what those actions cause (the new end-effector position, the
   distance to the target).
3. Verify that the simulation's physics agrees with the underlying maths (the
   analytic forward kinematics).

This experiment does all three — with no RL algorithm involved at all.

---

## What you will see

When you run with `--render foxglove` you watch the Buddy Jr arm sweep each
joint from its minimum angle (−90°) to its maximum (+90°) and back, while a
coloured target sphere sits at a randomly sampled, always-reachable location.
The arm is driven by a pure scripted policy (no learning), so the motion is
smooth and predictable.

In the terminal you see a line printed for every joint:

```
joint 0: base_yaw    (−90° → +90°)
joint 1: shoulder_pitch  (−90° → +90°)
joint 2: elbow_pitch     (−90° → +90°)
joint 3: camera_tilt     (−90° → +90°)
```

After all four sweeps a summary shows the worst FK residual (tip-position
error between the analytic kinematics and what the environment reports):

```
[02_world] Done.  steps=240  max FK residual=0.0001 mm  (tolerance=0.5 mm)  FK checks PASSED
```

If your installation is healthy that residual is well under 0.5 mm for every
single step — the analytic model and the simulation backend agree.

A plot is saved to `experiments/_outputs/02_world/02_world_sweep.png` showing:

- **Top row** — joint angle over sweep steps for all four joints.
- **Bottom row** — FK residual in millimetres per sweep step, with a red
  dashed tolerance line at 0.5 mm.

---

## How to run

### Minimal (terminal only, no viewer)

```bash
python experiments/02_world.py
```

### With live 3-D visualisation (Foxglove)

Start the Foxglove desktop app, create an open connection to
`ws://localhost:8765`, and then:

```bash
python experiments/02_world.py --render foxglove
```

The script starts its own WebSocket server and prints a `foxglove://` URL you
can click to open the app directly onto the right connection.

For Foxglove setup details see
[`docs/getting_started/foxglove_setup.md`](../getting_started/foxglove_setup.md).

### Quick smoke-test (CI / low-resource machine)

```bash
python experiments/02_world.py --quick
```

`--quick` uses a tiny step budget and skips plotting and Foxglove, finishing
in a few seconds on CPU.

### Full option reference

```
usage: 02_world [--quick] [--render {foxglove}] [--seed SEED] [--no-plot]

  --quick            Tiny step budget, no plots, no Foxglove (CI mode).
  --render foxglove  Stream live to Foxglove.
  --seed SEED        RNG seed for the environment (default: 0).
  --no-plot          Skip saving the matplotlib figure.
```

---

## What to look for

### In the 3-D viewer (Foxglove)

| What you see | What it means |
|---|---|
| Each arm segment swings independently in sequence | Each joint controls exactly one degree of freedom — the URDF, the FK chain, and the sim backend all agree. |
| The target sphere stays fixed while the arm moves | `reset()` samples a reachable target and holds it until the next `reset()`. |
| Distance metric drops when the tip passes near the sphere | The `info['distance']` field is the Euclidean tip-to-target distance in metres. |

### In the terminal output

Watch that **max FK residual** stays well below the 0.5 mm tolerance.  If it
is zero (or sub-nanometre) the kinematic backend is purely analytic —
no physics engine is involved — so the FK is exact by construction.  If you
swap in the PyBullet or MuJoCo backend the residual may be a few
micrometres due to floating-point differences; it should never exceed 1 mm.

### In the saved plot

- **Joint angle (top row):** should be a smooth triangle wave sweeping from
  −90° to +90° and back.  Any clipping at the limits means the scripted
  action saturated at ±1 for a few steps — expected and harmless.
- **FK residual (bottom row):** should be an essentially flat line at or
  below the red tolerance marker.

---

## The FK sanity check explained

At every step the script calls:

```python
fk_pos = rl_lab.robot.kinematics.forward(info["joint_q"]).position
env_pos = info["ee_pos"]
residual = np.linalg.norm(fk_pos - env_pos)
assert residual < 0.001  # 1 mm hard limit
```

`kinematics.forward` is a standalone, physics-engine-free implementation of
the joint transform chain from the URDF.  `info["ee_pos"]` is whatever the
environment's backend (kinematic, PyBullet, or MuJoCo) reports as the camera
tip.

The fact that these agree to sub-millimetre precision tells you three things:

1. The URDF joint origins are self-consistent.
2. The FK implementation in `rl_lab/robot/kinematics.py` faithfully mirrors
   those joint origins.
3. The environment backend computes the same camera-tip position.

Every later experiment uses `info["ee_pos"]` and `info["joint_q"]` in
observations and rewards.  This check gives you the right to trust them.

---

## The Gymnasium contract

This experiment is a good place to internalise the three-function Gymnasium
loop that every experiment in the lab uses:

```python
import gymnasium as gym
import rl_lab  # registers BuddyJrReach-v0

env = gym.make("BuddyJrReach-v0")

obs, info = env.reset(seed=0)
# obs  — shape (17,) float32 observation vector
# info — dict with {distance, is_success, joint_q, ee_pos, target}

for _ in range(200):
    action = env.action_space.sample()         # random 4-D continuous action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

Key points:

- `action` is a shape-(4,) vector in `[-1, 1]`, one delta per joint.  The
  env scales each delta by 0.1 rad before applying it.
- `terminated` means the tip reached the target (within 2 cm); `truncated`
  means the episode time limit was hit.  They are returned separately and
  should never be conflated.
- `info['joint_q']` is the actual joint state **after** the step, not the
  commanded target.  In the pure-kinematic backend they are the same; in a
  physics backend they may differ slightly due to dynamics.
- `obs[14:17]` is the tip→target vector (already included in the observation
  so an agent can see how far it is from the goal without reading `info`).

---

## Aha takeaway

> *RL doesn't start with an algorithm — it starts with a clean environment
> definition you can trust and observe.*

After this experiment you know:

- How to create and step through a Gymnasium environment.
- What each joint on Buddy Jr physically controls.
- That the FK implementation and the simulation agree — you have a testable
  ground truth.
- How to drive the arm with scripted actions rather than a learned policy
  (invaluable for debugging later).

**Now you understand the agent–environment loop (`reset`/`step`/`reward`) and
why a faithful sim plus a live viewer is non-negotiable before writing a
single line of RL.**

---

## Files involved

| Path | Role |
|---|---|
| `experiments/02_world.py` | The runnable experiment (this script). |
| `rl_lab/env/buddy_jr_reach_env.py` | `BuddyJrReachEnv` — the Gymnasium environment. |
| `rl_lab/robot/kinematics.py` | Analytic forward kinematics (`forward(q)`). |
| `rl_lab/robot/buddy_jr.py` | Joint constants, limits, servo mapping. |
| `rl_lab/viz/foxglove_bridge.py` | `FoxgloveStreamer` — live WebSocket publisher. |
| `urdf/buddy_jr.urdf` | The URDF model the environment and FK chain implement. |
| `experiments/_outputs/02_world/` | Output directory for saved plots. |

---

## What's next

- **Experiment 03** introduces the first RL algorithm: tabular Q-learning on
  a discretised version of this same environment.  Because you've seen the raw
  environment here, you'll be able to interpret the Q-table values in terms of
  real joint angles and distances.
