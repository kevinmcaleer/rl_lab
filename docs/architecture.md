# Buddy Jr RL Lab — Architecture Overview

This is the *map* of the lab. If [`PLAN.md`](PLAN.md) is the **why** (the design
decisions and the milestone roadmap) and [`experiments/README.md`](../experiments/README.md)
is the **what you do** (the 12-step learning ladder), this document is the
**how the pieces fit together** — the layers, who owns what, and exactly what
happens, in order, during a single training step.

Read this once before you start hacking on the code. It will save you from
wondering "where does the reward live?" or "why is the renderer a separate
thing from the physics?" — questions whose answers *are* the architecture.

> New to the project? Skim [`PLAN.md`](PLAN.md) §1–2 first (the stack decision),
> then come back here for the structure, then dive into
> [`experiments/README.md`](../experiments/README.md) to actually learn RL.

---

## 1. The big idea: clean layers, swappable parts

The lab is built as a stack of **six layers**. Each layer talks only to the
layer directly above and below it through a narrow, well-defined boundary. That
is not architectural fussiness — it is the single most important *teaching*
device in the whole project:

- **You can swap one layer without touching the others.** Change the physics
  engine (PyBullet → MuJoCo) and the RL code, the URDF, and the viewer never
  notice. Change the algorithm (Q-learning → PPO) and the environment, physics,
  and viewer never notice. *Watching* a swap work is itself a lesson (see §3
  and §4).
- **Each layer maps to exactly one idea you are learning.** The environment
  layer *is* the agent–environment loop. The algorithm layer *is* the learning
  rule. The viz layer *is* "the policy doesn't know what renders it." Keeping
  them physically separate in the code keeps them separate in your head.

```
  +-------------------------------------------------------------+
  |  experiments/   the 12-step curriculum (runnable lessons)   |   <- you start here
  |  one script per lesson; wires the layers below together     |
  +-------------------------------------------------------------+
        |  imports & configures everything beneath
        v
  +----------------------+         +----------------------------+
  |  algos               |         |  viz                       |
  |  rl_lab/algos/       |         |  rl_lab/viz/               |
  |  HOW to learn        | <-----> |  HOW to SEE it learn       |
  |  (Q-learn, DQN, PPO, |  obs/   |  (Foxglove bridge, scene   |
  |   SAC, SB3 wrappers) |  action |   + transforms + metrics)  |
  +----------------------+         +----------------------------+
        |  reset()/step()                ^  reads state to publish
        v                                |
  +-------------------------------------------------------------+
  |  env            rl_lab/env/                                 |
  |  the MDP: observation, action, reward, reset, termination   |
  |  (Gymnasium API — BuddyJrReachEnv + wrappers + registration) |
  +-------------------------------------------------------------+
        |  "apply this action / give me joint & pose state"
        v
  +-------------------------------------------------------------+
  |  sim            rl_lab/sim/                                 |
  |  physics behind a SimBackend interface                      |
  |  (PyBullet = default, MuJoCo = optional, same interface)    |
  +-------------------------------------------------------------+
        |  loads geometry, joints, limits, inertias FROM ...
        v
  +-------------------------------------------------------------+
  |  urdf           urdf/buddy_jr.urdf  (+ rl_lab/robot/)       |
  |  the SINGLE SOURCE OF TRUTH for the robot's shape & joints   |
  +-------------------------------------------------------------+
```

### Layer-by-layer responsibilities

| Layer | Owns (package) | Responsibility | Knows nothing about |
|-------|----------------|----------------|---------------------|
| **urdf** | `urdf/` + `rl_lab/robot/` | The robot definition: 4 revolute joints, two 80 mm links, the fixed camera tip, joint axes & limits, inertias. `rl_lab/robot/` adds the Python view of it — joint table, forward/inverse kinematics, and the **servo mapping** (`servo_deg = degrees(theta) + 90`, clamped `[0, 180]`). | RL, rewards, rendering |
| **sim** | `rl_lab/sim/` | Stepping physics. A `SimBackend` interface (`base.py`) with concrete `pybullet_sim.py` (default) and `mujoco_sim.py` (optional). Loads the URDF, applies joint targets, advances time, and reports joint angles + link world poses (forward kinematics). | What the action *means*, what a reward is, who is watching |
| **env** | `rl_lab/env/` | The **MDP**. Defines the observation space, action space, reward (`rewards.py`), `reset()`/`step()`, and the strict `terminated` vs `truncated` split. `BuddyJrReachEnv` is canonical; discrete/tabular variants are thin `wrappers.py`. `registration.py` exposes the `BuddyJrReach-v0` ids. This is the only layer that turns physics into an RL problem. | Which algorithm is learning; how it is rendered |
| **algos** | `rl_lab/algos/` | The **learning rule**. From-scratch teaching implementations (`tabular/`, `value_based/`, `policy_gradient/`) plus `sb3_integration.py` for Stable-Baselines3 (PPO/SAC/DQN/TD3). Consumes only `(obs, reward, done)` through the Gymnasium API. | Physics internals; how the arm is drawn |
| **viz** | `rl_lab/viz/` | Making learning **visible**. `foxglove_bridge.py` publishes the transform tree + scene markers + live metric channels; `schemas.py` defines the message shapes; `urdf_publisher.py` maps URDF links to frames; `live_metrics.py` feeds the plot panels. A `rviz/` subfolder holds the optional ROS2 path. Read-only: it observes state, it never changes it. | The learning rule; the reward formula |
| **experiments** | `experiments/` | The **curriculum**. One runnable, documented script per lesson (01..12) that *wires the layers together* for a specific teaching goal and tells you what to watch for. | Implementation details it doesn't need — it composes, it doesn't reimplement |

The golden rule: **dependencies point downward only.** `env` may call `sim`;
`sim` must never import `env`. `viz` reads state but never writes it. If you
ever find yourself wanting an upward import, the responsibility is in the wrong
layer.

---

## 2. Runtime data flow: one training step, end to end

Here is what actually happens, in order, when an algorithm takes a single step.
Follow the arrows top to bottom; this is the path every one of the 12
experiments runs thousands of times per second.

```
   [ algos ]  agent picks an action a_t from obs_t
      |        (4-vector of joint deltas, or a discrete jog)
      |  env.step(a_t)
      v
   [ env ]   BuddyJrReachEnv.step()
      |   1. map a_t -> POSITION_CONTROL joint targets, clamp to URDF limits
      |   2. ask the backend to advance physics
      |  backend.set_joint_targets(...) ; backend.step(substeps)
      v
   [ sim ]   SimBackend (PyBullet by default)
      |   steps the dynamics a few substeps for stability,
      |   then reports back joint angles q and the camera-tip world pose
      |  returns joint_q, ee_pose  (forward kinematics)
      v
   [ env ]   back in step():
      |   3. compute distance(tip, target) and the reward
      |   4. build obs_{t+1} (sin/cos of joints, tip, target, vector-to-goal)
      |   5. set terminated (success) / truncated (time limit) — kept separate
      |   returns (obs_{t+1}, reward, terminated, truncated, info)
      |   info always carries {distance, is_success, joint_q, ee_pos, target}
      |
      +--------> back to [ algos ]: store transition, update the policy
      |
      |  WHEN render_mode="foxglove", the SAME state fans out to viz
      v
   [ viz ]   FoxgloveBridge.publish(state)   (throttled to ~30 Hz)
      |   - one FrameTransform per joint (parent->child from current FK)
      |       so the URDF-shaped arm actually moves
      |   - a SceneUpdate: target sphere (green within tolerance),
      |       tip marker, and a tip->target line = the distance reward minimises
      |   - numeric channels: distance, reward, episode_return, success_rate
      v
   [ foxglove-sdk ]   one process, two sinks:
      |
      |--->  live WebSocket  ws://127.0.0.1:8765  --->  Foxglove desktop app
      |          (3D panel composes the transform tree + pins link
      |           geometry to each frame; plot panels show the metrics)
      |
      '--->  optional .mcap recording on disk
                 (same channels; replay & scrub later through the SAME layout)
```

Key things to notice in this flow — they recur all over the curriculum:

- **The agent only ever sees `(obs, reward, terminated, truncated, info)`.** It
  has no handle on PyBullet, no idea a renderer exists. That narrow interface is
  why you can drop in any algorithm in `rl_lab/algos/`.
- **Visualization is a read-only tap on the same state.** The bridge publishes
  what physics already produced; it never feeds back into learning. Turning the
  viewer off changes nothing about training except speed — proof that the
  policy doesn't know what renders it.
- **Joint angles → transform tree → geometry** is the universal rendering recipe.
  Foxglove receives a `FrameTransform` per joint and hangs each URDF link's
  shape off the matching frame. Same recipe, different sink, drives the ROS2
  path (§4) and, in Experiment 12, the **real** robot.
- **MCAP is the same data, written instead of streamed.** Anything you can
  watch live you can record and scrub later, which is how you debug a training
  run after it finished.

---

## 3. Why PyBullet is the default and MuJoCo is the swap

Both are excellent physics engines. The lab puts a thin `SimBackend` interface
(`rl_lab/sim/base.py`) in front of *both* so neither is hard-wired, and makes
**PyBullet the default**:

- **It mirrors the robot's own maths.** Buddy Jr's blog firmware aims the camera
  with hand-derived **law-of-cosines inverse kinematics**. PyBullet ships a
  native URDF loader and a built-in `calculateInverseKinematics`, so a lesson can
  put classical IK and a learned policy side by side with no extra tooling — the
  most pedagogically relevant path for makers porting to real hardware.
- **It is one `pip install` with an instant local look.** PyBullet has an
  optional built-in GUI window, so a beginner can *see the arm move before any RL
  exists* (Experiment 2) with zero viewer setup.

**MuJoCo stays first-class but optional** (the `[mujoco]` extra), behind the
exact same `SimBackend` interface. Its strengths — the cleanest arm64 wheels and
higher actuator fidelity — are real and worth keeping. Crucially:

> **Swapping the backend is itself an experiment.** Train a policy in PyBullet,
> then flip a config flag to run it in MuJoCo. If it still works, you have just
> done a **sim-to-sim transfer** — a free, hardware-safe rehearsal of the
> sim-to-real gap you tackle in Experiments 11–12. The disagreement between
> "which engine?" became a teaching moment instead of a fork.

The one Apple-Silicon caveat (an occasional x86_64-under-Rosetta PyBullet wheel)
is handled by a documented `platform.machine() == "arm64"` check and a URDF-load
guard test, not by changing the default.

---

## 4. Why Foxglove is the default viewer and ROS2/rviz is the advanced track

The choice here is driven almost entirely by the target machine: **a stock
Apple-Silicon Mac**, where native ROS2 + Gazebo are painful to install and run.

- **Foxglove (default)** is a polished, cross-platform *desktop app* that speaks
  a simple WebSocket protocol. The lab's `rl_lab/viz/foxglove_bridge.py` uses
  `foxglove-sdk` (a pure `pip install`) to serve `ws://127.0.0.1:8765` with the
  3D scene, the transform tree, and live reward/metric plots in one window — no
  ROS, no Homebrew, no compilers, no Rosetta. The same protocol later shows the
  **real** robot, so sim and hardware look identical in one layout.
- **ROS2 + `robot_state_publisher` + rviz2 (advanced track)** is offered as the
  `[ros2]` extra for people who already live in ROS. It is meant to run in
  **Docker or a Linux VM** — explicitly *not* natively on macOS. The RL code is
  byte-for-byte identical; only the visualization *sink* changes: instead of
  Foxglove `SceneUpdate` + `FrameTransform` messages, the bridge publishes
  `sensor_msgs/JointState` → `robot_state_publisher` → TF → an rviz2
  `RobotModel`. Same conceptual flow (joint angles → transform tree → meshes),
  different plumbing.

That two-viewer split is deliberate teaching: because *both* viewers consume the
same joint-state stream and neither touches the policy, you internalise early
that **visualization is decoupled from physics and from learning**. If your eyes
can't tell the Foxglove sim view from the rviz view (or from the real robot),
you have understood the abstraction.

---

## 5. One URDF, two consumers — and a path to the real robot

`urdf/buddy_jr.urdf` is the **single source of truth** for the robot's physical
definition, and it feeds two very different consumers from the same file:

```
                         urdf/buddy_jr.urdf
        (links, 4 revolute joints, axes, limits, inertias, fixed camera tip)
                          /                         \
              loaded by SIM                    described to VIZ
         (PyBullet / MuJoCo)                 (Foxglove / rviz)
   masses & inertias -> dynamics        link shapes -> 3D geometry
   joint axes & limits -> control       joint frames -> transform tree
                          \                         /
                   they MUST agree because they are the SAME file
```

Because physics and rendering both read the *one* URDF, the arm you see can
never silently disagree with the arm being simulated. Add a link or change a
limit once, and both sides update together. (`rl_lab/robot/` wraps this file in
Python — the joint table, FK/IK helpers, and the servo mapping below — so the
rest of the code never parses XML by hand.)

### From a trained policy to real SG90 servos

The same single-source-of-truth discipline is what makes sim-to-real a *short*
step rather than a rewrite. A trained policy outputs joint angles in **URDF
radians**; the real arm wants **servo degrees**. The whole bridge is one mapping,
owned by `rl_lab/robot/servo_map.py`:

```
   policy  --(URDF radians theta, per joint)-->  servo_map
                                                    |
                          servo_deg = degrees(theta) + 90   (URDF 0 rad = 90 deg, centre)
                          clamp to [0, 180]                 (and to the URDF +/-1.5708 limits)
                                                    |
                                                    v
          adafruit-circuitpython-servokit -> PCA9685 -> 4x SG90 servos (Buddy Jr)
```

In deployment (Experiment 12) the heavy training stack falls away: you load the
frozen policy, read the camera-detected target as the observation, run one
forward pass, map the action through `servo_map`, and drive the servos — while
publishing the *same* `JointState`/transform stream to Foxglove so the real arm
and the sim arm overlay in one view. The `[rpi]` extra
(`adafruit-circuitpython-servokit`) carries only what the Raspberry Pi 5 runtime
needs. See [`experiments/README.md`](../experiments/README.md) Experiments 11–12
for the safety shim (rate limiter, e-stop, dry-run) that wraps this path.

---

## 6. Where to go next

- **The design rationale and milestone roadmap:** [`PLAN.md`](PLAN.md) —
  the reconciled stack decision, the full directory layout, and the M1–M7
  milestones this architecture is built out across.
- **The hands-on learning ladder:** [`experiments/README.md`](../experiments/README.md)
  — the 12 runnable lessons, each of which is just a particular wiring of the
  layers described above.
- **The robot itself:** [`urdf/buddy_jr.urdf`](../urdf/buddy_jr.urdf) and the
  Buddy Jr build at <https://www.kevsrobots.com/blog/buddy_jr.html>.

Whenever a lesson confuses you, come back to the layer diagram in §1 and ask:
*which layer am I actually changing, and what does it promise the layers around
it?* That question, more than any single algorithm, is the skill this lab is
trying to teach.
