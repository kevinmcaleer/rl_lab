# Buddy Jr RL Lab — Experiments

A hands-on, progressive reinforcement-learning (RL) curriculum that takes you
from **zero RL knowledge** to **deploying a learned policy on a real robot arm**.

Every experiment teaches **one** concept, gives you something concrete to **run
and watch in a 3D viewer**, and ends with an **"aha" takeaway** plus a
**"now you understand X"** statement you can carry into your own projects.

The robot we learn on is **Buddy Jr** — a 4-DOF 3D-printed arm
([kevsrobots.com](https://www.kevsrobots.com/blog/buddy_jr.html)) whose job is to
**point a Raspberry Pi camera at a target in 3D space**.

---

## The robot we are teaching

Buddy Jr has **4 revolute joints**, all driven by SG90 hobby servos
(0–180°) via a PCA9685 PWM board over I2C from a Raspberry Pi 5
(`adafruit-circuitpython-servokit`). Servo power comes from a Pimoroni Yukon
bench module, not the Pi.

| Joint | Name | Axis | What it does |
|------:|------|------|--------------|
| 0 | Base | vertical (yaw) | rotates the whole arm left/right |
| 1 | Shoulder | horizontal (pitch) | raises/lowers the first 80 mm segment |
| 2 | Elbow | horizontal (pitch) | bends the second 80 mm segment |
| 3 | Camera / wrist | horizontal (pitch) | tilts the camera at the end |

`SHOULDER_LENGTH = 80 mm`, `ELBOW_LENGTH = 80 mm`. End-effector = the camera lens.
The reach envelope is therefore a ~160 mm sphere around the base.

> **Why this matters for RL.** The blog version of Buddy Jr aims the camera using
> *hand-derived inverse kinematics* (law of cosines + trig). This lab asks a
> different question: **can the arm learn to aim itself from experience, with no
> kinematics equations at all?** That is what RL gives us — and what makes it
> transfer to robots whose geometry you can't easily solve by hand.

---

## The stack (macOS-first, pip-installable)

You are on Apple Silicon / macOS, where native ROS2 + Gazebo are painful. So the
**default track is 100% pip-installable** and works out of the box:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib gymnasium pybullet stable-baselines3 torch \
            foxglove-sdk imageio
```

- **PyBullet** — the physics engine. Loads our Buddy Jr URDF, steps dynamics,
  computes contacts, and renders a built-in 3D window (no ROS needed).
- **Gymnasium** — the standard `env.reset()` / `env.step(action)` RL interface
  that every algorithm and library understands.
- **Stable-Baselines3 (SB3)** — battle-tested DQN / PPO / SAC implementations so
  you study *behaviour*, not buggy gradient math.
- **Foxglove** — the live 3D viewer. We publish the arm's joint states + target
  marker over the Foxglove WebSocket protocol so you can watch training in a
  polished cross-platform app, and later watch the **real** robot the same way.
- **matplotlib** — for the learning-curve / value-function plots that make the
  abstract concepts visible.

> **Optional advanced track (ROS2 + rviz).** If you already run ROS2 (e.g. in a
> Linux VM or container), each experiment notes how to publish the same
> `JointState` + `Marker` topics so you can visualise in **rviz** instead of
> Foxglove. The RL code is identical; only the viewer bridge changes. Skip this
> if you're on a stock Mac.

### Two viewers, one mental model

We deliberately use **two** visualisers so you internalise the most important
sim-to-real idea early: *the policy doesn't know or care what renders it.*

- **PyBullet's own GUI** — zero setup, great for fast debugging.
- **Foxglove (or rviz)** — the "production" viewer. The **exact same**
  joint-state messages drive the sim arm here and, in the final experiments, the
  **real** Buddy Jr. If your eyes can't tell sim from real in Foxglove, you've
  understood the abstraction.

### Repo layout you'll build toward

```
rl_lab/
├── docs/experiments.md          # this file
├── urdf/buddy_jr.urdf           # 4-DOF model (built in Experiment 2)
├── buddy_lab/
│   ├── viz/foxglove_bridge.py   # publishes JointState + target Marker
│   ├── envs/                    # one Gymnasium env per stage
│   └── agents/                  # bandit, tabular Q, plus SB3 configs
└── experiments/                 # one runnable script per experiment below
```

---

## How to read each experiment

Each section has the same five parts so you always know what you're getting:

1. **Concept** — the single RL idea being taught.
2. **Objective** — what you should be able to *do* afterwards.
3. **Build & run** — the concrete thing you create and execute.
4. **Watch for** — the specific behaviour to observe in the viewer/plots.
5. **Aha takeaway** — the one-sentence insight, and a *"now you understand X"*.

Difficulty ramps from "no robot at all" up to "real hardware". Do them in order;
each reuses the mental model and often the code of the previous one.

---

# Part I — Intuition (no robot yet)

## Experiment 1 — The Bandit Base: explore vs. exploit

- **Concept:** Reward, action, exploration vs. exploitation — the core tension of
  all RL, with **no state and no sequence** to distract you.
- **Objective:** Understand *why* a learner must sometimes act sub-optimally on
  purpose, and what ε (epsilon) controls.
- **Build & run:** Model Buddy Jr's **base joint** as a 5-armed bandit. The base
  can snap to one of 5 yaw "slots" (−90°, −45°, 0°, +45°, +90°). The target light
  is hidden behind one slot; each slot returns a noisy reward = how centred the
  (still imaginary) camera would be. Implement three action-selection rules in
  `experiments/01_bandit.py`: **greedy**, **ε-greedy**, and **optimistic initial
  values**. Run 1000 pulls each.
  - In the viewer: the base servo physically rotates to the chosen slot each step,
    and the chosen slot flashes green/red by reward. (Foxglove marker = the target
    glow.)
- **Watch for:** Pure-greedy locks onto the first decent slot and *never*
  discovers the true best one. ε-greedy keeps sampling and finds it. Plot
  cumulative reward and **regret** for all three.
- **Aha takeaway:** *A learner that only exploits what it already knows can be
  permanently wrong.* **Now you understand exploration vs. exploitation, reward
  signals, and the ε knob — the vocabulary the rest of the lab is built on.**

## Experiment 2 — Build the world: URDF + the viewer bridge

- **Concept:** The **environment** is a first-class object. Before an agent can
  learn, you must define states, actions, the reset, and the reward — and be able
  to *see* them.
- **Objective:** Author the Buddy Jr URDF and prove you can drive it from Python
  and watch it in Foxglove/PyBullet. This is the foundation every later
  experiment loads.
- **Build & run:** Write `urdf/buddy_jr.urdf`: base link + 4 revolute joints
  (`base_yaw`, `shoulder_pitch`, `elbow_pitch`, `camera_pitch`), 80 mm shoulder
  and elbow links, a small camera box as the tip, joint limits 0–π (matching the
  SG90 180° range). Then `experiments/02_world.py` loads it in PyBullet, sweeps
  each joint through its range, and streams `JointState` to Foxglove via
  `buddy_lab/viz/foxglove_bridge.py`. Add a movable red sphere = the camera
  **target**.
- **Watch for:** Every commanded joint angle moves the matching part; the
  forward-kinematics tip position printed in Python lands exactly where the
  camera box is in the viewer. Confirm the reach sphere is ~160 mm.
- **Aha takeaway:** *RL doesn't start with an algorithm — it starts with a clean
  environment definition you can trust and observe.* **Now you understand the
  agent–environment loop (`reset`/`step`/`reward`) and why a faithful sim +
  viewer is non-negotiable.**

---

# Part II — Tabular RL (the algorithm you can read by hand)

## Experiment 3 — Tabular Q-learning on a discretised reach

- **Concept:** **State, action, reward, next-state** (the MDP); the **Q-table**;
  the **Bellman update**; the **discount factor γ**.
- **Objective:** Train an agent with *no neural network* and inspect the literal
  numbers it learns, so the learning rule stops being magic.
- **Build & run:** Freeze the camera/wrist joint. Discretise the remaining three
  joints into coarse bins (e.g. base ×9, shoulder ×7, elbow ×7) and actions into
  {−1 bin, hold, +1 bin} per joint. Reward = +1 when the camera tip is within
  2 cm of a **fixed** target, small step penalty otherwise. Implement plain
  Q-learning in `experiments/03_qlearning.py` (a real NumPy table, no library).
- **Watch for:** Plot a **learning curve** (steps-to-target falling over
  episodes). In the viewer, the arm goes from random flailing to a clean, repeatable
  reach. Then *open the Q-table*: render the greedy action for each state as
  arrows and watch a "policy field" pointing toward the target form.
- **Aha takeaway:** *Learning is just repeatedly nudging a number toward
  "reward now + discounted best future".* **Now you understand the Bellman
  equation, Q-values, γ, and learning rate α — concretely, not symbolically.**

## Experiment 4 — Reward shaping & the discretisation wall

- **Concept:** **Reward design** (sparse vs. dense/shaped), and the **curse of
  dimensionality** that kills tabular methods.
- **Objective:** Feel *why* we are forced to abandon tables and move to function
  approximation — by hitting the wall yourself.
- **Build & run:** Two parts, reusing Experiment 3's env.
  1. Swap the sparse +1 reward for a **shaped** reward
     `−distance(tip, target)` and re-train. Then try a *badly* shaped reward
     (e.g. reward for being high up) and watch the agent **hack** it.
  2. Increase resolution: 9→25 bins per joint and **un-freeze** the wrist (3→4
     dims). Print the Q-table size and wall-clock-to-learn.
- **Watch for:** Shaped reward learns far faster but the bad shaping produces a
  confidently-wrong policy in the viewer (arm proudly points at the ceiling).
  The table size explodes (millions of cells); learning stalls because most
  states are never visited.
- **Aha takeaway:** *Reward design is where most RL projects succeed or fail, and
  tables don't scale.* **Now you understand reward shaping, reward hacking, and
  the curse of dimensionality — the exact problems neural networks were brought
  in to solve.**

---

# Part III — Deep value-based RL

## Experiment 5 — DQN: replace the table with a network

- **Concept:** **Function approximation** with a neural net; **experience
  replay** and the **target network**; why naive deep Q-learning is unstable.
- **Objective:** Train your first deep-RL agent and understand the two tricks
  (replay buffer + target net) that make it stable.
- **Build & run:** Keep discrete actions, but feed the **continuous** state
  (raw joint angles + target xyz) into a small MLP. Use SB3 `DQN` in
  `experiments/05_dqn.py` on the now-4-DOF reach env from Exp 4. Run once with
  the replay buffer + target net (defaults) and once after deliberately
  shrinking the buffer / syncing the target every step.
- **Watch for:** With the tricks, the learning curve climbs smoothly and the arm
  reaches **continuously** without bins. Without them, the curve oscillates or
  collapses — instability you can *see* as the policy "forgetting". Watch greedy
  rollouts live in Foxglove.
- **Aha takeaway:** *A network lets one policy generalise across infinitely many
  states, but only if you decorrelate samples and stabilise the target.* **Now
  you understand DQN, experience replay, target networks, and why deep RL needs
  stabilisation.**

## Experiment 6 — Generalisation: moving targets & domain randomisation

- **Concept:** **Generalisation** vs. memorisation; **domain randomisation** as
  the bridge toward sim-to-real robustness.
- **Objective:** See the difference between a policy that memorised one target
  and one that learned the *task*.
- **Build & run:** Take the Exp 5 DQN and change `reset()` to spawn the target at
  a **random reachable point** every episode, and add the target's position to
  the observation. Optionally randomise link length ±5% and add servo-angle
  noise. Evaluate on targets the agent never saw.
- **Watch for:** The fixed-target agent fails immediately on a new target; the
  randomised-target agent tracks brand-new targets in the viewer. Drag the target
  sphere live in Foxglove and watch the arm follow.
- **Aha takeaway:** *Train on variety or you'll only ever memorise one episode.*
  **Now you understand generalisation, observation design, and domain
  randomisation — your first concrete sim-to-real tool.**

---

# Part IV — Policy-based RL

## Experiment 7 — Policy gradients: REINFORCE from scratch

- **Concept:** Learning a **policy directly** (a distribution over actions)
  instead of values; the **score-function / log-prob gradient**; why **variance**
  is the enemy and a **baseline** helps.
- **Objective:** Understand the second great family of RL methods by writing the
  simplest member by hand.
- **Build & run:** On the discretised reach task (so you can keep it minimal),
  implement REINFORCE in `experiments/07_reinforce.py`: a softmax-policy MLP,
  collect whole episodes, push up log-probs of actions weighted by return.
  Then add a **value baseline** and compare.
- **Watch for:** Raw REINFORCE learns but its learning curve is *noisy* and slow;
  adding the baseline visibly smooths it. In the viewer the motion goes from
  twitchy to deliberate as the policy sharpens.
- **Aha takeaway:** *You can optimise behaviour directly, and the main battle is
  taming gradient variance.* **Now you understand policy gradients, stochastic
  policies, returns, and baselines — the foundation under PPO.**

## Experiment 8 — PPO: a policy gradient that actually behaves

- **Concept:** **Actor–critic** architecture; **advantage estimation (GAE)**; the
  **clipped surrogate objective** that keeps updates from blowing up.
- **Objective:** Use the modern default on-policy algorithm and understand what
  the "clip" is protecting you from.
- **Build & run:** SB3 `PPO` on the 4-DOF discrete reach env in
  `experiments/08_ppo.py`. Train, then sweep the **clip range** (e.g. 0.05, 0.2,
  0.6) and the number of epochs per batch.
- **Watch for:** Well-tuned PPO is dramatically more stable and sample-efficient
  than your hand-written REINFORCE. Too-large clip → unstable; too-small → crawls.
  Watch the reward curves stack up in matplotlib and the rollouts in Foxglove.
- **Aha takeaway:** *PPO = policy gradients + a critic + a trust region you can
  tune.* **Now you understand actor–critic, advantages, and why PPO is the
  go-to baseline for robotics RL.**

---

# Part V — Continuous control (the real arm's action space)

## Experiment 9 — Continuous PPO: smooth servo commands

- **Concept:** **Continuous action spaces** (real servos take a continuous angle,
  not "bin +1"); **action scaling** and **smoothness penalties**.
- **Objective:** Control all 4 joints with continuous outputs and learn to value
  motion quality, not just hitting the goal.
- **Build & run:** New env `BuddyReachContinuous-v0`: action = a 4-vector of joint
  *velocity* (or delta-angle) commands in [−1, 1], scaled to servo limits.
  Train SB3 `PPO` (Gaussian policy) in `experiments/09_ppo_continuous.py`.
  Add a reward term penalising large/jerky actions.
- **Watch for:** Without the smoothness term the arm reaches but **buzzes** — the
  kind of chatter that destroys SG90 gears. With it, the motion becomes smooth and
  servo-friendly. Compare both live in the viewer.
- **Aha takeaway:** *Continuous control is what real actuators need, and the
  reward must encode "be gentle", not just "be correct".* **Now you understand
  continuous action spaces, action scaling, and why smoothness/effort penalties
  matter for hardware.**

## Experiment 10 — SAC: sample-efficient continuous control + the full aim task

- **Concept:** **Off-policy continuous control (SAC)**; **entropy-regularised
  exploration**; **sample efficiency** vs. on-policy PPO.
- **Objective:** Solve the *real* Buddy Jr task — **aim the camera at a moving
  target using all 4 DOF** — and understand when to pick SAC over PPO.
- **Build & run:** Upgrade the reward to the true objective: the camera's
  **view ray** must point at the target (orientation, not just tip position), with
  the wrist joint now load-bearing. Train SB3 `SAC` in
  `experiments/10_sac_aim.py` and run PPO on the same task for comparison; plot
  reward vs. environment-steps.
- **Watch for:** SAC typically reaches good aiming with **far fewer environment
  steps** (it reuses a replay buffer). In Foxglove, render the camera's
  view-frustum and watch it lock onto and track the dragged target.
- **Aha takeaway:** *Off-policy methods squeeze more learning out of each sample,
  which is gold when "samples" are expensive robot moves.* **Now you understand
  SAC, the entropy bonus, and the PPO-vs-SAC trade-off for real robots.**

---

# Part VI — Sim-to-real: deploy toward the real Buddy Jr

## Experiment 11 — Closing the sim-to-real gap

- **Concept:** The **reality gap**; **domain randomisation**, **observation
  noise**, **action latency/rate limits**, and **safety clamping** as the tools
  that close it.
- **Objective:** Harden the Exp 10 policy so it has a fighting chance on real
  SG90 servos — *before* touching hardware.
- **Build & run:** In `experiments/11_robustify.py`, wrap the env to model real
  Buddy Jr: SG90 angle resolution + jitter, PCA9685 update rate, control latency,
  gravity sag, and clamp every command to **safe** servo ranges. Retrain/fine-tune
  the SAC policy under this randomisation.
- **Watch for:** The naive Exp 10 policy degrades under noise/latency; the
  robustified one stays accurate. Visualise the *distribution* of randomised
  parameters and the policy's success rate across them.
- **Aha takeaway:** *A policy that's perfect in a pristine sim will fail on
  real servos — robustness must be trained in, not hoped for.* **Now you
  understand the reality gap and the standard toolkit (randomisation, noise,
  latency, safety limits) for crossing it.**

## Experiment 12 — Deploy: same messages, real hardware

- **Concept:** **Policy export & inference deployment**; decoupling **training**
  from a lightweight **runtime**; **safe rollout** with a human in the loop.
- **Objective:** Run the trained policy in inference mode and send its actions to
  real SG90 servos via the PCA9685 — watching the real arm and the sim in the
  **same** Foxglove layout.
- **Build & run:** Export the SAC policy to a small inference module. Write
  `experiments/12_deploy.py` that runs on the Raspberry Pi 5: load the policy,
  read the (real or mocked) camera-detected target as the observation, run a
  forward pass, map the action to servo angles, and drive them with
  `adafruit-circuitpython-servokit` (PCA9685, Yukon-powered). Mirror the same
  `JointState` + target `Marker` to Foxglove. Include a **dry-run** mode
  (Foxglove only, servos detached) and a **rate limiter + e-stop** key.
- **Watch for:** The **same policy bytes** that drove the sim now move the real
  arm; the live Foxglove view of the hardware overlays the sim arm closely. Note
  every place reality diverges (servo dead-band, sag, target-detection lag) — your
  next round of randomisation.
- **Aha takeaway:** *Deployment is just inference + a thin, safe hardware shim;
  training and running are different jobs.* **Now you understand policy export,
  inference-only deployment, and safe rollout — you can now take an RL policy from
  sim to your own robot.**

---

## What you can now do on your own robot

By the end you have, end-to-end, the workflow professionals use:

1. **Model** a robot as a URDF + a Gymnasium environment (states, actions,
   reward, reset) — Exp 2–4.
2. **Choose** an algorithm by trade-off, not hype: tabular → DQN → PPO → SAC,
   discrete → continuous, on- vs. off-policy — Exp 3, 5, 8, 9, 10.
3. **Design rewards** that produce the behaviour you actually want, and recognise
   reward hacking — Exp 1, 4, 9.
4. **Make policies generalise and survive reality** with randomisation, noise,
   latency and safety limits — Exp 6, 11.
5. **Deploy** a trained policy to real actuators with a safe, lightweight runtime
   and a shared viewer — Exp 12.

Point this at your *own* mechanism — a different arm, a rover, a pan-tilt — by
swapping the URDF, the observation/reward, and the hardware shim. The RL stays
the same. **That is the whole point of the lab.**

---

## Concept-to-experiment map (quick reference)

| RL concept | First taught in |
|---|---|
| Reward, exploration vs. exploitation, ε | Exp 1 |
| Agent–environment loop, env definition | Exp 2 |
| MDP, Q-table, Bellman update, γ, α | Exp 3 |
| Reward shaping, reward hacking, curse of dimensionality | Exp 4 |
| Function approximation, replay buffer, target network (DQN) | Exp 5 |
| Generalisation, observation design, domain randomisation | Exp 6 |
| Policy gradients, stochastic policy, baselines (REINFORCE) | Exp 7 |
| Actor–critic, advantages/GAE, clipped objective (PPO) | Exp 8 |
| Continuous actions, action scaling, smoothness penalties | Exp 9 |
| Off-policy continuous control, entropy, sample efficiency (SAC) | Exp 10 |
| Reality gap, randomisation, noise, latency, safety | Exp 11 |
| Policy export, inference deployment, safe rollout | Exp 12 |
