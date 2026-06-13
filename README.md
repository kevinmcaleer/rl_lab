# Buddy Jr RL Lab

[![CI](https://github.com/kevinmcaleer/rl_lab/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinmcaleer/rl_lab/actions/workflows/ci.yml)

A **reinforcement-learning simulation and learning lab** built around the
[Buddy Jr](https://www.kevsrobots.com/blog/buddy_jr.html) 4-DOF robot arm.

You experiment in a 3D simulator, *watch* a policy learn live, and work through a
progressive ladder of experiments that take you from "what is a reward?" all the
way to deploying a trained policy on real SG90 servos. The goal is not a demo — it
is to genuinely **teach RL** so you can apply it in your own robotics projects.

---

## Why this lab

- **You learn by watching.** Every experiment streams the arm and the reward
  signal live into a 3D viewer, so abstract RL concepts (exploration, value
  estimates, advantage, sim-to-real noise) become things you can *see*.
- **It runs on a Mac.** The base lab and dev tooling are 100% `pip install` on
  Apple Silicon — no ROS2, no Gazebo, no Homebrew. The one native piece, the
  PyBullet physics backend, ships as the `[sim]` extra (one `conda install` on
  Apple Silicon, a plain wheel everywhere else — see Quickstart below).
- **It's a real robot.** Buddy Jr is a printable 4-servo arm with a Raspberry Pi
  camera on the end. The same code you train in sim can drive the real servos.

## The Mac-friendly default stack

| Layer | Tool | Why |
|-------|------|-----|
| Physics (default) | **PyBullet** | native URDF loader + built-in FK/IK that mirrors the blog's law-of-cosines math; ships as the `[sim]` extra (wheel on Linux/Windows, conda-forge on macOS arm64) |
| Physics (swappable) | **MuJoCo** | Official arm64 wheels, higher actuator fidelity; selectable behind the same backend interface (and itself a "does my policy transfer?" experiment) |
| Env API | **Gymnasium** | Standard `reset()/step()` interface you modify in experiments |
| Algorithms | **Stable-Baselines3** | One-liner PPO / SAC / DQN / TD3 on macOS CPU, plus from-scratch teaching implementations |
| Visualization | **Foxglove** | Cross-platform native desktop app; the lab streams live 3D + reward plots over a WebSocket and records `.mcap` files you can scrub later |
| Robot model | **`buddy_jr.urdf`** | Single source of truth: 4 revolute joints, two 80 mm links, camera-tip end-effector — feeds both physics and visualization |

An **optional advanced track** swaps the visualizer for ROS2 + `robot_state_publisher`
+ rviz2 (run in Docker / a Linux VM on macOS). The RL code is identical; only the
visualization sink changes — a nice lesson in how visualization is decoupled from physics.

---

## Quickstart

You need **Python 3.12**. Python 3.14 is not supported — RL and simulation wheels
lag the newest CPython releases by several months.

### Step 1 — Create a virtual environment and install the lab

```bash
# Create and activate a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install --upgrade pip

# Install the lab package and all dev tooling (torch, SB3, Gymnasium, etc.)
pip install -e ".[dev]"
```

That single command gives you the CLI (`rl-lab`), all nine algorithms, all twelve
experiment scripts, and the Foxglove bridge. It uses the fast **kinematic backend**
by default (no native dependencies, physics-free, great for learning the algorithms).

When you are ready to add real physics, install the `[sim]` extra:

```bash
# Linux / Windows — a pre-built wheel is available:
pip install -e ".[sim]"

# macOS Apple Silicon — no pybullet wheel exists; build from conda-forge instead:
conda install -c conda-forge pybullet
```

For MuJoCo (official arm64 wheels, higher-fidelity physics):

```bash
pip install -e ".[mujoco]"
```

### Step 2 — Set up Foxglove to see the arm in 3D

1. Download and install the free **Foxglove** desktop app from
   <https://foxglove.dev/download>.
2. Open Foxglove and choose **Open connection**.
3. Set the connection type to **Rosbridge / WebSocket** and enter the URL
   `ws://localhost:8765`. Click **Open**.
4. Import the pre-built layout so the panels are arranged correctly:
   in Foxglove go to **Layout → Import from file** and select
   `rl_lab/viz/layouts/buddy_jr.json` from this repo.

> The first time a script opens the WebSocket server, macOS will ask
> "Do you want the application to accept incoming network connections?" —
> this is localhost-only and safe to allow.

See [docs/getting_started/foxglove_setup.md](docs/getting_started/foxglove_setup.md)
for screenshots and troubleshooting.

### Step 3 — See the arm in 3D (no learning yet)

Run the world-building experiment. It loads `buddy_jr.urdf`, jogs each joint
through its full range, and streams everything to Foxglove in real time:

```bash
python experiments/02_world.py --render foxglove
```

Switch to Foxglove and you should see the 4-DOF arm moving. This confirms that
your URDF, kinematics, and WebSocket bridge are all working before you add any RL.

To run the same experiment in headless / quick mode (no Foxglove needed):

```bash
python experiments/02_world.py --quick
```

### Step 4 — Run your first learning experiments

Start with the simplest possible RL problem — a multi-armed bandit — so you can
see exploration, epsilon decay, and reward accumulation before there is any robot
involved:

```bash
python experiments/01_bandit.py
```

The terminal prints per-episode reward and the evolving action-value estimates.
Once you understand why the agent converges to the best action, move to full
tabular Q-learning on the arm:

```bash
python experiments/03_qlearning.py
```

This experiment discretises the joint space into a Q-table and runs the Bellman
update live. Watch the Q-values converge in the terminal output. Both scripts
accept `--quick` to reduce episode count for a fast smoke-test, and `--render foxglove`
to stream joint states into Foxglove while they run.

### Step 5 — Train with a deep RL algorithm and evaluate the result

The `rl-lab` CLI wraps Stable-Baselines3 and the from-scratch implementations
behind a single command. Train PPO on the reach task:

```bash
rl-lab train --algo ppo --env BuddyJrReach-v0
```

Checkpoints are saved to `checkpoints/` as training progresses. When training
finishes (or at any checkpoint), evaluate the saved policy:

```bash
rl-lab eval --checkpoint checkpoints/ppo_BuddyJrReach-v0_final.zip
```

Add `--render foxglove` to any command to stream the evaluation live into Foxglove.

To see all available algorithms and environments:

```bash
rl-lab list
```

Available algorithms: `qlearning`, `sarsa`, `dqn`, `reinforce`, `ppo_min`, `ppo`,
`sac`, `td3`, `ddpg`.

Available environments: `BuddyJrReach-v0` (continuous, Box(4) actions),
`BuddyJrReachDiscrete-v0` (Discrete(9) actions), `BuddyJrCameraPoint-v0`
(continuous, Box(17) observations with camera vector).

---

## The experiment ladder

Twelve experiments, each runnable and documented, each teaching one new idea:

| # | Experiment | You learn |
|---|-----------|----------|
| 1 | **Bandit base** | reward, action, explore vs. exploit, epsilon |
| 2 | **Build the world** | URDF + viewer bridge; an environment as a first-class object |
| 3 | **Tabular Q-learning** | the MDP, Q-tables, the Bellman update, gamma |
| 4 | **Reward shaping & the discretisation wall** | reward design, reward hacking, the curse of dimensionality |
| 5 | **DQN** | function approximation, replay buffer, target networks |
| 6 | **Generalisation & domain randomisation** | memorisation vs. generalisation; first sim-to-real bridge |
| 7 | **REINFORCE from scratch** | policy gradients, log-prob trick, value baselines |
| 8 | **PPO** | actor-critic, GAE, the clipped surrogate objective |
| 9 | **Continuous PPO** | continuous actions, action scaling, smoothness penalties |
| 10 | **SAC + the full aim task** | off-policy continuous control, entropy, sample efficiency |
| 11 | **Closing the sim-to-real gap** | randomisation, noise, latency, rate limits, safety clamps |
| 12 | **Deploy to real hardware** | inference-only export driving SG90 servos via PCA9685 |

Each experiment lives in `experiments/NN_name.py` and accepts the same flags:
`--quick` (fewer episodes, fast smoke-test), `--render foxglove` (live 3D stream),
and `--seed N` (reproducible runs). See [`experiments/README.md`](experiments/README.md)
for the full learning path and prerequisites.

## The robot

Buddy Jr is a 4-DOF arm driven by 4x SG90 hobby servos through a PCA9685 PWM board
from a Raspberry Pi 5:

| idx | joint | axis | role | sim limit |
|-----|-------|------|------|-----------|
| 0 | `base_yaw` | Z | rotate whole arm L/R | ±90° |
| 1 | `shoulder_pitch` | Y | raise/lower segment 1 (80 mm) | ±90° |
| 2 | `elbow_pitch` | Y | bend segment 2 (80 mm) | ±90° |
| 3 | `camera_tilt` | Y | tilt the camera end-effector | ±90° |

URDF radians map to servo degrees as `servo_deg = degrees(theta) + 90`, clamped to
`[0, 180]`. See [`urdf/README.md`](urdf/README.md) for frames, axes and limits.

## Repository layout

```
rl_lab/
  urdf/            buddy_jr.urdf + meshes (single source of truth)
  rl_lab/          installable package: robot/ sim/ env/ viz/ algos/ train/
  experiments/     the 12-step curriculum, each runnable + documented
  notebooks/       Jupyter companions
  deploy/          Raspberry Pi inference + servo calibration
  docs/            MkDocs site (concepts, robot, experiments, getting started)
  tests/           URDF load, env API, kinematics, rewards, algo smoke tests
```

## Documentation

- [Architecture overview](docs/architecture.md) — the layered design and data flow
- [Project plan & roadmap](docs/PLAN.md)
- [Experiment curriculum](experiments/README.md)
- [Foxglove setup](docs/getting_started/foxglove_setup.md)
- [ROS2 / rviz2 optional track](docs/getting_started/installation_ros2.md)
- [Wiring the real robot](docs/robot/wiring.md)

## Credits & license

Buddy Jr robot by Kevin McAleer / [kevsrobots.com](https://www.kevsrobots.com/blog/buddy_jr.html).
Released under the MIT License.
