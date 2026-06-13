# Buddy Jr RL Lab

[![CI](https://github.com/kevinmcaleer/rl_lab/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinmcaleer/rl_lab/actions/workflows/ci.yml)

A **reinforcement-learning simulation and learning lab** built around the
[Buddy Jr](https://www.kevsrobots.com/blog/buddy_jr.html) 4-DOF robot arm.

You experiment in a 3D simulator, *watch* a policy learn live, and work through a
progressive ladder of experiments that take you from "what is a reward?" all the
way to deploying a trained policy on real SG90 servos. The goal is not a demo — it
is to genuinely **teach RL** so you can apply it in your own robotics projects.

> Status: **early scaffolding / planning.** The Buddy Jr URDF exists and is
> validated; the package, environment, algorithms, and experiments are being
> built out milestone by milestone. See [`docs/PLAN.md`](docs/PLAN.md) and the
> GitHub issues/milestones for the roadmap.

---

## Why this lab

- **You learn by watching.** Every experiment streams the arm and the reward
  signal live into a 3D viewer, so abstract RL concepts (exploration, value
  estimates, advantage, sim-to-real noise) become things you can *see*.
- **It runs on a Mac.** The default stack is 100% `pip install` on Apple Silicon
  — no ROS2, no Gazebo, no Homebrew, no compilers, no Rosetta.
- **It's a real robot.** Buddy Jr is a printable 4-servo arm with a Raspberry Pi
  camera on the end. The same code you train in sim can drive the real servos.

## The Mac-friendly default stack

| Layer | Tool | Why |
|-------|------|-----|
| Physics (default) | **PyBullet** | `pip install pybullet`, native URDF loader, built-in FK + IK helper that mirrors the blog's law-of-cosines math |
| Physics (swappable) | **MuJoCo** | Official arm64 wheels, higher actuator fidelity; selectable behind the same backend interface (and itself a "does my policy transfer?" experiment) |
| Env API | **Gymnasium** | Standard `reset()/step()` interface you modify in experiments |
| Algorithms | **Stable-Baselines3** | One-liner PPO / SAC / DQN / TD3 on macOS CPU, plus from-scratch teaching implementations |
| Visualization | **Foxglove** | Cross-platform native desktop app; the lab streams live 3D + reward plots over a WebSocket and records `.mcap` files you can scrub later |
| Robot model | **`buddy_jr.urdf`** | Single source of truth: 4 revolute joints, two 80 mm links, camera-tip end-effector — feeds both physics and visualization |

An **optional advanced track** swaps the visualizer for ROS2 + `robot_state_publisher`
+ rviz2 (run in Docker / a Linux VM on macOS). The RL code is identical; only the
visualization sink changes — a nice lesson in how visualization is decoupled from physics.

## Quickstart (placeholder — being implemented)

```bash
# 1. Create a Python 3.12 venv (NOT 3.14 — RL/sim wheels lag the newest CPython)
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 2. Install the lab (default Mac-friendly stack)
pip install -e .            # pyproject extras: [mujoco] [ros2] [rpi] [dev]

# 3. Download the Foxglove desktop app (free) and open it.
#    Connect to the live server at ws://localhost:8765

# 4. See the robot move (no RL yet)
rl-lab sim hello            # loads buddy_jr.urdf, jogs joints, streams to Foxglove

# 5. Run your first training
rl-lab train --algo ppo --env BuddyJrReach-v0
```

The first time the WebSocket server opens, macOS may ask to "accept incoming
connections" — it is localhost-only and safe to allow.

## The experiment ladder

Twelve experiments, each runnable and documented, each teaching one new idea:

| # | Experiment | You learn |
|---|-----------|-----------|
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

See [`experiments/README.md`](experiments/README.md) for the full learning path
and prerequisites.

## The robot

Buddy Jr is a 4-DOF arm driven by 4× SG90 hobby servos through a PCA9685 PWM board
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

## Credits & license

Buddy Jr robot by Kevin McAleer / [kevsrobots.com](https://www.kevsrobots.com/blog/buddy_jr.html).
Released under the MIT License.
