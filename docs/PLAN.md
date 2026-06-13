# Buddy Jr RL Lab — Project Plan

This document reconciles five specialist design docs (Architecture, URDF, RL Env,
Curriculum, Repo/PM) into one coherent plan: the architecture decision, the chosen
stack and data flow, the RL environment design, the directory layout, and the
roadmap by milestone.

---

## 1. Architecture decision (the reconciliation)

There was one genuine conflict between the specialist docs:

- The **Architecture** doc recommended **MuJoCo** as the default physics engine.
- The **RL Env** and **Repo/PM** docs assumed **PyBullet** as the default.

**Decision: PyBullet is the default backend; MuJoCo is a first-class swappable
alternative behind a `SimBackend` interface.**

Rationale:
- The audience is makers/hobbyists porting to *real hardware*. PyBullet's native
  URDF loader and its built-in `calculateInverseKinematics` directly mirror the
  blog's law-of-cosines IK, so a lesson can compare RL against classical IK with no
  extra tooling. This is the most pedagogically relevant path for this audience.
- PyBullet has an optional `GUI` mode for a quick local look, lowering friction
  for a first run.
- MuJoCo's strengths (cleanest arm64 wheels, higher actuator fidelity) are real and
  worth keeping. We therefore keep a thin **engine-abstraction layer** so switching
  to MuJoCo is a config flag — which *itself* becomes the "does my policy transfer
  across simulators?" experiment, turning the disagreement into a teaching moment.
- **Apple-Silicon reality:** PyBullet has no macOS arm64 wheel and its source build
  fails on recent Xcode, so it is shipped as the optional `[sim]` extra rather than
  a core dependency. macOS users install it via `conda install -c conda-forge
  pybullet`; Linux/Windows get a plain wheel. The base + `[dev]` install stays pure
  pip on macOS, and `test_urdf_loads.py` guards the PyBullet load with
  `importorskip` so CI is green on macOS without it.

Everything else from the Architecture doc stands: **Foxglove** is the default
visualizer, **Gymnasium** is the env API, **Stable-Baselines3** is the default
algorithm library, **MCAP** is the recording format, and **ROS2 + rviz** is the
optional advanced track (run in Docker / a Linux VM, never natively on macOS).

A second, minor reconciliation: the Curriculum doc and the PM draft listed slightly
different experiment sets. The **authoritative curriculum is the 12 numbered
lessons** in the Curriculum doc; the PM draft's `experiments/` tree and write-up
tasks are folded into those 12.

## 2. Chosen stack

```
PyBullet (physics, default) ── MuJoCo (optional, same SimBackend interface)
        │
   Gymnasium env  ──>  Stable-Baselines3 (PPO/SAC/DQN/TD3) + from-scratch algos
        │                                   │
        └────────────── FoxgloveBridge ─────┘
                              │
                  foxglove-sdk live server  ws://127.0.0.1:8765  + MCAP recording
                              │
                    Foxglove desktop app (3D panel + reward plots)
```

Core deps: `numpy`, `gymnasium`, `pybullet`, `stable-baselines3`, `torch` (CPU),
`foxglove-sdk`, `tensorboard`, `matplotlib`. Optional extras: `[mujoco]`,
`[ros2]`, `[rpi]` (`adafruit-circuitpython-servokit`), `[dev]`.

Target runtime: **Python 3.12** venv on macOS Apple Silicon (avoid 3.14 — wheels lag).

## 3. Data flow (one training/inference step)

1. **Physics step.** The Gymnasium env receives a 4-vector action (normalised joint
   deltas). It maps them to `POSITION_CONTROL` joint targets clamped to `[0, π]`
   (or `[-π/2, +π/2]` per the URDF's ±1.5708 rad limits) and steps the sim a few
   substeps per `ctrl_dt` for stability.
2. **State extraction.** FK (`getLinkState`) gives the camera-tip world pose. The
   env computes distance-to-target and reward, and returns
   `(obs, reward, terminated, truncated, info)`. `info` always carries
   `{distance, is_success, joint_q, ee_pos, target}`.
3. **State → visualization.** When `render_mode="foxglove"`, a `FoxgloveBridge`
   publishes, throttled to ~30 Hz:
   - one `FrameTransform` per joint (parent→child from current FK) so the
     URDF-shaped arm moves;
   - a `SceneUpdate` with a target sphere (green within tolerance), a tip marker,
     and a tip→target line that visualises the distance the reward minimises;
   - numeric channels (`distance`, `reward`, `episode_return`, `success_rate`) for
     Foxglove plot panels next to the 3D view.
4. **Transport + record.** `foxglove-sdk` serves the live WebSocket and can
   simultaneously write the same channels to an `.mcap` file for offline scrubbing.
5. **Render.** The Foxglove 3D panel composes the transform tree and pins each
   link's geometry to its frame. Recorded MCAP replays through the same layout.

The advanced ROS2/rviz variant publishes `sensor_msgs/JointState` →
`robot_state_publisher` → TF → rviz2 `RobotModel`. Same conceptual flow
(joint angles → transform tree → meshes), different sink.

## 4. RL environment design (summary)

- **Task — "Reach":** move the camera tip to within `tol = 2 cm` of a target sampled
  uniformly inside the reachable shell (always physically reachable, so beginners
  never mistake an impossible goal for a broken agent). Reach radius ≈ 0.16 m
  (80 mm + 80 mm).
- **Canonical env:** `BuddyJrReachEnv` (continuous). Discrete/tabular variants are
  thin `gymnasium.Wrapper`s so lessons share identical physics, reward and viz and
  can be compared honestly.
- **Observation:** continuous `Box(float32)` ≈ 17-D — `sin/cos` of each of 4 joints
  (avoids the angle wraparound discontinuity), tip position, target position, and
  vector-to-goal, all scaled to ~[-1, 1]. Optional `include_velocity` and a
  `Dict` goal-env variant (for HER) behind flags.
- **Actions:** continuous `Box(4,)` mapped to per-step joint deltas (default jog
  ~5°) for PPO/SAC; `Discrete(9)` (hold + ±jog per joint) for tabular/DQN. Same
  underlying joint-target update for honest cross-algorithm comparison.
- **Reward modes:** `sparse` (hard baseline + HER lesson), `dense = -distance`
  + success bonus (recommended default), and `shaped` (potential-based progress
  + small control penalty). Reward scales kept on the same order (success bonus
  `+10`, not `+1000`) so the gradient is not drowned.
- **Termination vs truncation kept strictly separate:** `terminated` on success
  (and optional self-collision); `truncated` on the time limit. Called out
  explicitly in lesson 3/4 because beginners bootstrap wrongly on timeouts.
- **Registered ids:** `BuddyJrReach-v0`, `BuddyJrReachDiscrete-v0`,
  `BuddyJrCameraPoint-v0`.

## 5. Directory layout

```
rl_lab/
├── README.md  LICENSE  CITATION.cff  CONTRIBUTING.md  CHANGELOG.md
├── pyproject.toml      # PEP 621, core deps + extras [mujoco][ros2][rpi][dev]
├── Makefile  .gitignore  .pre-commit-config.yaml  .python-version (3.12)
├── .github/            # ci.yml, docs.yml, release.yml, issue/PR templates, dependabot
├── urdf/
│   ├── buddy_jr.urdf            # validated single source of truth (at repo root today)
│   ├── buddy_jr.urdf.xacro      # optional parametric master
│   ├── meshes/{visual,collision}/   # STL from the printed parts (optional upgrade)
│   ├── targets/                 # goal-marker URDFs
│   └── README.md                # joint table, frames, axes, limits
├── rl_lab/                      # installable package
│   ├── robot/    buddy_jr.py kinematics.py servo_map.py
│   ├── sim/      base.py pybullet_sim.py mujoco_sim.py loader.py
│   ├── env/      buddy_jr_reach_env.py spaces.py rewards.py wrappers.py registration.py
│   ├── viz/      foxglove_bridge.py schemas.py urdf_publisher.py live_metrics.py rviz/
│   ├── algos/    tabular/ value_based/ policy_gradient/ sb3_integration.py registry.py
│   ├── train/    train.py evaluate.py callbacks.py logger.py
│   ├── utils/    seeding.py checkpoint.py plotting.py
│   └── cli.py    # `rl-lab` console entry point
├── experiments/   01..12 (the curriculum) + README.md + template
├── notebooks/     rl_concepts, inspecting_the_env, analysis template
├── deploy/raspberrypi/   run_policy.py servo_calibration.py requirements-pi.txt
├── docs/          MkDocs Material (getting_started, concepts, robot, experiments, api)
└── tests/         urdf, kinematics, env api, rewards, spaces, servo_map, algos smoke, schemas
```

Note: `buddy_jr.urdf` currently lives at the repo root and is validated
(well-formed, single root, acyclic tree, unit axes, valid limits, positive-definite
inertias). An early task moves it under `urdf/` without changing the kinematics.

## 6. Roadmap by milestone

- **M1 — Foundations & tooling:** repo scaffolding, packaging, CI, pre-commit, the
  layered architecture doc, and moving/validating the Buddy Jr URDF.
- **M2 — Simulation & Foxglove visualization:** `SimBackend` + PyBullet (default) and
  MuJoCo (optional), URDF loader, kinematics (FK + law-of-cosines IK), and the live
  Foxglove bridge with metrics panels + a saved layout.
- **M3 — Gymnasium RL environment:** observation/action spaces, composable rewards,
  `BuddyJrReachEnv`, wrappers (normalise, domain randomisation), registration, and
  `check_env` conformance.
- **M4 — RL algorithms:** from-scratch teaching impls (tabular Q-learning, SARSA,
  DQN, REINFORCE, minimal PPO) + SB3 integration, the train/eval CLI, logging and
  callbacks.
- **M5 — Experiment curriculum:** the 12 runnable, documented experiments — the
  heart of the lab — each its own `experiment` issue.
- **M6 — Sim-to-real & Raspberry Pi deployment:** servo mapping/calibration,
  on-device inference, safety (clamp/rate-limit/e-stop), SSH deploy tooling.
- **M7 — Documentation, tutorials & packaging:** MkDocs site, concept primers,
  per-experiment pages, notebooks, troubleshooting/FAQ, PyPI release workflow.

Dependencies are roughly sequential (M1→M2→M3→M4→M5), with M6 depending on a trained
policy from M5 and M7 running alongside throughout but finalised last.
