# Markov Decision Processes (MDPs)

**The mathematical skeleton that every RL algorithm shares.**

---

## The problem RL solves

Reinforcement learning is a way of teaching an agent to make good decisions by
giving it feedback — reward — after each action. To write that idea down
precisely, researchers reached for a framework called the **Markov Decision
Process** (MDP). Every algorithm in this lab, from the simplest Q-table to SAC,
is solving an MDP. Understanding the pieces is what lets you read a learning
curve and know *what went wrong*.

---

## The five pieces

### S — State

The state is everything the agent knows about the world right now. For Buddy Jr
the Gymnasium environment returns a **17-dimensional observation vector**
at every step:

```
obs[0:4]   — sin of the four joint angles  (base_yaw, shoulder_pitch, elbow_pitch, camera_tilt)
obs[4:8]   — cos of the four joint angles  (same order)
obs[8:11]  — camera-tip position in world space  (x, y, z)
obs[11:14] — target position in world space  (x, y, z)
obs[14:17] — tip-to-target vector  (target_pos − tip_pos)
```

The last three elements are the most action-relevant: they tell the agent exactly
how far it needs to move in each direction. Everything else helps the agent
represent its own joint configuration.

### A — Action

The set of moves available to the agent in each state. `BuddyJrReach-v0` has a
**continuous** action space: a 4-element vector in `[−1, 1]`, one component per
joint. The environment scales each component by `0.1 rad` and adds it to the
current joint angle before clamping to the URDF limits of `±1.5708 rad (±90°)`.

The discrete variant, `BuddyJrReachDiscrete-v0`, collapses those continuous
moves to **9 jog commands**: no-op, plus-or-minus for each of the four joints.

### R — Reward

The scalar signal the environment sends after every step. It is the *only*
teaching signal an RL algorithm sees. The default reward in Buddy Jr is:

```
reward = −distance_to_target   (in metres)   every step
       + 1.0                                  only when distance < 2 cm (success)
```

Small, negative values mean the arm is far away; values close to zero mean it
is almost touching the target. Experiment 04 shows what happens when this signal
is changed or broken.

### T — Transition

The rule that maps `(state, action)` to the next state. In a simulation this is
the physics engine. In the kinematic backend it is the analytic forward-
kinematics chain. The agent never sees this rule directly — it can only observe
the *effects* of its actions through the resulting observations.

The key property of an MDP is the **Markov property**: the next state depends
only on the *current* state and the action taken, not on anything that happened
before. Buddy Jr's observation is designed to satisfy this — it contains enough
information (joint angles *and* the full tip-and-target pose) that the past
history adds nothing useful.

### γ — Discount factor

A number in `[0, 1]` (typically 0.97–0.99) that says how much a reward arriving
one step in the future is worth *today*. A reward `r` arriving `k` steps from
now contributes `γᵏ · r` to the current decision.

| γ value | What the agent cares about |
|---------|---------------------------|
| 0.0 | Only the very next reward — completely short-sighted |
| 0.5 | Rewards more than a few steps away are nearly worthless |
| 0.97 | A reward 10 steps away is worth `0.97¹⁰ ≈ 0.74` of one today |
| 0.99 | A reward 50 steps away is still worth `0.99⁵⁰ ≈ 0.60` of one today |

For Buddy Jr reaching tasks, episodes run up to 200 steps and success only pays
out when the tip arrives at the goal. A high γ (0.97–0.99) is necessary so the
agent values that distant success pay-off from the very first step of each
episode.

---

## The Bellman equation — the idea behind every value-based algorithm

The agent wants to know the **value** of being in a state: the total discounted
reward it can expect if it acts optimally from here. Call this `V*(s)`. The
Bellman equation expresses it recursively:

```
V*(s) = max_a  [ R(s, a) + γ · Σ_{s'} P(s' | s, a) · V*(s') ]
```

In English: the value of a state is the reward you get now plus the (discounted)
value of the best next state you can reach. Q-learning (Experiment 03) builds on
this by storing `Q(s, a) = R(s, a) + γ · max_{a'} Q(s', a')` in a table and
updating it one step at a time:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_{a'} Q(s', a') − Q(s, a) ]
                           └──────── Bellman target ────────────────┘
```

DQN (Experiment 05) replaces the table with a neural network but the same
Bellman target drives every gradient step.

---

## The agent–environment loop

Every episode follows the same cycle:

```
    ┌──────────────────────────────────────────────────────┐
    │  env.reset()  →  obs₀, info₀                        │
    │                                                      │
    │  for each step t:                                    │
    │      agent picks action aₜ from policy π(obs_t)     │
    │      env.step(aₜ)  →  obs_{t+1}, r_t, term, trunc   │
    │      agent learns from (obs_t, aₜ, r_t, obs_{t+1})  │
    │      if term or trunc: env.reset(); break            │
    └──────────────────────────────────────────────────────┘
```

Every experiment in this lab is just a particular filling-in of that loop: a
different policy, a different update rule, and a different analysis of what
came out.

---

## Summary table

| Symbol | Name | Buddy Jr meaning |
|:------:|------|-------------------|
| S | State | 17-D obs: joint angles, tip pos, target pos, tip→target vector |
| A | Action | 4-D continuous joint-delta, or 9 discrete jog commands |
| R | Reward | −distance + success bonus |
| T | Transition | Kinematics / PyBullet / MuJoCo physics step |
| γ | Discount | 0.97–0.99 — keeps distant success pay-off relevant |

---

## See it in action

| Experiment | What it shows |
|------------|---------------|
| [Experiment 02 — Build the World](../experiments/02_world.md) | Watch `reset()` / `step()` / `reward` in a scripted sweep with no learning — the bare MDP loop. |
| [Experiment 03 — Q-learning](../experiments/03_qlearning.md) | The Bellman update applied to a Q-table; `terminated` vs. `truncated` printed explicitly; discount γ as a hyperparameter to vary. |

---

## Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Ch. 3 —
  MDPs. Free PDF at <http://incompleteideas.net/book/the-book.html>
- Bellman, R. (1957), *Dynamic Programming* — the original statement of the
  Bellman equation.
