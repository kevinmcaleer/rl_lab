# Experiment 03 — Tabular Q-learning

> **You will learn:** what a Q-table is, how the Bellman update works,
> why tabular RL cannot scale to large state spaces, and how to read
> a policy field to understand what your agent has actually learned.

---

## The concept

### Markov Decision Processes (MDPs)

All RL problems share the same mathematical skeleton: an **MDP**.

| Symbol | Name | What it means here |
|:------:|------|---------------------|
| **S** | State | The discretised tip→target error vector (3 bins) |
| **A** | Action | One of 9 jog commands (no-op, ±base, ±shoulder, ±elbow, ±camera) |
| **R** | Reward | Dense shaping: negative distance to target, +1 on success |
| **T** | Transition | The kinematics backend moves the joints, computes a new tip position |
| **γ** | Discount | How much a reward tomorrow is worth vs. a reward right now |

At every step the agent sees the current state **s**, picks an action **a**,
receives reward **r**, and arrives in the next state **s′**.  Gymnasium gives
you two flags at the end of each episode:

* **`terminated = True`** — the tip reached within `success_tol` of the goal.
  The agent *won*. This is a genuine task success.
* **`truncated = True`** — the episode hit `max_steps` before the agent
  succeeded.  The agent *ran out of time*. No success.

These are conceptually different and the experiment prints the split explicitly.

---

### The Q-table

Q-learning stores one number for every **(state, action) pair** in a 2-D
table called the **Q-table**:

```
Q-table shape: (n_states, n_actions)  →  (343, 9) with bins=7
```

The value `Q[s, a]` represents the agent's current belief about the
*expected discounted cumulative reward* it will collect if it takes action `a`
in state `s` and then acts greedily ever after.

---

### The Bellman update

After each step `(s, a, r, s′, done)` the agent applies one update:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_{a′} Q(s′, a′) − Q(s, a)]
                          └──────── TD target ────────┘
                    └────────────── TD error (δ) ─────────────────┘
```

**Walking through the terms:**

| Term | Role |
|------|------|
| `α` (alpha=0.15) | Learning rate — how much to shift Q toward the new evidence |
| `r` | Immediate reward received on this step |
| `γ` (gamma=0.97) | Discount — a reward 10 steps later is worth γ¹⁰ ≈ 0.74 of the same reward now |
| `max_{a′} Q(s′, a′)` | The best Q-value we *believe* is achievable from s′ (greedy bootstrap) |
| `TD error δ` | How surprised we were: positive δ → better than expected → increase Q |

This is called **off-policy** because the bootstrap uses the *best* next action,
not the action the ε-greedy policy will actually take next.

---

### Observation discretisation and the curse of dimensionality

The reach environment has a **17-dimensional continuous** observation.
Pure tabular RL needs a finite state space.  The `TabularBuddyJr` wrapper
extracts the **3-D tip→target error vector** (indices 14–16) and bins each
dimension into `bins` buckets:

```
n_states = bins³
bins=7  →  343 states  (manageable)
bins=20 →  8 000 states (getting large)
bins=50 →  125 000 states (too slow to fill)
```

This is the **curse of dimensionality** in practice.  If we tried to bin all
17 dimensions: 7¹⁷ ≈ 2.3 × 10¹⁴ states — impossible to visit them all.
This is exactly why we need neural networks (Experiment 05, DQN).

---

### Epsilon-greedy exploration

The agent starts with `epsilon = 1.0` (random actions) and decays it each
episode:

```
epsilon ← max(epsilon_min, epsilon × epsilon_decay)
        = max(0.05, epsilon × 0.995)
```

Early on, exploration fills the Q-table.  Later, exploitation dominates.
The floor `epsilon_min = 0.05` ensures the agent never *completely* stops
exploring — important because the environment has some reset noise.

---

### Fixed target

To make Q-learning converge in a short session, the environment is configured
with a narrow `target_radius=(0.08, 0.09)` — a thin shell so the target is
always at roughly the same distance from the base.  This shrinks the
effective state space the Q-table needs to cover.

Experiment 04 (SARSA) uses a wider distribution to show what happens when
the target can appear anywhere in the workspace.

---

## How to run

### Full run (recommended first time)

```bash
python experiments/03_qlearning.py
```

This trains for **60 000 steps** (~2–5 minutes on a laptop CPU), then:

* Prints a results table showing terminated vs truncated counts.
* Saves two plots under `experiments/_outputs/03_qlearning/`.

### Quick smoke test

```bash
python experiments/03_qlearning.py --quick
```

Completes in a few seconds.  Used by the CI pipeline.

### Live Foxglove streaming

```bash
python experiments/03_qlearning.py --render foxglove
```

After training, streams one greedy episode to
[Foxglove Studio](https://foxglove.dev/download) so you can watch Buddy Jr
navigate to the target using the learned Q-table.

Connect Foxglove Studio to `ws://localhost:8765` before running.

### Reproducible seed

```bash
python experiments/03_qlearning.py --seed 42
```

---

## What to look at

### Console output

During training you will see progress every 500 steps:

```
  step    500  ep   27  return +3.12  ε=0.872  success_rate=0.02
  step   1000  ep   52  return +4.89  ε=0.762  success_rate=0.06
  ...
  step  60000  ep 1841  return +9.14  ε=0.050  success_rate=0.71
```

After training, a results table:

```
============================================================
EXPERIMENT 03 — Tabular Q-learning RESULTS
============================================================
  Training steps   : 60,000
  Episodes trained : ~1800
  Q-table shape    : (343, 9)
  Final epsilon    : 0.0500

Evaluation (greedy, 20 episodes):
  terminated (SUCCESS) :  14 / 20
  truncated  (TIMEOUT) :   6 / 20
  success rate         : 70%

KEY INSIGHT: terminated=success (arm reached goal) vs.
             truncated=timeout (episode hit max_steps).
A well-trained agent should have mostly terminated episodes.
============================================================
```

---

### Plot 1 — Learning curve

`experiments/_outputs/03_qlearning/learning_curve.png`

Shows cumulative reward per episode (faint blue) plus a rolling mean
(solid blue).  Expect:

* **Episodes 1–200**: near-zero or negative returns — the table is mostly
  zeros and exploration is random.
* **Episodes 200–800**: rising trend — Q is filling in useful values.
* **Episodes 800+**: the curve plateaus at a positive value — the agent
  has learned a reliable greedy path to the target.

High variance between adjacent episodes is normal for tabular RL — different
reset positions map to different visited states.

---

### Plot 2 — Policy field

`experiments/_outputs/03_qlearning/policy_field.png`

A 2-D heatmap showing the **greedy action** (colour/number) for each
(dx bin, dy bin) cell while the dz bin is fixed at its middle value.

| Element | Meaning |
|---------|---------|
| x-axis  | dx bin — tip-to-target x-error (low = target is to the left) |
| y-axis  | dy bin — tip-to-target y-error (low = target is behind) |
| Colour  | Greedy action chosen by argmax Q(s, ·) for that cell |
| Number in cell | Action index (0 = no-op, 1 = base+, 2 = base−, …) |

**What a well-trained policy looks like:**
The colours should form rough quadrants — base-rotation actions dominate
the left/right columns; shoulder/elbow actions appear more in the top/bottom
rows; the centre cell (dx≈0, dy≈0) should show action 0 (no-op) or a joint
that fine-tunes the dz dimension.

If the field looks random (all colours mixed without pattern), training
is still early — run for more steps.

---

## Hyper-parameters to experiment with

| Parameter | Default | Try | What changes |
|-----------|---------|-----|-------------|
| `bins` | 7 | 3, 5, 10 | Q-table size; more bins = finer resolution but slower to fill |
| `alpha` | 0.15 | 0.01, 0.5 | Low α = slower but stable; high α = fast but can oscillate |
| `gamma` | 0.97 | 0.5, 0.99 | Low γ = myopic (only cares about next step); high γ = far-sighted |
| `epsilon_decay` | 0.995 | 0.99, 0.9995 | Fast decay = less exploration; slow decay = more |
| `target_radius` | (0.08, 0.09) | (0.04, 0.155) | Wide band = harder (more states to cover) |

---

## What you understand now

After completing this experiment you can answer:

1. **What is a Q-table?** A lookup table mapping every (state, action) pair
   to an estimate of the expected discounted return.

2. **What is the Bellman update?** A one-step correction that shifts Q toward
   the observed reward plus the discounted best future value.

3. **Why does tabular RL struggle with continuous spaces?** Because the number
   of states grows exponentially with the number of dimensions being binned
   (curse of dimensionality).  DQN (Experiment 05) replaces the table with a
   neural network that generalises across unseen states.

4. **What is the difference between terminated and truncated?**
   `terminated` = genuine task success (goal reached).  `truncated` = time
   ran out.  A good agent maximises terminated and minimises truncated.

5. **What does gamma trade off?** How much future rewards matter relative to
   immediate ones.  γ close to 1 makes the agent plan far ahead; γ close to 0
   makes it short-sighted.

---

## Next steps

* **Experiment 04 — SARSA**: Same task, on-policy learning, wider target
  distribution.  Contrast on-policy vs off-policy convergence.
* **Experiment 05 — DQN**: Replace the Q-table with a neural network —
  generalises to the full 17-D observation without binning.

---

<!-- nav-footer -->
← Previous: [Build the world](02_world.md) &nbsp;|&nbsp; [All experiments](../experiments.md) &nbsp;|&nbsp; Next: [Reward shaping](04_reward_shaping.md) →
