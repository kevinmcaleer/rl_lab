# Experiment 01 — The Bandit Base: explore vs. exploit

**Part I — Intuition (no robot yet)**

---

## What this experiment teaches

Before we train Buddy Jr's arm to reach a target, we need the single most
important concept in all of reinforcement learning: **the tension between
exploring new possibilities and exploiting what you already know**.

This experiment introduces that tension in the simplest possible setting — one
where there is *no state, no sequence, no physics* — just a repeated choice
and a noisy reward signal.  Once you understand this, every algorithm in the
rest of the lab is just a more sophisticated version of the same idea.

By the end you will understand:

| Concept | Where it shows up later |
|---|---|
| Reward signal | Every RL algorithm |
| Action selection | Q-learning, DQN, PPO, SAC |
| Exploration vs. exploitation | ε in DQN; entropy bonus in SAC |
| ε (epsilon) parameter | Experiments 3, 5 |
| Optimism under uncertainty | Upper-Confidence-Bound, model-based RL |

---

## Background: the multi-armed bandit

Imagine a row of five slot machines (one-armed bandits).  Each machine pays
out a different average reward, but with noise — so you can't tell the best
one from a single pull.  You have a limited budget of pulls.  **How do you
find the best machine and earn as much as possible?**

This is the *multi-armed bandit* problem.  It has been studied for decades and
is the foundation of every A/B testing framework, clinical trial design, and
RL algorithm.

### Why Buddy Jr's base joint?

Buddy Jr's `base_yaw` joint rotates the whole arm left and right.  For a
camera-pointing task, getting the base angle roughly right is the first step —
everything else depends on it.

We model this joint as a **5-armed bandit**: the base can snap to one of five
discrete "slots" spread across its −90° to +90° range.  A hidden target lives
near one slot; rewards tell the agent how close the base direction is to the
target.

```
Slot index :  0      1      2      3      4
Angle      : -90°  -45°    0°   +45°   +90°
True mean  :  0.2   0.4   0.5    1.0    0.3   ← hidden from agent
```

The best slot is **slot 3 (+45°)** with mean reward 1.0.  The agent does not
know this; it must discover it from noisy samples.

### Regret: the cost of not knowing

We measure performance as **cumulative regret**:

```
regret(t) = Σ [ best_mean − reward(t) ]
```

Regret counts how much reward was *lost* by not always pulling the best arm.
A perfect oracle (always on slot 3) has zero regret.  A learning algorithm
accumulates regret early while exploring but should slow down once it finds the
best arm.

---

## Three strategies

### 1. Greedy

Always pick the arm with the **highest current Q-estimate** (sample average of
past rewards for that arm).  Q-values start at 0.

- **Pro:** Zero wasted pulls on exploration once confident.
- **Con:** Can lock onto a sub-optimal arm early and *never* escape.
  If slot 2 happens to return a reward of 0.8 on its first pull, and slot 3
  returns 0.3 on its first pull, the greedy agent will never try slot 3 again.

### 2. ε-Greedy (epsilon = 0.10)

Act greedily 90% of the time; pick a **uniformly random** arm the other 10%.

- **ε = 0** → pure greedy (no exploration).
- **ε = 1** → pure random (no exploitation).
- **ε = 0.1** → the typical sweet spot for bandit problems.

This tiny random fraction is usually enough to discover the best arm over time
without sacrificing much cumulative reward.  It is the same ε used in DQN
later in the curriculum.

### 3. Optimistic initial values (Q₀ = 2.0)

Start every Q-estimate at 2.0 — well above the true maximum (~1.0).  Use a
**greedy** rule (no ε).

- The agent pulls arm 0, receives ~0.2, updates Q[0] ≈ 1.1.
- Now arm 0's Q is lower than arms 1–4 (still at 2.0), so the agent tries arm 1.
- This continues until every arm has been tried enough times to bring its
  Q below the true best.
- **No random coin flip needed** — the optimism drives systematic exploration
  automatically.

---

## What to expect when you run it

### Cumulative reward

| Agent | Expected behaviour |
|---|---|
| Greedy | Rises fast early, then **plateaus** at a sub-optimal slope |
| ε-Greedy | Slightly noisier at first, then **overtakes greedy** as it finds slot 3 |
| Optimistic | Dips early (forced to try bad arms), then **climbs steeply** once calibrated |

The oracle line (always slot 3) shows the theoretical ceiling.

### Cumulative regret

Regret curves *always increase* — every non-optimal pull adds to the total.
Watch for:

- **Greedy** — regret grows at a **constant rate** indefinitely once stuck.
- **ε-Greedy** — regret growth *slows* as the best arm is found; a small
  residual slope remains (the 10% random pulls still sometimes land on bad arms).
- **Optimistic** — regret is high for the first ~50 pulls (exploring all arms),
  then **flattens** toward the optimal rate.

### % pulls on the best arm

This shows how quickly each agent converges to slot 3.

- Greedy often never exceeds 20% (it's stuck).
- ε-greedy converges toward ~90% (the 10% ε keeps it from 100%).
- Optimistic reaches near 100% and stays there.

### Final Q-value estimates (bar chart)

After 2000 pulls the bar chart shows what each agent *believes* about the arm
values.  A well-calibrated agent should have bar heights close to the true
means (dashed grey lines).  The greedy agent's chart often shows a heavily
inflated estimate for a wrong arm — it barely sampled the others.

---

## How to run

### Quick smoke test (a few seconds)

```bash
python experiments/01_bandit.py --quick
```

Runs 200 pulls per agent; prints a summary table; no plots.  Used by CI.

### Full run with plots

```bash
python experiments/01_bandit.py
```

Runs 2000 pulls per agent and saves four plots to
`experiments/_outputs/01_bandit/`:

| File | What it shows |
|---|---|
| `cumulative_reward.png` | Running total of rewards earned |
| `cumulative_regret.png` | Running total of missed reward |
| `pct_best_arm.png` | Rolling proportion of pulls on the best arm |
| `q_value_estimates.png` | Final Q-values vs. true arm means |

### With Foxglove streaming

```bash
python experiments/01_bandit.py --render foxglove
```

Starts a Foxglove WebSocket server on `ws://127.0.0.1:8765`.  After the
bandit run completes, replays the ε-greedy agent's pull sequence by physically
rotating Buddy Jr's `base_yaw` joint to the chosen slot at ~10 Hz.  Open
Foxglove Studio, connect to the local server, and watch the arm snap between
the five yaw positions.  The `/metrics` panel shows reward per pull.

### Reproducibility

```bash
python experiments/01_bandit.py --seed 42
```

All random number generators are seeded from `--seed`; the same seed produces
identical results across runs.

### Headless / no-plot

```bash
python experiments/01_bandit.py --no-plot
```

Skips plot generation.  Useful on servers without a display.

---

## Programmatic use

```python
from experiments import bandit_01  # or: import importlib; ...
metrics = bandit_01.run(quick=True, seed=0)
print(metrics)
# {
#   'final_regret_greedy': ...,
#   'final_regret_eps': ...,
#   'final_regret_opt': ...,
#   'best_arm_pct_greedy': ...,
#   'best_arm_pct_eps': ...,
#   'best_arm_pct_opt': ...,
#   'total_pulls': 200,
#   'seed': 0,
# }
```

---

## The "aha" takeaway

> **A learner that only exploits what it already knows can be permanently wrong.**

Pure greedy is not merely slow — it can be *stuck forever* on a sub-optimal
choice, confidently taking the wrong action every step.  A tiny random
fraction (ε) or an optimistic starting assumption is enough to prevent this.

**Now you understand:**

- What a **reward signal** is and how an agent learns from it.
- The **explore/exploit tension** — the fundamental trade-off in all of RL.
- What **ε (epsilon)** controls and why it appears in DQN, Q-learning, and SAC.
- That exploration doesn't always require randomness — **optimism** (initialising
  estimates high) can do the same job deterministically.

These concepts carry forward into every experiment that follows.  The only
thing that changes is how we represent *state* and how complex the action space
becomes.

---

## Connection to the rest of the lab

| Later experiment | Bandit concept re-used |
|---|---|
| Exp 03 — Tabular Q-learning | ε-greedy in the Q-table update loop |
| Exp 05 — DQN | ε-greedy exploration + annealing schedule |
| Exp 07 — REINFORCE | Stochastic policy as exploration mechanism |
| Exp 08 — PPO | Entropy bonus replaces explicit ε |
| Exp 10 — SAC | Maximum-entropy RL — built-in, principled exploration |

---

## Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
  Chapter 2 — Multi-armed Bandits.
  Free PDF at <http://incompleteideas.net/book/the-book.html>
- The original ε-greedy paper: Watkins (1989), *Learning from Delayed Rewards*.
- Upper Confidence Bound (UCB) — a provably optimal bandit algorithm that
  uses optimism formally: Auer et al. (2002), *Finite-Time Analysis of the
  Multiarmed Bandit Problem*, Machine Learning 47(2-3).
