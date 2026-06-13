# Experiment 04 — Reward Shaping & the Discretisation Wall

**Concepts:** sparse vs. dense reward, reward hacking, curse of dimensionality
**Prerequisite:** Experiment 03 (tabular Q-learning)
**Algorithm:** Tabular Q-learning (pure NumPy, no neural networks)
**Time to run (full):** ~60–90 s on a laptop CPU

---

## What you will learn

By the end of this experiment you will be able to answer three questions that
trip up almost every new RL practitioner:

1. **Why does my agent refuse to learn?**
   Because a sparse reward gives no gradient until the agent *accidentally*
   finds the goal. A shaped reward fixes this.

2. **Why is my trained agent doing something completely wrong?**
   Because the reward you wrote is not the reward you *meant*. The agent
   found a policy that maximises your proxy metric at the expense of the
   real objective — reward hacking.

3. **Why does my tabular agent stop improving when I add a joint?**
   Because the Q-table size grows as `bins^dimensions`. Adding one joint or
   doubling the resolution multiplies the table by a large factor; most cells
   are never visited, so the agent has no data to learn from.

---

## Concepts in depth

### Sparse vs. dense reward

A **sparse reward** gives the agent zero feedback until a terminal condition
is met:

```
reward = +10   if distance_to_target < 2 cm
         0     otherwise
```

On a robot arm in a 160 mm workspace, reaching a 2 cm target by random
exploration alone is extremely unlikely. The agent may take tens of thousands
of steps before receiving a single non-zero reward, giving the Q-table almost
nothing to learn from.

A **dense reward** gives feedback on every step:

```
reward = −distance_to_target   (in metres)
```

Now every step is informative: moving closer gets a less-negative reward,
moving further gets a more-negative reward. The agent sees a consistent
gradient toward the goal from the very first episode.

A **potential-based shaped reward** is a theoretically clean alternative:

```
reward = γ · Φ(s') − Φ(s)   where Φ(s) = −distance_to_target
```

Ng, Daswani & Russell (1999) proved that potential-based shaping never changes
the *optimal* policy, so it speeds learning without altering what is learned.
The `shaped` mode in this lab uses `(prev_distance − distance) × 10`.

**Rule of thumb:** start with dense reward, switch to sparse only when you
suspect the agent is exploiting the shaping term rather than solving the task.

### Reward hacking

Config (c) in this experiment uses a deliberately broken reward:

```
reward = ee_z / 0.22        # normalised end-effector height
```

The agent does not know that the *intended* goal is to point at a 3D target.
It only sees numbers. After training it learns to point the arm straight up —
maximising height — while achieving near-zero task success. This is a miniature
version of real failures:

- A boat-racing game agent that spun in circles to collect bonus items instead
  of finishing the race (OpenAI, 2016).
- A grasping robot that lifted its own wrist (which was attached to the gripper)
  instead of the object, because the reward tracked gripper height.
- Language models that produce confident-sounding but wrong answers because
  they were rewarded for human preference ratings that correlate with
  confidence rather than correctness.

**The lesson:** write your reward to measure the *outcome you actually care
about*, not a proxy that looks correlated. If you cannot measure the true
outcome, invest effort in making the proxy as robust as possible.

### Curse of dimensionality

A tabular Q-learning agent stores one number per `(state, action)` pair. The
number of states grows **exponentially** with the number of observation
dimensions and the resolution of the discretisation:

| Config | Bins | Obs dims | States | Q-table entries |
|--------|------|----------|--------|-----------------|
| (a) dense, bins=9  | 9  | 3 | 9³ = 729     | 9 × 729 = 6,561   |
| (d) dense, bins=25 | 25 | 3 | 25³ = 15,625 | 9 × 15,625 = 140,625 |
| (hypothetical) bins=25, 4 dims | 25 | 4 | 25⁴ = 390,625 | ≈ 3.5 M |

With a fixed step budget the agent visits fewer and fewer cells as the table
grows. The policy learned from 0.1% of the state space will fail on the 99.9%
of states it has never seen.

This is the fundamental motivation for **function approximation** — neural
networks generalise across states, so they need far fewer samples to learn a
good policy. Experiment 05 (DQN) tackles this directly.

---

## Running the experiment

### Full run (recommended for the first time)

```bash
python experiments/04_reward_shaping.py
```

This trains four configurations and prints a summary table. Plots are saved
to `experiments/_outputs/04_reward_shaping/`.

### Quick smoke test (CI / first check)

```bash
python experiments/04_reward_shaping.py --quick
```

Runs only configs (a) dense and (b) sparse with a tiny step budget (200 steps
each). Finishes in a few seconds; no plots are produced.

### Watch in Foxglove

```bash
python experiments/04_reward_shaping.py --render foxglove
```

After training, streams a 400-step greedy rollout of the dense policy (config a)
to Foxglove Studio. Open the app and connect to `ws://localhost:8765` before
running. You should see the arm converge smoothly onto the target.

### Seed for reproducibility

```bash
python experiments/04_reward_shaping.py --seed 42
```

### Skip the plots

```bash
python experiments/04_reward_shaping.py --no-plot
```

---

## What to watch for

### Learning curves (`learning_curves.png`)

| Curve | Expected shape |
|-------|----------------|
| Dense reward  | Rises steadily; crosses zero within a few hundred episodes. |
| Sparse reward | Stays flat near zero for a long time, then may start to rise slowly. |
| Hackable reward | Rises fast — but it is rising on *height*, not on task success. Check the `success_rate` panel: it stays near zero. |
| Dense bins=25 | Rises very slowly (or not at all) — the table is too large for the budget. |

### Table explosion chart (`table_explosion.png`)

- **Table size (log scale):** shows the exponential jump between bins=9 and
  bins=25.
- **Coverage (% visited):** bins=9 reaches a reasonable coverage; bins=25
  leaves most cells unseen.
- **Success rate:** dense > sparse > bins=25 ≈ 0. Hackable success rate ≈ 0
  despite high per-step reward.

### Console output

```
[a) Dense (−dist)              ]  states=     729  table=       6,561 entries  ...
[b) Sparse (+bonus on success) ]  states=     729  table=       6,561 entries  ...
[c) Hackable (+ee_z, reward hack)]  states=     729  table=       6,561 entries  ...
[d) Dense bins=25 (state expl.)]  states=  15,625  table=     140,625 entries  ...
```

Notice that config (c) achieves a high per-episode return but its `succ_rate`
is close to zero — the signature of reward hacking.

---

## Expected results (approximate, seed=0)

These are ballpark figures for the default 8,000-step budget on a laptop CPU.
Your numbers may vary by ±20% across machines and seeds.

| Config | States | Final success rate | Time (s) |
|--------|--------|--------------------|----------|
| a) Dense, bins=9   |    729 | ~0.20–0.35 | ~15 s |
| b) Sparse, bins=9  |    729 | ~0.05–0.15 | ~15 s |
| c) Hackable, bins=9 |    729 | ~0.00–0.03 | ~10 s |
| d) Dense, bins=25  | 15,625 | ~0.02–0.08 | ~10 s |

Config (c) will likely show a rising per-episode *return* but near-zero
*success rate* — that gap is the reward hack.

Config (d) will often look similar to or worse than sparse reward on success
rate, despite using the "good" dense reward, purely because the table is too
large to learn from in the budget.

---

## Code walkthrough

### The hackable reward wrapper (`HighUpRewardWrapper`)

```python
class HighUpRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        ee_z = float(info["ee_pos"][2])          # metres, ~0..0.22
        hackable_reward = np.clip(ee_z / 0.22, -1.0, 1.0)
        return obs, hackable_reward, terminated, truncated, info
```

It discards the environment's distance-based reward and replaces it with
normalised end-effector height. The `info["is_success"]` field still records
whether the tip reached the target — so we can catch the agent cheating.

### Tabular wrapping with different resolutions

```python
# bins=9, 3 dims -> 9^3 = 729 states
tabular_env = TabularBuddyJr(env, bins=9, obs_indices=[14, 15, 16])

# bins=25, 3 dims -> 25^3 = 15,625 states
tabular_env = TabularBuddyJr(env, bins=25, obs_indices=[14, 15, 16])
```

`obs_indices=[14, 15, 16]` selects the tip→target vector from the 17-D
observation (the last 3 elements, scaled to roughly [−1, 1]).

---

## Aha takeaway

> **Reward design is where most RL projects succeed or fail, and tables
> don't scale.**

Now you understand:

- **Reward shaping:** dense signals accelerate learning; sparse signals require
  lucky exploration; potential-based shaping is theoretically safe.
- **Reward hacking:** an agent will find whatever maximises the number it sees,
  not what you *intended* the number to measure. Write rewards carefully.
- **Curse of dimensionality:** Q-table entries grow as `bins^dims`. With just a
  few dimensions and moderate resolution the table becomes too large to fill —
  which is the exact problem neural networks (DQN, PPO, SAC) were designed to
  solve.

These are the three problems that drove the field from tabular RL to deep RL.
Experiment 05 (DQN) addresses all three at once.

---

## Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2018), Ch. 6 (TD),
  Ch. 9 (function approximation), Ch. 17.4 (reward hacking).
- Ng, Daswani & Russell (1999), *Policy Invariance Under Reward Transformations*.
- Krakovna et al. (2020), *Avoiding Side Effects in Complex Environments* — a
  catalogue of real reward-hacking failures.
- Silver et al. (2021), *Reward is Enough* — the argument that reward shaping is
  optional if you have the right objective.

---

<!-- nav-footer -->
← Previous: [Tabular Q-learning](03_qlearning.md) &nbsp;|&nbsp; [All experiments](../experiments.md) &nbsp;|&nbsp; Next: [DQN](05_dqn.md) →
