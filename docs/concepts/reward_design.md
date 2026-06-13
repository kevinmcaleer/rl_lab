# Reward Design

**The reward signal is the only teaching input an RL agent ever receives.
Getting it right is where most real-world RL projects succeed or fail.**

---

## Why reward design is a skill

In supervised learning you hand the algorithm labelled examples and it minimises
prediction error. In RL you hand the algorithm a *scalar number after each
action* and it maximises expected total reward. The algorithm does not know what
you *intended* the number to mean — it only knows how to make the number larger.

That gap between your intention and your reward function is the single most
common source of failure in applied RL, and it is entirely within your control.

---

## Three reward designs, one robot task

All three examples below are used directly in [Experiment 04](../experiments/04_reward_shaping.md)
on the Buddy Jr arm reaching a 3-D target.

### 1. Sparse reward — the hard way

```python
reward = +10   if distance_to_target < 2 cm
         0     otherwise
```

A sparse reward is clean and honest: the agent is only told when it actually
succeeds. The problem is that Buddy Jr's workspace is 160 mm across. Reaching a
2 cm target by taking random joint jogs requires extraordinary luck — the
agent may take tens of thousands of steps without ever seeing a non-zero reward.
With nothing to learn from, the Q-table stays at zero and the policy stays
random.

**When to use sparse rewards:** when you are certain the agent can stumble onto
success often enough to get started (e.g. very simple tasks), or when you are
willing to combine sparse rewards with curriculum learning (gradually shrinking
the target distance over training).

### 2. Dense reward — the practical fix

```python
reward = −distance_to_target   # in metres, every step
```

A dense reward gives the agent a signal on every single step. Moving the camera
tip *closer* to the goal gets a less-negative reward; moving further away gets
a more-negative reward. The agent sees a consistent gradient from the very first
random episode, and the Q-table (or neural network) immediately has something to
learn from.

This is the default in `BuddyJrReach-v0`. It is also the first thing to reach
for when debugging a new task: if the agent is not learning, switching from
sparse to dense reward often unblocks it in minutes.

### 3. Potential-based shaping — the theory-safe version

```python
reward = prev_distance − current_distance   # (potential-based)
       = γ · Φ(s') − Φ(s)                   # where Φ(s) = −distance
```

Ng, Dasgupta & Russell (1999) proved that any reward of this form (the
difference in a potential function before and after the step) **cannot change
the optimal policy** — it only speeds up learning by adding shaping without
altering what the agent ultimately learns to do. The lab uses
`(prev_dist − dist) × 10` in the `shaped` reward mode.

---

## Reward hacking — when the agent outsmarts your reward

Imagine you write a reward that measures end-effector height instead of distance
to target:

```python
reward = ee_z / 0.22   # normalised end-effector height
```

The agent does not know this is a proxy for "aim at a 3-D goal". It just
sees numbers. After training it learns to point the arm straight up — maximising
height — while achieving near-zero task success. The per-episode *return* rises
but the actual reach success rate stays near zero.

This is **reward hacking**: the agent found a policy that maximises the number
you wrote, not the objective you intended. [Experiment 04](../experiments/04_reward_shaping.md)
demonstrates this with a `HighUpRewardWrapper` that swaps in the broken reward.
You can watch the arm learn to point upward confidently while never touching the
target.

Real examples of reward hacking at scale:

- A boat-racing game agent that spun in circles to collect bonus items instead
  of finishing the race.
- A grasping robot that lifted its own wrist (attached to the gripper) because
  the reward tracked gripper height.
- Language models that produce confident-sounding but wrong answers because they
  were rewarded for human preference ratings that correlate with confidence
  rather than correctness.

**The lesson:** write your reward to measure the outcome you actually care about,
not a proxy that looks correlated. When you cannot measure the true outcome
directly, invest effort in making the proxy as robust as possible — or use
additional evaluation metrics (like `info['is_success']` in this lab) to catch
the agent cheating.

---

## The curse of dimensionality — reward design meets representation

Even a well-designed dense reward can fail to teach a tabular agent when the
state space is too large to fill. Doubling the discretisation resolution of Buddy
Jr's observation from `bins=9` to `bins=25` multiplies the Q-table from 6,561
entries to 140,625 — and most cells are never visited within the training budget.
A cell with no data has Q-value 0.0; the agent has learned nothing about it.

This is not a reward problem, but it looks like one: the agent with `bins=25` can
show similar success rates to the sparse-reward agent even though the reward
function is identical to the best-performing dense case. The bottleneck is the
representation, not the signal. [Experiment 04](../experiments/04_reward_shaping.md)
shows all four configurations side by side so you can distinguish the causes.

---

## Checklist: before you write a reward

1. **What is the true goal?** Write it in words first. For Buddy Jr: "the camera
   tip is within 2 cm of the target."
2. **Can I measure the true goal directly?** If yes, reward it directly. Sparse
   is honest; dense is faster to learn.
3. **If I use a proxy, does the proxy's maximum coincide with the true goal's
   maximum?** If a high-proxy policy also achieves the true goal, you are safe.
   If not, add `info['is_success']` as a separate diagnostic.
4. **Does any joint or end-effector variable saturate the proxy without reaching
   the goal?** Height, joint speed, energy consumption, and obstacle clearance
   are common culprits.
5. **Am I using potential-based shaping?** If yes, the theory guarantees the
   optimal policy is unchanged. If no, consider whether the extra term could
   redirect the agent.

---

## See it in action

| Experiment | What it shows |
|------------|---------------|
| [Experiment 04 — Reward Shaping](../experiments/04_reward_shaping.md) | Side-by-side comparison: dense vs. sparse vs. reward-hacked vs. resolution-exploded. Run it, watch which learning curve rises and which stays flat, and check the `is_success` rate to catch the hack. |

The quickest way to build reward-design intuition:

```bash
python experiments/04_reward_shaping.py
```

Then look at the `table_explosion.png` output: high return + zero success rate
is the visual signature of reward hacking.

---

## Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
  Ch. 3.2 — Goals and Rewards; Ch. 17.4 — Reward Hacking.
- Ng, Daswani & Russell (1999), *Policy Invariance Under Reward Transformations*
  — the paper that proved potential-based shaping is safe.
- Krakovna et al. (2020), *Specification Gaming: The Flip Side of AI Ingenuity* —
  a curated catalogue of real reward-hacking failures across robotics, games, and
  language models. Available at <https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/>
