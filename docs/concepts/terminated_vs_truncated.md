# Terminated vs. Truncated

**Two ways an episode can end — and why conflating them quietly breaks your
algorithm.**

---

## The Gymnasium distinction

Every call to `env.step(action)` returns five values:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

Gymnasium (the modern successor to OpenAI Gym, used throughout this lab)
returns `terminated` and `truncated` as *separate* booleans. Older Gym code
returned a single `done` flag. That single flag is gone in Gymnasium, and
for good reason.

---

## What each flag means

### `terminated = True` — the task finished

The episode ended because the agent reached a **terminal state** defined by the
MDP. On Buddy Jr:

```python
terminated = distance_to_target < 0.02   # camera tip within 2 cm of the goal
```

This is genuine success. The arm is pointing at the target. If a Q-learning
agent sees `terminated = True`, it can correctly set the bootstrap target to
just the immediate reward (no future reward to add, because the MDP is
finished):

```
Q(s, a) ← r   (no bootstrap)     when terminated
```

In the experiment console you will see this reported as **SUCCESS**.

### `truncated = True` — the clock ran out

The episode ended because a **time limit was hit** (`max_steps = 200` for
Buddy Jr), not because the task succeeded or failed in any meaningful way. The
agent simply ran out of time.

This is *not* a terminal state of the MDP — if the arm had one more step, the
world would keep going and the agent could potentially still reach the target.
The bootstrap target should *still* include the discounted future value:

```
Q(s, a) ← r + γ · max_{a'} Q(s', a')   (continue bootstrap)   when truncated
```

In the experiment console you will see this reported as **TIMEOUT**.

---

## Why the difference matters for learning

If you treat `truncated` as `terminated` — setting the bootstrap target to just
`r` with no future value — you are telling the algorithm: "the value of this
boundary state is zero." That is wrong. The agent was not actually in a terminal
state; it was in a perfectly valid intermediate state that happened to be the
last step the time limit allowed. Zeroing out the future value creates a
**systematic bias** at episode boundaries.

For short episodes (50 steps) the bias is small. For Buddy Jr's 200-step
episodes with a continuous-control policy, incorrectly zeroing future values at
boundary states can noticeably slow convergence or create a persistent
underestimation of Q near the time limit.

In code:

```python
# CORRECT — Gymnasium style
obs, r, terminated, truncated, info = env.step(a)
done = terminated or truncated

# Q-learning bootstrap
if terminated:
    target = r                                 # MDP is over
else:
    target = r + gamma * max(Q[next_obs])      # still going (or just truncated)

# WRONG — old single-done style (breaks at truncation)
done_old = terminated or truncated
target = r + gamma * max(Q[next_obs]) * (1 - done_old)  # zeros future at truncation too!
```

Stable-Baselines3 (used in Experiments 08–11) handles this correctly internally
using the `TimeLimit` wrapper's `info['TimeLimit.truncated']` flag. The
from-scratch implementations in this lab also handle it correctly. Check the
DQN implementation in `rl_lab/algos/value_based/dqn.py` for the exact line.

---

## Reading the results table

[Experiment 03 (Q-learning)](../experiments/03_qlearning.md) prints a results
table at the end of evaluation that splits `terminated` and `truncated`
explicitly so you can see the difference:

```
Evaluation (greedy, 20 episodes):
  terminated (SUCCESS) :  14 / 20
  truncated  (TIMEOUT) :   6 / 20
  success rate         : 70%
```

A well-trained agent should have mostly `terminated` episodes. A large number
of `truncated` episodes means one of:

- The agent is still learning (success rate climbing, give it more steps).
- The task is genuinely too hard for the current algorithm or state
  representation.
- The episode is too short (`max_steps` is too small for the reach distance).
- The reward is misspecified and the agent is not actually learning to reach
  the target.

---

## A quick diagnostic in your own code

```python
import gymnasium as gym
import rl_lab  # registers BuddyJrReach-v0

env = gym.make("BuddyJrReach-v0")
obs, info = env.reset(seed=0)

terminated_count = 0
truncated_count  = 0

for _ in range(10):          # 10 episodes
    obs, info = env.reset()
    for _ in range(200):     # up to 200 steps each
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            terminated_count += 1
            break
        if truncated:
            truncated_count += 1
            break

print(f"terminated (success): {terminated_count}")
print(f"truncated  (timeout): {truncated_count}")
# With random actions almost all episodes will be truncated — expected!
```

Run this before training. With random actions, almost every episode hits the
time limit (`truncated`). After training, most episodes should terminate early
(`terminated`) because the agent reaches the target. That shift from
`truncated` to `terminated` *is* learning.

---

## The historical context

Pre-Gymnasium, OpenAI Gym returned a single `done = terminated or truncated`
flag. Many classic RL implementations (and many online tutorials) still use
`done` and zero out the bootstrap at every episode end:

```python
target = r + gamma * max_Q * (1 - done)   # silently wrong at truncation
```

This is technically incorrect but often harmless when episodes are short and the
task is easy. It becomes a real problem in robotics tasks with long horizons
(like 200-step arm reaching) or when the time limit is tight relative to the
task difficulty. Gymnasium's split was introduced explicitly to eliminate this
class of subtle bug.

When you read older RL code or papers that pre-date Gymnasium, mentally
translate `done` to `terminated or truncated` and ask whether the
implementation handles the two cases differently — often it does not, and that
is a bug worth noting.

---

## See it in action

| Experiment | What it shows |
|------------|---------------|
| [Experiment 02 — Build the World](../experiments/02_world.md) | The raw Gymnasium loop: `reset()`, `step()`, `terminated`, `truncated` printed for a scripted sweep. No learning — just the loop itself. |
| [Experiment 03 — Q-learning](../experiments/03_qlearning.md) | Evaluation results table that splits `terminated` (SUCCESS) vs. `truncated` (TIMEOUT) across 20 greedy episodes. Watch how the split changes as you vary `max_steps` or training length. |

---

## Further reading

- Gymnasium documentation — *Core API: Termination and Truncation*:
  <https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Ch. 3.3
  — Episodic and Continuing Tasks. The textbook definition of terminal states.
- Towers et al. (2023), *Gymnasium* — the paper accompanying the library,
  which explains the design decision to separate the two flags.
