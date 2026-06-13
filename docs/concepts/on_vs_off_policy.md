# On-Policy vs. Off-Policy Learning

**The same experience can be used in two very different ways. Which way you
choose has a bigger impact on sample efficiency and stability than almost any
other algorithmic decision.**

---

## The core question

After the agent takes a step and observes `(state, action, reward, next_state)`,
what policy does it use to compute the learning target?

- **On-policy:** the target is based on what the *same* (current) policy would
  do next. The agent learns about the policy it is actually following.
- **Off-policy:** the target is based on the *best possible* next action,
  regardless of what the current policy would do. The agent can learn about a
  *different* (often greedy) policy even while following an exploratory one.

---

## Buddy Jr example: Q-learning vs. SARSA

Both Q-learning and SARSA use a table of Q-values and update them with a
one-step TD error. The only difference is one line in the update rule.

### Q-learning — off-policy

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_{a'} Q(s', a') − Q(s, a) ]
                                      ^^^^^^^^^^^^^^^^^
                                      greedy over next state
                                      (regardless of what ε-greedy would pick)
```

Q-learning says: "I took action `a` with my ε-greedy policy, but the value I
should assign to it assumes I act *greedily* ever after." The two policies —
the behaviour policy (ε-greedy, used to collect experience) and the target
policy (greedy, used to compute the update) — are **different**. That is the
definition of off-policy.

Because the target is always the *best* next action, Q-learning converges
directly to the optimal Q-function, `Q*`. It can also reuse experience stored
in a **replay buffer** — transitions collected under old policies are still valid
targets for a greedy policy. This is the key property DQN exploits.

### SARSA — on-policy

SARSA stands for State–Action–Reward–State–Action, naming all five quantities
involved in a single update:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · Q(s', a') − Q(s, a) ]
                                       ^^^^
                                       the actual next action the policy picks
                                       (including the ε random moves)
```

SARSA uses the *actual* next action `a'` that its ε-greedy policy will take —
not the best possible one. It is learning the value of the policy it is actually
running, including its exploratory random moves.

This makes SARSA **safer** in environments where exploratory actions are costly:
if a random move leads to a bad state, SARSA learns to avoid the (state, action)
pair that got it there; Q-learning does not, because it assumes you would act
greedily next.

### Practical consequence for Buddy Jr

With a narrow target band `(0.08–0.09 m)` the two algorithms converge similarly.
With a **wide target distribution** covering the full workspace, SARSA's
on-policy nature makes it slower to converge (it has to account for the full
ε-greedy policy, including the noise) but more conservative — it learns a policy
that works even with some exploration still happening.

---

## The key trade-off at a glance

| Property | On-policy (SARSA, PPO) | Off-policy (Q-learning, DQN, SAC) |
|----------|------------------------|------------------------------------|
| **What it learns** | The value of the current policy | The value of the greedy / optimal policy |
| **Experience reuse** | Each batch used once then discarded | Stored in a replay buffer, used many times |
| **Sample efficiency** | Lower — experience is discarded | Higher — each transition is replayed |
| **Stability** | Generally more stable | Requires tricks (target network, replay) |
| **Safety during training** | Conservative — risky exploratory moves hurt the target | Optimistic — assumes greedy behaviour after each step |

---

## How this distinction grows with the curriculum

The on/off-policy split is not just a tabular-RL concept. It persists through
every algorithm in the lab:

### On-policy family

- **SARSA** — the tabular archetype (Experiment 03).
- **REINFORCE** — collects whole episodes under the current policy, computes
  returns, updates once, discards the data (Experiment 07).
- **PPO** — collects a mini-batch under the current policy, does a handful of
  gradient steps with the clipped surrogate, then discards and re-collects
  (Experiments 08, 09). Each collected batch is used only a few times (typically
  3–10 epochs) before being thrown away.

### Off-policy family

- **Q-learning** — the tabular archetype (Experiment 03).
- **DQN** — neural Q-network, off-policy by the same max-bootstrap logic as
  Q-learning, enabled by a replay buffer (Experiment 05).
- **SAC** — stores every transition in a large replay buffer and replays each
  one many times. On continuous robot control tasks, SAC typically reaches a
  good policy in **far fewer environment steps** than PPO because every
  transition is reused. This is critical when "environment steps" correspond to
  real servo moves on a real arm (Experiment 10).

---

## When to choose each

**Choose on-policy (PPO) when:**

- Stability matters more than sample efficiency.
- The environment is fast to simulate (you can afford to re-collect data).
- You want a reliable default with few hyperparameters to tune.

**Choose off-policy (SAC, DQN) when:**

- Environment interactions are expensive (real hardware, slow physics).
- You need to re-use collected data efficiently.
- The action space is continuous and high-dimensional (SAC is purpose-built for
  this).

For Buddy Jr deployed on a Raspberry Pi 5 (Experiment 12), every servo move
wears the hardware and takes real time. Off-policy SAC's replay-buffer re-use
becomes genuinely valuable, not just a theoretical advantage.

---

## See it in action

| Experiment | What it shows |
|------------|---------------|
| [Experiment 03 — Q-learning](../experiments/03_qlearning.md) | Q-learning (off-policy, `max` bootstrap) vs. SARSA (on-policy, `actual-next-action` bootstrap) on the same discrete reach task. The console prints which update rule is active so you can compare convergence speed and final success rate. |
| [Experiment 10 — SAC vs. PPO](../experiments/10_sac_aim.md) | On-policy PPO vs. off-policy SAC on the continuous aim task. Plot mean return vs. environment steps: SAC's replay buffer makes it visibly more sample-efficient. |

---

## Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Ch. 6.4
  (SARSA) and Ch. 6.5 (Q-learning). The two update equations appear side by
  side with a clear explanation of why the difference matters.
- Mnih et al. (2015), *Human-level control through deep reinforcement learning*,
  Nature — DQN, the off-policy deep-learning milestone.
- Haarnoja et al. (2018), *Soft Actor-Critic: Off-Policy Maximum Entropy Deep
  Reinforcement Learning with a Stochastic Actor* — SAC's original paper.
