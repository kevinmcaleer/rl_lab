# Value Methods vs. Policy Methods

**There are two ways to build an RL agent. Understanding the difference tells
you which one to reach for — and why every modern algorithm combines both.**

---

## Two routes to good behaviour

An RL agent needs to answer one question: "given the state I am in, which action
should I take?" There are two fundamentally different answers to how you
represent the knowledge that lets the agent answer that question.

### Value methods

Learn the **value** of each (state, action) pair — how much total reward the
agent expects if it takes that action now and then acts optimally. Store this in
a table or neural network (`Q(s, a)`), and derive behaviour by acting greedily:

```
action = argmax_a  Q(s, a)
```

The policy is *implicit* — it falls out of the Q-function. The agent never
explicitly stores "in state `s` take action `a`"; it stores the expected reward
for every option and picks the best.

### Policy methods

Learn the **policy** directly — a function `π(a | s; θ)` that maps states to
a probability distribution over actions. The agent outputs action probabilities
and samples from them:

```
action ~ π(· | s; θ)
```

No Q-function is involved. The parameters θ are updated by pushing up the
probability of actions that led to high returns and pushing down those that led
to low returns (the **policy gradient** idea).

---

## Buddy Jr through both lenses

### Value approach: DQN on the discrete reach task

In [Experiment 05](../experiments/05_dqn.md), a neural network approximates
`Q(s, a)` for `BuddyJrReachDiscrete-v0`. The state is the 17-D observation;
the action is one of 9 discrete joint jogs. The network outputs **9 numbers**,
one per action, and the greedy policy picks the largest:

```
Input: obs (17,)
       ↓
  Linear(17→64) + ReLU
  Linear(64→64) + ReLU
  Linear(64→9)
       ↓
Output: [Q(s,jog_base+), Q(s,jog_base-), Q(s,jog_shoulder+), ..., Q(s,noop)]
        → pick action = argmax
```

The training target is the Bellman equation:
`y = r + γ · max_{a'} Q_target(s', a')`. Every gradient step pushes the
predicted Q-value toward `y`.

### Policy approach: REINFORCE on the discrete reach task

In [Experiment 07](../experiments/07_reinforce.md), a neural network directly
represents `π(a | s; θ)`. The same 17-D observation goes in; what comes out is
a **softmax probability distribution over 9 actions**:

```
Input: obs (17,)
       ↓
  Linear(17→64) + ReLU
  Linear(64→64) + ReLU
  Linear(64→9) + Softmax
       ↓
Output: [0.03, 0.41, 0.08, 0.12, 0.18, 0.06, 0.04, 0.05, 0.03]
        → sample from this distribution
```

The training update uses the **score-function trick** (log-derivative trick):

```
∇_θ J(θ) = E [ Σ_t  ∇_θ log π(a_t | s_t; θ)  ·  G_t ]
```

In English: increase the log-probability of actions that were followed by a
high return; decrease it for actions followed by a low return. No Q-table is
needed. The environment's dynamics are never differentiated through — the
gradient is estimated by sampling complete episodes.

---

## Comparing the two families

| Property | Value methods (Q-learning, DQN) | Policy methods (REINFORCE, PPO) |
|----------|--------------------------------|---------------------------------|
| **What is stored** | Q-function or value table | Policy network parameters |
| **Action selection** | Greedy argmax (deterministic) | Sample from probability distribution |
| **Action spaces** | Best for small discrete spaces | Handles both discrete and continuous naturally |
| **Exploration** | Explicit ε-greedy or UCB | Implicit: stochastic policy; entropy bonus |
| **Data usage** | Off-policy: can reuse old data | On-policy: usually needs fresh data |
| **Variance / stability** | Lower variance, can diverge with function approximation | Higher variance gradients, stabilised by baselines |
| **Continuous control** | Requires discretisation or separate max-step | Native: output mean + std of a Gaussian |

---

## Why modern algorithms combine both

Pure value methods struggle with continuous action spaces (you cannot do
`argmax` over infinitely many actions) and can diverge when using neural network
approximators. Pure policy methods have high-variance gradient estimates that
slow learning.

The solution — used by PPO, SAC, TD3, and DDPG — is the **actor-critic**
architecture:

```
  Actor  π(a | s; θ)      — learns the policy directly
  Critic V(s; φ) or Q(s, a; φ)  — estimates value to reduce gradient variance
```

The critic provides a **baseline** that the actor's gradient is measured against:

```
advantage  A(s, a) = Q(s, a) − V(s)   ("was this action better than average?")
policy gradient ∝  ∇_θ log π(a | s; θ)  ·  A(s, a)
```

Subtracting `V(s)` dramatically reduces the variance of the gradient without
introducing bias — the critic absorbs the "background level" of return so the
actor only sees the signal that comes from choosing one action over another.
This insight, first made rigorous in REINFORCE-with-baseline (Experiment 07),
underpins every experiment from Experiment 08 onward.

---

## The progression through this lab

```
 Tabular Q-learning  →  DQN           — value methods, discrete actions
      (Exp 03)            (Exp 05)

 REINFORCE  →  PPO  →  SAC            — policy methods (+ critic)
  (Exp 07)     (Exp 08)  (Exp 10)
```

After Experiment 05 you understand value methods deeply enough to see *why*
they struggle with continuous control. After Experiment 07 you understand policy
gradients deeply enough to see *why* variance reduction matters. Experiment 08
(PPO) and Experiment 10 (SAC) show how to combine both ideas into algorithms
that are practical on real hardware.

---

## See it in action

| Experiment | What it shows |
|------------|---------------|
| [Experiment 05 — DQN](../experiments/05_dqn.md) | Value method: Q-network trained with Bellman targets on the discrete reach task. Includes the ablation that shows *why* you need a replay buffer and a target network. |
| [Experiment 07 — REINFORCE](../experiments/07_reinforce.md) | Policy method: score-function gradient on the same task. Two agents compared: with and without a learned value baseline. The variance difference is plotted explicitly. |
| [Experiment 08 — PPO](../experiments/08_ppo.md) | Actor-critic: combines a policy network with a value critic via GAE. Shows how the combination outperforms pure REINFORCE. |

---

## Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
  Ch. 6 (TD / value methods), Ch. 13 (policy gradient methods),
  Ch. 13.5 (actor-critic). Free PDF at <http://incompleteideas.net/book/the-book.html>
- Williams (1992), *Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning* — the original REINFORCE paper.
- Mnih et al. (2015), *Human-level control through deep reinforcement learning*,
  Nature — DQN at scale.
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms* — PPO.
