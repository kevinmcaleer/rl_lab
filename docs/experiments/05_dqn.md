# Experiment 05 — DQN: Replace the Table with a Network

| | |
|---|---|
| **Concept** | Function approximation · experience replay · target network |
| **Env** | `BuddyJrReachDiscrete-v0`  (obs `Box(17,)`, action `Discrete(9)`) |
| **Algorithm** | From-scratch DQN (`rl_lab.algos.value_based.dqn.DQN`) |
| **Script** | `experiments/05_dqn.py` |
| **Output** | `experiments/_outputs/05_dqn/learning_curves.png` |

---

## The problem we solved last time — and why it broke

In Experiment 04 you ran tabular Q-learning with higher resolution (more bins)
and unlocked the fourth joint. The Q-table exploded in size — millions of cells,
almost none of them ever visited. Training stalled not because the algorithm was
wrong but because *you can't store one value per state when there are infinitely
many states*.

That is the **curse of dimensionality**, and it is the exact wall that led
researchers to replace tables with neural networks.

---

## The key idea: a network *generalises* across states

A Q-table maps **each discrete state** to Q-values independently — it learns
nothing about neighbouring states. A neural network instead learns a *function*
`Q(s, a; θ)` that **shares parameters across all states**. Once it has seen a
state near the goal, it can infer the Q-value of a similar nearby state it has
never visited, because the function is smooth.

This is called **function approximation**: we approximate the true Q-function
with a parameterised model (here, a small MLP) instead of a table.

The Q-network for `BuddyJrReachDiscrete-v0` has:

- **Input layer**: 17 neurons (the full obs vector — 4 joint angles + 4
  velocities + 3 ee pos + 3 target pos + 3 tip-to-target vector)
- **Two hidden layers**: 64 ReLU neurons each
- **Output layer**: 9 linear neurons, one per discrete action (±jog on each
  joint, plus a no-op)

The training rule is identical to Q-learning:

```
target  y = r + γ · max_a' Q_target(s', a') · (1 - done)
loss    L = ( Q_online(s, a) − y )²        MSE over a minibatch
```

But because the network generalises, this regression has to be handled
carefully. Two tricks are essential.

---

## The two stabilising tricks

### 1 — Experience replay

Consecutive environment steps are **highly correlated**: step *t* and step
*t+1* share the same episode, similar joint angles, and strongly related
observations. Gradient descent assumes data is approximately i.i.d.; training on
correlated pairs causes the gradient to point in a systematically biased
direction and the network can oscillate or diverge.

**The fix:** store every transition `(s, a, r, s', done)` in a **ring buffer**
(the *replay buffer*) and train on a **random minibatch** drawn from it. Random
sampling breaks the correlation. It also lets each transition contribute to
*multiple* gradient updates — expensive experience is reused.

```
Replay buffer (ring buffer, capacity = 50 000 transitions)
┌──────────────────────────────────────────────────────────┐
│  (s₀, a₀, r₀, s₁, d₀)   ← oldest (overwritten when full)│
│  (s₁, a₁, r₁, s₂, d₁)                                   │
│   …                                                      │
│  (sₙ, aₙ, rₙ, sₙ₊₁, dₙ) ← most recent                  │
└──────────────────────────────────────────────────────────┘
        ↓  sample 64 uniformly at random each gradient step
        →  decorrelated minibatch → stable gradient estimate
```

### 2 — Target network

If we compute the bootstrap target `y = r + γ max_a' Q(s', a')` using the
*same* network we are updating, the regression target moves every gradient step.
We are literally chasing a goal that shifts each time we take a step toward it —
like trying to hit a moving target. This often leads to oscillation or
divergence.

**The fix:** maintain a **second copy** of the Q-network (`Q_target`) whose
weights are frozen and synced to the online network only every `target_sync`
gradient steps (500 in the healthy run). The target computed against `Q_target`
is stable for 500 steps at a time, giving the regression something solid to
converge toward.

---

## The ablation: watch both tricks fail

`experiments/05_dqn.py` trains **two** DQN agents back-to-back:

| Condition | `buffer_size` | `target_sync` | What it shows |
|---|---|---|---|
| **Healthy** | 50 000 | 500 | Both tricks active → stable, rising curve |
| **Collapsed** | 256 | 1 | Both tricks broken → oscillating / flat curve |

The **collapsed** condition breaks both tricks at once:

- A buffer of 256 holds only ~1–2 episodes of transitions. Random sampling from
  it still draws from a highly correlated window — effectively the same as
  learning directly from the last few steps.
- `target_sync=1` syncs the target every gradient step, meaning `Q_target` and
  `Q_online` are always identical. There is no stable target to regress toward.

The resulting learning curve either oscillates wildly or collapses flat —
instability you can *see* rather than just read about.

---

## How to run

### Full training (≈ 50 000 steps per condition, saves plots)

```bash
python experiments/05_dqn.py
```

Expected wall-clock time: **2–5 minutes** on Apple Silicon M-series (CPU-only).

### Quick smoke-test (CI mode, a few seconds)

```bash
python experiments/05_dqn.py --quick
```

Useful to confirm the script is importable and runs without errors before a
long training session.

### Live 3D visualisation in Foxglove

```bash
# Start the Foxglove desktop app, then:
python experiments/05_dqn.py --render foxglove
```

After training completes the healthy DQN's greedy policy runs 5 rollouts
streamed live to Foxglove:

- The 3D arm moves in real time.
- The goal sphere turns green when the tip is within tolerance.
- The tip-to-target line shrinks as the policy improves.
- Live metrics (distance, reward, success rate) appear in the data panel.

Connect Foxglove to `ws://localhost:8765` (the default Foxglove WebSocket port).

### Additional flags

```
--seed N      reproducible run with seed N (default: 0)
--no-plot     skip saving the matplotlib figure
```

---

## What to watch for

### In the saved plot (`experiments/_outputs/05_dqn/learning_curves.png`)

The figure has two panels:

**Left — Episode Return (smoothed)**
The healthy (blue) curve rises steadily over 50 000 steps. The collapsed (red)
curve stays low, oscillates, or temporarily rises then collapses. The gap
between blue and red is the combined value of the two stabilising tricks.

**Right — Success Rate %**
The healthy agent's success rate climbs toward a reliable non-zero value. The
collapsed agent stays near zero or fluctuates with no clear upward trend. A
"success" means the camera tip reached within 2 cm of the target in a single
episode.

### In Foxglove (if `--render foxglove`)

After training, 5 greedy episodes play out live. You should see the arm:

1. Start from a random configuration (joints noisy around a resting pose).
2. Jog its joints in deliberate increments rather than randomly flailing.
3. Drive the tip progressively closer to the goal (watch the line shrink).
4. Sometimes reach the goal and terminate early (the greedy policy can succeed
   even before the full 200-step budget is used).

If the arm still seems fairly random by the end of a 50 000-step run it is
likely because the task is hard under a discrete action space — see the *Next
steps* section.

---

## Aha takeaway

> *A neural network can represent Q-values for infinitely many states — but only
> if you decorrelate the training data (replay buffer) and stabilise the
> regression target (target network). Remove either trick and the learning curve
> collapses.*

**Now you understand:**

- **Function approximation** — why neural networks replace tables.
- **Experience replay** — how a ring buffer breaks temporal correlations.
- **Target networks** — how a frozen copy prevents chasing a moving goal.
- **Why "deep" Q-learning needs special care** — gradient descent on RL data is
  fragile, and the two tricks above are the minimal patch.

These are the ideas that made DeepMind's 2015 Atari paper work (Mnih et al.,
*Human-level control through deep reinforcement learning*, Nature 2015) and that
every subsequent deep value-based algorithm builds on.

---

## Concept connections

| Concept | First introduced | This experiment |
|---|---|---|
| Q-values / Bellman update | Exp 03 | Now approximated by a neural network |
| Curse of dimensionality | Exp 04 | Solved by function approximation |
| Replay buffer | **Exp 05** | Breaks temporal correlation |
| Target network | **Exp 05** | Stabilises the regression target |
| Generalisation / moving targets | Exp 06 | Next: what happens when the goal moves |

---

## Next steps

Once you understand the healthy vs. collapsed contrast, try these self-directed
experiments by editing `experiments/05_dqn.py`:

1. **Find the buffer threshold.** Try `buffer_size` values of 512, 1 024, 4 096,
   16 384. At what size does the collapse disappear?
2. **Find the target-sync threshold.** Keep `buffer_size=50_000` fixed and sweep
   `target_sync` in {1, 10, 50, 200, 1000}. How infrequent does syncing need to
   be to stay stable?
3. **Turn off only one trick at a time.** Use `buffer_size=256, target_sync=500`
   (broken replay only) and then `buffer_size=50_000, target_sync=1` (broken
   target only). Which trick matters more for this environment?
4. **Add Huber loss.** Replace the MSE loss in
   `rl_lab/algos/value_based/dqn.py` with `torch.nn.HuberLoss()`. Does it
   reduce the wild spikes in the collapsed curve?
5. **Move to Experiment 06** — use the same healthy DQN but change `reset()` to
   randomise the target position every episode. Does the policy generalise?

---

## Code pointers

| File | What to look at |
|---|---|
| `rl_lab/algos/value_based/dqn.py` | `ReplayBuffer`, `QNetwork`, `DQN._learn()` |
| `rl_lab/algos/registry.py` | `make_algorithm("dqn", env, ...)` |
| `rl_lab/env/registration.py` | `BuddyJrReachDiscrete-v0` registration |
| `rl_lab/env/wrappers.py` | `DiscretizeBuddyJr` — how 9 discrete actions map to joint jogs |
| `rl_lab/viz/foxglove_bridge.py` | `FoxgloveStreamer.publish(...)` — live 3D streaming |
| `experiments/05_dqn.py` | `_train_condition()`, `run()`, `main()` |
