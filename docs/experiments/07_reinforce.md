# Experiment 07 — REINFORCE from Scratch: Policy Gradients and Variance

**Part IV — Policy-based RL**

| | |
|---|---|
| **Algorithm** | REINFORCE (Williams, 1992) — Monte-Carlo policy gradient |
| **Environment** | `BuddyJrReachDiscrete-v0` — Discrete(9) jog actions, Box(17) obs |
| **Script** | `experiments/07_reinforce.py` |
| **Prerequisite** | Experiment 06 (or familiarity with the DQN / discrete-action env) |
| **Training time** | ~4–8 min on CPU (M1 Pro) |

---

## Concept

So far in the curriculum you have learned *value-based* RL: estimate Q(s, a) and
act greedily on it. This experiment introduces the second great family — **policy
gradient** methods — where you skip the value function entirely and learn the
*policy* directly.

The central question is: **how do you push a neural network's parameters in the
direction of "take actions that lead to more reward" without knowing the gradient
of the environment?**

The answer is a beautiful identity called the **score-function trick** (also known
as the log-derivative trick), which is the mathematical engine inside REINFORCE,
PPO, and TRPO.

---

## The mathematics

### The objective

We want to maximise the expected return:

```
J(θ) = E_τ~πθ [ Σ_t r_t ]
```

where `τ` is a trajectory (episode) sampled by following policy `π(a | s; θ)`.

### The score-function gradient estimator

Taking the gradient of `J` with respect to `θ` looks impossible at first: how
do you differentiate through the environment dynamics? The trick is the
log-derivative identity:

```
∇_θ π(a | s; θ) = π(a | s; θ) · ∇_θ log π(a | s; θ)
```

This lets you express the gradient of the *expectation* as an *expectation of
a gradient* — something you can estimate by sampling:

```
∇_θ J(θ) = E [ Σ_t  ∇_θ log π(a_t | s_t; θ)  ·  G_t ]
```

where `G_t = Σ_{k≥t} γ^(k-t) r_k` is the discounted return from step `t`.

In English: **increase the log-probability of actions that were followed by a
large return; decrease it for actions followed by a small return.**

Because optimisers minimise (not maximise), the training loss is the negative:

```
L(θ) = − (1/N) · Σ_t  log π(a_t | s_t; θ)  ·  A_t
```

where `A_t` is the "advantage" — what REINFORCE actually multiplies the log-prob
by.

---

## The variance problem and baselines

### Why raw REINFORCE is noisy

`G_t` can be large even for mediocre actions if the episode happened to go well
by chance.  The gradient estimator is **unbiased** (it points in the right
direction on average) but has **high variance** — individual gradient steps can
push the policy in the wrong direction, making training slow and erratic.

This is the "twitchy" behaviour you will see in Foxglove: the arm moves
differently every time because the policy is not confident about any action.

### The baseline fix

You can subtract any quantity `b(s_t)` that does **not depend on the action**
from `G_t` without introducing bias (it cancels in expectation):

```
A_t = G_t − b(s_t)
```

A good baseline is the **state-value function** `V(s_t)` — the expected return
from state `s_t` under the current policy.  Subtracting it leaves the
*advantage*: "was this action better or worse than average from this state?"

In this experiment a second MLP (the "critic") learns `V(s_t)` by regression
onto the observed returns `G_t`.  Because `V(s_t)` captures most of the
"background level" of return at each state, `A_t = G_t − V(s_t)` has much
smaller variance than raw `G_t`.

**This is the key insight behind every modern actor-critic method.** PPO, A3C,
and SAC all use variants of this advantage estimation.

---

## What the experiment runs

Two REINFORCE agents are trained on `BuddyJrReachDiscrete-v0` for 80,000 env
steps each:

| Agent | Advantage signal | Critic? |
|-------|-----------------|--------|
| `no_baseline` | `G_t − mean(G)` (batch mean; very weak) | No |
| `with_baseline` | `G_t − V(s_t)` (learned value network) | Yes — separate MLP |

Both agents use identical hyperparameters for everything else:
`lr=5e-4`, `γ=0.99`, hidden layers `(64, 64)`, `max_steps=150`.

After training, four plots are saved under `experiments/_outputs/07_reinforce/`.

---

## How to run

### Full run (recommended the first time)

```bash
python experiments/07_reinforce.py
```

Trains both agents (~4–8 min), then saves four plots.

### With Foxglove (live 3D viewer)

```bash
# In one terminal — leave it running:
make foxglove

# In another terminal:
python experiments/07_reinforce.py --render foxglove
```

After training, the script streams five greedy rollouts of the *with-baseline*
policy to Foxglove.  Open `ws://localhost:8765` in Foxglove Studio to watch.

To compare the two agents visually, edit the last line of `_foxglove_demo` to
pass `algo_no` instead of `algo_wb` and re-run — the motion will be noticeably
more erratic.

### Quick / CI smoke test

```bash
python experiments/07_reinforce.py --quick
```

Runs 800 steps per agent (a few seconds), prints metrics, skips plots.

### Options

```
--quick          Tiny budget, no plots, no Foxglove.  Completes in < 5 s.
--render foxglove  Stream the trained policy to Foxglove after training.
--seed INT       Global RNG seed (default: 0).
--no-plot        Skip matplotlib plots (useful on headless servers).
```

---

## Expected results

### Learning curves (`learning_curves.png`)

The **no-baseline** plot (red) shows a very jagged episode-return curve — the
raw return varies wildly between episodes because the gradient signal is noisy.
The EMA-smoothed line climbs slowly.

The **with-baseline** plot (blue) is visibly less noisy even in the raw signal,
and the smoothed curve climbs earlier and more steeply.

Both agents' raw curves look messy — that is normal for REINFORCE. The key is
the *difference* between them, not the absolute noisiness.

### Advantage variance (`advantage_variance.png`)

This is the quantitative proof that the baseline helps.  Look for:

- **No baseline**: the advantage variance stays high throughout (the raw return
  carries a lot of "background level" noise that could be removed).
- **With baseline**: advantage variance is consistently lower — the critic
  learned to absorb most of the background return, leaving a cleaner signal.

The `var_ratio` printed at the end (`with_baseline / no_baseline`) should be
less than 1.0 after a full run.  Typical values: 0.3–0.7.

### Success rate / deliberateness (`success_rate.png`)

Early in training, both agents move randomly (high entropy = "twitchy").  As the
policy sharpens onto good actions, the success rate rises.  The with-baseline
agent typically rises faster and reaches a higher plateau within the 80 k step
budget.

In Foxglove you will see:

- **No-baseline policy**: the arm jogs in many directions before (if ever) landing
  near the target.  Joint velocities change direction frequently.
- **With-baseline policy**: after the policy converges, joint movements are more
  purposeful — each step is a deliberate jog toward the target rather than
  semi-random exploration.

### Console output

At the end of a full run you should see something like:

```
[07] Summary:
    No baseline   — return=-18.4  SR=0.021  avg_adv_var=142.3
    With baseline — return=-12.1  SR=0.087  avg_adv_var=54.7
    Variance ratio (with/no): 0.384  (baseline reduces variance)
```

Exact numbers vary with seed and training budget.

---

## Common questions

**Why is the success rate still low after 80 k steps?**

REINFORCE is a high-variance algorithm — it is not designed for sample
efficiency.  The point of this experiment is the *comparison* between the
agents, not achieving a high final success rate.  Experiment 08 (PPO) will
dramatically improve both learning speed and final performance using the same
core idea with several important engineering fixes.

**Can I train longer?**

Yes.  Set `_FULL_STEPS = 200_000` at the top of the script and re-run.  The
gap between the two agents tends to narrow as training continues — given enough
data even the no-baseline agent converges.  The baseline's advantage is most
visible in the **early** learning curve.

**Why does `var_ratio` sometimes exceed 1.0?**

With a short training budget or an unlucky seed, the value network may not have
learned a good `V(s)` yet — its predictions are too noisy to help.  This is
another teaching point: a *bad* baseline can increase variance.  Try `--seed 1`
or `--seed 2` and see whether the ordering holds.

**What is the difference between this and PPO?**

PPO (Experiment 08) adds three things on top of this experiment's with-baseline
REINFORCE:

1. **GAE (Generalised Advantage Estimation)** — a lower-variance advantage
   that blends Monte-Carlo returns with the critic's one-step TD error.
2. **Multiple epochs of minibatch updates** — reusing each rollout rather than
   discarding it after one gradient step.
3. **The clipped surrogate objective** — a constraint that prevents individual
   updates from moving the policy too far, avoiding sudden collapses.

If you understand why the baseline helps here, you already understand *why*
each of PPO's additions matters.

---

## Code highlights

The core implementation lives in `rl_lab/algos/policy_gradient/reinforce.py`.
Key lines to study:

```python
# --- Advantage (the difference between the two agents) ---
if self.use_baseline and self.value is not None:
    # detach() is critical: the baseline must not receive the policy gradient.
    advantages_t = returns_t - values_t.detach()
else:
    # Batch-mean baseline — unbiased but weak variance reduction.
    advantages_t = returns_t - returns_t.mean()

# --- Policy gradient loss (the score-function estimator) ---
dist = self._policy_dist(obs_t)       # Categorical distribution over actions
log_probs = dist.log_prob(act_t)      # log π(a_t | s_t; θ)
policy_loss = -(log_probs * advantages_t).sum()   # maximise → minimise negative
```

Notice `values_t.detach()` — the `.detach()` call is what makes the baseline
unbiased.  Without it, gradients would flow back through the value network
when updating the policy, entangling the two objectives.

---

## Aha takeaway

**You can optimise behaviour directly — and the main battle is taming gradient
variance.**

Now you understand:

- **Policy gradients** — the score-function / log-derivative trick that turns
  a hard expectation into a sampleable gradient.
- **Monte-Carlo returns** — why collecting whole episodes is necessary for
  REINFORCE (and why it is slow).
- **Baselines and advantages** — why subtracting `V(s)` reduces variance without
  introducing bias, and why this matters for learning speed.
- **Twitchy vs. deliberate motion** — how policy entropy translates to observable
  arm behaviour in the viewer.

These concepts are the foundation under PPO (Experiment 08), and they reappear
in every actor-critic method you will ever use.

---

<!-- nav-footer -->
← Previous: [Generalisation](06_generalisation.md) &nbsp;|&nbsp; [All experiments](../experiments.md) &nbsp;|&nbsp; Next: [PPO](08_ppo.md) →
