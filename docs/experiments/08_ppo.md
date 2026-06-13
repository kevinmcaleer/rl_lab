# Experiment 08 — PPO: a Policy Gradient That Actually Behaves

**Concept:** Actor–critic architecture, Generalised Advantage Estimation (GAE),
and the clipped surrogate objective that keeps updates from blowing up.

---

## Overview

REINFORCE (Experiment 07) works, but it is **noisy and slow**: each update uses
a single episode's raw return as the advantage estimate, which has very high
variance.  Proximal Policy Optimisation (PPO) fixes this with three additions:

1. **An actor–critic architecture** — a value network (the critic) turns raw
   returns into *advantages*, dramatically cutting gradient variance.

2. **GAE (Generalised Advantage Estimation)** — blends one-step TD and
   Monte-Carlo advantages through a tunable `lambda` parameter, trading a small
   amount of bias for much lower variance.

3. **A clipped surrogate objective** — instead of letting the policy update as
   far as gradient descent wants, PPO clips the probability-ratio `r_t` so the
   new policy can never stray too far from the one that collected the data.

Together these make PPO the **default on-policy baseline** for robotics RL.
If you are unsure which algorithm to try first, PPO is almost always the right
answer.

---

## The maths (just enough to read the code)

### Actor–critic advantage

The critic estimates the value of each state `V(s)`.  The advantage of taking
action `a` in state `s` is how much better the actual outcome was compared to
what the critic predicted:

```
A_t  =  r_t + gamma * V(s_{t+1}) * (1 - done)  -  V(s_t)
     =  TD residual  (one-step advantage)
```

Because `A_t` depends on the critic — a learned quantity, not the raw return —
its variance is much smaller than REINFORCE's Monte-Carlo `G_t`.

### GAE(lambda) — the variance knob

A single TD step is low-variance but biased if the critic is not perfect.
GAE smooths that bias away by accumulating a discounted sum of TD residuals:

```
delta_t  =  r_t + gamma * V(s_{t+1}) * (1 - done_t)  -  V(s_t)
A_t      =  delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
```

- `lambda = 0` → pure one-step TD (low variance, more bias).
- `lambda = 1` → Monte-Carlo return (unbiased, high variance).
- `lambda = 0.95` (the default) → a sweet spot that works well in practice.

### The clipped surrogate objective

Collect a rollout with the *current* ("old") policy `pi_old`, then perform
multiple SGD updates on that same rollout using the *updated* ("new") policy
`pi_new`.  The probability ratio measures how much the policy has changed:

```
r_t(theta)  =  pi_new(a_t | s_t) / pi_old(a_t | s_t)
```

Without any constraint, a large gradient step could move `pi_new` so far that
the rollout data is no longer representative — and learning collapses.  PPO
prevents this with a clip:

```
L_CLIP = E[ min( r_t * A_t ,  clip(r_t, 1-eps, 1+eps) * A_t ) ]
```

The `min` of the two terms ensures:

- When `A_t > 0` (a good action): credit is capped once `r_t > 1 + eps`.
  We cannot make the policy infinitely prefer this action just because one
  rollout went well.
- When `A_t < 0` (a bad action): the penalty is capped once `r_t < 1 - eps`.
  A single bad observation cannot drive the probability to zero.

`eps` is the `clip_range` hyper-parameter.  This experiment sweeps it so you
can see its effect directly.

---

## What this experiment does

The script runs two sweeps on `BuddyJrReachDiscrete-v0` (the 4-DOF arm with
`Discrete(9)` actions — same env as DQN and REINFORCE):

### Sweep A — clip-range (eps)

| `clip_range` | Effect |
|---|---|
| `0.05` | Very conservative: each update barely moves the policy.  Learning is slow but stable. |
| `0.2` | SB3 / PPO default.  Well-tuned balance between stability and speed. |
| `0.6` | Most samples are never clipped.  Policy can take large steps — may become unstable. |

All other hypers are fixed: `n_epochs=10`, `n_steps=512`.

### Sweep B — epochs per rollout (n_epochs)

| `n_epochs` | Effect |
|---|---|
| `2` | Nearly one-shot SGD.  Data is not fully exploited; sample-efficient? No. |
| `10` | PPO default.  Good balance of data reuse and policy drift. |
| `20` | Aggressive reuse.  Useful data exhausted after a few epochs; later epochs may overfit. |

All other hypers are fixed: `clip_range=0.2`, `n_steps=512`.

### Plots saved

After the full run (not `--quick`) four PNG files are written to
`experiments/_outputs/08_ppo/`:

| File | What it shows |
|---|---|
| `clip_range_sweep_return.png` | Smoothed episode-return curves for each clip value. |
| `clip_range_sweep_success.png` | Smoothed success-rate curves for each clip value. |
| `epochs_sweep_return.png` | Smoothed episode-return curves for each epoch count. |
| `epochs_sweep_success.png` | Smoothed success-rate curves for each epoch count. |

---

## How to run

### Minimal (terminal only, no viewer)

```bash
python experiments/08_ppo.py
```

### With live 3-D Foxglove visualisation

After training completes, the script streams three deterministic rollouts using
the best-performing clip-range run to Foxglove Studio.

Start the Foxglove desktop app, open a new connection to `ws://localhost:8765`,
and then:

```bash
python experiments/08_ppo.py --render foxglove
```

The script prints a `foxglove://` URL you can click to open the connection
automatically.  For setup instructions see
[`docs/getting_started/foxglove_setup.md`](../getting_started/foxglove_setup.md).

### Quick smoke-test (CI / low-resource machine)

```bash
python experiments/08_ppo.py --quick
```

Runs a tiny budget (512 steps per run, 2 clip values, 2 epoch values) with no
plots and no Foxglove.  Finishes in a few seconds on CPU.

### Headless (no plots, full training)

```bash
python experiments/08_ppo.py --no-plot
```

### Full option reference

```
usage: 08_ppo [--quick] [--render {foxglove}] [--seed SEED] [--no-plot]

  --quick            Tiny budget, no plots, no Foxglove (CI mode).
  --render foxglove  Stream post-training rollouts to Foxglove.
  --seed SEED        Global RNG seed (default: 0).
  --no-plot          Skip saving matplotlib figures.
```

---

## What to look for

### In the clip-range plot

| What you see | What it means |
|---|---|
| `clip=0.05` learns slowly or plateaus early | Updates are so small the policy barely explores. |
| `clip=0.2` reaches the best final return | The default works well: updates are neither too tiny nor too large. |
| `clip=0.6` is fast at first, then unstable or collapses | Large steps violate the proximal guarantee; the policy overshoots and has to recover. |

A key diagnostic is the **clip fraction** — the fraction of sampled transitions
whose ratio `r_t` was actually clipped.  With `eps=0.05` nearly everything gets
clipped, meaning the surrogate is almost never tight.  With `eps=0.6` almost
nothing is clipped: the clip provides no constraint at all.

### In the epochs plot

| What you see | What it means |
|---|---|
| `n_epochs=2` learns more slowly per wall-clock step | Each rollout is used for only two gradient steps; data is wasted. |
| `n_epochs=10` reaches good performance | Default; the clip keeps later epochs from overfitting. |
| `n_epochs=20` sometimes matches 10, sometimes degrades | Once the ratio drifts far enough that everything is clipped, extra epochs add noise. |

### In the Foxglove rollout

After training the best policy streams three deterministic episodes.  Watch for:

- The arm moving directly and deliberately toward the target — the learned
  value function now guides the actor efficiently.
- Smooth trajectories compared to REINFORCE's twitchy rollouts from Experiment
  07 (which had much higher advantage variance).
- The distance metric in the Foxglove panel dropping consistently toward zero.

---

## Expected results

On a typical CPU run (25 000 steps per configuration):

- The `clip=0.2` run should achieve a mean episode return noticeably higher
  than `clip=0.05` and at least as high as `clip=0.6`.
- All three clip runs should out-perform the hand-coded REINFORCE from
  Experiment 07 in both final return and learning speed.
- The `n_epochs=10` run should roughly match or exceed `n_epochs=20`, which
  in turn should beat `n_epochs=2`.

Exact numbers depend on the random seed and your machine.  The curriculum
cares more that you can **read the curves and explain the differences** than
that you hit a specific number.

---

## PPO vs. REINFORCE: a direct comparison

| Property | REINFORCE (Exp 07) | PPO (Exp 08) |
|---|---|---|
| Advantage estimate | Raw Monte-Carlo return `G_t` | GAE(lambda) from critic |
| Gradient variance | High — one episode at a time | Low — TD + EMA over rollout |
| Update trust region | None — can overshoot | Clipped ratio; stays proximal |
| Data reuse | One gradient step per episode | Multiple epochs per rollout |
| Sample efficiency | Low | Moderate (on-policy) |
| Stability | Fragile for large `lr` | Robust across a wide `lr` range |

The key sentence: **PPO = policy gradients + a critic + a trust region you can
tune.**

---

## Implementation notes

This experiment uses the **SB3 PPO** implementation via the lab's
`make_algorithm("ppo", env, ...)` registry, not the from-scratch `PPOMin`
taught in the `ppo_min` algorithm file.  Reasons:

1. SB3 PPO is publication-quality (used in real research); using it here shows
   you what the algorithm looks like in practice, not just in a teaching stub.
2. The from-scratch `PPOMin` is still available as `make_algorithm("ppo_min", ...)`;
   compare them for a deeper study.

The experiment uses `BuddyJrReachDiscrete-v0` (`Discrete(9)` actions) so it
is directly comparable to Experiments 05 (DQN) and 07 (REINFORCE).  PPO works
equally well on the continuous `BuddyJrReach-v0` env — Experiment 09 explores
that variant.

Plots are generated with matplotlib on the **Agg backend** (non-interactive),
so the script runs on any server without a display.

---

## Aha takeaway

> *PPO = policy gradients + a critic + a trust region you can tune.  The clip
> range is not magic — it is a budget for how much the policy is allowed to
> change per batch, and you just watched what happens when that budget is too
> tight or too loose.*

After this experiment you understand:

- **Actor–critic**: why having a separate value head reduces gradient variance.
- **GAE**: how to blend TD and Monte-Carlo advantages and why the lambda knob exists.
- **The clipped surrogate**: what the clip *does* mathematically and what too
  small / too large `eps` looks like empirically.
- **Why PPO is the go-to robotics baseline**: it is stable across a wide range
  of hypers and works on both discrete and continuous action spaces.

**Now you understand actor–critic, advantages, and why PPO is the go-to
baseline for robotics RL.**

---

## Files involved

| Path | Role |
|---|---|
| `experiments/08_ppo.py` | The runnable experiment (this script). |
| `rl_lab/algos/sb3_integration.py` | `SB3Algorithm` — wraps SB3 PPO behind the lab's protocol. |
| `rl_lab/algos/policy_gradient/ppo_min.py` | From-scratch PPO for comparison. |
| `rl_lab/train/callbacks.py` | `SimpleCallbackBridge` — adapts our callback for SB3. |
| `rl_lab/env/buddy_jr_reach_env.py` | `BuddyJrReachEnv` — the Gymnasium environment. |
| `rl_lab/env/wrappers.py` | `DiscretizeBuddyJr` — the Discrete(9) action wrapper. |
| `rl_lab/viz/foxglove_bridge.py` | `FoxgloveStreamer` — live WebSocket publisher. |
| `experiments/_outputs/08_ppo/` | Output directory for saved plots. |

---

## What's next

- **Experiment 09 — Continuous PPO** takes everything you learned here and
  applies it to the *continuous* action space (`Box(4,)`) that real SG90
  servos need, adding a smoothness penalty so the arm doesn't buzz.
- **Experiment 10 — SAC** shows the off-policy alternative: far more
  sample-efficient than PPO but restricted to continuous actions.
