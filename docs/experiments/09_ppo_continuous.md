# Experiment 09 — Continuous PPO: smooth servo commands

**Part V — Continuous control (the real arm's action space)**

---

## Concept

Every RL algorithm so far used a **discrete** action space: the agent chose from
a small menu of pre-defined joint nudges.  Real servos — including the SG90s on
Buddy Jr — receive **continuous angle commands**.  If you want to deploy a policy
to hardware you must eventually work in the continuous domain.

This experiment introduces two ideas that appear together whenever continuous
control is done for real hardware:

1. **Continuous action spaces and action scaling.**  PPO's Gaussian policy outputs
   real-valued numbers; those numbers must be scaled sensibly before they move
   the servo.  In `BuddyJrReach-v0` each action dimension is a delta-angle in
   `[-1, 1]`, scaled internally by `ACTION_SCALE = 0.1 rad/step` before being
   added to the current joint target.  A unit action therefore moves a joint by
   about 5.7 degrees — a safe default that keeps episodes from ending instantly
   on joint-limit violations.

2. **Smoothness / effort penalties.**  A policy that cares only about reaching
   the goal will happily oscillate the arm at full speed around the target — it
   receives the same reward either way.  On a real SG90 this behaviour strips
   gears, drains the power bank, and makes the arm vibrate itself off the bench.
   A **control penalty** `− w · ||a||²` added to the reward every step makes
   large actions costly.  The agent then learns to *approach* the target rather
   than thrash around it.

---

## The two agents

| Agent | `control_weight` | What it learns |
|-------|-----------------|----------------|
| **Buzzy** | 0.0 | Reach the target as fast as possible; no penalty for big or jerky actions. |
| **Smooth** | 0.1 | Reach the target *and* keep actions small; hardware-friendly motion. |

Both agents use identical PPO hyperparameters and train on identical episodes.
The *only* difference is the weight on the action-magnitude penalty in the
reward function.  This isolates the effect of reward design on motion quality.

---

## What you will see

### Training curves

After training, the script saves a two-panel figure to
`experiments/_outputs/09_ppo_continuous/curves.png`:

- **Left panel** — episode return over training for both agents.  Both curves
  should climb, but the smooth agent's curve may be slightly lower because it is
  also paying a control penalty that is not present in the buzzy agent's reward.
  In a well-tuned run the difference in *final* return is small; the difference
  in *behaviour* is large.

- **Right panel** — mean action L2 norm during evaluation.  This measures how
  big the actions are on average across all steps of all evaluation episodes.
  A lower value means calmer, servo-friendly motion.  You should see the smooth
  agent's bar noticeably shorter than the buzzy agent's bar.

### Foxglove viewer (optional)

When you add `--render foxglove` the script streams greedy evaluation rollouts
of the **smooth** agent to Foxglove Studio after training.  Open
`http://localhost:8765` in Foxglove to watch:

- The arm approaching the target (green sphere) with deliberate, controlled
  movements.
- The `distance` metric channel dropping and levelling off near zero once the
  arm reaches the target.
- The joint trajectories in the Time Series panel — you should see smooth
  curves, not high-frequency oscillations.

To compare side by side, re-run the evaluation with `--render foxglove` on the
buzzy agent by temporarily swapping the `render_eval` call in the source, or by
copying the trained model and re-evaluating.  The contrast in motion style is
immediate and unmistakable.

---

## How to run

```bash
# Full training run (saves plot, ~5 minutes on CPU)
python experiments/09_ppo_continuous.py

# Full run with live Foxglove streaming of the smooth agent's evaluation rollouts
python experiments/09_ppo_continuous.py --render foxglove

# Quick smoke test (a few seconds, no plot, used by CI)
python experiments/09_ppo_continuous.py --quick

# Reproducible run with a specific seed
python experiments/09_ppo_continuous.py --seed 42

# Skip saving the plot (useful when running headless and you only want metrics)
python experiments/09_ppo_continuous.py --no-plot
```

---

## Understanding the reward

The `BuddyJrReach-v0` environment uses `reward_mode="dense"` by default, which
computes:

```
reward = −distance(tip, target)
       + success_bonus  (if within tolerance)
       − control_weight · ||action||²
```

Each component plays a role:

| Term | Effect |
|------|--------|
| `−distance` | Pulls the arm toward the target every step (dense shaping). |
| `success_bonus` | Large one-time prize when the tip lands within 2 cm of the goal. |
| `−control_weight · \|\|a\|\|²` | Penalises large actions, encouraging smooth motion. |

With `control_weight=0` the reward is purely distance-minimising.  The policy
discovers that oscillating near the target keeps distance low but never quite
triggers the success bonus — so it buzzes.  With `control_weight=0.1` each
oscillation costs penalty proportional to the square of the action, so the
policy learns to damp the oscillation and hold still.

### The quadratic matters

The penalty is `||a||²`, not `||a||`.  The quadratic norm is *much* more
sensitive to large actions than to small ones.  A single large action costs four
times as much as two half-sized actions (2² = 4 vs 1 + 1 = 2).  This is
intentional: we want to discourage high-frequency, large-amplitude oscillations
without preventing the arm from making rapid corrections when it is far from the
target.

---

## Experiment API

The experiment follows the frozen interface used by all lab experiments:

```python
from experiments.ppo_continuous_09 import run

metrics = run(quick=False, render=None, seed=0)
# metrics is a dict with keys:
#   buzzy_success_rate   — fraction of eval episodes reaching the target
#   smooth_success_rate
#   buzzy_mean_return    — mean undiscounted episode return
#   smooth_mean_return
#   buzzy_mean_action_l2  — mean L2 norm of actions (servo effort proxy)
#   smooth_mean_action_l2
```

The `quick=True` flag reduces training to `512` steps and skips all plotting
and Foxglove output, so the CI smoke test finishes in a few seconds.

---

## Expected results

After 60 000 training steps on a laptop CPU (approximately 3–6 minutes):

| Metric | Buzzy | Smooth |
|--------|-------|--------|
| Success rate | 60–85 % | 50–80 % |
| Mean episode return | higher (no penalty) | slightly lower |
| Mean action L2 | 0.6–1.0 | 0.2–0.5 |

The exact numbers depend on the random seed and hardware.  The important
observation is that **the smooth agent's mean action L2 should be significantly
lower** — even if its success rate is similar.  If both action L2 values are
similar, try increasing `CONTROL_SMOOTH` in the script.

---

## What to try next

1. **Sweep `control_weight`.**  Edit `CONTROL_SMOOTH` in the script and try
   values from `0.0` (buzzy) through `0.01`, `0.05`, `0.1`, `0.3`.  At very
   high values the agent stops reaching the target and just stands still —
   you have over-penalised.  There is a sweet spot; finding it is part of
   reward engineering.

2. **Add a smoothness (jerk) penalty.**  The current penalty is on action
   *magnitude*.  You could add a penalty on the *change* between consecutive
   actions `||a_t − a_{t-1}||²` — this is the `smoothness_term` in
   `rl_lab/env/rewards.py`.  Patch the env to include it and observe whether
   the motion becomes even smoother.

3. **Compare with SAC (Experiment 10).**  SAC is an off-policy algorithm with
   built-in entropy regularisation that encourages diverse, exploratory actions.
   Run both PPO (this experiment) and SAC (Exp 10) with the same control weight
   and compare how many steps each takes to reach a given success rate.

4. **Try the camera-pointing task.**  Switch the env to
   `BuddyJrCameraPoint-v0`, which rewards *aiming* the camera at the target
   (not just placing the tip nearby).  The wrist joint now matters — observe
   how the policy uses all four DOF.

---

## Aha takeaway

> *Continuous control is what real actuators need, and the reward must encode
> "be gentle", not just "be correct".*

Now you understand:

- **Continuous action spaces** — real servos accept a continuous angle; the
  `Box(4)` action space is the natural fit.
- **Action scaling** — raw network outputs `[-1, 1]` are scaled to
  `~0.1 rad/step` so the arm moves at a safe speed.
- **Smoothness / effort penalties** — `−w · ||a||²` is the simplest and most
  widely used form; it directly penalises the behaviour that destroys servo
  hardware.
- **Reward design trade-off** — increasing `control_weight` makes the arm
  gentler but also makes the task harder to solve; finding the right balance
  is a core skill in applied RL.
