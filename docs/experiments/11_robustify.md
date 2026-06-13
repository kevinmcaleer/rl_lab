# Experiment 11 — Closing the sim-to-real gap

> Part VI — Sim-to-real: deploy toward the real Buddy Jr
> Script: [`experiments/11_robustify.py`](https://github.com/kevinmcaleer/rl_lab/blob/main/experiments/11_robustify.py)

A policy that is *perfect* in a pristine simulator routinely **fails on real
hardware**. This experiment shows you why — and hands you the standard toolkit
for fixing it, **before** you ever connect a servo.

---

## Concept

The **reality gap** is the mismatch between your clean simulator and the messy
real world. In sim, a commanded joint angle lands exactly where you asked,
instantly, and your sensors report the ground truth. On real Buddy Jr — four
SG90 hobby servos driven by a PCA9685 over I2C from a Raspberry Pi — *none* of
that is true.

The standard way to cross the gap is **domain randomisation**: during training
you deliberately perturb the environment so the policy learns behaviour that is
*robust* to the perturbations it will meet on hardware. Instead of memorising
one frictionless trajectory, it learns a strategy that still works when the arm
twitches, lags and lies to it.

This is the same idea you first met in Experiment 6 (randomising the *target*
to force generalisation). Here we randomise the **dynamics and sensing** to
force *sim-to-real* robustness.

## Objective

After this experiment you should be able to:

- name the four real-hardware imperfections that make a sim policy fail, and
- wrap any Gymnasium env to model them, and
- *measure* the payoff: a success-rate curve showing the robustified policy
  surviving noise that flattens the naive one.

---

## The toolkit: every knob maps to a real defect

The experiment wraps the continuous reach env in
[`DomainRandomization`](https://github.com/kevinmcaleer/rl_lab/blob/main/rl_lab/env/wrappers.py). Each knob models exactly
one thing that goes wrong on the real arm:

| `DomainRandomization` knob | Real-hardware analogue | Why it bites |
|---|---|---|
| `action_noise_std` | **SG90 command jitter** — dead-band, gear backlash and PWM quantisation (~0.5°) mean the servo never lands exactly on the commanded angle. | A policy tuned to land *exactly* on a target overshoots and oscillates when every command is a little off. |
| `obs_noise_std` | **Sensor noise** — the camera-based target detector (and any joint feedback) is a noisy *estimate*, not the clean ground truth the sim hands you. | The policy must act on a fuzzy state, not the exact one it trained on. |
| `action_rate_limit` | **PCA9685 update rate + control latency** — a real servo cannot teleport between angles each tick; it slews at a finite speed, so a command can only change so much per control step. | A policy that relies on instant, large corrections will lag reality and chase its own tail. |
| `randomize_radius` | **Task / calibration variation** — link lengths, mounting and the reachable target shell are never exactly what the CAD model says. | Forces the policy to generalise across slightly different workspaces. |

Separately — and crucially — **safe-range clamping** is *always on*, in two
places:

1. The env clips every action to `[-1, 1]` and every joint target to
   `JOINT_LIMITS` (±90°), so the policy can never command an impossible pose.
2. On hardware,
   [`radians_to_servo_degrees`](https://github.com/kevinmcaleer/rl_lab/blob/main/rl_lab/robot/buddy_jr.py) further clamps
   to the SG90's physical `[0, 180]°` range (`servo_deg = degrees(θ) + 90`).

Clamping is not optional cosmetics: it is what stops a confused policy from
stripping a gear. Randomisation makes the policy *robust*; clamping makes it
*safe*.

---

## Build & run

The script trains **two** policies on the *same* reach task and compares them:

1. a **naive** policy, trained only on the pristine, noise-free sim (the
   Experiment 10 setup), and
2. a **robustified** policy, trained inside `DomainRandomization` at a moderate
   severity so it meets jitter, latency and sensor noise during learning.

It then evaluates **both** across a sweep of randomisation severities and plots
the success-rate comparison.

```bash
# Full run: train both policies, sweep, and save the comparison plot.
python experiments/11_robustify.py

# Watch the robustified policy cope with noise live in Foxglove.
python experiments/11_robustify.py --render foxglove

# Fast smoke test (tiny budget, no plots, no Foxglove) — the CI path.
python experiments/11_robustify.py --quick

# Reproducible run on a chosen seed, with the plot suppressed.
python experiments/11_robustify.py --seed 7 --no-plot
```

You can also call it from Python and inspect the returned metrics dict. Because
the filename starts with a digit it is not a valid identifier, so load it by
path with `importlib`:

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "exp11", Path("experiments/11_robustify.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

metrics = mod.run(quick=False, seed=0)
print(metrics["robust_success_worst"], metrics["naive_success_worst"])
```

Running it as a script (`python experiments/11_robustify.py`) is the simplest
path and needs no import gymnastics.

### What the knobs are set to

A single **severity** number (0.0 → ~0.4) scales the knobs together so the sweep
reads as "increasingly unkind reality":

- `action_noise_std = severity` (SG90 jitter),
- `obs_noise_std = 0.5 × severity` (sensor noise),
- `action_rate_limit = max(0.1, 1 − 1.5 × severity)` (tighter slew at high severity),
- `randomize_radius = (0.04, 0.155)` (workspace variation).

Severity `0.0` is the pristine sim the naive policy was trained in.

---

## Watch for

- **The naive policy falls off a cliff.** On the plot
  (`experiments/_outputs/11_robustify/robustness_comparison.png`) its success
  rate is high at severity 0 and collapses as jitter/latency/noise rise — it
  *only* knew the frictionless world.
- **The robustified policy holds up.** It may give up a little success in the
  pristine sim (a small "robustness tax") but stays far more accurate as
  severity climbs. The gap between the two curves *is* the value of domain
  randomisation.
- **In Foxglove (`--render foxglove`)**, the robustified arm under heavy noise
  still settles onto the target instead of buzzing or overshooting — the
  motion you actually want on SG90 gears.

Headline numbers in the returned dict / printed summary:

| Metric | Meaning |
|---|---|
| `naive_success_clean` | naive policy success in the pristine sim (should be high) |
| `naive_success_worst` | naive policy at the harshest severity (should crater) |
| `robust_success_worst` | robustified policy at the harshest severity |
| `robustness_gain_worst` | `robust_success_worst − naive_success_worst` (the payoff) |

### Expected result

The exact figures depend on the training budget, seed and CPU, but the
**shape** is the lesson and is reproducible: the naive curve drops steeply with
severity while the robustified curve degrades gently, so `robustness_gain_worst`
is clearly **positive**. If you crank the full (non-`--quick`) budget up, the
separation becomes dramatic. In `--quick` mode the budget is deliberately tiny
(it is only a smoke test), so treat those numbers as "did it run", not "did it
learn".

---

## Aha takeaway

> *A policy that's perfect in a pristine sim will fail on real servos —
> robustness must be **trained in**, not hoped for.*

**Now you understand the reality gap and the standard toolkit — domain
randomisation (jitter, sensor noise, rate/latency limits) plus safe-range
clamping — for crossing it.** That toolkit is exactly what makes Experiment 12
(deploying to the real arm) have a fighting chance.

---

## Where this fits

- **Builds on:** Experiment 10 (the continuous SAC reach policy) and Experiment
  6 (domain randomisation for *generalisation*).
- **Leads to:** Experiment 12 — export the robustified policy and run it on the
  real Raspberry Pi + PCA9685 hardware, watching the real arm and the sim in
  the same Foxglove layout.

---

<!-- nav-footer -->
← Previous: [SAC + aim](10_sac_aim.md) &nbsp;|&nbsp; [All experiments](../experiments.md) &nbsp;|&nbsp; Next: [Deploy](12_deploy.md) →
