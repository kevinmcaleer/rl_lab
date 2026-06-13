# Experiment 06 — Generalisation & Domain Randomisation

**Concept:** An agent can *memorise* a solution or *learn the task*. These two look
the same during training, but diverge the moment you move the target somewhere new.
Domain randomisation — varying the task during training — is the primary tool that
forces genuine learning and is your first concrete step toward sim-to-real transfer.

---

## The core question

Take the DQN agent from Experiment 05 and ask: **what did it actually learn?**

There are two possibilities:

1. **Memorisation** — it learned a fixed trajectory: "always move joints like *this*
   to reach *that one specific point*". Perfect score during training; zero score
   anywhere else.

2. **Generalisation** — it learned the task: "look at the tip-to-target vector in
   the observation, and choose the action that shrinks it". Works on any target,
   because the policy conditions on the target position.

You cannot tell the difference by watching training metrics alone. The only honest
test is evaluation on **targets the agent never saw during training** — the
*generalisation evaluation*.

The difference in success rate between training targets and unseen targets is the
**generalisation gap**. A large gap means memorisation; a small gap means the agent
actually learned the task.

---

## What this experiment does

The script trains two DQN agents side by side:

| Condition | What changes each episode | What the agent must learn |
|---|---|---|
| **Fixed target** | Nothing — the target stays at the same point forever | A single hard-coded trajectory |
| **Random target** | The target is re-sampled at a random reachable position | A general reaching policy driven by the observation |

Both agents see the same observation space (including `obs[14:17]` = normalised
tip-to-target vector) and the same DQN hyperparameters. The only difference is
whether the target moves.

After training, **both agents** are evaluated on the same set of randomly sampled
unseen targets (a fresh set, with a different seed). This reveals the generalisation
gap without any ambiguity.

---

## Domain randomisation

Randomising the target position is the simplest form of **domain randomisation** —
deliberately injecting variation at training time so the policy is forced to learn a
general strategy rather than a memorised one.

The same principle applies to:

- Target position (this experiment)
- Link lengths and masses (Experiment 11 adds servo jitter and control latency)
- Sensor noise and observation perturbations
- Action delays and servo slew limits

The intuition is: *if you train on many variations of a task, your policy must
generalise across them. At test time — including on the real robot — you encounter
a new variation it has not seen. If the training distribution was wide enough, the
policy handles it.*

This is why domain randomisation is called "the first sim-to-real bridge": before
you worry about the camera, the lighting, the servo dead-band, or the link
flexibility of the real arm, you first ensure the policy can handle a different
*target position*. If it cannot handle that, it cannot handle anything real.

---

## What to look for

### Learning curves (training)

Both agents will reach a high training success rate. This is expected — the fixed
agent has an easy job (memorise one trajectory), and the random agent has a harder
job but a denser signal (every episode teaches something about the geometry of the
reaching task).

Watch whether:

- The fixed agent's learning curve rises faster (likely — its problem is simpler).
- The random agent's curve is noisier early on but may eventually match or exceed
  the fixed agent's training success rate.

> The learning curves look similar. The real test is what happens next.

### Generalisation bar chart

This is the key plot: success rate on **unseen** targets, one bar per agent.

The expected result after 40,000 training steps:

- **Fixed-target agent**: low or near-zero success on unseen targets. It learned
  to reach *one point* and the policy has no information to use when the target is
  somewhere different.
- **Random-target agent**: substantially higher success on unseen targets. It
  learned that `obs[14:17]` is the direction to the goal, and that shrinking it is
  the task.

The **generalisation gap** (random success − fixed success) is the headline metric.
A gap of +30 % or more is a clear demonstration of the concept; on a full 40,000
step run you should see 50 % or more.

### Per-episode scatter plot

Each dot is one evaluation episode. You will see:

- **Orange dots (fixed agent)**: scattered high on the distance axis, with little
  variation — the agent applies the same wrong motion pattern regardless of where
  the target is.
- **Blue dots (random agent)**: tightly clustered near the red success threshold
  (2 cm) on many episodes, with occasional failures when the target is in an
  especially awkward part of the workspace.

### Foxglove (live comparison)

With `--render foxglove` the script streams both agents in sequence to Foxglove
after training. Open Foxglove Studio, connect to `ws://localhost:8765`, and watch:

1. **Fixed-target agent**: the arm moves to the same position every episode,
   regardless of where the green target sphere is. The arm "ignores" the target
   completely after it was trained.
2. **Random-target agent**: the arm moves *toward* the target sphere, tracking
   wherever it appears. Each episode the arm finds a different path to a different
   goal.

The target sphere colour changes from red to green when the tip is within 2 cm.
The fixed agent's sphere almost never turns green on an unseen target; the random
agent's sphere turns green frequently.

---

## How to run

### Minimal (terminal only, no viewer)

```bash
python experiments/06_generalisation.py
```

This trains both agents for 40,000 steps each, evaluates on 50 unseen targets, and
saves three plots to `experiments/_outputs/06_generalisation/`.

### With live 3-D visualisation (Foxglove)

Start the Foxglove desktop app, create a connection to `ws://localhost:8765`, and:

```bash
python experiments/06_generalisation.py --render foxglove
```

Both agents are streamed to Foxglove in sequence after training — first the fixed
agent, then the random agent — so you can compare them directly.

For Foxglove setup details see
[`docs/getting_started/foxglove_setup.md`](../getting_started/foxglove_setup.md).

### Quick smoke-test (CI / low-resource machine)

```bash
python experiments/06_generalisation.py --quick
```

Uses a tiny training budget (600 steps per agent, 5 evaluation episodes) and
finishes in a few seconds on CPU. No plots, no Foxglove. The generalisation gap
may not be large at this budget, but the script runs cleanly end-to-end.

### Full option reference

```
usage: 06_generalisation [--quick] [--render {foxglove}] [--seed INT] [--no-plot]

  --quick              Tiny budget; finish in seconds (CI mode).
  --render foxglove    Stream live to Foxglove after training.
  --seed INT           Master random seed (default: 0).
  --no-plot            Skip saving the matplotlib plots (useful for quick checks).
```

### Typical output (full run)

```
=================================================================
Experiment 06 — Generalisation & Domain Randomisation
=================================================================
  quick=False  train_steps=40,000  eval_episodes=50  seed=0

--- Training: FIXED target (memorisation) (40,000 steps, seed=0) ---
    Training done in 45.2 s — 3814 episodes logged.

--- Training: RANDOM target (generalisation) (40,000 steps, seed=1) ---
    Training done in 47.8 s — 2903 episodes logged.

Evaluating both agents on 50 UNSEEN random targets …

───────────────────────────────────────────────────────
RESULTS (unseen targets)
───────────────────────────────────────────────────────
  Fixed-target agent  — success: 6%    mean dist: 8.3 cm
  Random-target agent — success: 62%   mean dist: 2.1 cm
  Generalisation gap  (+random − fixed): +56%
───────────────────────────────────────────────────────
  > The random-target agent generalises better to unseen targets.
  > The fixed-target agent memorised ONE trajectory — it has no
    general reaching policy.

Saving plots …
    Saved: experiments/_outputs/06_generalisation/learning_curves.png
    Saved: experiments/_outputs/06_generalisation/generalisation_gap.png
    Saved: experiments/_outputs/06_generalisation/per_episode_distances.png

Done. gap=+56%  fixed=6%  random=62%
```

---

## Saved plots

Three plots are written to `experiments/_outputs/06_generalisation/`:

| File | What it shows |
|---|---|
| `learning_curves.png` | Per-episode rolling success rate (100-episode window) during training, for both agents. |
| `generalisation_gap.png` | Bar chart: success rate and mean final distance on unseen targets for both agents. The generalisation gap is shown in the subtitle. |
| `per_episode_distances.png` | Scatter: per-episode final tip-to-target distance for all evaluation episodes. The red dashed line marks the 2 cm success threshold. |

---

## The observation design lesson

Look at `rl_lab/env/spaces.py` (specifically `build_observation`). The last three
elements of the 17-D observation vector are:

```
obs[14:17] = (target_pos - ee_pos) * POS_SCALE   # normalised tip→target vector
```

This is the key ingredient that makes the random-target agent possible. Without
this vector in the observation, the agent would have no way to know where the
target is — it would be forced to memorise trajectories rather than respond to the
goal. With it, the agent can condition every action on the direction it needs to
move, making genuine generalisation feasible.

This is also why **observation design is a first-class concern** in RL. An
observation that omits information the agent needs to solve the task *by
definition* prevents generalisation. The agent can only be as general as its
observation allows.

---

## Connection to sim-to-real

The fixed-target agent is an extreme illustration of a common failure mode in
robotics RL: training on a single scenario (or a narrow scenario distribution) and
expecting the policy to work on a real robot where nothing is quite the same.

On the real Buddy Jr, the target position is determined by a camera detection
pipeline — it is never in exactly the position the simulation used. If you trained
on one fixed target in sim, the real-robot performance is unpredictable.

The random-target agent is a first step toward robustness. Experiment 11 continues
this theme with full domain randomisation: perturbing link lengths, adding servo
jitter, introducing control latency, and injecting observation noise — all the
sources of variation you would expect on real hardware.

The mental model to carry forward:

> *The distribution of targets (or perturbations) you train on is the distribution
> your policy can handle. Make your training distribution match (or exceed) the
> distribution you expect at deployment.*

---

## Aha takeaway

> *Train on variety or you will only ever memorise one episode.*

After this experiment you know:

- The difference between memorisation and generalisation in RL, and how to measure
  it (the generalisation gap on held-out targets).
- Why the target position must be in the observation for a general reaching policy
  to be possible.
- What domain randomisation is: deliberately widening the training distribution to
  force the policy to learn a general strategy.
- Why domain randomisation is the first practical step toward sim-to-real transfer.

**Now you understand generalisation, observation design, and domain randomisation —
your first concrete sim-to-real tool.**

---

## Files involved

| Path | Role |
|---|---|
| `experiments/06_generalisation.py` | The runnable experiment (this script). |
| `rl_lab/env/buddy_jr_reach_env.py` | `BuddyJrReachEnv` — used as the base class for `_FixedTargetEnv`. |
| `rl_lab/env/wrappers.py` | `DiscretizeBuddyJr` wrapper applied to the fixed-target env. |
| `rl_lab/env/spaces.py` | `build_observation` — shows where `obs[14:17]` (tip→target vector) comes from. |
| `rl_lab/algos/value_based/dqn.py` | `DQN` — the algorithm trained in this experiment. |
| `rl_lab/algos/registry.py` | `make_algorithm("dqn", env)` factory. |
| `rl_lab/viz/foxglove_bridge.py` | `FoxgloveStreamer` — live WebSocket publisher for the comparison hook. |
| `experiments/_outputs/06_generalisation/` | Output directory for saved plots. |

---

## What's next

- **Experiment 07** introduces REINFORCE — the simplest policy-gradient algorithm.
  Unlike DQN (which learns a value function and derives a policy from it),
  REINFORCE directly optimises the policy parameters. The generalisation lesson
  from this experiment applies equally there: a policy that conditions on the
  target position generalises; one trained on a fixed target memorises.

- **Experiment 11** returns to domain randomisation with a full sim-to-real
  hardening pass: servo noise, control latency, link-length perturbations, and
  safety clamping — everything needed to give a trained policy a realistic chance
  on the real Buddy Jr arm.

---

<!-- nav-footer -->
← Previous: [DQN](05_dqn.md) &nbsp;|&nbsp; [All experiments](../experiments.md) &nbsp;|&nbsp; Next: [REINFORCE](07_reinforce.md) →
