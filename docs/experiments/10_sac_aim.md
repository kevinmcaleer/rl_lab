# Experiment 10 — SAC: sample-efficient continuous control + the full aim task

> **Part V — Continuous control (the real arm's action space)**
>
> Script: [`experiments/10_sac_aim.py`](../../experiments/10_sac_aim.py)
> Env: `BuddyJrCameraPoint-v0` · Algorithms: **SAC** vs. **PPO**

This is the experiment where Buddy Jr finally does its *real* job: not just put
the camera **near** a point in space, but **aim the camera's view ray straight at
it** using all four joints. Along the way you learn the most consequential
practical choice in deep-RL-for-robots — **on-policy vs. off-policy** — and the
**sample efficiency** that the off-policy Soft Actor-Critic (SAC) buys you over
the on-policy PPO you met in Experiments 8 and 9.

---

## The five-part summary

1. **Concept** — Off-policy continuous control (**SAC**); **entropy-regularised
   exploration**; **sample efficiency** vs. on-policy PPO.
2. **Objective** — Solve the true Buddy Jr task (aim the camera at a target with
   all 4 DOF) and learn *when to reach for SAC instead of PPO*.
3. **Build & run** — Train SB3 `SAC` and SB3 `PPO` on the **same** aim task for
   the **same** number of environment steps; plot mean return vs. env-steps.
4. **Watch for** — SAC's learning curve climbs much faster *per environment
   step*; in Foxglove the camera's view ray locks onto (and tracks) the target.
5. **Aha takeaway** — *Off-policy methods squeeze more learning out of each
   sample, which is gold when "samples" are expensive robot moves.* **Now you
   understand SAC, the entropy bonus, and the PPO-vs-SAC trade-off for real
   robots.**

---

## 1. The concept

### On-policy vs. off-policy — the one idea that matters here

- **PPO is on-policy.** It collects a fresh batch of experience with the
  *current* policy, does a handful of gradient steps, then **throws that batch
  away** and collects again. Stable and simple, but every environment step is
  used only once or twice.
- **SAC is off-policy.** It stores every transition `(state, action, reward,
  next_state)` in a **replay buffer** and replays each one *many* times during
  training. When a "sample" is a real servo move on a real arm, re-using it is
  worth a great deal.

That single difference is why SAC usually reaches a good policy in **far fewer
environment steps** than PPO on continuous-control tasks like this one.

### Entropy-regularised exploration

SAC does not only maximise reward. It maximises reward **plus the entropy of the
policy**:

```
J(π) = E[ Σ_t  r_t  +  α · H(π(·|s_t)) ]
```

The entropy term `H` rewards the policy for *staying uncertain* — for keeping a
spread of plausible actions instead of collapsing onto one too early. That keeps
exploration alive deep into training. The temperature `α` controls how much we
value exploration vs. reward, and Stable-Baselines3 **auto-tunes it** toward a
target entropy of `-dim(action)` (here `-4`, one per joint), so you do not have to
hand-set an exploration schedule the way you did with ε in the tabular
experiments.

### Why "aim" is harder than "reach"

Experiments 3–9 only asked the camera **tip** to land near the target. This
experiment uses `BuddyJrCameraPoint-v0`, which adds an **alignment reward**: the
cosine of the angle between the camera link's forward axis and the unit vector
pointing at the target. `+1` means the camera is staring straight down the line to
the target; `0` means it is pointing 90° away.

```
reward = (shaping / distance term)  +  align_weight · cos(camera_ray, to_target)
```

Crucially, the **camera-tilt (wrist) joint**, which was idle when only tip
*position* mattered, now becomes load-bearing — orientation depends on it. The
agent must coordinate all four joints, which is exactly the kind of coupled
continuous-control problem SAC excels at. The per-step `info["alignment"]` value
is what the experiment uses to score how well a trained policy actually aims.

---

## 2. What the script does

`experiments/10_sac_aim.py` follows the lab's frozen experiment interface
(`run()` / `main()` / `__main__`). In one `run()` it:

1. Trains **SAC** on `BuddyJrCameraPoint-v0` for a fixed step budget, recording
   `(env_step, mean_episode_return)` samples via a lightweight callback.
2. Trains **PPO** on the **same** task for the **same** budget, recording the same
   samples.
3. Evaluates both trained policies for several deterministic episodes and reports
   `mean_return`, `mean_alignment` (the terminal aim cosine), and `success_rate`.
4. Saves a **reward-vs-environment-steps** comparison plot (Agg backend, no
   window) to `experiments/_outputs/10_sac_aim/sac_vs_ppo_sample_efficiency.png`.
5. With `--render foxglove`, streams a greedy SAC rollout to Foxglove Studio so
   you can watch the view ray lock onto the target.

Both algorithms come straight from the registry with their CPU-friendly defaults,
so the code reads at a high level:

```python
from rl_lab.algos.registry import make_algorithm

algo = make_algorithm("sac", env, seed=seed)   # or "ppo"
algo.train(total_steps=total_steps, callback=recorder)
action, _ = algo.predict(obs, deterministic=True)
```

---

## 3. How to run it

> **Prerequisites.** A working install with `stable-baselines3`, `torch`,
> `gymnasium`, `matplotlib`, and (for streaming) `foxglove-sdk`. See
> [`docs/getting_started/installation_ros2.md`](../getting_started/installation_ros2.md)
> and [`docs/getting_started/foxglove_setup.md`](../getting_started/foxglove_setup.md).

### Quick smoke test (a few seconds, CPU)

```bash
python experiments/10_sac_aim.py --quick
```

`--quick` uses a tiny training budget, **writes no plots**, and **never touches
Foxglove**. It exists so continuous integration can confirm the experiment still
imports, trains, and evaluates without burning minutes — do not read anything into
the numbers a quick run prints.

### Full run (a couple of minutes, CPU)

```bash
python experiments/10_sac_aim.py
```

This trains both algorithms for the full budget, prints a summary, and writes the
comparison plot to `experiments/_outputs/10_sac_aim/`.

### Watch it in Foxglove

```bash
python experiments/10_sac_aim.py --render foxglove
```

After training, the script rolls out the SAC policy in an env with
`render_mode="foxglove"`; the environment publishes the URDF arm, the target
sphere, and live metrics. Open Foxglove Studio, connect to `ws://localhost:8765`,
and import the layout at `rl_lab/viz/layouts/buddy_jr.json`.

### Useful flags

| Flag | Effect |
|------|--------|
| `--quick` | Tiny budget, no plots, no Foxglove (CI smoke test). |
| `--render foxglove` | Stream the trained SAC policy to Foxglove Studio. |
| `--seed N` | Set the random seed (default `0`) for reproducible runs. |
| `--no-plot` | Full training + evaluation, but skip writing the plot. |

---

## 4. The Foxglove view: a frustum that locks onto a dragged target

The 3D panel of the Buddy Jr layout already shows the arm (`/tf` + `/robot`) and
the target sphere (`/scene`). For the aim task it helps to picture — and you can
add to the layout — the camera's **view frustum**: a thin cone (or a single ray)
projected forward along the camera link's local `+Z` axis, exactly the direction
the alignment reward measures.

What to watch:

- **Early training:** the frustum points in roughly random directions; the
  alignment value oscillates around `0` and the distance plot wobbles.
- **As SAC learns:** the frustum **swings around and locks onto** the amber
  target sphere — and the sphere turns green when the tip is also within
  tolerance. The alignment cosine climbs toward `+1`.
- **Drag the target live:** grab the target sphere in the 3D panel and move it.
  Because the policy generalises over target position (it is part of the
  observation — `obs[14:17]` is the tip→target vector), the frustum **tracks** the
  moving target like a pan-tilt camera following a subject. This is the same
  behaviour you saw the DQN do in Experiment 6, but now in full continuous control
  *and* with orientation, not just position.

This is also a sim-to-real rehearsal: the **same** `JointState` + marker messages
that animate this sim arm will, in Experiment 12, animate the **real** Buddy Jr in
the identical Foxglove layout.

---

## 5. Expected result

A full run prints something like (exact numbers vary with seed and budget):

```
Task: BuddyJrCameraPoint-v0  |  budget: 20000 env steps
  SAC  ->  return= ...   aim(cos)=+0.9..  success= 8..%
  PPO  ->  return= ...   aim(cos)=+0.6..  success= 3..%
  Learning-curve plot saved to: experiments/_outputs/10_sac_aim/sac_vs_ppo_sample_efficiency.png
```

The point is **not** the absolute scores — the budget here is deliberately modest
so the experiment runs on a laptop CPU in a couple of minutes. The point is the
**relationship between the two curves**:

- **SAC's curve rises sooner and steeper** per environment step. With its replay
  buffer reusing every transition many times, it extracts more learning from the
  same amount of experience.
- **PPO's curve is smoother but lags** at this small step budget. Given *many
  more* steps it would catch up (and PPO is often easier to tune and parallelise),
  but step-for-step on this continuous task SAC is more sample-efficient.
- **SAC's final aim cosine is closer to `+1`**, meaning the camera ends episodes
  pointing more squarely at the target.

If your SAC curve does **not** beat PPO, the usual culprits are: too small a step
budget (raise `_FULL_STEPS`), an unlucky seed (try `--seed 1`, `--seed 2`), or a
machine so slow that PPO's larger rollouts dominate wall-clock — remember the
x-axis is *environment steps*, not seconds.

---

## 6. When to pick SAC vs. PPO on your own robot

| Situation | Prefer |
|-----------|--------|
| Continuous actions, samples are **expensive** (real hardware, slow sim) | **SAC** (off-policy, replay buffer) |
| You can collect **lots** of cheap parallel sim steps | **PPO** (on-policy, easy to parallelise, very stable) |
| Discrete actions | PPO or DQN — SAC is continuous-only |
| You want the simplest thing that "just works" first | PPO, then try SAC for sample efficiency |

SAC, TD3, and DDPG are all off-policy continuous-control siblings in the registry
(`make_algorithm("td3", env)`); SAC's entropy regularisation usually makes it the
most robust of the three out of the box.

---

## Aha takeaway

> *Off-policy methods squeeze more learning out of each sample, which is gold when
> "samples" are expensive robot moves.*

**Now you understand SAC, the entropy bonus, and the PPO-vs-SAC trade-off for real
robots.** Next, in [Experiment 11](11_robustify.md), you harden this aim policy
against the messiness of real SG90 servos — noise, latency, and safety limits —
before it ever touches hardware.
