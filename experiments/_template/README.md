# Experiment Template

This directory is a **copyable skeleton** for adding a new experiment to the
Buddy Jr RL Teaching Lab curriculum.

---

## Quick start

```bash
# 1. Copy the template (pick the next free number, e.g. 13)
cp -r experiments/_template experiments/13_my_experiment

# 2. Rename the script
mv experiments/13_my_experiment/experiment.py \
   experiments/13_my_experiment/13_my_experiment.py

# 3. Rename the config
mv experiments/13_my_experiment/config.yaml \
   experiments/13_my_experiment/13_my_experiment.yaml

# 4. Smoke-test immediately (before you write any real code)
python experiments/13_my_experiment/13_my_experiment.py --quick
# Expected: prints a metrics dict and exits 0 in < 5 seconds.
```

Then work through the `TODO` markers in the script and config.

---

## File-by-file guide

### `experiment.py` (rename to `NN_name.py`)

The main Python file.  Every experiment in the lab must expose exactly
this interface (the **Frozen Interface**):

```python
def run(quick: bool = False,
        render: str | None = None,
        seed: int = 0) -> dict: ...

def main(argv: list[str] | None = None) -> int: ...

if __name__ == "__main__":
    raise SystemExit(main())
```

| Requirement | Details |
|-------------|---------|
| `quick=True` | Tiny step budget, no plots, no Foxglove. Must finish in **< 5 seconds on CPU**. This is what CI runs. |
| `render` | `None` (default) or `"foxglove"`. Always `None` when `quick=True`. |
| Return value | A `dict` with at least `"algo"`, `"total_steps"`, `"mean_return"`. |
| Plots | Saved under `experiments/_outputs/<name>/` (create with `mkdir(parents=True, exist_ok=True)`) and only when **not** in quick mode. |
| matplotlib | `matplotlib.use("Agg")` **before** `import matplotlib.pyplot`. This is a hard requirement — CI has no display. |
| `from __future__ import annotations` | First non-comment line of every file. |

**Key TODOs in the script:**

1. Replace the module docstring with your concept / objective / aha sections.
2. Set `EXPERIMENT_NAME` to your script's stem (e.g. `"13_my_experiment"`).
3. Set `ALGO` to the algorithm being taught.
4. Adjust hyperparameters in `make_algorithm(...)` to illustrate the concept.
5. Add experiment-specific subplots in `_plot_learning_curve()`.
6. Extend the returned metrics dict with anything you want to assert on in tests.

### `config.yaml` (rename to `NN_name.yaml`)

A plain YAML knobs file for the experiment's hyperparameters.  It is
**optional** — you can hard-code everything in the script.  Use it when:

- You want learners to experiment by editing a config without touching Python.
- You are implementing a hyperparameter sweep.
- Your experiment shares a training loop used by multiple configs (e.g.
  comparing `reward_mode: sparse` vs `reward_mode: dense` across two YAML
  files without duplicating the script).

Load it in your script with:

```python
import yaml
from pathlib import Path

cfg = yaml.safe_load(
    (Path(__file__).parent / "13_my_experiment.yaml").read_text()
)
alpha = cfg.get("alpha", 0.1)
```

### This `README.md`

Replace this file with the experiment-specific student guide.  Keep the same
five-section structure used in `experiments/README.md`:

1. **Concept** — the single RL idea.
2. **Objective** — what you can do afterwards.
3. **Build & run** — the concrete steps.
4. **Watch for** — what to observe in plots / Foxglove.
5. **Aha takeaway** — one-sentence insight + "now you understand X".

---

## Available APIs (cheat-sheet)

```python
import rl_lab                       # registers envs with Gymnasium
import gymnasium as gym

# --- Environments ---
# BuddyJrReach-v0          Box(4) action, Box(17) obs — continuous control
# BuddyJrReachDiscrete-v0  Discrete(9) action         — tabular / DQN
# BuddyJrCameraPoint-v0    full aim task
#
# Useful kwargs:
#   render_mode="foxglove"   live 3-D viewer
#   reward_mode="dense"      dense | sparse | shaped
#   max_steps=200
#   goal_env=True            Dict obs for HER-compatible algos
#   target_radius=(0.04, 0.155)   reachable target distance range (metres)
#   reset_noise=0.1          randomise initial joint angles
env = gym.make("BuddyJrReach-v0", reward_mode="dense")

# obs[14:17] = tip -> target Cartesian error vector (most informative feature)
# info keys: distance, is_success, joint_q, ee_pos, target

# --- Algorithm factory ---
from rl_lab.algos.registry import make_algorithm, recommended_env_id, ALGORITHMS

env_id = recommended_env_id("dqn")   # -> "BuddyJrReachDiscrete-v0"
algo = make_algorithm("dqn", env, seed=0, lr=1e-3, gamma=0.99)
history = algo.train(total_steps=50_000, callback=my_callback)
action, _ = algo.predict(obs, deterministic=True)
algo.save("runs/my_run/model")

# ALGORITHMS tuple: all valid name strings
print(ALGORITHMS)

# --- Wrappers ---
from rl_lab.env.wrappers import (
    DiscretizeBuddyJr,       # convert Box obs to Discrete(9) actions
    TabularBuddyJr,          # bin obs for tabular algos
    ActionRepeat,            # repeat each action n steps
    DomainRandomization,     # add noise / latency (Exp 11)
    NormalizeObservation,
    TimeLimit,
)

# --- Kinematics ---
from rl_lab.robot.kinematics import forward
pose = forward(q)   # q: (4,) joint angles in radians
pose.position       # (3,) metres
pose.orientation    # (4,) quaternion xyzw

# --- Foxglove streaming ---
from rl_lab.viz.foxglove_bridge import FoxgloveStreamer
streamer = FoxgloveStreamer(render_mode="foxglove")
streamer.publish(joint_q, p_ee, target_xyz, distance,
                 reward=r, episode_return=G, success_rate=sr)
streamer.close()

# --- Policy export (Exp 12 pattern) ---
from rl_lab.deploy.policy_export import export_algorithm, NumpyMLPPolicy
export_algorithm(algo, "runs/policy.npz")
policy = NumpyMLPPolicy.load("runs/policy.npz")
action = policy.predict(obs)
```

---

## Checklist before opening a PR

- [ ] `python experiments/NN_name.py --quick` exits 0 in < 5 seconds.
- [ ] `python experiments/NN_name.py` completes without error (full run).
- [ ] Plot(s) appear under `experiments/_outputs/NN_name/`.
- [ ] A test in `tests/test_smoke.py` (or a new file) imports the module and
      calls `run(quick=True)`, asserting the return type is `dict`.
- [ ] Module docstring covers all five sections (Concept / Objective /
      Build & run / Watch for / Aha takeaway).
- [ ] Inline comments explain *why*, not just *what*.
- [ ] `ruff check experiments/NN_name.py` reports no issues.
- [ ] Branch name follows `experiment/NN-short-title` convention.
- [ ] PR description links the GitHub issue that proposed this experiment.
