# Contributing to Buddy Jr RL Lab

Welcome — and thanks for wanting to help! This project exists to make reinforcement
learning accessible to makers, hobbyists, and robotics enthusiasts. Whether you are
fixing a typo, adding a new experiment, or improving the simulation environment, your
contribution matters.

No RL PhD required. If you have ever run a Python script and are curious about how
robots learn, you are in exactly the right place.

---

## Table of contents

1. [Setting up your development environment](#1-setting-up-your-development-environment)
2. [Running the tests](#2-running-the-tests)
3. [Code style](#3-code-style)
4. [Branch and pull-request flow](#4-branch-and-pull-request-flow)
5. [Commit message style](#5-commit-message-style)
6. [Proposing or adding a new experiment](#6-proposing-or-adding-a-new-experiment)
7. [Reporting bugs and suggesting features](#7-reporting-bugs-and-suggesting-features)
8. [Code of Conduct](#8-code-of-conduct)

---

## 1. Setting up your development environment

### Prerequisites

- **Python 3.12** (recommended; 3.10 and 3.11 are supported but 3.12 is what CI runs).
  Use [pyenv](https://github.com/pyenv/pyenv) or the [official installer](https://www.python.org/downloads/).
- **git** (any modern version).
- A C compiler is needed by PyBullet (`gcc` on Linux, Xcode Command Line Tools on macOS).

### Clone and install

```bash
# 1. Fork the repo on GitHub, then clone your fork:
git clone https://github.com/<your-username>/rl_lab.git
cd rl_lab

# 2. Create and activate a virtual environment:
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install the package in editable mode, including all dev tools:
pip install -e .[dev]

# 4. Install the pre-commit hooks (runs ruff + mypy before every commit):
pre-commit install
```

After step 3 you should have `rl-lab` available on your PATH:

```bash
rl-lab --help
```

### The physics backend (`[sim]`)

The default PyBullet backend is the `[sim]` extra. It is kept out of the core
deps because it has no macOS Apple-Silicon wheel and its source build fails on
recent Xcode:

```bash
pip install -e ".[sim]"            # Linux / Windows: a plain wheel
conda install -c conda-forge pybullet   # macOS Apple Silicon: use conda-forge
```

The pure-Python URDF tests run without PyBullet (they self-skip the load test
when the wheel is absent), so you can develop and pass CI on macOS without it.

### Optional extras

```bash
pip install -e .[mujoco]   # MuJoCo physics backend (requires a MuJoCo licence key)
pip install -e .[rpi]      # Raspberry Pi servo deploy (adafruit-circuitpython-servokit)
```

The `[ros2]` extra is an advanced track that runs in Docker or a Linux VM;
it has no pip dependencies and is documented separately in `docs/`.

---

## 2. Running the tests

```bash
# Run the full test suite:
pytest

# Or, if a Makefile target is available:
make test

# Run a single test file:
pytest tests/test_urdf.py -v

# Run only tests that don't need a GPU or display:
pytest -m "not slow"
```

All tests must pass before you open a pull request. CI (GitHub Actions) runs them on
every push and PR — you can see the same results locally before pushing.

### What the tests cover

- `tests/test_urdf.py` — the URDF loads, has a single root, acyclic link tree,
  valid joint limits, and positive-definite inertias.
- `tests/test_env_api.py` — Gymnasium `check_env` conformance.
- `tests/test_kinematics.py` — forward kinematics matches known poses.
- `tests/test_rewards.py` — reward modes return the expected values.
- `tests/test_servo_map.py` — servo angle clamping is correct.

If you add a new feature, please add or extend the relevant test file.

---

## 3. Code style

This project uses **ruff** for linting and formatting, with a line length of **100**.
The pre-commit hook runs ruff automatically — you can also run it manually:

```bash
ruff check .          # lint
ruff format .         # auto-format (equivalent to black)
mypy rl_lab/          # type-check the package
```

Key style points:

- 4-space indentation, LF line endings, a final newline in every file.
- Imports are isort-ordered (ruff handles this).
- Write comments for **learners**: explain the *why* behind non-obvious code,
  especially anything involving RL maths or PyBullet API quirks.
- Public functions and classes must have docstrings.
- Avoid acronyms in variable names without explanation (`obs` is fine because
  Gymnasium uses it everywhere; `td_err` should be `td_error` with a comment).

---

## 4. Branch and pull-request flow

```
main           <- always releasable; protected; only accepts PRs
feature/<name> <- your work branch
bugfix/<name>
experiment/<name>  <- new experiments or major changes to existing ones
docs/<name>
```

1. **Create a branch** from the latest `main`:
   ```bash
   git fetch origin
   git checkout -b feature/my-improvement origin/main
   ```

2. **Make your changes.** Keep commits focused — one logical change per commit.

3. **Run the tests and the pre-commit hooks:**
   ```bash
   pre-commit run --all-files
   pytest
   ```

4. **Push and open a pull request** against `main`.
   - Give the PR a clear title (see commit style below).
   - Fill in the PR template: what changed, how to test it, and any experiments
     or screenshots that show the change in action.
   - Link the relevant GitHub issue (e.g. `Closes #42`).

5. **A reviewer will look at your PR.** We aim to respond within a few days.
   Please do not force-push to a PR branch after review has started — add new
   commits instead so the diff is easy to follow.

6. **Merge.** Once approved, the maintainer squash-merges into `main`.

---

## 5. Commit message style

This project uses **Conventional Commits** (https://www.conventionalcommits.org).
Format:

```
<type>(<scope>): <short summary in imperative mood>

[Optional body: what and why, wrapped at 72 characters.]

[Optional footer: Closes #<issue>, Co-authored-by: …]
```

**Types:**

| Type | When to use |
|------|-------------|
| `feat` | A new feature or experiment |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `refactor` | Code change that is not a fix or feature |
| `chore` | Build tooling, CI, dependency bumps |
| `perf` | Performance improvement |

**Scopes** (optional but encouraged): `urdf`, `env`, `sim`, `viz`, `algos`,
`train`, `cli`, `exp01` … `exp12`, `rpi`, `ci`, `docs`.

**Examples:**

```
feat(env): add domain randomisation wrapper for Exp 11
fix(urdf): correct shoulder_pitch joint axis from X to Y
docs(exp03): clarify Q-table index convention in comments
chore(ci): pin ruff to 0.4.x to avoid formatter churn
```

The pre-commit hook will warn you if the message format is wrong.

---

## 6. Proposing or adding a new experiment

The curriculum is the heart of this lab. Experiments live in `experiments/` and
follow a strict structure so each one teaches **one concept** and fits the same
five-part template.

### Proposing an experiment

1. Open a GitHub issue with the label `experiment`.
2. Describe: the RL concept being taught, the learning objective, the "aha
   takeaway", and how it fits into the curriculum arc.
3. Discuss with maintainers before writing code — we want the curriculum to be
   coherent, not a grab-bag.

### Implementing an experiment

Read [`experiments/README.md`](experiments/README.md) first — it documents the full
curriculum arc, the robot, the stack, and exactly how each experiment is structured.

Every experiment file must:

- Start with a module-level docstring covering: concept, objective, run instructions,
  what to watch for, and the "aha takeaway".
- Use the standard five-section template:
  1. **Concept** — one RL idea.
  2. **Objective** — what the learner can do afterwards.
  3. **Build & run** — the concrete thing created and executed.
  4. **Watch for** — specific behaviour to observe.
  5. **Aha takeaway** — one-sentence insight + "now you understand X".
- Be runnable as a standalone script: `python experiments/XX_name.py`.
- Include inline comments aimed at a learner encountering the idea for the first time.
- Add or update a test in `tests/` that at least imports and instantiates the
  experiment's env without crashing.

An experiment template (`experiments/_template/`) is provided as a starting
point.

---

## 7. Reporting bugs and suggesting features

- **Bug?** Open a GitHub issue. Include: Python version, OS, full traceback, and the
  command you ran. For simulation bugs, a screenshot from Foxglove or PyBullet helps
  a lot.
- **Feature idea?** Open an issue with the label `enhancement`. Describe the use-case
  and who it helps (beginners? Raspberry Pi deployers? advanced users?).
- **Security issue?** Email kevinmcaleer@gmail.com directly rather than opening a
  public issue.

---

## 8. Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). By
participating you agree to abide by its terms. The enforcement contact is
kevinmcaleer@gmail.com.

In short: be kind, be patient, and remember that many contributors are learning
both Python and RL at the same time. That is the whole point of the lab.
