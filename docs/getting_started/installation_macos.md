# Installing Buddy Jr RL Lab on macOS (Apple Silicon)

This guide covers a complete setup on macOS with an Apple Silicon chip (M1, M2,
or M3 series). By the end you will have a working virtual environment, the
default install verified with `rl-lab --help`, and optional physics simulation
enabled if you want to run experiments with a full 3D physics backend.

If you are setting up Foxglove visualisation, see
[Foxglove Setup](foxglove_setup.md) after completing this page. For ROS2/rviz2,
see [ROS2 Installation](installation_ros2.md).

---

## Prerequisites

- macOS 13 Ventura or later (Monterey works but is untested).
- **Python 3.12.** See [Python version note](#python-version-note) below if you
  have Python 3.14 or another version already installed.
- A terminal (Terminal.app, iTerm2, or VS Code's integrated terminal all work).
- Git — already present on macOS after accepting the Xcode command-line tools
  prompt, or via `brew install git`.
- (Optional) [Homebrew](https://brew.sh) if you need to install Python.
- (Optional) [Miniforge or Miniconda](https://conda-forge.org/miniforge/) if
  you want the PyBullet physics backend (see [Physics simulation](#physics-simulation-optional)).

---

## Python version note

**Use Python 3.12. Avoid Python 3.14 (or any 3.13+) for now.**

This lab depends on `torch`, `stable-baselines3`, and `pybullet`. Prebuilt
binary wheels for these packages on macOS arm64 consistently trail new Python
releases by several months. At the time of writing, none of these packages
publish arm64 wheels for Python 3.14, which means `pip install` falls back to a
source build — and those builds often fail or produce broken binaries against the
current Xcode SDK.

Python 3.12 is the project's declared target (see `.python-version` at the repo
root) and is what CI tests against. It has stable, tested wheels for every core
dependency.

### Check your current Python

```bash
python3 --version
python3.12 --version
```

If `python3.12` is not found, install it:

```bash
# Option A — Homebrew (simplest)
brew install python@3.12

# Option B — python.org installer
# Download from https://www.python.org/downloads/release/python-3128/
# and run the .pkg file.
```

Verify after installation:

```bash
python3.12 --version   # should print Python 3.12.x
```

---

## 1. Clone the repository

```bash
git clone https://github.com/kevinmcaleer/rl_lab.git
cd rl_lab
```

---

## 2. Create a virtual environment

Always use `python3.12` explicitly here, regardless of what `python3` resolves
to on your machine:

```bash
python3.12 -m venv .venv
```

This creates a self-contained environment in the `.venv/` folder inside the
repo. Activate it:

```bash
source .venv/bin/activate
```

Your prompt will change to show `(.venv)`. Every `pip` and `python` command
from this point forward runs inside the venv, not your system Python.

> **Tip:** You need to run `source .venv/bin/activate` again whenever you open
> a new terminal tab or window. If you use VS Code, selecting the `.venv`
> interpreter (Command Palette > "Python: Select Interpreter") handles this
> automatically for the integrated terminal.

---

## 3. Install the default stack

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

The `-e` flag installs the package in **editable mode**: changes you make to the
source files under `rl_lab/` take effect immediately without reinstalling.

The `[dev]` extra includes the full developer toolchain on top of the core stack:
`pytest`, `ruff`, `black`, `mypy`, `mkdocs-material`, and `mkdocstrings`. This
is what you want for running experiments and contributing.

The install pulls in these core runtime packages (among others):

| Package | Purpose |
|---|---|
| `gymnasium` | Standard RL environment API (`BuddyJrReach-v0` etc.) |
| `stable-baselines3` | PPO, SAC, TD3, DDPG algorithms |
| `torch` (CPU build) | Tensor operations and neural network policies |
| `foxglove-sdk` | WebSocket bridge for live 3D visualisation |
| `numpy` | Array maths used throughout |
| `tensorboard` | Training metrics logging |
| `matplotlib` | Reward/episode plots |

PyBullet is deliberately **not** in the core dependencies. See the next
section for why and how to add it.

### Makefile shortcut

If you prefer, the `Makefile` wraps the steps above into a single command:

```bash
make install
```

This creates `.venv` with `python3.12` if it does not already exist, upgrades
`pip`, and runs `pip install -e ".[dev]"`. Useful if you cloned a fresh copy and
want to get going in one step.

---

## 4. Verify the installation

### Check the CLI entry point

```bash
rl-lab --help
```

Expected output (abbreviated):

```
usage: rl-lab [-h] {train,eval,list,viz} ...

Buddy Jr RL Lab command-line interface.

subcommands:
  train     Train an agent on a Buddy Jr environment.
  eval      Evaluate a saved checkpoint.
  list      List available environments and algorithms.
  viz       Launch the Foxglove visualisation bridge.
```

If you see `command not found: rl-lab`, the venv is not activated. Run
`source .venv/bin/activate` and try again.

### Run the test suite

```bash
pytest
```

All tests should pass. The PyBullet-specific test
(`tests/test_urdf_loads.py`) is automatically skipped when PyBullet is not
installed — that is expected and correct on a default install.

You should see output similar to:

```
........s...........                              [ 100%]
X passed, 1 skipped in Y.Zs
```

(The exact count grows as more tests are added; the important thing is no
failures.)

---

## 5. Physics simulation (optional) {#physics-simulation-optional}

The default install uses the **KinematicBackend** — a fast, physics-free solver
that is sufficient for the first several experiments. If you want full rigid-body
dynamics (contact forces, gravity, inertia) for experiments 09 onwards, you need
an actual physics engine.

You have two options: PyBullet (via conda) or MuJoCo (via pip).

### Option A — PyBullet via conda-forge (recommended for macOS)

PyBullet has **no prebuilt arm64 wheel on PyPI**. Attempting
`pip install pybullet` on Apple Silicon triggers a source build that currently
fails against recent Xcode SDKs. The conda-forge channel provides a working
binary package instead.

**Step 1.** Install Miniforge (a minimal conda installer that defaults to
conda-forge) if you do not already have it:

```bash
# Homebrew
brew install --cask miniforge

# Or download the shell installer directly:
# https://github.com/conda-forge/miniforge/releases/latest
# curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
# bash Miniforge3-MacOSX-arm64.sh
```

**Step 2.** Create a conda environment that includes Python 3.12 and PyBullet:

```bash
conda create -n rl_lab python=3.12
conda activate rl_lab
conda install -c conda-forge pybullet
```

**Step 3.** Install the rest of the lab into the same conda environment:

```bash
# Inside the conda env (not the pip .venv)
pip install -e ".[dev,sim]"
```

The `[sim]` extra declares PyBullet as the physics backend dependency. Because
you already installed `pybullet` from conda-forge, pip will detect it is
present and skip reinstalling it from PyPI.

**Step 4.** Verify PyBullet is importable:

```bash
python -c "import pybullet; print('pybullet', pybullet.__version__)"
```

> **Note on mixing pip and conda:** Only install pybullet via conda; install
> everything else via pip. Mixing conda and pip packages in the same environment
> is generally fine as long as you install conda packages first and pip packages
> after. Do not run `conda install torch` — the pip wheel from PyPI is faster
> and better maintained for Apple Silicon.

### Option B — PyBullet via pip (Linux / Windows only)

On Linux x86-64 and Windows, a prebuilt pybullet wheel exists on PyPI:

```bash
pip install -e ".[sim]"
```

This does not work reliably on macOS Apple Silicon. Use Option A above.

### Option C — MuJoCo via pip

MuJoCo ships clean arm64 wheels and is a full alternative physics backend:

```bash
pip install -e ".[mujoco]"
```

Verify:

```bash
python -c "import mujoco; print('mujoco', mujoco.__version__)"
```

To use MuJoCo as the backend in experiments, set the `BACKEND` environment
variable:

```bash
export BACKEND=mujoco
python experiments/09_ppo_continuous.py
```

The `SimBackend` abstraction means the RL code is identical — switching
backends is itself a learning exercise (see Experiment 06: Generalisation).

---

## 6. Quick smoke test

Once the install is complete, run a short training loop to confirm everything
works end to end:

```bash
python experiments/01_bandit.py --quick
```

This should run a few hundred steps and print a summary table to the terminal in
under a minute. No physics backend or GPU is needed.

To test the full environment stack (KinematicBackend):

```bash
rl-lab list                         # shows available envs and algorithms
rl-lab train --algo qlearning --env BuddyJrReachDiscrete-v0 --steps 5000
```

---

## Troubleshooting

### `rl-lab: command not found`

The venv is not active. Run:

```bash
source .venv/bin/activate
```

### `ModuleNotFoundError: No module named 'rl_lab'`

The editable install is missing. From inside the repo root with the venv
activated:

```bash
pip install -e ".[dev]"
```

### `pip install pybullet` fails with compiler errors

This is the known no-arm64-wheel issue. Do not try to force pip to build
PyBullet from source on macOS. Use the conda-forge path:

```bash
conda install -c conda-forge pybullet
```

### `torch` install is very slow or downloads a large file

Torch for macOS arm64 is distributed as a single large wheel (~200 MB). The
first install takes a few minutes on a typical broadband connection. Subsequent
installs use pip's cache. This is normal.

### Tests fail with `gymnasium` version errors

This usually means a stale venv from a previous install of a different
Gymnasium version. Delete and recreate the venv:

```bash
deactivate
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### `black` formatter version mismatch warnings

`black` is pinned in `pyproject.toml` to a specific version so that local and
CI formatting never drift. If you see a version conflict, let pip resolve it:

```bash
pip install --upgrade -e ".[dev]"
```

---

## What to do next

1. **Visualisation** — set up the Foxglove 3D viewer so you can watch the
   robot arm move during training:
   [Foxglove Setup](foxglove_setup.md)

2. **First experiment** — open `experiments/01_bandit.py` and read through it;
   it introduces the core RL loop in about 60 lines before any robot is
   involved. Run it with `python experiments/01_bandit.py`.

3. **Environment explorer** — the notebook
   `notebooks/inspecting_the_env.ipynb` lets you manually step through
   `BuddyJrReach-v0` and inspect observations, rewards, and episode
   termination before any learning happens.

4. **ROS2 / rviz2 (advanced)** — if you already work in a ROS2 ecosystem and
   want the full `robot_state_publisher` pipeline, see
   [ROS2 Installation](installation_ros2.md). This is not required for any of
   the 12 curriculum experiments.
