# Troubleshooting

This page collects the most common problems makers run into when setting up the
Buddy Jr RL Lab on macOS (especially Apple Silicon), together with the exact
steps to fix each one. The format is **symptom → cause → fix** throughout.

If your problem is not listed here, check the
[GitHub Issues](https://github.com/kevinmcaleer/rl_lab/issues) page or open a
new one.

---

## Contents

1. [PyBullet GUI window does not open on macOS](#1-pybullet-gui-window-does-not-open-on-macos)
2. [`pip install pybullet` fails on Apple Silicon](#2-pip-install-pybullet-fails-on-apple-silicon)
3. [Foxglove Studio shows "Connection refused"](#3-foxglove-studio-shows-connection-refused)
4. [macOS firewall blocks the WebSocket server](#4-macos-firewall-blocks-the-websocket-server)
5. [`pip install torch` (or pybullet) fails on Python 3.14](#5-pip-install-torch-or-pybullet-fails-on-python-314)
6. [`rl-lab: command not found`](#6-rl-lab-command-not-found)
7. [MuJoCo import error: `No module named 'mujoco'`](#7-mujoco-import-error-no-module-named-mujoco)

---

## 1. PyBullet GUI window does not open on macOS

### Symptom

You run an experiment with `--render human` (or pass `render_mode='human'` in
your own code) and PyBullet either:

- opens a blank black window that immediately closes, or
- prints `pybullet build time: ...` and hangs without displaying a window, or
- raises `pybullet.error: Cannot initialize GLFW` or a similar OpenGL error.

### Cause

PyBullet's built-in OpenGL GUI (`pybullet.GUI` connect mode) does not work
reliably on macOS — it predates Apple's Metal graphics stack and has not been
ported to arm64. Even on Intel Macs the GUI is fragile.

### Fix — option A: use DIRECT mode + Foxglove (recommended)

The lab is designed around this approach. PyBullet runs in headless `DIRECT`
mode (no window) and streams joint-angle data over WebSocket to Foxglove Studio,
which renders the arm in 3-D. You get a better view than the PyBullet window
and live reward plots alongside the robot.

1. Start the bridge in one terminal:

   ```bash
   python scripts/launch_foxglove_bridge.py
   ```

2. Run your experiment with the Foxglove render mode:

   ```bash
   python experiments/05_dqn.py --render foxglove
   # or with the CLI:
   rl-lab train --algo dqn --render foxglove
   ```

3. Open Foxglove Studio, connect to `ws://localhost:8765`, and import the
   layout at `rl_lab/viz/layouts/buddy_jr.json`.

See [`docs/getting_started/foxglove_setup.md`](getting_started/foxglove_setup.md)
for a full walkthrough.

### Fix — option B: use the conda-forge PyBullet build

If you specifically need the PyBullet GUI (for example to compare its view
with Foxglove), the conda-forge community maintains a PyBullet build that links
against native macOS system frameworks:

```bash
conda install -c conda-forge pybullet
```

This still does not guarantee a stable GUI on arm64 Macs, but it is the best
available option for native-window use on macOS. If the GUI freezes, switch to
option A.

---

## 2. `pip install pybullet` fails on Apple Silicon

### Symptom

Any of the following errors when running `pip install pybullet` or
`pip install -e ".[sim]"` on an Apple Silicon Mac (M1, M2, M3, M4):

```
ERROR: Could not find a version that satisfies the requirement pybullet
```

```
error: command '/usr/bin/clang' failed with exit code 1
```

```
clang: error: invalid version number in '-mmacosx-version-min=10.7'
```

### Cause

PyPI does not publish a prebuilt wheel for `pybullet` targeting macOS arm64
(the `macosx_*_arm64` platform tag). When pip cannot find a wheel it falls back
to a source build, which in turn fails because recent versions of Xcode have
removed APIs that pybullet's C extension relies on.

This is a known, long-standing upstream issue in the pybullet project. The rl
lab deliberately lists pybullet as an *optional* dependency (the `[sim]` extra)
rather than a core one so that `pip install -e .` succeeds on macOS without it.

### Fix

Install pybullet from the conda-forge channel, which maintains a compiled
arm64 binary:

```bash
# If you don't have conda yet, install miniforge (arm64-native):
# https://github.com/conda-forge/miniforge

conda install -c conda-forge pybullet
```

Then activate your conda environment before running the lab:

```bash
conda activate <your-env>
python experiments/05_dqn.py
```

> **Why not just use a venv?** The conda-forge pybullet binary is linked
> against system libraries that conda manages. Using it inside a `venv` that
> sits outside conda can create linking conflicts. If you mix conda and venv,
> make sure the conda environment is activated first so its `pybullet` takes
> precedence.

All other lab dependencies (`torch`, `stable-baselines3`, `gymnasium`,
`foxglove-sdk`) install fine from PyPI inside a regular venv or conda
environment.

---

## 3. Foxglove Studio shows "Connection refused"

### Symptom

You open Foxglove Studio, set the URL to `ws://localhost:8765`, click **Open**,
and see:

```
WebSocket connection to 'ws://localhost:8765/' failed:
Error in connection establishment: net::ERR_CONNECTION_REFUSED
```

or a dialog that says _"Connection refused"_ or _"Unable to connect to
ws://localhost:8765"_.

### Cause

The Foxglove WebSocket server is not running. Foxglove Studio is a *client*:
it does not include a server. The lab's bridge script (`scripts/launch_foxglove_bridge.py`)
must be running first, in a separate terminal, before you try to connect.

> Note: `rl-lab viz` is listed in `rl-lab --help` but is not yet fully wired.
> Until it is, use the script directly as described below.

### Fix

1. Open a terminal in the project root with the virtual environment active:

   ```bash
   source .venv/bin/activate   # or: conda activate <your-env>
   ```

2. Start the bridge server:

   ```bash
   python scripts/launch_foxglove_bridge.py
   ```

   You should see output similar to:

   ```
   Foxglove WebSocket server listening on ws://127.0.0.1:8765
   ```

   Leave this terminal open. The server must keep running while you use
   Foxglove Studio.

3. In Foxglove Studio, connect to `ws://localhost:8765`. You can use `localhost`
   or `127.0.0.1` — both resolve to the same loopback interface.

4. In a third terminal, start your experiment:

   ```bash
   python experiments/10_sac_aim.py --render foxglove
   # or:
   rl-lab train --algo ppo --render foxglove
   ```

Foxglove will start receiving data as soon as the experiment's first training
step runs.

---

## 4. macOS firewall blocks the WebSocket server

### Symptom

The bridge starts, Foxglove Studio tries to connect, but it still fails (or
hang-loops) even though the bridge is running. When you started the bridge, macOS
showed a dialog:

> _"Do you want the application python3.x to accept incoming network
> connections?"_

and you clicked **Deny** (or the dialog appeared and disappeared without you
clicking it).

### Cause

macOS Application Firewall blocks network connections to processes that have not
been explicitly allowed to receive incoming connections. Even though both the
bridge and Foxglove Studio are running on the same machine (loopback only), the
macOS firewall inspects loopback traffic when the firewall is in its default
"block all incoming connections" mode.

### Fix

**Option A — allow at the prompt (easiest)**

Re-run `python scripts/launch_foxglove_bridge.py`. macOS will show the
firewall dialog again. Click **Allow**. The server binds only to `127.0.0.1`
and is never reachable from outside your machine.

**Option B — add python to the firewall allow-list manually**

1. Open **System Settings > Network > Firewall**.
2. Click **Options...** (or **Firewall Options...** on older macOS).
3. Click the **+** button, navigate to your Python binary (e.g.
   `/Users/<you>/.venv/bin/python3.12`), select it, and set it to
   **Allow incoming connections**.
4. Click **OK** and relaunch the bridge.

If you use multiple Python versions or virtual environments, each binary may
need to be added separately. The simplest approach is to click Allow at the
prompt each time — macOS remembers the decision per binary path.

---

## 5. `pip install torch` (or pybullet) fails on Python 3.14

### Symptom

You have Python 3.14 installed (or your shell's `python3` points to 3.14) and
you see errors like:

```
ERROR: Could not find a version that satisfies the requirement torch
```

```
ERROR: No matching distribution found for stable-baselines3
```

or a source build is attempted for `torch` that eventually fails with C++
compile errors.

### Cause

Python wheel publishers — including PyTorch, pybullet, and several others in
the RL ecosystem — build and upload new wheels on a delay after each CPython
release. Python 3.14 shipped in 2025 and as of mid-2026, `torch`, `pybullet`,
and some `stable-baselines3` transitive dependencies still do not publish
`cp314` wheels for all platforms. pip falls back to a source build which
requires a compatible C++ compiler and can fail for multiple reasons.

### Fix

Use **Python 3.12**. This is the version the lab is tested and recommended on:

```bash
# Check what version you have:
python3 --version

# Install 3.12 if needed (macOS with Homebrew):
brew install python@3.12

# Create a fresh virtual environment pinned to 3.12:
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

If you also need pybullet, follow the [conda-forge path](#2-pip-install-pybullet-fails-on-apple-silicon)
(conda-forge distributes Python 3.12 arm64 via miniforge).

> **Why not 3.13?** PyTorch 2.x wheels exist for 3.13 on most platforms, so
> 3.13 generally works. However, 3.12 is the version CI runs against and is the
> safest choice if you want to match the lab's tested configuration exactly.

---

## 6. `rl-lab: command not found`

### Symptom

After cloning the repository you type `rl-lab train ...` (or `rl-lab --help`)
and your shell reports:

```
zsh: command not found: rl-lab
bash: rl-lab: command not found
```

### Cause

The `rl-lab` command is a *console script entry point* declared in
`pyproject.toml`. pip installs it into the virtual environment's `bin/`
directory when you run `pip install -e .`. If you have not done that install
step, or if the virtual environment is not active, the command is not on your
`PATH`.

Three sub-cases:

| Sub-case | What happened |
|---|---|
| A | You never ran `pip install -e .` at all. |
| B | You ran it but the venv was not active at the time, so pip installed into the system or a different environment. |
| C | You opened a new terminal tab and forgot to re-activate the venv. |

### Fix

**Step 1 — activate the virtual environment.**

```bash
# Standard venv:
source .venv/bin/activate

# conda:
conda activate <your-env>
```

You should see the environment name in your prompt, e.g. `(.venv)` or
`(rl_lab)`.

**Step 2 — install the package if you have not already.**

```bash
pip install -e .
# or, with all developer tools:
pip install -e ".[dev]"
```

**Step 3 — verify.**

```bash
rl-lab --version
# Should print: rl-lab 0.0.1

rl-lab list --what algos
# Should print the algorithm list.
```

If you switch between multiple Python environments (e.g. a conda env and a
venv) you need to install the package *inside whichever environment is active*
and then activate that environment before running `rl-lab`.

---

## 7. MuJoCo import error: `No module named 'mujoco'`

### Symptom

You pass `--backend mujoco` to a training command (or set
`SIM_BACKEND=mujoco` in your config), and the lab raises:

```
ModuleNotFoundError: No module named 'mujoco'
```

or:

```
ImportError: cannot import name 'MujocoSim' from 'rl_lab.sim'
```

### Cause

`mujoco` is an *optional* dependency. The default `pip install -e .` and
`pip install -e ".[dev]"` do not install it, because most users (especially
newcomers) run the lab on the default `KinematicBackend` (physics-free,
instantly available) and do not need MuJoCo for the first several experiments.

### Fix

Install the `[mujoco]` extra:

```bash
pip install -e ".[mujoco]"
```

This installs the `mujoco` Python bindings from PyPI. Unlike pybullet, the
officially distributed `mujoco` wheel *does* include an arm64 macOS binary, so
the above command works on Apple Silicon with a plain venv and Python 3.12.

Verify the install:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

After that, re-run your command — the MuJoCo backend will be available.

> **Which experiments use MuJoCo?** Experiment 11 (`11_robustify.py`) switches
> between PyBullet and MuJoCo to test policy robustness across simulators. All
> other experiments default to `KinematicBackend` and do not require either
> physics backend to be installed.
