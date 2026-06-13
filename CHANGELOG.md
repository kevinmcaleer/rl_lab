# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Repository scaffolding** — initial project layout: `rl_lab/` package with
  `robot/`, `sim/`, `env/`, `viz/`, `algos/`, `train/` sub-packages, each with
  an `__init__.py`; `experiments/`, `docs/`, `urdf/`, `tests/` directories.
- **Packaging** — `pyproject.toml` (PEP 621 / setuptools) with core dependencies
  (`numpy`, `gymnasium`, `pybullet`, `stable-baselines3`, `foxglove-sdk`) and
  optional extras `[mujoco]`, `[ros2]`, `[rpi]`, `[dev]`; `rl-lab` console
  entry point wired to `rl_lab.cli:main`.
- **Buddy Jr URDF** — `urdf/buddy_jr.urdf`: 4-DOF revolute-joint arm
  (`base_yaw` / `shoulder_pitch` / `elbow_pitch` / `camera_tilt`), two 80 mm
  links, SG90-compatible joint limits (±1.5708 rad), fixed `camera_link` and
  `camera_optical_frame`; validated for well-formedness, single root, acyclic
  link tree, unit axes, and positive-definite inertias.
- **Curriculum documentation** — `experiments/README.md`: 12-experiment
  progressive RL curriculum from multi-armed bandits through sim-to-real
  deployment; five-part per-experiment structure (Concept, Objective, Build & run,
  Watch for, Aha takeaway).
- **Project plan** — `docs/PLAN.md`: architecture decision (PyBullet default,
  MuJoCo optional behind `SimBackend`), full stack diagram, data-flow description,
  RL environment design, directory layout, and M1–M7 milestone roadmap.
- **CLI stub** — `rl_lab/cli.py`: skeleton `main()` entry point for the `rl-lab`
  command.
- **License** — MIT license (`LICENSE`), copyright 2026 Kevin McAleer.
- **`.gitignore`** — Python, venv, PyBullet, Foxglove, and macOS ignores.
- **Contributing guide** — `CONTRIBUTING.md`: venv setup, editable install,
  pre-commit, test instructions, Conventional Commit style, branch/PR flow, and
  experiment contribution process.
- **Code of Conduct** — `CODE_OF_CONDUCT.md`: Contributor Covenant v2.1,
  enforcement contact kevinmcaleer@gmail.com.
- **Changelog** — `CHANGELOG.md`: this file, Keep a Changelog 1.1.0 format.

---

<!-- next release line — do not edit above, add new sections below -->

[Unreleased]: https://github.com/kevinmcaleer/rl_lab/compare/HEAD...HEAD
