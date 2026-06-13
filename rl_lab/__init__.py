"""Buddy Jr RL Lab — a reinforcement-learning simulation & learning lab.

This package turns a simulated 4-DOF Buddy Jr robot arm into a structured
learning environment for Reinforcement Learning (RL).  It is organised into
six sub-packages that mirror the RL workflow:

    robot/   — URDF loading, joint helpers and (optionally) real-servo deploy
    sim/     — PyBullet physics wrapper and scene management
    env/     — Gymnasium environments (observations, actions, reward shaping)
    algos/   — algorithm helpers and hyper-parameter presets (SB3-backed)
    train/   — training loops, callbacks, and checkpoint utilities
    viz/     — Foxglove / WebSocket bridge for 3-D live visualisation

A convenience ``utils`` sub-package holds shared helpers (seeding, paths, …).

Quick-start
-----------
>>> import rl_lab
>>> print(rl_lab.__version__)
0.0.1

See ``experiments/`` for hands-on RL tutorials and ``docs/PLAN.md`` for the
development roadmap.
"""

from __future__ import annotations

from rl_lab.version import __version__

__all__ = ["__version__"]
