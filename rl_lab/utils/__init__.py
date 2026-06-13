"""Shared utilities for rl-lab: seeding, checkpoint paths, and small helpers.

This sub-package provides lightweight, dependency-free helpers that are used
across the robot/, sim/, env/, algos/, train/, and viz/ sub-packages.  Keeping
them here avoids circular imports and makes them easy to unit-test.

Planned helpers (stubs filled in as the package grows):

    seed_everything(seed: int) -> None
        Set NumPy, Python random, and Torch seeds in one call so experiments
        are reproducible.

    checkpoint_dir(run_name: str) -> pathlib.Path
        Return (and create) a consistent ``checkpoints/<run_name>/`` directory
        so every script saves models in the same place.

    latest_checkpoint(run_name: str) -> pathlib.Path | None
        Find the most-recently written ``.zip`` checkpoint for a run so you
        can resume training without hunting through directories.

All helpers will be importable as ``from rl_lab.utils import seed_everything``
once implemented.  See the GitHub issues for the tracking milestone.
"""

from __future__ import annotations

__all__: list[str] = []
