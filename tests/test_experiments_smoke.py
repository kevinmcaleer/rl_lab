"""Smoke test: every experiment's ``run(quick=True)`` executes without error.

Each ``experiments/NN_name.py`` exposes ``run(quick, render, seed)``. In quick
mode it trains for a tiny budget with no plots and no Foxglove, so the whole
12-experiment curriculum is exercised in well under a minute on CPU. The files
are loaded by path (their names start with digits, so they are not importable
as normal modules). torch/SB3 are core deps, so this runs in CI's quality job.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

import rl_lab  # noqa: F401,E402  (registers the envs)

_EXPERIMENTS = sorted(
    (Path(__file__).resolve().parent.parent / "experiments").glob("[0-9][0-9]_*.py")
)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_twelve_experiments_present() -> None:
    assert len(_EXPERIMENTS) == 12, [p.name for p in _EXPERIMENTS]


@pytest.mark.parametrize("path", _EXPERIMENTS, ids=lambda p: p.stem)
def test_experiment_runs_quick(path: Path) -> None:
    module = _load(path)
    assert hasattr(module, "run"), f"{path.name} must expose run()"
    result = module.run(quick=True, seed=0)
    assert isinstance(result, dict)
