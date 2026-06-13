"""Env preset loading tests (skipped if PyYAML is unavailable)."""

from __future__ import annotations

import pytest

pytest.importorskip("yaml")

import rl_lab  # noqa: F401  (registers envs)
from rl_lab.env.presets import list_presets, load_env_preset, make_env_from_preset


def test_three_presets_exist() -> None:
    assert set(list_presets()) == {"easy", "medium", "hard"}


def test_preset_fields() -> None:
    cfg = load_env_preset("easy")
    assert cfg["env_id"] == "BuddyJrReach-v0"
    assert isinstance(cfg["target_radius"], tuple) and len(cfg["target_radius"]) == 2
    assert cfg["success_tol"] > 0 and cfg["max_steps"] > 0


@pytest.mark.parametrize("name", ["easy", "medium", "hard"])
def test_make_env_from_preset(name: str) -> None:
    env = make_env_from_preset(name)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()
