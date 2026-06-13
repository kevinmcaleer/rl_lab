"""Load the easy/medium/hard env presets from ``rl_lab/config/env/*.yaml``.

The presets vary the target distribution, success tolerance, episode length and
reward mode so a learner (or the training CLI) can dial difficulty without
touching code::

    from rl_lab.env.presets import make_env_from_preset
    env = make_env_from_preset("easy")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

PRESET_DIR = Path(__file__).resolve().parents[1] / "config" / "env"


def list_presets() -> list[str]:
    """Names of the available presets (e.g. ``['easy', 'hard', 'medium']``)."""
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def load_env_preset(name: str) -> dict[str, Any]:
    """Load a preset YAML into a plain dict of env kwargs (plus ``env_id``)."""
    import yaml  # lazy: only needed when presets are used

    path = PRESET_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown preset {name!r}; available: {list_presets()}")
    data = yaml.safe_load(path.read_text())
    if "target_radius" in data:  # YAML lists -> tuple the env expects
        data["target_radius"] = tuple(data["target_radius"])
    return data


def make_env_from_preset(name: str, **overrides: Any) -> gym.Env:
    """Build a Gymnasium env configured by a preset, with optional overrides."""
    cfg = load_env_preset(name)
    cfg.update(overrides)
    env_id = cfg.pop("env_id", "BuddyJrReach-v0")
    max_steps = cfg.get("max_steps")
    # Match Gymnasium's TimeLimit to the preset so truncation is consistent.
    return gym.make(env_id, max_episode_steps=max_steps, **cfg)
