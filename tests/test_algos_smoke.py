"""Smoke tests: every algorithm trains a few steps on a tiny env without error.

The bar (per the M4 epic) is *runs cleanly and seeded*, not *learns well* — the
experiments milestone exercises learning. Tiny step counts keep the whole suite
well under a minute on CPU. torch/SB3 algorithms self-skip if those optional
deps are missing.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import rl_lab  # noqa: F401  (registers envs)
from rl_lab.algos.registry import ALGORITHMS, make_algorithm, recommended_env_id
from rl_lab.utils.seeding import set_global_seed

# Which optional dependency each algo needs (tabular ones need neither).
_NEEDS = {
    "dqn": "torch",
    "reinforce": "torch",
    "ppo_min": "torch",
    "ppo": "stable_baselines3",
    "sac": "stable_baselines3",
    "td3": "stable_baselines3",
    "ddpg": "stable_baselines3",
}

# Tiny, CPU-friendly hyperparameters per algo for a fast smoke run.
_TINY = {
    "qlearning": (300, {}),
    "sarsa": (300, {}),
    "dqn": (
        400,
        {
            "buffer_size": 500,
            "batch_size": 32,
            "learning_starts": 50,
            "target_sync": 50,
            "epsilon_decay_steps": 200,
        },
    ),
    "reinforce": (400, {}),
    "ppo_min": (512, {"n_steps": 256, "n_epochs": 2, "batch_size": 64}),
    "ppo": (512, {"n_steps": 256, "batch_size": 64}),
    "sac": (300, {"buffer_size": 500, "batch_size": 32, "learning_starts": 50}),
    "td3": (300, {"buffer_size": 500, "batch_size": 32, "learning_starts": 50}),
    "ddpg": (300, {"buffer_size": 500, "batch_size": 32, "learning_starts": 50}),
}


def _assert_valid_action(action, space: gym.spaces.Space) -> None:
    if isinstance(space, gym.spaces.Discrete):
        assert 0 <= int(action) < space.n
    else:
        arr = np.asarray(action, dtype=np.float32)
        assert arr.shape == space.shape


@pytest.mark.parametrize("name", ALGORITHMS)
def test_algo_trains_and_predicts(name: str) -> None:
    """Each registered algo trains a few steps and predicts a valid action."""
    if name in _NEEDS:
        pytest.importorskip(_NEEDS[name])
    total, hparams = _TINY[name]
    env = gym.make(recommended_env_id(name), max_steps=20)
    set_global_seed(0, env)
    algo = make_algorithm(name, env, seed=0, **hparams)
    algo.train(total)  # must not raise
    obs, _ = env.reset(seed=0)
    action, _state = algo.predict(obs, deterministic=True)
    _assert_valid_action(action, env.action_space)
    env.close()


def test_registry_contents() -> None:
    assert len(ALGORITHMS) == 9
    assert {"qlearning", "dqn", "ppo", "sac"} <= set(ALGORITHMS)


def test_unknown_algorithm_raises() -> None:
    env = gym.make("BuddyJrReachDiscrete-v0", max_steps=20)
    with pytest.raises(ValueError):
        make_algorithm("nope", env)
    env.close()


def test_qlearning_is_deterministic() -> None:
    """Two seeded Q-learning runs produce identical Q-tables."""

    def run() -> np.ndarray:
        env = gym.make("BuddyJrReachDiscrete-v0", max_steps=20)
        set_global_seed(0, env)
        algo = make_algorithm("qlearning", env, seed=0)
        algo.train(400)
        env.close()
        return np.asarray(algo.q)

    np.testing.assert_array_equal(run(), run())


def test_algo_config_presets_exist() -> None:
    """Each algo has a hyperparameter preset YAML that loads."""
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    cfg_dir = Path(rl_lab.__file__).resolve().parent / "config" / "algo"
    for name in ALGORITHMS:
        path = cfg_dir / f"{name}.yaml"
        assert path.exists(), f"missing preset {path}"
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict) and data
