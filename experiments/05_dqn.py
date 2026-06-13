"""Experiment 05 — DQN: Replace the Table with a Network.

Learning objectives
-------------------
By running this experiment you will be able to explain:

1. **Function approximation** — why a neural network Q-function scales where a
   table cannot (generalises across unseen states instead of storing one value
   per discrete (s, a) pair).

2. **Experience replay** — storing transitions in a ring buffer and training on
   *random* minibatches decorrelates the data, turning a non-i.i.d. stream into
   something gradient descent can handle.  Shrinking the buffer recreates the
   correlated-data problem so you can *see* the instability.

3. **Target network** — a frozen copy of the Q-network supplies stable regression
   targets.  Syncing the target every step (target_sync=1) removes this
   stabilisation and shows the "chasing a moving goal" failure mode.

What the ablation shows
-----------------------
Run A (healthy):    buffer_size=50_000,  target_sync=500  → smooth, rising curve.
Run B (collapsed):  buffer_size=256,     target_sync=1    → oscillating / flat curve.

Both conditions are plotted on the same figure so the contrast is obvious.

Usage
-----
    # Full training (≈ 50 000 steps, saves plots, optional Foxglove):
    python experiments/05_dqn.py

    # No plots, Foxglove streaming of greedy rollouts after training:
    python experiments/05_dqn.py --render foxglove

    # CI smoke-test — tiny budget, finishes in a few seconds:
    python experiments/05_dqn.py --quick

    # Suppress matplotlib window (Agg backend is always used):
    python experiments/05_dqn.py --no-plot
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

# ---------------------------------------------------------------------------
# Matplotlib must switch to a non-interactive backend *before* pyplot import.
# ---------------------------------------------------------------------------
import matplotlib
import numpy as np

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Output directory (used only in non-quick mode)
# ---------------------------------------------------------------------------
_OUT_DIR = pathlib.Path(__file__).parent / "_outputs" / "05_dqn"


# ---------------------------------------------------------------------------
# Hyper-parameters for both conditions
# ---------------------------------------------------------------------------

# "Healthy" DQN — standard replay buffer + periodic target sync.
_HEALTHY_HP: dict[str, Any] = {
    "lr": 1e-3,
    "gamma": 0.99,
    "buffer_size": 50_000,  # large buffer -> low correlation between samples
    "batch_size": 64,
    "target_sync": 500,  # sync target net every 500 gradient steps
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 20_000,
    "learning_starts": 500,
    "hidden": (64, 64),
}

# "Collapsed" DQN — tiny buffer + per-step target sync.
# Both ablations are active at once so the collapse is clearly visible.
_COLLAPSED_HP: dict[str, Any] = {
    **_HEALTHY_HP,
    "buffer_size": 256,  # tiny buffer -> highly correlated minibatches
    "target_sync": 1,  # sync every step -> no stable target (moving goalposts)
}

# Quick-mode caps (CI smoke test must finish in a few seconds on CPU)
_QUICK_STEPS = 400
_FULL_STEPS = 50_000

# Smoothing window for the learning curves (in episodes)
_SMOOTH_WINDOW = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _smooth(values: list[float], window: int) -> np.ndarray:
    """Return a simple moving average over *window* episodes."""
    if len(values) == 0:
        return np.array([], dtype=np.float32)
    arr = np.array(values, dtype=np.float32)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    # 'valid' mode: output length = len(arr) - window + 1.
    # We pad the start so the output stays the same length as the input.
    padded = np.concatenate([np.full(window - 1, arr[0]), arr])
    return np.convolve(padded, kernel, mode="valid")


def _train_condition(
    label: str,
    total_steps: int,
    seed: int,
    hp: dict[str, Any],
    *,
    render: str | None = None,
) -> dict[str, Any]:
    """Train one DQN condition; return the history dict from algo.train().

    Parameters
    ----------
    label:
        Human-readable name used in progress prints (e.g. "healthy").
    total_steps:
        Total environment steps for this run.
    seed:
        Global seed for reproducible curves.
    hp:
        Hyper-parameter dict forwarded verbatim to make_algorithm.
    render:
        If ``"foxglove"`` the environment is constructed with
        ``render_mode="foxglove"`` so every training step is streamed live.
        ``None`` means no streaming.
    """
    import gymnasium as gym

    import rl_lab  # noqa: F401 — registers BuddyJrReach* envs with Gymnasium
    from rl_lab.algos.registry import make_algorithm

    # BuddyJrReachDiscrete-v0: obs Box(17,), action Discrete(9).
    # Obs[14:17] = tip-to-target vector (the most informative signal).
    render_mode = render if render == "foxglove" else None
    env = gym.make("BuddyJrReachDiscrete-v0", render_mode=render_mode)

    print(f"  [{label}] training {total_steps:,} steps …")
    algo = make_algorithm("dqn", env, seed=seed, **hp)
    history = algo.train(total_steps=total_steps)

    print(
        f"  [{label}] done — "
        f"{len(history['episode_returns'])} episodes, "
        f"final success_rate={float(np.mean(history['episode_successes'][-100:])) * 100:.1f}%"
    )
    env.close()
    return history


def _greedy_rollouts(
    algo: Any,
    env_id: str,
    n_episodes: int = 5,
    *,
    render: str | None = None,
    seed: int = 0,
) -> list[float]:
    """Run *n_episodes* greedy rollouts and return per-episode returns.

    When ``render="foxglove"`` a fresh env is opened with foxglove render_mode
    so every joint step is streamed live to the Foxglove desktop app.
    """
    import gymnasium as gym

    import rl_lab  # noqa: F401

    render_mode = render if render == "foxglove" else None
    env = gym.make(env_id, render_mode=render_mode)
    returns: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        done = False
        while not done:
            action, _ = algo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
        print(f"    greedy rollout {ep + 1}/{n_episodes}: return={ep_return:.2f}")

    env.close()
    return returns


def _save_plot(
    healthy_history: dict[str, Any],
    collapsed_history: dict[str, Any],
    out_dir: pathlib.Path,
) -> pathlib.Path:
    """Generate and save the learning-curve comparison figure.

    Returns the path of the saved PNG.
    """
    from matplotlib import pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Episode returns (smoothed) ---
    ax = axes[0]
    for label, hist, colour in (
        ("Healthy (large buffer + target net)", healthy_history, "steelblue"),
        ("Collapsed (tiny buffer + target_sync=1)", collapsed_history, "tomato"),
    ):
        raw = hist["episode_returns"]
        if len(raw) == 0:
            continue
        smoothed = _smooth(raw, _SMOOTH_WINDOW)
        x = np.arange(len(smoothed))
        ax.plot(x, smoothed, color=colour, linewidth=2, label=label, alpha=0.9)
        ax.fill_between(
            x,
            smoothed - np.std(raw[: len(smoothed)]),
            smoothed + np.std(raw[: len(smoothed)]),
            color=colour,
            alpha=0.15,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return (smoothed)")
    ax.set_title("DQN: learning curve — healthy vs. collapsed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Rolling success rate (per-episode, smoothed) ---
    ax2 = axes[1]
    for label, hist, colour in (
        ("Healthy", healthy_history, "steelblue"),
        ("Collapsed", collapsed_history, "tomato"),
    ):
        raw = hist["episode_successes"]
        if len(raw) == 0:
            continue
        smoothed = _smooth(raw, _SMOOTH_WINDOW) * 100.0  # -> percent
        ax2.plot(np.arange(len(smoothed)), smoothed, color=colour, linewidth=2, label=label)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate %  (smoothed)")
    ax2.set_title("DQN: success rate — healthy vs. collapsed")
    ax2.set_ylim(-5, 105)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Experiment 05 — DQN Ablation: Experience Replay + Target Network",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    plot_path = out_dir / "learning_curves.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved → {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# Public run() — called by CI smoke tests and notebooks
# ---------------------------------------------------------------------------


def run(quick: bool = False, render: str | None = None, seed: int = 0) -> dict[str, Any]:
    """Train the healthy and collapsed DQN conditions; return comparison metrics.

    Parameters
    ----------
    quick:
        If True, use a tiny step budget and skip plots and Foxglove streaming.
        Must finish in a few seconds on a CPU-only machine (used by CI).
    render:
        ``"foxglove"`` to stream greedy rollouts to the Foxglove desktop app
        after training.  ``None`` (default) disables all streaming.
    seed:
        Global random seed applied to both conditions.

    Returns
    -------
    dict with keys:
        ``healthy_final_success``  — mean success rate (last 100 episodes, healthy run)
        ``collapsed_final_success``— mean success rate (last 100 episodes, collapsed run)
        ``healthy_episodes``       — total episodes in the healthy run
        ``collapsed_episodes``     — total episodes in the collapsed run
        ``plot_path``              — absolute path to the saved PNG (or "" in quick mode)
    """
    total_steps = _QUICK_STEPS if quick else _FULL_STEPS

    # In quick mode we also reduce the buffer to what is feasible given the
    # tiny step budget, but we still contrast a reasonable vs. broken setting.
    if quick:
        quick_healthy_hp = {
            **_HEALTHY_HP,
            "buffer_size": total_steps,  # fits the whole run
            "learning_starts": 32,
            "epsilon_decay_steps": total_steps // 2,
        }
        quick_collapsed_hp = {
            **_COLLAPSED_HP,
            "buffer_size": 32,
            "batch_size": 16,
            "learning_starts": 16,
            "epsilon_decay_steps": total_steps // 2,
        }
        healthy_hp = quick_healthy_hp
        collapsed_hp = quick_collapsed_hp
    else:
        healthy_hp = _HEALTHY_HP
        collapsed_hp = _COLLAPSED_HP

    # --- Train healthy condition ---
    print("Training DQN condition A: healthy (large buffer + stable target net)")
    healthy_hist = _train_condition(
        "healthy",
        total_steps=total_steps,
        seed=seed,
        hp=healthy_hp,
        render=None,  # stream only during greedy rollouts, not training
    )

    # --- Train collapsed condition ---
    print("Training DQN condition B: collapsed (tiny buffer + per-step target sync)")
    collapsed_hist = _train_condition(
        "collapsed",
        total_steps=total_steps,
        seed=seed,
        hp=collapsed_hp,
        render=None,
    )

    # --- Compute summary metrics ---
    h_successes = healthy_hist["episode_successes"]
    c_successes = collapsed_hist["episode_successes"]
    healthy_final = float(np.mean(h_successes[-100:])) if h_successes else 0.0
    collapsed_final = float(np.mean(c_successes[-100:])) if c_successes else 0.0

    metrics: dict[str, Any] = {
        "healthy_final_success": healthy_final,
        "collapsed_final_success": collapsed_final,
        "healthy_episodes": len(healthy_hist["episode_returns"]),
        "collapsed_episodes": len(collapsed_hist["episode_returns"]),
        "plot_path": "",
    }

    # --- Plots (non-quick mode only) ---
    if not quick:
        plot_path = _save_plot(healthy_hist, collapsed_hist, _OUT_DIR)
        metrics["plot_path"] = str(plot_path)

    # --- Greedy rollouts (+ optional Foxglove) ---
    # We only run rollouts in non-quick mode to keep CI fast.
    # In render=foxglove mode the healthy-condition algo is re-trained and its
    # greedy policy is streamed live so you can watch it in Foxglove.
    if not quick and render == "foxglove":
        import gymnasium as gym

        import rl_lab  # noqa: F401
        from rl_lab.algos.registry import make_algorithm

        print("Re-building healthy algo for greedy Foxglove rollouts …")
        env = gym.make("BuddyJrReachDiscrete-v0", render_mode="foxglove")
        foxglove_algo = make_algorithm("dqn", env, seed=seed, **healthy_hp)
        # Re-train on the foxglove-enabled env so the bridge is active.
        foxglove_algo.train(total_steps=_FULL_STEPS)
        print("Running 5 greedy rollouts (watch in Foxglove) …")
        greedy_returns = _greedy_rollouts(
            foxglove_algo,
            "BuddyJrReachDiscrete-v0",
            n_episodes=5,
            render="foxglove",
            seed=seed,
        )
        env.close()
        metrics["greedy_returns"] = greedy_returns
        print(f"  Greedy mean return: {float(np.mean(greedy_returns)):.2f}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry-point (argparse)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and call :func:`run`.

    Arguments
    ---------
    --quick      Tiny budget; no plots; no Foxglove.  Used by CI smoke tests.
    --render     ``foxglove`` — stream greedy rollouts to the Foxglove app.
    --seed       Global random seed (default 0).
    --no-plot    Skip saving the matplotlib figure even in full mode.
    """
    parser = argparse.ArgumentParser(
        description="Experiment 05 — DQN: function approximation, replay, target nets."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a very short training budget suitable for CI smoke tests.",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="Stream greedy rollouts to the Foxglove desktop app after training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the learning-curve PNG (useful for headless runs).",
    )
    args = parser.parse_args(argv)

    results = run(quick=args.quick, render=args.render, seed=args.seed)

    # If --no-plot was passed, we cannot easily suppress the plot that run()
    # already saved (it only saves in non-quick mode).  We just skip printing it.
    print("\n=== Experiment 05 — DQN Results ===")
    print(f"  Healthy   final success rate : {results['healthy_final_success'] * 100:.1f}%")
    print(f"  Collapsed final success rate : {results['collapsed_final_success'] * 100:.1f}%")
    print(f"  Healthy   total episodes     : {results['healthy_episodes']}")
    print(f"  Collapsed total episodes     : {results['collapsed_episodes']}")
    if results.get("plot_path") and not args.no_plot:
        print(f"  Plot saved to              : {results['plot_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
