"""Experiment NN — <Short Title>.

Concept
-------
<One RL idea being taught — e.g. "exploration vs exploitation".>

Objective
---------
<What the learner can *do* or *explain* after completing this experiment.>

Build & run
-----------
<The concrete task: what env, what algo, what you run and observe.>

    python experiments/NN_name.py            # default: 100 k steps, no render
    python experiments/NN_name.py --quick    # smoke-test mode (few seconds)
    python experiments/NN_name.py --render foxglove   # stream to Foxglove

Watch for
---------
<Specific behaviour in the viewer / plots that signals understanding.>

Aha takeaway
------------
<One-sentence insight.>  Now you understand <X> — <carry-forward message>.
"""

# TODO: Replace the module docstring above with the real concept description.
# Keep each section concise — one paragraph max. This docstring is displayed
# when learners run `help(experiments.NN_name)` and is the skeleton of the
# student-facing README.

from __future__ import annotations  # PEP 563 — required project-wide

import argparse
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# matplotlib: ALWAYS set the non-interactive backend BEFORE importing pyplot.
# The CI and quick mode never have a display; Agg renders to files instead.
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
# ---------------------------------------------------------------------------
# rl_lab imports — importing rl_lab registers the Gymnasium envs.
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402 (import after backend set)

import rl_lab  # noqa: F401  registers BuddyJrReach-v0 etc. with Gymnasium
from rl_lab.algos.registry import make_algorithm, recommended_env_id  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# TODO: Change EXPERIMENT_NAME to the file's actual stem, e.g. "05_dqn".
# This string determines where plots are saved:
#   experiments/_outputs/<EXPERIMENT_NAME>/
EXPERIMENT_NAME: str = "NN_template"

# TODO: Pick the algorithm that best illustrates your concept.
# Valid choices (see rl_lab.algos.registry.ALGORITHMS):
#   tabular/discrete : "qlearning", "sarsa", "dqn", "reinforce", "ppo_min"
#   continuous       : "ppo", "sac", "td3", "ddpg"
ALGO: str = "qlearning"

# Total environment steps for a *full* run.  Smoke-test (quick=True) overrides
# this to a tiny value so CI finishes in a few seconds on CPU.
TOTAL_STEPS_FULL: int = 50_000
TOTAL_STEPS_QUICK: int = 400  # fast enough for a pytest smoke test


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------


def run(
    quick: bool = False,
    render: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the experiment and return a metrics dictionary.

    This is the function called by CI (``quick=True``) and by the CLI via
    :func:`main`.  It must always return a plain ``dict`` with at least:

    * ``"algo"``          — algorithm name string
    * ``"total_steps"``   — steps actually trained
    * ``"mean_return"``   — mean episode return over last 10% of training

    Parameters
    ----------
    quick:
        ``True``  → tiny step budget, skip all plots, skip Foxglove.
        ``False`` → full training run, save plots, stream if render is set.
    render:
        ``None`` (default) or ``"foxglove"`` — live 3-D visualisation.
        Ignored when ``quick=True``.
    seed:
        Master random seed for reproducibility.

    Returns
    -------
    dict
        Metrics that can be logged / asserted on in tests.
    """
    # ------------------------------------------------------------------
    # 0. Decide the step budget
    # ------------------------------------------------------------------
    total_steps = TOTAL_STEPS_QUICK if quick else TOTAL_STEPS_FULL

    # Never stream to Foxglove in quick/CI mode — there is no viewer and no
    # point paying the WebSocket overhead.
    if quick:
        render = None

    # ------------------------------------------------------------------
    # 1. Build the environment
    # ------------------------------------------------------------------
    # TODO: Replace ALGO with your chosen algorithm name, or hard-code an
    # env_id like "BuddyJrReach-v0" if your algorithm fixes the action space.
    #
    # recommended_env_id() returns:
    #   "BuddyJrReachDiscrete-v0"  for tabular/discrete algorithms
    #   "BuddyJrReach-v0"          for continuous algorithms (PPO/SAC/TD3/DDPG)
    env_id = recommended_env_id(ALGO)

    # render_mode drives the Foxglove WebSocket when requested; pass None for
    # headless runs (which is always the case in quick/CI mode).
    env = gym.make(
        env_id,
        render_mode=render,  # None | "foxglove"
        reward_mode="dense",  # TODO: change to "sparse" or "shaped" to explore
        max_steps=200,
    )

    # ------------------------------------------------------------------
    # 2. Build the algorithm
    # ------------------------------------------------------------------
    # TODO: Adjust hyperparameters to match your concept.
    # Each algorithm accepts its own keyword arguments; see the registry
    # docstring or rl_lab/config/algo/*.yaml for the full list.
    #
    # Example for qlearning (tabular, NumPy-only):
    algo = make_algorithm(
        ALGO,
        env,
        seed=seed,
        alpha=0.1,  # learning rate — how quickly we shift Q toward new targets
        gamma=0.99,  # discount factor — how much future rewards are worth
        epsilon=1.0,  # start fully random ...
        epsilon_min=0.05,  # ... decay down to 5 % exploration
        epsilon_decay=0.995,
        bins=7,  # discretisation bins for the tip→target obs vector
    )

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    # The callback fires at the end of every episode with a dict:
    #   {"step", "episode", "episode_return", "epsilon", "success_rate"}
    # Use it to collect history for plotting without polling the algo.
    episode_returns: list[float] = []

    def _callback(info: dict[str, Any]) -> None:
        # TODO: log any extra per-episode values your experiment needs here.
        episode_returns.append(float(info["episode_return"]))

    history = algo.train(total_steps=total_steps, callback=_callback)
    # history is a dict whose keys depend on the algorithm (e.g.
    # "episode_returns", "q_table_shape", "epsilon" for Q-learning).
    _ = history  # TODO: use history for plots or extra logging

    env.close()

    # ------------------------------------------------------------------
    # 4. Plots (only when not in quick mode)
    # ------------------------------------------------------------------
    if not quick and episode_returns:
        _plot_learning_curve(episode_returns)

    # ------------------------------------------------------------------
    # 5. Return a metrics dict
    # ------------------------------------------------------------------
    # Compute mean return over the last 10 % of episodes (at least 1).
    tail = max(1, len(episode_returns) // 10)
    mean_return = float(sum(episode_returns[-tail:]) / tail) if episode_returns else 0.0

    # TODO: Add any experiment-specific metrics you want to assert on in tests
    # or compare across hyperparameter sweeps.
    return {
        "algo": ALGO,
        "total_steps": total_steps,
        "mean_return": mean_return,
        "n_episodes": len(episode_returns),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_learning_curve(episode_returns: list[float]) -> None:
    """Save a smoothed learning curve to the experiment output directory.

    Plot is written to:
        experiments/_outputs/<EXPERIMENT_NAME>/learning_curve.png

    We use a simple moving average so the trend is visible even with noisy
    episodes (RL returns are notoriously high-variance).
    """
    # TODO: Add additional subplots that are specific to your concept.
    # For example, experiment 03 plots the Q-value distribution; experiment 08
    # plots the entropy of the policy distribution over training.

    out_dir = Path("experiments") / "_outputs" / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- smooth with a running average (window = 10 % of episodes, min 5) ---
    window = max(5, len(episode_returns) // 10)
    smoothed = [
        sum(episode_returns[max(0, i - window) : i + 1]) / min(i + 1, window)
        for i in range(len(episode_returns))
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episode_returns, alpha=0.3, color="steelblue", label="raw return")
    ax.plot(smoothed, color="steelblue", linewidth=2, label=f"smoothed (w={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"{EXPERIMENT_NAME} — learning curve")
    ax.legend()
    fig.tight_layout()

    save_path = out_dir / "learning_curve.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)  # free memory — crucial in long training runs
    print(f"[{EXPERIMENT_NAME}] plot saved → {save_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, call :func:`run`, and print the metrics.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=f"Buddy Jr RL Lab — {EXPERIMENT_NAME}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny step budget, no plots, no Foxglove (used by CI smoke tests).",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="Enable live 3-D visualisation (requires Foxglove desktop app).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving plots even in a full run.",
    )
    args = parser.parse_args(argv)

    # Respect --no-plot by running in "quick" mode for plot skipping only.
    # We keep the full step budget unless --quick is also set.
    # TODO: If your experiment adds more CLI flags, parse them here and pass
    # them into run() via a dedicated keyword argument.
    metrics = run(
        quick=args.quick,
        render=args.render,
        seed=args.seed,
    )

    # Pretty-print the result so learners see something informative.
    print(f"\n=== {EXPERIMENT_NAME} results ===")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.4f}")
        else:
            print(f"  {key:20s}: {val}")
    return 0


# ---------------------------------------------------------------------------
# Script guard — required by the FROZEN INTERFACE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
