"""Experiment 09 — Continuous PPO: smooth servo commands.

Concept: Continuous action spaces, action scaling, and smoothness penalties.

This experiment trains two PPO agents on ``BuddyJrReach-v0`` (Box(4) delta-angle
actions) and compares them side by side:

* **buzzy**  — no control penalty (``control_weight=0``).  The agent cares only
  about reaching the target.  In the viewer you see rapid, twitchy oscillations
  around the goal — exactly the kind of high-frequency chatter that strips SG90
  servo gears and burns out the motor.
* **smooth** — a non-zero control penalty (``control_weight=0.1``) is added to
  the dense reward.  The agent learns to trade a small amount of positional
  accuracy for much calmer motion — what real hardware needs.

The output is a side-by-side reward/return curve saved under
``experiments/_outputs/09_ppo_continuous/`` and a console summary showing the
success rate and mean episode return for both variants.

How to run
----------
::

    # Standard run (trains both variants and plots curves)
    python experiments/09_ppo_continuous.py

    # Quick smoke-test (tiny budget, no plots, no Foxglove)
    python experiments/09_ppo_continuous.py --quick

    # Stream the greedy rollouts after training to Foxglove
    python experiments/09_ppo_continuous.py --render foxglove

Key hyperparameters
-------------------
* ``TOTAL_STEPS``  — env steps per variant (default 60 000).
* ``CONTROL_BUZZY`` — control weight for the "buzzy" variant (0 = no penalty).
* ``CONTROL_SMOOTH`` — control weight for the "smooth" variant (0.1 by default).
* ``N_EVAL_EPISODES`` — episodes used to evaluate each trained agent.

Aha takeaway
------------
*Continuous control is what real actuators need, and the reward must encode
"be gentle", not just "be correct".*  Now you understand continuous action
spaces, action scaling, and why smoothness/effort penalties matter for hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Output directory (created lazily when not in quick mode)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_OUTPUT_DIR = _HERE / "_outputs" / "09_ppo_continuous"

# ---------------------------------------------------------------------------
# Training budget (scaled down heavily in quick mode)
# ---------------------------------------------------------------------------
#: Full training budget per variant (env steps).
TOTAL_STEPS: int = 60_000

#: Budget used in quick / CI mode — just enough to confirm nothing raises.
QUICK_STEPS: int = 512

#: Control penalty weight for the "buzzy" agent.  0 = reach only, no gentleness.
CONTROL_BUZZY: float = 0.0

#: Control penalty weight for the "smooth" agent.  0.1 is noticeable but not
#: so large that the agent stops reaching the target entirely.
CONTROL_SMOOTH: float = 0.1

#: Number of evaluation episodes after training.
N_EVAL_EPISODES: int = 20


# ---------------------------------------------------------------------------
# Helper: build an environment with a custom control_weight
# ---------------------------------------------------------------------------


def _make_env(control_weight: float, render_mode: str | None = None) -> Any:
    """Create a ``BuddyJrReach-v0`` env with the given control penalty weight.

    The environment's ``reward_cfg`` is replaced *after* construction because
    ``BuddyJrReachEnv.__init__`` does not expose ``control_weight`` as a
    constructor argument.  Patching is safe: ``reward_cfg`` is a frozen
    dataclass, so we replace the whole object atomically.

    Parameters
    ----------
    control_weight:
        Weight applied to ``-||action||^2`` each step.  0 = buzzy,
        ~0.1 = smooth.
    render_mode:
        ``None`` for headless training; ``"foxglove"`` for live streaming.
    """
    import gymnasium as gym

    import rl_lab  # noqa: F401  (registers the env ids)

    _ = rl_lab
    from rl_lab.env.rewards import RewardConfig

    env = gym.make("BuddyJrReach-v0", render_mode=render_mode)

    # Patch reward config: keep mode=dense and success_tol but override
    # control_weight so the two variants differ *only* in this single knob.
    existing = env.unwrapped.reward_cfg  # type: ignore[union-attr]
    env.unwrapped.reward_cfg = RewardConfig(  # type: ignore[union-attr]
        mode=existing.mode,
        success_tol=existing.success_tol,
        success_bonus=existing.success_bonus,
        step_penalty=existing.step_penalty,
        control_weight=control_weight,
        limit_weight=existing.limit_weight,
        shaping_scale=existing.shaping_scale,
    )
    return env


# ---------------------------------------------------------------------------
# Helper: collect per-episode returns from SB3's ep_info_buffer
# ---------------------------------------------------------------------------


def _harvest_returns(model: Any) -> list[float]:
    """Extract per-episode returns recorded by SB3 during ``model.learn()``.

    SB3 keeps a rolling ``ep_info_buffer`` deque.  We drain it into a list
    so we can plot the learning curve.  In quick mode the buffer may be very
    short or empty — that is fine.
    """
    buf = getattr(model, "ep_info_buffer", None)
    if buf is None or len(buf) == 0:
        return []
    return [float(ep["r"]) for ep in buf if "r" in ep]


# ---------------------------------------------------------------------------
# Helper: evaluate a trained model for N episodes
# ---------------------------------------------------------------------------


def _evaluate(
    model: Any,
    control_weight: float,
    n_episodes: int,
    seed: int,
    render_mode: str | None = None,
) -> dict[str, float]:
    """Roll out *n_episodes* deterministic episodes and return summary metrics.

    A fresh environment is built for evaluation so the training env can be
    closed while evaluation runs (important for Foxglove streaming, where we
    want only one active server at a time).

    Returns
    -------
    dict
        ``{"success_rate": float, "mean_return": float, "mean_action_l2": float}``
        where ``mean_action_l2`` is the mean L2 norm of actions across all steps
        — a proxy for servo effort / jerkiness.
    """
    env = _make_env(control_weight=control_weight, render_mode=render_mode)
    returns: list[float] = []
    successes: list[bool] = []
    action_l2s: list[float] = []

    for ep in range(n_episodes):
        obs, _info = env.reset(seed=seed + ep)
        ep_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            # Track action magnitude as a measure of servo effort.
            action_l2s.append(float(np.linalg.norm(action)))
        returns.append(ep_return)
        successes.append(bool(info.get("is_success", False)))

    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_action_l2": float(np.mean(action_l2s)) if action_l2s else 0.0,
    }


# ---------------------------------------------------------------------------
# Helper: smooth a signal with a simple sliding-window average
# ---------------------------------------------------------------------------


def _smooth(values: list[float], window: int = 20) -> list[float]:
    """Return a sliding-window moving average (same length as input).

    Early elements use a smaller, left-aligned window so the first value is
    never NaN — the output array is always the same length as *values*.
    """
    out = []
    for i, _ in enumerate(values):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(values[lo : i + 1])))
    return out


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------


def run(quick: bool = False, render: str | None = None, seed: int = 0) -> dict[str, Any]:
    """Train and compare two PPO agents (buzzy vs smooth) on BuddyJrReach-v0.

    Parameters
    ----------
    quick:
        When ``True`` use a tiny training budget and skip all plots and Foxglove
        streaming.  This mode is used by the CI smoke test and must finish in a
        few seconds on CPU.
    render:
        ``"foxglove"`` to stream greedy evaluation rollouts to Foxglove Studio
        after training; ``None`` for headless operation.  Ignored when ``quick``
        is True.
    seed:
        Global random seed for reproducibility.

    Returns
    -------
    dict
        Small metrics dict with keys ``buzzy_success_rate``,
        ``smooth_success_rate``, ``buzzy_mean_return``, ``smooth_mean_return``,
        ``buzzy_mean_action_l2``, ``smooth_mean_action_l2``.
    """
    # ------------------------------------------------------------------ #
    # Imports — torch and SB3 are only loaded here (not at module import)  #
    # so ``import experiments.09_ppo_continuous`` stays cheap.             #
    # ------------------------------------------------------------------ #
    from stable_baselines3 import PPO

    total_steps = QUICK_STEPS if quick else TOTAL_STEPS
    eval_episodes = 2 if quick else N_EVAL_EPISODES
    render_eval = None if quick else render  # never render in quick mode

    # ------------------------------------------------------------------ #
    # PPO hyperparameters (CPU-friendly, same for both variants so the
    # only variable is the reward / control penalty).
    # ------------------------------------------------------------------ #
    ppo_kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "n_steps": 256 if quick else 512,
        "batch_size": 64,
        "n_epochs": 4 if quick else 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # small entropy bonus encourages exploration
        "learning_rate": 3e-4,
        "policy_kwargs": {"net_arch": [64, 64]},
        "device": "cpu",
        "verbose": 0,
        "seed": seed,
    }

    # ------------------------------------------------------------------ #
    # Train: buzzy variant (no control penalty)
    # ------------------------------------------------------------------ #
    print("\n--- Training BUZZY agent (control_weight=0) ---")
    env_buzzy = _make_env(control_weight=CONTROL_BUZZY)
    model_buzzy = PPO(env=env_buzzy, **ppo_kwargs)
    model_buzzy.learn(total_timesteps=total_steps, reset_num_timesteps=True)
    returns_buzzy = _harvest_returns(model_buzzy)
    env_buzzy.close()
    print(f"  Collected {len(returns_buzzy)} episode returns during training.")

    # ------------------------------------------------------------------ #
    # Train: smooth variant (with control penalty)
    # ------------------------------------------------------------------ #
    print(f"\n--- Training SMOOTH agent (control_weight={CONTROL_SMOOTH}) ---")
    env_smooth = _make_env(control_weight=CONTROL_SMOOTH)
    model_smooth = PPO(env=env_smooth, **ppo_kwargs)
    model_smooth.learn(total_timesteps=total_steps, reset_num_timesteps=True)
    returns_smooth = _harvest_returns(model_smooth)
    env_smooth.close()
    print(f"  Collected {len(returns_smooth)} episode returns during training.")

    # ------------------------------------------------------------------ #
    # Evaluate both agents
    # ------------------------------------------------------------------ #
    print("\n--- Evaluating trained agents ---")
    # Evaluate buzzy first (no Foxglove), then smooth (optional Foxglove).
    metrics_buzzy = _evaluate(
        model_buzzy, CONTROL_BUZZY, eval_episodes, seed=seed, render_mode=None
    )
    # Optionally stream smooth agent to Foxglove so learners can *see* the
    # difference in motion quality in the 3D viewer.
    metrics_smooth = _evaluate(
        model_smooth, CONTROL_SMOOTH, eval_episodes, seed=seed, render_mode=render_eval
    )

    print(
        f"\n  BUZZY  — success={metrics_buzzy['success_rate']:.0%}  "
        f"mean_return={metrics_buzzy['mean_return']:.1f}  "
        f"mean_action_L2={metrics_buzzy['mean_action_l2']:.3f}"
    )
    print(
        f"  SMOOTH — success={metrics_smooth['success_rate']:.0%}  "
        f"mean_return={metrics_smooth['mean_return']:.1f}  "
        f"mean_action_L2={metrics_smooth['mean_action_l2']:.3f}"
    )
    print(
        "\nNote: a lower mean_action_L2 means calmer motion.  "
        "Smooth should be noticeably lower than buzzy."
    )

    # ------------------------------------------------------------------ #
    # Plot learning curves (skipped in quick mode)
    # ------------------------------------------------------------------ #
    if not quick:
        _plot_curves(
            returns_buzzy=returns_buzzy,
            returns_smooth=returns_smooth,
            metrics_buzzy=metrics_buzzy,
            metrics_smooth=metrics_smooth,
        )

    # ------------------------------------------------------------------ #
    # Return metrics dict
    # ------------------------------------------------------------------ #
    return {
        "buzzy_success_rate": metrics_buzzy["success_rate"],
        "smooth_success_rate": metrics_smooth["success_rate"],
        "buzzy_mean_return": metrics_buzzy["mean_return"],
        "smooth_mean_return": metrics_smooth["mean_return"],
        "buzzy_mean_action_l2": metrics_buzzy["mean_action_l2"],
        "smooth_mean_action_l2": metrics_smooth["mean_action_l2"],
    }


# ---------------------------------------------------------------------------
# Plotting helper (only called when quick=False)
# ---------------------------------------------------------------------------


def _plot_curves(
    returns_buzzy: list[float],
    returns_smooth: list[float],
    metrics_buzzy: dict[str, float],
    metrics_smooth: dict[str, float],
) -> None:
    """Save a two-panel figure comparing the two PPO training runs.

    Panel 1 — Episode return during training.
        Raw returns are plotted as faint dots; a moving-average trend line is
        overlaid so the learning signal is easy to see even before convergence.

    Panel 2 — Mean action L2 norm (servo effort) during evaluation.
        A simple bar chart showing how much calmer the smooth agent is.

    The Agg backend is used so the figure is saved to disk without needing a
    display — safe on headless CI boxes and inside Jupyter.
    """
    import matplotlib

    matplotlib.use("Agg")  # must come before any pyplot import
    import matplotlib.pyplot as plt

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Experiment 09 — Continuous PPO: Buzzy vs Smooth",
        fontsize=13,
        fontweight="bold",
    )

    # ---- Panel 1: training return curves -------------------------------- #
    ax = axes[0]

    # Plot both raw traces and smoothed trend lines.
    for returns, label, color in [
        (returns_buzzy, "Buzzy (no control penalty)", "#e06c75"),
        (returns_smooth, f"Smooth (control_weight={CONTROL_SMOOTH})", "#61afef"),
    ]:
        if not returns:
            continue
        episodes = list(range(1, len(returns) + 1))
        # Raw returns as faint markers.
        ax.plot(episodes, returns, alpha=0.25, color=color, linewidth=0.8)
        # Moving-average trend line.
        trend = _smooth(returns, window=max(1, len(returns) // 10))
        ax.plot(episodes, trend, color=color, linewidth=2.0, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title("Training curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add annotation explaining what to look for.
    ax.annotate(
        "Both agents should reach similar\nfinal returns (goal matters),\n"
        "but the smooth agent gets there\nwith calmer, safer actions.",
        xy=(0.97, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#5c6370",
    )

    # ---- Panel 2: servo effort (action L2) bar chart -------------------- #
    ax2 = axes[1]
    labels = ["Buzzy", "Smooth"]
    values = [metrics_buzzy["mean_action_l2"], metrics_smooth["mean_action_l2"]]
    bar_colors = ["#e06c75", "#61afef"]
    bars = ax2.bar(labels, values, color=bar_colors, width=0.5, edgecolor="white")

    # Annotate bars with the numeric value.
    for bar, val in zip(bars, values, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax2.set_ylabel("Mean action L2 (lower = calmer)")
    ax2.set_title("Servo effort (evaluation)")
    ax2.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1.0)
    ax2.grid(True, axis="y", alpha=0.3)

    # Add success-rate comparison as a text box.
    info_text = (
        f"Buzzy  — success: {metrics_buzzy['success_rate']:.0%}  "
        f"return: {metrics_buzzy['mean_return']:.1f}\n"
        f"Smooth — success: {metrics_smooth['success_rate']:.0%}  "
        f"return: {metrics_smooth['mean_return']:.1f}"
    )
    ax2.text(
        0.5,
        -0.18,
        info_text,
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#5c6370",
        family="monospace",
    )

    fig.tight_layout()
    out_path = _OUTPUT_DIR / "curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, call :func:`run`, print metrics, return exit code.

    Arguments
    ---------
    --quick
        Tiny budget, no plots, no Foxglove. Used by the CI smoke test.
    --render {foxglove}
        Stream the greedy smooth-agent evaluation rollouts to Foxglove Studio
        after training.
    --seed INT
        Global random seed (default 0).
    --no-plot
        Skip saving the matplotlib figure even in full (non-quick) mode.
    """
    parser = argparse.ArgumentParser(
        description="Exp 09: Continuous PPO — buzzy vs smooth servo commands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny training budget, no plots, no Foxglove (CI smoke test mode).",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="Stream greedy evaluation to Foxglove Studio after training.",
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
        help="Skip the matplotlib output even in full mode.",
    )
    args = parser.parse_args(argv)

    # --no-plot is implemented by monkey-patching the plotting function off.
    if args.no_plot:
        global _plot_curves  # noqa: PLW0603
        _plot_curves = lambda **_kw: None  # noqa: E731

    metrics = run(quick=args.quick, render=args.render, seed=args.seed)

    print("\n=== Experiment 09 summary ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
