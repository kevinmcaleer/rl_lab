"""Experiment 06 — Generalisation & Domain Randomisation.

CONCEPT
-------
An agent can *memorise* a solution or *learn the task*.  These two look the same
during training — both achieve a high success rate on the targets they saw — but
diverge the moment you move the target somewhere new.

This experiment teaches the difference by training two DQN agents in parallel:

1. **FIXED agent** — the target is always at the same position every episode.
   The agent quickly learns a single hard-coded motion trajectory that reaches
   *that one point*.  Evaluation on unseen targets reveals it has memorised,
   not learned.

2. **RANDOM agent** — the target is re-sampled at a random reachable point on
   every episode reset, and its position is visible in the observation (obs[14:17]
   = normalised tip-to-target vector).  The agent must discover a *general*
   reaching policy — it cannot rely on memorisation because every episode is
   different.

The *generalisation gap* is the difference in success rate between the two agents
when evaluated on a held-out set of unseen targets: a large gap means the fixed
agent memorised; a small gap means the random agent generalised.

DOMAIN RANDOMISATION
--------------------
Randomising the target position is the simplest form of domain randomisation —
deliberately injecting variation at training time so the learned policy is
robust to variation at test time.  It is also the first practical tool on the
road from sim to real: a robot that reaches any reachable target in simulation
is much more likely to handle the positional uncertainty of a real-world scene
than one that was trained to hit a single point.

FROZEN EXPERIMENT INTERFACE
----------------------------
  run(quick, render, seed) -> dict   # main body; quick=True for CI
  main(argv) -> int                  # argparse entry point
  __main__                           # raises SystemExit(main())

Plots (skipped in quick mode) are saved under experiments/_outputs/06_generalisation/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Matplotlib — must use Agg before pyplot is imported so we never try to open
# a display window (required for headless CI and macOS non-main-thread use).
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402 (after backend switch)
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# RL Lab imports — rl_lab registers "BuddyJrReach*" envs on import.
# ---------------------------------------------------------------------------
import rl_lab  # noqa: F401  (side-effect: Gymnasium env registration)
from rl_lab.algos.registry import make_algorithm  # noqa: E402

# ---------------------------------------------------------------------------
# Output directory (created only when not in quick mode)
# ---------------------------------------------------------------------------
_OUTDIR = Path(__file__).parent / "_outputs" / "06_generalisation"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

# Full-run budgets — enough for a clear learning curve on a CPU in a few minutes.
_TRAIN_STEPS_FULL: int = 40_000
_EVAL_EPISODES_FULL: int = 50  # unseen-target evaluation episodes

# Quick-mode (CI smoke) budgets — must finish in a few seconds.
_TRAIN_STEPS_QUICK: int = 600
_EVAL_EPISODES_QUICK: int = 5

# DQN hyper-parameters (shared between both agents).
_DQN_HP: dict[str, Any] = {
    "lr": 1e-3,
    "gamma": 0.99,
    "buffer_size": 20_000,
    "batch_size": 64,
    "target_sync": 500,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 15_000,
    "learning_starts": 500,
    "hidden": (64, 64),
}

# Quick-mode overrides (tiny buffers + very short epsilon schedule).
_DQN_HP_QUICK: dict[str, Any] = {
    **_DQN_HP,
    "buffer_size": 500,
    "batch_size": 32,
    "learning_starts": 50,
    "target_sync": 50,
    "epsilon_decay_steps": 200,
}

# A fixed target used for the FIXED training condition.  We choose a point that
# is reachable and not at the arm's neutral position, so it actually requires
# learning some joint motion.  Coordinates in metres (x, y, z relative to base).
_FIXED_TARGET: np.ndarray = np.array([0.08, 0.06, 0.10], dtype=np.float64)

# Radius shell used for RANDOM training (same as the env's default).
_RANDOM_RADIUS: tuple[float, float] = (0.04, 0.155)

# Tolerance for success (must match the env default to be meaningful).
_SUCCESS_TOL: float = 0.02  # metres (2 cm)


# ---------------------------------------------------------------------------
# Helper: roll out one episode and return (total_return, is_success, final_dist)
# ---------------------------------------------------------------------------


def _rollout(env: gym.Env, algo: Any, *, seed: int | None = None) -> tuple[float, bool, float]:
    """Run one deterministic greedy episode; return (return, success, distance)."""
    obs, info = env.reset(seed=seed)
    ep_return = 0.0
    terminated = truncated = False
    dist = float(info.get("distance", 0.0))
    while not (terminated or truncated):
        action, _ = algo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        dist = float(info.get("distance", dist))
    return ep_return, bool(info.get("is_success", False)), dist


# ---------------------------------------------------------------------------
# Helper: evaluate an agent on N episodes using a *fresh* random-target env
# so the agent truly sees unseen targets.
# ---------------------------------------------------------------------------


def _evaluate(
    algo: Any,
    n_episodes: int,
    *,
    seed: int = 42,
    render: str | None = None,
) -> dict[str, Any]:
    """Evaluate *algo* on ``n_episodes`` random-target episodes.

    A fresh environment with random targets is created for each call, so the
    targets are drawn independently of whatever target distribution was used
    during training.  This is the "unseen-target" evaluation.

    Returns a dict with success_rate, mean_return, mean_dist, and per-episode
    lists (episode_returns, episode_successes, episode_distances).
    """
    # Evaluation always uses a random-target env so we probe generalisation,
    # regardless of what each agent was trained on.
    eval_render_mode = render if render == "foxglove" else None
    env = gym.make(
        "BuddyJrReachDiscrete-v0",
        render_mode=eval_render_mode,
        reward_mode="dense",
        max_steps=200,
    )
    returns: list[float] = []
    successes: list[bool] = []
    distances: list[float] = []
    try:
        for ep in range(n_episodes):
            ep_return, success, dist = _rollout(env, algo, seed=seed + ep)
            returns.append(ep_return)
            successes.append(success)
            distances.append(dist)
    finally:
        env.close()

    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_dist": float(np.mean(distances)),
        "episode_returns": returns,
        "episode_successes": successes,
        "episode_distances": distances,
    }


# ---------------------------------------------------------------------------
# Helper: build and train one agent, collecting a per-episode success log.
# ---------------------------------------------------------------------------


def _train_agent(
    label: str,
    env: gym.Env,
    total_steps: int,
    *,
    seed: int,
    hp: dict[str, Any],
) -> tuple[Any, list[float]]:
    """Train a DQN agent on *env* and return (algo, rolling_success_rates).

    ``rolling_success_rates`` is a list of (episode-number, success_rate) pairs
    sampled via the callback — handy for plotting the learning curve.
    """
    print(f"\n--- Training: {label} ({total_steps:,} steps, seed={seed}) ---")
    start = time.monotonic()

    algo = make_algorithm("dqn", env, seed=seed, **hp)

    # We collect the rolling success rate (window=100 eps) from the callback so
    # we can plot two learning curves and compare them later.
    logged_success: list[float] = []
    logged_steps: list[int] = []

    def _cb(metrics: dict[str, Any]) -> None:
        logged_success.append(float(metrics.get("success_rate", 0.0)))
        logged_steps.append(int(metrics.get("step", 0)))

    algo.train(total_steps, callback=_cb)

    elapsed = time.monotonic() - start
    print(f"    Training done in {elapsed:.1f} s — {len(logged_success)} episodes logged.")
    return algo, logged_success


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_learning_curves(
    fixed_curve: list[float],
    random_curve: list[float],
    outdir: Path,
) -> None:
    """Plot and save the two training learning curves on a shared axis."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(fixed_curve, color="tab:orange", linewidth=1.2, label="Fixed target (memorisation)")
    ax.plot(random_curve, color="tab:blue", linewidth=1.2, label="Random target (generalisation)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling success rate (100-ep window)")
    ax.set_title("Experiment 06 — Learning curves during training")
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = outdir / "learning_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")


def _plot_generalisation_gap(
    fixed_eval: dict[str, Any],
    random_eval: dict[str, Any],
    outdir: Path,
) -> None:
    """Bar chart comparing success rates and mean distances on unseen targets."""
    labels = ["Fixed-target\n(memorised)", "Random-target\n(generalised)"]
    success_rates = [fixed_eval["success_rate"], random_eval["success_rate"]]
    mean_dists = [fixed_eval["mean_dist"], random_eval["mean_dist"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    colours = ["tab:orange", "tab:blue"]
    bars1 = ax1.bar(labels, success_rates, color=colours, edgecolor="black", linewidth=0.7)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Success rate on UNSEEN targets")
    ax1.set_title("Generalisation: success rate")
    for bar, val in zip(bars1, success_rates, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.0%}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    bars2 = ax2.bar(labels, mean_dists, color=colours, edgecolor="black", linewidth=0.7)
    ax2.set_ylabel("Mean final distance to target (m)")
    ax2.set_title("Generalisation: final distance")
    # Mark the success tolerance threshold.
    ax2.axhline(_SUCCESS_TOL, linestyle="--", color="red", linewidth=1, label="success tol (2 cm)")
    ax2.legend(fontsize=9)
    for bar, val in zip(bars2, mean_dists, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.002,
            f"{val*100:.1f} cm",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    gap = random_eval["success_rate"] - fixed_eval["success_rate"]
    fig.suptitle(
        f"Generalisation gap (random − fixed success rate): {gap:+.0%}  "
        f"({'random is better' if gap > 0 else 'similar or fixed won'})",
        fontsize=10,
    )
    fig.tight_layout()
    path = outdir / "generalisation_gap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")


def _plot_per_episode_distances(
    fixed_eval: dict[str, Any],
    random_eval: dict[str, Any],
    outdir: Path,
) -> None:
    """Scatter plot of per-episode final distances on unseen targets."""
    n = len(fixed_eval["episode_distances"])
    xs = np.arange(n)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(
        xs,
        fixed_eval["episode_distances"],
        color="tab:orange",
        s=20,
        alpha=0.7,
        label="Fixed target agent",
    )
    ax.scatter(
        xs,
        random_eval["episode_distances"],
        color="tab:blue",
        s=20,
        alpha=0.7,
        label="Random target agent",
    )
    ax.axhline(_SUCCESS_TOL, linestyle="--", color="red", linewidth=1, label="success tol (2 cm)")
    ax.set_xlabel("Evaluation episode (unseen targets)")
    ax.set_ylabel("Final tip-to-target distance (m)")
    ax.set_title("Experiment 06 — Per-episode distances on unseen targets")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = outdir / "per_episode_distances.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")


# ---------------------------------------------------------------------------
# Foxglove live-tracking comparison hook
# ---------------------------------------------------------------------------


def _foxglove_comparison(
    fixed_algo: Any,
    random_algo: Any,
    n_episodes: int,
    *,
    seed: int,
) -> None:
    """Stream both agents on unseen random targets to Foxglove in sequence.

    This is the "drag-the-target live-tracking" hook described in the issue.
    Each agent plays the same sequence of random-target episodes so you can
    compare them side-by-side in Foxglove.  Watch how the fixed agent ignores
    the target position while the random agent tracks it.

    NOTE: both agents control the *same* Foxglove channel (/robot + /scene) so
    you see one arm at a time; the agent label is printed to stdout so you know
    which is which.  In a Foxglove layout you can open two panels and use MCAP
    replay to overlay them.
    """
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

    labels = ["FIXED-target agent", "RANDOM-target agent"]
    algos = [fixed_algo, random_algo]

    for label, algo in zip(labels, algos, strict=False):
        print(f"\n[Foxglove] Streaming {label} — {n_episodes} episodes …")
        env = gym.make(
            "BuddyJrReachDiscrete-v0",
            render_mode="foxglove",
            reward_mode="dense",
            max_steps=200,
        )
        streamer = FoxgloveStreamer("foxglove")
        try:
            if streamer.app_url:
                print(f"    Open: {streamer.app_url}")
            for ep in range(n_episodes):
                obs, info = env.reset(seed=seed + ep)
                ep_return = 0.0
                terminated = truncated = False
                while not (terminated or truncated):
                    action, _ = algo.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_return += float(reward)
                    streamer.publish(
                        joint_q=np.asarray(info.get("joint_q", [0.0] * 4)),
                        p_ee=np.asarray(info.get("ee_pos", [0.0] * 3)),
                        g=np.asarray(info.get("target", [0.0] * 3)),
                        dist=float(info.get("distance", 0.0)),
                        reward=float(reward),
                        episode_return=ep_return,
                    )
                print(
                    f"    ep {ep + 1:>3}/{n_episodes}  return={ep_return:7.2f}"
                    f"  success={info.get('is_success', False)}"
                )
        finally:
            env.close()
            streamer.close()
        print(f"[Foxglove] {label} done.")


# ---------------------------------------------------------------------------
# Public interface: run()
# ---------------------------------------------------------------------------


def run(
    quick: bool = False,
    render: str | None = None,
    seed: int = 0,
    *,
    no_plot: bool = False,
) -> dict[str, Any]:
    """Train and evaluate two agents: fixed-target vs random-target.

    Parameters
    ----------
    quick:
        When ``True`` use tiny step counts so the function finishes in a few
        seconds (used by the CI smoke test; no plots, no Foxglove).
    render:
        ``"foxglove"`` streams both agents to Foxglove after training; ``None``
        disables visualisation.  ``quick=True`` forces ``render=None``.
    seed:
        Master random seed for reproducibility.
    no_plot:
        When ``True`` skip saving matplotlib plots even if ``quick=False``.
        Useful for a full-budget run without writing image files (e.g. when
        testing on a headless server that lacks write access to _outputs/).

    Returns
    -------
    dict
        A metrics dict with at least:
        ``fixed_success_rate``, ``random_success_rate``, ``generalisation_gap``,
        ``fixed_mean_dist``, ``random_mean_dist``.
    """
    # ------------------------------------------------------------------
    # 0.  Resolve budgets based on quick mode.
    # ------------------------------------------------------------------
    train_steps = _TRAIN_STEPS_QUICK if quick else _TRAIN_STEPS_FULL
    eval_eps = _EVAL_EPISODES_QUICK if quick else _EVAL_EPISODES_FULL
    hp = _DQN_HP_QUICK if quick else _DQN_HP

    if quick:
        render = None  # never open Foxglove in quick mode

    print("=" * 65)
    print("Experiment 06 — Generalisation & Domain Randomisation")
    print("=" * 65)
    print(f"  quick={quick}  train_steps={train_steps:,}  eval_episodes={eval_eps}  seed={seed}")
    print()

    # ------------------------------------------------------------------
    # 1.  Build environments.
    #
    #     FIXED env: we subclass BuddyJrReachEnv and override _sample_target
    #     to always return the same point.  This is intentionally transparent
    #     — a learner can inspect how it works by reading this class.
    #
    #     RANDOM env: uses the standard BuddyJrReachDiscrete-v0 which already
    #     re-samples a random reachable target on every reset.
    # ------------------------------------------------------------------

    # FIXED training environment — same target every episode.
    # We build a plain continuous env and wrap it with DiscretizeBuddyJr.
    from rl_lab.env.buddy_jr_reach_env import BuddyJrReachEnv
    from rl_lab.env.wrappers import DiscretizeBuddyJr

    _fixed_target = _FIXED_TARGET.copy()

    class _FixedTargetEnv(BuddyJrReachEnv):
        """Identical to BuddyJrReachEnv but always places the target at the
        same fixed position so the agent can memorise one trajectory."""

        def _sample_target(self) -> np.ndarray:
            # Ignore the random sampler — always return the fixed point.
            return _fixed_target.copy()

    fixed_raw_env = _FixedTargetEnv(
        reward_mode="dense",
        max_steps=200,
    )
    fixed_env = DiscretizeBuddyJr(fixed_raw_env)

    # RANDOM training environment — re-samples target every episode.
    random_env = gym.make(
        "BuddyJrReachDiscrete-v0",
        reward_mode="dense",
        max_steps=200,
    )

    # ------------------------------------------------------------------
    # 2.  Train both agents.
    # ------------------------------------------------------------------
    fixed_algo, fixed_curve = _train_agent(
        "FIXED target (memorisation)",
        fixed_env,
        train_steps,
        seed=seed,
        hp=hp,
    )
    random_algo, random_curve = _train_agent(
        "RANDOM target (generalisation)",
        random_env,
        train_steps,
        seed=seed + 1,  # different seed so the two runs don't produce identical trajectories
        hp=hp,
    )

    fixed_env.close()
    random_env.close()

    # ------------------------------------------------------------------
    # 3.  Evaluate on unseen targets (random evaluation environment).
    #
    #     Both agents are evaluated on the SAME set of random targets (same
    #     seed), which the fixed agent definitely never saw during training.
    #     This reveals the generalisation gap cleanly.
    # ------------------------------------------------------------------
    print(f"\nEvaluating both agents on {eval_eps} UNSEEN random targets …")
    eval_seed = seed + 999  # clearly out-of-distribution seed
    fixed_eval = _evaluate(fixed_algo, eval_eps, seed=eval_seed, render=None)
    random_eval = _evaluate(random_algo, eval_eps, seed=eval_seed, render=None)

    gap = random_eval["success_rate"] - fixed_eval["success_rate"]

    print(f"\n{'─' * 55}")
    print("RESULTS (unseen targets)")
    print(f"{'─' * 55}")
    print(
        f"  Fixed-target agent  — success: {fixed_eval['success_rate']:.0%}"
        f"   mean dist: {fixed_eval['mean_dist']*100:.1f} cm"
    )
    print(
        f"  Random-target agent — success: {random_eval['success_rate']:.0%}"
        f"   mean dist: {random_eval['mean_dist']*100:.1f} cm"
    )
    print(f"  Generalisation gap  (+random − fixed): {gap:+.0%}")
    print(f"{'─' * 55}")

    if gap > 0:
        print("  > The random-target agent generalises better to unseen targets.")
        print("  > The fixed-target agent memorised ONE trajectory — it has no")
        print("    general reaching policy.")
    elif gap < -0.1:
        print("  > Unusually: the fixed agent did better on unseen targets this run.")
        print("  > Try a larger training budget or a different seed.")
    else:
        print("  > The two agents performed similarly — run longer for a clearer gap.")

    # ------------------------------------------------------------------
    # 4.  Save plots (skipped in quick mode or when no_plot is set).
    # ------------------------------------------------------------------
    if not quick and not no_plot:
        _OUTDIR.mkdir(parents=True, exist_ok=True)
        print("\nSaving plots …")
        _plot_learning_curves(fixed_curve, random_curve, _OUTDIR)
        _plot_generalisation_gap(fixed_eval, random_eval, _OUTDIR)
        _plot_per_episode_distances(fixed_eval, random_eval, _OUTDIR)

    # ------------------------------------------------------------------
    # 5.  Optional Foxglove streaming — compare both agents live.
    # ------------------------------------------------------------------
    if render == "foxglove":
        foxglove_eps = 5 if quick else 10
        _foxglove_comparison(
            fixed_algo,
            random_algo,
            foxglove_eps,
            seed=eval_seed,
        )

    # ------------------------------------------------------------------
    # 6.  Return a compact metrics dict.
    # ------------------------------------------------------------------
    return {
        "fixed_success_rate": fixed_eval["success_rate"],
        "random_success_rate": random_eval["success_rate"],
        "generalisation_gap": gap,
        "fixed_mean_dist": fixed_eval["mean_dist"],
        "random_mean_dist": random_eval["mean_dist"],
        "train_steps": train_steps,
        "eval_episodes": eval_eps,
    }


# ---------------------------------------------------------------------------
# CLI entry point: main()
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and call :func:`run`.

    Flags
    -----
    --quick       Tiny budget; finish in seconds (CI smoke test).
    --render      Enable live visualisation: ``foxglove``.
    --seed INT    Random seed (default 0).
    --no-plot     Skip saving plots even in full mode (useful for quick checks).
    """
    parser = argparse.ArgumentParser(
        prog="06_generalisation",
        description="Exp 06 — Generalisation vs memorisation + domain randomisation.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Use tiny budgets for a fast smoke run (no plots, no Foxglove).",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="Stream live to Foxglove after training (requires foxglove-sdk).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="INT",
        help="Master random seed (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Skip saving matplotlib plots (even in full mode).",
    )
    args = parser.parse_args(argv)

    metrics = run(
        quick=args.quick,
        render=args.render,
        seed=args.seed,
        no_plot=args.no_plot,
    )

    # Friendly summary to stdout (already printed inside run(), but a
    # compact one-liner is useful when the output is piped / logged).
    print(
        f"\nDone. gap={metrics['generalisation_gap']:+.1%}  "
        f"fixed={metrics['fixed_success_rate']:.0%}  "
        f"random={metrics['random_success_rate']:.0%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
