"""Experiment 07 — REINFORCE from Scratch: Policy Gradients and Variance.

Teaches
-------
* The score-function / log-derivative policy-gradient estimator.
* Why raw Monte-Carlo returns produce very noisy gradient estimates.
* How subtracting a learned value baseline (the "advantage") reduces variance
  without introducing bias — the insight behind every modern policy-gradient
  method including PPO.

What this script does
---------------------
Two REINFORCE agents are trained side-by-side on ``BuddyJrReachDiscrete-v0``
(Discrete(9) jog actions, Box(17) observation):

1. **No baseline** — the policy-gradient update is weighted by the raw
   discounted return G_t.  Lucky episodes with many positive steps can push
   probability mass in the wrong direction just because G_t happened to be
   large.  This is the "twitchy" mode you will see in the viewer.

2. **With value baseline** — a second MLP learns V(s) in parallel and the
   update is weighted by the *advantage* A_t = G_t − V(s_t).  Subtracting
   the baseline removes the "noise floor" of the return distribution,
   producing smaller and more consistent gradient steps.  This is the
   "deliberate" mode: the arm moves with more purpose.

After training, four plots are saved under
``experiments/_outputs/07_reinforce/``:

* ``learning_curves.png``    — raw + EMA-smoothed episode-return curves.
* ``advantage_variance.png`` — per-update advantage variance (the key
                               diagnostic: lower = more stable gradients).
* ``success_rate.png``       — rolling success rate (twitchy → deliberate).
* ``comparison.png``         — concise 2×2 summary for at-a-glance comparison.

Frozen experiment interface
---------------------------
  run(quick, render, seed) -> dict
  main(argv)               -> int
  if __name__ == "__main__": raise SystemExit(main())
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import rl_lab  # noqa: F401 — registers BuddyJr envs with Gymnasium
from rl_lab.algos.registry import make_algorithm
from rl_lab.utils.seeding import set_global_seed

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

_ENV_ID = "BuddyJrReachDiscrete-v0"  # Discrete(9) actions, Box(17) obs
_OUT_DIR = Path(__file__).parent / "_outputs" / "07_reinforce"

# Full-run hyperparameters.  Conservative values so the variance difference
# is visible without multi-hour compute.
_FULL_STEPS = 80_000  # env steps per agent
_QUICK_STEPS = 800  # env steps per agent in CI smoke mode
_LR = 5e-4  # Adam learning rate (shared by policy and value)
_GAMMA = 0.99  # discount factor
_HIDDEN = (64, 64)  # hidden-layer sizes for both MLPs
_MAX_EPISODE_STEPS = 150  # episode truncation

# This flag is set to True by main() when --no-plot is passed.
# It lets the frozen run() signature stay unchanged while still supporting
# the --no-plot CLI flag.
_SKIP_PLOTS: bool = False


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------


def _ema(values: list[float] | np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average.

    Parameters
    ----------
    values:
        Sequence of scalar observations.
    alpha:
        Smoothing factor in (0, 1].  Small alpha → heavy smoothing.

    Returns
    -------
    np.ndarray of the same length as *values*.

    The update rule is::

        s_0 = x_0
        s_i = alpha * x_i + (1 - alpha) * s_{i-1}
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    if len(arr) == 0:
        return out
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# Core experiment body
# ---------------------------------------------------------------------------


def run(quick: bool = False, render: str | None = None, seed: int = 0) -> dict[str, Any]:
    """Train REINFORCE with and without a value baseline; return a metrics dict.

    Parameters
    ----------
    quick:
        If ``True`` use a tiny step budget and skip all plotting and Foxglove
        rendering.  This mode must finish in a few seconds on CPU and is the
        path used by the CI smoke test.
    render:
        ``"foxglove"`` to stream the *with-baseline* agent's greedy rollouts
        to Foxglove Studio after training.  ``None`` skips rendering.
    seed:
        Global RNG seed for reproducible curves.

    Returns
    -------
    dict
        ``no_baseline_final_return``, ``no_baseline_final_sr``,
        ``with_baseline_final_return``, ``with_baseline_final_sr``,
        ``no_baseline_avg_adv_var``, ``with_baseline_avg_adv_var``,
        ``var_ratio`` (with_baseline / no_baseline; < 1 confirms baseline helps).
    """
    global _SKIP_PLOTS  # noqa: PLW0603

    total_steps = _QUICK_STEPS if quick else _FULL_STEPS
    do_plot = (not quick) and (not _SKIP_PLOTS)
    do_render = (render == "foxglove") and (not quick)

    # ------------------------------------------------------------------
    # 1. Train — no baseline  (raw discounted return as the weight)
    # ------------------------------------------------------------------
    print(f"\n[07] Training REINFORCE WITHOUT baseline  ({total_steps:,} steps) ...")
    t0 = time.perf_counter()

    env_no = gym.make(_ENV_ID, max_steps=_MAX_EPISODE_STEPS)
    set_global_seed(seed, env_no)
    # baseline=False: use the batch-mean return as a constant (very weak)
    # baseline, so A_t = G_t - mean(G).  The gradient is still unbiased but
    # the variance is much higher than with a learned V(s) baseline.
    algo_no = make_algorithm(
        "reinforce",
        env_no,
        seed=seed,
        lr=_LR,
        gamma=_GAMMA,
        baseline=False,
        hidden=_HIDDEN,
    )
    hist_no: dict[str, list[float]] = algo_no.train(total_steps=total_steps)
    env_no.close()

    t_no = time.perf_counter() - t0
    tail_no = max(1, len(hist_no["episode_return"]) // 5)
    print(
        f"    Done in {t_no:.1f} s  |  "
        f"final return ≈ {np.mean(hist_no['episode_return'][-tail_no:]):.2f}  |  "
        f"SR ≈ {np.mean(hist_no['success_rate'][-tail_no:]):.3f}"
    )

    # ------------------------------------------------------------------
    # 2. Train — with learned value baseline  (advantage = G_t - V(s_t))
    # ------------------------------------------------------------------
    print(f"\n[07] Training REINFORCE WITH baseline     ({total_steps:,} steps) ...")
    t1 = time.perf_counter()

    env_wb = gym.make(_ENV_ID, max_steps=_MAX_EPISODE_STEPS)
    set_global_seed(seed, env_wb)
    # baseline=True: a second MLP learns V(s) via MSE regression onto G_t.
    # The policy gradient is then weighted by A_t = G_t - V(s_t), which
    # has substantially lower variance because the baseline "soaks up" the
    # unconditional expectation of G_t at each state.
    algo_wb = make_algorithm(
        "reinforce",
        env_wb,
        seed=seed,
        lr=_LR,
        gamma=_GAMMA,
        baseline=True,
        hidden=_HIDDEN,
    )
    hist_wb: dict[str, list[float]] = algo_wb.train(total_steps=total_steps)
    env_wb.close()

    t_wb = time.perf_counter() - t1
    tail_wb = max(1, len(hist_wb["episode_return"]) // 5)
    print(
        f"    Done in {t_wb:.1f} s  |  "
        f"final return ≈ {np.mean(hist_wb['episode_return'][-tail_wb:]):.2f}  |  "
        f"SR ≈ {np.mean(hist_wb['success_rate'][-tail_wb:]):.3f}"
    )

    # ------------------------------------------------------------------
    # 3. Summary metrics
    # ------------------------------------------------------------------
    no_ret = float(np.mean(hist_no["episode_return"][-tail_no:]))
    no_sr = float(np.mean(hist_no["success_rate"][-tail_no:]))
    wb_ret = float(np.mean(hist_wb["episode_return"][-tail_wb:]))
    wb_sr = float(np.mean(hist_wb["success_rate"][-tail_wb:]))

    no_avar = float(np.mean(hist_no["advantage_var"])) if hist_no["advantage_var"] else 0.0
    wb_avar = float(np.mean(hist_wb["advantage_var"])) if hist_wb["advantage_var"] else 0.0
    # var_ratio < 1 confirms the baseline reduces advantage variance.
    var_ratio = (wb_avar / no_avar) if no_avar > 1e-12 else float("nan")

    helps = "baseline reduces variance" if var_ratio < 1.0 else "try longer training"
    print(
        f"\n[07] Summary:"
        f"\n    No baseline   — return={no_ret:.2f}  SR={no_sr:.3f}  "
        f"avg_adv_var={no_avar:.4f}"
        f"\n    With baseline — return={wb_ret:.2f}  SR={wb_sr:.3f}  "
        f"avg_adv_var={wb_avar:.4f}"
        f"\n    Variance ratio (with/no): {var_ratio:.4f}  ({helps})"
    )

    metrics: dict[str, Any] = {
        "no_baseline_final_return": no_ret,
        "no_baseline_final_sr": no_sr,
        "with_baseline_final_return": wb_ret,
        "with_baseline_final_sr": wb_sr,
        "no_baseline_avg_adv_var": no_avar,
        "with_baseline_avg_adv_var": wb_avar,
        "var_ratio": var_ratio,
    }

    # ------------------------------------------------------------------
    # 4. Plots  (skipped in quick mode or when --no-plot was passed)
    # ------------------------------------------------------------------
    if do_plot:
        _save_plots(hist_no, hist_wb)

    # ------------------------------------------------------------------
    # 5. Foxglove live rendering of the trained (with-baseline) policy
    # ------------------------------------------------------------------
    if do_render:
        _foxglove_demo(algo_wb, seed=seed)

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_plots(
    hist_no: dict[str, list[float]],
    hist_wb: dict[str, list[float]],
) -> None:
    """Save four diagnostic PNGs to ``_OUT_DIR``."""
    # Set Agg backend before importing pyplot so we never try to open a display
    # window.  This is safe in CI, headless servers, and macOS without a GUI.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Colour palette
    C_NO = "#e05c5c"  # warm red — no-baseline agent (noisy)
    C_NO_S = "#a01a1a"  # dark red — smoothed
    C_WB = "#4c9be8"  # blue — with-baseline agent (smoother)
    C_WB_S = "#0e4d91"  # dark blue — smoothed
    _ALPHA = 0.22  # opacity for raw (noisy) signal

    def _ep(h: dict[str, list[float]]) -> np.ndarray:
        return np.arange(1, len(h["episode_return"]) + 1)

    ep_no = _ep(hist_no)
    ep_wb = _ep(hist_wb)

    # --- Plot 1: Learning curves -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle(
        "REINFORCE Learning Curves — Raw Returns vs. EMA-Smoothed (α = 0.05)",
        fontsize=12,
        fontweight="bold",
    )
    for ax, hist, ep, c_raw, c_sm, lbl in (
        (axes[0], hist_no, ep_no, C_NO, C_NO_S, "No baseline (raw returns)"),
        (axes[1], hist_wb, ep_wb, C_WB, C_WB_S, "With value baseline (advantage)"),
    ):
        raw = np.asarray(hist["episode_return"], dtype=np.float64)
        sm = _ema(raw, alpha=0.05)
        ax.plot(ep, raw, color=c_raw, alpha=_ALPHA, lw=0.7, label="Raw")
        ax.plot(ep, sm, color=c_sm, lw=2.0, label="EMA-smoothed")
        # Annotate standard deviation — larger SD = noisier gradient signal.
        sd = float(np.std(raw))
        ax.text(
            0.97,
            0.04,
            f"σ = {sd:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=c_sm,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.7},
        )
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode return")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = _OUT_DIR / "learning_curves.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {p}")

    # --- Plot 2: Advantage variance ---------------------------------------
    # This is the core diagnostic: a lower advantage variance means the
    # gradient estimator has less noise — updates are more reliable.
    # The baseline should visibly reduce this.
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Advantage Variance Over Training (log scale)\n"
        "Lower = more stable policy-gradient signal",
        fontsize=12,
        fontweight="bold",
    )
    for hist, ep, c_raw, c_sm, lbl in (
        (hist_no, ep_no, C_NO, C_NO_S, "No baseline"),
        (hist_wb, ep_wb, C_WB, C_WB_S, "With baseline"),
    ):
        av = np.asarray(hist["advantage_var"], dtype=np.float64)
        ax.plot(ep, av, color=c_raw, alpha=_ALPHA, lw=0.7)
        ax.plot(ep, _ema(av, 0.05), color=c_sm, lw=2.0, label=lbl)
    # Horizontal lines at the mean values to make the ratio legible.
    mean_no = float(np.mean(hist_no["advantage_var"])) if hist_no["advantage_var"] else 0.0
    mean_wb = float(np.mean(hist_wb["advantage_var"])) if hist_wb["advantage_var"] else 0.0
    ax.axhline(
        mean_no, color=C_NO_S, ls="--", lw=1.0, alpha=0.7, label=f"Mean (no bl) = {mean_no:.3f}"
    )
    ax.axhline(
        mean_wb, color=C_WB_S, ls="--", lw=1.0, alpha=0.7, label=f"Mean (with bl) = {mean_wb:.3f}"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Advantage variance (per update)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = _OUT_DIR / "advantage_variance.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {p}")

    # --- Plot 3: Success rate  (twitchy → deliberate) --------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Rolling Success Rate — Twitchy → Deliberate\n"
        "(high early entropy = random flailing; low late entropy = purposeful reach)",
        fontsize=12,
        fontweight="bold",
    )
    for hist, ep, c_raw, c_sm, lbl in (
        (hist_no, ep_no, C_NO, C_NO_S, "No baseline"),
        (hist_wb, ep_wb, C_WB, C_WB_S, "With baseline"),
    ):
        sr = np.asarray(hist["success_rate"], dtype=np.float64)
        ax.plot(ep, sr, color=c_raw, alpha=_ALPHA, lw=0.7)
        ax.plot(ep, _ema(sr, 0.05), color=c_sm, lw=2.0, label=lbl)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling success rate (last 100 episodes)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = _OUT_DIR / "success_rate.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {p}")

    # --- Plot 4: 2×2 comparison summary ----------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Experiment 07 — REINFORCE: No Baseline (red) vs. With Baseline (blue)",
        fontsize=12,
        fontweight="bold",
    )

    # (0,0) Episode return
    ax = axes[0, 0]
    for hist, ep, c_r, c_s, lbl in (
        (hist_no, ep_no, C_NO, C_NO_S, "No baseline"),
        (hist_wb, ep_wb, C_WB, C_WB_S, "With baseline"),
    ):
        arr = np.asarray(hist["episode_return"], dtype=np.float64)
        ax.plot(ep, arr, color=c_r, alpha=_ALPHA, lw=0.6)
        ax.plot(ep, _ema(arr, 0.05), color=c_s, lw=1.8, label=lbl)
    ax.set_title("Episode Return", fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1) Advantage variance (log)
    ax = axes[0, 1]
    for hist, ep, c_r, c_s, lbl in (
        (hist_no, ep_no, C_NO, C_NO_S, "No baseline"),
        (hist_wb, ep_wb, C_WB, C_WB_S, "With baseline"),
    ):
        arr = np.asarray(hist["advantage_var"], dtype=np.float64)
        ax.plot(ep, arr, color=c_r, alpha=_ALPHA, lw=0.6)
        ax.plot(ep, _ema(arr, 0.05), color=c_s, lw=1.8, label=lbl)
    ax.set_yscale("log")
    ax.set_title("Advantage Variance (log scale)", fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Advantage variance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) Success rate
    ax = axes[1, 0]
    for hist, ep, c_r, c_s, lbl in (
        (hist_no, ep_no, C_NO, C_NO_S, "No baseline"),
        (hist_wb, ep_wb, C_WB, C_WB_S, "With baseline"),
    ):
        arr = np.asarray(hist["success_rate"], dtype=np.float64)
        ax.plot(ep, arr, color=c_r, alpha=_ALPHA, lw=0.6)
        ax.plot(ep, _ema(arr, 0.05), color=c_s, lw=1.8, label=lbl)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Success Rate", fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling SR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) Policy loss (stability indicator — fewer/smaller spikes = better)
    ax = axes[1, 1]
    for hist, ep, c_r, c_s, lbl in (
        (hist_no, ep_no, C_NO, C_NO_S, "No baseline"),
        (hist_wb, ep_wb, C_WB, C_WB_S, "With baseline"),
    ):
        arr = np.asarray(hist["loss"], dtype=np.float64)
        ax.plot(ep, arr, color=c_r, alpha=_ALPHA, lw=0.6)
        ax.plot(ep, _ema(arr, 0.05), color=c_s, lw=1.8, label=lbl)
    ax.set_title("Policy Loss (steadier = more stable)", fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = _OUT_DIR / "comparison.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {p}")
    print(f"\n[07] All plots saved under: {_OUT_DIR}")


# ---------------------------------------------------------------------------
# Foxglove demo
# ---------------------------------------------------------------------------


def _foxglove_demo(algo: Any, *, seed: int = 0, n_episodes: int = 5) -> None:
    """Run the trained (with-baseline) policy live in Foxglove.

    We open the env with ``render_mode="foxglove"`` so the environment
    itself publishes joint angles and the target position to Foxglove via
    its internal FoxgloveStreamer.  The *greedy* (deterministic) policy is
    used so you see the arm's sharpest, most deliberate behaviour.

    For comparison, you can swap *algo* for the no-baseline agent and
    observe the difference in motion smoothness — that contrast is the
    Foxglove teaching point of this experiment.
    """
    print(
        "\n[07] Foxglove demo — open ws://localhost:8765 in Foxglove Studio " "to watch the arm ..."
    )
    env_vis = gym.make(_ENV_ID, render_mode="foxglove", max_steps=_MAX_EPISODE_STEPS)
    set_global_seed(seed, env_vis)

    for ep_idx in range(n_episodes):
        obs, _ = env_vis.reset()
        done = False
        total_r = 0.0
        n_steps = 0
        while not done:
            action, _ = algo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env_vis.step(action)
            total_r += float(reward)
            n_steps += 1
            done = terminated or truncated
        print(
            f"    Episode {ep_idx + 1}/{n_episodes}  |  "
            f"return = {total_r:.2f}  |  steps = {n_steps}"
        )

    env_vis.close()
    print("[07] Foxglove demo complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for experiment 07.

    Examples::

        python experiments/07_reinforce.py                     # full run
        python experiments/07_reinforce.py --quick             # CI smoke test
        python experiments/07_reinforce.py --render foxglove   # live viewer
        python experiments/07_reinforce.py --seed 42           # reproducible
        python experiments/07_reinforce.py --no-plot           # skip matplotlib
    """
    parser = argparse.ArgumentParser(
        prog="07_reinforce",
        description=(
            "Experiment 07: REINFORCE policy gradients — no baseline vs. value baseline.\n"
            "Trains two agents side-by-side and plots learning curves, advantage variance,\n"
            "success rate, and a 2x2 comparison summary."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Smoke-test mode: tiny budget, no plots, no Foxglove.  Completes in < 5 s.",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="After training, stream the with-baseline policy to Foxglove Studio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global RNG seed (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        dest="no_plot",
        help="Skip matplotlib plots (useful on headless servers without a display).",
    )
    args = parser.parse_args(argv)

    # Communicate --no-plot to run() via the module-level flag without
    # altering the frozen run() signature.
    global _SKIP_PLOTS  # noqa: PLW0603
    _SKIP_PLOTS = args.no_plot

    metrics = run(quick=args.quick, render=args.render, seed=args.seed)
    print("\n[07] Final metrics:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
