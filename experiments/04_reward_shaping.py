"""Experiment 04 — Reward Shaping & the Discretisation Wall.

CONCEPT
-------
This experiment teaches three tightly-linked ideas that every RL practitioner
must encounter first-hand:

1. **Sparse vs. shaped reward** — a sparse signal (+10 only on success, 0
   otherwise) forces the agent to stumble upon the goal before it can learn
   anything. A dense/shaped signal (−distance or potential-based progress)
   guides every step, dramatically accelerating learning.

2. **Reward hacking** — if the reward does not precisely capture what you
   *mean*, the agent will find a policy that maximises the *proxy* while
   violating the *intent*. We demonstrate this with a "high-up" reward that
   produces a confidently-wrong policy.

3. **Curse of dimensionality** — the Q-table grows as bins^dims. Doubling
   the resolution from 9→25 bins (or unfreezing the wrist to add a 4th
   dimension) explodes the state count into millions of cells that will
   never be visited. The agent's learning stalls, motivating function
   approximation (Experiment 5 — DQN).

WHAT RUNS
---------
Four tabular Q-learning runs on *BuddyJrReachDiscrete-v0* (Discrete(9)
jog actions, Box(17) obs). For each run we print the Q-table size and
wall-clock time, then compare learning curves.

  (a) Dense reward  (−distance)                   bins=9,  obs_dims=[14,15,16]
  (b) Sparse reward (+bonus on success only)       bins=9,  obs_dims=[14,15,16]
  (c) Hackable reward (high-up proxy)              bins=9,  obs_dims=[14,15,16]
  (d) Dense reward, high resolution / 4 dims       bins=25, obs_dims=[14,15,16,+extra]
      → shows table explosion

In quick mode only runs (a) and (b) with a tiny budget so the CI test
finishes in a few seconds.

FROZEN EXPERIMENT INTERFACE
---------------------------
run(quick, render, seed) -> dict   — main body, no plots/Foxglove in quick
main(argv)               -> int    — argparse wrapper
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rl_lab.algos.tabular.q_learning import QLearning
from rl_lab.env.wrappers import TabularBuddyJr

# ---------------------------------------------------------------------------
# Output directory — plots land here (never in quick mode).
# ---------------------------------------------------------------------------
_OUTPUT_DIR = Path(__file__).parent / "_outputs" / "04_reward_shaping"


# ---------------------------------------------------------------------------
# Hackable reward wrapper
# ---------------------------------------------------------------------------


class HighUpRewardWrapper(gym.Wrapper):
    """Replace the env reward with a 'be as high as possible' proxy.

    This is a deliberately *broken* reward: the agent will learn to point the
    arm straight up rather than reaching the target. It's a toy version of real
    reward-hacking failures seen in the literature (boat-racing games, grasping
    robots that exploit simulator artefacts, etc.).

    The reward is: +ee_z (end-effector height in metres, roughly 0..0.22).
    No success bonus — success is irrelevant to this signal.
    """

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, _reward, terminated, truncated, info = self.env.step(action)
        # ee_pos is stored in info by BuddyJrReachEnv (raw metres)
        ee_z = float(info.get("ee_pos", [0.0, 0.0, 0.0])[2])
        # Clip to a sane [-1, +1] range so the scale matches the other configs.
        hackable_reward = float(np.clip(ee_z / 0.22, -1.0, 1.0))
        return obs, hackable_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(reward_mode: str, seed: int) -> gym.Env:
    """Return a *BuddyJrReachDiscrete-v0* with the requested reward mode."""
    env = gym.make(
        "BuddyJrReachDiscrete-v0",
        reward_mode=reward_mode,
        max_steps=200,
    )
    env.reset(seed=seed)
    return env


def _make_hackable_env(seed: int) -> gym.Env:
    """Return a discrete reach env with the high-up hackable reward."""
    base = gym.make(
        "BuddyJrReachDiscrete-v0",
        reward_mode="dense",  # underlying mode is overridden by the wrapper
        max_steps=200,
    )
    env = HighUpRewardWrapper(base)
    env.reset(seed=seed)
    return env


def _q_table_bytes(algo: QLearning) -> int:
    """Return the memory footprint of the Q-table in bytes."""
    return algo.q.nbytes


def _train_config(
    label: str,
    env: gym.Env,
    bins: int,
    obs_indices: list[int],
    total_steps: int,
    seed: int,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Wrap *env* with TabularBuddyJr, train Q-learning, return metrics dict.

    Parameters
    ----------
    label:        Human-readable name for console output.
    env:          Gymnasium environment (Discrete(9) action space assumed).
    bins:         Discretisation bins per obs dimension.
    obs_indices:  Which observation dimensions to bin (length 3 or 4).
    total_steps:  Approximate number of env.step() calls.
    seed:         Random seed forwarded to Q-learning.
    verbose:      Print one-liner summary to stdout.
    """
    # Wrap with the tabular discretiser. TabularBuddyJr bins only the
    # requested obs dimensions; the Q-table has bins^len(obs_indices) rows.
    tabular_env = TabularBuddyJr(env, bins=bins, obs_indices=obs_indices)

    n_states = int(tabular_env.observation_space.n)
    n_actions = int(tabular_env.action_space.n)
    table_entries = n_states * n_actions

    # QLearning normally auto-wraps a Box env, but here we pre-wrapped it.
    # We build QLearning with the *already-discrete* env by passing it
    # directly. QLearning's __init__ detects Discrete obs_space and skips
    # re-wrapping.
    algo = QLearning(
        tabular_env,
        seed=seed,
        alpha=0.15,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        # bins is passed here too, but QLearning only uses it if it wraps
        # the env itself — since the obs_space is already Discrete, it won't.
        bins=bins,
    )

    # --- Training -----------------------------------------------------------
    episode_returns: list[float] = []
    success_rates: list[float] = []
    recent_successes: list[float] = []
    _window = 50

    def _cb(info: dict[str, Any]) -> None:
        episode_returns.append(info["episode_return"])
        recent_successes.append(float(info.get("success_rate", 0.0)))
        if len(recent_successes) > _window:
            recent_successes.pop(0)
        success_rates.append(float(np.mean(recent_successes)))

    t0 = time.perf_counter()
    algo.train(total_steps=total_steps, callback=_cb)
    elapsed = time.perf_counter() - t0

    final_sr = float(np.mean(success_rates[-20:])) if len(success_rates) >= 20 else 0.0

    # Count how many Q-table cells were actually visited (non-zero).
    visited = int(np.count_nonzero(algo.q))
    pct_visited = 100.0 * visited / max(1, table_entries)

    if verbose:
        print(
            f"  [{label:30s}] "
            f"states={n_states:>8,d}  table={table_entries:>10,d} entries  "
            f"mem={_q_table_bytes(algo)/1024:.1f} KB  "
            f"visited={pct_visited:5.1f}%  "
            f"sr={final_sr:.3f}  "
            f"t={elapsed:.1f}s"
        )

    env.close()

    return {
        "label": label,
        "bins": bins,
        "n_obs_dims": len(obs_indices),
        "n_states": n_states,
        "n_actions": n_actions,
        "table_entries": table_entries,
        "table_bytes": _q_table_bytes(algo),
        "visited_cells": visited,
        "pct_visited": pct_visited,
        "elapsed_s": elapsed,
        "total_steps": total_steps,
        "episode_returns": episode_returns,
        "success_rates": success_rates,
        "final_success_rate": final_sr,
        "q_table": algo.q,  # kept for optional heatmap plot
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Moving-average smoothing for learning curves."""
    if len(values) < 2:
        return np.array(values, dtype=np.float64)
    k = min(window, len(values))
    return np.convolve(values, np.ones(k) / k, mode="same")


def _plot_learning_curves(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Save a 2-panel figure: per-episode return and rolling success rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 04 — Reward Shaping & the Discretisation Wall", fontsize=13)

    colors = ["#2196F3", "#F44336", "#FF9800", "#4CAF50"]

    for ax, key, ylabel, title in [
        (axes[0], "episode_returns", "Episode return", "Learning curves (per-episode return)"),
        (axes[1], "success_rates", "Success rate (rolling 50 ep)", "Success rate over training"),
    ]:
        for i, res in enumerate(results):
            vals = res[key]
            if not vals:
                continue
            x = np.arange(len(vals))
            ax.plot(
                x, _smooth(vals), label=res["label"], color=colors[i % len(colors)], linewidth=1.8
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "learning_curves.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


def _plot_table_explosion(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Bar chart showing Q-table sizes and % visited for each configuration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    entries = [r["table_entries"] for r in results]
    pct_vis = [r["pct_visited"] for r in results]
    sr = [r["final_success_rate"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Q-table size, coverage, and success rate", fontsize=12)

    bar_kw: dict[str, Any] = {
        "color": ["#2196F3", "#F44336", "#FF9800", "#4CAF50"],
        "edgecolor": "black",
    }

    axes[0].bar(labels, entries, **bar_kw)
    axes[0].set_ylabel("Q-table entries (states × actions)")
    axes[0].set_title("Table size\n(curse of dimensionality)")
    axes[0].set_yscale("log")
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")

    axes[1].bar(labels, pct_vis, **bar_kw)
    axes[1].set_ylabel("% cells visited during training")
    axes[1].set_title("Coverage\n(tiny = most states never seen)")
    axes[1].set_ylim(0, 100)
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")

    axes[2].bar(labels, sr, **bar_kw)
    axes[2].set_ylabel("Final success rate")
    axes[2].set_title("Task performance\n(hackable → near 0)")
    axes[2].set_ylim(0, 1)
    for tick in axes[2].get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")

    plt.tight_layout()
    out_path = out_dir / "table_explosion.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print a human-readable ASCII summary table to stdout."""
    header = (
        f"{'Config':<30s}  {'bins':>4}  {'dims':>4}  {'states':>10}  "
        f"{'entries':>12}  {'KB':>7}  {'visited%':>8}  {'succ_rate':>9}  {'time(s)':>7}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  EXPERIMENT 04 SUMMARY — Reward Shaping & the Discretisation Wall")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        print(
            f"  {r['label']:<28s}  {r['bins']:>4d}  {r['n_obs_dims']:>4d}  "
            f"{r['n_states']:>10,d}  {r['table_entries']:>12,d}  "
            f"{r['table_bytes']/1024:>7.1f}  {r['pct_visited']:>8.2f}%  "
            f"{r['final_success_rate']:>9.3f}  {r['elapsed_s']:>7.1f}"
        )
    print(sep)
    print()
    print("KEY LESSONS:")
    print("  1. Dense reward learns faster (more gradient signal per step).")
    print("  2. Sparse reward learns slowly — the agent rarely finds the bonus.")
    print("  3. Hackable reward achieves high proxy score but ZERO task success.")
    print("  4. More bins / more dims → exponentially larger table + near-0% coverage.")
    print("     This is the curse of dimensionality — the motivation for DQN (Exp 05).")
    print()


# ---------------------------------------------------------------------------
# Foxglove rollout helper
# ---------------------------------------------------------------------------


def _foxglove_rollout(env: gym.Env, algo: QLearning, n_steps: int = 300) -> None:
    """Run a greedy rollout and stream it to Foxglove.

    This is called *only* when render='foxglove' and not in quick mode. It
    requires a Foxglove desktop app open at ws://localhost:8765.
    """
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

    streamer = FoxgloveStreamer("foxglove")
    obs, info = env.reset()
    total_r = 0.0
    for _ in range(n_steps):
        action, _ = algo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += float(reward)
        q = info.get("joint_q", np.zeros(4))
        ee = info.get("ee_pos", np.zeros(3))
        target = info.get("target", np.zeros(3))
        dist = float(info.get("distance", 0.0))
        streamer.publish(q, ee, target, dist, reward=float(reward), episode_return=total_r)
        if terminated or truncated:
            obs, info = env.reset()
            total_r = 0.0
    streamer.close()


# ---------------------------------------------------------------------------
# run() — the frozen experiment interface entry point
# ---------------------------------------------------------------------------


def run(
    quick: bool = False,
    render: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run all reward-shaping and discretisation experiments.

    Parameters
    ----------
    quick:
        If True, use tiny step budgets so the CI smoke test finishes in a
        few seconds. Skips all plotting and Foxglove streaming.
    render:
        "foxglove" to stream a greedy rollout after training. None = no stream.
    seed:
        Master random seed for reproducibility.

    Returns
    -------
    dict
        Compact metrics dict with results for each config.
    """
    # Tuned so the full run completes in ~60-90 s on a laptop CPU.
    # quick=True: ~200 steps each (smoke-test only, no real learning).
    steps_ab = 8_000 if not quick else 200  # dense vs sparse  (3-dim, bins=9)
    steps_c = 6_000 if not quick else 200  # hackable reward   (3-dim, bins=9)
    steps_d = 3_000 if not quick else 200  # big table         (3-dim, bins=25)

    configs_to_run = ["dense", "sparse", "hackable", "big_table"]
    if quick:
        # In quick mode we only run (a) dense and (b) sparse to keep CI fast.
        configs_to_run = ["dense", "sparse"]

    print()
    print("Experiment 04 — Reward Shaping & the Discretisation Wall")
    print("=" * 60)
    print(f"  quick={quick}  render={render}  seed={seed}")
    print()
    print(
        f"  {'Config':<30s}  {'states':>10}  {'entries':>12}  "
        f"{'visited%':>8}  {'sr':>6}  {'time(s)':>7}"
    )
    print("  " + "-" * 78)

    results: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # (a) DENSE reward — −distance each step + success bonus
    # -----------------------------------------------------------------------
    # This is the "good" reward: every step gives a learning signal proportional
    # to how far the tip is from the target. The agent can learn without ever
    # accidentally hitting the goal because the gradient points in the right
    # direction from the very first episode.
    # -----------------------------------------------------------------------
    if "dense" in configs_to_run:
        env_a = _make_env(reward_mode="dense", seed=seed)
        res_a = _train_config(
            "a) Dense (−dist)",
            env_a,
            bins=9,
            obs_indices=[14, 15, 16],
            total_steps=steps_ab,
            seed=seed,
            verbose=True,
        )
        results.append(res_a)

    # -----------------------------------------------------------------------
    # (b) SPARSE reward — +10 only when tip is within 2 cm of target
    # -----------------------------------------------------------------------
    # The agent receives zero reward for hundreds of random steps, then
    # suddenly +10 when it stumbles upon the goal. Early in training this
    # almost never happens, so the Q-table stays nearly zero and the agent
    # learns much slower — or not at all within this budget.
    # -----------------------------------------------------------------------
    if "sparse" in configs_to_run:
        env_b = _make_env(reward_mode="sparse", seed=seed)
        res_b = _train_config(
            "b) Sparse (+bonus on success)",
            env_b,
            bins=9,
            obs_indices=[14, 15, 16],
            total_steps=steps_ab,
            seed=seed,
            verbose=True,
        )
        results.append(res_b)

    # -----------------------------------------------------------------------
    # (c) HACKABLE reward — +ee_z (be as high as possible)
    # -----------------------------------------------------------------------
    # The reward no longer measures progress toward the target; it rewards
    # *height*. The agent confidently learns to point the arm straight up.
    # It will achieve near-zero task success while racking up high proxy scores.
    # This is reward hacking — the agent maximises the metric, not the goal.
    # -----------------------------------------------------------------------
    if "hackable" in configs_to_run:
        env_c = _make_hackable_env(seed=seed)
        res_c = _train_config(
            "c) Hackable (+ee_z, reward hack)",
            env_c,
            bins=9,
            obs_indices=[14, 15, 16],
            total_steps=steps_c,
            seed=seed,
            verbose=True,
        )
        results.append(res_c)

    # -----------------------------------------------------------------------
    # (d) HIGH RESOLUTION — bins=25 (3 dims still)
    # -----------------------------------------------------------------------
    # With 25 bins per dimension the Q-table has 25^3 × 9 = 140,625 entries.
    # That is ~17× larger than the bins=9 table (9^3 × 9 = 6,561 entries).
    # In the same step budget only a tiny fraction of cells will be visited.
    # The agent barely learns anything — the curse of dimensionality in action.
    #
    # Note: we keep obs_indices=[14,15,16] (3 dims) here to isolate the effect
    # of resolution from the effect of adding a 4th dimension. The docstring
    # mentions "9 vs 25 bins (and/or un-freezing the wrist)"; using 4 dims
    # with bins=25 would yield 25^4 × 9 ≈ 3.5M entries — effectively never
    # visited — so we demonstrate the wall with 3 dims at high resolution.
    # -----------------------------------------------------------------------
    if "big_table" in configs_to_run:
        env_d = _make_env(reward_mode="dense", seed=seed)
        res_d = _train_config(
            "d) Dense bins=25 (state explosion)",
            env_d,
            bins=25,
            obs_indices=[14, 15, 16],
            total_steps=steps_d,
            seed=seed,
            verbose=True,
        )
        results.append(res_d)

    # -----------------------------------------------------------------------
    # Print human-readable summary table
    # -----------------------------------------------------------------------
    _print_summary_table(results)

    # -----------------------------------------------------------------------
    # Optional Foxglove rollout — show the dense policy (config a) live
    # -----------------------------------------------------------------------
    if render == "foxglove" and not quick and results:
        best_result = results[0]  # dense policy from config (a)
        print("  [foxglove] Streaming greedy rollout of the dense policy ...")
        print("  Open Foxglove Studio at ws://localhost:8765 to watch.")
        # Re-create the env + trained agent for the live demo.
        demo_env = _make_env(reward_mode="dense", seed=seed)
        demo_tabular_env = TabularBuddyJr(demo_env, bins=9, obs_indices=[14, 15, 16])
        demo_algo = QLearning(demo_tabular_env, seed=seed, bins=9)
        # Restore the trained Q-table from config (a).
        demo_algo.q = best_result["q_table"].copy()
        _foxglove_rollout(demo_env, demo_algo, n_steps=400)

    # -----------------------------------------------------------------------
    # Plotting (skip in quick mode)
    # -----------------------------------------------------------------------
    if not quick:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _plot_learning_curves(results, _OUTPUT_DIR)
        _plot_table_explosion(results, _OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Build and return the compact metrics dict
    # -----------------------------------------------------------------------
    metrics: dict[str, Any] = {}
    for res in results:
        key = res["label"]
        metrics[key] = {
            "bins": res["bins"],
            "n_obs_dims": res["n_obs_dims"],
            "n_states": res["n_states"],
            "table_entries": res["table_entries"],
            "pct_visited": res["pct_visited"],
            "final_success_rate": res["final_success_rate"],
            "elapsed_s": res["elapsed_s"],
        }
    return metrics


# ---------------------------------------------------------------------------
# main() — argparse entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and call :func:`run`.

    Flags
    -----
    --quick      Tiny budget, no plots, no Foxglove (for CI smoke tests).
    --render     {foxglove} — stream a greedy rollout after training.
    --seed       Integer random seed (default 0).
    --no-plot    Skip all matplotlib output even in non-quick mode.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 04 — Reward Shaping & the Discretisation Wall.\n\n"
            "Compares tabular Q-learning under dense, sparse, and hackable "
            "reward functions, and shows the state-space explosion from finer "
            "discretisation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Tiny budget / no plots — used by the CI smoke test.",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="Stream a greedy rollout to Foxglove after training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        dest="no_plot",
        action="store_true",
        default=False,
        help="Skip matplotlib output.",
    )
    args = parser.parse_args(argv)

    # --no-plot is honoured by monkey-patching quick inside run() indirectly:
    # we temporarily rename the plotting functions so they are no-ops.
    if args.no_plot and not args.quick:
        global _plot_learning_curves, _plot_table_explosion  # noqa: PLW0603

        def _plot_learning_curves(_r: Any, _d: Any) -> None:  # type: ignore[misc]
            pass

        def _plot_table_explosion(_r: Any, _d: Any) -> None:  # type: ignore[misc]
            pass

    metrics = run(quick=args.quick, render=args.render, seed=args.seed)

    # Print a concise one-liner per config so the output is readable in logs.
    print("Returned metrics:")
    for label, m in metrics.items():
        print(
            f"  {label}: states={m['n_states']:,d}  visited={m['pct_visited']:.1f}%  "
            f"sr={m['final_success_rate']:.3f}  t={m['elapsed_s']:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
