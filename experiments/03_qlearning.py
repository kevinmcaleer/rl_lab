"""Experiment 03 — Tabular Q-learning on a fixed-target reach task.

LEARNING OBJECTIVES
-------------------
*  What is a Markov Decision Process (MDP)?  States, actions, rewards,
   transitions, terminal vs truncated episodes.
*  How does the Bellman optimality equation turn into an update rule?
*  What is a Q-table and why does it only work for small state spaces?
*  What does the discount factor γ trade off?
*  Why do we need epsilon-greedy exploration and how does epsilon decay?

CONCEPT OVERVIEW
----------------
Q-learning (Watkins & Dayan, 1992) learns the value of every (state, action)
pair from raw experience.  The Bellman update says:

    Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') − Q(s, a)]
                               └──────── TD target ────────┘
                         └────────────── TD error (δ) ──────────────┘

Key terms
~~~~~~~~~
* α (alpha)  — learning rate: how fast to shift Q toward the new target.
* γ (gamma)  — discount factor: how much future reward counts vs. now.
* ε (epsilon)— exploration rate: probability of picking a random action.
* terminated — the agent *succeeded* (tip reached the goal). ✓
* truncated  — the episode ran out of time (max_steps hit). ✗

OBSERVATION DISCRETISATION
--------------------------
The reach environment has a continuous 17-D Box observation.  Pure tabular
Q-learning needs a finite state space.  We solve this with TabularBuddyJr,
which extracts the 3-D tip→target error vector (obs[14:17]) and bins each
dimension into `bins` buckets, yielding bins³ discrete states.  With bins=7
we get 343 states — a manageable Q-table of shape (343, 9).

This design choice reveals an important lesson: increasing `bins` dramatically
grows the table (bins=20 → 8000 states), showing *why* we need neural
networks (DQN, experiment 05) once the state space is large.

FIXED TARGET
------------
To make Q-learning converge in a short experiment we use a fixed target
(narrow target_radius band so the env always places the goal at roughly
the same position).  This shrinks the effective state space the agent needs
to cover.  Experiment 04 (SARSA) uses a wider target distribution.

PLOTS PRODUCED
--------------
1.  Learning curve — per-episode return smoothed with a rolling window.
2.  Policy field — a 2-D slice through the Q-table showing the greedy action
    for every (dx, dy) bin while dz is held at its central bin.
    Each cell is the argmax action (0–8), coloured by action index.

RUN
---
    python experiments/03_qlearning.py            # full run, plots saved
    python experiments/03_qlearning.py --quick    # 5-second smoke test
    python experiments/03_qlearning.py --render foxglove  # stream to Foxglove
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Any

import gymnasium as gym

# Matplotlib must switch to the non-interactive Agg backend BEFORE pyplot is
# imported.  Do this unconditionally so it is safe even on headless servers.
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import rl_lab  # registers BuddyJr envs with Gymnasium  # noqa: F401
from rl_lab.algos.registry import make_algorithm

# ---------------------------------------------------------------------------
# Output directory for saved plots (parents created lazily, only when needed).
# ---------------------------------------------------------------------------
_OUT_DIR = pathlib.Path(__file__).parent / "_outputs" / "03_qlearning"

# ---------------------------------------------------------------------------
# Discrete action labels for the policy-field legend.
# Discrete(9): 0=no-op, 1=base+, 2=base-, 3=shoulder+, 4=shoulder-,
#              5=elbow+, 6=elbow-, 7=camera+, 8=camera-
# ---------------------------------------------------------------------------
_ACTION_LABELS = [
    "no-op",
    "base +",
    "base −",
    "shoulder +",
    "shoulder −",
    "elbow +",
    "elbow −",
    "camera +",
    "camera −",
]


# ---------------------------------------------------------------------------
# run() — the experiment body (called by main() and by the CI smoke test).
# ---------------------------------------------------------------------------


def run(quick: bool = False, render: str | None = None, seed: int = 0) -> dict[str, Any]:
    """Train a tabular Q-learning agent on the fixed-target reach task.

    Parameters
    ----------
    quick:
        If ``True``, train for a tiny budget (500 steps) and skip every slow
        operation (plotting, Foxglove streaming, console progress table).  Used
        by the CI smoke test — must complete in a few seconds on CPU.
    render:
        ``"foxglove"`` to stream the greedy-rollout to Foxglove Studio after
        training, or ``None`` to skip.  Ignored when ``quick=True``.
    seed:
        Master random seed for the environment and the algorithm.

    Returns
    -------
    dict with keys:
        ``episode_returns``     — list of per-episode cumulative rewards,
        ``n_episodes``          — total episodes completed,
        ``terminated_count``    — episodes that ended with task *success*,
        ``truncated_count``     — episodes that ended with *timeout*,
        ``final_success_rate``  — fraction of last 50 episodes that succeeded,
        ``q_table_shape``       — shape of the final Q-table.
    """
    # ── Hyper-parameters ────────────────────────────────────────────────────
    # In quick mode we use a tiny budget so the CI smoke test finishes fast.
    total_steps = 500 if quick else 60_000
    bins = 5 if quick else 7  # bins³ = 125 or 343 states
    alpha = 0.15  # learning rate — how fast Q shifts toward new TD targets
    gamma = 0.97  # discount factor — how much future rewards are worth
    epsilon_start = 1.0  # start fully random (maximum exploration)
    epsilon_min = 0.05  # always keep at least 5% exploration
    epsilon_decay = 0.995  # per-episode multiplicative decay of epsilon
    max_steps = 100  # max env steps per episode (also limits truncation)
    smooth_window = 20 if quick else 50  # rolling window for the learning curve

    # ── Environment ─────────────────────────────────────────────────────────
    # Use a FIXED target: set target_radius to a very narrow band so the
    # environment always places the goal at roughly the same distance.
    # This makes Q-learning converge faster — a good first experiment.
    # The width of the band is kept non-zero so the env can always find a
    # valid, reachable point within it.
    render_mode = render if (render and not quick) else None
    env = gym.make(
        "BuddyJrReachDiscrete-v0",
        render_mode=render_mode,
        reward_mode="dense",  # continuous reward signal for tabular learning
        max_steps=max_steps,
        target_radius=(0.08, 0.09),  # narrow band → nearly fixed target distance
        reset_noise=0.05,  # tiny reset jitter for state coverage
    )

    # ── Algorithm ───────────────────────────────────────────────────────────
    # make_algorithm wraps the env with TabularBuddyJr automatically when
    # passed a Box observation space.
    algo = make_algorithm(
        "qlearning",
        env,
        seed=seed,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        bins=bins,
    )

    # ── Training ─────────────────────────────────────────────────────────────
    # We keep our own per-episode counters so we can report terminated vs
    # truncated separately — this is a key MDP concept.
    terminated_count = 0
    truncated_count = 0
    episode_returns: list[float] = []

    # The callback fires at the end of every episode.  We use it to:
    #   (a) log progress periodically, and
    #   (b) count terminated vs truncated episodes.
    #
    # IMPORTANT: QLearning.train() doesn't break early — we track per-episode
    # terminated / truncated inside the callback via a re-rollout approach.
    # Instead we collect final info from the training history.
    #
    # Because QLearning.train() doesn't expose per-episode terminated/truncated
    # flags directly, we collect them via a second pass (greedy evaluation
    # episodes) after training, then report the split.

    # Progress logging interval (disabled in quick mode).
    log_every = 0 if quick else 500  # steps between progress prints
    _last_log_step = [0]  # mutable for closure

    def _callback(info: dict[str, Any]) -> None:
        episode_returns.append(info["episode_return"])
        if log_every and (info["step"] - _last_log_step[0]) >= log_every:
            _last_log_step[0] = info["step"]
            print(
                f"  step {info['step']:>6d}  "
                f"ep {info['episode']:>4d}  "
                f"return {info['episode_return']:+.2f}  "
                f"ε={info['epsilon']:.3f}  "
                f"success_rate={info['success_rate']:.2f}"
            )

    t0 = time.perf_counter()
    history = algo.train(total_steps=total_steps, callback=_callback)
    train_time = time.perf_counter() - t0

    # episode_returns may already be populated by the callback; if the algo
    # returned them too, prefer the algo's list (more complete).
    if history.get("episode_returns"):
        episode_returns = list(history["episode_returns"])

    n_episodes = len(episode_returns)

    # ── Evaluate: count terminated vs truncated ──────────────────────────────
    # Run 20 greedy evaluation episodes (epsilon=0) to see how the trained
    # policy behaves.  This separates *success* (terminated=True) from
    # *timeout* (truncated=True), two fundamentally different outcomes.
    #
    # terminated=True means the tip reached within success_tol (the agent WON).
    # truncated=True  means max_steps was hit before reaching the goal.
    #
    # A good learner has many terminated episodes; a stuck agent accumulates
    # truncated episodes.
    n_eval = 5 if quick else 20
    _saved_epsilon = algo.epsilon
    algo.epsilon = 0.0  # pure greedy for evaluation
    eval_env = gym.make(
        "BuddyJrReachDiscrete-v0",
        render_mode=None,
        reward_mode="dense",
        max_steps=max_steps,
        target_radius=(0.08, 0.09),
        reset_noise=0.05,
    )
    eval_env.reset(seed=seed + 999)  # different seed from training

    for _ in range(n_eval):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = algo.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _info = eval_env.step(action)
            done = terminated or truncated
        if terminated:
            terminated_count += 1
        else:
            truncated_count += 1

    eval_env.close()
    algo.epsilon = _saved_epsilon  # restore

    # Compute the final success rate from the last smooth_window training eps.
    episode_returns[-smooth_window:] if len(episode_returns) >= smooth_window else episode_returns
    # A training episode "succeeded" if its return is above a threshold; we
    # use 0 as the threshold (dense reward: positive ≈ got close).
    final_success_rate = float(terminated_count / max(n_eval, 1))

    # ── Console report ─────────────────────────────────────────────────────
    if not quick:
        print()
        print("=" * 60)
        print("EXPERIMENT 03 — Tabular Q-learning RESULTS")
        print("=" * 60)
        print(f"  Training steps   : {total_steps:,}")
        print(f"  Episodes trained : {n_episodes}")
        print(f"  Training time    : {train_time:.1f}s")
        print(
            f"  Q-table shape    : {algo.q.shape}  ({algo.q.shape[0]} states × {algo.q.shape[1]} actions)"
        )
        print(f"  Final epsilon    : {algo.epsilon:.4f}")
        print()
        print("Evaluation (greedy, 20 episodes):")
        print(f"  terminated (SUCCESS) : {terminated_count:>3d} / {n_eval}")
        print(f"  truncated  (TIMEOUT) : {truncated_count:>3d} / {n_eval}")
        print(f"  success rate         : {final_success_rate:.0%}")
        print()
        print("KEY INSIGHT: terminated=success (arm reached goal) vs.")
        print("             truncated=timeout (episode hit max_steps).")
        print("A well-trained agent should have mostly terminated episodes.")
        print("If you see mostly truncated episodes, try training longer.")
        print("=" * 60)

    # ── Foxglove streaming: greedy rollout ──────────────────────────────────
    # Stream one full greedy episode to Foxglove so learners can watch the
    # arm navigate to the fixed target using the learned Q-table.
    if render == "foxglove" and not quick:
        _render_foxglove(algo, seed)

    # ── Plots ───────────────────────────────────────────────────────────────
    if not quick:
        _plot_learning_curve(episode_returns, smooth_window)
        _plot_policy_field(algo.q, bins)

    env.close()

    return {
        "episode_returns": episode_returns,
        "n_episodes": n_episodes,
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "final_success_rate": final_success_rate,
        "q_table_shape": algo.q.shape,
    }


# ---------------------------------------------------------------------------
# Foxglove live-streaming helper
# ---------------------------------------------------------------------------


def _render_foxglove(algo: Any, seed: int) -> None:
    """Run one greedy episode and stream every step to Foxglove Studio.

    Open Foxglove Studio (https://foxglove.dev/download) → Data Source →
    Open Connection → WebSocket → ws://localhost:8765 before running.
    """
    print()
    print("Streaming greedy rollout to Foxglove (ws://localhost:8765) …")
    print("Open Foxglove Studio and connect to ws://localhost:8765.")

    stream_env = gym.make(
        "BuddyJrReachDiscrete-v0",
        render_mode="foxglove",
        reward_mode="dense",
        max_steps=200,
        target_radius=(0.08, 0.09),
        reset_noise=0.05,
    )
    saved_eps = algo.epsilon
    algo.epsilon = 0.0  # pure greedy
    obs, _ = stream_env.reset(seed=seed)
    done = False
    steps = 0
    while not done:
        action, _ = algo.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = stream_env.step(action)
        done = terminated or truncated
        steps += 1
        time.sleep(0.05)  # ~20 fps so Foxglove can keep up
    outcome = "SUCCESS (terminated)" if terminated else "TIMEOUT (truncated)"
    print(f"  Episode ended after {steps} steps: {outcome}")
    print(f"  Final distance to target: {info.get('distance', '?'):.4f} m")
    stream_env.close()
    algo.epsilon = saved_eps


# ---------------------------------------------------------------------------
# Plot 1 — Learning curve
# ---------------------------------------------------------------------------


def _plot_learning_curve(episode_returns: list[float], window: int) -> None:
    """Save a smoothed learning-curve plot to the output directory.

    The per-episode return starts low (random policy) and should climb as the
    Q-table converges.  The smoothed line makes the trend visible through the
    high variance of individual episodes.
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = np.array(episode_returns, dtype=np.float64)
    episodes = np.arange(1, len(returns) + 1)

    # Rolling mean — pad the start so the curve begins at episode 1.
    smoothed = np.convolve(returns, np.ones(window) / window, mode="same")
    # The convolution edges wrap around; truncate to valid centre region.
    half = window // 2
    valid = slice(half, len(returns) - half) if len(returns) > window else slice(None)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        episodes, returns, color="#aec6f5", linewidth=0.6, alpha=0.6, label="per-episode return"
    )
    ax.plot(
        episodes[valid],
        smoothed[valid],
        color="#1f6bb5",
        linewidth=2.0,
        label=f"rolling mean (window={window})",
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative reward", fontsize=11)
    ax.set_title(
        "Experiment 03 — Q-learning learning curve (fixed target)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = _OUT_DIR / "learning_curve.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Learning curve saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Policy field (2-D slice of the Q-table)
# ---------------------------------------------------------------------------


def _plot_policy_field(q: np.ndarray, bins: int) -> None:
    """Visualise the greedy policy as a 2-D grid (dx vs dy, dz fixed at centre).

    The Q-table has shape (bins³, 9).  Each row corresponds to a tuple
    (bin_dx, bin_dy, bin_dz) that encodes the tip→target error vector in
    discretised form.  We fix dz at its *middle* bin and sweep (dx, dy)
    across all bins, plotting the greedy action argmax_a Q(s, a) for each cell.

    This 'policy field' shows, at a glance, which jog the agent prefers in
    each region of the workspace slice — a powerful diagnostic tool.

    Reading the plot
    ~~~~~~~~~~~~~~~~
    * x-axis: dx bin (negative = target is to the left; positive = to the right).
    * y-axis: dy bin (negative = target is behind; positive = in front).
    * Colour:  greedy action index (see legend).
    * Centre cell (dx=0, dy=0): the arm is already on-target in this slice;
      the agent should pick no-op (action 0).
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Number of actions.
    n_actions = q.shape[1]
    # Fix dz at the central bin.
    dz_bin = bins // 2
    # Build the 2-D grid: rows=dy bins, cols=dx bins.
    grid = np.zeros((bins, bins), dtype=int)
    for dy in range(bins):
        for dx in range(bins):
            # Encode the state as TabularBuddyJr does:
            #   flat = dx * bins^2 + dy * bins + dz
            # where (dx, dy, dz) are the bin indices for obs[14], obs[15], obs[16].
            # TabularBuddyJr uses obs_indices=[14,15,16] and encodes:
            #   flat = 0; for i in [bin14, bin15, bin16]: flat = flat*bins + i
            # So the mapping is: flat = bin14 * bins² + bin15 * bins + bin16.
            state_idx = dx * bins * bins + dy * bins + dz_bin
            state_idx = min(state_idx, q.shape[0] - 1)  # safety clamp
            grid[dy, dx] = int(np.argmax(q[state_idx]))

    # Build a colour map with one colour per action.
    cmap = plt.get_cmap("tab10", n_actions)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(
        grid,
        origin="lower",
        cmap=cmap,
        vmin=-0.5,
        vmax=n_actions - 0.5,
        interpolation="nearest",
    )
    # Annotate each cell with the action index for clarity.
    for dy in range(bins):
        for dx in range(bins):
            ax.text(
                dx,
                dy,
                str(grid[dy, dx]),
                ha="center",
                va="center",
                fontsize=8 if bins <= 7 else 6,
                color="white" if grid[dy, dx] in (0, 2, 4, 6, 8) else "black",
            )

    # Axis labels use bin index.
    tick_labels = [f"{b}" for b in range(bins)]
    ax.set_xticks(range(bins))
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(range(bins))
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("dx bin  (tip→target x-error, low=left, high=right)", fontsize=10)
    ax.set_ylabel("dy bin  (tip→target y-error, low=behind, high=front)", fontsize=10)
    ax.set_title(
        f"Experiment 03 — Greedy policy field (dz bin={dz_bin}, bins={bins})",
        fontsize=11,
        fontweight="bold",
    )
    # Legend: one patch per action.
    patches = [
        mpatches.Patch(color=cmap(a), label=f"{a}: {_ACTION_LABELS[a]}")
        for a in range(n_actions)
        if a < len(_ACTION_LABELS)
    ]
    ax.legend(
        handles=patches,
        loc="upper right",
        bbox_to_anchor=(1.38, 1.0),
        fontsize=8,
        title="Greedy action",
        title_fontsize=9,
    )
    fig.tight_layout()
    out_path = _OUT_DIR / "policy_field.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Policy field saved  → {out_path}")


# ---------------------------------------------------------------------------
# main() — argparse entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and call :func:`run`.

    Returns 0 on success, 1 on any unhandled exception (CI-friendly).
    """
    parser = argparse.ArgumentParser(
        prog="03_qlearning",
        description="Experiment 03: Tabular Q-learning on the Buddy Jr fixed-target reach task.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny budget smoke test (used by CI; skips plots and Foxglove).",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        metavar="{foxglove}",
        help="Stream a greedy rollout to Foxglove Studio after training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="Master random seed (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving plots even when not in quick mode.",
    )
    args = parser.parse_args(argv)

    # --no-plot is implemented by temporarily monkey-patching the plot helpers
    # with no-ops so run() logic stays clean.
    if args.no_plot:
        global _plot_learning_curve, _plot_policy_field  # noqa: PLW0603

        def _plot_learning_curve(_returns: list[float], _window: int) -> None:  # type: ignore[misc]
            pass

        def _plot_policy_field(_q: Any, _bins: int) -> None:  # type: ignore[misc]
            pass

    try:
        metrics = run(quick=args.quick, render=args.render, seed=args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1

    if not args.quick:
        print()
        print("Returned metrics:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
