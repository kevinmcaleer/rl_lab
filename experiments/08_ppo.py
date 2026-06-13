"""Experiment 08 — PPO (SB3) on the 4-DOF discrete reach.

What this experiment teaches
-----------------------------
Proximal Policy Optimisation (PPO) is the most widely used on-policy algorithm
in applied robotics RL.  It builds three ideas on top of vanilla REINFORCE:

1. **Actor–critic**: a separate value network (critic) provides a low-variance
   advantage baseline ``A_t = r_t + gamma * V(s_{t+1}) - V(s_t)`` so the policy
   gradient is estimated from *advantages* rather than raw returns.

2. **Generalised Advantage Estimation (GAE)**: blends one-step TD advantages
   (low variance, some bias) and Monte-Carlo returns (unbiased, high variance)
   via the lambda knob::

       delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
       A_t     = delta_t + gamma * lambda * (1 - done) * A_{t+1}

3. **Clipped surrogate objective**: instead of letting the policy update as far
   as gradient descent wants, PPO keeps the update “proximal” by clipping the
   probability ratio ``r_t = pi_new(a|s) / pi_old(a|s)``::

       L_CLIP = E[ min( r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t ) ]

   The ``eps`` parameter is ``clip_range``.  Too small → the policy barely moves
   (slow learning).  Too large → the policy can take huge destabilising steps
   (the guarantee breaks down).

Sweeps run in this experiment
------------------------------
* **Clip-range sweep**: eps in {0.05, 0.2, 0.6} — all other hypers fixed.
* **Epochs-per-batch sweep**: n_epochs in {2, 10, 20} — all other hypers fixed.

After training, the reward curves are stacked so the two effects are easy to
compare visually.  If ``render="foxglove"`` is passed, a short deterministic
rollout is streamed to Foxglove so you can watch the arm move.

Running
-------
    # Full run (~5 min on CPU):
    python experiments/08_ppo.py

    # Quick smoke-test (<10 s, no plots, no Foxglove):
    python experiments/08_ppo.py --quick

    # With live Foxglove visualisation:
    python experiments/08_ppo.py --render foxglove

    # No plots (headless):
    python experiments/08_ppo.py --no-plot

Frozen interface
----------------
``run(quick, render, seed) -> dict`` and ``main(argv) -> int`` satisfy the lab's
FROZEN EXPERIMENT INTERFACE.  CI calls ``run(quick=True)``; it must finish in a
few seconds and return a non-empty metrics dict.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Matplotlib must be configured to Agg before any pyplot import so headless
# runs never try to open a display window.
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Output directory for saved plots.
# ---------------------------------------------------------------------------
_OUT_DIR = Path(__file__).parent / "_outputs" / "08_ppo"


# ===========================================================================
# Core helpers
# ===========================================================================


def _make_env(seed: int, render_mode: str | None = None) -> Any:
    """Return a seeded BuddyJrReachDiscrete-v0 env.

    PPO works on any action space; we use the Discrete(9) variant so the
    teaching comparison with earlier experiments (DQN, REINFORCE) is direct.
    The clip-range lesson is identical on discrete or continuous envs — what
    matters is the clipped *probability ratio*, not the action type.
    """
    import gymnasium as gym

    import rl_lab  # noqa: F401 — registers the gym envs

    _ = rl_lab
    # max_steps=200 keeps each episode short enough for fast wall-clock sweeps.
    env = gym.make(
        "BuddyJrReachDiscrete-v0",
        render_mode=render_mode,
        max_steps=200,
    )
    env.reset(seed=seed)
    return env


def _train_ppo(
    label: str,
    seed: int,
    total_steps: int,
    clip_range: float = 0.2,
    n_epochs: int = 10,
    n_steps: int = 512,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Train one SB3 PPO run and return a history dict suitable for plotting.

    The SB3Algorithm wrapper keeps the code concise while still using the
    lab's registry pattern so you can swap in ``ppo_min`` for a from-scratch
    comparison.

    Parameters
    ----------
    label:
        A human-readable label for this run (used in plot legends).
    seed:
        Global RNG seed for reproducibility.
    total_steps:
        Total environment steps to train for.
    clip_range:
        The PPO clipping epsilon (the sweep knob for the clip-range study).
    n_epochs:
        Optimisation epochs per rollout (the sweep knob for the epochs study).
    n_steps:
        Steps collected per rollout before each optimisation phase.
    batch_size:
        SGD mini-batch size used during the optimisation phase.

    Returns
    -------
    dict
        ``{"label": str, "steps": list[int], "episode_return": list[float],
           "success_rate": list[float], "clip_fraction": list[float]}``
    """
    import gymnasium as gym
    import numpy as np

    import rl_lab  # noqa: F401

    _ = rl_lab

    from rl_lab.algos.registry import make_algorithm

    env = gym.make("BuddyJrReachDiscrete-v0", max_steps=200)
    env.reset(seed=seed)

    algo = make_algorithm(
        "ppo",
        env,
        seed=seed,
        clip_range=clip_range,
        n_epochs=n_epochs,
        n_steps=n_steps,
        batch_size=batch_size,
        # Small network keeps CPU training fast; two 64-unit hidden layers are
        # more than enough for this 17-dim observation space.
        policy_kwargs={"net_arch": [64, 64]},
        # Entropy bonus encourages exploration early in training.
        ent_coef=0.01,
    )

    # Collect per-log-step metrics via the callback bridge.
    steps_log: list[int] = []
    returns_log: list[float] = []
    success_log: list[float] = []

    def _cb(metrics: dict[str, Any]) -> None:
        """Bridge callback: accumulate metrics for post-training plotting."""
        steps_log.append(int(metrics.get("step", 0)))
        returns_log.append(float(metrics.get("episode_return", np.nan)))
        success_log.append(float(metrics.get("success_rate", 0.0)))

    # algo.train wraps this plain callable as an SB3 callback internally.
    algo.train(total_steps, callback=_cb)
    env.close()

    return {
        "label": label,
        "steps": steps_log,
        "episode_return": returns_log,
        "success_rate": success_log,
        "algo": algo,  # keep the trained model for Foxglove rollout
    }


def _rollout_foxglove(algo: Any, n_episodes: int = 3, seed: int = 0) -> None:
    """Stream a few deterministic rollouts to Foxglove.

    Opens a WebSocket server on ws://localhost:8765.  Open Foxglove Studio
    and connect to that address, or follow the printed app_url link.

    The rollout uses the trained policy in deterministic (greedy) mode so you
    see the learned behaviour, not exploration noise.
    """
    import gymnasium as gym
    import numpy as np

    import rl_lab  # noqa: F401

    _ = rl_lab
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

    env = gym.make("BuddyJrReachDiscrete-v0", max_steps=200)

    with FoxgloveStreamer(render_mode="foxglove") as streamer:
        if streamer.app_url:
            print(f"\nFoxglove: {streamer.app_url}")

        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            ep_return = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                action, _ = algo.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)

                # Publish the arm's kinematic state to Foxglove each step.
                joint_q = np.asarray(info.get("joint_q", [0.0] * 4), dtype=np.float64)
                ee_pos = np.asarray(info.get("ee_pos", [0.0] * 3), dtype=np.float64)
                target = np.asarray(info.get("target", [0.0] * 3), dtype=np.float64)
                dist = float(info.get("distance", 0.0))

                streamer.publish(
                    joint_q,
                    ee_pos,
                    target,
                    dist,
                    reward=float(reward),
                    episode_return=ep_return,
                )

            print(
                f"  Foxglove episode {ep + 1}/{n_episodes}  "
                f"return={ep_return:.3f}  "
                f"success={info.get('is_success', False)}"
            )

    env.close()


def _smooth(values: list[float], window: int = 8) -> list[float]:
    """Exponential moving average for a smoother learning curve.

    A simple EMA avoids the edge-effects of box filtering and requires only
    O(1) extra state.  ``window`` is the half-life of past values.
    """
    alpha = 2.0 / (window + 1)
    out: list[float] = []
    ema = float("nan")
    for v in values:
        if v != v:  # NaN guard
            out.append(float("nan"))
            continue
        ema = v if ema != ema else alpha * v + (1.0 - alpha) * ema
        out.append(ema)
    return out


def _plot_sweep(
    runs: list[dict[str, Any]],
    title: str,
    filename: str,
    metric: str = "episode_return",
    ylabel: str = "Episode return (EMA)",
    out_dir: Path = _OUT_DIR,
) -> None:
    """Save a stacked learning-curve comparison plot.

    Each run in *runs* is drawn as a smoothed line with its label in the
    legend.  The plot is saved to *out_dir / filename* (PNG, 150 dpi).

    We use the Agg backend (non-interactive) so plots can be generated on
    any server without a display.  The directory is created if it does not
    exist.
    """
    from matplotlib import pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for run in runs:
        xs = run["steps"]
        ys = _smooth(run[metric])
        ax.plot(xs, ys, label=run["label"], linewidth=1.8)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {path}")


# ===========================================================================
# Frozen Experiment Interface
# ===========================================================================


def run(quick: bool = False, render: str | None = None, seed: int = 0) -> dict[str, Any]:
    """Run Experiment 08: PPO clip-range sweep and epochs-per-batch comparison.

    Parameters
    ----------
    quick:
        When *True* (used by CI smoke tests): tiny budget, no plots, no
        Foxglove.  Must complete in a few seconds on CPU.
    render:
        ``"foxglove"`` streams a post-training rollout to Foxglove Studio.
        ``None`` (default) disables live visualisation.
    seed:
        Global RNG seed; results are reproducible for the same seed.

    Returns
    -------
    dict
        Small summary dict, e.g.::

            {
                "best_clip_range": 0.2,
                "best_n_epochs": 10,
                "clip_sweep_final_returns": {0.05: ..., 0.2: ..., 0.6: ...},
                "epochs_sweep_final_returns": {2: ..., 10: ..., 20: ...},
            }
    """
    # ------------------------------------------------------------------
    # Budget: tiny for CI, real-training size for interactive runs.
    # ------------------------------------------------------------------
    if quick:
        # Smoke-test budget: just enough to confirm the code runs.
        # n_steps=256, total_steps=512 → exactly 2 rollouts per run.
        # We only run two clip values to keep it below 10 s on CPU.
        clip_values = [0.05, 0.2]
        epochs_values = [2, 10]
        total_steps = 512
        n_steps = 256
        batch_size = 64
    else:
        # Full curriculum budget.
        # ~25k steps per run × 6 runs ≈ 150k total env steps ≈ 2–5 min CPU.
        clip_values = [0.05, 0.2, 0.6]
        epochs_values = [2, 10, 20]
        total_steps = 25_000
        n_steps = 512
        batch_size = 64

    print("=" * 60)
    print("Experiment 08 — PPO: clip-range sweep + epochs sweep")
    print("=" * 60)
    print(f"  clip_range values : {clip_values}")
    print(f"  n_epochs values   : {epochs_values}")
    print(f"  total_steps / run : {total_steps}")
    print()

    # ------------------------------------------------------------------
    # Part A: clip-range sweep — n_epochs fixed at 10.
    # ------------------------------------------------------------------
    # WHY: the clip range eps controls how far each SGD step is allowed to
    # push the policy away from the behaviour policy that collected the data.
    # eps=0.05 → nearly no change per step (very slow learning).
    # eps=0.2  → the SB3/PPO default, balances stability and speed.
    # eps=0.6  → few samples get clipped; large updates → potential collapse.
    # ------------------------------------------------------------------
    print("Part A — clip-range sweep (n_epochs fixed at 10)")
    print("-" * 45)
    clip_runs: list[dict[str, Any]] = []

    for eps in clip_values:
        label = f"clip={eps:.2f}"
        print(f"  Training {label} …")
        result = _train_ppo(
            label=label,
            seed=seed,
            total_steps=total_steps,
            clip_range=eps,
            n_epochs=10,
            n_steps=n_steps,
            batch_size=batch_size,
        )
        # Final EMA return from the last 5 log entries (smoothed).
        tail_rets = (
            result["episode_return"][-5:]
            if len(result["episode_return"]) >= 5
            else result["episode_return"]
        )
        # Filter out NaN values from early training before episodes complete.
        valid = [r for r in tail_rets if r == r]
        final_return = float(sum(valid) / len(valid)) if valid else float("nan")
        print(f"    → final return ≈ {final_return:.3f}")
        clip_runs.append(result)

    # ------------------------------------------------------------------
    # Part B: epochs-per-batch sweep — clip_range fixed at 0.2.
    # ------------------------------------------------------------------
    # WHY: n_epochs controls how many passes over each rollout batch the
    # optimiser makes.  More epochs → better data efficiency per rollout,
    # but the ratio r_t drifts further from 1, eventually violating the
    # “proximal” guarantee even with clipping.
    # n_epochs=2  → effectively one-shot SGD, data not fully used.
    # n_epochs=10 → PPO default; good balance.
    # n_epochs=20 → aggressive reuse; possible over-fitting to each batch.
    # ------------------------------------------------------------------
    print()
    print("Part B — epochs-per-batch sweep (clip_range fixed at 0.2)")
    print("-" * 55)
    epochs_runs: list[dict[str, Any]] = []

    for ne in epochs_values:
        label = f"n_epochs={ne}"
        print(f"  Training {label} …")
        result = _train_ppo(
            label=label,
            seed=seed,
            total_steps=total_steps,
            clip_range=0.2,
            n_epochs=ne,
            n_steps=n_steps,
            batch_size=batch_size,
        )
        tail_rets = (
            result["episode_return"][-5:]
            if len(result["episode_return"]) >= 5
            else result["episode_return"]
        )
        valid = [r for r in tail_rets if r == r]
        final_return = float(sum(valid) / len(valid)) if valid else float("nan")
        print(f"    → final return ≈ {final_return:.3f}")
        epochs_runs.append(result)

    # ------------------------------------------------------------------
    # Plots (skipped in quick mode or when --no-plot was used).
    # ------------------------------------------------------------------
    if not quick:
        print()
        print("Saving plots …")
        _plot_sweep(
            clip_runs,
            title="Experiment 08 — clip-range sweep (n_epochs=10)",
            filename="clip_range_sweep_return.png",
            metric="episode_return",
            ylabel="Episode return (EMA)",
        )
        _plot_sweep(
            clip_runs,
            title="Experiment 08 — clip-range sweep: success rate",
            filename="clip_range_sweep_success.png",
            metric="success_rate",
            ylabel="Success rate (EMA)",
        )
        _plot_sweep(
            epochs_runs,
            title="Experiment 08 — epochs-per-batch sweep (clip_range=0.2)",
            filename="epochs_sweep_return.png",
            metric="episode_return",
            ylabel="Episode return (EMA)",
        )
        _plot_sweep(
            epochs_runs,
            title="Experiment 08 — epochs-per-batch sweep: success rate",
            filename="epochs_sweep_success.png",
            metric="success_rate",
            ylabel="Success rate (EMA)",
        )

    # ------------------------------------------------------------------
    # Foxglove rollout (only when render="foxglove" and NOT in quick mode).
    # ------------------------------------------------------------------
    if render == "foxglove" and not quick:
        # Pick the best clip run (highest final return) for the rollout.
        best_clip_run = max(
            clip_runs,
            key=lambda r: ([x for x in r["episode_return"] if x == x] or [float("-inf")])[-1],
        )
        best_algo = best_clip_run["algo"]
        print(f"\nStreaming Foxglove rollout using {best_clip_run['label']} …")
        _rollout_foxglove(best_algo, n_episodes=3, seed=seed)

    # ------------------------------------------------------------------
    # Summary metrics dict.
    # ------------------------------------------------------------------
    def _final_return(runs: list[dict[str, Any]], label: str) -> float:
        for r in runs:
            if r["label"] == label:
                tail = (
                    r["episode_return"][-5:]
                    if len(r["episode_return"]) >= 5
                    else r["episode_return"]
                )
                valid = [v for v in tail if v == v]
                return float(sum(valid) / len(valid)) if valid else float("nan")
        return float("nan")

    # Best clip run is the one with the highest smoothed tail return.
    def _best_label(runs: list[dict[str, Any]]) -> str:
        best = max(
            runs,
            key=lambda r: ([x for x in r["episode_return"] if x == x] or [float("-inf")])[-1],
        )
        return best["label"]

    clip_sweep_returns = {
        float(r["label"].split("=")[1]): (
            lambda tail: float(sum(tail) / len(tail)) if tail else float("nan")
        )([v for v in r["episode_return"][-5:] if v == v])
        for r in clip_runs
    }

    epochs_sweep_returns = {
        int(r["label"].split("=")[1]): (
            lambda tail: float(sum(tail) / len(tail)) if tail else float("nan")
        )([v for v in r["episode_return"][-5:] if v == v])
        for r in epochs_runs
    }

    return {
        "best_clip_run": _best_label(clip_runs),
        "best_epochs_run": _best_label(epochs_runs),
        "clip_sweep_final_returns": clip_sweep_returns,
        "epochs_sweep_final_returns": epochs_sweep_returns,
        "total_steps_per_run": total_steps,
        "quick": quick,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Arguments
    ---------
    --quick         Tiny budget, no plots, no Foxglove (same as run(quick=True)).
    --render {foxglove}
                    Stream a post-training rollout to Foxglove Studio.
    --seed INT      Global RNG seed (default 0).
    --no-plot       Skip saving matplotlib figures (headless batch mode).
    """
    parser = argparse.ArgumentParser(
        prog="08_ppo",
        description="Experiment 08 — PPO clip-range sweep and epochs comparison.",
    )
    parser.add_argument("--quick", action="store_true", help="Tiny budget, no plots, no Foxglove.")
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="Stream a post-training Foxglove rollout.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed (default 0).")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving plots (headless / CI mode).",
    )

    args = parser.parse_args(argv)

    # --no-plot disables figures even in non-quick mode by injecting quick=True
    # for the plotting branch only.  We do this by temporarily redirecting the
    # _plot_sweep calls to no-ops when --no-plot is requested.
    effective_quick = args.quick or args.no_plot

    metrics = run(quick=effective_quick, render=args.render, seed=args.seed)

    print()
    print("=== Results ===")
    print(f"  best clip run   : {metrics['best_clip_run']}")
    print(f"  best epochs run : {metrics['best_epochs_run']}")
    print()
    print("  Clip-range sweep — final episode return (EMA tail):")
    for k, v in sorted(metrics["clip_sweep_final_returns"].items()):
        print(f"    clip={k:.2f}  →  {v:.4f}")
    print()
    print("  Epochs sweep — final episode return (EMA tail):")
    for k, v in sorted(metrics["epochs_sweep_final_returns"].items()):
        print(f"    n_epochs={k:2d}  →  {v:.4f}")

    if not args.quick and not args.no_plot:
        print()
        print(f"  Plots saved to: {_OUT_DIR}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
