"""Experiment 11 — Closing the sim-to-real gap (robustification).

Concept
-------
A policy that is *perfect* in a pristine simulator routinely **fails on real
hardware**. The mismatch between sim and reality is the **reality gap**, and the
standard toolkit for crossing it is **domain randomisation**: deliberately
perturbing the environment during training so the policy learns behaviour that
survives the messiness of real servos and sensors.

Buddy Jr's "reality" is four SG90 hobby servos driven by a PCA9685 PWM board
over I2C from a Raspberry Pi. Each real-world imperfection maps onto exactly one
knob of :class:`~rl_lab.env.wrappers.DomainRandomization`:

==========================  =====================================================
DomainRandomization knob    Real-hardware analogue it models
==========================  =====================================================
``action_noise_std``        SG90 command **jitter**: the servo never lands
                            exactly on the commanded angle (dead-band, gear
                            backlash, PWM quantisation ~0.5 deg).
``obs_noise_std``           **Sensor noise**: the camera-based target detector
                            and any joint feedback are noisy estimates, not the
                            clean ground truth the sim hands you.
``action_rate_limit``       **PCA9685 update rate + control latency**: a real
                            servo cannot teleport between angles each tick; it
                            slews at a finite speed, so a command can only change
                            so much per control step.
``randomize_radius``        **Task / calibration variation**: link lengths,
                            mounting and the reachable target shell are never
                            exactly what the CAD model says.
==========================  =====================================================

Separately, every commanded joint angle is **clamped to the safe servo range**
inside the env itself (``BuddyJrReachEnv`` clips actions to ``[-1, 1]`` and the
joint targets to ``JOINT_LIMITS``), and on hardware
:func:`rl_lab.robot.buddy_jr.radians_to_servo_degrees` further clamps to the
SG90's physical ``[0, 180] deg`` — so the policy can never command a pose that
strips a gear.

What this experiment does
-------------------------
1. Trains a **naive** policy on the clean, noise-free env (the Exp 10 setup).
2. Trains a **robustified** policy on the *same* task wrapped in
   :class:`DomainRandomization`.
3. Evaluates **both** policies across a sweep of randomised parameters and
   plots the success-rate comparison, so you can *see* the naive policy fall
   off a cliff under noise/latency while the robustified one holds up.

Frozen experiment interface
----------------------------
``run(quick, render, seed) -> dict`` and ``main(argv) -> int`` as described in
the lab's experiment contract. ``quick=True`` uses a tiny budget, makes no plots
and never opens Foxglove (it is the CI smoke test and must finish in seconds).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# We robustify the continuous reach policy from Experiment 10. SAC is the
# natural choice (off-policy, sample-efficient) but the experiment is written
# against the generic Algorithm protocol, so any continuous-action algo works.
ALGO_NAME = "sac"
ENV_ID = "BuddyJrReach-v0"

# Output directory for plots (only written when not in quick mode).
OUTPUT_DIR = Path(__file__).resolve().parent / "_outputs" / "11_robustify"

# The grid of randomisation severities we sweep at evaluation time. Each level
# scales the SG90 jitter, sensor noise and slew limit together, modelling
# "increasingly hostile reality". Level 0.0 is the pristine sim.
SWEEP_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)

# Module-level toggle so the CLI ``--no-plot`` flag can suppress the figure
# without changing the frozen ``run(quick, render, seed)`` signature. The plot
# is also always skipped in quick mode.
_PLOT_ENABLED = True


# ---------------------------------------------------------------------------
# Domain-randomisation knobs at a given severity
# ---------------------------------------------------------------------------
def _randomization_kwargs(level: float) -> dict[str, Any]:
    """Return ``DomainRandomization`` kwargs for a severity ``level`` in [0, ~).

    The knobs are scaled together so a single number sweeps "how unkind reality
    is". See the module docstring for the real-hardware analogue of each knob.
    """
    if level <= 0.0:
        # Pristine sim: no perturbation at all (and no rate limit).
        return {
            "action_noise_std": 0.0,
            "obs_noise_std": 0.0,
            "action_rate_limit": None,
            "randomize_radius": None,
        }
    return {
        # SG90 command jitter, as a fraction of the [-1, 1] action range.
        "action_noise_std": level,
        # Camera/joint sensor noise on the (already [-1,1]-scaled) observation.
        "obs_noise_std": 0.5 * level,
        # Servo slew limit: at high severity an action may only change a little
        # per control step (models PCA9685 update rate + control latency).
        # Smaller => tighter limit, so we map higher severity to a tighter cap.
        "action_rate_limit": max(0.1, 1.0 - 1.5 * level),
        # Modest task/calibration variation in the reachable target shell.
        "randomize_radius": (0.04, 0.155),
    }


def make_clean_env(seed: int, *, render_mode: str | None = None) -> Any:
    """Build the pristine continuous reach env (the Experiment 10 setup)."""
    import gymnasium as gym

    import rl_lab  # noqa: F401  (registers the BuddyJr envs)

    env = gym.make(ENV_ID, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def make_randomized_env(seed: int, level: float, *, render_mode: str | None = None) -> Any:
    """Build the reach env wrapped in :class:`DomainRandomization` at ``level``."""
    from rl_lab.env.wrappers import DomainRandomization

    env = make_clean_env(seed, render_mode=render_mode)
    env = DomainRandomization(env, **_randomization_kwargs(level))
    # Re-seed so the wrapper's own np_random (used for the noise) is deterministic.
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _train_steps(quick: bool) -> int:
    """Total env steps for training (tiny in quick mode for the CI smoke test)."""
    return 300 if quick else 20_000


def _train_hparams(quick: bool) -> dict[str, Any]:
    """SAC hyper-parameters; in quick mode start learning almost immediately."""
    if quick:
        # Defaults wait 1000 steps before any gradient update; with a 300-step
        # budget that would mean "no learning at all". Lower it so the quick
        # smoke test actually exercises the train loop, while staying fast.
        return {"learning_starts": 50, "buffer_size": 1_000, "batch_size": 64}
    return {}


def _train_policy(env: Any, seed: int, quick: bool) -> Any:
    """Train one policy on ``env`` and return the fitted algorithm."""
    from rl_lab.algos.registry import make_algorithm

    algo = make_algorithm(ALGO_NAME, env, seed=seed, **_train_hparams(quick))
    algo.train(_train_steps(quick))
    return algo


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _evaluate(
    algo: Any,
    seed: int,
    level: float,
    n_episodes: int,
    *,
    streamer: Any = None,
) -> dict[str, float]:
    """Roll a trained policy out on a randomised env and measure success.

    Returns a small metrics dict with the success rate and mean final distance.
    The policy was trained once; here we only *evaluate* it under the given
    randomisation severity (no further learning).
    """
    # Use a distinct seed per level so episodes differ but stay reproducible.
    env = make_randomized_env(seed + 1_000 + int(level * 1000), level)
    successes = 0
    final_dists: list[float] = []
    try:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            terminated = truncated = False
            ep_success = False
            while not (terminated or truncated):
                action, _ = algo.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_success = ep_success or bool(info.get("is_success", False))
                if streamer is not None and streamer.enabled:
                    streamer.publish(
                        info["joint_q"],
                        info["ee_pos"],
                        info["target"],
                        float(info["distance"]),
                        reward=float(reward),
                    )
            successes += int(ep_success)
            final_dists.append(float(info["distance"]))
    finally:
        env.close()
    return {
        "success_rate": successes / max(1, n_episodes),
        "mean_final_distance": float(np.mean(final_dists)) if final_dists else float("nan"),
    }


def _sweep(algo: Any, seed: int, levels: tuple[float, ...], n_episodes: int) -> dict[float, float]:
    """Evaluate ``algo`` across every randomisation ``level`` -> success rate."""
    return {lvl: _evaluate(algo, seed, lvl, n_episodes)["success_rate"] for lvl in levels}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_comparison(
    levels: tuple[float, ...],
    naive: dict[float, float],
    robust: dict[float, float],
    out_path: Path,
) -> None:
    """Save the success-rate-vs-randomisation comparison figure (Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = list(levels)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [naive[x] for x in xs], "o-", color="#d1495b", label="naive (clean-sim only)")
    ax.plot(xs, [robust[x] for x in xs], "s-", color="#2e86ab", label="robustified (randomised)")
    ax.set_xlabel("randomisation severity\n(SG90 jitter / sensor noise / slew limit)")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Closing the sim-to-real gap: success across randomised parameters")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The experiment body (frozen interface)
# ---------------------------------------------------------------------------
def run(quick: bool = False, render: str | None = None, seed: int = 0) -> dict:
    """Train a naive and a robustified policy and compare them across noise.

    Parameters
    ----------
    quick:
        If ``True``, use a tiny training budget, a 2-level sweep, no plots and
        no Foxglove. Used by the CI smoke test; finishes in a few seconds on CPU.
    render:
        ``None`` or ``"foxglove"``. When ``"foxglove"`` (and not ``quick``) the
        robustified policy's evaluation rollouts are streamed to Foxglove.
    seed:
        Base random seed for reproducibility.

    Returns
    -------
    dict
        Metrics: per-level success rates for both policies plus headline
        robustness numbers.
    """
    # In quick mode keep everything tiny: 2 sweep levels, 3 eval episodes.
    levels: tuple[float, ...] = (0.0, 0.20) if quick else SWEEP_LEVELS
    n_eval = 3 if quick else 20

    # --- 1) Naive policy: trained only on the pristine simulator. ----------
    naive_algo = _train_policy(make_clean_env(seed), seed, quick)

    # --- 2) Robustified policy: trained under domain randomisation. --------
    # We train at a moderate severity (0.20) so the policy meets realistic
    # jitter/latency/noise during learning — the whole point of the experiment.
    robust_env = make_randomized_env(seed, level=0.20)
    robust_algo = _train_policy(robust_env, seed, quick)

    # --- 3) Optional live view of the robustified policy under noise. ------
    streamer = None
    if render == "foxglove" and not quick:
        from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

        streamer = FoxgloveStreamer("foxglove")
        # One streamed rollout at high severity so you can watch it cope.
        _evaluate(robust_algo, seed, level=0.30, n_episodes=1, streamer=streamer)
        streamer.close()
        streamer = None

    # --- 4) Sweep both policies across the randomisation grid. -------------
    naive_sweep = _sweep(naive_algo, seed, levels, n_eval)
    robust_sweep = _sweep(robust_algo, seed, levels, n_eval)

    # --- 5) Plot (skipped in quick mode or when plotting is disabled). -----
    plot_path: str | None = None
    if not quick and _PLOT_ENABLED:
        out = OUTPUT_DIR / "robustness_comparison.png"
        _plot_comparison(levels, naive_sweep, robust_sweep, out)
        plot_path = str(out)

    # Headline numbers: success at the highest swept severity, and how much
    # robustification recovers there.
    worst = max(levels)
    naive_worst = naive_sweep[worst]
    robust_worst = robust_sweep[worst]

    metrics: dict[str, Any] = {
        "algo": ALGO_NAME,
        "levels": list(levels),
        "naive_success_by_level": {f"{lvl:.2f}": naive_sweep[lvl] for lvl in levels},
        "robust_success_by_level": {f"{lvl:.2f}": robust_sweep[lvl] for lvl in levels},
        "naive_success_clean": naive_sweep[levels[0]],
        "naive_success_worst": naive_worst,
        "robust_success_worst": robust_worst,
        "robustness_gain_worst": robust_worst - naive_worst,
        "plot": plot_path,
    }
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point (frozen interface)
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Parse args, run the experiment, print the headline metrics."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny budget, no plots, no Foxglove (CI smoke test).",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="stream the robustified policy's rollouts to Foxglove.",
    )
    parser.add_argument("--seed", type=int, default=0, help="base random seed.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip writing the comparison plot even when not in quick mode.",
    )
    args = parser.parse_args(argv)

    # Honour --no-plot by disabling the figure before run() executes, so we
    # never even open matplotlib. The frozen run() signature stays untouched.
    global _PLOT_ENABLED
    _PLOT_ENABLED = not args.no_plot

    render = None if args.quick else args.render
    metrics = run(quick=args.quick, render=render, seed=args.seed)

    print("Experiment 11 — Closing the sim-to-real gap")
    print(f"  algorithm                : {metrics['algo']}")
    print(f"  naive success (clean sim): {metrics['naive_success_clean']:.2f}")
    print(f"  naive success (worst)    : {metrics['naive_success_worst']:.2f}")
    print(f"  robust success (worst)   : {metrics['robust_success_worst']:.2f}")
    print(f"  robustness gain (worst)  : {metrics['robustness_gain_worst']:+.2f}")
    if metrics.get("plot"):
        print(f"  plot                     : {metrics['plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
