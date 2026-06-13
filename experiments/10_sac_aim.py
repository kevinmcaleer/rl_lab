"""Experiment 10 — SAC: sample-efficient continuous control + the full aim task.

This experiment is the climax of Part V of the lab. We solve the *real* Buddy Jr
objective — **aim the camera's view ray at a target in 3D using all four joints**
— and use it to teach the single most important practical distinction in deep
reinforcement learning for robots: **on-policy vs. off-policy** learning, and the
**sample efficiency** that off-policy methods buy you.

Concepts taught
---------------
* **Off-policy continuous control (SAC).** Soft Actor-Critic keeps a *replay
  buffer* of past transitions and re-uses every one of them many times. PPO, by
  contrast, throws each rollout away after a single batch of updates. When a
  "sample" is an expensive robot move, re-use is gold.
* **Entropy-regularised exploration.** SAC does not just maximise reward; it
  maximises ``reward + alpha * entropy(policy)``. The entropy bonus keeps the
  Gaussian policy from collapsing to a single action too early, so it explores
  the joint space far more thoroughly than a plain deterministic actor. SB3
  auto-tunes the temperature ``alpha`` toward a target entropy of ``-dim(action)``.
* **Sample efficiency vs. PPO.** We train *both* SAC and PPO on the *same* aim
  task for the *same* number of environment steps, then plot mean episode return
  against environment steps. SAC's curve typically climbs much faster per step.

The task — "aim", not just "reach"
-----------------------------------
We use the ``BuddyJrCameraPoint-v0`` environment. This is the reach task plus an
**alignment reward**: the dot product between the camera link's forward axis and
the unit vector pointing at the target (``+1`` means the camera is staring
straight at it). The wrist / camera-tilt joint, idle in the pure-reach task,
becomes load-bearing here because orientation now matters, not just tip position.
``info["alignment"]`` reports the cosine similarity each step so we can score how
well the trained policy actually *aims*.

Foxglove view (described, not required to run)
----------------------------------------------
With ``--render foxglove`` we roll out the trained SAC policy in an env whose
``render_mode="foxglove"``. The environment streams the URDF arm, the target
sphere and the live metrics to Foxglove Studio. Conceptually you can picture (and
add to the layout) the camera's **view frustum** — a thin cone projected along
the camera link's forward axis. As the policy improves, that frustum swings
around and **locks onto** the target; drag the target sphere live and the frustum
tracks it, exactly like a pan-tilt security camera following a moving subject.
The same stream is what will later drive the *real* arm in Experiment 12.

Frozen experiment interface
---------------------------
* ``run(quick=False, render=None, seed=0) -> dict`` — the experiment body.
* ``main(argv=None) -> int`` — argparse front-end.
* ``python experiments/10_sac_aim.py`` runs ``main()``.

``quick=True`` uses a tiny training budget, writes **no** plots, and never touches
Foxglove — it is the CI smoke test and must finish in a few seconds on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Output location for saved plots (created lazily, only when not in quick mode).
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "10_sac_aim"
OUTPUT_DIR = Path(__file__).resolve().parent / "_outputs" / EXPERIMENT_NAME

# The task every algorithm in this experiment is trained on. The camera-point
# variant adds the alignment reward (align_weight defaults to 1.0 for this id).
ENV_ID = "BuddyJrCameraPoint-v0"


# ---------------------------------------------------------------------------
# Training budgets.
# ---------------------------------------------------------------------------
# Two regimes:
#   * quick  -> a handful of steps so the CI smoke test finishes in seconds.
#   * full   -> enough steps to *see* SAC pull ahead of PPO on the learning
#               curve while still running in a couple of minutes on a laptop CPU.
# These are deliberately modest: the teaching point is the *shape* of the two
# curves relative to each other, not a publication-grade final score.
_QUICK_STEPS = 600
_FULL_STEPS = 20_000

# Episodes used to score the trained policies at the end.
_QUICK_EVAL_EPISODES = 2
_FULL_EVAL_EPISODES = 20

# Steps streamed to Foxglove when --render foxglove is requested.
_FOXGLOVE_STEPS = 600


def _make_env(seed: int, *, render_mode: str | None = None) -> Any:
    """Build the aim task env. Imports gymnasium + rl_lab lazily.

    ``import rl_lab`` is what registers ``BuddyJrCameraPoint-v0`` with Gymnasium,
    so it must happen before ``gym.make``.
    """
    import gymnasium as gym

    import rl_lab  # noqa: F401  (side-effect import: registers the env ids)

    env = gym.make(ENV_ID, render_mode=render_mode)
    # Seed the action space so env.action_space.sample() (if ever used) and the
    # initial reset are reproducible across runs.
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


class _ReturnRecorder:
    """A plain ``Callable[[dict], None]`` that records (step, episode_return).

    The lab's ``SB3Algorithm.train(callback=...)`` accepts exactly this shape and
    internally wraps it in ``SimpleCallbackBridge``, which calls us every
    ``log_every`` env steps with a metrics dict that includes ``"step"`` and
    (once at least one episode has finished) ``"episode_return"``. We simply
    accumulate those samples so we can plot reward-vs-environment-steps afterward.
    """

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.returns: list[float] = []

    def __call__(self, metrics: dict[str, Any]) -> None:
        # Only record points where SB3 actually had a finished-episode return to
        # report; early in training the episode buffer may still be empty.
        if "episode_return" not in metrics:
            return
        self.steps.append(int(metrics.get("step", 0)))
        self.returns.append(float(metrics["episode_return"]))


def _train_one(
    name: str,
    *,
    total_steps: int,
    seed: int,
) -> tuple[Any, _ReturnRecorder]:
    """Train a single algorithm on the aim task and capture its learning curve.

    Returns the trained algorithm wrapper and the recorder holding the
    (step, mean episode return) samples gathered during training.

    Note on cadence: ``SB3Algorithm.train`` wraps our recorder in
    ``SimpleCallbackBridge``, which fires every 1 000 env steps. The full run
    (~20 000 steps) therefore yields ~20 curve points — plenty to show the SAC
    vs. PPO trend. A quick run is far shorter than 1 000 steps, so it simply
    produces zero points; that is fine because quick mode never plots.
    """
    from rl_lab.algos.registry import make_algorithm

    env = _make_env(seed)
    algo = make_algorithm(name, env, seed=seed)
    recorder = _ReturnRecorder()

    # Pass a plain callable; SB3Algorithm wraps it in SimpleCallbackBridge.
    algo.train(total_steps=total_steps, callback=recorder)

    env.close()
    return algo, recorder


def _evaluate_aim(algo: Any, *, episodes: int, seed: int) -> dict[str, float]:
    """Roll out a trained policy and score how well it *aims*.

    Returns mean episode return, mean terminal alignment (cosine of the angle
    between the camera ray and the to-target direction; +1 is perfect), and the
    success rate (fraction of episodes whose tip reached the target tolerance).
    """
    import numpy as np

    env = _make_env(seed)
    returns: list[float] = []
    final_alignment: list[float] = []
    successes: list[float] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + 1000 + ep)
        ep_return = 0.0
        terminated = truncated = False
        align = float(info.get("alignment", 0.0))
        while not (terminated or truncated):
            action, _ = algo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            align = float(info.get("alignment", align))
        returns.append(ep_return)
        final_alignment.append(align)
        successes.append(1.0 if info.get("is_success", False) else 0.0)

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "mean_alignment": float(np.mean(final_alignment)),
        "success_rate": float(np.mean(successes)),
    }


def _plot_curves(
    sac_rec: _ReturnRecorder,
    ppo_rec: _ReturnRecorder,
    *,
    out_path: Path,
) -> None:
    """Save the reward-vs-environment-steps comparison plot (Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")  # headless: never opens a window, safe in CI
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    if sac_rec.steps:
        ax.plot(sac_rec.steps, sac_rec.returns, marker="o", label="SAC (off-policy)")
    if ppo_rec.steps:
        ax.plot(ppo_rec.steps, ppo_rec.returns, marker="s", label="PPO (on-policy)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episode return")
    ax.set_title("Aim task: sample efficiency of SAC vs. PPO")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _stream_foxglove(algo: Any, *, seed: int, steps: int) -> None:
    """Roll out the trained SAC policy with Foxglove streaming enabled.

    The env is built with ``render_mode="foxglove"`` so the environment itself
    publishes the URDF arm, the target sphere and the live metrics each step (see
    ``BuddyJrReachEnv._maybe_render``). Open Foxglove Studio and load
    ``rl_lab/viz/layouts/buddy_jr.json``; the arm will swing the camera onto the
    target. Add a frustum/cone marker along the camera's forward axis to *see*
    the view ray lock on; drag the target sphere and watch it track.
    """
    env = _make_env(seed, render_mode="foxglove")
    try:
        obs, _info = env.reset(seed=seed)
        for _ in range(steps):
            action, _ = algo.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                obs, _info = env.reset()
    finally:
        env.close()


def run(quick: bool = False, render: str | None = None, seed: int = 0) -> dict:
    """Train SAC and PPO on the aim task, compare sample efficiency, return metrics.

    Parameters
    ----------
    quick:
        When True use a tiny budget, skip all plotting and skip Foxglove. This is
        the CI smoke-test path and must finish in a few seconds on CPU.
    render:
        ``"foxglove"`` streams a trained-policy rollout to Foxglove Studio;
        ``None`` disables streaming. Ignored when ``quick`` is True.
    seed:
        Global seed for reproducibility (forwarded to both algorithms and the
        env resets).

    Returns
    -------
    dict
        A small metrics dict with the final scores for both algorithms and the
        budgets used, e.g.::

            {
                "env_id": "BuddyJrCameraPoint-v0",
                "total_steps": 20000,
                "sac": {"mean_return": ..., "mean_alignment": ..., "success_rate": ...},
                "ppo": {"mean_return": ..., "mean_alignment": ..., "success_rate": ...},
                "sac_curve_points": 20,
                "ppo_curve_points": 20,
            }
    """
    # The frozen public entry point always plots in a full run; --no-plot is a
    # CLI-only convenience handled by main() via the shared _run() helper.
    return _run(quick=quick, render=render, plot=not quick, seed=seed)


def _run(*, quick: bool, render: str | None, plot: bool, seed: int) -> dict:
    """Shared body for :func:`run` and the ``--no-plot`` CLI path.

    Trains SAC and PPO on the same aim task for the same budget, scores both, and
    (optionally) writes the comparison plot and/or streams a rollout to Foxglove.
    Quick mode forces plotting and streaming off so the CI smoke test is fast.
    """
    total_steps = _QUICK_STEPS if quick else _FULL_STEPS
    eval_episodes = _QUICK_EVAL_EPISODES if quick else _FULL_EVAL_EPISODES

    # --- Train both algorithms on the SAME aim task for the SAME budget -------
    # SAC first (the star of the show), then PPO as the on-policy baseline.
    sac_algo, sac_rec = _train_one("sac", total_steps=total_steps, seed=seed)
    ppo_algo, ppo_rec = _train_one("ppo", total_steps=total_steps, seed=seed)

    # --- Score the trained policies on how well they aim ----------------------
    sac_metrics = _evaluate_aim(sac_algo, episodes=eval_episodes, seed=seed)
    ppo_metrics = _evaluate_aim(ppo_algo, episodes=eval_episodes, seed=seed)

    metrics: dict[str, Any] = {
        "env_id": ENV_ID,
        "total_steps": total_steps,
        "sac": sac_metrics,
        "ppo": ppo_metrics,
        "sac_curve_points": len(sac_rec.steps),
        "ppo_curve_points": len(ppo_rec.steps),
    }

    # --- Plots + Foxglove only outside quick mode -----------------------------
    if quick:
        return metrics

    if plot:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = OUTPUT_DIR / "sac_vs_ppo_sample_efficiency.png"
        _plot_curves(sac_rec, ppo_rec, out_path=plot_path)
        metrics["plot_path"] = str(plot_path)

    if render == "foxglove":
        # Stream the better-aiming SAC policy so the view ray visibly locks on.
        _stream_foxglove(sac_algo, seed=seed, steps=_FOXGLOVE_STEPS)

    return metrics


def main(argv: list[str] | None = None) -> int:
    """Command-line front-end for the experiment.

    Flags
    -----
    --quick           Tiny budget, no plots, no Foxglove (CI smoke test).
    --render foxglove Stream the trained SAC policy to Foxglove Studio.
    --seed N          Random seed (default 0).
    --no-plot         Skip writing the comparison plot even in a full run.
    """
    parser = argparse.ArgumentParser(
        description="Experiment 10 — SAC vs. PPO on the Buddy Jr camera-aim task."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny training budget, no plots, no Foxglove (used by the CI smoke test).",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="Stream a trained-policy rollout to Foxglove Studio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the comparison plot (still trains and evaluates).",
    )
    args = parser.parse_args(argv)

    # --no-plot trains + evaluates (+ optionally streams) but skips the plot.
    # Go through the shared _run() helper so there is a single code path.
    metrics = _run(
        quick=args.quick,
        render=args.render,
        plot=not (args.no_plot or args.quick),
        seed=args.seed,
    )

    # Print a compact human-readable summary.
    sac = metrics["sac"]
    ppo = metrics["ppo"]
    print(f"Task: {metrics['env_id']}  |  budget: {metrics['total_steps']} env steps")
    print(
        "  SAC  ->  return={mean_return:7.3f}  aim(cos)={mean_alignment:+.3f}  "
        "success={success_rate:.0%}".format(**sac)
    )
    print(
        "  PPO  ->  return={mean_return:7.3f}  aim(cos)={mean_alignment:+.3f}  "
        "success={success_rate:.0%}".format(**ppo)
    )
    if "plot_path" in metrics:
        print(f"  Learning-curve plot saved to: {metrics['plot_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
