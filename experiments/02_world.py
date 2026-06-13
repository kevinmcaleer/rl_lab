"""Experiment 02 — Build the world: joint sweep + FK sanity check.

Concept: The **environment** is a first-class object.  Before an agent can
learn anything you must first be able to (a) define what actions do, (b) see
what they cause, and (c) verify that the simulation's physics agrees with the
maths.

What this script does
---------------------
1. Creates a ``BuddyJrReach-v0`` Gymnasium environment with a movable target
   sphere at a random, always-reachable location.
2. Performs a **joint sweep**: cycles each of the four joints from its lower
   joint limit to its upper limit while holding the others at zero, giving a
   clean visual tour of what each actuator controls.
3. Optionally streams every step to **Foxglove** (``--render foxglove``) so
   you can watch the arm move in 3-D, see the target sphere, and follow the
   distance metric live.
4. At every step runs an **FK sanity check**: calls
   ``rl_lab.robot.kinematics.forward(q)`` independently of the environment and
   asserts that the resulting tip position agrees with ``info['ee_pos']`` from
   ``env.step()`` to within 0.5 mm.  If the URDF, the kinematic chain, and the
   simulation backend are all consistent this assertion *always* passes — and
   that is exactly the point: you now have a trusted, testable environment.
5. Saves a multi-panel matplotlib figure showing joint angle trajectories and
   the FK residual (tip-position error) over time.

Teaching goals
--------------
* Understand the Gymnasium ``reset() / step()`` contract.
* See that every ``info`` dict carries ``{joint_q, ee_pos, target, distance,
  is_success}`` — the full state of the world in one call.
* Gain confidence in the FK implementation before any agent touches it.
* Learn to drive the environment *without* an agent (pure scripted actions) so
  you can test assumptions at any time during the curriculum.

After running this, every subsequent experiment can say "the environment and
the kinematics agree" and you will know exactly what that means.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must come before any pyplot import; safe in quick mode too

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import rl_lab  # noqa: F401 – registers Gymnasium environments on import
from rl_lab.robot import buddy_jr as bj
from rl_lab.robot import kinematics as kin

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

#: Tolerance for the FK vs. env ee_pos sanity check (metres).
FK_TOLERANCE: float = 5e-4  # 0.5 mm

#: Number of steps per joint during the sweep (full scale: quick mode uses fewer).
_STEPS_FULL: int = 60
_STEPS_QUICK: int = 8

#: The FK residual that would cause us to raise an assertion error.
_ASSERT_HARD_TOL: float = 1e-3  # 1 mm — hard error limit


# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------


def _joint_sweep(
    env: gym.Env,
    joint_idx: int,
    n_steps: int,
    streamer: Any | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Sweep joint ``joint_idx`` from –limit to +limit over ``n_steps`` steps.

    Returns three parallel lists collected at each env step:
    - ``q_hist``      : joint angle vectors (shape (4,)) from ``info['joint_q']``.
    - ``ee_hist``     : end-effector positions from ``info['ee_pos']``.
    - ``fk_hist``     : tip positions computed by the analytic FK.
    """
    limit = bj.JOINT_LIMIT  # +/- π/2 rad

    q_hist: list[np.ndarray] = []
    ee_hist: list[np.ndarray] = []
    fk_hist: list[np.ndarray] = []

    # Sweep angle schedule: from -limit to +limit, then back.
    angles = np.concatenate(
        [
            np.linspace(-limit, limit, n_steps // 2 + 1),
            np.linspace(limit, -limit, n_steps - n_steps // 2),
        ]
    )

    # Start from the current env state (after reset, q ≈ 0).
    obs, info = env.reset(seed=joint_idx)
    q_current = info["joint_q"].copy()

    for target_angle in angles:
        # Build a target q: all joints at zero except the swept one.
        q_target = np.zeros(bj.NUM_JOINTS, dtype=np.float64)
        q_target[joint_idx] = target_angle

        # Convert to an action delta: action = (q_target - q_current) / scale
        # The env applies: q_new = q_current + action * ACTION_SCALE (0.1 rad).
        # We saturate to [-1, 1] so the arm moves toward the target in steps.
        delta = q_target - q_current
        action = np.clip(delta / 0.1, -1.0, 1.0)

        obs, _reward, terminated, truncated, info = env.step(action)

        q_now: np.ndarray = info["joint_q"]
        ee_now: np.ndarray = info["ee_pos"]
        fk_now: np.ndarray = kin.forward(q_now).position

        q_hist.append(q_now.copy())
        ee_hist.append(ee_now.copy())
        fk_hist.append(fk_now.copy())

        # FK sanity check — hard assertion.
        residual = float(np.linalg.norm(fk_now - ee_now))
        if residual > _ASSERT_HARD_TOL:
            raise AssertionError(
                f"FK vs env ee_pos mismatch at joint {joint_idx} sweep step: "
                f"residual={residual:.6f} m (limit={_ASSERT_HARD_TOL} m)\n"
                f"  FK tip  = {fk_now}\n"
                f"  env tip = {ee_now}"
            )

        # Stream to Foxglove if enabled.
        if streamer is not None:
            target: np.ndarray = info["target"]
            dist: float = info["distance"]
            streamer.publish(q_now, ee_now, target, dist)

        q_current = q_now.copy()

        if terminated or truncated:
            obs, info = env.reset()
            q_current = info["joint_q"].copy()

    return q_hist, ee_hist, fk_hist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_plots(
    joint_histories: list[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]],
    output_dir: Path,
) -> None:
    """Save a multi-panel figure: joint angle trajectories + FK residuals.

    Panel layout:
    - Row 0: joint angle over sweep steps for all four joints (one subplot each).
    - Row 1: FK residual (||fk_pos - ee_pos||, mm) for all four joints.
    """
    fig, axes = plt.subplots(2, bj.NUM_JOINTS, figsize=(14, 6))
    fig.suptitle("Experiment 02 — Joint sweep & FK sanity check", fontsize=13)

    colours = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]

    for j, (q_hist, ee_hist, fk_hist) in enumerate(joint_histories):
        steps = np.arange(len(q_hist))
        # Extract the angle for this joint across the sweep.
        angles_rad = np.array([q[j] for q in q_hist])
        angles_deg = np.degrees(angles_rad)
        residuals_mm = np.array(
            [np.linalg.norm(fk - ee) * 1000 for fk, ee in zip(fk_hist, ee_hist, strict=False)]
        )

        # ---- top row: joint angle trajectory ----
        ax_top = axes[0, j]
        ax_top.plot(steps, angles_deg, color=colours[j], linewidth=1.5)
        ax_top.axhline(y=0, color="grey", linewidth=0.5, linestyle="--")
        ax_top.set_title(bj.JOINT_NAMES[j].replace("_", " ").title(), fontsize=10)
        ax_top.set_ylabel("Angle (deg)" if j == 0 else "")
        ax_top.set_ylim(-100, 100)
        ax_top.grid(True, alpha=0.3)

        # ---- bottom row: FK residual ----
        ax_bot = axes[1, j]
        ax_bot.plot(steps, residuals_mm, color=colours[j], linewidth=1.2)
        ax_bot.axhline(
            y=FK_TOLERANCE * 1000,
            color="red",
            linewidth=0.8,
            linestyle="--",
            label=f"tol={FK_TOLERANCE*1000:.1f} mm",
        )
        ax_bot.set_xlabel("Sweep step")
        ax_bot.set_ylabel("FK residual (mm)" if j == 0 else "")
        ax_bot.set_ylim(-0.02, max(residuals_mm.max() * 1.3, FK_TOLERANCE * 2000))
        ax_bot.legend(fontsize=7)
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "02_world_sweep.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[02_world] Plot saved → {fig_path}")


# ---------------------------------------------------------------------------
# Public experiment interface
# ---------------------------------------------------------------------------


def run(quick: bool = False, render: str | None = None, seed: int = 0) -> dict:
    """Run the joint-sweep and FK sanity check experiment.

    Parameters
    ----------
    quick:
        When ``True`` use a tiny step budget and skip plotting / Foxglove.
        Used by the CI smoke test; finishes in a few seconds on CPU.
    render:
        ``"foxglove"`` — stream joint states and target marker to a running
        Foxglove WebSocket server (``ws://127.0.0.1:8765``).
        ``None`` — no live visualisation.
    seed:
        RNG seed forwarded to the environment's ``reset()``.

    Returns
    -------
    dict with keys:
        ``max_fk_residual_mm``  : worst FK vs env-ee_pos error across all steps
                                  (should be < 0.5 mm on a healthy install).
        ``n_steps_total``       : total number of env steps executed.
        ``fk_checks_passed``    : ``True`` if every step passed the FK assertion.
    """
    n_steps = _STEPS_QUICK if quick else _STEPS_FULL

    # ------------------------------------------------------------------ env --
    render_mode = render if render in ("foxglove",) else None
    env = gym.make("BuddyJrReach-v0", render_mode=render_mode)
    env.reset(seed=seed)

    # ------------------------------------------------------------------ viz --
    streamer = None
    if render == "foxglove" and not quick:
        from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

        streamer = FoxgloveStreamer("foxglove")
        url = streamer.app_url
        if url:
            print(f"[02_world] Open Foxglove → {url}")
        else:
            print("[02_world] Foxglove server running on ws://127.0.0.1:8765")

    # --------------------------------------------------------------- sweep --
    all_histories: list[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = []
    n_steps_total = 0
    max_fk_residual_m = 0.0
    fk_checks_passed = True

    print(f"[02_world] Sweeping {bj.NUM_JOINTS} joints × {n_steps} steps each …")

    try:
        for j, joint in enumerate(bj.JOINTS):
            print(
                f"  joint {j}: {joint.name}  ({math.degrees(joint.lower):.0f}° → "
                f"{math.degrees(joint.upper):.0f}°)"
            )
            try:
                q_hist, ee_hist, fk_hist = _joint_sweep(env, j, n_steps, streamer)
            except AssertionError as exc:
                # Propagate the assertion but still tidy up.
                fk_checks_passed = False
                print(f"  [FAIL] {exc}")
                all_histories.append(([], [], []))
                continue

            all_histories.append((q_hist, ee_hist, fk_hist))
            n_steps_total += len(q_hist)

            # Gather per-joint worst residual.
            for fk_p, ee_p in zip(fk_hist, ee_hist, strict=False):
                r = float(np.linalg.norm(fk_p - ee_p))
                max_fk_residual_m = max(max_fk_residual_m, r)

    finally:
        env.close()
        if streamer is not None:
            streamer.close()

    max_fk_residual_mm = max_fk_residual_m * 1000.0

    print(
        f"[02_world] Done.  steps={n_steps_total}  "
        f"max FK residual={max_fk_residual_mm:.4f} mm  "
        f"(tolerance={FK_TOLERANCE*1000:.1f} mm)  "
        f"FK checks {'PASSED' if fk_checks_passed else 'FAILED'}"
    )

    # ------------------------------------------------------------ plot -------
    if not quick and any(len(q) > 0 for q, _, _ in all_histories):
        output_dir = Path(__file__).parent / "_outputs" / "02_world"
        _save_plots(all_histories, output_dir)

    return {
        "max_fk_residual_mm": max_fk_residual_mm,
        "n_steps_total": n_steps_total,
        "fk_checks_passed": fk_checks_passed,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and call ``run()``; returns 0 on success."""
    parser = argparse.ArgumentParser(
        prog="02_world",
        description=(
            "Experiment 02 — Build the world: sweep each joint through its full "
            "range, optionally stream to Foxglove, and verify FK against the env."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny step budget, no plots, no Foxglove (CI smoke test mode).",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="Stream live visualisation to a running Foxglove desktop app.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the environment (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the matplotlib figure even in full mode.",
    )
    args = parser.parse_args(argv)

    metrics = run(quick=args.quick, render=args.render, seed=args.seed)

    # Print a tidy summary.
    print("\n=== Experiment 02 results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if not metrics["fk_checks_passed"]:
        print("\n[02_world] ERROR: FK sanity check failed — see output above.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
