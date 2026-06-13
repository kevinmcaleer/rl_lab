"""Experiment 12 — Deploy to real hardware (inference-only, dry-run on the laptop).

Concept
-------
You have trained a policy.  Now what?  The final step of any robotics RL project
is **deployment**: exporting the learned weights from the training framework and
running *pure inference* on the physical robot — no torch, no gym, no SB3.

This experiment walks through the complete M6 sim-to-real pipeline:

1. **Export**: convert any trained (or, in quick mode, a tiny random) policy to a
   lightweight ``.npz`` archive via :func:`~rl_lab.deploy.policy_export.export_algorithm`
   or :func:`~rl_lab.deploy.policy_export.export_mlp_to_npz`. The archive is
   self-describing: it carries the MLP weights *and* metadata (obs_dim, act_dim,
   action_type) so it can be loaded on any machine that has NumPy — and only NumPy.

2. **Inference**: load the archive with :class:`~rl_lab.deploy.policy_export.NumpyMLPPolicy`.
   ``policy.predict(obs)`` runs a pure-NumPy ReLU-MLP forward pass — identical to
   the Raspberry Pi runtime.

3. **Safety layer**: every action goes through:
   a. :func:`~rl_lab.robot.safety.clamp_joint_limits` — clips to the URDF joint limits.
   b. :class:`~rl_lab.robot.servo_map.ServoMap.to_servo_degrees` — converts radians
      to PCA9685 servo degrees (sign-corrected, trimmed).
   c. :class:`~rl_lab.robot.safety.RateLimiter` — caps the per-step servo change
      so a glitchy action can never slam the SG90 gears.
   d. :class:`~rl_lab.robot.safety.EmergencyStop` — ESC / Space halts the loop.

4. **Dry-run**: commands are *printed* instead of sent to hardware. Paste the same
   ``.npz`` onto a Pi and run :file:`deploy/raspberrypi/run_policy.py` with
   ``--no-dry-run`` to actually move the servos.

5. **Optional Foxglove mirror**: when ``render='foxglove'`` the commanded joint
   state is streamed to Foxglove so you can watch the arm execute the dry-run in
   3-D — the same view you would see on the real robot.

Frozen experiment interface
---------------------------
``run(quick, render, seed) -> dict`` and ``main(argv) -> int`` as described in the
lab's experiment contract. ``quick=True`` exports a tiny random policy and dry-runs
~5 steps; it must finish in a few seconds on CPU.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Output directory for saved plots (created lazily; skipped in quick mode).
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "12_deploy"
OUTPUT_DIR = Path(__file__).resolve().parent / "_outputs" / EXPERIMENT_NAME

# Module-level toggle so --no-plot can suppress figures without touching the
# frozen run() signature.
_PLOT_ENABLED = True

# ---------------------------------------------------------------------------
# Safety constants — mirror the defaults used in deploy/raspberrypi/run_policy.py
# ---------------------------------------------------------------------------
# Maximum servo degrees any joint may move in a single control step.
_RATE_LIMIT_DEG: float = 15.0
# Control-loop rate for the dry-run (Hz). Not time-critical on the laptop.
_LOOP_HZ: float = 20.0
# Action scale: each action unit moves a joint by this many radians (matches env).
_ACTION_SCALE: float = 0.1
# Obs vector dimension produced by the reach env (Box(17)).
_OBS_DIM: int = 17
# Action dimension (Box(4), continuous per-joint delta).
_ACT_DIM: int = 4
# Position scale: workspace ≈ 0.22 m -> ~[-1, 1] (matches spaces.POS_SCALE).
_POS_SCALE: float = 4.5


# ---------------------------------------------------------------------------
# Step 1 — Train a tiny policy (full mode) or build a random one (quick mode)
# ---------------------------------------------------------------------------


def _train_policy(seed: int, *, quick: bool) -> Any:
    """Train a minimal SAC policy on BuddyJrReach-v0 and return the algo object.

    In quick mode we skip training and return a random-weights stand-in that
    still satisfies the Algorithm protocol (it has .predict and .save).
    We do *not* import torch / SB3 until we actually need them, keeping import
    time low and the quick path entirely dependency-free.
    """
    if quick:
        # Quick mode: return a lightweight stand-in — NumpyMLPPolicy already
        # implements the same .predict signature, so it works as a "algo".
        # We don't call export_algorithm on it; instead _export_policy handles
        # the tiny random MLP branch directly.
        return None  # signal: use the random-MLP path in _export_policy

    import gymnasium as gym

    import rl_lab  # noqa: F401 — registers BuddyJrReach-v0
    from rl_lab.algos.registry import make_algorithm

    env = gym.make("BuddyJrReach-v0")
    env.reset(seed=seed)
    # A very short SAC run — enough to produce a non-trivial weight matrix that
    # shows the export/inference pipeline works end-to-end. The key lesson here
    # is the export/deploy path, not the training score.
    algo = make_algorithm(
        "sac",
        env,
        seed=seed,
        learning_starts=200,
        buffer_size=5_000,
        batch_size=64,
    )
    algo.train(total_steps=2_000)
    env.close()
    return algo


# ---------------------------------------------------------------------------
# Step 2 — Export the policy to .npz
# ---------------------------------------------------------------------------


def _build_tiny_random_mlp(seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a minimal 2-layer MLP with random weights (obs_dim=17, act_dim=4).

    Used in quick mode so we can exercise the export/inference/safety pipeline
    without any training: a random policy still produces valid numpy arrays and
    servo commands that the safety layer must clamp.
    """
    rng = np.random.default_rng(seed)
    # Hidden layer: 17 -> 16
    W1 = rng.standard_normal((_OBS_DIM, _OBS_DIM)).astype(np.float32) * 0.1
    b1 = np.zeros(_OBS_DIM, dtype=np.float32)
    # Output layer: 17 -> 4 (joint deltas in [-1, 1])
    W2 = rng.standard_normal((_ACT_DIM, _OBS_DIM)).astype(np.float32) * 0.1
    b2 = np.zeros(_ACT_DIM, dtype=np.float32)
    return [(W1, b1), (W2, b2)]


def _export_policy(algo: Any | None, path: str, seed: int) -> str:
    """Export *algo* (or a random MLP when algo is None) to a .npz file.

    Returns the resolved path (with .npz suffix) that was written.
    """
    if algo is None:
        # Quick-mode path: hand-craft a tiny random MLP and use export_mlp_to_npz
        # directly. This exercises the same serialisation code that full export uses.
        from rl_lab.deploy.policy_export import export_mlp_to_npz

        layers = _build_tiny_random_mlp(seed)
        meta = {
            "action_type": "continuous",
            "obs_dim": _OBS_DIM,
            "act_dim": _ACT_DIM,
            "algo": "random-mlp (quick mode)",
        }
        export_mlp_to_npz(layers, meta, path)
    else:
        # Full-mode path: extract weights from the trained SB3 model.
        from rl_lab.deploy.policy_export import export_algorithm

        export_algorithm(algo, path)

    # np.savez appends .npz automatically when missing; normalise the return path.
    npz_path = path if path.endswith(".npz") else path + ".npz"
    return npz_path


# ---------------------------------------------------------------------------
# Step 3 — Build an observation vector (no gym needed at inference time)
# ---------------------------------------------------------------------------


def _build_obs(q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Assemble the 17-D observation the policy was trained on.

    On the real Pi we have no joint velocities (SG90s give no feedback), so
    the velocity slots are left out of the 17-D observation.  The formula
    matches :func:`rl_lab.env.spaces.build_observation` exactly:

        [sin(q), cos(q), ee_pos * POS_SCALE, target * POS_SCALE,
         (target - ee) * POS_SCALE]
    """
    from rl_lab.robot.kinematics import forward_position

    ee = forward_position(q)
    # sin/cos encoding avoids the discontinuity at +/-pi in raw joint angles.
    obs = np.concatenate(
        [
            np.sin(q),  # 4 values
            np.cos(q),  # 4 values
            ee * _POS_SCALE,  # 3 values (camera-tip position)
            target * _POS_SCALE,  # 3 values (target position)
            (target - ee) * _POS_SCALE,  # 3 values (tip-to-target vector)
        ]
    ).astype(
        np.float32
    )  # total: 17
    # Clip to [-1, 1] so observations stay in the range the policy was trained on.
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4 — Dry-run inference loop
# ---------------------------------------------------------------------------


def _dry_run(
    policy: Any,  # NumpyMLPPolicy
    *,
    target: np.ndarray,
    max_steps: int,
    seed: int,
    streamer: Any | None = None,
) -> dict[str, Any]:
    """Run the inference loop in dry-run mode: print servo commands, drive nothing.

    Mirrors the logic in ``deploy/raspberrypi/run_policy.py`` so the laptop
    dry-run and the Pi live-run are identical except for the final ``kit.servo``
    call.

    Returns a metrics dict with per-step servo commands, distances, and a flag
    indicating whether the target was reached before ``max_steps``.
    """
    from rl_lab.robot import buddy_jr as bj
    from rl_lab.robot import kinematics as kin
    from rl_lab.robot import safety
    from rl_lab.robot.servo_map import ServoMap

    # Set up the servo map with default 1:1 calibration (sign=+1, offset=0 deg).
    # On real hardware you would load a JSON calibration produced by
    # deploy/raspberrypi/servo_calibration.py.
    servo_map = ServoMap()

    # Seed the joint state at the neutral pose (all zeros).
    np.random.default_rng(seed)
    q = np.zeros(bj.NUM_JOINTS, dtype=np.float64)

    # Initialise the rate limiter from the current (neutral) servo pose.
    current_deg = servo_map.to_servo_degrees(q)
    rate_limiter = safety.RateLimiter(max_delta_deg=_RATE_LIMIT_DEG)
    rate_limiter.reset(current_deg)

    # The emergency stop is started here so it behaves identically to the Pi
    # runner. On a laptop without a real TTY (CI) it is a safe no-op.
    estop = safety.EmergencyStop()
    estop.start_keyboard_listener()

    # Accumulators for the returned metrics.
    all_servo_cmds: list[list[float]] = []
    all_distances: list[float] = []
    reached = False

    print("  Dry-run inference loop (no hardware):")
    print(f"  target = ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) m")
    print(f"  max_steps={max_steps}, rate_limit={_RATE_LIMIT_DEG} deg/step")
    print()

    for step in range(max_steps):
        # --- E-stop check (would halt the real robot immediately) ----------
        if estop.engaged:
            print("  EMERGENCY STOP engaged — halting dry-run.")
            break

        # --- Observe -------------------------------------------------------
        obs = _build_obs(q, target)

        # --- Policy inference (pure NumPy — identical to the Pi) -----------
        action, _ = policy.predict(obs, deterministic=True)
        # action is float32 ndarray of shape (4,) in [-1, 1] (tanh output)
        delta = np.asarray(action, dtype=np.float64).ravel()
        # Guard: clip delta to [-1, 1] in case of numerical edge cases.
        delta = np.clip(delta, -1.0, 1.0)

        # --- Integrate the delta and clamp to joint limits -----------------
        # ACTION_SCALE=0.1 rad/unit matches the training environment.
        q = safety.clamp_joint_limits(q + delta * _ACTION_SCALE)

        # --- Sim2real: radians -> PCA9685 servo degrees -------------------
        target_deg = servo_map.to_servo_degrees(q)

        # --- Rate limit: cap the per-step change to protect SG90 gears ----
        target_deg = rate_limiter.apply(target_deg, current_deg)
        # After rate-limiting, back-project to radians so the next step's FK
        # reflects what the servo actually did (slewed, not teleported).
        q = servo_map.to_radians(target_deg)
        current_deg = target_deg.copy()

        # Safety invariant: every command must be a valid SG90 angle.
        assert np.all(target_deg >= 0.0) and np.all(
            target_deg <= 180.0
        ), f"Safety violation: servo command out of range {target_deg}"

        # --- Forward kinematics: compute the tip position from q -----------
        ee = kin.forward_position(q)
        dist = float(np.linalg.norm(ee - target))

        # --- Print the dry-run servo command (what would go to PCA9685) ----
        cmd_str = "  ".join(
            f"ch{ch}={deg:6.2f}" for ch, deg in zip(servo_map.channels, target_deg, strict=False)
        )
        print(f"  step {step:4d}  dist={dist:.4f} m  [{cmd_str}]")

        all_servo_cmds.append(target_deg.tolist())
        all_distances.append(dist)

        # --- Optional Foxglove mirror -------------------------------------
        if streamer is not None:
            with contextlib.suppress(Exception):  # never crash the dry-run
                streamer.publish(q, ee, target, dist, force=True)

        # --- Success check ------------------------------------------------
        # Same tolerance as the training env (2 cm default).
        if dist <= 0.02:
            print(f"  TARGET REACHED at step {step} (dist={dist:.4f} m).")
            reached = True
            break

    return {
        "steps_run": len(all_distances),
        "reached": reached,
        "final_distance": all_distances[-1] if all_distances else float("nan"),
        "min_distance": float(np.min(all_distances)) if all_distances else float("nan"),
        "servo_commands": all_servo_cmds,
        "distances": all_distances,
    }


# ---------------------------------------------------------------------------
# Step 5 — Plot servo trajectories and distance curve
# ---------------------------------------------------------------------------


def _plot_results(
    result: dict[str, Any],
    *,
    out_dir: Path,
) -> str:
    """Save two figures: per-joint servo commands over time, and distance curve.

    Uses the Agg backend so the function is safe in headless CI and on the Pi.
    """
    import matplotlib

    matplotlib.use("Agg")  # must be set before importing pyplot
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    servo_cmds = np.array(result["servo_commands"])  # shape (T, 4)
    distances = np.array(result["distances"])  # shape (T,)
    steps = np.arange(len(distances))
    joint_names = ["base_yaw", "shoulder_pitch", "elbow_pitch", "camera_tilt"]
    colors = ["#2e86ab", "#a23b72", "#f18f01", "#c73e1d"]

    # --- Figure 1: servo commands per joint --------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, color) in enumerate(zip(joint_names, colors, strict=True)):
        ax.plot(steps, servo_cmds[:, i], color=color, label=name, linewidth=1.5)
    ax.axhline(90, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="neutral (90°)")
    ax.set_xlabel("Control step")
    ax.set_ylabel("Servo command (degrees)")
    ax.set_ylim(0, 180)
    ax.set_title("Experiment 12 — Dry-run servo commands per joint")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    servo_path = out_dir / "servo_commands.png"
    fig.savefig(servo_path, dpi=120)
    plt.close(fig)

    # --- Figure 2: distance to target over time ----------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(steps, distances, color="#2e86ab", linewidth=1.8, label="tip-to-target distance")
    ax2.axhline(
        0.02, color="green", linestyle="--", linewidth=1.0, label="success threshold (2 cm)"
    )
    ax2.set_xlabel("Control step")
    ax2.set_ylabel("Distance to target (m)")
    ax2.set_title("Experiment 12 — Camera-tip distance to target")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    dist_path = out_dir / "distance_curve.png"
    fig2.savefig(dist_path, dpi=120)
    plt.close(fig2)

    return str(servo_path)


# ---------------------------------------------------------------------------
# Frozen experiment interface
# ---------------------------------------------------------------------------


def run(quick: bool = False, render: str | None = None, seed: int = 0) -> dict:
    """Export a policy to .npz and dry-run the M6 inference path.

    Parameters
    ----------
    quick:
        When ``True``: export a tiny random MLP and dry-run only 5 steps.
        No plots, no Foxglove. Finishes in < 5 s on CPU (CI smoke-test mode).
    render:
        ``None`` or ``"foxglove"``.  When ``"foxglove"`` (and not quick)
        the dry-run joint state is mirrored live to Foxglove Studio so you
        can watch the arm execute the dry-run commands in 3-D.
    seed:
        Random seed (used for the tiny MLP in quick mode and the env in full mode).

    Returns
    -------
    dict
        Metrics: export_path, steps_run, final_distance, min_distance, reached,
        plus the paths of any saved plots.
    """
    # Pick a target inside the reachable workspace (same shell as the training env).
    # Target = (x=0.10 m, y=0.02 m, z=0.12 m) — a point the shoulder can reach.
    target = np.array([0.10, 0.02, 0.12], dtype=np.float64)
    max_steps = 5 if quick else 200

    # --- 1. Train (or skip in quick mode) ---------------------------------
    print(
        "[12_deploy] Step 1/4 — "
        + ("building random MLP (quick mode)" if quick else "training SAC policy")
    )
    algo = _train_policy(seed, quick=quick)

    # --- 2. Export to .npz -----------------------------------------------
    print("[12_deploy] Step 2/4 — exporting policy to .npz")
    # Use a temp file in quick mode to avoid leaving artefacts; a named output
    # directory file in full mode so the archive persists for inspection.
    if quick:
        fd, tmp_name = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        export_base = tmp_name[:-4]  # strip .npz; np.savez will add it back
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        export_base = str(OUTPUT_DIR / "policy")

    npz_path = _export_policy(algo, export_base, seed)
    print(f"  Exported -> {npz_path}")

    # --- 3. Load via NumpyMLPPolicy (pure NumPy — identical to the Pi) ----
    print("[12_deploy] Step 3/4 — loading policy via NumpyMLPPolicy")
    from rl_lab.deploy.policy_export import NumpyMLPPolicy

    policy = NumpyMLPPolicy.load(npz_path)
    print(
        f"  action_type={policy.meta['action_type']}  "
        f"obs_dim={policy.meta['obs_dim']}  "
        f"act_dim={policy.meta['act_dim']}"
    )
    print("  layers: " + ", ".join(f"({W.shape[1]}→{W.shape[0]})" for (W, _b) in policy.layers))

    # --- 4. Optional Foxglove streamer ------------------------------------
    streamer = None
    if render == "foxglove" and not quick:
        from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

        streamer = FoxgloveStreamer("foxglove")
        print("  Foxglove streaming enabled (connect to ws://localhost:8765).")
        if streamer.app_url is not None:
            print(f"  Desktop: {streamer.app_url}")

    # --- 5. Dry-run inference loop ----------------------------------------
    print(f"[12_deploy] Step 4/4 — dry-run ({max_steps} steps, no hardware)")
    try:
        result = _dry_run(
            policy,
            target=target,
            max_steps=max_steps,
            seed=seed,
            streamer=streamer,
        )
    finally:
        if streamer is not None:
            streamer.close()

    # --- 6. Plot (skipped in quick mode or when _PLOT_ENABLED is False) ---
    plot_path: str | None = None
    if not quick and _PLOT_ENABLED:
        plot_path = _plot_results(result, out_dir=OUTPUT_DIR)
        print(f"  Plots saved under {OUTPUT_DIR}")

    metrics: dict[str, Any] = {
        "export_path": npz_path,
        "action_type": policy.meta["action_type"],
        "obs_dim": policy.meta["obs_dim"],
        "act_dim": policy.meta["act_dim"],
        "n_layers": len(policy.layers),
        "steps_run": result["steps_run"],
        "reached": result["reached"],
        "final_distance": result["final_distance"],
        "min_distance": result["min_distance"],
        "plot": plot_path,
    }
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point (frozen interface)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, run the experiment, print headline metrics."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny budget: export a random MLP and dry-run 5 steps. No plots. (CI mode)",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="Mirror the dry-run joint state to Foxglove Studio (ws://localhost:8765).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the training env / random MLP (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the servo-command and distance plots.",
    )
    args = parser.parse_args(argv)

    # Honour --no-plot by disabling the figure *before* run() executes.
    # The frozen run() signature stays untouched.
    global _PLOT_ENABLED
    _PLOT_ENABLED = not args.no_plot

    render = None if args.quick else args.render
    metrics = run(quick=args.quick, render=render, seed=args.seed)

    print()
    print("Experiment 12 — Deploy to real hardware (dry-run)")
    print(f"  export path      : {metrics['export_path']}")
    print(
        f"  policy type      : {metrics['action_type']} "
        f"(obs={metrics['obs_dim']}, act={metrics['act_dim']}, layers={metrics['n_layers']})"
    )
    print(f"  steps run        : {metrics['steps_run']}")
    print(f"  reached target   : {metrics['reached']}")
    print(f"  final distance   : {metrics['final_distance']:.4f} m")
    print(f"  min distance     : {metrics['min_distance']:.4f} m")
    if metrics.get("plot"):
        print(f"  plots            : {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
