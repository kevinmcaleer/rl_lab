#!/usr/bin/env python3
"""run_policy.py -- on-device inference loop for Buddy Jr on a Raspberry Pi.

Run a trained policy on the real robot (or, by default, in a safe *dry-run* that
only prints the servo commands it *would* send). The policy is a torch-free
:class:`~rl_lab.deploy.policy_export.NumpyMLPPolicy` exported from any algorithm
with ``rl-lab export`` / :func:`rl_lab.deploy.policy_export.export_algorithm`, so
nothing heavyweight has to be installed on the Pi: just NumPy.

The loop mirrors :class:`~rl_lab.env.buddy_jr_reach_env.BuddyJrReachEnv` exactly,
so a policy behaves the same on hardware as it did in sim:

1. Build the observation the policy expects (17-D, or 21-D with velocity) from
   the current joint state and the target, using
   :func:`rl_lab.env.spaces.build_observation`.
2. ``predict`` an action (pure NumPy; no torch).
3. Interpret the action as a joint-target *delta* (discrete jog -> continuous via
   :func:`rl_lab.env.spaces.discrete_to_continuous`; continuous straight through),
   scaled by ``action_scale`` (0.1, matching the env).
4. Apply safety: clamp to the joint limits, convert to servo degrees through the
   calibrated :class:`~rl_lab.robot.servo_map.ServoMap`, rate-limit the change per
   step, and honour an :class:`~rl_lab.robot.safety.EmergencyStop`.
5. **Dry-run** (the default): print the per-step servo-degree command and assert
   it is within ``[0, 180]``. Nothing is imported from ``adafruit``.
   **--no-dry-run**: lazily import ``adafruit_servokit`` and drive the servos.

Forward kinematics ([`rl_lab.robot.kinematics.forward`]) is used to estimate the
camera-tip pose from the commanded joint angles (SG90s give no feedback), which
also lets us mirror the live state to Foxglove with ``--foxglove``.

Usage
-----
    # safe dry-run (default): prints servo commands, drives nothing
    python deploy/raspberrypi/run_policy.py --policy policy.npz --target 0.1 0.0 0.12

    # actually drive the servos (on the Pi, with the PCA9685 wired up)
    python deploy/raspberrypi/run_policy.py --policy policy.npz \
        --target 0.1 0.0 0.12 --no-dry-run --calibration servo_cal.json

    # visualise the on-device run live in Foxglove
    python deploy/raspberrypi/run_policy.py --policy policy.npz \
        --target 0.1 0.0 0.12 --foxglove
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from rl_lab.env import spaces as S
from rl_lab.robot import buddy_jr as bj
from rl_lab.robot import kinematics as kin
from rl_lab.robot import safety
from rl_lab.robot import servo_map as servo_map_mod

if TYPE_CHECKING:  # imported lazily at runtime; here only for type checkers
    from rl_lab.deploy.policy_export import NumpyMLPPolicy
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

# Public alias kept for readability at call sites.
ServoMap = servo_map_mod.ServoMap

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Max change to each joint target per step (rad) for a unit action -- this is
#: ``BuddyJrReachEnv.ACTION_SCALE`` so the policy behaves the same on hardware.
ACTION_SCALE: float = 0.1

#: Default control rate (Hz). SG90s are slow; 20 Hz is a sane, gentle default.
DEFAULT_HZ: float = 20.0

#: Default per-step rate limit (degrees). Caps how far any servo can jump in a
#: single control step so a glitchy command can't slam the arm. ``None`` => off.
DEFAULT_RATE_LIMIT_DEG: float = 15.0

#: Distance (m) below which we declare the target "reached" and stop.
SUCCESS_TOL: float = 0.02


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_policy",
        description=(
            "On-device inference loop for Buddy Jr: run an exported NumpyMLPPolicy "
            "to reach a target. Dry-run by default (prints servo commands only)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policy",
        required=True,
        metavar="PATH",
        help="Path to an exported NumpyMLPPolicy .npz file.",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Target camera-tip position in metres (world frame), e.g. 0.1 0.0 0.12.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=DEFAULT_HZ,
        help=f"Control loop rate in Hz (default: {DEFAULT_HZ:g}).",
    )
    # Dry-run defaults to True; pass --no-dry-run to actually drive the servos.
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print servo commands instead of driving hardware (default: on). "
        "Use --no-dry-run to drive the real servos via the PCA9685.",
    )
    parser.add_argument(
        "--foxglove",
        action="store_true",
        help="Mirror the live joint state + target marker to Foxglove.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Stop after this many control steps even if the target isn't reached "
        "(default: 400).",
    )
    parser.add_argument(
        "--rate-limit-deg",
        type=float,
        default=DEFAULT_RATE_LIMIT_DEG,
        metavar="DEG",
        help="Max change per servo per step in degrees; <= 0 disables it "
        f"(default: {DEFAULT_RATE_LIMIT_DEG:g}).",
    )
    parser.add_argument(
        "--calibration",
        metavar="PATH",
        default=None,
        help="Path to a ServoMap JSON calibration {signs,offsets_deg,channels}. "
        "If omitted, a default 1:1 mapping is used.",
    )
    parser.add_argument(
        "--success-tol",
        type=float,
        default=SUCCESS_TOL,
        metavar="M",
        help=f"Stop once the tip is within this distance of the target "
        f"(default: {SUCCESS_TOL:g} m).",
    )
    parser.add_argument(
        "--seed-q",
        type=float,
        nargs=bj.NUM_JOINTS,
        default=None,
        metavar="RAD",
        help="Initial joint angles (rad), e.g. 0 0 0 0. Defaults to all zeros.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_servo_map(path: str | None) -> ServoMap:
    """Build a :class:`ServoMap` from a calibration file, or a 1:1 default."""
    if path is None:
        return ServoMap()
    return ServoMap.from_file(path)


def _build_obs(policy: NumpyMLPPolicy, q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build the observation the policy was trained on (17-D or 21-D).

    The on-device path has no joint-velocity sensing (SG90s give no feedback), so
    velocities are reported as zero when a 21-D observation is required.
    """
    obs_dim = int(policy.meta.get("obs_dim", S.OBS_DIM))
    include_velocity = obs_dim == S.OBS_DIM_WITH_VELOCITY
    ee = kin.forward_position(q)
    qd = np.zeros(bj.NUM_JOINTS, dtype=np.float64)
    return S.build_observation(q, qd, ee, target, include_velocity=include_velocity)


def _action_to_delta(policy: NumpyMLPPolicy, action: Any) -> np.ndarray:
    """Interpret a policy action as a unit joint-target delta in ``[-1, 1]^4``.

    Discrete actions are mapped through :func:`spaces.discrete_to_continuous` so a
    discrete and a continuous agent issue the *same* kind of update -- exactly as
    :class:`BuddyJrReachEnv` does.
    """
    action_type = str(policy.meta.get("action_type", "continuous"))
    if action_type == "discrete":
        return S.discrete_to_continuous(int(np.asarray(action).reshape(-1)[0]))
    delta = np.asarray(action, dtype=np.float64).reshape(-1)
    if delta.shape[0] != bj.NUM_JOINTS:
        raise ValueError(f"continuous action has {delta.shape[0]} dims, expected {bj.NUM_JOINTS}")
    return np.clip(delta, -1.0, 1.0)


def _make_streamer(enabled: bool, success_tol: float) -> FoxgloveStreamer | None:
    """Start a Foxglove streamer when requested; otherwise return ``None``."""
    if not enabled:
        return None
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

    streamer = FoxgloveStreamer("foxglove", success_radius=success_tol)
    print("  Foxglove streaming enabled.")
    if streamer.app_url is not None:
        print(f"  Desktop : {streamer.app_url}")
    return streamer


def _make_servo_kit() -> Any:
    """Lazily import and construct the PCA9685-backed ServoKit (hardware only)."""
    from adafruit_servokit import ServoKit  # noqa: PLC0415  (lazy on purpose)

    return ServoKit(channels=16)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(
    policy: NumpyMLPPolicy,
    target: np.ndarray,
    *,
    servo_map: ServoMap,
    hz: float = DEFAULT_HZ,
    dry_run: bool = True,
    max_steps: int = 400,
    rate_limit_deg: float | None = DEFAULT_RATE_LIMIT_DEG,
    success_tol: float = SUCCESS_TOL,
    streamer: FoxgloveStreamer | None = None,
    q0: np.ndarray | None = None,
    sleep: bool = True,
) -> int:
    """Run the inference loop. Returns 0 on success, 1 on E-stop/failure.

    Pure logic: hardware and Foxglove are injected, so this is unit-testable and
    importable without any device libraries.
    """
    target = np.asarray(target, dtype=np.float64).reshape(3)
    period = 1.0 / hz if hz > 0 else 0.0
    channels = servo_map.channels

    # Joint state we *command*; FK gives the resulting tip (no feedback on SG90s).
    q: np.ndarray = (
        bj.clamp_to_limits(np.asarray(q0, dtype=np.float64))
        if q0 is not None
        else np.zeros(bj.NUM_JOINTS, dtype=np.float64)
    )

    # Safety: rate limiter (seeded at the current servo pose) + emergency stop.
    current_deg: np.ndarray = servo_map.to_servo_degrees(q)
    limiter: safety.RateLimiter | None = None
    if rate_limit_deg is not None and rate_limit_deg > 0:
        limiter = safety.RateLimiter(max_delta_deg=float(rate_limit_deg))
        limiter.reset(current_deg)

    estop = safety.EmergencyStop()
    estop.start_keyboard_listener()  # lazy; safe no-op when there is no TTY

    kit = _make_servo_kit() if not dry_run else None

    mode = "DRY-RUN (no hardware)" if dry_run else "LIVE (driving servos)"
    print(f"Buddy Jr policy runner -- {mode}")
    print(f"  target   : ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) m")
    print(f"  channels : {channels}")
    print(f"  rate     : {hz:g} Hz, max-steps {max_steps}")
    print(
        f"  obs_dim  : {int(policy.meta.get('obs_dim', S.OBS_DIM))}, "
        f"action    : {policy.meta.get('action_type', 'continuous')}"
    )
    print()

    reached = False
    estopped = False
    try:
        for step in range(int(max_steps)):
            if estop.engaged:
                print("  EMERGENCY STOP engaged -- halting.")
                estopped = True
                break

            t0 = time.monotonic()

            # 1. observe -> 2. predict -> 3. interpret as a joint delta.
            obs = _build_obs(policy, q, target)
            action, _ = policy.predict(obs, deterministic=True)
            delta = _action_to_delta(policy, action)

            # 4. integrate the delta and clamp to the joint limits.
            q = safety.clamp_joint_limits(q + delta * ACTION_SCALE)

            # 5. to servo degrees, then rate-limit the per-step change.
            target_deg: np.ndarray = servo_map.to_servo_degrees(q)
            if limiter is not None:
                target_deg = limiter.apply(target_deg, current_deg)
                # The actually-commanded pose drives the next step's FK so the
                # estimated tip stays consistent with what the servos do.
                q = servo_map.to_radians(target_deg)
            current_deg = target_deg

            # Safety invariant: every command must be a valid SG90 angle.
            assert np.all(target_deg >= 0.0) and np.all(
                target_deg <= 180.0
            ), f"servo command out of range: {target_deg}"

            ee = kin.forward_position(q)
            dist = float(np.linalg.norm(ee - target))

            # 6. drive (or print).
            _emit(kit, channels, target_deg, dry_run, step, dist)

            if streamer is not None:
                streamer.publish(q, ee, target, dist, force=True)

            if dist <= success_tol:
                print(f"  TARGET REACHED at step {step} (dist={dist:.4f} m).")
                reached = True
                break

            if sleep and period > 0:
                elapsed = time.monotonic() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\n  Ctrl-C -- stopping.")
        estopped = True
    finally:
        if streamer is not None:
            streamer.close()

    if estopped:
        return 1
    if not reached:
        print(f"  Stopped after {max_steps} steps without reaching the target.")
    return 0


def _emit(
    kit: Any,
    channels: tuple[int, ...],
    target_deg: np.ndarray,
    dry_run: bool,
    step: int,
    dist: float,
) -> None:
    """Print (dry-run) or send (live) the per-channel servo angles."""
    pairs = "  ".join(f"ch{ch}={deg:6.2f}" for ch, deg in zip(channels, target_deg, strict=False))
    print(f"  step {step:4d}  dist={dist:.4f} m  [{pairs}]")
    if dry_run:
        return
    for ch, deg in zip(channels, target_deg, strict=False):
        kit.servo[int(ch)].angle = float(deg)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns an exit code (0 = clean, 1 = error/E-stop)."""
    args = _build_parser().parse_args(argv)

    # Lazy: keep module import cheap and torch-free for tools that just import.
    from rl_lab.deploy.policy_export import NumpyMLPPolicy

    try:
        policy = NumpyMLPPolicy.load(args.policy)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not load policy {args.policy!r}: {exc}", file=sys.stderr)
        return 1

    try:
        servo_map = _load_servo_map(args.calibration)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not load calibration {args.calibration!r}: {exc}", file=sys.stderr)
        return 1

    rate_limit = args.rate_limit_deg if args.rate_limit_deg and args.rate_limit_deg > 0 else None
    q0 = np.asarray(args.seed_q, dtype=np.float64) if args.seed_q is not None else None

    streamer: FoxgloveStreamer | None
    try:
        streamer = _make_streamer(args.foxglove, args.success_tol)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not start Foxglove server: {exc}", file=sys.stderr)
        return 1

    return run(
        policy,
        np.asarray(args.target, dtype=np.float64),
        servo_map=servo_map,
        hz=args.hz,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        rate_limit_deg=rate_limit,
        success_tol=args.success_tol,
        streamer=streamer,
        q0=q0,
    )


if __name__ == "__main__":
    sys.exit(main())
