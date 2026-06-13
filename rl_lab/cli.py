"""Command-line interface for the Buddy Jr RL Lab.

After installing the package (``pip install -e .``) the ``rl-lab`` command
becomes available on your PATH.

Usage examples
--------------
    rl-lab --version
    rl-lab --help
    rl-lab list --what algos
    rl-lab list --what envs
    rl-lab train --algo dqn --timesteps 5000
    rl-lab train --algo ppo --env BuddyJrReach-v0 --timesteps 50000
    rl-lab eval  --checkpoint runs/dqn_20240101_120000/model --episodes 5
    rl-lab sim   --scene hello
    rl-lab viz   --port 8765

Heavy sub-package imports (torch, SB3, algo classes) are deferred inside the
subcommand handler functions so ``rl-lab --help`` stays near-instant.
"""

from __future__ import annotations

import argparse
import sys

from rl_lab.version import __version__

# ---------------------------------------------------------------------------
# Friendly stub helpers (kept for sim / viz which are not yet wired)
# ---------------------------------------------------------------------------

_NOT_YET = (
    "This subcommand is not fully implemented yet — it is being built out "
    "incrementally.\n"
    "See docs/PLAN.md and the GitHub issues for the roadmap:\n"
    "  https://github.com/kevinmcaleer/rl_lab/issues"
)


def _stub(subcommand: str, args: argparse.Namespace) -> int:  # noqa: ARG001
    """Print a friendly 'not yet' message and return 0 (success)."""
    print(f"rl-lab {subcommand}: {_NOT_YET}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    """Handler for ``rl-lab train``.

    Validates the algo choice, then delegates to
    :func:`rl_lab.train.train.train`.  All heavy imports happen inside that
    function so this handler itself is lightweight.

    Parameters
    ----------
    args.algo:
        Algorithm key (lower-cased before use), e.g. ``ppo``, ``dqn``.
    args.env:
        Optional environment id override.
    args.timesteps:
        Total training steps (default 50 000).
    args.seed:
        Global random seed.
    args.render:
        ``'foxglove'`` or ``'human'`` or ``None``.
    args.logdir:
        Root directory for run sub-directories.
    """
    # Deferred import — keeps `rl-lab --help` fast.
    from rl_lab.train.train import train

    try:
        checkpoint_path = train(
            algo=args.algo,
            env_id=args.env or None,
            total_steps=args.timesteps,
            seed=args.seed,
            logdir=args.logdir,
            render=args.render or None,
        )
        print(f"\nDone. Checkpoint: {checkpoint_path}")
        return 0
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Training failed: {exc}", file=sys.stderr)
        return 2


def _cmd_eval(args: argparse.Namespace) -> int:
    """Handler for ``rl-lab eval``.

    Delegates to :func:`rl_lab.train.evaluate.evaluate` and prints the
    ``success_rate`` and ``mean_return`` metrics.

    Parameters
    ----------
    args.checkpoint:
        Path to the saved model (required).
    args.env:
        Optional environment id override.
    args.episodes:
        Number of evaluation episodes.
    args.render:
        ``'foxglove'`` or ``'human'`` or ``None``.
    args.record_mcap:
        Path for MCAP recording (optional).
    """
    # Deferred import — keeps `rl-lab --help` fast.
    from rl_lab.train.evaluate import evaluate

    try:
        results = evaluate(
            checkpoint=args.checkpoint,
            env_id=args.env or None,
            episodes=args.episodes,
            render=args.render or None,
            record_mcap=getattr(args, "record_mcap", None) or None,
            seed=args.seed,
        )
        print(f"\nEvaluation results ({results['episodes']} episodes):")
        print(f"  success_rate : {results['success_rate']:.3f}")
        print(f"  mean_return  : {results['mean_return']:.3f}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


def _cmd_list(args: argparse.Namespace) -> int:
    """Handler for ``rl-lab list``.

    Prints the available algorithm keys or Gymnasium env ids.

    Parameters
    ----------
    args.what:
        ``'algos'`` or ``'envs'``.
    """
    # Defer heavy imports.
    import rl_lab  # noqa: F401 — ensures envs are registered

    if args.what == "algos":
        from rl_lab.algos.registry import ALGORITHMS

        print("Available algorithms:")
        for key in sorted(ALGORITHMS):
            print(f"  {key}")
    else:  # 'envs'
        import gymnasium as gym

        # Show only the Buddy Jr envs registered by this package.
        buddy_envs = [eid for eid in gym.registry if eid.startswith("BuddyJr")]
        print("Available environments:")
        for eid in sorted(buddy_envs):
            print(f"  {eid}")
    return 0


def _cmd_sim(args: argparse.Namespace) -> int:
    """Stub for ``rl-lab sim`` (friendly 'not yet' message)."""
    return _stub("sim", args)


def _cmd_viz(args: argparse.Namespace) -> int:
    """Stub for ``rl-lab viz`` (friendly 'not yet' message)."""
    return _stub("viz", args)


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser.

    Kept in its own function so it can be imported and tested independently
    of ``main``.  Each subcommand gets its own sub-parser so
    ``rl-lab <sub> --help`` works correctly.
    """
    # --- top-level parser --------------------------------------------------
    parser = argparse.ArgumentParser(
        prog="rl-lab",
        description=(
            "Buddy Jr RL Lab — learn Reinforcement Learning on a "
            "simulated 4-DOF robot arm.\n"
            "\n"
            "Run 'rl-lab <subcommand> --help' for per-command options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # --- subcommand router -------------------------------------------------
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # -----------------------------------------------------------------------
    # train — kick off a training run
    # -----------------------------------------------------------------------
    # Determine algo choices lazily — fall back to a generic list if the
    # registry cannot be imported (e.g. during docs build without deps).
    try:
        from rl_lab.algos.registry import ALGORITHMS

        algo_choices = sorted(ALGORITHMS)
    except Exception:  # noqa: BLE001  # pragma: no cover
        algo_choices = None  # type: ignore[assignment]

    train_p = subparsers.add_parser(
        "train",
        help="Train an RL algorithm on a Buddy Jr Gymnasium environment.",
        description=(
            "Train one of the lab's RL algorithms on a Buddy Jr Gymnasium\n"
            "environment.  Checkpoints and TensorBoard logs are written to\n"
            "<logdir>/<run_name>/ automatically.\n"
            "\n"
            "Example:\n"
            "  rl-lab train --algo dqn --timesteps 5000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_p.add_argument(
        "--algo",
        required=True,
        choices=algo_choices,
        metavar="ALGO",
        help=(
            "RL algorithm key.  Available: "
            + (", ".join(algo_choices) if algo_choices else "see rl-lab list --what algos")
            + "."
        ),
    )
    train_p.add_argument(
        "--env",
        default=None,
        metavar="ENV_ID",
        help=(
            "Gymnasium environment id (default: registry-recommended env for " "the chosen algo)."
        ),
    )
    train_p.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        metavar="N",
        help="Total environment steps to train for (default: 50 000).",
    )
    train_p.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="SEED",
        help="Global random seed (default: 0).",
    )
    train_p.add_argument(
        "--render",
        choices=["foxglove", "human"],
        default=None,
        metavar="MODE",
        help="Render mode: 'foxglove' (live WebSocket) or 'human' (local GUI).",
    )
    train_p.add_argument(
        "--logdir",
        default="runs",
        metavar="DIR",
        help="Root directory for run sub-directories (default: runs/).",
    )
    train_p.set_defaults(func=_cmd_train)

    # -----------------------------------------------------------------------
    # eval — evaluate a saved policy
    # -----------------------------------------------------------------------
    eval_p = subparsers.add_parser(
        "eval",
        help="Evaluate a trained policy checkpoint.",
        description=(
            "Load a saved policy and run deterministic evaluation episodes,\n"
            "reporting mean return and success rate.\n"
            "\n"
            "Example:\n"
            "  rl-lab eval --checkpoint runs/dqn_20240101_120000/model --episodes 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_p.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help=(
            "Path to the model checkpoint (without extension — the algorithm\n"
            "resolves the extension itself, e.g. .npz or .zip)."
        ),
    )
    eval_p.add_argument(
        "--env",
        default=None,
        metavar="ENV_ID",
        help="Override the environment id stored in the checkpoint metadata.",
    )
    eval_p.add_argument(
        "--episodes",
        type=int,
        default=10,
        metavar="N",
        help="Number of evaluation episodes to run (default: 10).",
    )
    eval_p.add_argument(
        "--render",
        choices=["foxglove", "human"],
        default=None,
        metavar="MODE",
        help="Render mode: 'foxglove' (live WebSocket) or 'human' (local GUI).",
    )
    eval_p.add_argument(
        "--record-mcap",
        dest="record_mcap",
        default=None,
        metavar="PATH",
        help="Record the evaluation to an MCAP file for later replay in Foxglove.",
    )
    eval_p.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="SEED",
        help="Seed used for env resets (default: 0).",
    )
    eval_p.set_defaults(func=_cmd_eval)

    # -----------------------------------------------------------------------
    # list — enumerate available envs / algos
    # -----------------------------------------------------------------------
    list_p = subparsers.add_parser(
        "list",
        help="List available environments or algorithms.",
        description=(
            "Discover what the lab has to offer.\n"
            "\n"
            "Examples:\n"
            "  rl-lab list --what algos\n"
            "  rl-lab list --what envs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    list_p.add_argument(
        "--what",
        choices=["algos", "envs"],
        default="algos",
        metavar="CATEGORY",
        help="Category to list: algos or envs (default: algos).",
    )
    list_p.set_defaults(func=_cmd_list)

    # -----------------------------------------------------------------------
    # sim — launch a named scene (stub — not yet implemented)
    # -----------------------------------------------------------------------
    sim_p = subparsers.add_parser(
        "sim",
        help="Launch a simulation scene by name (e.g. 'hello').",
        description=(
            "Open a named simulation scene.  The 'hello' scene loads the Buddy Jr\n"
            "arm in an empty world — a good first sanity check.\n"
            "\n"
            "(Not yet wired — see GitHub issues for the roadmap.)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sim_p.add_argument(
        "--scene",
        default="hello",
        metavar="SCENE",
        help="Scene name to load: hello, reach, stack … (default: hello).",
    )
    sim_p.set_defaults(func=_cmd_sim)

    # -----------------------------------------------------------------------
    # viz — start the Foxglove WebSocket bridge (stub)
    # -----------------------------------------------------------------------
    viz_p = subparsers.add_parser(
        "viz",
        help="Start the Foxglove WebSocket server for 3-D visualisation.",
        description=(
            "Start the Foxglove WebSocket bridge so you can connect\n"
            "Foxglove Studio (https://foxglove.dev/studio) to watch the\n"
            "robot arm live in 3-D.\n"
            "\n"
            "(Not yet wired — see GitHub issues for the roadmap.)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    viz_p.add_argument(
        "--port",
        type=int,
        default=8765,
        metavar="PORT",
        help="WebSocket port to listen on (default: 8765).",
    )
    viz_p.set_defaults(func=_cmd_viz)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* and dispatch to the appropriate subcommand handler.

    This is the function wired up as the ``rl-lab`` console_scripts entry
    point in pyproject.toml.  It follows the convention of returning an
    integer exit code (0 = success) so the caller can do::

        raise SystemExit(main())

    Parameters
    ----------
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]`` when *None*.
        Pass an explicit list in tests to avoid touching ``sys.argv``.

    Returns
    -------
    int
        Exit code — 0 for success, non-zero for errors.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)

    # If no subcommand was given, print the top-level help and exit cleanly.
    if args.subcommand is None:
        parser.print_help()
        return 0

    # Every sub-parser calls set_defaults(func=_cmd_*), so args.func is set.
    return args.func(args)


if __name__ == "__main__":
    # Allow running as ``python -m rl_lab.cli`` during development.
    raise SystemExit(main())
