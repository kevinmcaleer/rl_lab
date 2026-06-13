"""Command-line interface for the Buddy Jr RL Lab.

After installing the package (``pip install -e .``) the ``rl-lab`` command
becomes available on your PATH.  Every subcommand currently prints a friendly
"not yet" message so you can explore the CLI structure even before the full
implementation lands.

Usage examples
--------------
    rl-lab --version
    rl-lab --help
    rl-lab train --help
    rl-lab train --algo PPO --env BuddyReach-v0 --timesteps 100000
    rl-lab eval  --help
    rl-lab sim   --help
    rl-lab sim   --scene hello
    rl-lab viz   --port 8765
    rl-lab list  --what envs

The real implementation is tracked in the GitHub issues — see docs/PLAN.md
for the full roadmap.
"""

from __future__ import annotations

import argparse
import sys

from rl_lab.version import __version__

# ---------------------------------------------------------------------------
# Stub handler helpers
# ---------------------------------------------------------------------------
# Each subcommand eventually calls a real function in the appropriate
# sub-package (e.g. rl_lab.train.run_training).  For now every handler
# prints a clear message that tells the learner where to look next.

_NOT_YET = (
    "This subcommand is not implemented yet — it is being built out "
    "incrementally.\n"
    "See docs/PLAN.md and the GitHub issues for the roadmap:\n"
    "  https://github.com/kevinmcaleer/rl_lab/issues"
)


def _stub(subcommand: str, args: argparse.Namespace) -> int:  # noqa: ARG001
    """Print a friendly 'not yet' message and return 0 (success).

    Returning 0 means the shell sees a clean exit, which is handy when
    composing ``rl-lab`` calls in scripts that test command availability.
    """
    print(f"rl-lab {subcommand}: {_NOT_YET}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand handler functions
# ---------------------------------------------------------------------------
# One function per subcommand.  Each receives the parsed Namespace so it can
# inspect the flags the user passed in.  Later we replace the body with a
# real implementation (usually a single delegation call).


def _cmd_train(args: argparse.Namespace) -> int:
    """Stub for ``rl-lab train``.

    Eventually this will launch a Stable-Baselines3 training run and stream
    live metrics to Foxglove.

    Parameters
    ----------
    args.algo:
        Algorithm name, e.g. ``PPO``, ``SAC``, ``TD3``.
    args.env:
        Gymnasium environment ID, e.g. ``BuddyReach-v0``.
    args.timesteps:
        Total environment steps to train for.
    args.checkpoint:
        Optional path to a ``.zip`` checkpoint to resume from.
    """
    return _stub("train", args)


def _cmd_eval(args: argparse.Namespace) -> int:
    """Stub for ``rl-lab eval``.

    Eventually this will load a trained policy, run evaluation episodes, and
    report mean return and success rate.

    Parameters
    ----------
    args.checkpoint:
        Path to the ``.zip`` model checkpoint to evaluate.
    args.episodes:
        Number of evaluation episodes to run.
    args.render:
        Whether to open a Foxglove viewer during evaluation.
    """
    return _stub("eval", args)


def _cmd_sim(args: argparse.Namespace) -> int:
    """Stub for ``rl-lab sim``.

    Eventually this will launch a PyBullet scene by name (e.g. ``hello``
    loads the Buddy Jr arm in the empty world so you can see it move).

    Parameters
    ----------
    args.scene:
        Scene name to load, e.g. ``hello``, ``reach``, ``stack``.
    """
    return _stub("sim", args)


def _cmd_viz(args: argparse.Namespace) -> int:
    """Stub for ``rl-lab viz``.

    Eventually this will start the Foxglove WebSocket server so you can
    connect the Foxglove Studio desktop app for 3-D live visualisation.

    Parameters
    ----------
    args.port:
        WebSocket port to listen on (default 8765 matches Foxglove's default).
    """
    return _stub("viz", args)


def _cmd_list(args: argparse.Namespace) -> int:
    """Stub for ``rl-lab list``.

    Eventually this will enumerate available environments, experiments, and
    algorithms so learners can discover what the lab has to offer.

    Parameters
    ----------
    args.what:
        Category to list: ``envs``, ``experiments``, or ``algos``.
    """
    return _stub("list", args)


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser.

    We keep parser construction in its own function so it can be imported
    and tested without running ``main``.  Each subcommand gets its own
    sub-parser so ``rl-lab <sub> --help`` works out of the box.
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
        # Keep the newlines we wrote above — argparse strips them by default
        # unless we use RawDescriptionHelpFormatter.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --version prints the package version string and exits cleanly.
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # --- subcommand router -------------------------------------------------
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="subcommand",  # args.subcommand will be the chosen name or None
        metavar="<subcommand>",
    )

    # -----------------------------------------------------------------------
    # train — kick off a training run
    # -----------------------------------------------------------------------
    train_p = subparsers.add_parser(
        "train",
        help="Train an RL policy on a Buddy Jr Gymnasium environment.",
        description=(
            "Train a Stable-Baselines3 policy on one of the rl-lab "
            "Gymnasium environments.  Checkpoints are saved to "
            "checkpoints/<run-name>/ automatically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_p.add_argument(
        "--algo",
        default="PPO",
        metavar="ALGO",
        help="RL algorithm to use, e.g. PPO, SAC, TD3 (default: PPO).",
    )
    train_p.add_argument(
        "--env",
        default="BuddyReach-v0",
        metavar="ENV_ID",
        help="Gymnasium environment ID to train on (default: BuddyReach-v0).",
    )
    train_p.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        metavar="N",
        help="Total environment steps to train for (default: 100 000).",
    )
    train_p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to a .zip checkpoint to resume training from (optional).",
    )
    # Wire up the handler so argparse calls _cmd_train(args) automatically.
    train_p.set_defaults(func=_cmd_train)

    # -----------------------------------------------------------------------
    # eval — evaluate a saved policy
    # -----------------------------------------------------------------------
    eval_p = subparsers.add_parser(
        "eval",
        help="Evaluate a trained policy checkpoint.",
        description=(
            "Load a saved policy and run evaluation episodes, reporting "
            "mean return and success rate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_p.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to the .zip model checkpoint to evaluate.",
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
        action="store_true",
        help="Open a Foxglove viewer to watch the evaluation in 3-D.",
    )
    eval_p.set_defaults(func=_cmd_eval)

    # -----------------------------------------------------------------------
    # sim — launch a named PyBullet scene
    # -----------------------------------------------------------------------
    sim_p = subparsers.add_parser(
        "sim",
        help="Launch a PyBullet scene by name (e.g. 'hello').",
        description=(
            "Open a named simulation scene.  The 'hello' scene loads "
            "the Buddy Jr arm in an empty world — a good first sanity check."
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
    # viz — start the Foxglove WebSocket bridge
    # -----------------------------------------------------------------------
    viz_p = subparsers.add_parser(
        "viz",
        help="Start the Foxglove WebSocket server for 3-D visualisation.",
        description=(
            "Start the Foxglove WebSocket bridge so you can connect "
            "Foxglove Studio (https://foxglove.dev/studio) to watch the "
            "robot arm live in 3-D."
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

    # -----------------------------------------------------------------------
    # list — enumerate available envs / experiments / algos
    # -----------------------------------------------------------------------
    list_p = subparsers.add_parser(
        "list",
        help="List available environments, experiments, or algorithms.",
        description=(
            "Discover what the lab has to offer.  Use --what to choose "
            "the category: envs, experiments, or algos."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    list_p.add_argument(
        "--what",
        choices=["envs", "experiments", "algos"],
        default="envs",
        help="Category to list: envs, experiments, or algos (default: envs).",
    )
    list_p.set_defaults(func=_cmd_list)

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
        Argument list to parse.  Defaults to ``sys.argv[1:]`` when *None*,
        which is the normal case when the command is invoked from the shell.
        Pass an explicit list in tests to avoid touching ``sys.argv``.

    Returns
    -------
    int
        Exit code — 0 for success, non-zero for errors.
    """
    # Use sys.argv[1:] when no explicit argv is provided.  This matches the
    # standard argparse convention.
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)

    # If no subcommand was given, print the top-level help and exit cleanly.
    # (argparse does not do this automatically when subparsers are optional.)
    if args.subcommand is None:
        parser.print_help()
        return 0

    # Every sub-parser calls set_defaults(func=_cmd_*) so args.func is always
    # set once a subcommand has been matched.  We call it and return its code.
    return args.func(args)


if __name__ == "__main__":
    # Allow running as ``python -m rl_lab.cli`` during development.
    raise SystemExit(main())
