"""Command-line entry point for the Buddy Jr RL Lab.

This is an intentional stub. The real CLI (``rl-lab sim hello``, ``rl-lab train``,
etc.) is built out under the M1/M2 milestones — see the GitHub issues.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print("rl-lab: the CLI is not implemented yet — it is being built out.")
    print("See README.md and docs/PLAN.md for the roadmap, or run the experiments")
    print("in experiments/ directly.")
    if argv:
        print(f"(received args: {argv})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
