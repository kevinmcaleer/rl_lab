#!/usr/bin/env python3
"""deploy_to_pi.py -- Copy a trained/exported policy + config to a Raspberry Pi over SSH.

Usage
-----
    python scripts/deploy_to_pi.py \\
        --host raspberrypi.local \\
        --policy runs/buddy_reach/model.npz \\
        --config runs/buddy_reach/servo_map.json

    # Dry-run: prints commands without executing them
    python scripts/deploy_to_pi.py --host pi5.local --policy model.npz --dry-run

The script
----------
1. Creates the remote destination directory via ssh mkdir -p.
2. Copies each file via scp.
3. Verifies each file landed via ssh 'test -f <path>'.
4. Prints the exact command to run on the Pi.

No hard dependency on paramiko: plain scp/ssh subprocesses are used.  If you
want to pass custom SSH options (e.g. a different identity file) set
``DEPLOY_SSH_OPTS`` in the environment, e.g.::

    DEPLOY_SSH_OPTS="-i ~/.ssh/pi_key" python scripts/deploy_to_pi.py ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_USER: str = "pi"
_DEFAULT_DEST: str = "~/rl_lab_deploy"
# The on-device runner script path relative to the Pi home.
_PI_RUNNER: str = "deploy/raspberrypi/run_policy.py"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deploy_to_pi",
        description="Deploy a trained policy + config to a Raspberry Pi over SSH/SCP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment:\n"
            "  DEPLOY_SSH_OPTS   Extra flags forwarded to every ssh/scp call.\n"
            "                    Example: DEPLOY_SSH_OPTS='-i ~/.ssh/pi_key'\n"
        ),
    )
    p.add_argument(
        "--host",
        required=True,
        help="Hostname or IP address of the Raspberry Pi (e.g. raspberrypi.local).",
    )
    p.add_argument(
        "--user",
        default=_DEFAULT_USER,
        help=f"SSH username on the Pi (default: {_DEFAULT_USER!r}).",
    )
    p.add_argument(
        "--policy",
        required=True,
        metavar="PATH",
        help="Local path to the exported policy .npz file.",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Local path to an optional config/calibration file (e.g. servo_map.json).",
    )
    p.add_argument(
        "--dest",
        default=_DEFAULT_DEST,
        metavar="DIR",
        help=f"Destination directory on the Pi (default: {_DEFAULT_DEST!r}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extra_ssh_opts() -> list[str]:
    """Return extra SSH flags from the environment as a list of tokens."""
    raw = os.environ.get("DEPLOY_SSH_OPTS", "").strip()
    if not raw:
        return []
    # Simple split — does not handle quoted tokens with spaces inside them.
    return raw.split()


def _run(
    cmd: list[str],
    *,
    dry_run: bool,
    description: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Print and optionally execute *cmd*.

    In dry-run mode the command is printed with a ``[DRY-RUN]`` prefix and a
    dummy :class:`subprocess.CompletedProcess` with returncode 0 is returned
    so callers do not need special-case logic.

    Parameters
    ----------
    cmd:
        The command + arguments list passed to :func:`subprocess.run`.
    dry_run:
        When *True*, only print; never execute.
    description:
        Human-readable label printed before the command.
    check:
        Forwarded to :func:`subprocess.run` when *dry_run* is *False*.
    """
    display = " ".join(cmd)
    if dry_run:
        print(f"[DRY-RUN] {description}")
        print(f"          {display}")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    print(f"{description}")
    print(f"  $ {display}")
    result = subprocess.run(  # noqa: S603
        cmd,
        check=check,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    return result


def _ssh_target(user: str, host: str) -> str:
    """Return the ``user@host`` string used by ssh/scp."""
    return f"{user}@{host}"


def _mkdir_remote(
    target: str,
    remote_dir: str,
    *,
    dry_run: bool,
    extra_opts: list[str],
) -> None:
    """Create *remote_dir* on the Pi with ``mkdir -p`` over SSH."""
    cmd = ["ssh", *extra_opts, target, f"mkdir -p {remote_dir}"]
    _run(cmd, dry_run=dry_run, description=f"Creating remote directory {remote_dir!r}")


def _scp_file(
    local_path: Path,
    target: str,
    remote_dir: str,
    *,
    dry_run: bool,
    extra_opts: list[str],
) -> None:
    """Copy *local_path* to *remote_dir* on the Pi via scp."""
    remote_dest = f"{target}:{remote_dir}/"
    cmd = ["scp", *extra_opts, str(local_path), remote_dest]
    _run(
        cmd,
        dry_run=dry_run,
        description=f"Copying {local_path.name!r} -> {remote_dest}",
    )


def _verify_file(
    target: str,
    remote_path: str,
    *,
    dry_run: bool,
    extra_opts: list[str],
) -> bool:
    """Return *True* if *remote_path* exists on the Pi.

    In dry-run mode always returns *True* (we cannot check the remote).
    """
    if dry_run:
        print(f"[DRY-RUN] Verify {remote_path!r} exists on Pi (skipped in dry-run)")
        return True

    cmd = ["ssh", *extra_opts, target, f"test -f {remote_path} && echo OK || echo MISSING"]
    print(f"Verifying {remote_path!r} on Pi ...")
    result = subprocess.run(  # noqa: S603
        cmd,
        check=False,
        text=True,
        capture_output=True,
    )
    landed = result.stdout.strip() == "OK"
    status = "OK" if landed else "MISSING"
    print(f"  {status}: {remote_path}")
    return landed


def _print_run_command(remote_dir: str, policy_name: str, config_name: str | None) -> None:
    """Print the command the user should run on the Pi to execute the policy."""
    policy_remote = f"{remote_dir}/{policy_name}"
    runner = f"python3 {_PI_RUNNER}"
    parts = [runner, f"--policy {policy_remote}"]
    if config_name is not None:
        config_remote = f"{remote_dir}/{config_name}"
        parts.append(f"--config {config_remote}")
    parts.append("--no-dry-run")
    run_cmd = " ".join(parts)
    print()
    print("=" * 72)
    print("Run this command on the Pi:")
    print()
    print(f"    {run_cmd}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    """Entry point; returns 0 on success, non-zero on failure."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    dry_run: bool = args.dry_run
    extra_opts = _extra_ssh_opts()

    # ------------------------------------------------------------------
    # Validate local paths
    # ------------------------------------------------------------------
    policy_path = Path(args.policy)
    if not dry_run and not policy_path.exists():
        print(f"ERROR: policy file not found: {policy_path}", file=sys.stderr)
        return 1

    config_path: Path | None = None
    if args.config is not None:
        config_path = Path(args.config)
        if not dry_run and not config_path.exists():
            print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
            return 1

    target = _ssh_target(args.user, args.host)
    # Normalise dest: strip trailing slash so we can safely append '/<name>'.
    remote_dir = args.dest.rstrip("/")

    print(f"Deploying to {target}:{remote_dir}")
    if dry_run:
        print("(dry-run mode -- no commands will be executed)")
    print()

    # ------------------------------------------------------------------
    # 1. Create remote directory
    # ------------------------------------------------------------------
    try:
        _mkdir_remote(target, remote_dir, dry_run=dry_run, extra_opts=extra_opts)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: could not create remote directory: {exc}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # 2. Copy files
    # ------------------------------------------------------------------
    files_to_copy: list[Path] = [policy_path]
    if config_path is not None:
        files_to_copy.append(config_path)

    for local_file in files_to_copy:
        try:
            _scp_file(local_file, target, remote_dir, dry_run=dry_run, extra_opts=extra_opts)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: scp failed for {local_file}: {exc}", file=sys.stderr)
            return 3

    # ------------------------------------------------------------------
    # 3. Verify each file landed
    # ------------------------------------------------------------------
    all_ok = True
    for local_file in files_to_copy:
        remote_path = f"{remote_dir}/{local_file.name}"
        ok = _verify_file(target, remote_path, dry_run=dry_run, extra_opts=extra_opts)
        if not ok:
            all_ok = False

    if not all_ok:
        print(
            "ERROR: one or more files are missing on the Pi after transfer.",
            file=sys.stderr,
        )
        return 4

    # ------------------------------------------------------------------
    # 4. Print the run command
    # ------------------------------------------------------------------
    _print_run_command(
        remote_dir,
        policy_path.name,
        config_path.name if config_path is not None else None,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
