"""Training run logger: TensorBoard + CSV, no hard deps at import time.

Every from-scratch algorithm and the SB3 adapter call
:meth:`RunLogger.log_scalars` once per episode or once per update.
The logger fans the data out to two sinks:

1. **TensorBoard** (``torch.utils.tensorboard.SummaryWriter``) — lazy-imported
   so the module loads even without ``torch``. If ``torch`` / ``tensorboard``
   are unavailable, this sink is silently skipped and only the CSV is written.
2. **CSV** (``metrics.csv`` inside the run directory) — always written;
   robust to missing keys (new keys are unioned into the header on first
   appearance).

Usage::

    logger = RunLogger("runs", "dqn_seed0")
    logger.log_scalars(step=100, scalars={"episode_return": 3.2, "loss": 0.04})
    logger.close()

Where to plug in Weights & Biases (out of scope for M4 but left as a hook):
  Search for the ``# W&B HOOK`` comment below.
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Any


class RunLogger:
    """Persist per-step / per-episode scalar metrics to TensorBoard + CSV.

    Parameters
    ----------
    logdir:
        Root directory for all runs (e.g. ``"runs"`` or ``"logs"``).
    run_name:
        Name of *this* run (e.g. ``"dqn_seed42_20260613_130500"``).
        A subdirectory ``logdir/run_name/`` is created on init.

    The directory tree looks like::

        logdir/
          run_name/
            events.out.tfevents.*   <- TensorBoard (if torch available)
            metrics.csv             <- always written
    """

    def __init__(self, logdir: str, run_name: str) -> None:
        self.logdir = logdir
        self.run_name = run_name

        # Construct and create the run directory.
        self.run_dir: Path = Path(logdir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------ #
        # TensorBoard sink (lazy — skipped if torch/tensorboard unavailable). #
        # ------------------------------------------------------------------ #
        self._tb_writer: Any = None  # SummaryWriter | None
        try:
            # torch.utils.tensorboard does NOT require a GPU; it only needs the
            # tensorboard package.  We import lazily so `import rl_lab.train`
            # stays cheap even when torch is absent (e.g. tabular-only setups).
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]

            self._tb_writer = SummaryWriter(log_dir=str(self.run_dir))
        except ImportError:
            # torch or tensorboard not installed -> CSV-only mode.
            self._tb_writer = None

        # ------------------------------------------------------------------ #
        # CSV sink — open in append mode so a resumed run extends the file.   #
        # ------------------------------------------------------------------ #
        self._csv_path: Path = self.run_dir / "metrics.csv"
        # Track which columns we have written so we can extend the header when
        # new keys appear (union-over-time header strategy).
        self._csv_columns: list[str] = []  # ordered, starts empty
        self._csv_file: io.TextIOWrapper | None = None
        self._csv_writer: csv.DictWriter | None = None  # type: ignore[type-arg]

        # We do NOT open the CSV here; we open it lazily on first log_scalars
        # call so we know which columns to write.

        # Record wall-clock start so we can log elapsed_s if desired.
        self._start_time: float = time.time()

        # W&B HOOK — initialise wandb here if you want W&B support:
        #   import wandb
        #   wandb.init(project="rl_lab", name=run_name, dir=logdir, config={})
        #   self._wandb_run = wandb.run
        # Then in log_scalars() call wandb.log(scalars, step=step).
        # In close() call wandb.finish().

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        """Write one row of scalar metrics to TensorBoard and the CSV.

        Parameters
        ----------
        step:
            Global training step (x-axis in TensorBoard and CSV).
        scalars:
            Mapping of metric name -> value.  New keys appearing after the
            first call are automatically added to the CSV header (the columns
            appear at the end; earlier rows get an empty string for new keys).

        Notes
        -----
        * ``step`` and ``elapsed_s`` are automatically prepended to the CSV row
          so every row is self-describing.
        * TensorBoard receives each scalar under its own tag, which produces one
          curve per metric in the TensorBoard UI.
        """
        # ------------------------------------------------------------------ #
        # 1. TensorBoard.                                                      #
        # ------------------------------------------------------------------ #
        if self._tb_writer is not None:
            for tag, value in scalars.items():
                # add_scalar(tag, scalar_value, global_step)
                # Using tag as-is (e.g. "episode_return", "loss") so curves
                # are labelled naturally in the TensorBoard UI.
                self._tb_writer.add_scalar(tag, float(value), global_step=step)

        # ------------------------------------------------------------------ #
        # 2. CSV.                                                              #
        # ------------------------------------------------------------------ #
        # Build the full row including the two always-present meta columns.
        elapsed_s = time.time() - self._start_time
        row: dict[str, Any] = {"step": step, "elapsed_s": round(elapsed_s, 3)}
        row.update({k: float(v) for k, v in scalars.items()})

        # Check whether any new keys have appeared since the last call.
        new_keys = [k for k in row if k not in self._csv_columns]
        if new_keys:
            self._csv_columns.extend(new_keys)
            # If the file is already open we need to re-open with the wider
            # header.  Because CSV headers are written on construction and
            # DictWriter uses extrasaction='ignore' by default, the safest
            # approach is to close and reopen, appending a "# columns changed"
            # sentinel so old readers still parse the existing rows.
            #
            # Strategy: keep a single writer; use extrasaction='ignore' and
            # write the header once on first open (so we never need to rewrite
            # past rows).  New-key rows will have empty strings for columns
            # that did not exist when those rows were written; that is clearly
            # visible and acceptable for a teaching tool.
            if self._csv_file is not None:
                # Flush existing state, re-create writer with extended fieldnames.
                # DictWriter.fieldnames is not settable directly, so we close
                # and reopen in append mode.
                self._csv_file.flush()
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
                # Fall through to the open-if-None block below.

        if self._csv_file is None:
            # Open (or reopen) the CSV file in append mode.
            # If this is a fresh file the DictWriter will write the header;
            # if we are re-opening after a columns change we skip the header
            # to avoid duplicating it.
            file_exists_and_non_empty = (
                self._csv_path.exists() and self._csv_path.stat().st_size > 0
            )
            self._csv_file = open(self._csv_path, "a", newline="", encoding="utf-8")  # noqa: SIM115
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_columns,
                extrasaction="ignore",
                restval="",  # empty string for missing keys in older rows
            )
            if not file_exists_and_non_empty:
                # Write header only for a brand-new file.
                self._csv_writer.writeheader()

        # Write the row; extrasaction='ignore' silently drops keys not in
        # fieldnames (shouldn't happen because we extended them above).
        assert self._csv_writer is not None
        self._csv_writer.writerow(row)
        self._csv_file.flush()  # type: ignore[union-attr]

        # W&B HOOK — log here too:
        #   if self._wandb_run is not None:
        #       import wandb
        #       wandb.log(scalars, step=step)

    def close(self) -> None:
        """Flush and close all sinks.  Safe to call more than once."""
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None

        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        # W&B HOOK — finish the run here:
        #   if self._wandb_run is not None:
        #       import wandb
        #       wandb.finish()
        #       self._wandb_run = None

    # ---------------------------------------------------------------------- #
    # Context-manager support so callers can use `with RunLogger(...) as lg:` #
    # ---------------------------------------------------------------------- #

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        tb_status = "TensorBoard+CSV" if self._tb_writer is not None else "CSV-only"
        return f"RunLogger(run_dir={self.run_dir!r}, mode={tb_status!r})"
