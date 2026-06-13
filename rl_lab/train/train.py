"""Training entry point for the Buddy Jr RL Lab.

The public API is a single function :func:`train` that:

1. Resolves the algorithm key and default env id via the registry.
2. Creates a Gymnasium environment (optionally with render_mode or goal_env).
3. Seeds every RNG for reproducibility.
4. Instantiates the algorithm and trains it with a logging callback.
5. Saves a self-describing checkpoint (model weights + JSON metadata).
6. Returns the checkpoint directory path.

Design notes
------------
* All heavy imports (torch, SB3, algorithm classes) happen *inside* the
  function body so ``from rl_lab.train.train import train`` does not add
  latency to the CLI help screen.
* The ``run_name`` is derived from ``{algo}_{timestamp}`` when not supplied,
  giving unique, human-readable run directories.
* Checkpoint layout::

      runs/<run_name>/model          ← algo.save() artefact
      runs/<run_name>/model.meta.json← metadata sidecar (load_metadata reads this)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any


def train(
    algo: str,
    env_id: str | None = None,
    *,
    total_steps: int = 50_000,
    seed: int = 0,
    hparams: dict[str, Any] | None = None,
    logdir: str = "runs",
    render: str | None = None,
    run_name: str | None = None,
) -> str:
    """Train an RL algorithm on a Buddy Jr environment and save a checkpoint.

    Parameters
    ----------
    algo:
        Algorithm key, e.g. ``"ppo"``, ``"dqn"``, ``"qlearning"``.
        Case-insensitive; must be a key in :data:`rl_lab.algos.registry.ALGORITHMS`.
    env_id:
        Gymnasium environment id, e.g. ``"BuddyJrReach-v0"``.
        When *None* the registry picks a sensible default for *algo*.
    total_steps:
        Approximate number of environment steps to train for.  The exact count
        may differ slightly depending on how the algorithm batches updates.
    seed:
        Master seed forwarded to :func:`~rl_lab.utils.seeding.set_global_seed`
        (seeds Python, NumPy, PyTorch, and the env).
    hparams:
        Extra hyper-parameters forwarded verbatim to the algorithm constructor.
        E.g. ``{"learning_rate": 3e-4, "n_steps": 512}``.
    logdir:
        Root directory under which ``<run_name>/`` is created.
    render:
        Render mode string.  ``"foxglove"`` streams live to Foxglove Studio;
        ``"human"`` opens a local viewer (env-dependent); ``None`` disables.
    run_name:
        Explicit run name.  Auto-generated from ``{algo}_{timestamp}`` when
        *None*.

    Returns
    -------
    str
        Absolute path to the saved checkpoint file (without extension — algo
        implementations add their own extension, e.g. ``.npz`` or ``.zip``).
    """
    # ------------------------------------------------------------------
    # 0.  Lazy imports — keep CLI --help fast
    # ------------------------------------------------------------------
    import gymnasium as gym

    import rl_lab  # ensure envs are registered
    from rl_lab.algos.registry import make_algorithm, recommended_env_id
    from rl_lab.train.callbacks import make_logging_callback
    from rl_lab.train.logger import RunLogger
    from rl_lab.utils.checkpoint import save_checkpoint
    from rl_lab.utils.seeding import set_global_seed

    _ = rl_lab  # imported for side-effect (register_envs)

    hparams = hparams or {}
    algo_key = algo.lower()

    # ------------------------------------------------------------------
    # 1.  Resolve environment id
    # ------------------------------------------------------------------
    resolved_env_id: str = env_id or recommended_env_id(algo_key)

    # HER needs a goal-conditioned observation space so the replay buffer
    # can assemble (obs, achieved_goal, desired_goal) tuples.
    make_kwargs: dict[str, Any] = {}
    if algo_key == "her":
        make_kwargs["goal_env"] = True
    if render is not None and render != "foxglove":
        # Pass render_mode directly for non-Foxglove modes (e.g. 'human').
        make_kwargs["render_mode"] = render

    # ------------------------------------------------------------------
    # 2.  Build run name and output directory
    # ------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_name: str = run_name or f"{algo_key}_{timestamp}"
    run_dir = Path(logdir) / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "model"

    # ------------------------------------------------------------------
    # 3.  Create environment
    # ------------------------------------------------------------------
    env = gym.make(resolved_env_id, **make_kwargs)

    # ------------------------------------------------------------------
    # 4.  Seed everything
    # ------------------------------------------------------------------
    # set_global_seed also calls env.reset(seed=seed) as a side-effect.
    set_global_seed(seed, env=env)

    # ------------------------------------------------------------------
    # 5.  Create logger (TensorBoard + stdout)
    # ------------------------------------------------------------------
    logger = RunLogger(str(run_dir), "tb_logs")

    # ------------------------------------------------------------------
    # 6.  Optionally create a Foxglove streamer
    # ------------------------------------------------------------------
    streamer = None
    if render == "foxglove":
        # Import lazily — foxglove-sdk not required for CPU-only runs.
        from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

        streamer = FoxgloveStreamer(render_mode="foxglove")
        print(f"Foxglove live stream started at {streamer.app_url}")

    # ------------------------------------------------------------------
    # 7.  Instantiate algorithm
    # ------------------------------------------------------------------
    algorithm = make_algorithm(algo_key, env, seed=seed, **hparams)

    # ------------------------------------------------------------------
    # 8.  Build the logging callback and train
    # ------------------------------------------------------------------
    callback = make_logging_callback(logger, streamer)

    print(
        f"Training {algo_key!r} on {resolved_env_id!r} " f"for {total_steps:,} steps  (seed={seed})"
    )
    print(f"Run directory: {run_dir}")

    try:
        _history = algorithm.train(total_steps, callback=callback)
    finally:
        # Always close logger/streamer even if training raises.
        logger.close()
        if streamer is not None:
            streamer.close()
        env.close()

    # ------------------------------------------------------------------
    # 9.  Save checkpoint + metadata sidecar
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {
        "algo": algo_key,
        "env_id": resolved_env_id,
        "hparams": hparams,
        "seed": seed,
        "total_steps": total_steps,
        "run_name": resolved_run_name,
    }
    save_checkpoint(algorithm, str(checkpoint_path), metadata)

    print(f"Checkpoint saved to {checkpoint_path}")
    return str(checkpoint_path)
