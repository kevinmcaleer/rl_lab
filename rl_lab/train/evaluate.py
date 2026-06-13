"""Evaluation / roll-out entry point for the Buddy Jr RL Lab.

The public API is :func:`evaluate`, which:

1. Reads the self-describing metadata sidecar next to the checkpoint to
   recover the algorithm name, env id, and hyper-parameters that were used
   at training time — no need to remember flags.
2. Rebuilds the env and algorithm via the registry.
3. Loads the saved weights into the algorithm (``algo.load(checkpoint)``).
4. Rolls out *episodes* deterministic episodes, collecting per-episode return
   and the terminal ``info['is_success']`` flag.
5. Optionally streams to Foxglove (live WebSocket) or records an MCAP file.
6. Returns a summary dict with ``success_rate``, ``mean_return``, and
   ``episodes``.

Design notes
------------
* Like :mod:`rl_lab.train.train`, heavy imports are deferred so the CLI stays
  fast at parse time.
* The Foxglove streamer is kept in scope for the full evaluation and closed in
  a ``finally`` block to avoid resource leaks.
* ``info['is_success']`` is read at the *end of each episode* (when
  ``terminated or truncated`` is True) because the reach environments only
  set it to True on the terminal step.
"""

from __future__ import annotations

from typing import Any


def evaluate(
    checkpoint: str,
    *,
    env_id: str | None = None,
    episodes: int = 10,
    render: str | None = None,
    record_mcap: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a saved checkpoint and return a summary of performance.

    Parameters
    ----------
    checkpoint:
        Path to the saved model (without extension — the algorithm's own
        ``load`` method resolves the extension, e.g. ``.npz`` or ``.zip``).
    env_id:
        Override the environment id.  When *None* the env id stored in the
        checkpoint metadata is used (recommended).
    episodes:
        Number of full episodes to run.
    render:
        ``"foxglove"`` streams live to Foxglove Studio; ``"human"`` opens a
        local viewer; *None* disables.
    record_mcap:
        If given, record all steps to this ``*.mcap`` file (for later replay
        in Foxglove Studio).  Can be combined with ``render='foxglove'``.
    seed:
        Seed used for env reset and action-space sampling.

    Returns
    -------
    dict
        ``{"success_rate": float, "mean_return": float, "episodes": int}``
    """
    # ------------------------------------------------------------------
    # 0.  Lazy imports
    # ------------------------------------------------------------------
    import gymnasium as gym
    import numpy as np

    import rl_lab  # register envs
    from rl_lab.algos.registry import make_algorithm, recommended_env_id
    from rl_lab.utils.checkpoint import load_metadata
    from rl_lab.utils.seeding import set_global_seed

    _ = rl_lab

    # ------------------------------------------------------------------
    # 1.  Read self-describing metadata
    # ------------------------------------------------------------------
    metadata = load_metadata(checkpoint)
    algo_key: str = metadata.get("algo", "ppo")
    resolved_env_id: str = env_id or metadata.get("env_id") or recommended_env_id(algo_key)
    hparams: dict[str, Any] = metadata.get("hparams", {})

    # ------------------------------------------------------------------
    # 2.  Build environment
    # ------------------------------------------------------------------
    make_kwargs: dict[str, Any] = {}
    # HER needs goal-conditioned obs (Dict space) to rebuild the algorithm.
    if algo_key == "her":
        make_kwargs["goal_env"] = True
    if render is not None and render != "foxglove":
        make_kwargs["render_mode"] = render

    env = gym.make(resolved_env_id, **make_kwargs)
    set_global_seed(seed, env=env)

    # ------------------------------------------------------------------
    # 3.  Build algorithm and load weights
    # ------------------------------------------------------------------
    algorithm = make_algorithm(algo_key, env, seed=seed, **hparams)
    algorithm.load(checkpoint)

    # ------------------------------------------------------------------
    # 4.  Optionally set up Foxglove streamer
    # ------------------------------------------------------------------
    streamer = None
    if render == "foxglove" or record_mcap is not None:
        from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

        # Determine render mode: live, mcap-only, or both.
        fox_mode = "foxglove" if render == "foxglove" else "mcap"

        streamer = FoxgloveStreamer(render_mode=fox_mode, mcap_path=record_mcap)  # type: ignore[arg-type]
        if render == "foxglove" and streamer.app_url:
            print(f"Foxglove live stream: {streamer.app_url}")

    # ------------------------------------------------------------------
    # 5.  Roll out evaluation episodes
    # ------------------------------------------------------------------
    episode_returns: list[float] = []
    successes: list[bool] = []

    try:
        for ep_idx in range(episodes):
            obs, info = env.reset(seed=seed + ep_idx)
            ep_return: float = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                action, _state = algorithm.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)

                # Stream to Foxglove if enabled.
                if streamer is not None and streamer.enabled:
                    # Extract kinematic fields from info (set by BuddyJrReachEnv).
                    joint_q: np.ndarray = np.asarray(
                        info.get("joint_q", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64
                    )
                    ee_pos: np.ndarray = np.asarray(
                        info.get("ee_pos", [0.0, 0.0, 0.0]), dtype=np.float64
                    )
                    target: np.ndarray = np.asarray(
                        info.get("target", [0.0, 0.0, 0.0]), dtype=np.float64
                    )
                    dist: float = float(info.get("distance", 0.0))
                    streamer.publish(
                        joint_q,
                        ee_pos,
                        target,
                        dist,
                        reward=float(reward),
                        episode_return=ep_return,
                    )

            # Record terminal success flag.  BuddyJrReachEnv stores this in
            # info['is_success'] (set True on the terminal step only).
            episode_returns.append(ep_return)
            successes.append(bool(info.get("is_success", False)))

            print(
                f"  Episode {ep_idx + 1:>{len(str(episodes))}}/{episodes}  "
                f"return={ep_return:7.3f}  "
                f"success={successes[-1]}"
            )

    finally:
        env.close()
        if streamer is not None:
            streamer.close()

    # ------------------------------------------------------------------
    # 6.  Summarise
    # ------------------------------------------------------------------
    success_rate: float = float(np.mean(successes)) if successes else 0.0
    mean_return: float = float(np.mean(episode_returns)) if episode_returns else 0.0

    return {
        "success_rate": success_rate,
        "mean_return": mean_return,
        "episodes": episodes,
    }
