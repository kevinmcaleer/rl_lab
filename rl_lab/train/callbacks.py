"""Training callbacks — the glue between algorithms, the logger, and the viz.

Callbacks in this lab serve two roles:

1. **Logging**: relay per-episode / per-update metrics to a
   :class:`~rl_lab.train.logger.RunLogger` (and optionally a
   :class:`~rl_lab.viz.foxglove_bridge.FoxgloveStreamer`).
2. **Checkpointing**: wrap SB3's ``CheckpointCallback`` behind a one-liner
   factory so the CLI and notebooks do not need to know SB3 internals.

The from-scratch algorithms (Q-learning, SARSA, DQN, REINFORCE, PPO-min) call
the plain ``Callable[[dict], None]`` returned by :func:`make_logging_callback`.
The SB3 adapter uses :class:`SimpleCallbackBridge` (a ``BaseCallback``) and
:class:`FoxgloveCallback`.

All SB3 / torch imports are intentionally lazy — importing this module is
cheap even without those packages installed.
"""

from __future__ import annotations

import collections
import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only imported for type annotations (never executed at import time).
    from rl_lab.train.logger import RunLogger
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert *value* to float without raising; return *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# (a) make_logging_callback                                                    #
#     Used by ALL from-scratch algorithms (tabular + neural).                  #
# --------------------------------------------------------------------------- #


def make_logging_callback(
    logger: RunLogger,
    streamer: FoxgloveStreamer | None = None,
) -> Callable[[dict[str, Any]], None]:
    """Return a callback suitable for passing to ``algo.train(callback=...)`.

    The from-scratch algorithms call ``callback(metrics)`` at the end of each
    logged episode with a dict like::

        {
            "step":            1_000,   # total env steps so far
            "episode_return":  3.2,     # undiscounted sum of rewards
            "loss":            0.041,   # algorithm-specific (may be absent)
            "distance":        0.08,    # tip-to-target at episode end (may be absent)
            "is_success":      False,   # bool or 0/1 (may be absent)
        }

    The callback:

    * Forwards all numeric scalars to the :class:`~rl_lab.train.logger.RunLogger`.
    * Tracks a rolling ``success_rate`` over the last 100 episodes (consistent
      with the ``is_success`` flag in ``info``) and includes it in the log.
    * If a *streamer* is provided, **no extra publish call is made here** —
      the environment itself calls ``streamer.publish(...)`` per step, so
      calling it here too would duplicate frames.  The streamer argument is
      retained only in case a future version needs episode-level summary
      events (e.g. logging episode_return as a separate channel).

    Parameters
    ----------
    logger:
        An open :class:`~rl_lab.train.logger.RunLogger` instance.
    streamer:
        Optional :class:`~rl_lab.viz.foxglove_bridge.FoxgloveStreamer`.  If
        given the callback is a no-op for viz (the env drives rendering).

    Returns
    -------
    Callable[[dict], None]
        The callback function.  Thread-safe for single-threaded algorithms
        (no locking — this is a teaching lab, not production multi-worker code).
    """
    # Rolling window for tracking success rate.
    # deque with maxlen auto-discards oldest entries — O(1) append.
    _recent_successes: collections.deque[float] = collections.deque(maxlen=100)

    def _callback(metrics: dict[str, Any]) -> None:
        """Inner callback called once per logged episode/update."""
        step = int(metrics.get("step", 0))

        # ------------------------------------------------------------------ #
        # Track success rate.                                                  #
        #                                                                      #
        # Algorithms pass either a boolean "is_success" or the raw info dict   #
        # entry. We accept both bool and numeric (1/0) forms.                  #
        # ------------------------------------------------------------------ #
        if "is_success" in metrics:
            _recent_successes.append(1.0 if metrics["is_success"] else 0.0)

        # Compute rolling success rate (mean over up-to-100 recent episodes).
        success_rate: float = (
            sum(_recent_successes) / len(_recent_successes) if _recent_successes else 0.0
        )

        # ------------------------------------------------------------------ #
        # Build the scalars dict that goes to the logger.                      #
        # Only include numeric values; skip step itself (it's the x-axis).     #
        # ------------------------------------------------------------------ #
        scalars: dict[str, float] = {}
        _skip_keys = {"step"}  # already used as x-axis
        for key, value in metrics.items():
            if key in _skip_keys:
                continue
            # Silently skip non-numeric values (e.g. a numpy array or string)
            # so the callback never crashes on unexpected payload shapes.
            with contextlib.suppress(TypeError, ValueError):
                scalars[key] = float(value)

        # Always include the derived success_rate so every run has it.
        scalars["success_rate"] = success_rate

        # Log to TensorBoard + CSV.
        logger.log_scalars(step=step, scalars=scalars)

        # Streamer note: the env's own _maybe_render() already calls
        # streamer.publish() per step; we deliberately do NOT call it here
        # to avoid double-publishing.  If you ever add episode-boundary
        # events (e.g. a Foxglove annotation marker at the end of each
        # episode) this is the right place.
        _ = streamer  # referenced to silence linters

    return _callback


# --------------------------------------------------------------------------- #
# (b) SimpleCallbackBridge                                                     #
#     Adapts our Callable callback for Stable-Baselines3's BaseCallback API.   #
# --------------------------------------------------------------------------- #


def SimpleCallbackBridge(
    callback: Callable[[dict[str, Any]], None],
    log_every: int = 1000,
):
    """Wrap a plain ``callback(metrics: dict)`` as a Stable-Baselines3 callback.

    Returns a ``BaseCallback`` instance that, every ``log_every`` env steps,
    builds a metrics dict from SB3's ``ep_info_buffer`` / logger and forwards it
    to ``callback``. Implemented as a factory so ``stable_baselines3`` is imported
    lazily (this module stays importable without it).
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - exercised only without SB3
        raise ImportError(
            "stable-baselines3 is required to use SimpleCallbackBridge.  "
            "Install it with: pip install stable-baselines3"
        ) from exc

    _user_callback = callback
    _log_every = max(1, int(log_every))

    class _Bridge(BaseCallback):  # type: ignore[misc]
        """SB3-compatible callback that delegates to the user callback."""

        def __init__(self, verbose: int = 0) -> None:
            super().__init__(verbose=verbose)
            self._user_callback = _user_callback
            self._log_every = _log_every

        def _build_metrics(self) -> dict[str, Any]:
            metrics: dict[str, Any] = {"step": self.num_timesteps}
            ep_buf = getattr(self.model, "ep_info_buffer", None)
            if ep_buf and len(ep_buf) > 0:
                returns = [float(ep["r"]) for ep in ep_buf if "r" in ep]
                if returns:
                    metrics["episode_return"] = sum(returns) / len(returns)
                successes = [float(ep["is_success"]) for ep in ep_buf if "is_success" in ep]
                if successes:
                    metrics["success_rate"] = sum(successes) / len(successes)
                    metrics["is_success"] = successes[-1]
            try:
                sb3_log_vals = getattr(self.model.logger, "name_to_value", {})
                for sb3_key in (
                    "train/loss",
                    "train/actor_loss",
                    "train/critic_loss",
                    "train/value_loss",
                ):
                    if sb3_key in sb3_log_vals:
                        metrics[sb3_key.split("/", 1)[-1]] = float(sb3_log_vals[sb3_key])
            except Exception:  # pragma: no cover - defensive; SB3 API may vary
                pass
            return metrics

        def _on_step(self) -> bool:
            if self.num_timesteps % self._log_every == 0:
                self._user_callback(self._build_metrics())
            return True

        def _on_training_end(self) -> None:
            self._user_callback(self._build_metrics())

    return _Bridge()


# --------------------------------------------------------------------------- #
# (c) FoxgloveCallback                                                         #
#     Optional SB3 callback that attaches a FoxgloveStreamer for SB3 runs.     #
# --------------------------------------------------------------------------- #


def FoxgloveCallback(streamer: FoxgloveStreamer, publish_every: int = 500):
    """SB3 callback that streams scalar metrics to Foxglove during SB3 runs.

    A thin helper: in SB3 runs joint_q/ee_pos are not readily available, so this
    publishes only the scalar metrics it can read from ``ep_info_buffer``. For
    full 3-D streaming, set ``render_mode="foxglove"`` on the environment instead.
    Returns a ``BaseCallback`` instance (SB3 imported lazily).
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - exercised only without SB3
        raise ImportError(
            "stable-baselines3 is required to use FoxgloveCallback.  "
            "Install it with: pip install stable-baselines3"
        ) from exc

    import numpy as np

    _streamer = streamer
    _publish_every = max(1, int(publish_every))

    class _FG(BaseCallback):  # type: ignore[misc]
        """SB3 callback that publishes scalar metrics to Foxglove."""

        def __init__(self, verbose: int = 0) -> None:
            super().__init__(verbose=verbose)
            self._streamer = _streamer
            self._publish_every = _publish_every

        def _on_step(self) -> bool:
            if not self._streamer.enabled:
                return True
            if self.num_timesteps % self._publish_every != 0:
                return True
            ep_buf = getattr(self.model, "ep_info_buffer", None)
            episode_return = 0.0
            success_rate = 0.0
            if ep_buf and len(ep_buf) > 0:
                returns = [float(ep["r"]) for ep in ep_buf if "r" in ep]
                if returns:
                    episode_return = sum(returns) / len(returns)
                successes = [float(ep["is_success"]) for ep in ep_buf if "is_success" in ep]
                if successes:
                    success_rate = sum(successes) / len(successes)
            self._streamer.publish(
                joint_q=np.zeros(4, dtype=np.float64),
                p_ee=np.zeros(3, dtype=np.float64),
                g=np.zeros(3, dtype=np.float64),
                dist=0.0,
                reward=0.0,
                episode_return=episode_return,
                success_rate=success_rate,
            )
            return True

    return _FG()


# --------------------------------------------------------------------------- #
# (d) make_sb3_checkpoint_callback                                             #
#     One-liner factory for SB3's CheckpointCallback.                          #
# --------------------------------------------------------------------------- #


def make_sb3_checkpoint_callback(
    save_freq: int,
    save_path: str,
) -> Any:
    """Return an SB3 ``CheckpointCallback`` that saves every *save_freq* steps.

    This thin factory exists so the CLI and notebooks never need to import SB3
    directly just to set up checkpointing.

    Parameters
    ----------
    save_freq:
        Save a model checkpoint every this many env steps.
    save_path:
        Directory (will be created) where ``.zip`` checkpoints are written.
        SB3 names them ``<algo>_<num_timesteps>_steps.zip``.

    Returns
    -------
    stable_baselines3.common.callbacks.CheckpointCallback

    Raises
    ------
    ImportError
        If ``stable-baselines3`` is not installed.

    Example
    -------
    ::

        ckpt_cb = make_sb3_checkpoint_callback(save_freq=10_000, save_path="runs/ppo")
        model.learn(total_timesteps=100_000, callback=ckpt_cb)
    """
    try:
        from stable_baselines3.common.callbacks import CheckpointCallback  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for make_sb3_checkpoint_callback.  "
            "Install it with: pip install stable-baselines3"
        ) from exc

    return CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix="model",
        verbose=1,
    )
