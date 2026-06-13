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


class SimpleCallbackBridge:
    """Bridge between a plain ``Callable[[dict], None]`` and SB3's callback API.

    SB3 expects a ``stable_baselines3.common.callbacks.BaseCallback`` subclass.
    This class inherits from ``BaseCallback`` (lazy-imported so the module
    stays importable without SB3) and every ``log_every`` env steps it builds
    a metrics dict from ``self.model.logger`` / ``self.locals`` and forwards
    it to the user-supplied *callback*.

    How SB3 calls BaseCallback
    --------------------------
    SB3 calls ``callback._on_step()`` after every env step.  It also calls
    ``callback._on_rollout_end()`` at the end of each rollout (PPO) and
    ``callback._on_training_end()`` once training finishes.  We only hook
    ``_on_step()`` here; subclasses can override the others.

    Metrics dict shape
    ------------------
    The dict we build on each log event looks like::

        {
            "step":           <num_timesteps>,
            "episode_return": <mean ep reward from SB3's EP_REW_MEAN if available>,
            "loss":           <actor/critic loss from SB3 logger if available>,
            "success_rate":   <EP_SUCCESS_RATE from SB3 ep_info_buffer if available>,
        }

    Parameters
    ----------
    callback:
        The plain callable to forward metrics to (e.g. the one returned by
        :func:`make_logging_callback`).
    log_every:
        How many env steps between successive callback calls.  1 000 is a
        reasonable default that keeps the CSV manageable while still giving
        smooth TensorBoard curves.
    """

    def __init__(
        self,
        callback: Callable[[dict[str, Any]], None],
        log_every: int = 1000,
    ) -> None:
        # Lazy-import BaseCallback so the module is importable without SB3.
        try:
            from stable_baselines3.common.callbacks import BaseCallback  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required to use SimpleCallbackBridge.  "
                "Install it with: pip install stable-baselines3"
            ) from exc

        # Dynamically create the class at construction time (rather than at
        # class-definition time) so we can inherit from BaseCallback without
        # forcing an SB3 import when the module is first loaded.
        #
        # We do this by creating an inner class and then *replacing* self with
        # an instance of it.  Python allows __class__ reassignment for this
        # pattern via __new__/__init__ but the cleanest teaching approach is
        # to subclass dynamically.
        #
        # For simplicity in a teaching lab we use a slightly different pattern:
        # store the user callback as an attribute and delegate from the SB3
        # hook methods.  We build the true BaseCallback subclass here.

        _user_callback = callback
        _log_every = max(1, int(log_every))

        class _Bridge(BaseCallback):  # type: ignore[misc]
            """Inner SB3-compatible callback; delegates to the user callback."""

            def __init__(self, verbose: int = 0) -> None:
                super().__init__(verbose=verbose)
                self._user_callback = _user_callback
                self._log_every = _log_every

            def _on_step(self) -> bool:
                """Called by SB3 after every env step.

                Returns True to continue training (False would stop early).
                """
                # Only log every _log_every steps to keep overhead low.
                if self.num_timesteps % self._log_every != 0:
                    return True

                # ---------------------------------------------------------- #
                # Build the metrics dict from whatever SB3 makes available.   #
                #                                                              #
                # SB3 stores rolling episode stats in ep_info_buffer (a deque #
                # of dicts with keys 'r' (return), 'l' (length), 't' (time)). #
                # ---------------------------------------------------------- #
                metrics: dict[str, Any] = {"step": self.num_timesteps}

                # Episode return: mean of the last few completed episodes.
                ep_buf = getattr(self.model, "ep_info_buffer", None)
                if ep_buf and len(ep_buf) > 0:
                    returns = [float(ep["r"]) for ep in ep_buf if "r" in ep]
                    if returns:
                        metrics["episode_return"] = sum(returns) / len(returns)

                # Success rate: SB3 stores is_success in ep_info_buffer too
                # when the env returns it in the info dict (HerReplayBuffer
                # and GoalEnv setups).
                if ep_buf and len(ep_buf) > 0:
                    successes = [float(ep["is_success"]) for ep in ep_buf if "is_success" in ep]
                    if successes:
                        metrics["success_rate"] = sum(successes) / len(successes)
                        # Also expose raw flag for the downstream callback's
                        # rolling-window tracker.
                        metrics["is_success"] = successes[-1]

                # Loss values: SB3 exposes them via model.logger.
                # The SB3 logger's name_to_value dict holds the most recently
                # dumped scalars (dumped at each gradient step).
                try:
                    sb3_log_vals = getattr(self.model.logger, "name_to_value", {})
                    # Common SB3 keys: train/loss, train/actor_loss,
                    # train/critic_loss, train/value_loss.
                    for sb3_key in (
                        "train/loss",
                        "train/actor_loss",
                        "train/critic_loss",
                        "train/value_loss",
                    ):
                        if sb3_key in sb3_log_vals:
                            # Strip the "train/" prefix for a cleaner log key.
                            short_key = sb3_key.split("/", 1)[-1]
                            metrics[short_key] = float(sb3_log_vals[sb3_key])
                except Exception:  # pragma: no cover — defensive; SB3 API may vary
                    pass

                self._user_callback(metrics)
                return True  # True = keep training

            def _on_training_end(self) -> None:
                """Called once training finishes; emit a final log entry."""
                # Emit whatever SB3 has buffered so the last episode is recorded.
                # Reuse _on_step logic by faking the modulus condition.
                orig_ts = self.num_timesteps
                self.num_timesteps = 0  # force modulus == 0 check to pass
                # Temporarily set to 0 so the modulus check passes.
                # (num_timesteps % 0 would error, so we use a flag instead.)
                self.num_timesteps = orig_ts
                # Build and forward the final metrics dict.
                metrics: dict[str, Any] = {"step": orig_ts}
                ep_buf = getattr(self.model, "ep_info_buffer", None)
                if ep_buf and len(ep_buf) > 0:
                    returns = [float(ep["r"]) for ep in ep_buf if "r" in ep]
                    if returns:
                        metrics["episode_return"] = sum(returns) / len(returns)
                self._user_callback(metrics)

        # Store the inner class as our "instance" by changing __class__.
        # This is the standard trick for making a factory function that
        # returns an instance of a dynamically-defined subclass.
        #
        # Because __init__ must return None we can't do `return _Bridge()`,
        # so we mutate self to look and behave exactly like a _Bridge.
        #
        # Simpler approach for a teaching codebase: just make SimpleCallbackBridge
        # a factory *function* that returns a BaseCallback instance.  But the
        # spec says "class SimpleCallbackBridge(BaseCallback)" so we honour that
        # by using __class__ reassignment.
        self.__class__ = _Bridge  # type: ignore[assignment]
        _Bridge.__init__(self)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# (c) FoxgloveCallback                                                         #
#     Optional SB3 callback that attaches a FoxgloveStreamer for SB3 runs.     #
# --------------------------------------------------------------------------- #


class FoxgloveCallback:
    """SB3 callback that streams robot state + metrics to Foxglove during SB3 runs.

    This is intentionally a *thin stub*: in SB3 runs the env may or may not be
    wrapped, and extracting joint angles from an SB3 VecEnv is cumbersome.
    The callback therefore publishes only the *scalar* metrics (distance, reward,
    episode_return, success_rate) that it can read from SB3's ep_info_buffer,
    skipping the full 3-D robot-pose publishing (which requires joint_q and
    ee_pos — available only from the unwrapped env).

    For full 3-D streaming, set ``render_mode="foxglove"`` directly on the
    environment instead of using this callback.

    Parameters
    ----------
    streamer:
        A :class:`~rl_lab.viz.foxglove_bridge.FoxgloveStreamer` instance
        (already constructed by the caller so *this* callback does not own
        its lifetime).
    publish_every:
        How many env steps between streamer.publish() calls.  Defaults to
        500 (roughly 30 fps at 15 000 steps/second on a laptop).
    """

    def __init__(
        self,
        streamer: FoxgloveStreamer,
        publish_every: int = 500,
    ) -> None:
        # Lazy SB3 import.
        try:
            from stable_baselines3.common.callbacks import BaseCallback  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required to use FoxgloveCallback.  "
                "Install it with: pip install stable-baselines3"
            ) from exc

        import numpy as np

        _streamer = streamer
        _publish_every = max(1, int(publish_every))
        _np = np

        class _FG(BaseCallback):  # type: ignore[misc]
            """Inner SB3 callback that publishes metrics to Foxglove."""

            def __init__(self, verbose: int = 0) -> None:
                super().__init__(verbose=verbose)
                self._streamer = _streamer
                self._publish_every = _publish_every

            def _on_step(self) -> bool:
                if not self._streamer.enabled:
                    return True
                if self.num_timesteps % self._publish_every != 0:
                    return True

                # Pull whatever scalar we can from the ep_info_buffer.
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

                # We do not have joint_q / ee_pos / target here, so we publish
                # zeros for the geometry and rely on the caller to use a
                # render_mode="foxglove" env for full 3-D streaming.
                _zero3 = _np.zeros(3, dtype=_np.float64)
                _zero4 = _np.zeros(4, dtype=_np.float64)
                self._streamer.publish(
                    joint_q=_zero4,
                    p_ee=_zero3,
                    g=_zero3,
                    dist=0.0,
                    reward=0.0,
                    episode_return=episode_return,
                    success_rate=success_rate,
                )
                return True

            def _on_training_end(self) -> None:
                """Called once by SB3 when training finishes."""
                # No cleanup needed — the caller owns the streamer lifetime.
                pass

        # Same __class__ reassignment pattern as SimpleCallbackBridge.
        self.__class__ = _FG  # type: ignore[assignment]
        _FG.__init__(self)  # type: ignore[arg-type]


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
