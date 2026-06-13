"""Observation and action space builders for the Buddy Jr reach task.

The canonical observation is a **17-D** continuous vector, scaled to roughly
``[-1, 1]`` so a network sees well-conditioned inputs:

==========================  ====  =====================================
component                   dims  notes
==========================  ====  =====================================
sin/cos of the 4 joints       8   avoids the +/-pi wraparound discontinuity
camera-tip position           3   scaled by :data:`POS_SCALE`
target position               3   scaled by :data:`POS_SCALE`
vector tip -> target          3   scaled, then clipped to [-1, 1]
==========================  ====  =====================================

With ``include_velocity=True`` the 4 joint velocities are appended (21-D). The
``goal_env=True`` variant returns a ``Dict`` of ``observation`` /
``achieved_goal`` / ``desired_goal`` (raw metre positions) for HER.

Actions are either a continuous ``Box(4,)`` in ``[-1, 1]`` (per-joint jog deltas)
or a ``Discrete(9)`` jog (no-op + nudge each of 4 joints +/-).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl_lab.robot import buddy_jr as bj

OBS_DIM: int = 17
OBS_DIM_WITH_VELOCITY: int = 21

#: Scale applied to metre positions so the ~0.22 m workspace maps into ~[-1, 1].
POS_SCALE: float = 4.5
#: Scale applied to joint velocities (rad/s) — the 6 rad/s limit -> ~[-1, 1].
VEL_SCALE: float = 1.0 / 6.0

#: Number of discrete jog actions: no-op + (+/- on each of the 4 joints).
N_DISCRETE_ACTIONS: int = 1 + 2 * bj.NUM_JOINTS  # = 9


def build_observation(
    q: np.ndarray,
    qd: np.ndarray,
    ee_pos: np.ndarray,
    target_pos: np.ndarray,
    include_velocity: bool = False,
) -> np.ndarray:
    """Assemble the scaled observation vector (clipped to ``[-1, 1]``)."""
    q = np.asarray(q, dtype=np.float64)
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    parts = [
        np.sin(q),
        np.cos(q),
        ee_pos * POS_SCALE,
        target_pos * POS_SCALE,
        (target_pos - ee_pos) * POS_SCALE,
    ]
    if include_velocity:
        parts.append(np.asarray(qd, dtype=np.float64) * VEL_SCALE)
    obs = np.concatenate(parts)
    expected = OBS_DIM_WITH_VELOCITY if include_velocity else OBS_DIM
    assert obs.shape == (expected,), f"observation is {obs.shape}, expected ({expected},)"
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


def make_observation_space(
    include_velocity: bool = False, goal_env: bool = False
) -> gym.spaces.Space:
    """Build the observation space (a ``Box``, or a HER ``Dict`` if ``goal_env``)."""
    n = OBS_DIM_WITH_VELOCITY if include_velocity else OBS_DIM
    box = gym.spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
    if not goal_env:
        return box
    goal_box = gym.spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32)
    return gym.spaces.Dict(
        {"observation": box, "achieved_goal": goal_box, "desired_goal": goal_box}
    )


def make_action_space(discrete: bool = False) -> gym.spaces.Space:
    """Continuous ``Box(4,)`` jog deltas in ``[-1, 1]``, or a ``Discrete(9)`` jog."""
    if discrete:
        return gym.spaces.Discrete(N_DISCRETE_ACTIONS)
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(bj.NUM_JOINTS,), dtype=np.float32)


def discrete_to_continuous(action: int) -> np.ndarray:
    """Map a ``Discrete(9)`` jog index to the equivalent continuous ``[-1,1]^4``.

    Index 0 is a no-op; indices 1..8 nudge joint ``(i-1)//2`` by +1 (odd) or
    -1 (even), so a discrete agent and a continuous agent issue the *same* kind
    of joint-target update.
    """
    delta = np.zeros(bj.NUM_JOINTS, dtype=np.float32)
    if action == 0:
        return delta
    joint = (action - 1) // 2
    delta[joint] = 1.0 if (action - 1) % 2 == 0 else -1.0
    return delta
