"""Composable reward terms and the three reward modes for the reach task.

Design goals:

* **What is rewarded == what is plotted.** The distance to the target is
  computed once per step (from cached FK) and feeds both the reward and
  ``info['distance']`` / the Foxglove metrics.
* **Composable terms.** Each term is a small pure function so a lesson can
  inspect or re-weight it, and ``test_rewards.py`` can check it in isolation.
* **Sane scale.** The success bonus is ``+10`` (not ``+1000``) so it does not
  swamp the per-step shaping; all modes stay on a comparable scale.

Three modes (chosen by ``RewardConfig.mode``):

* ``sparse`` — ``+bonus`` on success, an optional small per-step penalty.
* ``dense``  — ``-distance`` (+ bonus on success, − small control penalty).
* ``shaped`` — potential-based progress ``(prev_distance − distance)`` + a small
  control penalty (+ bonus). Potential-based shaping keeps the optimal policy
  unchanged while speeding learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_lab.robot import buddy_jr as bj

SUCCESS_BONUS: float = 10.0


@dataclass(frozen=True)
class RewardConfig:
    """Knobs for the reward function."""

    mode: str = "dense"  # "sparse" | "dense" | "shaped"
    success_tol: float = 0.02  # metres; tip within this of the target == success
    success_bonus: float = SUCCESS_BONUS
    step_penalty: float = 0.0  # sparse-mode per-step penalty (e.g. 0.01)
    control_weight: float = 0.01  # penalise large/jerky actions
    limit_weight: float = 0.0  # penalise approaching joint limits (off by default)
    shaping_scale: float = 10.0  # scales potential-based progress in "shaped"


# --------------------------------------------------------------------------- #
# Individual terms (pure functions — each independently testable)
# --------------------------------------------------------------------------- #
def distance_term(distance: float) -> float:
    """Closer is better: ``-distance`` (metres)."""
    return -float(distance)


def success_term(distance: float, tol: float, bonus: float = SUCCESS_BONUS) -> float:
    """``+bonus`` once the tip is within ``tol`` of the target, else 0."""
    return float(bonus) if distance <= tol else 0.0


def control_term(action: np.ndarray) -> float:
    """Penalise large actions: ``-||action||^2`` (encourages economical motion)."""
    a = np.asarray(action, dtype=np.float64)
    return -float(np.sum(a * a))


def smoothness_term(action: np.ndarray, prev_action: np.ndarray) -> float:
    """Penalise jerky changes between consecutive actions."""
    a = np.asarray(action, dtype=np.float64)
    p = np.asarray(prev_action, dtype=np.float64)
    return -float(np.sum((a - p) ** 2))


def joint_limit_term(q: np.ndarray, soft_fraction: float = 0.9) -> float:
    """Penalise joints driven past ``soft_fraction`` of their limit."""
    q = np.asarray(q, dtype=np.float64)
    over = np.clip(np.abs(q) / bj.JOINT_LIMIT - soft_fraction, 0.0, None)
    return -float(np.sum(over * over))


def is_success(distance: float, tol: float) -> bool:
    return bool(distance <= tol)


# --------------------------------------------------------------------------- #
# Mode combiner
# --------------------------------------------------------------------------- #
def compute_reward(
    cfg: RewardConfig,
    *,
    distance: float,
    prev_distance: float,
    action: np.ndarray,
    q: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Return ``(reward, components)`` for the configured mode.

    ``components`` breaks the reward down (handy for plots and tests). The same
    ``distance`` value is used here and reported in ``info``.
    """
    success = is_success(distance, cfg.success_tol)
    bonus = success_term(distance, cfg.success_tol, cfg.success_bonus)
    control = cfg.control_weight * control_term(action)
    limit = cfg.limit_weight * joint_limit_term(q)

    if cfg.mode == "sparse":
        reward = bonus - cfg.step_penalty
    elif cfg.mode == "dense":
        reward = distance_term(distance) + bonus + control + limit
    elif cfg.mode == "shaped":
        progress = cfg.shaping_scale * (float(prev_distance) - float(distance))
        reward = progress + bonus + control + limit
    else:
        raise ValueError(f"unknown reward mode {cfg.mode!r} (sparse|dense|shaped)")

    components = {
        "distance": float(distance),
        "bonus": float(bonus),
        "control": float(control),
        "limit": float(limit),
        "is_success": float(success),
        "reward": float(reward),
    }
    return float(reward), components
