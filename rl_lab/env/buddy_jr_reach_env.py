"""``BuddyJrReachEnv`` — the canonical Gymnasium environment for the reach task.

The agent jogs the 4 joints to drive the camera tip onto a randomly placed,
always-reachable target. It wraps a :class:`~rl_lab.sim.base.SimBackend` (the
physics-free :class:`~rl_lab.sim.kinematic.KinematicBackend` by default, so it
runs anywhere; swap in PyBullet/MuJoCo for real dynamics), composes its reward
from :mod:`rl_lab.env.rewards`, and can stream itself live to Foxglove.

Gymnasium contract notes:

* ``terminated`` means *task success* (tip within tolerance); ``truncated``
  means the time limit was hit. They are returned separately, never conflated.
* ``info`` always carries ``{distance, is_success, joint_q, ee_pos, target}``.
* ``render_mode`` may be ``"human"``, ``"foxglove"`` or ``"rgb_array"``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from rl_lab.env import rewards as R
from rl_lab.env import spaces as S
from rl_lab.robot import buddy_jr as bj
from rl_lab.robot import kinematics as kin
from rl_lab.sim.base import SimBackend

#: Max change applied to each joint target per step (radians) for a unit action.
ACTION_SCALE: float = 0.1

#: Supported render modes (also published in ``metadata``).
RENDER_MODES: list[str] = ["human", "foxglove", "rgb_array"]


def make_backend(backend: str | SimBackend, ctrl_dt: float = 1.0 / 60.0) -> SimBackend:
    """Construct a backend from a name (or pass a ready instance through)."""
    if isinstance(backend, SimBackend):
        return backend
    name = backend.lower()
    if name == "kinematic":
        from rl_lab.sim.kinematic import KinematicBackend

        return KinematicBackend(ctrl_dt=ctrl_dt)
    if name == "pybullet":
        from rl_lab.sim.pybullet_sim import PyBulletBackend

        return PyBulletBackend("direct")
    if name == "mujoco":
        from rl_lab.sim.mujoco_sim import MujocoBackend

        return MujocoBackend(ctrl_dt=ctrl_dt)
    raise ValueError(f"unknown backend {backend!r} (kinematic|pybullet|mujoco)")


class BuddyJrReachEnv(gym.Env):
    """Reach a target with the camera tip of the Buddy Jr arm."""

    metadata = {"render_modes": RENDER_MODES, "render_fps": 30}

    def __init__(
        self,
        backend: str | SimBackend = "kinematic",
        *,
        render_mode: str | None = None,
        reward_mode: str = "dense",
        max_steps: int = 200,
        include_velocity: bool = False,
        goal_env: bool = False,
        camera_point: bool = False,
        align_weight: float = 0.0,
        action_scale: float = ACTION_SCALE,
        success_tol: float = 0.02,
        target_radius: tuple[float, float] = (0.04, 0.155),
        reset_noise: float = 0.1,
        viz_every_n_episodes: int = 1,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in RENDER_MODES:
            raise ValueError(f"invalid render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.include_velocity = include_velocity
        self.goal_env = goal_env
        self.camera_point = camera_point
        # Camera-pointing variant rewards aiming the camera at the target.
        self.align_weight = float(align_weight) if align_weight else (1.0 if camera_point else 0.0)
        self.action_scale = float(action_scale)
        self.target_radius = target_radius
        self.reset_noise = float(reset_noise)
        self.viz_every_n_episodes = max(1, int(viz_every_n_episodes))
        self.reward_cfg = R.RewardConfig(
            mode=reward_mode,
            success_tol=success_tol,
            limit_weight=0.0,
        )

        self._backend = make_backend(backend)
        self.observation_space = S.make_observation_space(include_velocity, goal_env)
        self.action_space = S.make_action_space(discrete=False)

        self._target = np.zeros(3, dtype=np.float64)
        self._prev_distance = 0.0
        self._steps = 0
        self._episode = -1
        self._episode_return = 0.0
        self._viz: Any = None

    # --------------------------------------------------------------- helpers
    def _sample_target(self) -> np.ndarray:
        """Sample an always-reachable target in the shell (z above the bench)."""
        base = np.array([0.0, 0.0, bj.BASE_HEIGHT])
        lo, hi = self.target_radius
        for _ in range(200):
            r = self.np_random.uniform(lo, hi)
            az = self.np_random.uniform(-bj.JOINT_LIMIT, bj.JOINT_LIMIT)
            el = self.np_random.uniform(-0.4, 1.0)
            p = base + r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            if p[2] >= 0.02 and kin.is_reachable(p):
                return p
        # Fallback: a configuration we know is reachable.
        return kin.forward(np.array([0.0, 0.5, -0.5, 0.0])).position

    def _ee_pos(self) -> np.ndarray:
        return self._backend.get_camera_tip_pose()[0]

    def _camera_alignment(self, ee: np.ndarray, quat_xyzw: np.ndarray) -> float:
        """Cosine similarity between the camera's forward axis and the to-target dir.

        ``+1`` means the camera points straight at the target. The forward axis
        is the camera link's local +Z rotated into the world frame.
        """
        to_target = self._target - ee
        n = float(np.linalg.norm(to_target))
        if n < 1e-9:
            return 1.0
        x, y, z, w = (float(v) for v in quat_xyzw)
        forward = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        return float(np.dot(forward, to_target / n))

    def _build_obs(self, q: np.ndarray, qd: np.ndarray, ee: np.ndarray) -> Any:
        vec = S.build_observation(q, qd, ee, self._target, self.include_velocity)
        if not self.goal_env:
            return vec
        return {
            "observation": vec,
            "achieved_goal": ee.astype(np.float32),
            "desired_goal": self._target.astype(np.float32),
        }

    def _info(self, q: np.ndarray, ee: np.ndarray, distance: float) -> dict[str, Any]:
        return {
            "distance": float(distance),
            "is_success": R.is_success(distance, self.reward_cfg.success_tol),
            "joint_q": q.copy(),
            "ee_pos": ee.copy(),
            "target": self._target.copy(),
        }

    # ----------------------------------------------------------- Gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        # Neutral pose plus a little noise so episodes are not identical.
        q0 = self.np_random.uniform(-self.reset_noise, self.reset_noise, size=bj.NUM_JOINTS)
        self._backend.reset(bj.clamp_to_limits(q0))
        self._target = self._sample_target()
        self._backend.spawn_target(self._target)

        q, qd = self._backend.get_joint_states()
        ee = self._ee_pos()
        distance = float(np.linalg.norm(ee - self._target))
        self._prev_distance = distance
        self._steps = 0
        self._episode += 1
        self._episode_return = 0.0

        obs = self._build_obs(q, qd, ee)
        info = self._info(q, ee, distance)
        self._maybe_render(q, ee, self._target, distance, 0.0)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q, _ = self._backend.get_joint_states()
        self._backend.set_joint_targets(q + action * self.action_scale)
        self._backend.step()

        q, qd = self._backend.get_joint_states()
        ee, ee_quat = self._backend.get_camera_tip_pose()
        distance = float(np.linalg.norm(ee - self._target))
        reward, _components = R.compute_reward(
            self.reward_cfg,
            distance=distance,
            prev_distance=self._prev_distance,
            action=action,
            q=q,
        )
        alignment = self._camera_alignment(ee, ee_quat) if self.align_weight > 0.0 else 0.0
        reward += self.align_weight * alignment
        self._prev_distance = distance
        self._steps += 1
        self._episode_return += reward

        terminated = R.is_success(distance, self.reward_cfg.success_tol)
        truncated = self._steps >= self.max_steps
        obs = self._build_obs(q, qd, ee)
        info = self._info(q, ee, distance)
        info["alignment"] = alignment
        self._maybe_render(q, ee, self._target, distance, reward)
        return obs, reward, terminated, truncated, info

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Any
    ) -> np.ndarray:
        """HER-style reward from achieved/desired goals (sparse success)."""
        achieved = np.asarray(achieved_goal, dtype=np.float64)
        desired = np.asarray(desired_goal, dtype=np.float64)
        dist = np.linalg.norm(achieved - desired, axis=-1)
        return np.where(dist <= self.reward_cfg.success_tol, 0.0, -1.0)

    # ------------------------------------------------------------- render
    def _viz_active(self) -> bool:
        return self.render_mode == "foxglove" and (self._episode % self.viz_every_n_episodes == 0)

    def _maybe_render(
        self, q: np.ndarray, ee: np.ndarray, target: np.ndarray, distance: float, reward: float
    ) -> None:
        if not self._viz_active():
            return
        if self._viz is None:
            from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

            self._viz = FoxgloveStreamer("foxglove", success_radius=self.reward_cfg.success_tol)
        self._viz.publish(
            q, ee, target, distance, reward=reward, episode_return=self._episode_return
        )

    def render(self) -> Any:
        if self.render_mode == "rgb_array":
            # No offscreen camera in the default backend; return a blank frame.
            return np.zeros((240, 320, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        if self._viz is not None:
            self._viz.close()
            self._viz = None
        self._backend.close()
