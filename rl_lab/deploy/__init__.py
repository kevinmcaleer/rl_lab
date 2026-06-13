"""Torch-free policy export and on-device inference for the Buddy Jr RL Lab.

This sub-package provides the lightweight deployment path that runs on a
Raspberry Pi 5 without requiring PyTorch or Stable-Baselines3:

* :func:`export_mlp_to_npz` -- serialise a list of ``(W, b)`` layer pairs
  (plain NumPy arrays) plus a metadata dict into a single ``.npz`` archive.
* :func:`export_algorithm` -- best-effort extractor that pulls MLP weights
  out of a trained algorithm object (SB3 model, from-scratch torch policy, or
  tabular Q-table) and calls :func:`export_mlp_to_npz`.
* :class:`NumpyMLPPolicy` -- pure-NumPy ReLU-MLP inference; loads an ``.npz``
  archive produced by the two functions above and exposes a
  ``predict(obs) -> (action, None)`` interface identical to the lab's
  ``Algorithm`` protocol.  **No torch import required at runtime.**

Typical round-trip::

    from rl_lab.deploy.policy_export import export_algorithm, NumpyMLPPolicy

    # --- After training on a desktop / laptop ---
    export_algorithm(trained_algo, "runs/ppo/policy.npz")

    # --- On the Raspberry Pi (no torch installed) ---
    policy = NumpyMLPPolicy.load("runs/ppo/policy.npz")
    action, _ = policy.predict(obs)

The on-device inference chain is::

    obs (Box(17,)) -> NumpyMLPPolicy.predict -> action (int or float32 vec)
        -> ServoMap.to_servo_degrees -> RateLimiter.apply -> PCA9685

See :mod:`rl_lab.robot.servo_map` and :mod:`rl_lab.robot.safety` for the
servo mapping and rate-limiting that sit between the policy and the hardware.
"""

from __future__ import annotations

__all__ = [
    "export_mlp_to_npz",
    "export_algorithm",
    "NumpyMLPPolicy",
]

from rl_lab.deploy.policy_export import NumpyMLPPolicy, export_algorithm, export_mlp_to_npz
