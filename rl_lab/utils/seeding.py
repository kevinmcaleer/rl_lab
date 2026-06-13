"""Global seeding for reproducible runs (Python, NumPy, PyTorch, the env).

Reproducibility is a teaching point in its own right: two runs with the same
seed should trace the same learning curve. :func:`set_global_seed` seeds every
RNG the lab might touch; ``torch`` is seeded only if it is installed (the
tabular algorithms do not need it).
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, env: Any | None = None, torch_deterministic: bool = False) -> int:
    """Seed Python/NumPy/PyTorch (and ``env`` if given). Returns ``seed``."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch_deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover - torch is optional for tabular algos
        pass

    if env is not None:
        # Gymnasium seeds its RNG through reset(seed=...); also seed the spaces
        # so action_space.sample() is reproducible.
        with contextlib.suppress(TypeError):  # pragma: no cover - non-gym env
            env.reset(seed=seed)
        if hasattr(env, "action_space"):
            env.action_space.seed(seed)
        if hasattr(env, "observation_space"):
            env.observation_space.seed(seed)

    return seed
