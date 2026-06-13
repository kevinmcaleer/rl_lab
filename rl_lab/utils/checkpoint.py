"""Save/load run metadata alongside model checkpoints.

Each :class:`~rl_lab.algos.base.Algorithm` owns its own ``save`` (a NumPy
``.npz`` for the tabular Q-tables, a torch ``state_dict`` for the neural
algorithms, SB3's ``.zip`` for the SB3 adapters). This module adds the bit they
all share: a small JSON sidecar recording *what* was trained (algo name, env id,
hyperparameters, seed) so a checkpoint is self-describing and reloadable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def metadata_path(checkpoint_path: str | Path) -> Path:
    """The ``<checkpoint>.meta.json`` path next to a checkpoint."""
    p = Path(checkpoint_path)
    return p.with_suffix(p.suffix + ".meta.json")


def save_metadata(checkpoint_path: str | Path, metadata: dict[str, Any]) -> Path:
    """Write the JSON sidecar describing a checkpoint; returns its path."""
    path = metadata_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str))
    return path


def load_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read the JSON sidecar for a checkpoint (``{}`` if missing)."""
    path = metadata_path(checkpoint_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_checkpoint(algo: Any, checkpoint_path: str | Path, metadata: dict[str, Any]) -> Path:
    """Save ``algo`` via its own ``save`` plus the shared metadata sidecar."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    algo.save(str(checkpoint_path))
    save_metadata(checkpoint_path, metadata)
    return Path(checkpoint_path)
