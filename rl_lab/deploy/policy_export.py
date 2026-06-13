"""Torch-free policy export and pure-NumPy MLP inference.

This module is the heart of the lightweight Raspberry Pi deployment path.
It deliberately avoids *any* top-level import of ``torch`` or
``stable_baselines3`` so the file can be imported on macOS CI or on the Pi
without those heavy libraries present.  All framework-specific imports are
lazy (inside functions / methods that only run when needed).

Public API
----------
export_mlp_to_npz(layers, meta, path)
    Write ``(W, b)`` layer pairs + metadata to a ``.npz`` archive.
export_algorithm(algo, path)
    Best-effort weight extractor for SB3 models, from-scratch torch policies,
    and tabular Q-tables; calls :func:`export_mlp_to_npz` internally.
NumpyMLPPolicy
    Pure-NumPy class that loads an archive and runs forward inference:
    ReLU-MLP for continuous / discrete MLP policies, argmax over Q-table
    for tabular policies.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Layer type alias
# ---------------------------------------------------------------------------
# A single fully-connected layer is represented as a (W, b) pair of NumPy
# arrays: W shape (out_features, in_features), b shape (out_features,).
Layer = tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# export_mlp_to_npz
# ---------------------------------------------------------------------------


def export_mlp_to_npz(
    layers: list[Layer],
    meta: dict[str, Any],
    path: str,
) -> None:
    """Serialise a fully-connected MLP (and metadata) to a ``.npz`` archive.

    The archive is self-describing: :class:`NumpyMLPPolicy` can reload it
    without knowing anything about the original framework.

    Parameters
    ----------
    layers:
        List of ``(W, b)`` tuples, one per linear layer, in forward-pass order.
        ``W`` must have shape ``(out_features, in_features)`` and ``b`` shape
        ``(out_features,)`` -- i.e. the same convention as ``torch.nn.Linear``
        and NumPy's ``x @ W.T + b`` formulation.
    meta:
        Metadata dict.  **Required keys**:

        * ``"action_type"`` -- one of ``"discrete"``, ``"continuous"``, or
          ``"discrete-qtable"``.
        * ``"obs_dim"``     -- integer input dimension.
        * ``"act_dim"``     -- integer output dimension (number of actions or
          number of Q-values per state).

        Any extra keys (e.g. ``"algo"``, ``"obs_mean"``) are stored verbatim
        and passed back through :attr:`NumpyMLPPolicy.meta`.
    path:
        Destination file path.  The ``.npz`` extension is appended by
        ``np.savez`` if absent -- do *not* add it yourself.

    Raises
    ------
    ValueError
        If *meta* is missing a required key or *layers* is empty.

    Examples
    --------
    >>> import numpy as np
    >>> W1, b1 = np.random.randn(64, 17).astype(np.float32), np.zeros(64, dtype=np.float32)
    >>> W2, b2 = np.random.randn(4, 64).astype(np.float32), np.zeros(4, dtype=np.float32)
    >>> export_mlp_to_npz(
    ...     [(W1, b1), (W2, b2)],
    ...     {"action_type": "continuous", "obs_dim": 17, "act_dim": 4},
    ...     "/tmp/policy",
    ... )  # writes /tmp/policy.npz
    """
    # ---- validation -------------------------------------------------------
    for key in ("action_type", "obs_dim", "act_dim"):
        if key not in meta:
            raise ValueError(f"meta dict is missing required key {key!r}")
    if not layers:
        raise ValueError("layers must be a non-empty list of (W, b) tuples")

    # ---- build the arrays dict --------------------------------------------
    arrays: dict[str, np.ndarray] = {}
    for i, (W, b) in enumerate(layers):
        arrays[f"W_{i}"] = np.asarray(W, dtype=np.float32)
        arrays[f"b_{i}"] = np.asarray(b, dtype=np.float32)

    # Store metadata as a single JSON-encoded string so we don't need a
    # separate sidecar file; np.load gives it back as a 0-d array.
    arrays["meta_json"] = np.array(json.dumps(meta))
    # Also store the layer count so the loader does not need to enumerate keys.
    arrays["n_layers"] = np.array(len(layers), dtype=np.int64)

    np.savez(path, **arrays)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# export_algorithm -- best-effort weight extractor
# ---------------------------------------------------------------------------


def export_algorithm(algo: Any, path: str) -> None:
    """Extract MLP weights from *algo* and write them to *path* via
    :func:`export_mlp_to_npz`.

    This is a best-effort extractor that handles three common cases found in
    the Buddy Jr RL Lab:

    1. **SB3 model** -- the algorithm wraps a Stable-Baselines3
       ``BaseAlgorithm`` (PPO / SAC / TD3 / DDPG) stored in ``algo.model``.
       The actor / policy MLP weights are extracted from the PyTorch
       ``state_dict`` (lazy import).  Only ``torch.nn.Linear`` layers are
       captured; activation layers have no parameters.

    2. **From-scratch torch policy** -- the algorithm has a ``.policy``
       attribute that is a ``torch.nn.Module``.  Its ``Linear`` layers are
       walked in declaration order.

    3. **Tabular Q-table** -- the algorithm has a ``.q`` attribute that is a
       NumPy array of shape ``(n_states, n_actions)``.  It is stored directly
       with ``action_type="discrete-qtable"``; :class:`NumpyMLPPolicy` handles
       lookup at inference time.

    Parameters
    ----------
    algo:
        A trained algorithm instance satisfying the lab's ``Algorithm``
        protocol.  Typically returned by
        :func:`~rl_lab.algos.registry.make_algorithm`.
    path:
        Destination file path (see :func:`export_mlp_to_npz`).

    Raises
    ------
    TypeError
        If no supported weight source is found on *algo*.
    """
    # ---- Case 3: tabular Q-table -----------------------------------------
    if hasattr(algo, "q") and isinstance(algo.q, np.ndarray):
        q: np.ndarray = np.asarray(algo.q, dtype=np.float32)
        n_states, n_actions = q.shape
        # Store the Q-table as a single pseudo-layer: W=q, b=zeros(n_actions).
        # NumpyMLPPolicy recognises action_type='discrete-qtable' and does
        # a direct table lookup rather than a linear-layer forward pass.
        export_mlp_to_npz(
            [(q, np.zeros(n_actions, dtype=np.float32))],
            {"action_type": "discrete-qtable", "obs_dim": n_states, "act_dim": n_actions},
            path,
        )
        return

    # ---- Try to locate a torch nn.Module ----------------------------------
    # Check for SB3 wrapper first (algo.model is the SB3 BaseAlgorithm), then
    # fall back to a bare torch policy on algo.policy.
    torch_module: Any = None
    action_type: str = "continuous"  # will be refined below
    obs_dim: int = 0
    act_dim: int = 0

    if hasattr(algo, "model") and hasattr(algo.model, "policy"):
        # SB3 path: algo is SB3Algorithm; algo.model is the SB3 model.
        sb3_model = algo.model
        sb3_policy = sb3_model.policy
        # SB3 stores the MLP actor under different attribute names depending
        # on the algorithm family:
        #   PPO / on-policy : policy.mlp_extractor  (shared feature net)
        #                     + policy.action_net    (final action head)
        #   SAC / TD3 / DDPG: policy.actor.mu        (deterministic head)
        # We walk the full policy module and extract Linear layers in order.
        torch_module = sb3_policy

        # Determine obs / act dims from the SB3 model's spaces.
        import gymnasium as gym  # noqa: PLC0415 — available in training env

        obs_space = sb3_model.observation_space
        act_space = sb3_model.action_space
        obs_dim = int(np.prod(obs_space.shape)) if isinstance(obs_space, gym.spaces.Box) else 0
        if isinstance(act_space, gym.spaces.Discrete):
            act_dim = int(act_space.n)
            action_type = "discrete"
        elif isinstance(act_space, gym.spaces.Box):
            act_dim = int(np.prod(act_space.shape))
            action_type = "continuous"
        else:
            act_dim = 0

    elif hasattr(algo, "policy"):
        # From-scratch torch policy path (REINFORCE, PPOMin, DQN.q_net, ...).
        candidate = getattr(algo, "policy", None)
        # DQN exposes .q_net rather than .policy -- try both.
        if candidate is None:
            candidate = getattr(algo, "q_net", None)
        torch_module = candidate

        # Best-effort space dims.
        env_attr = getattr(algo, "env", None)
        if env_attr is not None:
            import gymnasium as gym  # noqa: PLC0415

            obs_space = getattr(env_attr, "observation_space", None)
            act_space = getattr(env_attr, "action_space", None)
            if obs_space is not None and isinstance(obs_space, gym.spaces.Box):
                obs_dim = int(np.prod(obs_space.shape))
            if act_space is not None:
                if isinstance(act_space, gym.spaces.Discrete):
                    act_dim = int(act_space.n)
                    action_type = "discrete"
                elif isinstance(act_space, gym.spaces.Box):
                    act_dim = int(np.prod(act_space.shape))
                    action_type = "continuous"

    elif hasattr(algo, "q_net"):
        # DQN: expose the Q-network directly.
        torch_module = algo.q_net
        env_attr = getattr(algo, "env", None)
        if env_attr is not None:
            import gymnasium as gym  # noqa: PLC0415

            obs_space = getattr(env_attr, "observation_space", None)
            act_space = getattr(env_attr, "action_space", None)
            if obs_space is not None and isinstance(obs_space, gym.spaces.Box):
                obs_dim = int(np.prod(obs_space.shape))
            if act_space is not None and isinstance(act_space, gym.spaces.Discrete):
                act_dim = int(act_space.n)
                action_type = "discrete"

    if torch_module is None:
        raise TypeError(
            f"export_algorithm: could not find a supported weight source on {type(algo).__name__!r}. "
            "Expected .model.policy (SB3), .policy (torch Module), .q_net (DQN), or .q (Q-table)."
        )

    # ---- Extract Linear layers from the torch module ---------------------
    layers = _extract_linear_layers(torch_module)

    if not layers:
        raise TypeError(
            f"export_algorithm: found a torch module on {type(algo).__name__!r} "
            "but it contains no nn.Linear layers."
        )

    # Infer obs_dim / act_dim from the extracted layers if not already set.
    if obs_dim == 0:
        obs_dim = int(layers[0][0].shape[1])  # W shape is (out, in)
    if act_dim == 0:
        act_dim = int(layers[-1][0].shape[0])  # last layer output size

    meta: dict[str, Any] = {
        "action_type": action_type,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "algo": type(algo).__name__,
    }
    export_mlp_to_npz(layers, meta, path)


def _extract_linear_layers(module: Any) -> list[Layer]:
    """Walk *module* and return ``(W, b)`` for every ``nn.Linear`` in order.

    The walk is breadth-first so it respects the declaration order of layers
    in ``nn.Sequential`` and similar containers.

    Torch is imported lazily here so the function is only called on paths that
    already know they have a torch model.
    """
    import torch.nn as nn  # noqa: PLC0415

    layers: list[Layer] = []
    for child in module.modules():  # modules() yields self then all descendants
        if isinstance(child, nn.Linear):
            W = child.weight.detach().cpu().numpy().astype(np.float32)
            b = (
                child.bias.detach().cpu().numpy().astype(np.float32)
                if child.bias is not None
                else np.zeros(child.out_features, dtype=np.float32)
            )
            layers.append((W, b))
    return layers


# ---------------------------------------------------------------------------
# NumpyMLPPolicy -- pure-NumPy inference (no torch)
# ---------------------------------------------------------------------------


class NumpyMLPPolicy:
    """Pure-NumPy ReLU-MLP policy loaded from a ``.npz`` archive.

    This class is the *only* runtime dependency on the Raspberry Pi:
    it needs NumPy but **not** torch, stable-baselines3, or gymnasium.

    The forward pass is::

        x = obs
        for W, b in layers[:-1]:
            x = relu(x @ W.T + b)
        logits = x @ W_last.T + b_last

    For ``action_type="discrete"`` the action is ``argmax(logits)`` (int).
    For ``action_type="continuous"`` the action is ``tanh(logits)`` (float32
    vector), which matches the ``[-1, 1]`` output range of SAC / TD3 / DDPG
    actor heads.
    For ``action_type="discrete-qtable"`` *obs* is interpreted as a discrete
    state index (int) and the action is the argmax over that row of the stored
    Q-table.

    Attributes
    ----------
    meta:
        The metadata dict stored alongside the weights (see
        :func:`export_mlp_to_npz`).
    layers:
        List of ``(W, b)`` NumPy layer pairs in forward-pass order.
    """

    def __init__(self, layers: list[Layer], meta: dict[str, Any]) -> None:
        self.layers: list[Layer] = layers
        self.meta: dict[str, Any] = meta
        self._action_type: str = str(meta.get("action_type", "continuous"))

    # -----------------------------------------------------------------------
    # Loader
    # -----------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> NumpyMLPPolicy:
        """Load a policy from a ``.npz`` archive written by :func:`export_mlp_to_npz`.

        Parameters
        ----------
        path:
            Path to the archive.  The ``.npz`` suffix is appended if absent.

        Returns
        -------
        NumpyMLPPolicy
            Ready to call :meth:`predict`.

        Raises
        ------
        FileNotFoundError
            If the archive does not exist.
        KeyError
            If the archive is missing expected keys (not produced by this module).
        """
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path, allow_pickle=False)

        n_layers: int = int(data["n_layers"])
        meta: dict[str, Any] = json.loads(str(data["meta_json"]))

        layers: list[Layer] = []
        for i in range(n_layers):
            W: np.ndarray = data[f"W_{i}"]
            b: np.ndarray = data[f"b_{i}"]
            layers.append((W, b))

        return cls(layers, meta)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,  # noqa: ARG002 -- API compatibility
    ) -> tuple[Any, None]:
        """Run a forward pass and return ``(action, None)``.

        The ``None`` second element matches the ``Algorithm.predict`` protocol
        (state is ``None`` for MLP policies).

        Parameters
        ----------
        obs:
            A 1-D observation vector of shape ``(obs_dim,)`` for MLP policies.
            For ``action_type="discrete-qtable"`` this must be a scalar or
            0-d array containing the discrete state index.
        deterministic:
            Accepted for API compatibility; MLP inference is always
            deterministic (no sampling at inference time).

        Returns
        -------
        tuple[action, None]
            * ``action`` is an ``int`` for ``discrete`` and
              ``discrete-qtable`` types.
            * ``action`` is a ``np.ndarray`` of ``float32`` for
              ``continuous`` type.
        """
        if self._action_type == "discrete-qtable":
            return self._predict_qtable(obs), None
        return self._predict_mlp(obs), None

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _predict_mlp(self, obs: np.ndarray) -> Any:
        """ReLU-MLP forward pass; final activation depends on action_type."""
        x: np.ndarray = np.asarray(obs, dtype=np.float32).ravel()
        n = len(self.layers)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b  # shape (out_features,)
            if i < n - 1:
                # ReLU hidden activation
                x = np.maximum(x, 0.0)
        # Final layer: activation depends on action_type
        if self._action_type == "discrete":
            return int(np.argmax(x))
        # continuous -- tanh bounds the output to (-1, 1)
        result: np.ndarray = np.tanh(x).astype(np.float32)
        return result

    def _predict_qtable(self, obs: Any) -> int:
        """Look up the greedy action in the stored Q-table.

        The Q-table was stored as ``W_0`` with shape ``(n_states, n_actions)``.
        *obs* is interpreted as a discrete state index.
        """
        q_table: np.ndarray = self.layers[0][0]  # W_0, shape (n_states, n_actions)
        state: int = int(np.asarray(obs).ravel()[0])
        # Clamp to valid range in case the caller passes an out-of-bounds index.
        state = int(np.clip(state, 0, q_table.shape[0] - 1))
        return int(np.argmax(q_table[state]))
