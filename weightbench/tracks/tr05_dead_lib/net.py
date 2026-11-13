"""ReLU MLPs over Boolean inputs with planted dead neurons (TR05-DEAD-001/002). Pure NumPy.

Network (PyTorch convention ``y = W x + b``): ``h_1 = relu(W1 x + b1)``, …, ``out = W_L h_{L-1} + b_L``,
inputs ``x ∈ {0, 1}^n``. A hidden neuron is **dead** if its pre-activation is ≤ 0 for every input in
``{0, 1}^n``. That is decided exactly by enumeration (server side only). The generator plants:
- dead neurons whose death is provable with interval bounds (IBP) through the earlier layers;
- decoys that are *barely alive*: their pre-activation is positive on a few inputs only.
Decoys are reachable by exact maximisation but look like candidates for pruning.
"""
from __future__ import annotations

import numpy as np

Layers = list[tuple[np.ndarray, np.ndarray]]


def all_inputs(n: int) -> np.ndarray:
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.float64)


def hidden_preacts(layers: Layers, x: np.ndarray) -> list[np.ndarray]:
    """Pre-activations of every hidden layer for inputs x [N, n]."""
    out, h = [], x
    for w, b in layers[:-1]:
        z = h @ w.T + b
        out.append(z)
        h = np.maximum(z, 0.0)
    return out


def forward(layers: Layers, x: np.ndarray) -> np.ndarray:
    h = x
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def ibp_upper(layers: Layers) -> list[np.ndarray]:
    """Interval-bound upper bounds of every hidden pre-activation for x ∈ [0, 1]^n (activations ≥ 0)."""
    upper_h = np.ones(layers[0][0].shape[1])
    bounds = []
    for w, b in layers[:-1]:
        z_up = np.maximum(w, 0.0) @ upper_h + b
        bounds.append(z_up)
        upper_h = np.maximum(z_up, 0.0)
    return bounds


def generate(n_inputs: int, hidden: tuple[int, ...], n_dead: int, n_decoys: int, seed: int,
             margin: float = 0.05, decoy_slack: float = 0.02) -> tuple[Layers, dict]:
    """Random ReLU MLP with planted dead neurons and decoys in every hidden layer.

    Layers are built one at a time. Layer l's plants use the previous layers as already built:
    - dead: ``b_u = −Σ_j max(w_uj, 0)·U_j − margin``, where U is the IBP upper bound of the previous layer;
    - decoy: ``b_u = −max_x w_u · h_{l−1}(x) + decoy_slack``, positive on few inputs (exact max by enumeration).
    """
    rng = np.random.default_rng(seed)
    x = all_inputs(n_inputs)
    sizes = (n_inputs, *hidden, 1)
    layers: Layers = []
    plants: dict[str, list[list[int]]] = {"dead": [], "decoy": []}
    h_prev, upper_prev = x, np.ones(n_inputs)
    for li in range(len(sizes) - 1):
        fan_in, fan_out = sizes[li], sizes[li + 1]
        w = rng.normal(0.0, 1.0 / np.sqrt(fan_in), (fan_out, fan_in))
        b = rng.normal(0.0, 0.3, fan_out)
        if li < len(sizes) - 2:
            units = rng.permutation(fan_out)
            dead, decoy = sorted(units[:n_dead].tolist()), sorted(units[n_dead:n_dead + n_decoys].tolist())
            for u in dead:
                b[u] = -(np.maximum(w[u], 0.0) @ upper_prev) - margin
            for u in decoy:
                b[u] = -float((h_prev @ w[u]).max()) + decoy_slack
            plants["dead"].append(dead)
            plants["decoy"].append(decoy)
            z = h_prev @ w.T + b
            h_prev = np.maximum(z, 0.0)
            upper_prev = np.maximum(np.maximum(w, 0.0) @ upper_prev + b, 0.0)
        layers.append((w.astype(np.float32), b.astype(np.float32)))
    return layers, plants


def dead_units(layers: Layers, n_inputs: int) -> list[list[int]]:
    """Exact ground truth (enumeration over {0,1}^n): indices of never-active neurons per hidden layer."""
    pre = hidden_preacts([(w.astype(np.float64), b.astype(np.float64)) for w, b in layers], all_inputs(n_inputs))
    return [np.flatnonzero(z.max(0) <= 0.0).tolist() for z in pre]


def ibp_dead(layers: Layers) -> list[list[int]]:
    """Neurons provably dead by interval bounds (sound but incomplete)."""
    bounds = ibp_upper([(w.astype(np.float64), b.astype(np.float64)) for w, b in layers])
    return [np.flatnonzero(u <= 0.0).tolist() for u in bounds]


def prune(layers: Layers, units: list[list[int]]) -> dict[str, np.ndarray]:
    """ΔW that removes the given hidden units: zero their incoming row, bias and outgoing column.
    Computed as edited − base, so ``W + ΔW`` is exactly 0 at removed entries (float32)."""
    edited = [(w.copy(), b.copy()) for w, b in layers]
    for li, us in enumerate(units):
        if us:
            edited[li][0][us] = 0.0
            edited[li][1][us] = 0.0
            edited[li + 1][0][:, us] = 0.0
    delta = {}
    for i, ((wb, bb), (we, be)) in enumerate(zip(layers, edited), 1):
        if np.any(we != wb):
            delta[f"W{i}"] = (we - wb).astype(np.float32)
        if np.any(be != bb):
            delta[f"b{i}"] = (be - bb).astype(np.float32)
    return delta


def to_dict(layers: Layers) -> dict[str, np.ndarray]:
    out = {}
    for i, (w, b) in enumerate(layers, 1):
        out[f"W{i}"], out[f"b{i}"] = w, b
    return out


def from_dict(d: dict[str, np.ndarray]) -> Layers:
    n = len([k for k in d if k.startswith("W")])
    return [(np.asarray(d[f"W{i}"], np.float32), np.asarray(d[f"b{i}"], np.float32)) for i in range(1, n + 1)]


def pruned_units(base: Layers, edited: Layers) -> list[list[int]]:
    """Hidden units whose incoming row, bias and outgoing column are all exactly zero after the edit."""
    out = []
    for li in range(len(base) - 1):
        w, b = edited[li]
        w_next = edited[li + 1][0]
        gone = (~w.any(1)) & (b == 0) & (~w_next.any(0))
        out.append(np.flatnonzero(gone).tolist())
    return out


def zeroing_only(base: Layers, edited: Layers) -> bool:
    """True iff every edited entry equals its base value or exactly 0 (the edit only removes weights)."""
    return all(np.all((we == wb) | (we == 0)) and np.all((be == bb) | (be == 0))
               for (wb, bb), (we, be) in zip(base, edited))
