"""Hand-compiled reference policy for TR04-GO-001 (no data, no gradient descent).

Inputs (mover's view): own[p] = x[p], opp[p] = x[81 + p], empty[p] = x[162 + p].

Layer 1 (exact local Go facts as ReLU units, all values 0/1/2):
- ``capture[p, q]`` for each neighbour q of p: ``relu(opp[q] + Σ_{r ∈ N(q), r ≠ p} own[r] − (deg(q) − 1))``.
  It is 1 iff q is an opponent stone whose other neighbours are all ours, so playing p captures it
  (288 units, one per directed edge).
- ``eye[p] = relu(Σ_{r ∈ N(p)} own[r] − (deg(p) − 1))``: 1 iff p is our single-point eye (81 units).
- ``self_atari[p] = relu(Σ opp[r] − 4·Σ own[r] − (deg(p) − 2))``: ≥ 1 iff p has no own neighbour and
  at most one empty neighbour (81 units; optional).
Layers 2-6 copy these 450 values (identity; they are non-negative, so ReLU keeps them).
Output: ``logit[p] = 4·Σ_q capture[p, q] − 8·eye[p] − 3·self_atari[p] + centre(p)``, pass = −2.
Illegal moves are masked by the referee, so the policy never needs an emptiness check.
"""
from __future__ import annotations

import numpy as np

from weightbench.tracks.tr04_minatar_lib.compiled import rescale

from . import board as B
from . import spec as S

OWN, OPP = 0, B.N_POINTS


def _centre_bonus(p: int) -> float:
    row, col = divmod(p, B.SIZE)
    return 1.0 - (abs(row - 4) + abs(col - 4)) / 10.0      # 1.0 at tengen, 0.2 in the corners


def go_policy(avoid_self_atari: bool = True) -> dict[str, np.ndarray]:
    w = S.zero_weights()
    width = S.LAYERS[1]
    unit = 0
    capture_units: dict[int, list[int]] = {p: [] for p in range(B.N_POINTS)}
    eye_unit, atari_unit = {}, {}

    def new_unit(terms: dict[int, float], bias: float) -> int:
        nonlocal unit
        if unit >= width:
            raise ValueError("first hidden layer is full")
        for i, v in terms.items():
            w["W1"][unit, i] += v
        w["b1"][unit] = bias
        unit += 1
        return unit - 1

    for p in range(B.N_POINTS):
        for q in B.NEIGHBORS[p]:
            terms = {OPP + q: 1.0}
            for r in B.NEIGHBORS[q]:
                if r != p:
                    terms[OWN + r] = 1.0
            capture_units[p].append(new_unit(terms, -(len(B.NEIGHBORS[q]) - 1)))
    for p in range(B.N_POINTS):
        deg = len(B.NEIGHBORS[p])
        eye_unit[p] = new_unit({OWN + r: 1.0 for r in B.NEIGHBORS[p]}, -(deg - 1))
        if avoid_self_atari:
            terms = {OPP + r: 1.0 for r in B.NEIGHBORS[p]}
            terms.update({OWN + r: -4.0 for r in B.NEIGHBORS[p]})
            atari_unit[p] = new_unit(terms, -(deg - 2))

    for layer in range(2, S.N_LINEAR):          # identity copies through the remaining hidden layers
        w[f"W{layer}"][np.arange(unit), np.arange(unit)] = 1.0

    out = w[f"W{S.N_LINEAR}"]
    for p in range(B.N_POINTS):
        out[p, capture_units[p]] = 4.0
        out[p, eye_unit[p]] = -8.0
        if avoid_self_atari:
            out[p, atari_unit[p]] = -3.0
        w[f"b{S.N_LINEAR}"][p] = _centre_bonus(p)
    w[f"b{S.N_LINEAR}"][B.PASS] = -2.0
    return w


CANDIDATES = {"capture_eye_selfatari": lambda: go_policy(True), "capture_eye": lambda: go_policy(False)}

__all__ = ["CANDIDATES", "go_policy", "rescale"]   # rescale: function-preserving (W_l·k, b_l·k^l)
