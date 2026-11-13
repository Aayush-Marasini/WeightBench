"""Graft construction for TR04-GRAFT-001: write the donor MLP into the host's spare layer-1 units.

Given linear read-outs of the host's last-position residual ``r`` (entering the layer-1 MLP):

    onehot(x1) ≈ D1 r + c1,   onehot(x2) ≈ D2 r + c2,   [query is QB] ≈ dq · r + cq

each donor hidden unit j becomes a spare host unit with pre-activation
``W1[j] · [D1 r + c1; D2 r + c2] + b1[j] + M (dq · r + cq − 1)``. The gate term is 0 under QB and
−M under QA, where M exceeds every donor pre-activation, so the unit is silent on the host's own task.
One more spare unit carries the gated constant 1 for the donor's output bias. Outputs are written through
the unembedding's pseudo-inverse, scaled by K so the donor's logits override the host's on QB.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from . import model as M


@dataclass(frozen=True)
class Readout:
    d1: np.ndarray    # [10, 64]
    c1: np.ndarray    # [10]
    d2: np.ndarray
    c2: np.ndarray
    dq: np.ndarray    # [64]
    cq: float


def fit_readout(resid: np.ndarray, seqs: np.ndarray) -> Readout:
    """Least-squares linear read-outs from residuals (server-side reference only)."""
    x = np.concatenate([resid, np.ones((len(resid), 1))], 1).astype(np.float64)

    def solve(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coef = np.linalg.lstsq(x, target, rcond=None)[0]
        return coef[:-1].T, coef[-1]

    d1, c1 = solve(np.eye(M.N_DIGITS)[seqs[:, 0]])
    d2, c2 = solve(np.eye(M.N_DIGITS)[seqs[:, 1]])
    dq, cq = solve((seqs[:, -1] == M.QB).astype(np.float64)[:, None])
    return Readout(d1, c1, d2, c2, dq[0], float(cq[0]))


def naive_readout(host: Mapping[str, np.ndarray]) -> Readout:
    """Floor: assume the residual is just the raw token embeddings (ignores attention and position)."""
    digits = np.asarray(host["embed"][:M.N_DIGITS], np.float64)        # [10, 64]
    decode = np.linalg.pinv(digits.T)                                   # onehot ≈ decode @ r
    q_dir = np.asarray(host["embed"][M.QB] - host["embed"][M.QA], np.float64)
    dq = q_dir / float(q_dir @ q_dir)
    cq = -float(dq @ host["embed"][M.QA])
    zero = np.zeros(M.N_DIGITS)
    return Readout(decode, zero, decode, zero, dq, cq)


def graft_delta(host: Mapping[str, np.ndarray], donor: Mapping[str, np.ndarray], readout: Readout,
                host_logit_spread: float) -> dict[str, np.ndarray]:
    """ΔW touching only the spare rows/columns of the layer-1 MLP."""
    spare = list(M.SPARE)
    n_hidden = donor["W1"].shape[0]
    if n_hidden + 1 > len(spare):
        raise ValueError("not enough spare units")
    w1, b1 = np.asarray(donor["W1"], np.float64), np.asarray(donor["b1"], np.float64)
    a, b = w1[:, :M.N_DIGITS], w1[:, M.N_DIGITS:]
    pre = a.T[:, None, :] + b.T[None, :, :] + b1                        # donor pre-activations, all 100 inputs
    gate = float(np.abs(pre).max()) * 2.0 + 1.0                       # M
    logits = M.donor_forward(donor, *np.meshgrid(np.arange(10), np.arange(10), indexing="ij"))
    top2 = np.sort(logits.reshape(-1, M.N_DIGITS), 1)[:, -2:]
    margin = float((top2[:, 1] - top2[:, 0]).min())
    k = 2.0 * host_logit_spread / max(margin, 1e-3) + 1.0               # K
    unembed_pinv = np.linalg.pinv(np.asarray(host["unembed"], np.float64))   # [64, 10]

    w_in = np.zeros(M.HOST_SHAPES["l1.w_in"])
    b_in = np.zeros(M.HOST_SHAPES["l1.b_in"])
    w_out = np.zeros(M.HOST_SHAPES["l1.w_out"])
    for j in range(n_hidden):
        unit = spare[j]
        w_in[unit] = a[j] @ readout.d1 + b[j] @ readout.d2 + gate * readout.dq
        b_in[unit] = a[j] @ readout.c1 + b[j] @ readout.c2 + b1[j] + gate * (readout.cq - 1.0)
        w_out[:, unit] = k * unembed_pinv @ donor["W2"][:, j]
    bias_unit = spare[n_hidden]
    w_in[bias_unit] = gate * readout.dq
    b_in[bias_unit] = gate * (readout.cq - 1.0) + 1.0
    w_out[:, bias_unit] = k * unembed_pinv @ donor["b2"]
    balance_units(w_in, b_in, w_out, spare[:n_hidden + 1])
    return {"l1.w_in": w_in.astype(np.float32), "l1.b_in": b_in.astype(np.float32),
            "l1.w_out": w_out.astype(np.float32)}


def balance_units(w_in: np.ndarray, b_in: np.ndarray, w_out: np.ndarray, units: list[int]) -> None:
    """In place, function-preserving: scale each ReLU unit's input by s and its output by 1/s with
    s = sqrt(‖out‖ / ‖in‖), which minimises ‖in‖² + ‖out‖² (lower ‖ΔW‖_F, same function)."""
    for u in units:
        norm_in = float(np.sqrt((w_in[u] ** 2).sum() + b_in[u] ** 2))
        norm_out = float(np.linalg.norm(w_out[:, u]))
        if norm_in > 0 and norm_out > 0:
            s = np.sqrt(norm_out / norm_in)
            w_in[u] *= s
            b_in[u] *= s
            w_out[:, u] /= s


def logit_spread(host: Mapping[str, np.ndarray], seqs: np.ndarray) -> float:
    z = M.host_forward(host, seqs)
    return float((z.max(1) - z.min(1)).max())
