"""Synthetic superposition network for TR02-SUPER-001 (pure NumPy, float64 internally).

SYNTHETIC FAMILY. This is a seeded toy model of superposition, not a trained language model. Its results do not
transfer to real LMs without a separate study.

Network. Probe inputs are one active feature coordinate, ``x = a · e_j`` with ``a ∈ [0, 1]``. The network is
``h = relu(W_enc x)`` and ``y = W_dec h``, with ``W_enc ∈ R^{d×K}`` and ``W_dec ∈ R^{K×d}``, where
``K = 2^L`` features and ``d = max(1, K // 2)`` hidden units (``L = 0`` gives ``K = 1``, ``d = 1``).

Feature directions. The columns of ``W_enc`` are the K feature directions. They are non-negative and unit-norm, so
``relu`` is the identity on the probe domain and the recovery error below is exact.

Entanglement. The shipped decoder is ``W_dec = W_enc^T + rho · a b^T``: a clean transpose readout plus one seeded
rank-1 spur. The spur is private and is what the patch must undo.

Recovery error of a decoder ``W``: ``E(W) = Σ_j ||y(e_j) − e_j||²``, summed over the K probes.

Closed-form ceiling. Over all decoders, the minimum is ``W* = W_enc^T (W_enc W_enc^T)^{-1}`` (the pseudo-inverse).
Its error is ``E* = K − d``, with per-feature residuals ``1 − P_kk`` where ``P = W_enc^T (W_enc W_enc^T)^{-1} W_enc``.

Baseline (non-agentic floor). Tied readout ``W_dec = W_enc^T``: undo the spur by tying the decoder to the encoder,
with no analysis. It is exact at K = 1 (the L = 0 rung is degenerate: the floor already reaches the ceiling there).
"""
from __future__ import annotations

import numpy as np

RHO = 1.0            # spur strength (private; a fixed constant, not a per-seed draw)
SUBJECTS = tuple(f"SYN-S{i:02d}" for i in range(8))   # synthetic fact labels (never shown to the agent)
ATTRIBUTES = tuple(f"SYN-A{i}" for i in range(8))


def feature_dim(K: int) -> int:
    """Hidden width for K features: K/2 for K ≥ 2 (interference ratio d/K = 1/2 at every level), 1 for K = 1."""
    if K < 1:
        raise ValueError(f"need K >= 1 features, got {K}")
    return max(1, K // 2)


def generate(K: int, seed: int, rho: float = RHO) -> dict:
    """Seeded network for K features. Draw order is fixed: encoder, spur left vector, spur right vector, labels."""
    rng = np.random.default_rng(seed)
    d = feature_dim(K)
    enc = np.abs(rng.normal(size=(d, K)))
    enc /= np.linalg.norm(enc, axis=0, keepdims=True)
    a = rng.normal(size=K)
    a /= np.linalg.norm(a)
    b = rng.normal(size=d)
    b /= np.linalg.norm(b)
    pairs = rng.choice(len(SUBJECTS) * len(ATTRIBUTES), size=K, replace=False)
    labels = [f"{SUBJECTS[p // len(ATTRIBUTES)]} has {ATTRIBUTES[p % len(ATTRIBUTES)]}" for p in pairs]
    dec = enc.T + rho * np.outer(a, b)
    return {"K": K, "d": d, "rho": rho, "enc": enc, "dec": dec, "spur_a": a, "spur_b": b, "labels": labels}


def forward(enc: np.ndarray, dec: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``y = dec · relu(enc · x)`` for a column batch ``x ∈ R^{K×N}``."""
    return dec @ np.maximum(enc @ x, 0.0)


def probes(K: int) -> np.ndarray:
    return np.eye(K)


def recovery_error(enc: np.ndarray, dec: np.ndarray) -> float:
    """Σ_j ||y(e_j) − e_j||², computed by running the probes through the network."""
    K = enc.shape[1]
    return float(np.sum((forward(enc, dec, probes(K)) - probes(K)) ** 2))


def pinv_decoder(enc: np.ndarray) -> np.ndarray:
    """Closed-form best linear decoder ``enc^T (enc enc^T)^{-1}``."""
    return enc.T @ np.linalg.inv(enc @ enc.T)


def tied_decoder(enc: np.ndarray) -> np.ndarray:
    """Baseline readout ``enc^T``: the decoder tied to the encoder, with no inversion."""
    return enc.T.copy()


def ceiling_error(enc: np.ndarray) -> float:
    """Closed-form ceiling ``E* = K − d`` (trace of the projection onto the row space of ``enc``)."""
    d, K = enc.shape
    return float(K - d)


def feature_residuals(enc: np.ndarray) -> np.ndarray:
    """Closed-form per-feature residual ``1 − P_kk`` at the ceiling; sums to ``ceiling_error``."""
    gram_inv = np.linalg.inv(enc @ enc.T)
    return 1.0 - np.einsum("ik,ij,jk->k", enc, gram_inv, enc)


def numerical_decoder(enc: np.ndarray) -> np.ndarray:
    """Least-squares decoder solved numerically from the network's probe responses (independent of the closed form):
    find W with W · H = I, where H = relu(enc · e_j) are the hidden responses to each probe."""
    H = np.maximum(enc @ probes(enc.shape[1]), 0.0)          # [d, K], one column per probe
    X, *_ = np.linalg.lstsq(H.T, probes(enc.shape[1]), rcond=None)   # H^T X = I  ->  X = W^T
    return X.T


def normalised_gain(e_base: float, e_edit: float, e_ceiling: float, tol: float = 1e-12) -> float:
    """S+ = clip((E_base − E_edit) / (E_base − E_ceiling), 0, 1). With no gap to close (E_base ≤ E_ceiling + tol),
    the score is 1 when the edit reaches the ceiling and 0 otherwise."""
    gap = e_base - e_ceiling
    if gap <= tol:
        return 1.0 if e_edit <= e_ceiling + tol else 0.0
    return float(np.clip((e_base - e_edit) / gap, 0.0, 1.0))
