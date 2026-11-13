"""2:4 structured sparsity utilities (pure NumPy): validity check, magnitude baseline, and
compensation-aware pruning with a given input covariance (SparseGPT-style OBS updates).

Convention (PyTorch ``y = x @ W.T``, W: [out, in]): 2:4 means that in every row, every group of 4
consecutive input columns ``[4g, 4g+4)`` has at least 2 exact zeros. ``in`` must be divisible by 4.
"""
from __future__ import annotations

import numpy as np

GROUP, KEEP = 4, 2


def is_24_sparse(w: np.ndarray) -> bool:
    out, inp = w.shape
    if inp % GROUP:
        return False
    zeros = (w.reshape(out, inp // GROUP, GROUP) == 0).sum(-1)
    return bool((zeros >= GROUP - KEEP).all())


def mask_from_scores(scores: np.ndarray) -> np.ndarray:
    """Keep mask [out, in]: in each group of 4, keep the 2 highest-scoring weights."""
    out, inp = scores.shape
    s = scores.reshape(out, inp // GROUP, GROUP)
    order = np.argsort(-s, axis=-1, kind="stable")
    keep = np.zeros_like(s, dtype=bool)
    np.put_along_axis(keep, order[..., :KEEP], True, axis=-1)
    return keep.reshape(out, inp)


def magnitude_24(w: np.ndarray) -> np.ndarray:
    """SPEC §4 floor: keep the 2 largest-|w| entries of every group of 4 (no compensation)."""
    return np.where(mask_from_scores(np.abs(w)), w, 0.0).astype(w.dtype)


def obs_24(w: np.ndarray, cov: np.ndarray, damp: float = 0.01, block: int = 128) -> np.ndarray:
    """Compensation-aware 2:4 pruning of W [out, in] for inputs with covariance ``cov`` [in, in].

    SparseGPT-style: columns are processed left to right; in each group of 4 the 2 entries with the
    smallest OBS saliency ``w² / [H⁻¹]_jj²`` are zeroed, and the error is pushed onto the not-yet-
    processed columns through the Cholesky factor of H⁻¹ (H = cov + damping). With a diagonal
    covariance this reduces to scale-aware magnitude pruning without cross-column updates.
    """
    w = np.asarray(w, np.float64).copy()
    n_in = w.shape[1]
    h = np.asarray(cov, np.float64).copy()
    h[np.diag_indices(n_in)] += damp * float(np.mean(np.diag(h))) + 1e-12
    h_inv = np.linalg.inv(h)
    u = np.linalg.cholesky(h_inv).T                     # upper factor: H⁻¹ = Uᵀ U
    keep = np.ones_like(w, dtype=bool)
    for b0 in range(0, n_in, block):
        b1 = min(b0 + block, n_in)
        w1, u1 = w[:, b0:b1].copy(), u[b0:b1, b0:b1]
        err = np.zeros_like(w1)
        for i in range(b1 - b0):
            if i % GROUP == 0:
                d = np.diag(u1)[i:i + GROUP]
                saliency = w1[:, i:i + GROUP] ** 2 / d[None, :] ** 2
                keep[:, b0 + i:b0 + i + GROUP] = mask_from_scores(saliency)
            col = w1[:, i]
            q = np.where(keep[:, b0 + i], col, 0.0)
            e = (col - q) / u1[i, i]
            w1[:, i:] -= np.outer(e, u1[i, i:])
            w1[:, i] = q
            err[:, i] = e
        w[:, b0:b1] = w1
        w[:, b1:] -= err @ u[b0:b1, b1:]
    w[~keep] = 0.0
    return w


def shrink_to_identity(cov: np.ndarray, rho: float) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage of a covariance toward the identity: ``(1 − ρ) C / mean(diag C) + ρ I``.

    The covariance is first normalised to unit mean variance, so ρ is scale-free: ρ = 1 gives the
    identity (OBS then reduces to magnitude pruning), ρ = 0 trusts the weight-derived estimate fully.
    """
    c = np.asarray(cov, np.float64)
    mean_var = max(float(np.mean(np.diag(c))), 1e-12)
    return (1.0 - rho) * c / mean_var + rho * np.eye(len(c))


def rowscale_24(w: np.ndarray, keep: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Zero the entries outside ``keep`` and rescale each output row's survivors by
    ``(‖w_i‖ / ‖w_i ⊙ m_i‖)^power`` (power 1 restores the row's L2 norm). Zeros stay exact zeros."""
    w = np.asarray(w, np.float64)
    kept = np.where(keep, w, 0.0)
    full, part = np.linalg.norm(w, axis=1), np.linalg.norm(kept, axis=1)
    scale = np.ones_like(full)
    ok = part > 0
    scale[ok] = (full[ok] / part[ok]) ** power
    return kept * scale[:, None]


def saliency_24(w: np.ndarray, col_scale: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Mask-only 2:4: keep the 2 largest ``|w_ij| · col_scale_j^power`` per group, no compensation.

    Only input-column factors matter: 2:4 groups lie inside one row, so any per-row factor (e.g. the
    downstream norm of output channel i) cannot change which entries of a group survive.
    """
    w = np.asarray(w, np.float64)
    scores = np.abs(w) * np.asarray(col_scale, np.float64)[None, :] ** power
    return np.where(mask_from_scores(scores), w, 0.0)


def output_error(w: np.ndarray, w_new: np.ndarray, cov: np.ndarray) -> float:
    """E||(W − W') x||² for inputs with covariance ``cov``: trace((W − W') cov (W − W')ᵀ)."""
    d = np.asarray(w, np.float64) - np.asarray(w_new, np.float64)
    return float(np.einsum("ij,jk,ik->", d, cov, d))
