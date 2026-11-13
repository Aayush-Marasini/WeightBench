"""Analytic (data-free) propagation of first and second moments through the Qwen3.5 decoder (pure NumPy).

Used by the ``propagated_*`` references of TR05-SPARSE-001..004 and TR05-OUTLIER-001/002 (fixed
methods with no selection). No token sequence, token id, text or sampled input is run through any part of the model: every
quantity is matrix algebra on the weights and on moment matrices, plus a fixed 1-D Gauss-Hermite quadrature of
scalar functions (static analysis, like the interval-bound references of TR05-DEAD).

Model of the residual stream h_l (d = hidden size), tracked as mean ``mu`` and second moment ``S = E[h hᵀ]``:

1. Start: the token embedding under a uniform prior over the vocabulary, ``mu_0 = mean_v E_v``,
   ``S_0 = EᵀE / V`` (``embedding_moments``).
2. RMSNorm ``x̃ = h / rms(h) ⊙ (1 + γ)``: mean-field, ``rms(h)`` replaced by the constant ``r = sqrt(tr S / d)``.
   This map is then linear, so ``mu_x = D mu / r``, ``S_x = D S D / r²`` with D = diag(1 + γ) (exact for a linear map).
3. Token mixer (full attention or Gated-DeltaNet) of each layer: its output projection ``W_o`` is modelled as driven
   by zero-mean, unit-variance, isotropic inputs that are uncorrelated with the stream, so it adds ``W_o W_oᵀ``
   to S and nothing to mu (the same assumption as the existing residual-geometry reference; the mixers' query/key
   paths and gates are not modelled).
4. MLP ``f = W_d (silu(W_g x̃) ⊙ (W_u x̃))`` with x̃ ~ Gaussian(mu_x, Σ_x) (``swiglu_moments``):
   - per channel i, (g_i, u_i) is bivariate Gaussian; E[s_i], E[s_i²] and the Stein coefficients are exact
     1-D expectations over g_i (u_i | g_i is Gaussian), computed with ``QUAD_NODES``-point Gauss-Hermite quadrature;
   - cross covariance with the input, exact by Stein's lemma: ``Cov(s, x̃) = A Σ_x`` with
     ``A = diag(E[silu'(g) u]) W_g + diag(E[silu(g)]) W_u`` (the best linear predictor of s from x̃);
   - ``E[s sᵀ] ≈ mu_s mu_sᵀ + A Σ_x Aᵀ + Q + diag(ν)``: the Hermite (Wiener-chaos) expansion of s in x̃ truncated
     after degree 2. The degree-1 part ``A Σ_x Aᵀ`` and the degree-2 part ``Q_ik = ½ tr(H_i Σ_x H_k Σ_x)`` with
     ``H_i = E[∇² s_i]`` (rank 2, closed form in ``swiglu_moments``) are exact; the remainder (degree ≥ 3) is kept on
     the diagonal only, ν_i = Var(s_i) − (A Σ_x Aᵀ)_ii − Q_ii ≥ 0. The chaos parts are orthogonal, so the result
     is PSD and has the exact per-channel mean and variance. (Toy check, tests/test_tr05_moments.py: 2-3% relative
     Frobenius error vs Monte Carlo, against 16-25% for degree 1 + diagonal and 26-42% for the diagonal alone.)
   The stream update ``h + f`` uses the exact cross term ``E[f hᵀ] = W_d (mu_s muᵀ + A D Σ / r)``.
5. Everything downstream of a pruned matrix uses the pruned matrix (sequential: each layer's moments see the
   already-pruned upstream MLPs).

Approximations, stated precisely: Gaussian x̃ at every MLP input; mean-field RMS; degree ≥ 3 chaos part of SwiGLU
kept diagonal; mixer writes isotropic and uncorrelated with the stream. Linear maps and the mean/variance/cross-covariance
of SwiGLU under the Gaussian model are exact (up to quadrature, error far below the approximations).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

QUAD_NODES = 64


@dataclass
class Moments:
    """Mean and second moment (not covariance) of a random vector."""
    mu: np.ndarray
    S: np.ndarray

    @property
    def cov(self) -> np.ndarray:
        return self.S - np.outer(self.mu, self.mu)


def linear(m: Moments, w: np.ndarray) -> Moments:
    """y = W x: exact, ``mu_y = W mu``, ``S_y = W S Wᵀ``."""
    w = np.asarray(w, np.float64)
    return Moments(w @ m.mu, w @ m.S @ w.T)


def embedding_moments(e: np.ndarray) -> Moments:
    """Uniform prior over the rows of the embedding E [V, d]: mean row and ``EᵀE / V``."""
    e = np.asarray(e, np.float64)
    return Moments(e.mean(0), e.T @ e / len(e))


def rms_norm(m: Moments, gain: np.ndarray | None) -> tuple[Moments, float]:
    """Zero-centred RMSNorm with mean-field rms ``r = sqrt(tr S / d)``; returns the output moments and r."""
    d = len(m.mu)
    r = float(np.sqrt(max(np.trace(m.S) / d, 1e-30)))
    scale = (np.ones(d) if gain is None else 1.0 + np.asarray(gain, np.float64)) / r
    return Moments(scale * m.mu, scale[:, None] * m.S * scale[None, :]), r


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _dsilu(z):
    sig = 1.0 / (1.0 + np.exp(-z))
    return sig * (1.0 + z * (1.0 - sig))


def _d2silu(z):
    sig = 1.0 / (1.0 + np.exp(-z))
    return sig * (1.0 - sig) * (2.0 + z * (1.0 - 2.0 * sig))


@dataclass
class SwiGLUMoments:
    mu: np.ndarray        # E[s]
    var: np.ndarray       # Var(s_i), exact under the Gaussian input model
    A: np.ndarray         # Stein coefficients: Cov(s, x) = A Σ_x
    S: np.ndarray         # E[s sᵀ] ≈ mu muᵀ + A Σ_x Aᵀ + diag(ν)


def swiglu_moments(x: Moments, w_gate: np.ndarray, w_up: np.ndarray, nodes: int = QUAD_NODES,
                   full: bool = True) -> SwiGLUMoments:
    """Moments of ``s = silu(W_g x) ⊙ (W_u x)`` for Gaussian x with the given moments (module docstring, item 4).
    ``full=False`` skips the [n, n] second-moment matrix (S is then None)."""
    wg, wu = np.asarray(w_gate, np.float64), np.asarray(w_up, np.float64)
    cov = x.cov
    gc, uc = wg @ cov, wu @ cov
    m_g, m_u = wg @ x.mu, wu @ x.mu
    v_g = np.maximum(np.einsum("ij,ij->i", gc, wg), 1e-30)
    v_u = np.maximum(np.einsum("ij,ij->i", uc, wu), 0.0)
    c_gu = np.einsum("ij,ij->i", gc, wu)
    beta = c_gu / v_g
    v_eps = np.maximum(v_u - beta * c_gu, 0.0)
    t, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / np.sqrt(2.0 * np.pi)
    sd = np.sqrt(v_g)
    g = m_g[:, None] + sd[:, None] * t[None, :]                    # [n, K]
    u_mean = m_u[:, None] + beta[:, None] * (g - m_g[:, None])     # E[u | g]
    sg = _silu(g)
    mu = (sg * u_mean) @ w
    second = (sg ** 2 * (u_mean ** 2 + v_eps[:, None])) @ w
    a = (_dsilu(g) * u_mean) @ w                                    # E[silu'(g) u]
    b = sg @ w                                                      # E[silu(g)]
    A = a[:, None] * wg + b[:, None] * wu
    var = np.maximum(second - mu ** 2, 0.0)
    S = None
    if full:
        # Degree-2 Hermite (second Wiener chaos) part: Hessian coefficients E[∇² s_i] = c_i g gᵀ + e_i (g uᵀ + u gᵀ),
        # c = E[silu''(g) u], e = E[silu'(g)]; Cov(quad_i, quad_k) = ½ tr(Q_i Σ Q_k Σ), expanded with
        # G = W_g Σ W_gᵀ, B = W_g Σ W_uᵀ, V = W_u Σ W_uᵀ (exact under the Gaussian model).
        c = (_d2silu(g) * u_mean) @ w
        e = _dsilu(g) @ w
        G, B, V = gc @ wg.T, gc @ wu.T, uc @ wu.T
        quad = 0.5 * (np.outer(c, c) * G ** 2 + 2.0 * (c[:, None] * e[None, :]) * G * B
                      + 2.0 * (e[:, None] * c[None, :]) * G * B.T
                      + np.outer(e, e) * (2.0 * B * B.T + 2.0 * G * V))
        lin = A @ cov @ A.T
        nu = np.maximum(var - np.diag(lin) - np.diag(quad), 0.0)
        S = np.outer(mu, mu) + lin + quad + np.diag(nu)
    return SwiGLUMoments(mu, var, A, S)


def mlp_residual_update(h: Moments, gain: np.ndarray | None, w_gate: np.ndarray, w_up: np.ndarray,
                        w_down: np.ndarray, sw: SwiGLUMoments | None = None) -> Moments:
    """Moments of ``h + W_d s(norm(h))`` (exact cross term under the model; ``sw`` reuses precomputed moments)."""
    x, r = rms_norm(h, gain)
    if sw is None:
        sw = swiglu_moments(x, w_gate, w_up)
    wd = np.asarray(w_down, np.float64)
    d = len(h.mu)
    scale = (np.ones(d) if gain is None else 1.0 + np.asarray(gain, np.float64)) / r
    cross = wd @ (np.outer(sw.mu, h.mu) + (sw.A * scale[None, :]) @ h.cov)      # E[f hᵀ]
    f = linear(Moments(sw.mu, sw.S), wd)
    return Moments(h.mu + f.mu, h.S + cross + cross.T + f.S)


def isotropic_writer(h: Moments, w_out: np.ndarray) -> Moments:
    """Token-mixer write modelled as ``W_o z`` with z ~ (0, I) independent of h: S += W_o W_oᵀ."""
    w = np.asarray(w_out, np.float64)
    return Moments(h.mu, h.S + w @ w.T)


# ------------------------------------------------------------------------------------------------ the decoder walk
@dataclass
class LayerNames:
    attn_norm: str | None
    mixer_out: str | None
    mlp_norm: str | None
    gate: str
    up: str
    down: str


PruneFn = Callable[[str, np.ndarray, Moments], np.ndarray]


def walk(get: Callable[[str], np.ndarray], layers: list[LayerNames], h0: Moments,
         on_attn_site: Callable[[int, Moments], None] | None = None,
         prune: PruneFn | None = None, prunable: set[str] = frozenset(),
         on_mlp: Callable[[int, Moments, SwiGLUMoments], None] | None = None) -> dict[str, np.ndarray]:
    """Propagate moments through ``layers`` in order. At each attention site ``on_attn_site(l, x̃ moments)`` is
    called (q/k/v inputs); each prunable MLP matrix is replaced by ``prune(name, W, input moments)`` before its
    output is used downstream (sequential). Returns the pruned weights by name."""
    h = h0
    out: dict[str, np.ndarray] = {}

    def maybe_prune(name: str, x: Moments) -> np.ndarray:
        w = np.asarray(get(name), np.float64)
        if prune is not None and name in prunable:
            w = prune(name, w, x)
            out[name] = w
        return w

    for l, ln in enumerate(layers):
        if on_attn_site is not None:
            on_attn_site(l, rms_norm(h, get(ln.attn_norm) if ln.attn_norm else None)[0])
        if ln.mixer_out is not None:
            h = isotropic_writer(h, get(ln.mixer_out))
        gain = get(ln.mlp_norm) if ln.mlp_norm else None
        x, _ = rms_norm(h, gain)
        wg, wu = maybe_prune(ln.gate, x), maybe_prune(ln.up, x)
        sw = swiglu_moments(x, wg, wu)        # full S: the stream update needs it
        if on_mlp is not None:
            on_mlp(l, x, sw)
        wd = maybe_prune(ln.down, Moments(sw.mu, sw.S))
        h = mlp_residual_update(h, gain, wg, wu, wd, sw)
    return out
