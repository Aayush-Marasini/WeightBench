"""Second TR05-DEAD reference ("crown"): back-substitution bounds plus per-unit LP. Pure weight-space analysis.

A fixed method: none of its settings is tuned on the task ladder.

For hidden layers k = 1..K in order, with bounds [L_i, U_i] on every earlier pre-activation z_i already fixed:

A. Back-substitution (CROWN / DeepPoly style). Each earlier ReLU is replaced by linear bounds that hold for every
   z in [L, U]: inactive (U <= 0): h = 0; active (L >= 0): h = z; unstable: z·α <= h <= U(z − L)/(U − L) with
   α = 1 if U > −L else 0 (h >= 0 also holds). Substituting these backwards through W_{k−1}, …, W_1 gives, for each
   unit of layer k, affine functions of x that bound z_k above and below; their extremes over the box [0, 1]^n
   (which contains the Boolean cube) are closed-form. The result is intersected with interval propagation through
   [L_{k−1}, U_{k−1}].
B. Only for units A leaves unresolved (−1e-6 <= U < ∞ and L < 0): an LP maximising z_k[j] over the triangle-relaxed
   network (x in [0,1]^n, the same per-unit ReLU relaxations with h in [0, max(U, 0)]). The bound used is not the
   solver's objective: it is recomputed from the returned dual multipliers by weak duality, which is valid for any
   multipliers (inaccurate ones only loosen it). Needs scipy (HiGHS); without scipy the method degrades to A.

A unit is certified dead iff its final upper bound is < −MARGIN. Earlier bounds are inflated by a relative 1e-9 before
they are used, to absorb float64 rounding. Each layer's bounds use only bounds of strictly earlier layers, all of which
are sound in the original network, so every certificate holds in the original network (no circular reasoning).
No input is ever fed through the network: only weight matrices times bound vectors / coefficient matrices.
"""
from __future__ import annotations

import time

import numpy as np

MARGIN = 1e-6      # certified dead iff upper bound < −MARGIN (ground truth is max <= 0; decoys peak at +0.02)
INFLATE = 1e-9     # relative widening of intermediate bounds against float64 rounding

Layers = list[tuple[np.ndarray, np.ndarray]]


def _relax(lo: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit linear ReLU bounds a_l·z <= relu(z) <= a_u·z + c_u, valid for z in [lo, up]."""
    a_u, c_u, a_l = np.zeros_like(up), np.zeros_like(up), np.zeros_like(up)
    act = lo >= 0
    a_u[act], a_l[act] = 1.0, 1.0
    uns = (lo < 0) & (up > 0)
    s = up[uns] / (up[uns] - lo[uns])
    a_u[uns], c_u[uns] = s, -s * lo[uns]
    a_l[uns] = (up[uns] > -lo[uns]).astype(float)
    return a_u, c_u, a_l


def _backsub(layers: Layers, k: int, lows: list[np.ndarray], ups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Upper and lower bounds of z_k (0-based hidden layer index k) over x in [0,1]^n by back-substitution."""
    w, b = layers[k]
    lam_u, off_u = w.copy(), b.copy()          # upper bound: lam_u · h_{k-1} + off_u
    lam_l, off_l = w.copy(), b.copy()          # lower bound: lam_l · h_{k-1} + off_l
    for i in range(k - 1, -1, -1):
        a_u, c_u, a_l = _relax(lows[i], ups[i])
        pu, nu = np.maximum(lam_u, 0), np.minimum(lam_u, 0)
        pl, nl = np.maximum(lam_l, 0), np.minimum(lam_l, 0)
        m_u = pu * a_u + nu * a_l              # coefficients on z_i
        off_u = off_u + pu @ c_u
        m_l = pl * a_l + nl * a_u
        off_l = off_l + nl @ c_u
        wi, bi = layers[i]
        lam_u, off_u = m_u @ wi, off_u + m_u @ bi
        lam_l, off_l = m_l @ wi, off_l + m_l @ bi
    return np.maximum(lam_u, 0).sum(1) + off_u, np.minimum(lam_l, 0).sum(1) + off_l


def _interval(w: np.ndarray, b: np.ndarray, lo: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hl, hu = np.maximum(lo, 0), np.maximum(up, 0)
    wp, wn = np.maximum(w, 0), np.minimum(w, 0)
    return wp @ hu + wn @ hl + b, wp @ hl + wn @ hu + b


class _LP:
    """Triangle-relaxed network up to hidden layer k (exclusive) as an LP over v = (x, h_1, ..., h_{k-1})."""

    def __init__(self, layers: Layers, k: int, lows: list[np.ndarray], ups: list[np.ndarray]):
        import scipy.sparse as sp

        n = layers[0][0].shape[1]
        sizes = [n] + [layers[i][0].shape[0] for i in range(k)]
        starts = np.cumsum([0] + sizes)
        nv = int(starts[-1])
        lb, ub = np.zeros(nv), np.zeros(nv)
        ub[:n] = 1.0
        # Sparse rows (COO triplets): each constraint touches one previous-layer block and one h variable.
        ub_r, ub_c, ub_v, b_ub = [], [], [], []
        eq_r, eq_c, eq_v, b_eq = [], [], [], []
        for i in range(k):
            w, bias = layers[i]
            lo, up = lows[i], ups[i]
            p0, q0 = int(starts[i]), int(starts[i + 1])
            cols = np.arange(p0, p0 + w.shape[1])
            ub[q0:q0 + len(up)] = np.maximum(up, 0)
            for j in range(len(up)):
                if up[j] <= 0:
                    continue                                   # inactive: h = 0 via bounds
                if lo[j] >= 0:                                 # active: z − h = 0
                    r = len(b_eq)
                    eq_r += [r] * (len(cols) + 1); eq_c += [*cols, q0 + j]; eq_v += [*w[j], -1.0]
                    b_eq.append(-bias[j])
                    continue
                r = len(b_ub)                                  # z − h <= 0
                ub_r += [r] * (len(cols) + 1); ub_c += [*cols, q0 + j]; ub_v += [*w[j], -1.0]
                b_ub.append(-bias[j])
                sl = up[j] / (up[j] - lo[j])                   # h − s·z <= s·(b − L)
                ub_r += [r + 1] * (len(cols) + 1); ub_c += [*cols, q0 + j]; ub_v += [*(-sl * w[j]), 1.0]
                b_ub.append(sl * (bias[j] - lo[j]))
        self.n_vars = nv
        self.lb, self.ub = lb, ub
        self.b_ub, self.b_eq = np.array(b_ub), np.array(b_eq)
        self.a_ub = sp.csr_matrix((ub_v, (ub_r, ub_c)), shape=(len(b_ub), nv))
        self.a_eq = sp.csr_matrix((eq_v, (eq_r, eq_c)), shape=(len(b_eq), nv))
        self.w_prev = (int(starts[k]), int(starts[k + 1]))   # block of h_{k-1}

    def upper(self, w_row: np.ndarray, bias: float) -> float | None:
        """Sound upper bound of w_row · h_{k-1} + bias over the relaxation (weak duality); None if HiGHS fails."""
        from scipy.optimize import linprog

        c = np.zeros(self.n_vars)
        c[self.w_prev[0]:self.w_prev[1]] = w_row
        res = linprog(-c, A_ub=self.a_ub if len(self.b_ub) else None, b_ub=self.b_ub if len(self.b_ub) else None,
                      A_eq=self.a_eq if len(self.b_eq) else None, b_eq=self.b_eq if len(self.b_eq) else None,
                      bounds=np.stack([self.lb, self.ub], 1), method="highs")
        if res.status != 0:
            return None
        # max c·v = −min(−c·v).  For any y >= 0, mu: −c·v >= (−c + A_ubᵀy + A_eqᵀmu)·v − y·b_ub − mu·b_eq, so
        # max c·v <= y·b_ub + mu·b_eq + max_box (c − A_ubᵀy − A_eqᵀmu)·v.
        y = np.maximum(-res.ineqlin.marginals, 0) if len(self.b_ub) else np.zeros(0)
        mu = -res.eqlin.marginals if len(self.b_eq) else np.zeros(0)
        red = c - self.a_ub.T @ y - self.a_eq.T @ mu
        red = np.asarray(red).ravel()
        box = np.where(red > 0, red * self.ub, red * self.lb).sum()
        return float(y @ self.b_ub + mu @ self.b_eq + box + bias)


def have_lp() -> bool:
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        return False
    return True


def bounds(layers: Layers, use_lp: bool = True, lp_budget_s: float | None = None) -> tuple[list[np.ndarray], dict]:
    """Sound upper bounds of every hidden pre-activation over x in [0,1]^n, and a record of what was done."""
    L = [(np.asarray(w, np.float64), np.asarray(b, np.float64)) for w, b in layers[:-1]]
    use_lp = use_lp and have_lp()
    t0 = time.time()
    lows: list[np.ndarray] = []
    ups: list[np.ndarray] = []      # padded bounds, used by later layers
    raw: list[np.ndarray] = []      # unpadded upper bounds, used for certification
    info = {"lp": use_lp, "lp_calls": 0, "lp_tightened": 0, "lp_failed": 0, "lp_skipped_budget": 0}
    for k, (w, b) in enumerate(L):
        if k == 0:
            up, lo = np.maximum(w, 0).sum(1) + b, np.minimum(w, 0).sum(1) + b      # exact over the box
        else:
            up, lo = _backsub(L, k, lows, ups)
            iu, il = _interval(w, b, lows[-1], ups[-1])
            up, lo = np.minimum(up, iu), np.maximum(lo, il)
        if use_lp and k > 0:
            todo = np.flatnonzero((up >= -MARGIN) & (lo < 0))
            lp = _LP(L, k, lows, ups) if len(todo) else None
            for j in todo:
                if lp_budget_s is not None and time.time() - t0 > lp_budget_s:
                    info["lp_skipped_budget"] += 1
                    continue
                info["lp_calls"] += 1
                v = lp.upper(w[j], float(b[j]))
                if v is None:
                    info["lp_failed"] += 1
                elif v < up[j]:
                    up[j] = v
                    info["lp_tightened"] += 1
        raw.append(up)
        lows.append(lo - INFLATE * (1.0 + np.abs(lo)))
        ups.append(up + INFLATE * (1.0 + np.abs(up)))
    info["seconds"] = round(time.time() - t0, 2)
    return raw, info


def crown_dead(layers: Layers, use_lp: bool = True, lp_budget_s: float | None = None) -> tuple[list[list[int]], dict]:
    """Units certified dead (upper bound < −MARGIN) per hidden layer, and the bounds record."""
    ups, info = bounds(layers, use_lp, lp_budget_s)
    return [np.flatnonzero(u < -MARGIN).tolist() for u in ups], info
