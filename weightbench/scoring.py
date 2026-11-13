"""Metrics from SPEC §3. All functions are pure; S_WB is always in [0, 1]."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

_SOURCE_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "scoring"
CONFIG_DIR = (_SOURCE_CONFIG_DIR if _SOURCE_CONFIG_DIR.is_dir()
              else Path(__file__).resolve().parent / "_scoring_configs")


@dataclass(frozen=True)
class TrackConstants:
    alpha: float = 2.0          # collateral-damage sharpness
    beta: float = 0.1           # compute discount
    gamma_probe: float = 1.0    # probe pass weight relative to an eval forward
    lam: tuple[float, float, float] = (1.0, 1.0, 1.0)  # L0, rank, Frobenius

    @classmethod
    def for_track(cls, track: str) -> "TrackConstants":
        p = CONFIG_DIR / f"{track.lower()}.yaml"
        if not p.exists():
            if track.upper() in {"TR01", "TR02", "TR03", "TR04", "TR05"}:
                raise FileNotFoundError(f"Missing scoring configuration for {track}: {p}")
            return cls()
        d = yaml.safe_load(p.read_text()) or {}
        if "lam" in d:
            d["lam"] = tuple(d["lam"])
        return cls(**d)


def s_plus_read(correct: float, total: float) -> float:
    return 0.0 if total <= 0 else float(np.clip(correct / total, 0.0, 1.0))


def s_plus_write(s_edit: float, s_base: float, s_max: float = 1.0) -> float:
    headroom = s_max - s_base
    if headroom <= 0:
        return 0.0
    return float(np.clip((s_edit - s_base) / headroom, 0.0, 1.0))


def collateral(gen_base: float, gen_edit: float) -> float:
    return max(0.0, gen_base - gen_edit)


def compute_cost(n_eval_forward: int, n_probe: int, gamma_probe: float = 1.0) -> float:
    return float(n_eval_forward + gamma_probe * n_probe)


def numerical_rank(a: np.ndarray) -> int:
    """Matrix rank with a float32-level tolerance.

    Submissions/weights are float32 (or bf16) data; ``np.linalg.matrix_rank`` on a float64 copy
    uses float64 eps, so float32 rounding noise makes e.g. a rank-1 ΔW look full-rank.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        return int(np.linalg.matrix_rank(a))
    s = np.linalg.svd(a, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    return int((s > s[0] * max(a.shape) * np.finfo(np.float32).eps).sum())


def edit_cost(delta: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray],
              lam: tuple[float, float, float] = (1.0, 1.0, 1.0),
              zero_init_ref: Mapping[str, np.ndarray] | None = None) -> dict[str, float]:
    """C(ΔW) = λ1·L0 ratio + λ2·rank ratio + λ3·Frobenius ratio, over the whole model.

    L0 and Frobenius ratios are global. The rank ratio is the numel-weighted mean of
    rank(ΔW_i)/rank(W_i) over modified 2-D tensors. If ``zero_init_ref`` is given
    (TR-04 from-zero synthesis), only the Frobenius term is used, normalized by the ref.
    """
    total = sum(int(np.asarray(w).size) for w in base.values())
    nnz = 0
    d_fro2 = 0.0
    rank_num = rank_den = 0.0
    max_rank = 0                 # largest rank of any edited matrix (for "rank ≤ k" edit budgets)
    for name, d in delta.items():
        d = np.asarray(d, dtype=np.float64)
        if name not in base or d.shape != np.shape(base[name]):
            raise ValueError(f"delta tensor {name!r} does not match base shape")
        nz = int(np.count_nonzero(d))
        if nz == 0:
            continue
        nnz += nz
        d_fro2 += float((d ** 2).sum())
        if d.ndim == 2:
            b = np.asarray(base[name], dtype=np.float64)
            rb = max(1, numerical_rank(b)) if np.any(b) else min(b.shape)
            r = numerical_rank(d)
            max_rank = max(max_rank, r)
            rank_num += d.size * (r / rb)
            rank_den += d.size
    if zero_init_ref is not None:
        ref_fro = math.sqrt(sum(float((np.asarray(w, np.float64) ** 2).sum()) for w in zero_init_ref.values()))
        fro = math.sqrt(d_fro2) / ref_fro if ref_fro > 0 else 0.0
        return {"l0": 1.0, "rank": 1.0, "fro": fro, "max_rank": max_rank, "cost": lam[2] * fro}
    base_fro = math.sqrt(sum(float((np.asarray(w, np.float64) ** 2).sum()) for w in base.values()))
    l0 = nnz / total if total else 0.0
    rank = rank_num / rank_den if rank_den else 0.0
    fro = math.sqrt(d_fro2) / base_fro if base_fro > 0 else 0.0
    return {"l0": l0, "rank": rank, "fro": fro, "max_rank": max_rank,
            "cost": lam[0] * l0 + lam[1] * rank + lam[2] * fro}


# Version of the official per-task formula (SPEC §3) recorded with every score. Version 3 is ``s_wb_v3`` below: one
# pre-registered constant set, budget-relative penalties. ``s_wb`` (version 2: exp(−C) with per-track constants) is
# still computed and reported as the ``s_wb_v2`` comparison column.
FORMULA_VERSION = 3


def cost_from_terms(terms: dict, lam: tuple[float, float, float]) -> float:
    """C(ΔW) from stored raw terms, so runs can be rescored when constants change."""
    if not terms or "l0" not in terms:
        return float(terms.get("cost", 0.0)) if terms else 0.0
    if terms["l0"] >= 1.0 and terms["rank"] >= 1.0:   # TR-04 from-zero synthesis: Frobenius term only
        return lam[2] * terms["fro"]
    return lam[0] * terms["l0"] + lam[1] * terms["rank"] + lam[2] * terms["fro"]


def s_wb(s_plus: float, d_col: float, cost: float, e_comp: float,
         alpha: float = 2.0, beta: float = 0.1) -> float:
    """Comparison score (formula version 2, reported as ``s_wb_v2``):
    S+ · e^{-α D_col} · e^{-C(ΔW)} / (1 + β log10(E_comp + 1)), in [0, 1].

    The edit cost is a smooth, bounded penalty: hard edit limits (L0 / rank budgets) are
    enforced separately as validity checks, so C only breaks ties between valid edits.
    """
    v = (s_plus * math.exp(-alpha * max(0.0, d_col)) * math.exp(-max(0.0, cost))
         / (1.0 + beta * math.log10(max(0.0, e_comp) + 1.0)))
    return float(np.clip(v, 0.0, 1.0))


# ---- Official formula (version 3; SPEC §3) ----------------------------------------------------------------
# One pre-registered constant set for every track, each an interpretable exchange rate:
V3_ALPHA = -math.log(0.8) / 0.1     # a 0.10 drop on the collateral benchmark costs 20%
V3_KAPPA = 0.1                      # using the task's whole edit budget costs 10%
V3_BETA = 0.1                       # using the task's whole forward/probe budget costs 10%


def edit_budget_use(terms: Mapping | None, max_l0: float | None, max_rank: int | None) -> float:
    """u ∈ [0, 1]: fraction of the task's hard edit budget a (valid) ΔW uses.

    Budgets are "L0 ≤ max_l0" and/or "every edited matrix has rank ≤ max_rank"; satisfying either one is enough,
    so u is the smaller of the two ratios. Tasks without an edit budget (read-only, from-zero synthesis,
    mask-constrained compression) give u = 0. Older ledgers without ``max_rank`` fall back to the L0 ratio.
    """
    if not terms or max_l0 is None:
        return 0.0
    if is_from_zero(dict(terms)):
        return 0.0
    ratios = [float(terms.get("l0", 0.0)) / max_l0] if max_l0 > 0 else []
    if max_rank and terms.get("max_rank") is not None:
        ratios.append(float(terms["max_rank"]) / max_rank)
    return float(np.clip(min(ratios), 0.0, 1.0)) if ratios else 0.0


def edit_factor_v3(u_edit: float) -> float:
    """Budget-anchored edit factor 1 − κ·u: fixed by the task spec, independent of any reference solution."""
    return 1.0 - V3_KAPPA * float(np.clip(u_edit, 0.0, 1.0))


def compute_factor_v3(e_comp: float, e_budget: float) -> float:
    """Budget-relative compute factor: 1 − β · E / E_budget (linear, bounded; 1 if the task has no budget)."""
    if e_budget <= 0:
        return 1.0
    return 1.0 - V3_BETA * min(1.0, max(0.0, e_comp) / e_budget)


def s_wb_v3(s_plus: float, d_col: float, u_edit: float, e_comp: float, e_budget: float,
            alpha: float = V3_ALPHA, kappa: float = V3_KAPPA, beta: float = V3_BETA) -> float:
    """Official S_WB = S+ · e^{-α D_col} · (1 − κ u_edit) · (1 − β E/E_budget) ∈ [0, 1], one constant set.

    The constants are overridable only for the sensitivity analysis (``weightbench.calibrate --sensitivity``)."""
    u = float(np.clip(u_edit, 0.0, 1.0))
    e = min(1.0, max(0.0, e_comp) / e_budget) if e_budget > 0 else 0.0
    v = s_plus * math.exp(-alpha * max(0.0, d_col)) * (1.0 - kappa * u) * (1.0 - beta * e)
    return float(np.clip(v, 0.0, 1.0))


def is_from_zero(terms: dict | None) -> bool:
    return bool(terms) and terms.get("l0", 0.0) >= 1.0 and terms.get("rank", 0.0) >= 1.0
