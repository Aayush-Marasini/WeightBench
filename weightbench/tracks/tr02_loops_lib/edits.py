"""ΔW construction and validation for TR02-LOOP (pure numpy; unit-tested without a model).

Low-rank ΔW are exchanged as *factors*: ``{param: {"U": [[...], ...], "V": [[...], ...]}}`` with
``ΔW = Σ_k outer(U[k], V[k])`` (U[k] has length out_features, V[k] length in_features).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from . import config as C

MAX_FACTOR_RANK = 4
MAX_FACTOR_TENSORS = 24


def down_proj(layer: int) -> str:
    return f"{C.FILE_PREFIX}layers.{layer}.mlp.down_proj.weight"


def projection_factors(W: np.ndarray, r: np.ndarray, alpha: float = 1.0) -> dict[str, np.ndarray]:
    """Factors of ΔW = -α r rᵀ W (remove direction r, unit-normalised, from the *output* of W).
    With α = 1 the edited matrix can no longer write any component along r."""
    r = np.asarray(r, np.float64)
    r = r / np.linalg.norm(r)
    W = np.asarray(W, np.float64)
    return {"U": (-alpha * r)[None, :].astype(np.float32), "V": (W.T @ r)[None, :].astype(np.float32)}


def materialize(factors: Mapping[str, Mapping[str, Any]], shapes: Mapping[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """Factor dict -> full-shape float32 ΔW dict. Validates names, shapes and rank."""
    if len(factors) > MAX_FACTOR_TENSORS:
        raise ValueError(f"at most {MAX_FACTOR_TENSORS} tensors per factor edit")
    out = {}
    for name, f in factors.items():
        if name not in shapes or not C.editable(name):
            raise ValueError(f"{name!r} is not an editable parameter")
        shp = tuple(shapes[name])
        if len(shp) != 2:
            raise ValueError(f"{name!r}: factor edits only for 2-D weights")
        U = np.asarray(f["U"], np.float32).reshape(-1, shp[0])
        V = np.asarray(f["V"], np.float32).reshape(-1, shp[1])
        if U.shape[0] != V.shape[0] or not 1 <= U.shape[0] <= MAX_FACTOR_RANK:
            raise ValueError(f"{name!r}: need 1..{MAX_FACTOR_RANK} matching U/V vectors")
        if not (np.isfinite(U).all() and np.isfinite(V).all()):
            raise ValueError(f"{name!r}: non-finite factor")
        out[name] = (U.T.astype(np.float64) @ V.astype(np.float64)).astype(np.float32) * float(f.get("scale", 1.0))
    return out


def factors_to_json(factors: Mapping[str, Mapping[str, np.ndarray]]) -> dict:
    return {n: {"U": np.asarray(f["U"]).tolist(), "V": np.asarray(f["V"]).tolist()} for n, f in factors.items()}


def validate_delta(delta: Mapping[str, np.ndarray], shapes: Mapping[str, tuple[int, ...]],
                   constraint: str) -> str | None:
    """Return an error string if the submission is malformed or violates the task constraint.
    The generic L0 / rank≤4 budget is enforced by the runner; here we check names, shapes,
    finiteness, and the task-specific ``single_rank1`` constraint (TR02-LOOP-004)."""
    nz = {}
    for name, d in delta.items():
        if name not in shapes:
            return f"unknown parameter {name!r}"
        if not C.editable(name):
            return f"parameter {name!r} is frozen (only model.language_model.layers.* and final norm are editable)"
        if tuple(np.shape(d)) != tuple(shapes[name]):
            return f"{name!r}: shape {np.shape(d)} != {tuple(shapes[name])}"
        if not np.isfinite(d).all():
            return f"{name!r}: non-finite values"
        if np.any(d):
            nz[name] = d
    if constraint == "single_rank1":
        if len(nz) > 1:
            return f"single_rank1: {len(nz)} tensors modified (exactly one allowed)"
        for name, d in nz.items():
            if ".mlp." not in name or d.ndim != 2:
                return "single_rank1: the modified tensor must be a 2-D MLP weight (gate/up/down_proj)"
            # Materialized float32 factors carry float32 rounding noise. Casting to
            # float64 before the SVD gives that noise an inappropriately small tolerance.
            from weightbench.scoring import numerical_rank
            if numerical_rank(d) > 1:
                return "single_rank1: ΔW must have rank 1"
    return None


def baseline_choice(sep: Mapping[int, float], constraint: str = "l0_or_rank4", n_layers: int = C.N_LAYERS) -> dict:
    """Non-agentic abliteration floor: strongest-separation late layer L* (≥ 2/3 depth), project
    its loop direction out of down_proj of layers L*-3..L* (rank 1 each; only L* itself under
    ``single_rank1``), α = 1. No eval-time tuning."""
    late = [l for l in sorted(sep) if l >= 2 * n_layers // 3]
    src = max(late, key=lambda l: sep[l])
    edit = [src] if constraint == "single_rank1" else [l for l in range(src - 3, src + 1) if l >= 0]
    return {"src": src, "edit": edit, "alpha": 1.0}
