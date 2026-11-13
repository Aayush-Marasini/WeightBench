"""Pure-NumPy ΔW construction and constraint checks for TR-02 sycophancy (no torch, no model).

Conventions: weights use the HF/PyTorch layout ``W[out, in]``. For attention,
``o_proj.weight`` is ``[hidden, n_heads*head_dim]`` (columns ``h*hd:(h+1)*hd`` read head h)
and ``q_proj.weight`` is ``[n_heads*head_dim, hidden]`` (rows of head h).
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

_ATTN_RX = re.compile(r"^(?P<pre>.*\blayers\.(?P<l>\d+)\.self_attn\.)(?P<p>[qkvo])_proj\.(?P<kind>weight|bias)$")


def attn_param_map(names) -> dict[int, dict[str, str]]:
    """layer -> {"q","k","v","o" (+ "q_bias", ...)} -> parameter name. Model-agnostic via regex."""
    out: dict[int, dict[str, str]] = defaultdict(dict)
    for n in names:
        m = _ATTN_RX.match(n)
        if m:
            key = m["p"] if m["kind"] == "weight" else f"{m['p']}_bias"
            out[int(m["l"])][key] = n
    return dict(out)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def diff_means_direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit vector along mean(a) - mean(b); a, b are [N, d]."""
    return unit(np.asarray(a, np.float64).mean(0) - np.asarray(b, np.float64).mean(0))


def project_out_output(W: np.ndarray, u: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ΔW removing (alpha=1) / reversing (alpha=2) the component of W's *output* along unit u.

    ΔW = -alpha · u (uᵀ W): rank 1.
    """
    u = unit(u)
    return (-alpha * np.outer(u, u @ np.asarray(W, np.float64))).astype(np.float32)


def head_slice_rank1(W_o: np.ndarray, head: int, head_dim: int, d: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ΔW_o that removes head ``head``'s write along head-space unit direction d (rank 1).

    Only columns of that head are non-zero: ΔW[:, s] = -alpha · (W[:, s] d) dᵀ.
    """
    d = unit(d)
    s = slice(head * head_dim, (head + 1) * head_dim)
    out = np.zeros_like(W_o, dtype=np.float32)
    out[:, s] = (-alpha * np.outer(np.asarray(W_o, np.float64)[:, s] @ d, d)).astype(np.float32)
    return out


def touched_heads(delta: dict[str, np.ndarray], names, n_heads: int, head_dim: int,
                  tol: float = 0.0) -> tuple[set[tuple[int, int]], list[str]]:
    """Which (layer, head) pairs a ΔW touches, plus violations of the head-only rule.

    Allowed: q_proj rows / q_proj bias entries and o_proj columns. Anything else
    (k/v (shared across heads under GQA), MLP, norms, embeddings) is a violation.
    """
    pm = attn_param_map(names)
    rev = {n: (l, k) for l, d in pm.items() for k, n in d.items()}
    heads: set[tuple[int, int]] = set()
    bad: list[str] = []
    for name, d in delta.items():
        d = np.asarray(d)
        nz = np.abs(d) > tol
        if not nz.any():
            continue
        if name not in rev or rev[name][1] not in ("q", "o", "q_bias"):
            bad.append(name)
            continue
        l, k = rev[name]
        if k == "o":
            cols = nz.reshape(d.shape[0], n_heads, head_dim).any(axis=(0, 2))
        else:
            cols = nz.reshape(n_heads, head_dim, *d.shape[1:]).reshape(n_heads, -1).any(axis=1)
        heads |= {(l, int(h)) for h in np.nonzero(cols)[0]}
    return heads, bad


def check_head_limit(delta: dict[str, np.ndarray], names, n_heads: int, head_dim: int,
                     max_heads: int) -> tuple[bool, str, list[tuple[int, int]]]:
    heads, bad = touched_heads(delta, names, n_heads, head_dim)
    if bad:
        return False, f"head-only edit rule: tensors outside q_proj/o_proj head slices modified: {bad[:5]}", sorted(heads)
    if len(heads) > max_heads:
        return False, f"head-only edit rule: {len(heads)} heads touched > {max_heads}", sorted(heads)
    return True, "", sorted(heads)


def check_shapes(delta: dict[str, np.ndarray], shapes: dict[str, tuple[int, ...]]) -> str | None:
    """None if every ΔW tensor names a real parameter with the right shape, else an error message."""
    for k, d in delta.items():
        if k not in shapes:
            return f"unknown parameter {k!r}"
        if tuple(np.shape(d)) != tuple(shapes[k]):
            return f"shape mismatch for {k}: {tuple(np.shape(d))} vs {tuple(shapes[k])}"
        if not np.all(np.isfinite(d)):
            return f"non-finite values in {k}"
    return None


def merge(*deltas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for d in deltas:
        for k, v in d.items():
            out[k] = (out[k] + v).astype(np.float32) if k in out else np.asarray(v, np.float32)
    return out


def max_rank(delta: dict[str, np.ndarray]) -> int:
    return max((int(np.linalg.matrix_rank(v)) for v in delta.values() if np.ndim(v) == 2 and np.any(v)), default=0)
