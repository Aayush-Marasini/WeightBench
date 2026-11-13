"""Shared utilities: report parsing/scoring, hidden-unit symmetries, spectral baseline."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

N_THREADS = int(os.environ.get("WB_THREADS", "3"))   # conservative default; raise on larger machines
N_PROCS = int(os.environ.get("WB_PROCS", "3"))


def torch_setup(seed: int, threads: int | None = None):
    import torch

    torch.set_num_threads(threads or N_THREADS)
    torch.manual_seed(seed)
    return torch


def device() -> str:
    """cuda > cpu (MPS deliberately skipped: tiny batched MLPs, determinism)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def norm_label(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def score_report(report: Mapping[str, Any] | None, truth: Mapping[str, str],
                 candidates: Sequence[str]) -> tuple[float, dict[str, Any]]:
    """Accuracy of a {model_name: label} report. Missing/invalid entries count as wrong."""
    report = report if isinstance(report, Mapping) else {}
    # accept either a flat mapping or {"labels": {...}}
    if isinstance(report.get("labels"), Mapping):
        report = report["labels"]
    cand = {norm_label(c) for c in candidates}
    per: dict[str, Any] = {}
    n_ok = 0
    for name, gt in truth.items():
        raw = report.get(name)
        got = norm_label(raw) if isinstance(raw, str) else None
        if got is not None and got not in cand:
            got = None
        ok = got == norm_label(gt)
        n_ok += ok
        per[name] = {"got": got, "truth": gt, "correct": bool(ok)}
    acc = n_ok / len(truth) if truth else 0.0
    return float(acc), {"accuracy": acc, "n_correct": n_ok, "n_items": len(truth), "items": per}


def permute_hidden(layers: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator,
                   positive_scale: float = 0.0, count: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Apply a function-preserving random permutation (and, for ReLU nets, positive per-unit
    rescaling with log-std ``positive_scale``) to ``count`` seeded hidden-layer bases.
    None transforms all bases with the original RNG sequence. Layers use PyTorch convention."""
    hidden = len(layers) - 1
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= hidden):
        raise ValueError(f"scrambled-layer count must be between 1 and {hidden}")
    sites = range(hidden) if count is None or count == hidden else sorted(rng.choice(hidden, count, replace=False))
    out = [(w.copy(), b.copy()) for w, b in layers]
    for i in sites:
        w, b = out[i]
        wn, bn = out[i + 1]
        p = rng.permutation(w.shape[0])
        s = np.exp(rng.normal(0, positive_scale, w.shape[0])) if positive_scale > 0 else np.ones(w.shape[0])
        out[i] = ((w[p] * s[:, None]).astype(np.float32), (b[p] * s).astype(np.float32))
        out[i + 1] = ((wn[:, p] / s[None, :]).astype(np.float32), bn.astype(np.float32))
    return out


def save_mlp(path: Path, layers: list[tuple[np.ndarray, np.ndarray]]) -> None:
    d = {}
    for i, (w, b) in enumerate(layers, 1):
        d[f"W{i}"] = np.asarray(w, np.float32)
        d[f"b{i}"] = np.asarray(b, np.float32)
    np.savez(path, **d)


def load_mlp(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    with np.load(path) as z:
        n = len([k for k in z.files if k.startswith("W")])
        return [(z[f"W{i}"], z[f"b{i}"]) for i in range(1, n + 1)]


def spectral_features(layers: list[tuple[np.ndarray, np.ndarray]], k: int = 8) -> np.ndarray:
    """Generic, label-free weight statistics: top-k singular values, Frobenius norm, stable
    rank of every weight matrix, plus bias mean/std. Permutation-invariant."""
    f: list[float] = []
    for w, b in layers:
        s = np.linalg.svd(w.astype(np.float64), compute_uv=False)
        top = np.zeros(k)
        top[: min(k, len(s))] = s[:k]
        fro = float(np.sqrt((s ** 2).sum()))
        f += list(np.log(top + 1e-6)) + [np.log(fro + 1e-6), float((s ** 2).sum() / (s[0] ** 2 + 1e-12))]
        f += [float(b.mean()), float(b.std()), float(np.abs(w).mean()), float(w.std())]
    return np.asarray(f)


def nearest_centroid(train_x: np.ndarray, train_y: Sequence[str], test_x: np.ndarray) -> list[str]:
    mu, sd = train_x.mean(0), train_x.std(0) + 1e-6
    tx, qx = (train_x - mu) / sd, (test_x - mu) / sd
    labels = sorted(set(train_y))
    cents = np.stack([tx[[y == l for y in train_y]].mean(0) for l in labels])
    d = ((qx[:, None, :] - cents[None]) ** 2).sum(-1)
    return [labels[i] for i in d.argmin(1)]


def spectral_baseline(art_dir: Path, names: Sequence[str]) -> dict[str, str]:
    """SPEC §4 floor: nearest-centroid on spectral features, centroids from a private bank of
    labeled shadow models trained by the generator with independent seeds."""
    shadow = json.loads((art_dir / "private" / "shadow.json").read_text())
    sx = np.stack([spectral_features(load_mlp(art_dir / "private" / "shadow" / f)) for f in shadow])
    sy = [shadow[f] for f in shadow]
    qx = np.stack([spectral_features(load_mlp(art_dir / "public" / "models" / f"{n}.npz")) for n in names])
    return dict(zip(names, nearest_centroid(sx, sy, qx)))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def draw_labels(rng: np.random.Generator, candidates: Sequence[str], n: int, min_distinct: int) -> list[str]:
    """Labels drawn with replacement (so elimination doesn't work), with at least min_distinct classes."""
    while True:
        y = [candidates[i] for i in rng.integers(0, len(candidates), n)]
        if len(set(y)) >= min_distinct:
            return y
