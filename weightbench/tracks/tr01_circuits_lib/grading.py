"""Report parsing and scoring for TR01-ESC (pure Python, no torch)."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

LANG_ALIASES = {
    "python": ("python", "py", "python code", "python source"),
    "english": ("english", "en", "eng"),
    "german": ("german", "de", "deutsch", "ger"),
    "romaji": ("romaji", "japanese", "ja", "jp", "japanese romaji", "romanized japanese", "rōmaji"),
}


def norm_lang(v: Any) -> str | None:
    if not isinstance(v, str):
        return None
    s = re.sub(r"[^a-zō ]", " ", v.strip().lower())
    s = re.sub(r"\s+", " ", s).strip()
    for lang, al in LANG_ALIASES.items():
        if s in al:
            return lang
    hits = {lang for lang, al in LANG_ALIASES.items() for a in al if len(a) > 3 and a in s}
    return hits.pop() if len(hits) == 1 else None


def _model_key(k: Any) -> str:
    s = str(k).strip()
    m = re.fullmatch(r"(?:model[_\s-]?)?([A-Za-z])(?:\.npz)?", s, flags=re.I)
    return m.group(1).upper() if m else s


def score_language_id(report: dict | None, truth: dict[str, str]) -> tuple[float, dict]:
    """report: {"A": "python", ...} (also accepts {"assignments": {...}} and keys like 'model_A')."""
    rep = report if isinstance(report, dict) else {}
    if isinstance(rep.get("assignments"), dict):
        rep = rep["assignments"]
    got = {_model_key(k): norm_lang(v) for k, v in rep.items()}
    per = {X: {"truth": t, "got": got.get(X), "ok": got.get(X) == t} for X, t in truth.items()}
    acc = sum(p["ok"] for p in per.values()) / max(1, len(truth))
    return float(acc), {"per_model": per}


def parse_head(x: Any) -> tuple[int, int] | None:
    """Accepts [l, h], (l, h), {"layer": l, "head": h}, "L1H3", "1.3", "1,3", "1:3", "layer 1 head 3"."""
    try:
        if isinstance(x, dict):
            l = x.get("layer", x.get("l"))
            h = x.get("head", x.get("h"))
            return int(l), int(h)
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return int(x[0]), int(x[1])
        if isinstance(x, str):
            nums = re.findall(r"\d+", x)
            if len(nums) == 2:
                return int(nums[0]), int(nums[1])
    except (TypeError, ValueError):
        return None
    return None


def parse_head_report(report: dict | None, field: str, models: list[str]) -> tuple[dict[str, set], list]:
    """report[field] = {"A": [[l, h], ...], ...}. Also accepts the per-model dict at top level."""
    rep = report or {}
    block = rep.get(field, rep) if isinstance(rep, dict) else {}
    if not isinstance(block, dict):
        return {X: set() for X in models}, [f"{field} is not an object"]
    out: dict[str, set] = {X: set() for X in models}
    bad: list = []
    for k, v in block.items():
        X = _model_key(k)
        if X not in out:
            continue
        if not isinstance(v, (list, tuple)):
            bad.append({k: v})
            continue
        for item in v:
            hd = parse_head(item)
            if hd is None:
                bad.append(item)
            else:
                out[X].add(hd)
    return out, bad


def truth_sets(scores: dict[str, list], pos: float, neg: float) -> tuple[dict[str, set], dict[str, set]]:
    """Positive heads (score >= pos) and ignored heads (neg <= score < pos) per model."""
    P, I = {}, {}
    for X, s in scores.items():
        a = np.asarray(s)
        P[X] = {(int(l), int(h)) for l, h in zip(*np.nonzero(a >= pos))}
        I[X] = {(int(l), int(h)) for l, h in zip(*np.nonzero((a >= neg) & (a < pos)))}
    return P, I


def f1_heads(pred: dict[str, set], pos: dict[str, set], ignore: dict[str, set],
             n_layers: int, n_heads: int) -> tuple[float, dict]:
    """Pooled F1 over (model, layer, head). Ignored heads count neither way; out-of-range heads are FPs.
    If there are no positives and no predictions, F1 = 1."""
    tp = fp = fn = 0
    per = {}
    for X in pos:
        p = {hd for hd in pred.get(X, set()) if hd not in ignore[X]}
        oor = {hd for hd in p if not (0 <= hd[0] < n_layers and 0 <= hd[1] < n_heads)}
        t, f, n = len(p & pos[X]), len(p - pos[X]), len(pos[X] - p)
        tp, fp, fn = tp + t, fp + f, fn + n
        per[X] = {"tp": t, "fp": f, "fn": n, "out_of_range": len(oor)}
    if tp + fp + fn == 0:
        return 1.0, {"tp": 0, "fp": 0, "fn": 0, "per_model": per}
    f1 = 2 * tp / (2 * tp + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return float(f1), {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4), "recall": round(rec, 4),
                       "per_model": per}
