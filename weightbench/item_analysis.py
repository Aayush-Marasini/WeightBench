"""Item (task) analysis for benchmark robustness.

For every task:
  * validity gate (pre-registered): reference − baseline ≥ MIN_MARGIN and reference ≥ MIN_REF, i.e. the task is
    shown solvable by a non-agentic weight-space method and not solved by the generic floor. Tasks that fail are
    *provisional*: reported, but excluded from the headline score.
  * discrimination across the capability ladder (Haiku < Sonnet < Opus): Spearman ρ between run score and ladder
    rank, the between-model share of variance (ICC-like η²), saturation (all models ≥ 0.95) and floor (all ≤ 0.05)
    flags, and within-model SD (run-to-run noise).

Used by ``weightbench.report``; also ``python -m weightbench.item_analysis``.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev

MIN_MARGIN = 0.20
MIN_REF = 0.50
LADDER = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]   # increasing capability
SAT, FLOOR = 0.95, 0.05


def _spearman(x: list[float], y: list[float]) -> float | None:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def analyse(rows: list[dict], key: str = "s_wb") -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[(r["task"], r["system"])].append(float(r.get(key) or 0.0))
    out: dict[str, dict] = {}
    for t in sorted({t for t, _ in by}):
        base = mean(by[(t, "baseline")]) if (t, "baseline") in by else None
        ref = mean(by[(t, "reference")]) if (t, "reference") in by else None
        valid = base is not None and ref is not None and ref - base >= MIN_MARGIN and ref >= MIN_REF
        models = {m: by[(t, m)] for m in LADDER if (t, m) in by}
        means = {m: mean(v) for m, v in models.items()}
        xs = [LADDER.index(m) for m, v in models.items() for _ in v]
        ys = [s for v in models.values() for s in v]
        rho = _spearman(xs, ys)
        eta2 = None
        if len(ys) >= 3 and len(models) >= 2:
            grand = mean(ys)
            sst = sum((y - grand) ** 2 for y in ys)
            ssb = sum(len(v) * (mean(v) - grand) ** 2 for v in models.values())
            eta2 = ssb / sst if sst > 0 else 0.0
        noise = mean(pstdev(v) for v in models.values() if len(v) > 1) if any(len(v) > 1 for v in models.values()) else None
        flags = []
        if not valid:
            flags.append("provisional: " + ("no reference/baseline" if base is None or ref is None else
                                            f"ref−base={ref - base:+.2f}, ref={ref:.2f}"))
        if len(means) >= 2 and all(v >= SAT for v in means.values()):
            flags.append("saturated")
        if len(means) >= 2 and all(v <= FLOOR for v in means.values()):
            flags.append("floor")
        if eta2 is not None and noise is not None and len(means) >= 2 and eta2 < 0.2 and "saturated" not in flags:
            flags.append("low discrimination")
        out[t] = {"baseline": base, "reference": ref, "valid": valid, "model_means": means, "spearman_rho": rho,
                  "eta2_between_models": eta2, "within_model_sd": noise, "flags": flags}
    return out


def markdown(items: dict[str, dict]) -> list[str]:
    f = lambda v: "–" if v is None else f"{v:.2f}"
    lines = ["", "## Item analysis (task validity and discrimination)", "",
             f"Validity gate (pre-registered): reference − baseline ≥ {MIN_MARGIN} and reference ≥ {MIN_REF}. "
             "Provisional tasks are excluded from the headline score. ρ = Spearman correlation of run score with the "
             "capability ladder (Haiku < Sonnet < Opus); η² = between-model share of score variance; SD = mean "
             "within-model run-to-run SD.", "",
             "| Task | Baseline | Reference | Valid | Haiku | Sonnet | Opus | ρ | η² | SD | Flags |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, d in items.items():
        mm = d["model_means"]
        lines.append(f"| {t} | {f(d['baseline'])} | {f(d['reference'])} | {'yes' if d['valid'] else '**no**'} | "
                     + " | ".join(f(mm.get(m)) for m in LADDER)
                     + f" | {f(d['spearman_rho'])} | {f(d['eta2_between_models'])} | {f(d['within_model_sd'])} | "
                     f"{'; '.join(d['flags']) or '–'} |")
    valid = [t for t, d in items.items() if d["valid"]]
    lines += ["", f"Valid tasks: {len(valid)}/{len(items)}. Saturated: "
              f"{', '.join(t for t, d in items.items() if 'saturated' in d['flags']) or 'none'}."]
    return lines


def headline_valid_only(per_task: dict[str, dict[str, float]], items: dict[str, dict]) -> dict[str, float]:
    """Macro mean over tracks of per-task S_WB, restricted to tasks that pass the validity gate."""
    out = {}
    for s, pt in per_task.items():
        tr = defaultdict(list)
        for t, v in pt.items():
            if items.get(t, {}).get("valid"):
                tr[t.split("-")[0]].append(v)
        out[s] = mean(mean(v) for v in tr.values()) if tr else None
    return out


if __name__ == "__main__":
    from weightbench.report import load, official_scores
    rows = load()
    official_scores(rows)
    print("\n".join(markdown(analyse(rows))))
