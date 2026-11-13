"""Scoring-constant tools (SPEC §3).

    python -m weightbench.calibrate --sensitivity          # official formula: leaderboard robustness
    python -m weightbench.calibrate [--runs runs/] [--write TR03,TR04,TR05]   # per-track constants (s_wb_v2)

The official formula (formula version 3, ``scoring.s_wb_v3``) uses one pre-registered constant set and needs no
calibration; ``--sensitivity`` checks that rankings hold when each constant is halved or doubled. The rest of this
module calibrates the per-track constants of the comparison score (formula version 2, the ``s_wb_v2`` column):

The constants only *discount* S+, so they cannot move a baseline with S+ = 0 off 0, and no choice of constants
alone can place references and baselines at fixed target scores. Each constant is set instead against an
explicit, documented target:

- λ (edit cost): scaled per track so the median **reference** loses ``REF_EDIT_LOSS`` (10%) to edit cost,
  i.e. median e^{-C} = 0.90. Edit cost stays a tie-breaker between working edits, not a gate.
- β (compute): using a track's **full** forward/probe budget costs ``FULL_BUDGET_LOSS`` (10%):
  1 / (1 + β log10(E_max + 1)) = 0.90.
- α (collateral): a drop of ``COLLATERAL_POINTS`` (0.10) on the general benchmarks costs
  ``COLLATERAL_LOSS`` (20%): α = −ln(0.8) / 0.10. Tracks without a collateral metric keep α as configured.
- γ_probe: 1 (one probe pass ≈ one forward pass) until probe vs. forward compute is measured.

Raw cost terms (L0, rank, Frobenius) are stored in every ledger, so all runs can be rescored afterwards
(``python -m weightbench.runner rescore``).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median

import yaml

from weightbench import scoring

REF_EDIT_LOSS = 0.10
FULL_BUDGET_LOSS = 0.10
COLLATERAL_POINTS, COLLATERAL_LOSS = 0.10, 0.20


def target_alpha() -> float:
    return -math.log(1.0 - COLLATERAL_LOSS) / COLLATERAL_POINTS


def target_beta(e_max: float) -> float:
    if e_max <= 0:
        return 0.1
    return (1.0 / (1.0 - FULL_BUDGET_LOSS) - 1.0) / math.log10(e_max + 1.0)


def lambda_scale(costs: list[float]) -> float | None:
    """Factor k such that median(k · C_ref) = −ln(1 − REF_EDIT_LOSS); None if references have no edit cost."""
    positive = [c for c in costs if c > 0]
    if not positive:
        return None
    return -math.log(1.0 - REF_EDIT_LOSS) / median(positive)


def load_runs(runs: Path) -> list[dict]:
    """result.json + the last ledger evaluation (raw edit-cost terms) of every run."""
    rows = []
    for rj in sorted(runs.glob("*/result.json")):
        res = json.loads(rj.read_text())
        ledger = rj.parent / "ledger.jsonl"
        terms = None
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                e = json.loads(line)
                if e.get("kind") == "evaluation":
                    terms = e["payload"].get("edit_cost_terms")
        rows.append({**res, "terms": terms})
    return rows


def budget_e_max(track: str) -> float:
    from weightbench.task import load_all

    c = scoring.TrackConstants.for_track(track)
    e = [t.budget.n_forward + c.gamma_probe * t.budget.n_probe for t in load_all().values() if t.track == track]
    return max(e) if e else 0.0


def propose(rows: list[dict], track: str) -> dict:
    current = scoring.TrackConstants.for_track(track)
    refs = [r for r in rows if r["track"] == track and r["agent"] == "reference" and r.get("valid")]
    costs = [scoring.cost_from_terms(r["terms"], current.lam) for r in refs if r.get("terms")]
    k = lambda_scale(costs)
    has_collateral = any(r.get("d_col", 0) > 0 for r in rows if r["track"] == track)
    lam = tuple(round(x * k, 4) for x in current.lam) if k is not None else current.lam
    proposed = scoring.TrackConstants(alpha=round(target_alpha(), 2) if has_collateral else current.alpha,
                                      beta=round(target_beta(budget_e_max(track)), 3), gamma_probe=1.0, lam=lam)
    return {"track": track, "current": asdict(current), "proposed": asdict(proposed), "n_reference_runs": len(refs),
            "median_reference_cost_current": median(costs) if costs else None,
            "lambda_scale": k, "e_max": budget_e_max(track)}


def write_config(track: str, constants: dict) -> None:
    path = scoring.CONFIG_DIR / f"{track.lower()}.yaml"
    header = [line for line in path.read_text().splitlines() if line.startswith("#")] if path.exists() else []
    body = {**constants, "lam": list(constants["lam"])}
    path.write_text("\n".join(header + ["# Calibrated with `python -m weightbench.calibrate` (targets in its docstring)."])
                    + "\n" + yaml.safe_dump(body, sort_keys=False))


# ---- sensitivity of the leaderboard to the official-formula constants --------------------------------------
SENSITIVITY_FACTORS = (0.5, 1.0, 2.0)     # each constant at half / default / double its pre-registered value


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall's τ between two rankings of the same items (1 = identical order, −1 = reversed)."""
    pos = {item: i for i, item in enumerate(b)}
    pairs = [(x, y) for i, x in enumerate(a) for y in a[i + 1:]]
    if not pairs:
        return 1.0
    concordant = sum(1 for x, y in pairs if pos[x] < pos[y])
    return (2 * concordant - len(pairs)) / len(pairs)


def system_means(rows: list[dict], key: str) -> dict[str, float]:
    """Overall mean of ``key`` per system: mean over tasks of the mean over samples (as in the leaderboard)."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r.get("system") or r.get("model") or r["agent"]][r["task"]].append(r[key])
    return {s: sum(sum(v) / len(v) for v in tasks.values()) / len(tasks) for s, tasks in per.items()}


def sensitivity(rows: list[dict]) -> dict:
    """Recompute every system's overall official S_WB on a 3×3×3 grid of (α, κ, β) and compare rankings to the default.

    Reported: Kendall τ to the default ranking (min over the grid), whether the top system changes, and the
    largest absolute change of any system's mean. A ranking claim should hold on the whole grid."""
    from weightbench import report

    report.add_v3(rows)
    default = system_means(rows, "s_wb_v3")
    default_rank = sorted(default, key=default.get, reverse=True)
    grid = []
    for fa in SENSITIVITY_FACTORS:
        for fk in SENSITIVITY_FACTORS:
            for fb in SENSITIVITY_FACTORS:
                consts = {"alpha": scoring.V3_ALPHA * fa, "kappa": scoring.V3_KAPPA * fk, "beta": scoring.V3_BETA * fb}
                report.add_v3(rows, key="_s_sens", **consts)
                means = system_means(rows, "_s_sens")
                rank = sorted(means, key=means.get, reverse=True)
                grid.append({**{k: round(v, 4) for k, v in consts.items()}, "tau": kendall_tau(default_rank, rank),
                             "top": rank[0] if rank else None,
                             "max_abs_change": max((abs(means[s] - default[s]) for s in means), default=0.0)})
    for r in rows:
        r.pop("_s_sens", None)
    return {"default_ranking": default_rank, "default_means": default, "n_systems": len(default),
            "min_tau": min(g["tau"] for g in grid), "top_changes": sorted({g["top"] for g in grid} - {default_rank[0]})
            if default_rank else [], "max_abs_change": max(g["max_abs_change"] for g in grid), "grid": grid}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parent.parent / "runs"))
    ap.add_argument("--write", default="", help="comma-separated tracks whose configs to overwrite")
    ap.add_argument("--sensitivity", action="store_true", help="leaderboard robustness to the v3 constants")
    a = ap.parse_args(argv)
    if a.sensitivity:
        from weightbench import report

        rows = report.load()
        result = sensitivity(rows)
        print(json.dumps({k: v for k, v in result.items() if k != "grid"}, indent=1))
        return
    rows = load_runs(Path(a.runs))
    by_track = defaultdict(list)
    for r in rows:
        by_track[r["track"]].append(r)
    report = [propose(rows, t) for t in sorted(by_track)]
    print(json.dumps(report, indent=1))
    for item in report:
        if item["track"] in set(filter(None, a.write.split(","))):
            write_config(item["track"], item["proposed"])
            print(f"wrote configs/scoring/{item['track'].lower()}.yaml")


if __name__ == "__main__":
    main()
