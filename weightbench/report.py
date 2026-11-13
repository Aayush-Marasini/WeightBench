"""Aggregate runs/*/result.json into ``summary.json`` and ``RESULTS.md`` in the results directory
(``WB_RESULTS_DIR``, default ``results`` at the repo root).

    python -m weightbench.report
"""
from __future__ import annotations

import json
import os
import sys
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from weightbench import conditions

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = Path(os.environ.get("WB_RESULTS_DIR", ROOT / "results"))   # set to write elsewhere, e.g. a gitignored dir


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (0.0, 0.0)
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def track_of(task_id: str) -> str:
    return task_id.split("-")[0]


def macro_mean(per_task: dict[str, float]) -> float:
    """The WeightBench score (SPEC §3.3): mean over tracks of the mean over that track's tasks, so every track
    weighs the same however many tasks it has."""
    by_track: dict[str, list[float]] = defaultdict(list)
    for task, value in per_task.items():
        by_track[track_of(task)].append(value)
    return mean(mean(v) for v in by_track.values()) if by_track else 0.0


def macro_bootstrap_ci(per_task: dict[str, float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of ``macro_mean``: tasks are resampled with replacement within each track (stratified bootstrap)."""
    by_track: dict[str, list[float]] = defaultdict(list)
    for task, value in per_task.items():
        by_track[track_of(task)].append(value)
    if not by_track:
        return 0.0, 0.0
    rng = random.Random(seed)
    stats = sorted(mean(mean(rng.choices(v, k=len(v))) for v in by_track.values()) for _ in range(n))
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def paired_diff_ci(a: dict[str, float], b: dict[str, float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% stratified-bootstrap CI of macro_mean(a) − macro_mean(b) over the tasks both systems have."""
    common = sorted(set(a) & set(b))
    return macro_bootstrap_ci({t: a[t] - b[t] for t in common}, n, seed)


def tie_groups(per_task: dict[str, dict[str, float]], systems: list[str]) -> dict[str, int]:
    """SPEC §3.3 statistical ties: rank systems by their WeightBench score. A system joins the current group if
    the paired bootstrap CI of its difference to the group's leader contains 0; otherwise it starts a new
    group. Returns system -> group number (1 = best)."""
    ranked = sorted((s for s in systems if per_task.get(s)), key=lambda s: -macro_mean(per_task[s]))
    groups: dict[str, int] = {}
    leader, group = None, 0
    for s in ranked:
        if leader is not None:
            lo, hi = paired_diff_ci(per_task[leader], per_task[s])
            if lo <= 0.0 <= hi:
                groups[s] = group
                continue
        group += 1
        leader = s
        groups[s] = group
    return groups


def edit_budget_use(row: dict) -> float:
    """u_edit of a result row: stored by the runner, or recomputed from the ledger terms and the task's budget."""
    from weightbench import scoring
    from weightbench.task import REGISTRY, load_all

    if "edit_budget_use" in row:
        return float(row["edit_budget_use"])
    load_all()
    task = REGISTRY.get(row["task"])
    if task is None:
        return 0.0
    return scoring.edit_budget_use(row.get("edit_cost_terms"), task.budget.max_l0, task.budget.max_rank)


def add_v3(rows: list[dict], key: str = "s_wb_v3", **constants: float) -> None:
    """Official formula (version 3, SPEC §3) under ``key``: one constant set for all tracks, edit and compute penalties
    relative to the task's own budgets. ``constants`` (alpha / kappa / beta) override the defaults for the
    sensitivity analysis only."""
    from weightbench import scoring

    for r in rows:
        if not r.get("valid") or r.get("s_plus", 0.0) == 0.0:
            r[key] = 0.0
            continue
        r[key] = scoring.s_wb_v3(r["s_plus"], r.get("d_col", 0.0), edit_budget_use(r), r.get("e_comp", 0.0),
                                 r.get("e_budget", 0.0), **constants)


def _review(run_dir: Path) -> dict:
    """Human review.json wins in either direction; otherwise an auto_review.json that cleared (weightbench.review)."""
    from weightbench.review import review_status
    return review_status(run_dir)


def _cleared(run_dir: Path) -> bool:
    return _review(run_dir)["cleared"]


FLOORS = ("baseline", "reference")


def floor_system(result: dict) -> str:
    """System name of a floor run: ``baseline`` / ``reference``, or ``reference-<variant>`` for a run of a task's
    non-default reference variant (``variant`` != ``default_variant`` in result.json). Results without the fields
    (all runs before variants were recorded, and the default variant) keep the bare name."""
    agent = str(result.get("agent"))
    variant = result.get("variant")
    if agent == "reference" and variant and variant != result.get("default_variant"):
        return f"{agent}-{variant}"
    return agent


def _latest_floors(rows: list[dict]) -> list[dict]:
    """Floors are deterministic, so a later sample of the same (floor, task, seed) only exists because the floor
    was redefined (e.g. a stronger reference); keep the latest one. Agent samples are all kept. A non-default
    reference variant is its own floor system, so it never replaces the default reference."""
    latest: dict[tuple, int] = {}
    for r in rows:
        if r["agent"] in FLOORS:
            key = (floor_system(r), r["task"], r.get("seed", 0))
            latest[key] = max(latest.get(key, -1), r.get("sample", 0))
    return [r for r in rows if r["agent"] not in FLOORS
            or r.get("sample", 0) == latest[(floor_system(r), r["task"], r.get("seed", 0))]]


def load() -> list[dict]:
    """All results (only the latest sample of each deterministic floor, see ``_latest_floors``). A quarantined run
    (SPEC §1 audit) counts as S_WB = S+ = 0 until a reviewer clears it with ``review.json`` = {"cleared": true}."""
    rows = []
    for p in sorted(RUNS.glob("*/result.json")):
        r = json.loads(p.read_text())
        if r["task"].startswith("SMOKE") or r.get("level") is not None:   # off-default knob levels: not the suite
            continue
        r["system"] = r["model"] if r["agent"] == "claude" else floor_system(r)
        r["system"] = conditions.system_name(r["system"], r.get("condition", "full"))
        status = _review(p.parent) if r.get("quarantined") else {"cleared": True, "by": None}
        if status["by"] == "auto" and status["cleared"]:
            r["auto_cleared"] = status["reviewer"]           # labelled separately in the report
        if r.get("quarantined") and not status["cleared"]:
            r["s_wb"] = r["s_plus"] = 0.0
            r["audit_flags"] = sorted(set(r.get("audit_flags", [])) | {"QUARANTINED"})
        rows.append(r)
    return _latest_floors(rows)


def official_scores(rows: list[dict]) -> None:
    """S_WB = official formula for every row. A result.json scored with formula version 2 carries that score in
    ``s_wb``; it is kept as the ``s_wb_v2`` comparison column."""
    for r in rows:
        if "s_wb_v2" not in r:
            r["s_wb_v2"] = r.get("s_wb") if r.get("formula_version", 2) == 2 else None
        if "QUARANTINED" in r.get("audit_flags", []):
            r["s_wb_v2"] = 0.0
    add_v3(rows)
    for r in rows:
        r["s_wb"] = r["s_wb_v3"]


def exclusion_reason(r: dict) -> str | None:
    """Why a run stays out of every mean, by the leaderboard rule (``leaderboard.classify_run``): red-team ``cmd`` runs,
    invalid runs, and quarantined runs without an approving review.json. None = the run counts."""
    if r.get("agent") == "cmd":
        return "red-team cmd run"
    if not r.get("valid"):
        return "invalid run (agent_auth_failed)" if r.get("agent_auth_failed") else "invalid run"
    if "QUARANTINED" in r.get("audit_flags", []):
        return "quarantined, no approving review.json"
    return None


def split_excluded(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    """(rows that count, [(run_id, reason)] for the rest). ``load()`` still returns every run; only means exclude."""
    kept, excluded = [], []
    for r in rows:
        reason = exclusion_reason(r)
        if reason is None:
            kept.append(r)
        else:
            excluded.append((str(r.get("run_id", f"{r.get('system')}/{r.get('task')}")), reason))
    return kept, excluded


def excluded_lines(excluded: list[tuple[str, str]]) -> list[str]:
    if not excluded:
        return []
    counts: dict[str, int] = defaultdict(int)
    for _, reason in excluded:
        counts[reason] += 1
    return [f"Excluded from all means (leaderboard rule): {len(excluded)} run(s): "
            + ", ".join(f"{n} {reason}" for reason, n in sorted(counts.items())) + ". "
            + "Runs: " + ", ".join(run_id for run_id, _ in excluded) + ".", ""]


def auto_cleared_lines(rows: list[dict]) -> list[str]:
    """Quarantined runs counted because an automatic review cleared them (no human review.json): their own section."""
    auto = [r for r in rows if r.get("auto_cleared")]
    if not auto:
        return []
    return ["Auto-cleared (automated reviewer, no human review.json; docs/AUTO_REVIEW.md): "
            + ", ".join(f"{r.get('run_id')} [{r['auto_cleared']}]" for r in auto) + ".", ""]


def main() -> None:
    rows, excluded = split_excluded(load())
    official_scores(rows)
    OUT.mkdir(exist_ok=True)
    by = defaultdict(list)                       # (system, task) -> runs
    for r in rows:
        by[(r["system"], r["task"])].append(r)
    systems = sorted({s for s, _ in by}, key=lambda s: (s not in ("baseline", "reference"), s))
    tasks = sorted({t for _, t in by})
    tracks = sorted({t.split("-")[0] for t in tasks})

    # Every task of the evaluated suite counts for every system: a task a system did not complete (crash, timeout,
    # skipped) scores 0 instead of silently dropping out of its average. The suite = every task any system ran.
    per_task = {s: {t: mean(x["s_wb"] for x in by[(s, t)]) if (s, t) in by else 0.0 for t in tasks} for s in systems}
    per_task_sp = {s: {t: mean(x["s_plus"] for x in by[(s, t)]) if (s, t) in by else 0.0 for t in tasks}
                   for s in systems}
    missing = {s: [t for t in tasks if (s, t) not in by] for s in systems}
    summary = {"systems": {}, "n_runs": len(rows),
               "excluded_runs": [{"run_id": i, "reason": x} for i, x in excluded],
               "auto_cleared_runs": [{"run_id": r.get("run_id"), "reviewer": r["auto_cleared"]}
                                     for r in rows if r.get("auto_cleared")]}
    for s in systems:
        tr = {}
        for k in tracks:
            vals = [v for t, v in per_task[s].items() if t.startswith(k)]
            if vals:
                tr[k] = {"mean": mean(vals), "ci95": bootstrap_ci(vals), "n_tasks": len(vals)}
        runs = [x for (ss, _), xs in by.items() if ss == s for x in xs]
        cost = sum((x.get("usage") or {}).get("total_cost_usd") or 0 for x in runs)
        flags = sorted({f for x in runs for f in x.get("audit_flags", [])})
        summary["systems"][s] = {"tracks": tr, "overall": macro_mean(per_task[s]) if per_task[s] else None,
                                 "overall_ci95": macro_bootstrap_ci(per_task[s]), "per_task": per_task[s],
                                 "per_task_s_plus": per_task_sp[s],
                                 "overall_s_plus": macro_mean(per_task_sp[s]) if per_task_sp[s] else None,
                                 "n_runs": len(runs), "api_equiv_cost_usd": round(cost, 2), "audit_flags": flags,
                                 "missing_tasks_scored_0": missing[s]}
    groups = tie_groups(per_task, systems)
    for s, g in groups.items():
        summary["systems"][s]["tie_group"] = g
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    lines = ["# WeightBench v0.1 results", "", f"{len(rows)} runs. Mean S_WB per track (95% bootstrap CI over tasks); "
             "**WeightBench score** = mean of the track means (every track weighs the same; SPEC §3.3). "
             "Group: systems in the same group are statistical ties (paired stratified bootstrap over tasks). "
             "Invalid, red-team and quarantined-uncleared runs are excluded from every mean, as on the leaderboard.", ""]
    lines += excluded_lines(excluded)
    lines += auto_cleared_lines(rows)
    lines += ["| Group | System | " + " | ".join(tracks) + " | WeightBench score | Runs | API-equiv $ | Audit flags |",
              "|---|---|" + "---|" * (len(tracks) + 4)]
    for s in sorted(systems, key=lambda x: (groups.get(x, 10 ** 6), x)):
        d = summary["systems"][s]
        cells = []
        for k in tracks:
            t = d["tracks"].get(k)
            cells.append(f"{t['mean']:.3f} [{t['ci95'][0]:.2f}, {t['ci95'][1]:.2f}]" if t else "–")
        lo, hi = d["overall_ci95"]
        ov = f"**{d['overall']:.3f}** [{lo:.2f}, {hi:.2f}]" if d["overall"] is not None else "–"
        flags = ", ".join(d["audit_flags"]) or "–"
        lines.append(f"| {groups.get(s, '–')} | {s} | " + " | ".join(cells)
                     + f" | {ov} | {d['n_runs']} | {d['api_equiv_cost_usd']} | {flags} |")
    lines += ["", "## Mean S+ (task gain / accuracy before edit-cost, collateral and compute discounts)", "",
              "| System | " + " | ".join(tracks) + " | Overall |", "|---|" + "---|" * (len(tracks) + 1)]
    for s in systems:
        cells = []
        for k in tracks:
            v = [x for t, x in per_task_sp[s].items() if t.startswith(k)]
            cells.append(f"{mean(v):.3f}" if v else "–")
        o = summary["systems"][s]["overall_s_plus"]
        lines.append(f"| {s} | " + " | ".join(cells) + f" | {o:.3f} |" if o is not None else f"| {s} | –|")
    per_task_v2 = {s: {t: mean(x["s_wb_v2"] for x in by[(s, t)]) for t in tasks
                       if (s, t) in by and all(x.get("s_wb_v2") is not None for x in by[(s, t)])} for s in systems}
    lines += ["", "## For comparison: mean S_WB under the previous formula v2 (per-track constants)", "",
              "| System | " + " | ".join(tracks) + " | Overall |", "|---|" + "---|" * (len(tracks) + 1)]
    for s in systems:
        cells = []
        for k in tracks:
            v = [x for t, x in per_task_v2[s].items() if t.startswith(k)]
            cells.append(f"{mean(v):.3f}" if v else "–")
        overall = f"{macro_mean(per_task_v2[s]):.3f}" if per_task_v2[s] else "–"
        lines.append(f"| {s} | " + " | ".join(cells) + f" | {overall} |")
    lines += ["", "## Per task (mean S_WB over samples; S+ in parentheses)", "", "| Task | " + " | ".join(systems) + " |",
              "|---|" + "---|" * len(systems)]
    for t in tasks:
        cells = ["0.000 (not run)" if t in missing[s] else f"{per_task[s][t]:.3f} ({per_task_sp[s][t]:.2f})"
                 for s in systems]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    from weightbench import item_analysis
    items = item_analysis.analyse(rows)
    hv = item_analysis.headline_valid_only(per_task, items)
    lines += item_analysis.markdown(items)
    lines += ["", "Headline WeightBench score over valid tasks only: "
              + ", ".join(f"{s} {v:.3f}" for s, v in hv.items() if v is not None)]
    summary["item_analysis"] = items
    summary["headline_valid_only"] = hv
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _print_safely("\n".join(lines))


def _print_safely(text: str) -> None:
    """Print, replacing characters the console encoding cannot show (e.g. '−' on a Windows cp1252 console)."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


if __name__ == "__main__":
    main()
