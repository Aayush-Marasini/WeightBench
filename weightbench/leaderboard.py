"""Shared, versioned leaderboard in the repo (a single source of reported numbers).

Each contributor runs benchmarks on their own machine and exports compact per-run records; the aggregate is rebuilt
from all records, so nobody has to run every benchmark alone and every reported number traces to a file.

    weightbench-leaderboard export --contributor <name> [--runs runs/]   # runs/*/result.json -> leaderboard/submissions/<name>/
    weightbench-leaderboard build                                        # -> leaderboard/LEADERBOARD.md, leaderboard.csv, runs.csv

A record keeps the scores and the evidence needed to trust them: SHA-256 of the run's ledger and transcript, the
harness git commit, formula version, audit flags and quarantine status. Quarantined runs count as 0 (SPEC §3.3).
"""
from __future__ import annotations

import argparse
import ast
import csv
import functools
import glob
import hashlib
import json
import platform
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from weightbench import conditions, knobs, report

ROOT = Path(__file__).resolve().parent.parent
BOARD = ROOT / "leaderboard"
RECORD_FIELDS = ("run_id", "task", "track", "tier", "agent", "model", "seed", "sample", "s_wb", "s_plus", "s_wb_v2",
                 "d_col", "edit_budget_use", "e_comp", "e_budget", "valid", "quarantined", "audit_flags", "wall_s",
                 "formula_version", "ledger_ok", "condition", "level", "access_plan_sha256", "harness_source_sha256")


def _variant(record: dict) -> bool:
    """Old exports omitted level metadata; their immutable ID still identifies variants."""
    return record.get('level') is not None or bool(re.search(
        r'__s\d+__r\d+__L\d+(?:__(?:weights_behavior|behavior_only))?$', str(record.get('run_id', ''))))


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def system_name(result: dict) -> str:
    """Leaderboard system: the model for agent runs (Claude Code keeps the bare id), else baseline / reference."""
    agent, model = result.get("agent"), result.get("model")
    if agent in ("baseline", "reference") or not model:
        return report.floor_system(result)
    name = model if agent == "claude" else f"{agent}-{model}"
    return conditions.system_name(name, result.get("condition") or "full")


def record_from_run(run_dir: Path, contributor: str) -> dict | None:
    """Compact, self-describing record of one finished run (None if the run has no result)."""
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    if _variant(result):
        return None
    record = {key: result.get(key) for key in RECORD_FIELDS}
    if result.get("variant") is not None:          # reference variant; absent from older results and records
        record.update(variant=result["variant"], default_variant=result.get("default_variant"))
    review_path = run_dir / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    record.update(
        review_cleared=bool(review.get("cleared")),
        review_reviewer=review.get("reviewer"),
        review_reason=review.get("reason"),
    )
    record.update(
        system=system_name(result),
        contributor=contributor,
        cost_usd=(result.get("usage") or {}).get("total_cost_usd"),
        ledger_sha256=_sha256(run_dir / "ledger.jsonl"),
        transcript_sha256=_sha256(run_dir / "transcript.jsonl"),
        harness_commit=_git_commit(),
        host=f"{platform.system()} {platform.machine()}",
        exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return record


def export(runs: Path, contributor: str, board: Path = BOARD) -> list[Path]:
    out_dir = board / "submissions" / contributor
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        record = record_from_run(run_dir, contributor)
        if record is None:
            continue
        path = out_dir / f"{run_dir.name}.json"
        path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        written.append(path)
    return written


def load_records(board: Path = BOARD) -> list[dict]:
    """All submitted records; if two contributors submitted the same run id, the later export wins."""
    latest: dict[str, dict] = {}
    for path in sorted((board / "submissions").glob("*/*.json")):
        record = json.loads(path.read_text())
        if _variant(record):
            continue
        key = record["run_id"]
        if key not in latest or record.get("exported_at", "") >= latest[key].get("exported_at", ""):
            latest[key] = record
    return list(latest.values())


def _score(record: dict) -> float:
    """S_WB of a valid run; a quarantined run counts 0 unless a human reviewer cleared it (review.json)."""
    if not record.get("valid"):
        return 0.0
    if record.get("quarantined") and not record.get("review_cleared"):
        return 0.0
    return float(record.get("s_wb") or 0.0)


def build(board: Path = BOARD) -> dict:
    """Aggregate every record into LEADERBOARD.md, leaderboard.csv (one row per system) and runs.csv (all runs)."""
    Path(board).mkdir(parents=True, exist_ok=True)
    records = load_records(board)
    by = defaultdict(list)
    for r in records:
        by[(r["system"], r["task"])].append(r)
    systems = sorted({s for s, _ in by}, key=lambda s: (s not in ("baseline", "reference"), s))
    tasks = sorted({t for _, t in by})
    tracks = sorted({report.track_of(t) for t in tasks})
    per_task = {s: {t: mean(_score(x) for x in by[(s, t)]) if (s, t) in by else 0.0 for t in tasks} for s in systems}
    per_task_sp = {s: {t: mean(float(x.get("s_plus") or 0.0) for x in by[(s, t)]) if (s, t) in by else 0.0
                       for t in tasks} for s in systems}
    groups = report.tie_groups(per_task, systems)
    summary = {}
    for s in systems:
        track_means = {k: mean(v for t, v in per_task[s].items() if report.track_of(t) == k) for k in tracks}
        summary[s] = {"weightbench_score": report.macro_mean(per_task[s]), "ci95": report.macro_bootstrap_ci(per_task[s]),
                      "s_plus": report.macro_mean(per_task_sp[s]), "tracks": track_means, "tie_group": groups.get(s),
                      "n_runs": sum(len(by[(s, t)]) for t in tasks if (s, t) in by),
                      "missing_tasks_scored_0": [t for t in tasks if (s, t) not in by],
                      "contributors": sorted({x["contributor"] for t in tasks if (s, t) in by for x in by[(s, t)]})}
    _write_markdown(board, systems, tasks, tracks, per_task, per_task_sp, summary, len(records))
    _write_csvs(board, systems, tracks, summary, records)
    return summary


def _write_markdown(board, systems, tasks, tracks, per_task, per_task_sp, summary, n_records) -> None:
    order = sorted(systems, key=lambda s: (summary[s]["tie_group"] or 10 ** 6, -summary[s]["weightbench_score"]))
    lines = ["# WeightBench leaderboard", "",
             f"Built from {n_records} run records in `leaderboard/submissions/` (one folder per contributor). "
             "Regenerate with `weightbench-leaderboard build`; never edit this file by hand.",
             "S_WB = official formula v3 (SPEC §3). WeightBench score = mean of track means over the tasks in this "
             "table; a task a system did not run counts as 0. 95% CI: stratified bootstrap over tasks. "
             "Systems in the same group are statistical ties. Quarantined runs count as 0.", "",
             "| Group | System | " + " | ".join(tracks) + " | WeightBench score | S+ | Runs | Contributors |",
             "|---|---|" + "---|" * (len(tracks) + 4)]
    for s in order:
        d = summary[s]
        cells = [f"{d['tracks'][k]:.3f}" for k in tracks]
        lo, hi = d["ci95"]
        lines.append(f"| {d['tie_group']} | {s} | " + " | ".join(cells) + f" | **{d['weightbench_score']:.3f}** "
                     f"[{lo:.2f}, {hi:.2f}] | {d['s_plus']:.3f} | {d['n_runs']} | {', '.join(d['contributors'])} |")
    lines += ["", "## Per task (mean S_WB over seeds / samples; S+ in parentheses)", "",
              "| Task | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for t in tasks:
        cells = ["–" if t in summary[s]["missing_tasks_scored_0"] else f"{per_task[s][t]:.3f} ({per_task_sp[s][t]:.2f})"
                 for s in order]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    (board / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")


def _write_csvs(board, systems, tracks, summary, records) -> None:
    with (board / "leaderboard.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "weightbench_score", "ci95_low", "ci95_high", "s_plus", *tracks, "tie_group", "n_runs"])
        for s in systems:
            d = summary[s]
            w.writerow([s, f"{d['weightbench_score']:.4f}", f"{d['ci95'][0]:.4f}", f"{d['ci95'][1]:.4f}",
                        f"{d['s_plus']:.4f}", *[f"{d['tracks'][k]:.4f}" for k in tracks], d["tie_group"], d["n_runs"]])
    columns = ["system", "contributor", *RECORD_FIELDS, "cost_usd", "harness_commit", "host", "ledger_sha256"]
    with (board / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=lambda x: (x["task"], x["system"], x.get("seed") or 0, x.get("sample") or 0)):
            w.writerow({**r, "audit_flags": ";".join(r.get("audit_flags") or [])})


# ---------------------------------------------------------------------------------------------------------------
# Current board: rebuilt directly from runs/*/result.json + review.json (no submissions folder).
# Only valid, non-stale runs are scored; a quarantined run is scored only if review.json has cleared=true (the
# reviewed-run rule wins). Every exclusion is listed with its reason; nothing is deleted.
# Staleness is per task: a run is stale only if its task's description file under ``tasks/`` or its own task class
# (the ``class`` with ``id = "<TASK>"`` in weightbench/tracks/*.py) changed after the run started (first ledger
# entry). Track-wide edits do not make a run stale. Shared modules (SHARED) are reported per run, not as stale.
# An optional per-agent cutoff (``banner_ts``) marks Codex runs that started before it as stale.
# ---------------------------------------------------------------------------------------------------------------
SHARED = ["weightbench/scoring.py", "weightbench/runner.py", "weightbench/task.py", "weightbench/harness",
          "weightbench/manifest.py", "weightbench/knobs.py", "weightbench/server.py", "weightbench/ledger.py"]


def _git_ts(paths: list[str]) -> float | None:
    """Unix time of the last commit touching any of ``paths`` (None when there is none or no git repo)."""
    if not paths:
        return None
    out = _git(["log", "-1", "--format=%ct", "--", *paths])
    return float(out) if out else None


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _git_show(spec: str) -> str | None:
    """Text of ``<rev>:<path>`` from git (None if it does not exist)."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "show", spec], capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return None
    return out.stdout if out.returncode == 0 else None


def _task_class_source(text: str | None, task: str) -> str | None:
    """Source of the class whose ``id = "<task>"``, or None if the text does not define it."""
    if not text:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if (isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "id" for t in stmt.targets)
                    and isinstance(stmt.value, ast.Constant) and stmt.value.value == task):
                return ast.get_source_segment(text, node)
    return None


@functools.lru_cache(maxsize=None)
def _task_class_files(task: str) -> tuple[str, ...]:
    """Track modules (repo-relative) that currently define the task's class."""
    files = []
    for f in sorted(glob.glob(str(ROOT / "weightbench" / "tracks" / "*.py"))):
        if _task_class_source(Path(f).read_text(encoding="utf-8"), task) is not None:
            files.append(Path(f).relative_to(ROOT).as_posix())
    return tuple(files)


def _class_changed_since(task: str, start: float | None) -> bool:
    """True if the task's class source differs between the last commit before ``start`` and HEAD, in any module
    that defines it (or if the module has no commit before ``start``). Other classes in the same module do not count."""
    if start is None:
        return False
    for rel in _task_class_files(task):
        commits = _git(["log", "--format=%H %ct", "--", rel]).splitlines()
        if not commits:
            continue                                  # no git history for this module: nothing to compare
        before = [line.split()[0] for line in commits if int(line.split()[1]) <= start]
        if not before:
            return True                               # module first committed after the run started
        if _task_class_source(_git_show(f"{before[0]}:{rel}"), task) != _task_class_source(_git_show(f"HEAD:{rel}"), task):
            return True
    return False


def _run_start(run_dir: Path) -> float | None:
    """Start of a run: time of its first ledger entry (falls back to the result file's mtime)."""
    ledger = run_dir / "ledger.jsonl"
    if ledger.exists():
        with ledger.open(encoding="utf-8") as f:
            first = json.loads(f.readline() or "{}")
        if first.get("ts") is not None:
            return float(first["ts"])
    result = run_dir / "result.json"
    return result.stat().st_mtime if result.exists() else None


def _wired_tasks() -> set[str]:
    return {t for knob in knobs.KNOBS.values() for t in knob.wired}


def classify_run(run_dir: Path, *, clock=_git_ts, banner_ts: float | None = None,
                 class_changed=_class_changed_since) -> dict | None:
    """One row per run with ``included`` and the first exclusion ``reason`` (None if the run has no result.json)."""
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    from weightbench.review import review_status
    status = review_status(run_dir)       # human review.json wins; else an auto_review that cleared (labelled)
    cleared = status["cleared"]
    auto_cleared = cleared and status["by"] == "auto"
    task = result.get("task") or run_dir.name.split("__")[0]
    start = _run_start(run_dir)
    card_ts = clock([f"tasks/{task}.md"])
    stale_spec = start is not None and ((card_ts is not None and card_ts > start) or class_changed(task, start))
    shared_ts = clock(SHARED)
    shared_changed = start is not None and shared_ts is not None and shared_ts > start
    stale_banner = (result.get("agent") == "codex" and banner_ts is not None and start is not None
                    and start < banner_ts)
    if result.get("agent") == "cmd":
        reason = "red-team cmd run"
    elif result.get("valid") is not True:
        reason = "invalid run"
    elif result.get("quarantined") and not cleared:
        reason = ("quarantined, no approving review.json" if status["by"] == "human" or not status["auto_path"]
                  else f"quarantined, auto review {status['auto_verdict']}: needs human review")
    elif _variant(result):
        reason = "level variant, not canonical" if task in _wired_tasks() else "level not wired"
    elif stale_spec:
        reason = "stale: task description or task class changed after run start"
    elif stale_banner:
        reason = "stale: Codex run started before the configured cutoff"
    else:
        reason = None
    flags = sorted(result.get("audit_flags") or [])
    return {
        "run_id": result.get("run_id", run_dir.name), "task": task, "track": result.get("track") or "",
        "agent": result.get("agent") or "", "model": result.get("model") or "", "system": system_name(result),
        "seed": result.get("seed"), "sample": result.get("sample"),
        "s_wb": float(result.get("s_wb") or 0.0), "valid": result.get("valid") is True,
        "quarantined": bool(result.get("quarantined")), "review_cleared": cleared,
        "reviewer": status["reviewer"], "audit_flags": ";".join(flags),
        "reviewed_flags": ";".join(status["flags_reviewed"]) if cleared else "",
        "auto_cleared": auto_cleared, "auto_review": status["auto_path"] or "",
        "stale": stale_spec or stale_banner, "shared_changed": shared_changed, "included": reason is None,
        "reason": reason or "",
    }


def classify_runs(runs: Path, *, clock=_git_ts, banner_ts: float | None = None,
                  class_changed=_class_changed_since) -> list[dict]:
    rows = []
    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        row = classify_run(run_dir, clock=clock, banner_ts=banner_ts, class_changed=class_changed)
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda r: r["run_id"])


def rebuild(runs: Path, board: Path = BOARD, *, clock=_git_ts, banner_ts: float | None = None,
            class_changed=_class_changed_since) -> dict:
    """Regenerate leaderboard/LEADERBOARD.md, leaderboard.csv (system x task cells) and runs.csv (every run)."""
    Path(board).mkdir(parents=True, exist_ok=True)
    rows = classify_runs(runs, clock=clock, banner_ts=banner_ts, class_changed=class_changed)
    included = [r for r in rows if r["included"]]
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in included:
        cells[(r["system"], r["task"])].append(r)
    _write_rebuild_files(board, rows, cells)
    reasons: dict[str, int] = defaultdict(int)
    for r in rows:
        if not r["included"]:
            reasons[r["reason"]] += 1
    return {"runs": len(rows), "included": len(included), "excluded": dict(sorted(reasons.items())),
            "stale_listed": sum(1 for r in rows if r["stale"])}


def _cell_stats(group: list[dict]) -> dict:
    scores = [r["s_wb"] for r in group]
    return {"n_runs": len(group), "n_seeds": len({r["seed"] for r in group}),
            "n_samples": len({r["sample"] for r in group}), "mean": mean(scores), "lo": min(scores), "hi": max(scores)}


def _write_rebuild_files(board: Path, rows: list[dict], cells: dict) -> None:
    systems = sorted({s for s, _ in cells})
    tasks = sorted({t for _, t in cells})
    agent_of = {r["system"]: r["agent"] for r in rows}
    stats = {key: _cell_stats(group) for key, group in cells.items()}
    with (board / "leaderboard.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "agent", "task", "track", "n_runs", "n_seeds", "n_samples", "mean_s_wb", "min_s_wb",
                    "max_s_wb"])
        for (system, task), group in sorted(cells.items()):
            d = stats[(system, task)]
            w.writerow([system, agent_of[system], task, report.track_of(task), d["n_runs"], d["n_seeds"],
                        d["n_samples"], f"{d['mean']:.4f}", f"{d['lo']:.4f}", f"{d['hi']:.4f}"])
    columns = ["run_id", "system", "agent", "model", "task", "track", "seed", "sample", "s_wb", "valid", "quarantined",
               "review_cleared", "reviewer", "audit_flags", "reviewed_flags", "stale", "shared_changed", "included",
               "reason", "auto_cleared"]
    with (board / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "s_wb": f"{r['s_wb']:.4f}"})
    lines = ["# WeightBench leaderboard (rebuilt from runs/)", "",
             "Generated from `runs/*/result.json` and `runs/*/review.json` by "
             "`weightbench-leaderboard rebuild --runs <runs dir>`. Never edit by hand.", "",
             "## Rules", "",
             "A run is scored only if all hold: `valid` is true; agent is not `cmd` (red-team); it is not quarantined, "
             "or it is quarantined and its `review.json` has `cleared: true` (the reviewer is listed below), or it has no "
             "`review.json` and its latest `auto_review.json` cleared it (listed separately as auto-cleared); it is a "
             "canonical level (or a wired knob level); neither its task description (`tasks/<ID>.md`, if present) nor its "
             "own task class changed after the run started (staleness is per task: track-wide edits do not count). "
             "Shared modules (scoring, runner, task, harness, manifest, knobs, server, "
             "ledger) that changed after a run started are reported in their own section and do not exclude it.",
             "Every mean is over runs of one system on one task only. No cross-task score is computed, because "
             "systems ran different task sets. Cells show mean S_WB, with runs, seeds and samples.", "",
             "## Counts", "", "| Outcome | Runs |", "|---|---|",
             f"| included | {sum(1 for r in rows if r['included'])} |",
             f"| included, shared code changed after start (reported) | "
             f"{sum(1 for r in rows if r['included'] and r['shared_changed'])} |"]
    excluded: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["included"]:
            excluded[r["reason"]].append(r)
    for reason in sorted(excluded):
        lines.append(f"| excluded: {reason} | {len(excluded[reason])} |")
    lines += ["", "## Per system (agent)", "", "| System | Agent | Tasks | Runs | Seeds | Samples |",
              "|---|---|---|---|---|---|"]
    for s in systems:
        tks = [t for (x, t) in cells if x == s]
        grp = [r for (x, _), g in cells.items() if x == s for r in g]
        lines.append(f"| {s} | {agent_of[s]} | {len(tks)} | {len(grp)} | {len({r['seed'] for r in grp})} | "
                     f"{len({r['sample'] for r in grp})} |")
    lines += ["", "## Per task (mean S_WB; n runs, seeds, samples)", "",
              "| Task | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for t in tasks:
        row = []
        for s in systems:
            if (s, t) in stats:
                d = stats[(s, t)]
                row.append(f"{d['mean']:.3f} (n={d['n_runs']}, {d['n_seeds']}s, {d['n_samples']}k)")
            else:
                row.append("–")
        lines.append(f"| {t} | " + " | ".join(row) + " |")
    lines += ["", "## Excluded runs", ""]
    for reason in sorted(excluded):
        lines += [f"### {reason} ({len(excluded[reason])})", ""]
        lines += [f"- `{r['run_id']}`" for r in excluded[reason]] + [""]
    reviewed = [r for r in rows if r["included"] and r["review_cleared"] and not r.get("auto_cleared")]
    lines += ["## Included with an approving review.json", ""]
    lines += [f"- `{r['run_id']}`: quarantined={'yes' if r['quarantined'] else 'no'}; flags "
              f"{r['audit_flags'] or 'none'}; reviewer: {r['reviewer'] or 'unnamed'}" for r in reviewed] or ["- none"]
    auto = [r for r in rows if r["included"] and r.get("auto_cleared")]
    lines += ["", "## Included as auto-cleared (automated reviewer, no human review.json)", "",
              "Advisory clearance by `weightbench review --auto` (docs/AUTO_REVIEW.md); a human `review.json` overrides "
              "it in either direction.", ""]
    lines += [f"- `{r['run_id']}`: auto-cleared; flags {r['audit_flags'] or 'none'}; {r['reviewer']}; evidence "
              f"`runs/{r['run_id']}/{r['auto_review']}`" for r in auto] or ["- none"]
    pending = [r for r in rows if not r["included"] and r["reason"].startswith("quarantined, auto review")]
    lines += ["", "## Quarantined, auto review did not clear (needs human review)", ""]
    lines += [f"- `{r['run_id']}`: {r['reason'].split('auto review ')[1].split(':')[0]}; flags {r['audit_flags']}; "
              f"evidence `runs/{r['run_id']}/{r['auto_review']}`" for r in pending] or ["- none"]
    shared = [r for r in rows if r["included"] and r["shared_changed"]]
    lines += ["", "## Included, shared code changed after the run started (reported, not stale)", ""]
    lines += [f"- `{r['run_id']}`" for r in shared] or ["- none"]
    (board / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="weightbench-leaderboard", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="runs/*/result.json -> leaderboard/submissions/<contributor>/")
    ex.add_argument("--contributor", required=True)
    ex.add_argument("--runs", default=str(ROOT / "runs"))
    sub.add_parser("build", help="legacy: aggregate all submissions into LEADERBOARD.md and CSVs")
    rb = sub.add_parser("rebuild", help="regenerate the current board from runs/ and review.json")
    rb.add_argument("--runs", default=str(ROOT / "runs"))
    a = ap.parse_args(argv)
    if a.cmd == "export":
        written = export(Path(a.runs), a.contributor)
        print(f"exported {len(written)} runs to leaderboard/submissions/{a.contributor}/")
    elif a.cmd == "rebuild":
        out = rebuild(Path(a.runs))
        print(f"rebuilt leaderboard: {out['included']} included of {out['runs']} runs; excluded {out['excluded']}; "
              f"stale runs listed: {out['stale_listed']} -> leaderboard/LEADERBOARD.md")
    else:
        summary = build()
        print(f"built leaderboard for {len(summary)} systems -> leaderboard/LEADERBOARD.md")


if __name__ == "__main__":
    main()
