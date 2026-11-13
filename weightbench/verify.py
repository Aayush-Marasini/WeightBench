"""Verify a finished run (and optionally its leaderboard record) from the files alone.

    weightbench-verify runs/<run_id>                                   # one run directory
    weightbench-verify runs/<run_id> --record leaderboard/submissions/<who>/<run_id>.json
    weightbench-verify --record leaderboard/submissions/<who>/<run_id>.json   # run dir = runs/<record.run_id>
    weightbench-verify --all runs/                                     # every run directory, summary table
    weightbench-verify ... --json                                      # machine-readable

Pure reading: nothing is re-run. Checks: the ledger hash chain; result.json scores equal what
``runner.score_from_ledger`` recomputes from the ledger with the current formula; the ledger has a run_start, a
submission (or the documented no-submission case) and exactly the one evaluation the result reports; quarantine
status agrees with audit.json and the ledger's agent_exit entry; a leaderboard record's hashes and scores match.
Exit code 0 if no check FAILs (WARN / SKIP do not fail), 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from weightbench.ledger import Ledger
from weightbench.manifest import sha256_file

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
TOL = 1e-9
SCORE_FIELDS = ("s_wb", "s_plus", "valid", "n_forward", "n_probe")
NO_SUBMISSION_REASON = "no submission"

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class RunReport:
    run_dir: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    def to_dict(self) -> dict[str, Any]:
        return {"run_dir": self.run_dir, "ok": self.ok, "checks": [asdict(c) for c in self.checks]}


# ---- helpers -------------------------------------------------------------------------------------------------------

def same_value(a: Any, b: Any, tol: float = TOL) -> bool:
    """Equality with a float tolerance (bools compared exactly, numbers by |a - b| <= tol)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_task(result: dict[str, Any]):
    from weightbench.task import REGISTRY, load_all

    load_all()
    cls = REGISTRY.get(result.get("task"))
    if cls is None:
        return None
    return cls().at_level(result.get("level"))


# ---- checks --------------------------------------------------------------------------------------------------------

def check_ledger_chain(rep: RunReport, led: Ledger | None) -> None:
    if led is None:
        rep.add("ledger_chain", FAIL, "ledger.jsonl missing")
        return
    try:
        intact = led.verify()
    except (KeyError, json.JSONDecodeError) as e:
        rep.add("ledger_chain", FAIL, f"unreadable ledger entry: {type(e).__name__}: {e}")
        return
    n = sum(1 for _ in led.entries())
    if intact:
        rep.add("ledger_chain", PASS, f"{n} entries, hash chain intact")
    else:
        rep.add("ledger_chain", FAIL, "hash chain broken (an entry was edited, removed or reordered)")


def check_formula_version(rep: RunReport, result: dict[str, Any]) -> bool:
    """WARN (not tampering) if result.json was scored with another formula version. Returns True if they match."""
    from weightbench import scoring

    stored = result.get("formula_version")
    if stored == scoring.FORMULA_VERSION:
        rep.add("formula_version", PASS, f"v{stored}")
        return True
    rep.add("formula_version", WARN, f"result.json scored with v{stored}, current formula is "
            f"v{scoring.FORMULA_VERSION}; s_wb not compared (run `weightbench rescore`)")
    return False


def check_scores(rep: RunReport, result: dict[str, Any], led: Ledger, same_formula: bool) -> None:
    from weightbench.runner import score_from_ledger

    task = _load_task(result)
    if task is None:
        rep.add("scores", FAIL, f"unknown task {result.get('task')!r}: cannot recompute")
        return
    if not any(True for _ in led.entries("evaluation")):
        rep.add("scores", FAIL, "ledger has no evaluation entry")
        return
    recomputed = score_from_ledger(task, led, None)
    fields = [f for f in SCORE_FIELDS if same_formula or f != "s_wb"]
    bad = [f"{f}: result={result.get(f)!r} ledger={recomputed.get(f)!r}"
           for f in fields if not same_value(result.get(f), recomputed.get(f))]
    if bad:
        rep.add("scores", FAIL, "result.json differs from the ledger: " + "; ".join(bad))
    else:
        rep.add("scores", PASS, "recomputed " + ", ".join(f"{f}={recomputed.get(f)}" for f in fields))


def _structure_problems(result: dict[str, Any], entries: list[dict[str, Any]]) -> list[str]:
    kinds = [e["kind"] for e in entries]
    problems = []
    if not kinds or kinds[0] != "run_start":
        problems.append("first entry is not run_start")
    if kinds.count("run_start") != 1:
        problems.append(f"{kinds.count('run_start')} run_start entries (expected 1)")
    starts = [e for e in entries if e["kind"] == "run_start"]
    if starts:
        start = starts[0]["payload"]
        condition = result.get("condition", "full")
        if start.get("condition", "full") != condition:
            problems.append("run_start access condition differs from result.json")
        from weightbench import conditions
        try:
            conditions.validate(condition)
            if condition != "full":
                inputs = [e['payload'] for e in entries if e['kind'] == 'access_inputs']
                if len(inputs) != 1 or not isinstance(inputs[0], dict) or inputs[0].get('condition') != condition:
                    problems.append("missing or inconsistent access-input evidence")
                else:
                    files = inputs[0].get('files')
                    subject = inputs[0].get('subject_weights')
                    source = inputs[0].get('harness_source_sha256')
                    if source is not None and result.get('harness_source_sha256') != source:
                        problems.append('harness-source provenance differs from result.json')
                    plan_hashes = [start.get('access_plan_sha256'), inputs[0].get('access_plan_sha256'),
                                   result.get('access_plan_sha256')]
                    if any(h is not None for h in plan_hashes) and len(set(plan_hashes)) != 1:
                        problems.append('access-plan provenance differs across run records')
                    if not isinstance(files, dict) or not all(isinstance(files.get(k), dict) for k in ('metadata', 'weights')):
                        problems.append('malformed access-file evidence')
                    elif condition == 'behavior_only' and files['weights']:
                        problems.append("behavior-only run received weight files")
                    elif not isinstance(subject, dict) or (condition == 'weights_behavior' and files['weights'] != subject):
                        problems.append('inconsistent subject-weight evidence')
                    if inputs[0].get('local_model_execution') is not False:
                        problems.append('matched run does not prohibit local model execution')
                allowed = conditions.profile(result['task']).behavior_probes
                if any(e['kind'] == 'probe' and e['payload'].get('kind') not in allowed for e in entries):
                    problems.append("matched run used an internal or undeclared probe")
        except (ValueError, KeyError, TypeError):
            problems.append("invalid access condition/profile")
        for key in ("task", "agent", "model", "seed", "sample", "variant"):   # variant: absent on both in older runs
            if start.get(key) != result.get(key):
                problems.append(f"run_start {key}={start.get(key)!r} but result.json {key}={result.get(key)!r}")
    evals = [i for i, k in enumerate(kinds) if k == "evaluation"]
    if len(evals) != 1:
        problems.append(f"{len(evals)} evaluation entries (expected exactly 1)")
        return problems
    ev_index = evals[0]
    if ev_index != len(kinds) - 1:
        problems.append("entries after the evaluation")
    ev = entries[ev_index]["payload"]
    subs = [i for i, k in enumerate(kinds) if k == "submission"]
    reason = (ev.get("details") or {}).get("reason")
    if not subs and not (reason == NO_SUBMISSION_REASON and not ev.get("valid")):
        problems.append("no submission entry and the evaluation is not the no-submission case")
    if subs and subs[-1] > ev_index:
        problems.append("submission after the evaluation")
    if "details" in result and canonical(result["details"]) != canonical(ev.get("details")):
        problems.append("result.json details differ from the ledger's evaluation")
    if not same_value(result.get("d_col"), ev.get("d_col")):
        problems.append(f"d_col: result={result.get('d_col')!r} ledger={ev.get('d_col')!r}")
    return problems


def check_ledger_structure(rep: RunReport, result: dict[str, Any], led: Ledger) -> None:
    entries = list(led.entries())
    problems = _structure_problems(result, entries)
    if problems:
        rep.add("ledger_structure", FAIL, "; ".join(problems))
        return
    n_sub = sum(1 for e in entries if e["kind"] == "submission")
    what = f"{n_sub} submission(s)" if n_sub else "no-submission case"
    rep.add("ledger_structure", PASS, f"run_start, {what}, 1 evaluation")


def check_audit(rep: RunReport, run_dir: Path, result: dict[str, Any], led: Ledger) -> None:
    claimed = bool(result.get("quarantined"))
    problems = []
    audit = _read_json(run_dir / "audit.json")
    review = _read_json(run_dir / "review.json") or {}
    if any(led.entries("runtime_trace_limit")) and not claimed and not review.get("cleared"):
        problems.append("runtime trace event limit reached without quarantine or explicit review")
    if audit is not None and bool(audit.get("quarantined")) != claimed:
        problems.append(f"result.json quarantined={claimed} but audit.json quarantined={bool(audit.get('quarantined'))}")
    exits = [e["payload"] for e in led.entries("agent_exit")]
    for flags in [(audit or {}).get("flags"), *(e.get("audit_flags") for e in exits)]:
        if (isinstance(flags, list) and any(f in ("transcript_malformed", "egress_malformed") for f in flags)
                and not claimed and not review.get("cleared")):
            problems.append("malformed audit evidence without quarantine or explicit review")
            break
    if exits and bool(exits[-1].get("quarantined")) != claimed:
        problems.append(f"result.json quarantined={claimed} but ledger agent_exit quarantined="
                        f"{bool(exits[-1].get('quarantined'))}")
    if problems:
        rep.add("audit", FAIL, "; ".join(problems))
        return
    if audit is None and not exits:
        is_floor = result.get("agent") in ("baseline", "reference")
        status = SKIP if is_floor and not claimed else WARN
        rep.add("audit", status, "no audit.json / agent_exit (expected only for baseline / reference runs)")
        return
    cleared = " (cleared by review.json)" if review.get("cleared") else ""
    rep.add("audit", PASS, f"quarantined={claimed}{cleared}")


def check_record(rep: RunReport, run_dir: Path, result: dict[str, Any], record: dict[str, Any]) -> None:
    from weightbench.leaderboard import RECORD_FIELDS

    problems = []
    for key, name in (("ledger_sha256", "ledger.jsonl"), ("transcript_sha256", "transcript.jsonl")):
        actual = sha256_file(run_dir / name)
        if record.get(key) != actual:
            problems.append(f"{key}: record={record.get(key)} file={actual}")
    for key in RECORD_FIELDS:
        if key in record and not same_value(record.get(key), result.get(key)):
            problems.append(f"{key}: record={record.get(key)!r} result={result.get(key)!r}")
    if problems:
        rep.add("record", FAIL, "; ".join(problems))
    else:
        rep.add("record", PASS, f"hashes and {len(RECORD_FIELDS)} fields match result.json")


# ---- driver --------------------------------------------------------------------------------------------------------

def verify_run(run_dir: Path, record: dict[str, Any] | None = None) -> RunReport:
    rep = RunReport(str(run_dir))
    result = _read_json(run_dir / "result.json")
    ledger_path = run_dir / "ledger.jsonl"
    led = Ledger(ledger_path) if ledger_path.exists() else None
    check_ledger_chain(rep, led)
    if result is None:
        rep.add("result", FAIL, "result.json missing")
        return rep
    if led is None:
        return rep
    same_formula = check_formula_version(rep, result)
    check_scores(rep, result, led, same_formula)
    check_ledger_structure(rep, result, led)
    check_audit(rep, run_dir, result, led)
    if record is not None:
        check_record(rep, run_dir, result, record)
    return rep


def verify_all(runs: Path) -> list[RunReport]:
    dirs = sorted(p for p in runs.iterdir() if p.is_dir() and (p / "result.json").exists())
    return [verify_run(d) for d in dirs]


def format_report(rep: RunReport) -> str:
    lines = [f"Verifying {rep.run_dir}"]
    for c in rep.checks:
        lines.append(f"  {c.status:<4}  {c.name:<16} {c.detail}")
    lines.append(f"RESULT: {'PASS' if rep.ok else 'FAIL'}")
    return "\n".join(lines)


def format_summary(reports: list[RunReport]) -> str:
    width = max([len(Path(r.run_dir).name) for r in reports] + [6])
    lines = [f"{'run':<{width}}  status  failed / warned checks", "-" * (width + 32)]
    for r in reports:
        failed = [c.name for c in r.checks if c.status == FAIL]
        warned = [f"{c.name}(warn)" for c in r.checks if c.status == WARN]
        lines.append(f"{Path(r.run_dir).name:<{width}}  {'PASS' if r.ok else 'FAIL':<6}  "
                     f"{', '.join(failed + warned) or '-'}")
    n_ok = sum(r.ok for r in reports)
    lines.append(f"{n_ok}/{len(reports)} runs pass")
    return "\n".join(lines)


def _resolve_run_dir(run_dir: str | None, record: dict[str, Any] | None) -> Path:
    if run_dir:
        return Path(run_dir)
    if record is None or not record.get("run_id"):
        raise SystemExit("give a run directory, --record, or --all")
    return RUNS / record["run_id"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="weightbench-verify", description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", nargs="?", help="runs/<run_id>")
    ap.add_argument("--record", help="leaderboard/submissions/<who>/<run_id>.json to check against the run")
    ap.add_argument("--all", metavar="RUNS", help="verify every run directory under RUNS")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)
    if a.all:
        reports = verify_all(Path(a.all))
        print(json.dumps([r.to_dict() for r in reports], indent=1) if a.json else format_summary(reports))
        sys.exit(0 if all(r.ok for r in reports) else 1)
    record = _read_json(Path(a.record)) if a.record else None
    if a.record and record is None:
        raise SystemExit(f"record not found: {a.record}")
    rep = verify_run(_resolve_run_dir(a.run_dir, record), record)
    print(json.dumps(rep.to_dict(), indent=1) if a.json else format_report(rep))
    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
