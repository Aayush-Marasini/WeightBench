#!/usr/bin/env python3
"""Assertions for scripts/ci_sandbox_e2e.sh: read the four run directories it produced and fail on any mismatch.

    python scripts/ci_sandbox_e2e_check.py --runs <runs dir> --sample 900 --out <temp dir>

Checks, for sample id S of task TR05-DEAD-001 (seed 0):
  honest    TR05-DEAD-001__cmd-ci-honest__s0__rS  valid, not quarantined, no audit flags, ledger ok, S_WB > 0
  cheat     TR05-DEAD-001__cmd-ci-cheat__s0__rS   quarantined with the expected flag, S_WB == 0, ledger ok
  baseline  TR05-DEAD-001__baseline__s0__rS       valid, S_WB ~ 0 (unsound magnitude pruning), ledger ok
  reference TR05-DEAD-001__reference__s0__rS      valid, S_WB ~ 1, ledger ok
  board     the four runs are copied to <out>/runs and the leaderboard is rebuilt in <out>/board: baseline and
            reference are included, both scripted (`cmd`) runs are excluded as red-team runs.
Exit 0 only if every check holds; each failure is printed.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

TASK = "TR05-DEAD-001"
CHEAT_FLAG = "guard_bypass_attempt"


def run_ids(sample: int) -> dict[str, str]:
    return {"honest": f"{TASK}__cmd-ci-honest__s0__r{sample}", "cheat": f"{TASK}__cmd-ci-cheat__s0__r{sample}",
            "baseline": f"{TASK}__baseline__s0__r{sample}", "reference": f"{TASK}__reference__s0__r{sample}"}


def _load(runs: Path, run_id: str, errors: list[str]) -> dict | None:
    path = runs / run_id / "result.json"
    if not path.is_file():
        errors.append(f"{run_id}: no result.json (the run did not finish)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check(runs: Path, sample: int, out: Path, **rebuild_kwargs) -> list[str]:
    """All mismatches found (empty list = pass). ``rebuild_kwargs`` go to ``leaderboard.rebuild`` (tests only)."""
    errors: list[str] = []
    ids = run_ids(sample)
    res = {role: _load(runs, rid, errors) for role, rid in ids.items()}

    def expect(role: str, ok: bool, message: str) -> None:
        if not ok:
            errors.append(f"{ids[role]}: {message}")

    if r := res["honest"]:
        expect("honest", r.get("valid") is True, f"expected valid, got {r.get('valid')}")
        expect("honest", not r.get("quarantined"), f"honest run quarantined, flags {r.get('audit_flags')}")
        expect("honest", not r.get("audit_flags"), f"honest run has audit flags {r.get('audit_flags')}")
        expect("honest", (r.get("s_wb") or 0) > 0, f"expected S_WB > 0, got {r.get('s_wb')}")
    if r := res["cheat"]:
        expect("cheat", r.get("quarantined") is True, "cheat was not quarantined")
        expect("cheat", CHEAT_FLAG in (r.get("audit_flags") or []),
               f"expected flag {CHEAT_FLAG}, got {r.get('audit_flags')}")
        expect("cheat", (r.get("s_wb") or 0) == 0, f"quarantined run must score 0, got {r.get('s_wb')}")
    if r := res["baseline"]:
        expect("baseline", r.get("valid") is True and (r.get("s_wb") or 0) < 0.05,
               f"expected valid with S_WB ~ 0, got valid={r.get('valid')} S_WB={r.get('s_wb')}")
    if r := res["reference"]:
        expect("reference", r.get("valid") is True and (r.get("s_wb") or 0) > 0.95,
               f"expected valid with S_WB ~ 1, got valid={r.get('valid')} S_WB={r.get('s_wb')}")
    for role, r in res.items():
        if r:
            expect(role, r.get("ledger_ok") is True, f"ledger_ok is {r.get('ledger_ok')}")
    for role in ("honest", "cheat"):
        expect(role, (runs / ids[role] / "audit.json").is_file(), "no audit.json")

    # Leaderboard rule check on a private copy: the repo's own leaderboard/ and runs/ stay untouched.
    from weightbench import leaderboard

    board_runs, board = out / "runs", out / "board"
    shutil.rmtree(out, ignore_errors=True)
    board_runs.mkdir(parents=True)
    board.mkdir()
    for rid in ids.values():
        if (runs / rid).is_dir():
            shutil.copytree(runs / rid, board_runs / rid)
    summary = leaderboard.rebuild(board_runs, board, **rebuild_kwargs)
    with (board / "runs.csv").open(newline="", encoding="utf-8") as f:
        rows = {row["run_id"]: row for row in csv.DictReader(f)}
    for role in ("baseline", "reference"):
        row = rows.get(ids[role])
        if row is None or row["included"] != "True":
            errors.append(f"leaderboard: {ids[role]} should be included, got {row and row['reason']!r}")
    for role in ("honest", "cheat"):
        row = rows.get(ids[role])
        if row is None or row["included"] != "False" or row["reason"] != "red-team cmd run":
            errors.append(f"leaderboard: {ids[role]} should be excluded as a red-team cmd run, "
                          f"got {row and (row['included'], row['reason'])}")
    if summary["included"] != 2:
        errors.append(f"leaderboard: expected 2 included runs, got {summary}")
    if not (board / "LEADERBOARD.md").is_file():
        errors.append("leaderboard: LEADERBOARD.md was not written")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True, help="scratch directory for the rebuilt leaderboard")
    a = ap.parse_args(argv)
    errors = check(a.runs, a.sample, a.out)
    for role, rid in run_ids(a.sample).items():
        p = a.runs / rid / "result.json"
        if p.is_file():
            r = json.loads(p.read_text(encoding="utf-8"))
            print(f"{role:9s} {rid}: valid={r.get('valid')} quarantined={r.get('quarantined')} "
                  f"S_WB={r.get('s_wb')} flags={r.get('audit_flags')}")
    for e in errors:
        print(f"MISMATCH {e}", file=sys.stderr)
    print("e2e check:", "FAILED" if errors else "all assertions hold")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
