"""The end-of-bench console table follows the leaderboard rule: invalid and uncleared-quarantined runs are excluded."""
import json

import pytest

from weightbench import report


def _row(run_id, system, task, s_plus, *, valid=True, flags=(), agent=None, **extra):
    return {"run_id": run_id, "system": system, "agent": agent or system, "task": task, "valid": valid,
            "s_plus": s_plus, "s_wb": s_plus, "d_col": 0.0, "e_comp": 0, "e_budget": 0, "edit_budget_use": 0.0,
            "formula_version": 3, "audit_flags": list(flags), "usage": {}, **extra}


def test_exclusion_reasons_match_the_leaderboard_rule():
    assert report.exclusion_reason(_row("a", "m", "T", 1.0)) is None
    assert report.exclusion_reason(_row("a", "m", "T", 0.0, valid=False)) == "invalid run"
    assert report.exclusion_reason(_row("a", "m", "T", 0.0, valid=False, agent_auth_failed="x")) \
        == "invalid run (agent_auth_failed)"
    assert report.exclusion_reason(_row("a", "m", "T", 0.0, flags=["QUARANTINED"])) \
        == "quarantined, no approving review.json"
    assert report.exclusion_reason(_row("a", "cmd", "T", 1.0, agent="cmd")) == "red-team cmd run"


def test_invalid_zero_does_not_drag_the_mean_and_exclusions_are_counted(tmp_path, monkeypatch, capsys):
    rows = [_row("T1__m__s0__r0", "m", "TR04-A", 1.0),
            _row("T1__m__s0__r1", "m", "TR04-A", 0.0, valid=False),
            _row("T2__m__s0__r0", "m", "TR04-B", 0.0, flags=["QUARANTINED"]),
            _row("T1__base__s0__r0", "baseline", "TR04-A", 0.0), _row("T2__base__s0__r0", "baseline", "TR04-B", 0.0)]
    monkeypatch.setattr(report, "load", lambda: rows)
    monkeypatch.setattr(report, "OUT", tmp_path)
    report.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["systems"]["m"]["per_task"]["TR04-A"] == pytest.approx(1.0)   # not 0.5: the invalid 0.0 is out
    assert summary["systems"]["m"]["n_runs"] == 1
    assert summary["excluded_runs"] == [{"run_id": "T1__m__s0__r1", "reason": "invalid run"},
                                        {"run_id": "T2__m__s0__r0", "reason": "quarantined, no approving review.json"}]
    out = capsys.readouterr().out
    assert "Excluded from all means (leaderboard rule): 2 run(s): 1 invalid run, 1 quarantined, no approving" in out
    assert "T1__m__s0__r1" in out and "T2__m__s0__r0" in out
    assert "Excluded" in (tmp_path / "RESULTS.md").read_text(encoding="utf-8")


def test_no_excluded_line_when_every_run_counts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(report, "load", lambda: [_row("r", "m", "TR04-A", 0.5)])
    monkeypatch.setattr(report, "OUT", tmp_path)
    report.main()
    assert "Excluded" not in capsys.readouterr().out
    assert json.loads((tmp_path / "summary.json").read_text())["excluded_runs"] == []
