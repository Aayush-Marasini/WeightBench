"""weightbench-verify: a real (toy) run passes; each kind of tampering fails the right check."""
import json

import pytest

from weightbench import leaderboard, runner, scoring
from weightbench import verify as V
from weightbench.ledger import Ledger


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A real SMOKE-000 reference run (8x8 SVD), produced by runner.run into tmp dirs."""
    monkeypatch.setattr(runner, "ART", tmp_path / "artifacts")
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    res = runner.run("SMOKE-000", "reference")
    return tmp_path / "runs" / res["run_id"]


def _status(rep, name):
    return {c.name: c.status for c in rep.checks}[name]


def _edit_json(path, **changes):
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def test_clean_run_passes(run_dir):
    rep = V.verify_run(run_dir)
    assert rep.ok, V.format_report(rep)
    for name in ("ledger_chain", "formula_version", "scores", "ledger_structure"):
        assert _status(rep, name) == V.PASS
    assert _status(rep, "audit") == V.SKIP           # reference run: no agent audit


def test_tampered_ledger_line_fails_chain(run_dir):
    path = run_dir / "ledger.jsonl"
    lines = path.read_text().splitlines()
    ev = json.loads(lines[-1])
    ev["payload"]["s_plus"] = 0.5
    lines[-1] = json.dumps(ev)
    path.write_text("\n".join(lines) + "\n")
    rep = V.verify_run(run_dir)
    assert not rep.ok and _status(rep, "ledger_chain") == V.FAIL


def test_edited_score_fails_scores(run_dir):
    _edit_json(run_dir / "result.json", s_wb=0.123)
    rep = V.verify_run(run_dir)
    assert _status(rep, "scores") == V.FAIL and _status(rep, "ledger_chain") == V.PASS


def test_formula_version_mismatch_is_a_warning(run_dir):
    _edit_json(run_dir / "result.json", formula_version=scoring.FORMULA_VERSION - 1, s_wb=0.5)
    rep = V.verify_run(run_dir)
    assert _status(rep, "formula_version") == V.WARN and _status(rep, "scores") == V.PASS and rep.ok


def test_quarantine_flip_fails_audit(run_dir):
    (run_dir / "audit.json").write_text(json.dumps({"quarantined": True, "flags": ["egress_denied"]}))
    rep = V.verify_run(run_dir)
    assert _status(rep, "audit") == V.FAIL
    _edit_json(run_dir / "result.json", quarantined=True)
    assert _status(V.verify_run(run_dir), "audit") == V.PASS


def test_second_evaluation_fails_structure(run_dir):
    Ledger(run_dir / "ledger.jsonl").append("evaluation", {"s_plus": 1.0, "d_col": 0.0, "valid": True})
    rep = V.verify_run(run_dir)
    assert _status(rep, "ledger_chain") == V.PASS and _status(rep, "ledger_structure") == V.FAIL


def test_no_submission_case_passes(tmp_path):
    from weightbench.task import REGISTRY, load_all

    load_all()
    task = REGISTRY["SMOKE-000"]()
    d = tmp_path / "SMOKE-000__x__s0__r0"
    led = Ledger(d / "ledger.jsonl")
    led.append("run_start", {"task": "SMOKE-000", "agent": "claude", "model": "m", "sample": 0, "seed": 0})
    led.append("agent_exit", {"quarantined": False})
    ev = runner._evaluate_and_log(task, tmp_path, None, led)
    res = {"task": "SMOKE-000", "agent": "claude", "model": "m", "sample": 0, "seed": 0, "quarantined": False,
           **runner.score_from_ledger(task, led, ev), "details": ev.details}
    (d / "result.json").write_text(json.dumps(res))
    rep = V.verify_run(d)
    assert rep.ok, V.format_report(rep)
    assert "no-submission" in [c for c in rep.checks if c.name == "ledger_structure"][0].detail


def test_leaderboard_record(run_dir, tmp_path):
    record = leaderboard.record_from_run(run_dir, "tester")
    assert _status(V.verify_run(run_dir, record), "record") == V.PASS
    assert _status(V.verify_run(run_dir, {**record, "s_wb": 0.99}), "record") == V.FAIL
    assert _status(V.verify_run(run_dir, {**record, "ledger_sha256": "0" * 64}), "record") == V.FAIL


def test_cli_exit_codes_and_all(run_dir, capsys):
    with pytest.raises(SystemExit) as e:
        V.main([str(run_dir), "--json"])
    assert e.value.code == 0 and json.loads(capsys.readouterr().out)["ok"] is True
    _edit_json(run_dir / "result.json", valid=False)
    with pytest.raises(SystemExit) as e:
        V.main(["--all", str(run_dir.parent)])
    assert e.value.code == 1 and "0/1 runs pass" in capsys.readouterr().out
