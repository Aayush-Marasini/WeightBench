"""Trace saturation integrity with real local HTTP/ledger flow and a stubbed agent."""
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from weightbench import runner, server, verify
from weightbench.harness import agents
from weightbench.ledger import Ledger
from weightbench.task import Budget, Evaluation, Task


class StaticTask(Task):
    id, track, tier = "TR99-LIMIT-TEST", "TR01", "foundational"
    budget = Budget(n_forward=0, n_probe=0)
    def prompt(self, art_dir):
        return "Protocol fixture only; no subject inference."
    def evaluate(self, art_dir, submission):
        return Evaluation(1.)              # protocol fixture only; no subject execution


def post(url, endpoint, body, token=None):
    request = urllib.request.Request(url + endpoint, json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **({"X-WB-Token": token} if token else {})})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_trace_limit_marker_is_once_only_under_concurrent_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MAX_TRACE_EVENTS", 1)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    state = server.RunState(StaticTask(), tmp_path, tmp_path, ledger, tmp_path)
    assert state.trace({"pid": 1})[0] == 200
    with ThreadPoolExecutor(4) as pool:
        codes = list(pool.map(lambda _: state.trace({"pid": 1})[0], range(20)))
    assert codes == [429] * 20 and state.n_trace_events == 1
    markers = list(ledger.entries("runtime_trace_limit"))
    assert len(markers) == 1 and markers[0]["payload"] == {"cap": 1, "accepted": 1}
    assert len(list(ledger.entries("runtime_trace"))) == 1 and ledger.verify()
    before = len(list(ledger.entries()))
    assert state.trace({"pid": []})[0] == 400
    assert len(list(ledger.entries())) == before


def test_runner_quarantines_trace_saturation_without_subject_or_agent_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MAX_TRACE_EVENTS", 1)
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "WORK", tmp_path / "work")
    monkeypatch.setattr(runner, "GUARDS", tmp_path / "guards")
    monkeypatch.setattr(runner, "get_task", lambda *args: StaticTask())
    monkeypatch.setattr(runner, "nondefault_level", lambda *args: None)
    artifact = tmp_path / "art"
    (artifact / "public").mkdir(parents=True)
    monkeypatch.setattr(runner, "build", lambda *args, **kwargs: artifact)
    def launch(spec, model, workdir, transcript, wall_clock, turns, guard, ports, flop_cap, url, token):
        transcript.write_text(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}}))
        assert post(url, "/trace", {"pid": 123}, token) == 200
        assert post(url, "/trace", {"pid": 123}, token) == 429
        assert post(url, "/trace", {"pid": 123}, token) == 429
        assert post(url, "/submit", {"type": "report", "report": {}}, token) == 200
        return 0
    monkeypatch.setattr(runner, "_launch_agent", launch)
    spec = agents.get("cmd", "unused", local_ports=[8000])
    result = runner.run(StaticTask.id, "cmd", "fixture", agent_spec=spec)
    assert result["quarantined"] and result["audit_flags"] == ["trace_limit_exceeded"]
    directory = runner.RUNS / result["run_id"]
    ledger = Ledger(directory / "ledger.jsonl")
    assert ledger.verify() and len(list(ledger.entries("runtime_trace_limit"))) == 1
    assert next(ledger.entries("agent_exit"))["payload"]["quarantined"]


def test_verifier_rejects_limit_marker_with_clean_quarantine_metadata(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("runtime_trace_limit", {"cap": 1, "accepted": 1})
    result = {"agent": "cmd", "quarantined": False}
    (tmp_path / "audit.json").write_text(json.dumps({"quarantined": False}))
    report = verify.RunReport(str(tmp_path))
    verify.check_audit(report, tmp_path, result, ledger)
    assert not report.ok and "event limit" in report.checks[0].detail
    (tmp_path / "review.json").write_text(json.dumps({"cleared": True}))
    reviewed = verify.RunReport(str(tmp_path))
    verify.check_audit(reviewed, tmp_path, result, ledger)
    assert reviewed.ok
