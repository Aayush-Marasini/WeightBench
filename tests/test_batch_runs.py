"""Batch CLI control-flow correctness; no subject or LLM benchmark execution."""
import json
from types import SimpleNamespace

import pytest

from weightbench import report, runner


@pytest.fixture
def batch(tmp_path, monkeypatch):
    state = SimpleNamespace(builds=[], calls=[], reports=[], selected=[], fail_task=None)
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "load_all", lambda: {})
    monkeypatch.setattr(runner, "needs_agent_sandbox", lambda args: False)
    monkeypatch.setattr(runner, "agent_spec_from_args", lambda args: SimpleNamespace(name="test-agent"))
    monkeypatch.setattr(runner.backends, "detect", lambda: SimpleNamespace(name="test"))
    def select(*args):
        state.selected.append(args)
        return ["TR01-TEST-001", "TR04-TEST-001"]
    monkeypatch.setattr(runner, "select_tasks", select)
    monkeypatch.setattr(runner, "build", lambda *args: state.builds.append(args))
    monkeypatch.setattr(report, "main", lambda: state.reports.append(True))
    def run(tid, agent, model=None, sample=0, seed=0, max_turns=200, *args):
        state.calls.append((tid, agent, model, sample, seed, max_turns))
        if tid == state.fail_task:
            raise RuntimeError("planted batch failure")
        return {"run_id": runner.make_run_id(tid, runner.run_tag(agent, model), seed, sample),
                "s_wb": 1., "s_plus": 1., "valid": True}
    monkeypatch.setattr(runner, "run", run)
    return state


def test_bench_cli_honors_starting_sample_and_floor_sample(batch):
    runner.main(["bench", "--model", "model", "--sample", "7", "--samples", "2", "--with-floors"])
    assert len(batch.calls) == 8 and len(batch.builds) == 2 and batch.reports == [True]
    for tid in ("TR01-TEST-001", "TR04-TEST-001"):
        jobs = [call for call in batch.calls if call[0] == tid]
        assert [(j[1], j[3]) for j in jobs] == [("claude", 7), ("claude", 8), ("baseline", 7), ("reference", 7)]


def test_sweep_cli_honors_sample_range_task_filters_and_turn_limit(batch):
    runner.main(["sweep", "--models", "a,b", "--sample", "7", "--samples", "2", "--jobs", "2",
                 "--tracks", "TR04", "--tiers", "foundational", "--max-turns", "9"])
    assert batch.selected == [("*", "TR04", "foundational")]
    assert len(batch.calls) == 12 and len(batch.builds) == 2
    assert all(call[5] == 9 for call in batch.calls)
    for tid in ("TR01-TEST-001", "TR04-TEST-001"):
        assert sorted((call[2], call[3]) for call in batch.calls if call[0] == tid and call[1] == "claude") == [
            ("a", 7), ("a", 8), ("b", 7), ("b", 8)]
        assert sorted((call[1], call[3]) for call in batch.calls if call[0] == tid and call[1] != "claude") == [
            ("baseline", 7), ("reference", 7)]


@pytest.mark.parametrize("command", ["bench", "sweep"])
def test_batch_preflights_all_interrupted_ids_before_build_or_run(batch, command):
    # A collision in the final task must stop earlier fresh tasks from starting.
    tag = "model" if command == "bench" else "baseline"
    directory = runner.RUNS / runner.make_run_id("TR04-TEST-001", tag, 0, 7)
    directory.mkdir(parents=True)
    (directory / "ledger.jsonl").write_bytes(b"retained partial evidence")
    with pytest.raises(FileExistsError, match="fresh --sample"):
        runner.main([command, "--model", "model", "--models", "model", "--sample", "7"])
    assert not batch.builds and not batch.calls
    assert (directory / "ledger.jsonl").read_bytes() == b"retained partial evidence"


@pytest.mark.parametrize("command", ["bench", "sweep"])
def test_batch_finishes_independent_runs_but_exits_nonzero_on_failure(batch, command, capsys):
    batch.fail_task = "TR01-TEST-001"
    with pytest.raises(SystemExit) as failed:
        runner.main([command, "--model", "model", "--models", "model"])
    assert failed.value.code == 1
    assert any(call[0] == "TR04-TEST-001" for call in batch.calls)
    assert "planted batch failure" in capsys.readouterr().err


@pytest.mark.parametrize("command,args", [("bench", ["--sample", "-1"]),
    ("bench", ["--samples", "0"]), ("sweep", ["--samples", "-1"]), ("sweep", ["--jobs", "0"])])
def test_invalid_batch_ranges_do_not_start_work(batch, command, args):
    with pytest.raises(SystemExit):
        runner.main([command, "--model", "model", *args])
    assert not batch.builds and not batch.calls


def test_batch_rejects_colliding_normalized_system_ids(batch):
    with pytest.raises(ValueError, match="duplicate run ID"):
        runner._pending_batch_runs([("TR01-TEST-001", "cmd", "org/model", 0),
                                    ("TR01-TEST-001", "cmd", "org_model", 0)], seed=0)
    assert not batch.builds and not batch.calls


@pytest.fixture
def cached_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ART", tmp_path / "artifacts")
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    # The existing SMOKE reference uses a static 8x8 SVD and logs through the real runner.
    result = runner.run("SMOKE-000", "reference")
    return runner.RUNS / result["run_id"]


def test_completed_cache_is_verified_without_reexecution(cached_floor, monkeypatch):
    before = {p.name: p.read_bytes() for p in cached_floor.iterdir()}
    monkeypatch.setattr(runner, "run", lambda *args: pytest.fail("cache must not execute a run"))
    assert runner._pending_batch_runs([("SMOKE-000", "reference", None, 0)], 0) == []
    assert before == {p.name: p.read_bytes() for p in cached_floor.iterdir()}
    with pytest.raises(FileExistsError, match="--force"):
        runner._pending_batch_runs([("SMOKE-000", "reference", None, 0)], 0, force=True)
    assert before == {p.name: p.read_bytes() for p in cached_floor.iterdir()}


@pytest.mark.parametrize("change", [{"s_wb": .123}, {"formula_version": -1}])
def test_corrupt_or_stale_completed_cache_is_not_silently_reused(cached_floor, change):
    path = cached_floor / "result.json"
    value = json.loads(path.read_text())
    value.update(change)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="cached run failed verification"):
        runner._pending_batch_runs([("SMOKE-000", "reference", None, 0)], 0)


def test_internally_valid_cache_must_match_requested_run_identity(cached_floor):
    copied_identity = cached_floor.with_name(cached_floor.name.replace("__r0", "__r1"))
    cached_floor.rename(copied_identity)
    with pytest.raises(ValueError, match="cached run failed verification"):
        runner._pending_batch_runs([("SMOKE-000", "reference", None, 1)], 0)


def test_report_keeps_latest_floor_sample_and_every_agent_sample():
    from weightbench import report

    rows = [{"agent": "reference", "task": "T", "seed": 0, "sample": 0, "s_wb": 0.1},
            {"agent": "reference", "task": "T", "seed": 0, "sample": 1, "s_wb": 0.8},
            {"agent": "reference", "task": "T", "seed": 1, "sample": 0, "s_wb": 0.7},
            {"agent": "baseline", "task": "T", "seed": 0, "sample": 0, "s_wb": 0.0},
            {"agent": "claude", "task": "T", "seed": 0, "sample": 0, "s_wb": 0.3},
            {"agent": "claude", "task": "T", "seed": 0, "sample": 1, "s_wb": 0.5}]
    kept = report._latest_floors(rows)
    assert [r["s_wb"] for r in kept] == [0.8, 0.7, 0.0, 0.3, 0.5]


def test_interrupted_run_message_names_ids_and_the_flag_to_continue(batch):
    for tid, sample in (("TR01-TEST-001", 7), ("TR04-TEST-001", 8)):
        directory = runner.RUNS / runner.make_run_id(tid, "model", 0, sample)
        directory.mkdir(parents=True)
        (directory / "ledger.jsonl").write_bytes(b"partial")
    with pytest.raises(FileExistsError) as err:
        runner.main(["bench", "--model", "model", "--sample", "7", "--samples", "2"])
    msg = str(err.value)
    assert "TR01-TEST-001__model__s0__r7 (interrupted: no result.json)" in msg    # every colliding id is named
    assert "TR04-TEST-001__model__s0__r8 (interrupted: no result.json)" in msg
    assert "--sample 9" in msg                                                     # first sample past all collisions
    assert not batch.builds and not batch.calls                                    # id semantics unchanged
    assert (runner.RUNS / "TR01-TEST-001__model__s0__r7" / "ledger.jsonl").read_bytes() == b"partial"
    assert not (runner.RUNS / "TR04-TEST-001__model__s0__r7").exists()             # nothing reserved by the preflight
