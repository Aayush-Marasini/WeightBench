"""Model-free checks of threaded API submission storage and input rejection."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from weightbench import server
from weightbench.ledger import Ledger
from weightbench.task import Budget


class StaticTask:
    probe_kinds = {"echo": "probe"}
    budget = Budget(n_probe=1)

    def probe(self, art_dir, kind, payload):
        return payload


@pytest.fixture
def state(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    return server.RunState(StaticTask(), tmp_path, tmp_path / "run",
                           Ledger(tmp_path / "ledger.jsonl"), work)


def stored(state):
    return json.loads((state.run_dir / "submission/submission.json").read_text())


def test_parallel_submissions_have_matching_files_and_ledger_order(state, monkeypatch):
    # Hold the first copy open while every other request has entered submit().
    entered, release = threading.Event(), threading.Event()
    copy = server.shutil.copy

    def blocked_copy(src, dst):
        if Path(src).name == "first.npz":
            entered.set()
            assert release.wait(10)
        return copy(src, dst)

    monkeypatch.setattr(server.shutil, "copy", blocked_copy)
    for index in range(8):
        (state.workdir / f"{index}.npz").write_bytes(str(index).encode())
    (state.workdir / "first.npz").write_bytes(b"first")
    waiting = threading.Barrier(8)

    def submit(index):
        waiting.wait(timeout=10)
        return state.submit({"type": "delta_w", "path": f"{index}.npz", "report": {"id": index}})

    with ThreadPoolExecutor(max_workers=9) as pool:
        first = pool.submit(state.submit, {"type": "delta_w", "path": "first.npz"})
        assert entered.wait(10)
        others = [pool.submit(submit, i) for i in range(8)]
        release.set()
        assert first.result()[0] == 200
        assert all(f.result()[0] == 200 for f in others)
    events = list(state.ledger.entries("submission"))
    assert [e["payload"]["n"] for e in events] == list(range(1, 10))
    assert state.n_submissions == 9 and state.ledger.verify()
    assert (state.run_dir / "submission/delta.npz").read_text() == str(stored(state)["report"]["id"])
    assert [p.name for p in state.run_dir.iterdir()] == ["submission"]


@pytest.mark.parametrize("path", [None, [], {}, 42, "", "bad\x00path", "../private.npz"])
def test_invalid_submission_retains_accepted_report(state, path):
    state.submit({"type": "report", "report": {"answer": "accepted"}})
    assert state.submit({"type": "delta_w", "path": path})[0] == 400
    assert stored(state) == {"type": "report", "report": {"answer": "accepted"}}
    assert state.n_submissions == len(list(state.ledger.entries("submission"))) == 1


@pytest.mark.parametrize("operation", ["copy", "install"])
def test_failed_storage_preserves_previous_delta(state, monkeypatch, operation):
    (state.workdir / "old.npz").write_bytes(b"accepted")
    (state.workdir / "new.npz").write_bytes(b"replacement")
    state.submit({"type": "delta_w", "path": "old.npz"})
    if operation == "copy":
        def fail_copy(src, dst):
            Path(dst).write_bytes(b"partial")
            raise OSError("private storage detail")
        monkeypatch.setattr(server.shutil, "copy", fail_copy)
    else:
        rename = Path.rename

        def fail_install(path, target):
            if path.name == "candidate":
                raise OSError("private storage detail")
            return rename(path, target)
        monkeypatch.setattr(Path, "rename", fail_install)
    code, out = state.submit({"type": "delta_w", "path": "new.npz"})
    assert code == 500 and "private storage detail" not in out["error"]
    assert (state.run_dir / "submission/delta.npz").read_bytes() == b"accepted"
    assert state.n_submissions == len(list(state.ledger.entries("submission"))) == 1
    assert [p.name for p in state.run_dir.iterdir()] == ["submission"]


@pytest.mark.parametrize("payload", [None, [], ["file"], "text", 1, False])
def test_invalid_probe_payload_spends_no_budget(state, payload):
    assert state.probe("echo", payload)[0] == 400
    assert state.used == {"forward": 0, "probe": 0}
    assert not list(state.ledger.entries())
    assert state.probe("echo", {})[0] == 200
    assert state.probe("echo", {})[0] == 429


def test_submission_types_can_replace_each_other(state):
    (state.workdir / "edit.npz").write_bytes(b"edit")
    assert state.submit({"type": "delta_w", "path": "edit.npz", "report": {"id": 1}})[0] == 200
    assert state.submit({"type": "refusal", "report": {"reason": "impossible"}})[0] == 200
    assert not (state.run_dir / "submission/delta.npz").exists()
    assert stored(state) == {"type": "refusal", "report": {"reason": "impossible"}}
    assert state.submit({"type": "report", "report": []})[0] == 400
    assert state.submit({"type": "unknown"})[0] == 400
    assert state.n_submissions == 2 and state.ledger.verify()


def test_delta_rejected_at_submit_for_a_task_that_cannot_score_one(tmp_path):
    # A read-only task rejects a ΔW at submit time, not only at scoring.
    from weightbench.task import Task

    class ReadOnly(Task):
        id, track = "T-RO", "TR-01"
        probe_kinds = {}
        budget = Budget(n_probe=0)

    class Write(ReadOnly):
        def base_weights(self, art_dir):
            return {}

    work = tmp_path / "work"
    work.mkdir()
    (work / "delta.npz").write_bytes(b"x")
    for task, code in ((ReadOnly, 400), (Write, 200)):
        run = tmp_path / task.__name__
        st = server.RunState.__new__(server.RunState)
        server.RunState.__init__(st, task.__new__(task), tmp_path, run, Ledger(tmp_path / f"{task.__name__}.jsonl"), work)
        status, body = st.submit({"type": "delta_w", "path": "delta.npz"})
        assert status == code, body
        if code == 400:
            assert "does not accept a ΔW" in body["error"] and "submit-report" in body["error"]
            assert not (run / "submission").exists()
    assert st.submit({"type": "report", "report": {}})[0] == 200


def test_prompt_states_the_turn_cap():
    from weightbench.runner import turn_cap_note

    note = turn_cap_note(80)
    assert "80 turns" in note and "early" in note and "scores 0" in note
