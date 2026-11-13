"""Run completion must include admitted HTTP work without waiting forever on idle clients."""
import json
import socket
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from weightbench import runner, server
from weightbench.harness import agents
from weightbench.ledger import Ledger
from weightbench.task import Budget, Evaluation, Task


class StaticTask(Task):
    id, track, tier = "TR99-SHUTDOWN-TEST", "TR01", "foundational"
    budget = Budget(n_probe=1)
    probe_kinds = {"echo": "probe"}

    def prompt(self, art_dir):
        return "HTTP protocol fixture; no subject inference."

    def evaluate(self, art_dir, submission):
        return Evaluation(1.)


def post(url, endpoint, body, token=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-WB-Token"] = token
    request = urllib.request.Request(url + endpoint, json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status


def test_server_close_waits_for_admitted_probe_and_its_ledger_entry(tmp_path):
    entered, release, closing, closed = (threading.Event() for _ in range(4))
    ledger = Ledger(tmp_path / "ledger.jsonl")
    task = StaticTask()

    def probe(*args):
        entered.set()
        assert release.wait(10)
        return {"done": True}

    task.probe = probe
    state = server.RunState(task, tmp_path, tmp_path, ledger, tmp_path)
    srv = server.make_server(state)
    serving = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    serving.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def close():
        srv.shutdown()
        closing.set()
        srv.server_close()
        closed.set()

    with ThreadPoolExecutor(2) as pool:
        request = pool.submit(post, url, "/probe", {"kind": "echo"})
        cleanup = None
        try:
            assert entered.wait(10)
            cleanup = pool.submit(close)
            assert closing.wait(10)
            assert not closed.wait(.1), "server closed while a probe could still write its ledger entry"
            assert not list(ledger.entries("probe"))
        finally:
            release.set()
            if cleanup is None:
                close()
        assert request.result(timeout=10) == 200
        cleanup.result(timeout=10)
    serving.join(timeout=10)
    assert not serving.is_alive() and closed.is_set()
    assert len(list(ledger.entries("probe"))) == 1 and ledger.verify()
    with pytest.raises(OSError):
        socket.create_connection(srv.server_address, timeout=1)


def test_partial_request_times_out_before_server_close(tmp_path):
    admitted = threading.Event()
    state = server.RunState(StaticTask(), tmp_path, tmp_path, Ledger(tmp_path / "ledger.jsonl"), tmp_path)
    srv = server.make_server(state, request_timeout_s=.1)
    finish_request = srv.finish_request

    def entered_request(*args):
        admitted.set()
        finish_request(*args)

    srv.finish_request = entered_request
    serving = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    serving.start()
    client = socket.create_connection(srv.server_address, timeout=5)
    client.sendall(b"POST /trace HTTP/1.0\r\nContent-Length: 20\r\n\r\n{")
    try:
        assert admitted.wait(5)
        with ThreadPoolExecutor(1) as pool:
            cleanup = pool.submit(lambda: (srv.shutdown(), srv.server_close()))
            try:
                response = client.recv(4096)
                assert b" 408 " in response
                cleanup.result(timeout=5)
            finally:
                client.close()
        assert not list(state.ledger.entries()) and state.used == {"forward": 0, "probe": 0}
    finally:
        client.close()
        srv.shutdown()
        srv.server_close()
        serving.join(timeout=5)


def test_runner_drains_pending_trace_before_audit(tmp_path, monkeypatch):
    entered, release, closing = (threading.Event() for _ in range(3))
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "WORK", tmp_path / "work")
    monkeypatch.setattr(runner, "GUARDS", tmp_path / "guards")
    monkeypatch.setattr(runner, "get_task", lambda *args: StaticTask())
    monkeypatch.setattr(runner, "nondefault_level", lambda *args: None)
    artifact = tmp_path / "art"
    (artifact / "public").mkdir(parents=True)
    monkeypatch.setattr(runner, "build", lambda *args, **kwargs: artifact)
    make_server, trace = runner.make_server, server.RunState.trace

    def held_trace(state, body):
        entered.set()
        assert release.wait(10)
        return trace(state, body)

    def tracked_server(*args, **kwargs):
        srv = make_server(*args, **kwargs)
        close = srv.server_close

        def tracked_close():
            closing.set()
            close()

        srv.server_close = tracked_close
        return srv

    monkeypatch.setattr(server.RunState, "trace", held_trace)
    monkeypatch.setattr(runner, "make_server", tracked_server)
    with ThreadPoolExecutor(2) as pool:
        requests = []

        def launch(spec, model, workdir, transcript, wall_clock, turns, guard, ports, flop_cap, url, token):
            transcript.write_text(json.dumps({"type": "assistant", "message": {"content": []}}))
            assert post(url, "/submit", {"type": "report", "report": {}}, token) == 200
            requests.append(pool.submit(post, url, "/trace", {"pid": 123, "blocked": ["socket:example.invalid"]}, token))
            assert entered.wait(10)
            return 0

        monkeypatch.setattr(runner, "_launch_agent", launch)
        spec = agents.get("cmd", "unused", local_ports=[8000])
        run = pool.submit(runner.run, StaticTask.id, "cmd", "fixture", agent_spec=spec)
        try:
            assert closing.wait(10), "runner did not close/drain its API server"
            assert not run.done()
        finally:
            release.set()
        result = run.result(timeout=10)
        assert requests[0].result(timeout=10) == 200
    assert result["quarantined"] and "blocked_network" in result["audit_flags"]
    ledger = Ledger(runner.RUNS / result["run_id"] / "ledger.jsonl")
    kinds = [entry["kind"] for entry in ledger.entries()]
    assert kinds.index("runtime_trace") < kinds.index("agent_exit") < kinds.index("evaluation")
    assert ledger.verify()
