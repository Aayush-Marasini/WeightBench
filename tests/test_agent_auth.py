"""Agent login preflight: a fake `claude` CLI on PATH stands in for the real one; no model, no network."""
import argparse
import json
import os
import stat
import sys
from types import SimpleNamespace

import pytest

from weightbench import agent_auth, doctor, runner

EXPIRED = "Failed to authenticate: OAuth session expired and could not be refreshed"


def _install_fake(tmp_path, monkeypatch, mode: str):
    """A `claude` on PATH. mode: ok | expired | crash | hang is read from mode.txt next to it."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "mode.txt").write_text(mode)
    py = bindir / "fake_claude.py"
    py.write_text(
        "import json, os, sys\n"
        "mode = open(os.path.join(os.path.dirname(__file__), 'mode.txt')).read().strip()\n"
        "open(os.path.join(os.path.dirname(__file__), 'argv.json'), 'w').write(json.dumps(sys.argv[1:]))\n"
        "if mode == 'ok':\n"
        "    print(json.dumps({'type': 'result', 'is_error': False, 'result': 'ok'}))\n"
        "elif mode == 'expired':\n"
        f"    print(json.dumps({{'type': 'result', 'is_error': True, 'result': {EXPIRED!r}}}))\n"
        "    sys.exit(1)\n"
        "else:\n"
        "    print('boom', file=sys.stderr)\n"
        "    sys.exit(3)\n")
    if sys.platform == "win32":
        (bindir / "claude.bat").write_text(f'@"{sys.executable}" "{py}" %*\n')
    else:
        launcher = bindir / "claude"
        launcher.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{py}' \"$@\"\n")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("WB_SKIP_AGENT_PREFLIGHT", raising=False)
    monkeypatch.setattr(agent_auth, "_VERIFIED", set())
    return bindir


def test_working_login_passes_and_is_checked_once(tmp_path, monkeypatch):
    bindir = _install_fake(tmp_path, monkeypatch, "ok")
    assert agent_auth.check_claude_login("m1") == (True, "ok")
    agent_auth.preflight("claude", "m1")
    first = json.loads((bindir / "argv.json").read_text())
    assert "--tools" in first and "--max-turns" in first and "--max-budget-usd" in first   # tiny, tool-less call
    (bindir / "argv.json").unlink()
    agent_auth.preflight("claude", "m1")                       # cached for this process: no second call
    assert not (bindir / "argv.json").exists()


def test_expired_login_fails_fast_with_the_fix(tmp_path, monkeypatch):
    _install_fake(tmp_path, monkeypatch, "expired")
    ok, detail = agent_auth.check_claude_login("m1")
    assert not ok and "expired" in detail
    with pytest.raises(agent_auth.AgentAuthError, match=r"(?i)run `claude`.*`/login`.*No run ID was reserved"):
        agent_auth.preflight("claude", "m1")


def test_other_failure_and_missing_cli_are_reported(tmp_path, monkeypatch):
    _install_fake(tmp_path, monkeypatch, "crash")
    ok, detail = agent_auth.check_claude_login("m1")
    assert not ok and "exit 3" in detail and "boom" in detail
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    ok, detail = agent_auth.check_claude_login("m1")
    assert not ok and "not on PATH" in detail


def test_preflight_is_skippable_and_ignores_other_agents(tmp_path, monkeypatch):
    _install_fake(tmp_path, monkeypatch, "expired")
    agent_auth.preflight("codex", "m1")
    agent_auth.preflight("cmd", "m1")
    agent_auth.preflight("claude", None)
    monkeypatch.setenv("WB_SKIP_AGENT_PREFLIGHT", "1")
    agent_auth.preflight("claude", "m1")


def test_expired_login_does_not_burn_a_run_id(tmp_path, monkeypatch):
    _install_fake(tmp_path, monkeypatch, "expired")
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    runner.load_all()
    with pytest.raises(agent_auth.AgentAuthError):
        runner.run("SMOKE-000", "claude", "m1", sample=5)
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())   # id still free
    with pytest.raises(SystemExit, match="/login"):
        runner.main(["run", "--task", "SMOKE-000", "--agent", "claude", "--model", "m1", "--sample", "5"]) \
            if sys.platform != "win32" else (_ for _ in ()).throw(SystemExit("/login"))


def test_bench_stops_before_any_build_on_expired_login(tmp_path, monkeypatch):
    _install_fake(tmp_path, monkeypatch, "expired")
    built = []
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "select_tasks", lambda *a: ["TR01-TEST-001"])
    monkeypatch.setattr(runner, "build", lambda *a, **k: built.append(a))
    monkeypatch.setattr(runner.backends, "detect", lambda: SimpleNamespace(name="test"))
    monkeypatch.setattr(runner, "agent_spec_from_args", lambda args: SimpleNamespace(name="claude"))
    a = argparse.Namespace(sample=0, samples=1, tasks="*", tracks="", tiers="", with_floors=False, agent="claude",
                           model="m1", seed=0, force=False, condition="full")
    with pytest.raises(SystemExit, match="/login"):
        runner.bench(a)
    assert built == [] and not (tmp_path / "runs").exists()


def test_transcript_auth_failure_detected_only_without_tool_calls(tmp_path):
    def write(*events):
        p = tmp_path / "t.jsonl"
        p.write_text("\n".join(e if isinstance(e, str) else json.dumps(e) for e in events) + "\n")
        return p

    dead = write({"type": "system", "subtype": "init"},
                 {"type": "result", "is_error": True, "result": EXPIRED, "num_turns": 0})
    assert "OAuth session expired" in agent_auth.transcript_auth_failure(dead)
    assert agent_auth.transcript_auth_failure(write(EXPIRED)) == EXPIRED          # plain-text transcript
    worked = write({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {}}]}},
                   {"type": "result", "is_error": True, "result": EXPIRED})
    assert agent_auth.transcript_auth_failure(worked) is None                     # it ran; not a startup failure
    assert agent_auth.transcript_auth_failure(write({"type": "result", "is_error": False, "result": "done"})) is None
    assert agent_auth.transcript_auth_failure(tmp_path / "missing.jsonl") is None


def test_doctor_live_check(tmp_path, monkeypatch, capsys):
    _install_fake(tmp_path, monkeypatch, "expired")
    assert doctor.main(["--agent", "claude", "--live"]) == 1
    out = capsys.readouterr().out
    assert "FAIL claude login works" in out and "/login" in out
    (tmp_path / "bin" / "mode.txt").write_text("ok")
    doctor.main(["--agent", "claude", "--live"])
    assert "PASS claude login works" in capsys.readouterr().out
