import tempfile
"""`weightbench bench`: agent adapters, task selection, sandbox backends (argument construction), net bridge."""
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from weightbench import runner
from weightbench.harness import agents, backends, netbridge


# ------------------------------------------------------------------------------------- agent adapters
def test_builtin_agents_have_egress_allowlists_and_noninteractive_commands(tmp_path):
    claude = agents.get("claude").command("claude-opus-5-5", "do the task", tmp_path / "PROMPT.md", 50)
    assert claude[:2] == ["claude", "-p"] and "--dangerously-skip-permissions" in claude and "WebFetch,WebSearch" in claude
    assert "--strict-mcp-config" in claude and json.loads(claude[claude.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert agents.get("claude").egress_hosts == ("api.anthropic.com",)
    codex = agents.get("codex").command("gpt-x", "do the task", tmp_path / "PROMPT.md", 50)
    assert codex[:2] == ["codex", "exec"] and codex[-1] == "do the task" and "api.openai.com" in agents.get("codex").egress_hosts


def test_allow_host_extends_a_builtin_agents_own_endpoints():
    """A built-in CLI on a gateway (Claude Code -> OpenRouter) keeps its own hosts and *adds* the gateway's."""
    spec = agents.get("claude", allow_hosts=["openrouter.ai"])
    assert spec.egress_hosts == ("api.anthropic.com", "openrouter.ai")
    assert spec.home_state == (".claude", ".claude.json", ".config/claude")     # unchanged
    # Asking again, or for a host the adapter already allows, changes nothing.
    assert agents.get("claude", allow_hosts=["api.anthropic.com"]).egress_hosts == ("api.anthropic.com",)
    assert agents.get("claude", allow_hosts=["openrouter.ai", "openrouter.ai"]).egress_hosts == (
        "api.anthropic.com", "openrouter.ai")


@pytest.mark.parametrize("floor", ["baseline", "reference"])
def test_floor_agents_get_no_agent_spec(floor):
    """`baseline`/`reference` are run by the harness, not by a CLI adapter: no spec, no ValueError.

    `bench --with-floors`, the e2e floors and `test_run_identity` rely on `run` not deriving a spec for them (otherwise
    they fail with `ValueError: unknown agent 'reference'` before anything runs)."""
    import argparse

    a = argparse.Namespace(agent=floor, agent_cmd=None, allow_host="api.anthropic.com", agent_home="",
                           allow_port="", python_agent=False, agent_transcript_format="text")
    assert runner.agent_spec_from_args(a) is None
    a.agent = "claude"          # a real adapter, with a gateway host added
    assert runner.agent_spec_from_args(a).egress_hosts == ("api.anthropic.com",)


def test_custom_agent_template_and_required_allowlist(tmp_path):
    spec = agents.get("cmd", "my-agent --model {model} --prompt-file {prompt_file} --turns {max_turns}",
                      ["api.example.com"], [".my-agent"])
    assert spec.command("m1", "p", tmp_path / "PROMPT.md", 7) == [
        "my-agent", "--model", "m1", "--prompt-file", str(tmp_path / "PROMPT.md"), "--turns", "7"]
    assert spec.egress_hosts == ("api.example.com",) and spec.home_state == (".my-agent",)
    with pytest.raises(ValueError, match="allow-host"):
        agents.get("cmd", "my-agent {prompt}", [])
    local = agents.get("cmd", "my-agent {prompt}", [], local_ports=[8000], python_runtime=True)
    assert local.local_ports == (8000,) and local.python_runtime and local.egress_hosts == ()


def test_proxy_allow_entries_parse_host_and_port():
    from weightbench.harness import sandbox

    assert sandbox.parse_allow_entry("api.example.com") == ("api.example.com", 443)
    assert sandbox.parse_allow_entry("LLM.lab.internal:8443") == ("llm.lab.internal", 8443)
    allow = {sandbox.parse_allow_entry(e) for e in ("api.openai.com", "*.oaiusercontent.com")}
    assert sandbox.host_allowed("sdmntpreastus3.oaiusercontent.com", 443, allow)
    assert not sandbox.host_allowed("oaiusercontent.com", 443, allow)               # wildcard = subdomains only
    assert not sandbox.host_allowed("evil-oaiusercontent.com", 443, allow)
    assert not sandbox.host_allowed("sdmntpr.oaiusercontent.com", 80, allow)        # port must match
    assert sandbox.host_allowed("API.OPENAI.COM", 443, allow) and not sandbox.host_allowed("example.com", 443, allow)


def test_python_agent_root_exemption_is_one_shot_and_audited(tmp_path):
    """The agent root runs unguarded; its children are guarded; a second claim quarantines the run."""
    import subprocess

    from weightbench.harness import audit, sandbox

    pub = tmp_path / "public"
    pub.mkdir()
    guard = sandbox.install_guard(tmp_path / "guard", pub)
    child = ("import subprocess, sys\n"
             "import json; print('root_blocked', 'json' in sys.modules)\n"
             "r = subprocess.run([sys.executable, '-c', \"import importlib.util as u, sys\\n"
             "try:\\n import minatar\\nexcept ImportError as e:\\n print(type(e).__name__)\"],"
             " capture_output=True, text=True)\n"
             "print('child', r.stdout.strip())\n")
    env = {**__import__("os").environ, "PYTHONPATH": str(guard / "lib"), "WB_TRACE_DIR": str(guard / "trace"),
           "WB_AGENT_ROOT": "1"}
    out = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    traces = [json.loads(p.read_text()) for p in (guard / "trace").glob("*.json")]
    assert sum(1 for t in traces if t.get("agent_root")) == 1                  # only the root was exempt
    claims = [{"pid": 1, "agent_root": True}, {"pid": 2, "agent_root": True}]
    t = tmp_path / "t.jsonl"
    t.write_text("")
    assert "agent_root_spoof" in audit.audit_run(t, guard / "trace", tmp_path / "e", server_traces=claims,
                                                 python_agent=True)["flags"]
    assert "agent_root_spoof" not in audit.audit_run(t, guard / "trace", tmp_path / "e", server_traces=claims[:1],
                                                     python_agent=True)["flags"]
    assert "agent_root_spoof" in audit.audit_run(t, guard / "trace", tmp_path / "e", server_traces=claims[:1])["flags"]
    with pytest.raises(ValueError, match="agent-cmd"):
        agents.get("cmd")
    with pytest.raises(ValueError, match="unknown agent"):
        agents.get("gpt-cli")


def test_run_tag():
    assert runner.run_tag("baseline", None) == "baseline"
    assert runner.run_tag("claude", "claude-opus-5-5") == "claude-opus-5-5"
    assert runner.run_tag("codex", "gpt-5") == "codex-gpt-5"
    assert runner.run_tag("cmd", "org/model:v1") == "cmd-org_model_v1"


# ------------------------------------------------------------------------------------- Windows support
def test_windows_paths_and_wsl_delegation(monkeypatch):
    from weightbench import platform_compat as PC

    if sys.platform == "win32":
        assert PC.to_wsl_path(Path(r"C:\Users\dev\src\weightbench")) == "/mnt/c/Users/dev/src/weightbench"
    monkeypatch.setenv("HF_TOKEN", "abc def")
    cmd = runner.wsl_command(["bench", "--agent", "claude", "--model", "m"], Path("/repo"))
    assert cmd[:4] == ["wsl.exe", "-e", "bash", "-lc"]
    inner = cmd[4]
    assert "wb-venv/bin/weightbench\" bench --agent claude --model m" in inner and "WB_HEAVY=local" in inner
    assert "HF_TOKEN='abc def'" in inner                                       # passed through, shell-quoted


def test_only_agent_runs_need_the_sandbox():
    import argparse

    def ns(cmd, agent="claude"):
        return argparse.Namespace(cmd=cmd, agent=agent)

    assert runner.needs_agent_sandbox(ns("bench")) and runner.needs_agent_sandbox(ns("run"))
    assert not runner.needs_agent_sandbox(ns("run", "reference")) and not runner.needs_agent_sandbox(ns("build"))
    assert not runner.needs_agent_sandbox(ns("list")) and not runner.needs_agent_sandbox(ns("rescore"))


def test_cross_platform_file_lock(tmp_path):
    from weightbench import platform_compat as PC

    with PC.file_lock(tmp_path / "x.lock"):
        (tmp_path / "inside").write_text("ok")
    assert (tmp_path / "inside").read_text() == "ok"
    (tmp_path / "a.bin").write_bytes(b"123")
    PC.clone_file(tmp_path / "a.bin", tmp_path / "b.bin")
    assert (tmp_path / "b.bin").read_bytes() == b"123"


# ------------------------------------------------------------------------------------- task selection
class _T:
    def __init__(self, track, tier):
        self.track, self.tier = track, tier


def test_select_tasks_by_track_tier_and_glob(monkeypatch):
    fake = {"SMOKE-000": _T("SMOKE", "foundational"), "TR01-A": _T("TR01", "foundational"),
            "TR04-A": _T("TR04", "intermediate"), "TR04-B": _T("TR04", "frontier")}
    monkeypatch.setattr(runner, "REGISTRY", fake)
    assert runner.select_tasks() == ["TR01-A", "TR04-A", "TR04-B"]                    # smoke task never benchmarked
    fake["TR02-HARD"] = _T("TR02", "frontier")
    fake["TR05-EASY"] = _T("TR05", "foundational")
    assert runner.select_tasks() == ["TR01-A", "TR05-EASY", "TR04-A", "TR02-HARD", "TR04-B"]   # easy -> hard
    del fake["TR02-HARD"], fake["TR05-EASY"]
    assert runner.select_tasks(tracks="TR04") == ["TR04-A", "TR04-B"]
    assert runner.select_tasks(tracks="tr-04,TR01", tiers="frontier") == ["TR04-B"]
    assert runner.select_tasks("TR04-A") == ["TR04-A"]
    assert runner.select_tasks("TR01-*,TR04-B") == ["TR01-A", "TR04-B"]                # several patterns


# ------------------------------------------------------------------------------------- linux backend arguments
@pytest.fixture
def box(tmp_path):
    work, guard, venv = tmp_path / "work", tmp_path / "guard", tmp_path / "venv"
    for d in (work, guard / "lib", guard / "trace", guard / "sock", venv / "bin", tmp_path / "home" / ".claude"):
        d.mkdir(parents=True)
    return backends.SandboxSpec(workdir=work, guard_root=guard, agent_venv=venv, deny_read=[tmp_path / "repo"],
                                ports={"api": 5001, "proxy": 5002}, home_state=(".claude", ".missing"),
                                home=tmp_path / "home")


def _pairs(args, flag):
    return [(args[i + 1], args[i + 2]) for i, a in enumerate(args) if a == flag]


def test_bwrap_isolates_network_home_and_tmp(box):
    args = backends.bwrap_args(box)
    assert args[0] == "bwrap" and "--unshare-all" in args and "--share-net" not in args and "--die-with-parent" in args
    tmpfs = [args[i + 1] for i, a in enumerate(args) if a == "--tmpfs"]
    assert "/tmp" in tmpfs and str(box.home) in tmpfs                                 # fresh /tmp, empty $HOME
    binds = dict(_pairs(args, "--bind"))
    assert str(box.workdir) in binds and str(box.guard_root / "trace") in binds
    assert str(box.home / ".claude") in binds and str(box.home / ".missing") not in binds  # only existing state
    ro = dict(_pairs(args, "--ro-bind"))
    assert str(box.guard_root / "lib") in ro and str(box.guard_root / "lib") not in binds  # guard code read-only
    assert str(box.agent_venv) in ro
    assert args.index("--tmpfs") < args.index(str(box.workdir))                      # workdir mounted after /tmp


def test_linux_wrap_bridges_api_and_proxy_through_unix_sockets(box):
    cmd = backends.LinuxBwrapBackend().wrap(box, ["claude", "-p", "x"])
    tail = cmd[cmd.index("--") + 1:]
    # -S as well: the bridge is harness code and must not load the agent guard (its CPU is not the agent's)
    assert tail[:4] == [str(box.agent_venv / "bin" / "python"), "-I", "-S",
                        str(box.guard_root / "lib" / "wb_netbridge.py")]
    assert f"5001={box.guard_root / 'sock' / 'api.sock'}" in tail and f"5002={box.guard_root / 'sock' / 'proxy.sock'}" in tail
    assert tail[-3:] == ["claude", "-p", "x"]


def test_detect_rejects_native_windows(monkeypatch):
    monkeypatch.setenv("WB_SANDBOX", "windows")
    with pytest.raises(RuntimeError, match="WSL2"):
        backends.detect()


# ------------------------------------------------------------------------------------- net bridge
def test_netbridge_arg_parsing():
    maps, cmd, limits = netbridge.parse_args(["--map", "5001=/s/api.sock", "--map", "5002=/s/p.sock",
                                              "--rlimit-cpu", "3600", "--rlimit-fsize", "1024", "--", "claude", "-p"])
    assert maps == [(5001, "/s/api.sock"), (5002, "/s/p.sock")] and cmd == ["claude", "-p"]
    assert limits == {"cpu": 3600, "fsize": 1024}
    assert netbridge.parse_args(["--", "x"])[2] == {}
    with pytest.raises(SystemExit):
        netbridge.parse_args(["--map", "1=/x"])


def test_linux_wrap_passes_resource_limits(box):
    box.cpu_seconds = 7200
    cmd = backends.LinuxBwrapBackend().wrap(box, ["claude"])
    tail = cmd[cmd.index("--") + 1:]
    assert tail[tail.index("--rlimit-cpu") + 1] == "7200"
    assert tail[tail.index("--rlimit-fsize") + 1] == str(backends.DEFAULT_FSIZE_BYTES)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits")
def test_file_size_limit_is_enforced(tmp_path):
    import errno
    import subprocess

    script = tmp_path / "write.py"
    script.write_text("import sys, resource\nprint('limit', *resource.getrlimit(resource.RLIMIT_FSIZE))\n"
                      "try:\n    with open(sys.argv[1], 'wb') as f:\n        f.write(b'x' * 4096)\n"
                      "    print('wrote')\nexcept OSError as e:\n    print('blocked', e.errno)\n")
    # Test POSIX enforcement on local storage (as used by the default agent workdir),
    # not Lustre's small-write semantics. /dev/shm avoids a compute node's full /tmp.
    local_tmp = "/dev/shm" if sys.platform == "linux" and Path("/dev/shm").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="wb-fsize-", dir=local_tmp) as temp:
        output = Path(temp) / "big.bin"
        # An unrestricted control proves space/quota errors cannot make this test pass.
        output.write_bytes(b'x' * 4096)
        assert output.stat().st_size == 4096
        output.unlink()
        out = subprocess.run([sys.executable, str(Path(netbridge.__file__)), "--rlimit-fsize", "1024", "--",
                              sys.executable, str(script), str(output)], capture_output=True, text=True)
        assert "limit 1024 1024" in out.stdout, out.stdout + out.stderr
        assert f"blocked {errno.EFBIG}" in out.stdout, out.stdout + out.stderr
        assert output.stat().st_size <= 1024


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX") or sys.platform == "win32", reason="unix sockets")
def test_netbridge_round_trip():
    """loopback client → TcpToUnix → unix socket → UnixToTcp → echo server, as inside / outside the sandbox."""
    echo = socket.socket()
    echo.bind(("127.0.0.1", 0))
    echo.listen(1)

    def serve():
        c, _ = echo.accept()
        c.sendall(c.recv(64).upper())
        c.close()

    threading.Thread(target=serve, daemon=True).start()
    # Both Linux and macOS limit socket paths; shared-filesystem pytest roots can be longer.
    with tempfile.TemporaryDirectory(prefix=".wbnb-", dir=Path.home()) as short:
        sock_path = str(Path(short) / "b.sock")
        host = netbridge.UnixToTcp(sock_path, echo.getsockname()[1]).start()
        inner = netbridge.TcpToUnix(0, sock_path)
        threading.Thread(target=inner.serve_forever, daemon=True).start()
        try:
            with socket.create_connection(("127.0.0.1", inner.server_address[1]), timeout=5) as c:
                c.sendall(b"ping")
                assert c.recv(64) == b"PING"
        finally:
            inner.shutdown()
            host.stop()
            echo.close()


def test_codex_transcript_audits_agent_actions_not_command_output(tmp_path):
    from weightbench.harness import audit

    t = tmp_path / "codex.jsonl"
    events = [
        {"type": "item.completed", "item": {"type": "command_execution", "command": "cat TASK.md", "status": "completed",
                                            "aggregated_output": "h = relu(W1 @ x + b1)\nlogits = W2 @ h  # pip install x"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "I will bound each neuron."}},
    ]
    t.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    assert audit.transcript_flags(t, transcript_format="codex-json") == set()            # TASK.md text ignored
    events.append({"type": "item.completed", "item": {"type": "command_execution", "status": "completed",
                                                      "command": "python -c 'import transformers'"}})
    t.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    assert "model_execution_library" in audit.transcript_flags(t, transcript_format="codex-json")
    assert agents.get("codex").transcript_format == "codex-json"


def test_audit_text_transcripts_are_checked_line_by_line(tmp_path):
    from weightbench.harness import audit

    t = tmp_path / "codex.jsonl"
    t.write_text(json.dumps({"type": "exec", "cmd": "pip install transformers"}) + "\n")
    assert "network_attempt" in audit.transcript_flags(t, transcript_format="text")
    assert "network_attempt" not in audit.transcript_flags(t)          # claude format: not an assistant message


# ------------------------------------------------------------------------------------- stall watchdog
def test_stall_seconds_default_and_override(monkeypatch):
    monkeypatch.delenv("WB_AGENT_STALL_S", raising=False)
    assert runner.stall_seconds() == runner.STALL_DEFAULT_S
    monkeypatch.setenv("WB_AGENT_STALL_S", "0")
    assert runner.stall_seconds() == 0                                 # opt out: wall clock only
    monkeypatch.setenv("WB_AGENT_STALL_S", "45")
    assert runner.stall_seconds() == 45
    monkeypatch.setenv("WB_AGENT_STALL_S", " 45 ")
    assert runner.stall_seconds() == 45
    for bad in ("soon", "-1"):
        monkeypatch.setenv("WB_AGENT_STALL_S", bad)
        with pytest.raises(ValueError):
            runner.stall_seconds()


@pytest.fixture
def fast_poll(monkeypatch):
    monkeypatch.setattr(runner, "STALL_POLL_S", 0.02)


def _spawn(script):
    return subprocess.Popen([sys.executable, "-c", script])


def test_wait_agent_kills_a_silent_process(tmp_path, fast_poll):
    """A wedged CLI (no bytes written for stall_s) is killed, and the run says so instead of hanging to wall-clock."""
    t = tmp_path / "transcript.jsonl"
    with t.open("w"):
        proc = _spawn("import time; time.sleep(60)")
        t0 = time.monotonic()
        rc = runner._wait_agent(proc, t, time.monotonic() + 60, stall_s=1)
    assert rc == runner.STALL_EXIT and rc == -8
    assert time.monotonic() - t0 < 20
    assert proc.poll() is not None                                     # no sandbox left running


def test_wait_agent_does_not_kill_a_writing_process(tmp_path, fast_poll):
    """Growth resets the clock: an agent that keeps emitting transcript bytes is left alone to finish."""
    t = tmp_path / "transcript.jsonl"
    script = ("import time\n"
              "for _ in range(8):\n"
              f"    with open({str(t)!r}, 'a') as f:\n"
              "        f.write('assistant\\n')\n"
              "    time.sleep(0.3)\n"
              "raise SystemExit(3)\n")
    with t.open("w"):
        proc = _spawn(script)
        rc = runner._wait_agent(proc, t, time.monotonic() + 60, stall_s=1)
    assert rc == 3                                                     # the agent's own exit code, not STALL_EXIT


def test_wait_agent_wall_clock_deadline_still_wins(tmp_path, fast_poll):
    t = tmp_path / "transcript.jsonl"
    with t.open("w"):
        proc = _spawn("import time; time.sleep(60)")
        assert runner._wait_agent(proc, t, time.monotonic() + 0.5, stall_s=0) == -9
    assert proc.poll() is not None
