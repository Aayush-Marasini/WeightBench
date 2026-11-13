"""The agent guard must load even under `python -I` / `-E` (Claude Code uses -I for untrusted files)."""
import os
import json
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from weightbench.harness import audit, sandbox


def test_guard_pth_written_and_env(tmp_path):
    venv = tmp_path / "venv"
    sp = venv / "lib" / f"python3.{sys.version_info.minor}" / "site-packages"
    sp.mkdir(parents=True)
    g = tmp_path / "guard"
    sandbox.install_guard(g, tmp_path)
    env = sandbox.agent_env({}, venv, g, 1)
    assert env["WB_GUARD_LIB"] == str(g / "lib")
    assert (sp / "zz_wb_guard.pth").read_text() == sandbox.GUARD_PTH


@pytest.fixture(scope="module")
def agent_venv(tmp_path_factory):
    """Bare, disposable interpreter: no downloads, subject weights or external venv."""
    if sys.platform == "win32":
        pytest.skip("agent subprocesses use the POSIX runtime through WSL2 on Windows")
    directory = tmp_path_factory.mktemp("guard-agent-venv")
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(directory)
    return directory


@pytest.mark.parametrize("flags,loaded", [([], True), (["-I"], True), (["-E"], True), (["-S"], False)])
def test_guard_loads_under_isolated_mode(tmp_path, agent_venv, flags, loaded, record_property):
    g = tmp_path / "guard"
    sandbox.install_guard(g, tmp_path)
    base = os.environ.copy()
    for key in ("WB_AGENT_ROOT", "WB_URL", "WB_TOKEN", "WB_FLOP_CAP"):
        base.pop(key, None)
    env = sandbox.agent_env(base, agent_venv, g, 1)
    code = """import json, socket, sys
guard = sys.modules.get('sitecustomize')
blocked = None
if guard is not None and hasattr(guard, 'TRACE'):
    # The guard rejects this documentation-only address before any network traffic.
    with socket.socket() as connection:
        try:
            connection.connect(('192.0.2.1', 443))
        except Exception as exc:
            blocked = type(exc).__name__
print(json.dumps({'guard': getattr(guard, '__file__', None), 'blocked': blocked}))
"""
    r = subprocess.run([str(agent_venv / "bin/python"), *flags, "-c", code],
                       env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    result = json.loads(r.stdout)
    traces = [json.loads(p.read_text()) for p in (g / "trace").glob("*.json")]
    record_property("startup_flags", " ".join(flags) or "ordinary")
    record_property("guard_loaded", bool(result["guard"]))
    record_property("blocked_exception", result["blocked"] or "none")
    record_property("trace_count", len(traces))
    if loaded:
        assert Path(result["guard"]).resolve() == (g / "lib/sitecustomize.py").resolve()
        assert result["blocked"] == "GuardViolation"
        assert len(traces) == 1 and "socket:192.0.2.1" in traces[0]["blocked"]
    else:
        assert result == {"guard": None, "blocked": None} and not traces


def test_audit_flags_dash_S_not_dash_I(tmp_path):
    transcript = tmp_path / "commands.txt"
    transcript.write_text("python -S x.py\n")
    flags = audit.transcript_flags(transcript, transcript_format="text")
    assert "guard_bypass_attempt" in flags
    transcript.write_text("python -I x.py\npython -E -c 1\n")
    flags = audit.transcript_flags(transcript, transcript_format="text")
    assert "guard_bypass_attempt" not in flags
