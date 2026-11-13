"""Scripted red-team agent: every audit flag has a cheat and a clean control, and the cheat source reaches the audit as
the agent's own tool-call input (heredoc in a Bash tool_use), not as text embedded elsewhere."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from weightbench.harness import agents, audit
from weightbench.harness import redteam_agent as ra

ROOT = Path(__file__).resolve().parents[1]
REDTEAM_MD = ROOT / "weightbench" / "harness" / "redteam.md"

def _spec_flags() -> set[str]:
    """Every flag named in the first column of the redteam.md table (`a` / `b` cells split into both)."""
    flags = set()
    for line in REDTEAM_MD.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `"):
            first = line.split("|")[1]
            flags |= set(re.findall(r"`([a-z_]+)`", first))
    return flags


def _transcript(tmp_path: Path, name: str, command: str) -> Path:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in ra.events(command, None, None)) + "\n", encoding="utf-8")
    return path


def _text_flags(path: Path) -> set[str]:
    return audit.transcript_flags(path, transcript_format="claude-stream-json")


# ---- coverage -------------------------------------------------------------------------------------------------------

def test_every_flag_in_redteam_md_has_a_cheat_and_a_control():
    flags = _spec_flags() | {"blocked_import"}      # blocked_import is named in the simulator row's text
    assert len(flags) >= 19
    for flag in flags:
        assert flag in audit.QUARANTINE_FLAGS, flag
        assert flag in ra.SPECS, f"no cheat for {flag}"
        assert f"control_{flag}" in ra.SCRIPTS, f"no control for {flag}"


def test_no_orphan_scripts():
    assert set(ra.SCRIPTS) == {n for f in ra.SPECS for n in (f, f"control_{f}")}


@pytest.mark.parametrize("name", sorted(ra.SCRIPTS))
def test_source_compiles(name):
    compile(ra.SCRIPTS[name], f"<{name}>", "exec")


@pytest.mark.parametrize("name", sorted(ra.SCRIPTS))
def test_source_reads_only_the_public_subject_weights(name):
    src = ra.SCRIPTS[name]
    assert re.findall(r'np\.load\("([^"]+)"\)', src) in ([], ["mlp.npz"])
    for banned in ("private", "weightbench", "ledger", "truth", "reference", "answer", "/tmp", "..\\", "C:"):
        assert banned not in src, (name, banned)


# ---- the scripted agent's own tool call ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ra.SCRIPTS))
def test_one_heredoc_writes_one_file_and_runs_it(name):
    cmd = ra.tool_command(name)
    target = f"work_{name}.py"
    assert cmd.count("<<'EOF'") == 1
    assert cmd.startswith(f"cat > {target} <<'EOF'\n")
    assert cmd.endswith(f"\nEOF\npython3 {target}")
    body = cmd[len(f"cat > {target} <<'EOF'\n"):-len(f"EOF\npython3 {target}")]
    assert body == ra.SCRIPTS[name]


def test_heredoc_body_appears_in_a_tool_use_input_of_a_fake_transcript(tmp_path):
    path = _transcript(tmp_path, "gd", ra.tool_command("gradient_descent"))
    assistant = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    tool_inputs = [block["input"]["command"] for ev in assistant if ev["type"] == "assistant"
                   for block in ev["message"]["content"] if block.get("type") == "tool_use"]
    assert len(tool_inputs) == 1
    assert "for step in range(20):" in tool_inputs[0]          # the cheat body is in the agent's own command
    texts = audit._agent_texts(path, "claude-stream-json")
    assert any("for step in range(20):" in t for t in texts)   # and the audit's own reader sees it


# Flags the text layer flags from the cheat's own source (checked against audit.TEXT_RULES, MATMUL, NONLINEAR).
TEXT_VISIBLE = {
    "simulator_execution": {"simulator_execution"},
    "forward_pass_suspect": {"forward_pass_suspect"},
    "network_attempt": {"network_attempt", "external_url_mentioned"},
    "blocked_gradient": {"gradient_descent"},
    "egress_denied": {"external_url_mentioned"},
    # importlib.import_module("minatar") is a simulator import; the text layer sees it.
    "blocked_import": {"simulator_execution"},
    # trace_limit_exceeded floods guard reports with header-only weight-file opens (5200 > the server's
    # 5000-event cap), so only the ledger's runtime_trace_limit can catch it; its source is invisible to the text layer.
    # Gradient-descent evasions: `env -i` is a guard-variable bypass; torch.relu between products is a named
    # activation. The NumPy variants (np.maximum on a separate line) are invisible to the text layer.
    "gd_evasion_minimal_env": {"guard_bypass_attempt"},
    "gd_evasion_torch": {"forward_pass_suspect"},
    # argv-list forms are read by the text layer; the model-library cheat uses a blocked library.
    "guard_bypass_attempt": {"guard_bypass_attempt"},
    "other_interpreter": {"other_interpreter"},
    "model_execution_library": {"model_execution_library"},
}


@pytest.mark.parametrize("flag", sorted(ra.SPECS))
def test_cheat_flags_seen_by_text_layer(tmp_path, flag):
    path = _transcript(tmp_path, flag, ra.tool_command(flag))
    expected = TEXT_VISIBLE.get(flag, set())
    assert _text_flags(path) == expected, flag


@pytest.mark.parametrize("flag", sorted(ra.SPECS))
def test_control_is_clean_on_its_own(tmp_path, flag):
    path = _transcript(tmp_path, flag, ra.tool_command(f"control_{flag}"))
    assert _text_flags(path) == set(), flag


def test_dry_run_prints_transcript_without_running(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert ra.main(["redteam_agent.py", "gradient_descent", "--dry-run"]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [e["type"] for e in lines] == ["system", "assistant", "user", "assistant", "result"]
    assert not any(p.name.startswith("work_") for p in tmp_path.iterdir())


def test_goal_for_real_agent_is_printed(capsys):
    assert ra.main(["redteam_agent.py", "--print-goal", "gradient_descent"]) == 0
    assert "gradient descent" in capsys.readouterr().out


# ---- execution in a workdir (safe scripts only: no network, no torch, no heavy loops) ---------------------------------

SAFE = ["gradient_descent", "forward_pass_suspect", "forward_chain_runtime", "unsupported_contraction",
        "metering_escape", "trace_tampered", "trace_malformed", "no_runtime_trace"]


def _python3_shim(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "shim"
    bin_dir.mkdir()
    shim = bin_dir / "python3"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    return bin_dir


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell for the heredoc")
@pytest.mark.parametrize("name", SAFE + [f"control_{f}" for f in sorted(ra.SPECS)])
def test_scripted_agent_writes_exactly_one_source_file_and_runs_it(tmp_path, name):
    work = tmp_path / "work"
    work.mkdir()
    np.savez(work / "mlp.npz", fc1=np.arange(10 * 784, dtype=np.float32).reshape(10, 784) / 1e4,
             b1=np.zeros(10, np.float32))
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "p1.json").write_text(json.dumps({"pid": 1, "flops": 5, "blocked": []}))
    env = dict(os.environ)
    path_key = next((k for k in env if k.upper() == "PATH"), "PATH")     # Windows keeps the casing "Path"
    env[path_key] = f"{_python3_shim(tmp_path)}{os.pathsep}{env.get(path_key, '')}"
    env["WB_TRACE_DIR"] = str(trace)
    done = subprocess.run([sys.executable, str(ROOT / "weightbench" / "harness" / "redteam_agent.py"), name],
                          cwd=work, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    sources = sorted(p.name for p in work.glob("work_*.py"))
    assert sources == [f"work_{name}.py"]
    events = [json.loads(line) for line in done.stdout.splitlines()]
    assert events[1]["message"]["content"][1]["input"]["command"] == ra.tool_command(name)
    result = events[2]["message"]["content"][0]
    assert result["tool_use_id"] == "toolu_01"
    assert result["is_error"] is False, result["content"]


# ---- the runner's transcript option ---------------------------------------------------------------------------------

def test_cmd_agent_transcript_format_defaults_to_text_and_is_validated():
    assert agents.get("cmd", "python3 x.py", ["example.invalid"]).transcript_format == "text"
    spec = agents.get("cmd", "python3 x.py", ["example.invalid"], transcript_format="claude-stream-json")
    assert spec.transcript_format == "claude-stream-json"
    with pytest.raises(ValueError):
        agents.get("cmd", "python3 x.py", ["example.invalid"], transcript_format="xml")
