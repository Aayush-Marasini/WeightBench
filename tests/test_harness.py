"""Harness: sandbox profile, egress proxy, runtime guard (in real subprocesses), audit, quarantine."""
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from weightbench import report, runner
from weightbench.harness import audit as A
from weightbench.harness import sandbox as S


# ------------------------------------------------------------------------------------- profile
def _rule_lines(prof: str, prefix: str) -> list[str]:
    return [line for line in prof.splitlines() if line.startswith(prefix)]


def test_path_forms_cover_private_aliases():
    assert {"/tmp/wb-work/x", "/private/tmp/wb-work/x"} <= set(S.path_forms("/tmp/wb-work/x"))
    assert {"/var/folders/a", "/private/var/folders/a"} <= set(S.path_forms("/private/var/folders/a"))
    assert S.path_forms("/Users/a/repo") == ["/Users/a/repo"]


def test_profile_rules_and_precedence():
    home, work, guard = Path("/Users/a"), Path("/tmp/wb-work/r1"), Path("/tmp/wb-guard/r1")
    prof = S.macos_profile(work, [Path("/Users/a/repo"), Path("/tmp/wb-work")], guard, home=home)
    lines = prof.splitlines()
    assert prof.startswith("(version 1)\n(allow default)")
    deny_home = next(i for i, l in enumerate(lines) if l.startswith('(deny file-read* (subpath "/Users/a")'))
    allow_home = next(i for i, l in enumerate(lines) if l.startswith("(allow file-read*") and "/Users/a/.claude" in l)
    deny_repo = next(i for i, l in enumerate(lines) if '(subpath "/Users/a/repo")' in l and l.startswith("(deny file-read*"))
    deny_write = lines.index("(deny file-write*)")
    allow_work = next(i for i, l in enumerate(lines) if l.startswith("(allow file-read* file-write*"))
    assert deny_home < allow_home < deny_repo < deny_write < allow_work           # later rules win
    assert '(subpath "/private/tmp/wb-work")' in lines[deny_repo]                  # both spellings denied
    assert '(subpath "/private/tmp/wb-work/r1")' in lines[allow_work]              # own workdir re-allowed
    trace_rule = next(l for l in lines if l.startswith("(allow file-write*") and "wb-guard/r1/trace" in l)
    assert "wb-guard/r1/lib" not in trace_rule                                     # guard code stays read-only
    assert not any("Desktop" in l or "Documents" in l for l in _rule_lines(prof, "(allow"))
    exec_rule = next(l for l in lines if l.startswith("(deny process-exec"))
    assert '(literal "/usr/bin/clang")' in exec_rule and '(literal "/usr/bin/perl")' in exec_rule
    assert prof.index("(deny network-outbound)") < prof.index('(remote ip "localhost:*")')
    assert '(regex #"^/Users/a/\\.claude\\.json")' in prof


def test_runner_profile_keeps_repo_and_runs_denied():
    prof = runner._sandbox_profile(Path("/tmp/wb-work/x"))
    deny_lines = [l for l in prof.splitlines() if l.startswith("(deny file-read*")]
    for p in (runner.ROOT, runner.WORK, runner.GUARDS):
        for form in S.path_forms(p):
            assert any(f'(subpath "{form}")' in l for l in deny_lines), form
    assert "(deny file-write*)" in prof and "(deny network-outbound)" in prof


def test_agent_env(tmp_path):
    env = S.agent_env({"PATH": "/usr/bin", "VIRTUAL_ENV": "x", "PYTHONHOME": "y"}, Path("/venv"), tmp_path, 4242, 1e9,
                      wb_url="http://127.0.0.1:9", wb_token="t0k", tmpdir=tmp_path / "tmp")
    assert env["HTTPS_PROXY"] == env["http_proxy"] == "http://127.0.0.1:4242"
    assert env["NO_PROXY"] == "127.0.0.1,localhost" and env["PYTHONPATH"] == str(tmp_path / "lib")
    assert "VIRTUAL_ENV" not in env and "PYTHONHOME" not in env and env["WB_FLOP_CAP"] == "1000000000.0"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["WB_URL"] == "http://127.0.0.1:9" and env["WB_TOKEN"] == "t0k" and env["TMPDIR"] == str(tmp_path / "tmp")


# ------------------------------------------------------------------------------------- Weight API server
class _FakeTask:
    probe_kinds = {"echo": "probe"}

    class budget:  # noqa: N801 - mimics Task.budget
        n_forward, n_probe = 0, 3

    def __init__(self, art_dir):
        self.art_dir = art_dir

    def probe(self, art_dir, kind, payload):
        if payload.get("boom") == "value":
            raise ValueError(f"bad input near {art_dir}/private/answer.json")
        if payload.get("boom") == "internal":
            raise RuntimeError("secret internal state 0.731")
        if "file" in payload:
            return {"bytes": Path(payload["file"]).read_bytes().decode()}
        return {"ok": True}


@pytest.fixture
def api(tmp_path):
    from weightbench.ledger import Ledger
    from weightbench.server import RunState, make_server

    work, art = tmp_path / "work", tmp_path / "art"
    work.mkdir()
    art.mkdir()
    led = Ledger(tmp_path / "ledger.jsonl")
    state = RunState(_FakeTask(art), art, tmp_path / "run", led, work, token="s3cret")
    srv = make_server(state)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state, led, work
    srv.shutdown()


def _post(url: str, path: str, body: dict, token: str | None = "s3cret", raw: bytes | None = None,
          length: str | None = None) -> tuple[int, dict]:
    import urllib.error
    import urllib.request

    data = raw if raw is not None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-WB-Token"] = token
    req = urllib.request.Request(url + path, data=data, method="POST", headers=headers)
    if length is not None:
        req.add_unredirected_header("Content-Length", length)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_server_requires_run_token(api):
    url, state, led, _ = api
    assert _post(url, "/probe", {"kind": "echo"}, token=None)[0] == 401
    assert _post(url, "/probe", {"kind": "echo"}, token="other-run")[0] == 401
    assert state.used["probe"] == 0                                     # rejected calls are not metered
    code, out = _post(url, "/probe", {"kind": "echo"})
    assert code == 200 and out["result"] == {"ok": True}


def test_server_rejects_oversized_body(api):
    url, *_ = api
    from weightbench import server

    port = int(url.rsplit(":", 1)[1])
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall((f"POST /probe HTTP/1.1\r\nHost: x\r\nX-WB-Token: s3cret\r\n"
               f"Content-Length: {server.MAX_BODY_BYTES + 1}\r\n\r\n{{}}").encode())
    status = s.recv(64).decode().split("\r\n")[0]          # answered from the header alone, body never read
    s.close()
    assert " 413 " in status + " "


def test_probe_errors_redacted_for_agent_but_logged(api):
    url, state, led, _ = api
    out = _post(url, "/probe", {"kind": "echo", "payload": {"boom": "value"}})[1]["result"]["error"]
    assert "<private>" in out and str(state.art_dir) not in out
    out = _post(url, "/probe", {"kind": "echo", "payload": {"boom": "internal"}})[1]["result"]["error"]
    assert "0.731" not in out and "internal evaluator error" in out
    logged = [e["payload"].get("error", "") for e in led.entries("probe")]
    assert any("0.731" in e for e in logged)                            # full detail stays in the ledger


def test_probe_file_is_copied_before_the_task_reads_it(api):
    url, state, _, work = api
    (work / "cand.txt").write_text("candidate")
    out = _post(url, "/probe", {"kind": "echo", "payload": {"file": "cand.txt"}})[1]
    assert out["result"] == {"bytes": "candidate"}
    assert not list((state.run_dir / "probe_inputs").glob("*"))         # private copy removed afterwards
    assert _post(url, "/probe", {"kind": "echo", "payload": {"file": "../art/x"}})[0] == 400


@pytest.mark.parametrize("payload", [None, [], "file", False])
def test_http_probe_rejects_nonobject_payload_without_spending_budget(api, payload):
    url, state, ledger, _ = api
    assert _post(url, "/probe", {"kind": "echo", "payload": payload})[0] == 400
    assert state.used == {"forward": 0, "probe": 0}
    assert not list(ledger.entries())


def test_trace_endpoint_appends_to_ledger(api):
    url, _, led, _ = api
    code, _ = _post(url, "/trace", {"pid": 7, "flops": 5, "weight_matmuls": 1, "forward_chains": 1,
                                    "blocked": ["import:jax"], "final": True, "junk": "x" * 10})
    assert code == 200
    (entry,) = list(led.entries("runtime_trace"))
    assert entry["payload"]["pid"] == 7 and "junk" not in entry["payload"] and led.verify()


BAD_TRACE_FIELDS = [
    {"pid": []}, {"pid": {}}, {"pid": 0}, {"pid": True},
    {"flops": "not-a-counter"}, {"flops": -1}, {"flops": True},
    {"flops": float("nan")}, {"flops": float("inf")}, {"weight_matmuls": .5},
    {"forward_chains": []}, {"metering_escapes": {}},
    {"blocked": "socket:example.invalid"}, {"blocked": [42]}, {"blocked": ["x"] * 51},
    {"argv": {"command": "x"}}, {"argv": [42]},
    {"final": "false"}, {"agent_root": [True]},
]


@pytest.mark.parametrize("bad", BAD_TRACE_FIELDS)
def test_trace_api_rejects_malformed_evidence_without_using_event_allowance(api, monkeypatch, bad):
    from weightbench import server

    monkeypatch.setattr(server, "MAX_TRACE_EVENTS", 1)
    url, state, ledger, _ = api
    status, response = _post(url, "/trace", {"pid": 22, **bad})
    assert status == 400 and "trace" in response["error"]
    assert state.n_trace_events == 0 and not list(ledger.entries("runtime_trace"))
    assert ledger.verify() and state.used == {"forward": 0, "probe": 0}
    assert _post(url, "/trace", {"pid": 22, "flops": 10, "final": True})[0] == 200
    assert _post(url, "/trace", {"pid": 22, "final": True})[0] == 429
    assert state.n_trace_events == 1 and len(list(ledger.entries("runtime_trace"))) == 1


@pytest.mark.parametrize("bad", BAD_TRACE_FIELDS)
@pytest.mark.parametrize("source", ["file", "server"])
def test_malformed_stored_trace_is_quarantined_even_with_other_valid_traces(tmp_path, bad, source):
    transcript = tmp_path / "t.txt"
    transcript.write_text("python solve.py")
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    clean = {"pid": 11, "flops": 10, "blocked": []}
    (trace_dir / "11.json").write_text(json.dumps(clean))
    server = [{**clean, "final": True}]
    invalid = {"pid": 22, **bad}
    if source == "file":
        (trace_dir / "22.json").write_text(json.dumps(invalid))
    else:
        server.append(invalid)
    result = A.audit_run(transcript, trace_dir, tmp_path / "egress", server_traces=server,
                         transcript_format="text")
    assert result["quarantined"] and "trace_malformed" in result["flags"]
    assert result["trace"]["processes"] == 1 and result["trace"]["flops"] == 10
    assert len(result["trace"]["malformed"]) == 1


@pytest.mark.parametrize("contents", ["not JSON", "[]", "null", "42"])
def test_invalid_trace_documents_cannot_disappear_from_audit(tmp_path, contents):
    transcript = tmp_path / "t.txt"
    transcript.write_text("ls")
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "invalid.json").write_text(contents)
    result = A.audit_run(transcript, trace_dir, tmp_path / "none", transcript_format="text")
    assert result["quarantined"] and result["trace"]["malformed"] == ["invalid.json"]


def test_trace_validation_preserves_legacy_optional_fields_and_integral_floats(api, tmp_path):
    url, _, ledger, _ = api
    legacy = {"flops": 10., "weight_matmuls": None, "blocked": None,
              "pid": None, "final": None, "agent_root": None, "argv": None}
    assert _post(url, "/trace", legacy)[0] == 200
    payload = next(ledger.entries("runtime_trace"))["payload"]
    assert payload["flops"] == 10 and payload["blocked"] == [] and "pid" not in payload
    trace_dir = tmp_path / "legacy"
    trace_dir.mkdir()
    (trace_dir / "legacy.json").write_text(json.dumps(legacy))
    summary = A.trace_summary(trace_dir)
    assert summary["processes"] == 1 and summary["flops"] == 10 and summary["malformed"] == []


def test_trace_metadata_bounds_match_files_and_server_for_tamper_detection(api, tmp_path):
    url, _, ledger, _ = api
    event = {"pid": 22, "blocked": ["socket:" + "x" * 300], "argv": ["x" * 3000] * 8, "final": True}
    assert _post(url, "/trace", event)[0] == 200
    payload = next(ledger.entries("runtime_trace"))["payload"]
    assert len(payload["blocked"][0]) == 200 and len(payload["argv"]) == 4
    assert all(len(arg) == 2000 for arg in payload["argv"])
    trace_dir = tmp_path / "bounded"
    trace_dir.mkdir()
    (trace_dir / "22.json").write_text(json.dumps(event))
    summary = A.trace_summary(trace_dir, [payload])
    assert not summary["malformed"] and not summary["tampered"]
    assert A.trace_flags(summary, python_used=True) == {"blocked_network"}


# ------------------------------------------------------------------------------------- proxy
def _connect_via(proxy_port: int, host: str, port: int = 443) -> tuple[str, socket.socket]:
    s = socket.socket()               # raw socket: tests may monkeypatch socket.create_connection
    s.settimeout(5)
    s.connect(("127.0.0.1", proxy_port))
    s.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
    status = s.recv(128).decode().split("\r\n")[0]
    return status, s


def test_proxy_denies_unlisted_hosts_and_logs(tmp_path):
    proxy = S.EgressProxy(tmp_path / "egress.jsonl").start()
    try:
        assert " 403 " in _connect_via(proxy.port, "example.com")[0] + " "
        assert " 403 " in _connect_via(proxy.port, "api.anthropic.com", 80)[0] + " "     # wrong port
    finally:
        proxy.stop()
    log = [json.loads(l) for l in (tmp_path / "egress.jsonl").read_text().splitlines()]
    assert [(e["host"], e["allowed"]) for e in log] == [("example.com", False), ("api.anthropic.com", False)]


def test_proxy_tunnels_allowlisted_host(tmp_path, monkeypatch):
    echo = socket.socket()
    echo.bind(("127.0.0.1", 0))
    echo.listen(1)

    def serve():
        c, _ = echo.accept()
        c.sendall(c.recv(64).upper())
        c.close()

    threading.Thread(target=serve, daemon=True).start()
    real = socket.create_connection
    monkeypatch.setattr(S.socket, "create_connection",
                        lambda addr, timeout=None: real(("127.0.0.1", echo.getsockname()[1]), timeout=timeout))
    proxy = S.EgressProxy(tmp_path / "egress.jsonl").start()
    try:
        status, s = _connect_via(proxy.port, "api.anthropic.com")
        assert " 200 " in status + " "
        s.sendall(b"ping")
        assert s.recv(64) == b"PING"
    finally:
        proxy.stop()


# ------------------------------------------------------------------------------------- runtime guard
@pytest.fixture
def guard(tmp_path):
    pub = tmp_path / "public"
    pub.mkdir()
    np.savez(pub / "w.npz", W1=np.zeros((8, 4), np.float32), W2=np.zeros((3, 8), np.float32), b=np.zeros(8))
    root = S.install_guard(tmp_path / "guard", pub)
    return root


def run_guarded(root: Path, code: str, **extra_env) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(root / "lib"), "WB_TRACE_DIR": str(root / "trace"),
           "WB_SHAPES": str(root / "lib" / "shapes.json"), **extra_env}
    try:
        return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        # Preserve the child's faulthandler stack in JUnit instead of a truncated bytes repr.
        stdout = (error.stdout or b"").decode(errors="replace")
        stderr = (error.stderr or b"").decode(errors="replace")
        pytest.fail(f"guard subprocess timed out after {error.timeout}s\n{stdout}\n{stderr}")


def traces(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in (root / "trace").glob("*.json")]


def test_install_guard_writes_shapes(guard):
    assert json.loads((guard / "lib" / "shapes.json").read_text()) == [[3, 8], [8, 4]]
    assert (guard / "lib" / "sitecustomize.py").read_text() == S.GUARD_SOURCE.read_text()


def test_guard_blocks_internet_but_not_localhost(guard):
    code = """
import socket
srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
socket.create_connection(("127.0.0.1", srv.getsockname()[1]), timeout=2)
try:
    socket.create_connection(("1.1.1.1", 443), timeout=2); print("INTERNET_OK")
except PermissionError as e:
    print("BLOCKED", e)
"""
    out = run_guarded(guard, code)
    assert "BLOCKED" in out.stdout and "INTERNET_OK" not in out.stdout, out.stderr
    assert any(b.startswith("socket:1.1.1.1") for t in traces(guard) for b in t["blocked"])


def test_guard_blocks_simulators_and_keeps_optional_imports_working(guard):
    code = """
try:
    import minatar; print("IMPORTED")
except ModuleNotFoundError as e:
    print("BLOCKED", e)
try:
    import jax
except ImportError:
    print("OPTIONAL_OK")
"""
    out = run_guarded(guard, code)
    assert "BLOCKED" in out.stdout and "WeightBench sandbox" in out.stdout and "OPTIONAL_OK" in out.stdout, out.stderr


def test_guard_blocks_gradient_descent(guard):
    code = """
import faulthandler
faulthandler.dump_traceback_later(45)
import torch
x = torch.ones(3, requires_grad=True)
for name, fn in [("backward", lambda: (x * 2).sum().backward()),
                 ("grad", lambda: torch.autograd.grad((x * 2).sum(), x)),
                 ("step", lambda: torch.optim.SGD([x], lr=0.1).step()),
                 ("adam", lambda: torch.optim.Adam([x]))]:
    try:
        fn(); print(name, "RAN")
    except PermissionError:
        print(name, "BLOCKED")
faulthandler.cancel_dump_traceback_later()
"""
    out = run_guarded(guard, code)
    assert out.stdout.split() == ["backward", "BLOCKED", "grad", "BLOCKED", "step", "BLOCKED", "adam", "BLOCKED"], \
        out.stderr


def test_guard_traces_forward_chains_torch_and_numpy(guard):
    code = """
import numpy as np, torch
W1, W2 = torch.randn(8, 4), torch.randn(3, 8)
x = torch.randn(5, 4)
y = torch.relu(x @ W1.T) @ W2.T                     # hand-written 2-layer forward pass
a1, a2 = np.random.randn(8, 4), np.random.randn(3, 8)
np.matmul(np.maximum(np.matmul(np.random.randn(5, 4), a1.T), 0), a2.T)
u, s, vt = np.linalg.svd(a1)                       # static analysis: not a chain
"""
    out = run_guarded(guard, code)
    assert out.returncode == 0, out.stderr
    t = traces(guard)[0]
    assert t["forward_chains"] >= 2 and t["weight_matmuls"] >= 4 and t["flops"] > 0


def test_guard_meters_numpy_matmul_operator_on_loaded_weights(guard, tmp_path):
    """`@` is a C-level operator; weights loaded through np.load are a metered ndarray subclass, so a
    hand-written layer stack written with `@` is traced like the torch / np.matmul versions."""
    np.savez(tmp_path / "w.npz", W1=np.ones((8, 4), np.float32), W2=np.ones((3, 8), np.float32))
    code = (f"import numpy as np\nw = np.load(r'{tmp_path / 'w.npz'}')\nW1, W2 = w['W1'], w['W2']\n"
            "x = np.ones((5, 4), np.float32)\nh = np.maximum(x @ W1.T, 0)\ny = h @ W2.T\n"
            "u, s, vt = np.linalg.svd(W1)\nprint(type(y).__name__, float(y.sum()))\n")
    proc = run_guarded(guard, code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[0] == "WeightArray" and float(proc.stdout.split()[1]) == 5 * 3 * 32.0   # same values
    (t,) = traces(guard)
    assert t["weight_matmuls"] >= 2 and t["forward_chains"] >= 1 and t["flops"] > 0


def test_guard_metering_survives_asarray_and_records_plain_views(guard, tmp_path):
    np.savez(tmp_path / "w.npz", W1=np.ones((8, 4), np.float32))
    code = (f"import numpy as np\nW = np.load(r'{tmp_path / 'w.npz'}')['W1']\n"
            "a = np.asarray(W)\nb = np.array(W, dtype=np.float64)\nprint(type(a).__name__, type(b).__name__)\n"
            "plain = W.view(np.ndarray)\nprint(type(plain).__name__)\n")
    proc = run_guarded(guard, code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["WeightArray", "WeightArray", "ndarray"]
    (t,) = traces(guard)
    assert t["metering_escapes"] == 1
    rep = A.audit_run(write_transcript(tmp_path / "t.jsonl", [{"command": "python a.py"}]), guard / "trace",
                      tmp_path / "none.jsonl")
    assert "metering_escape" in rep["flags"] and rep["quarantined"]


def test_weights_the_task_allows_to_evaluate_are_not_metered(tmp_path):
    pub = tmp_path / "public"
    pub.mkdir()
    np.savez(pub / "host.npz", w=np.zeros((64, 64), np.float32))
    np.savez(pub / "donor.npz", W1=np.zeros((32, 20), np.float32), W2=np.zeros((10, 32), np.float32))
    assert S.weight_shapes(pub) == [[10, 32], [32, 20], [64, 64]]
    assert S.weight_shapes(pub, ["donor.npz"]) == [[64, 64]]
    root = S.install_guard(tmp_path / "guard", pub, ["donor.npz"])
    assert json.loads((root / "lib" / "shapes.json").read_text()) == [[64, 64]]
    from weightbench.tracks import tr04_graft

    assert tr04_graft.CircuitGrafting.meter_exclude == ("donor.npz",)


def test_guard_static_analysis_has_no_chain(guard):
    code = """
import numpy as np
W1, W2 = np.random.randn(8, 4), np.random.randn(3, 8)
np.matmul(W2, W1); np.linalg.svd(W1); np.dot(W1.T, W1)
"""
    out = run_guarded(guard, code)
    assert out.returncode == 0, out.stderr
    assert traces(guard)[0]["forward_chains"] == 0


def test_guard_flop_cap(guard):
    out = run_guarded(guard, "import numpy as np\nnp.matmul(np.ones((50, 50)), np.ones((50, 50)))", WB_FLOP_CAP="1000")
    assert out.returncode != 0 and "FLOP cap" in out.stderr


def test_einsum_shape_planner_hand_computed_cases(guard):
    cases = [
        ("ij,jk->ik", [(5, 4), (4, 8)], [320, True]),
        ("bi,oi->bo", [(5, 4), (8, 4)], [320, True]),
        ("ij,jk", [(5, 4), (4, 8)], [320, True]),
        ("...ij,...jk->...ik", [(2, 5, 4), (1, 4, 8)], [640, True]),
        ("bij,bjk->bik", [(1, 5, 4), (2, 4, 8)], [640, True]),
        ("...ij,...jk->...ik", [(0, 5, 4), (1, 4, 8)], [0, True]),
        ("i,i->", [(4,), (4,)], [8, True]),
        ("i,j->ij", [(4,), (3,)], [12, False]),
        ("ij,ij->ij", [(3, 4), (3, 4)], [12, False]),
        ("ij,ij->", [(3, 4), (3, 4)], [24, True]),
        ("ij->ji", [(3, 4)], [0, False]),
        ("ii->i", [(3, 3)], [0, False]),
        ("ii->", [(3, 3)], [2, False]),
        ("ij->i", [(3, 4)], [9, False]),
        ("...ij->...ji", [(2, 3, 4)], [0, False]),
        ("i,i,i->", [(4,), (4,), (4,)], None),
        ("ii,ij->j", [(4, 4), (4, 3)], None),
        ("...ij->j", [(2, 3, 4)], None),
        ("i,j->", [(4,), (3,)], None),
    ]
    code = "import json,sys\nplan=sys.modules['sitecustomize']._einsum_plan\n"
    code += f"cases=json.loads({json.dumps(cases)!r})\n"
    code += "print(json.dumps([plan(equation, shapes) for equation,shapes,expected in cases]))"
    result = run_guarded(guard, code)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [row[2] for row in cases]


@pytest.mark.parametrize("library,optimize", [("numpy", False), ("numpy", True), ("torch", False), ("torch-list", False)])
def test_einsum_forward_chain_and_exact_flop_proxy(guard, tmp_path, library, optimize):
    weights = tmp_path / "weights.npz"
    np.savez(weights, W1=np.ones((8, 4), np.float32), W2=np.ones((3, 8), np.float32))
    if library == "numpy":
        code = (f"import numpy as np\nw=np.load(r'{weights}')\nx=np.ones((5,4),np.float32)\n"
                f"h=np.einsum('bi,oi->bo',x,w['W1'],optimize={optimize})\n"
                "h=np.maximum(h,0)\n"
                f"y=np.einsum('bi,oi->bo',h,w['W2'],optimize={optimize})\n"
                "print(type(y).__name__,float(y.sum()))")
    else:
        code = ("import torch\nx=torch.ones(5,4); w1=torch.ones(8,4); w2=torch.ones(3,8)\n"
                "h=torch.relu(torch.einsum('bi,oi->bo',x,w1))\n"
                "y=torch.einsum('bi,oi->bo',h,w2)\nprint(type(y).__name__,float(y.sum()))")
        if library == "torch-list":
            code = ("import torch\nx=torch.ones(5,4); w1=torch.ones(8,4); w2=torch.ones(3,8)\n"
                    "h=torch.relu(torch.einsum(x,[0,1],w1,[2,1],[0,2]))\n"
                    "y=torch.einsum(h,[0,1],w2,[2,1],[0,2])\nprint(type(y).__name__,float(y.sum()))")
    result = run_guarded(guard, code)
    assert result.returncode == 0, result.stderr
    expected_type = "WeightArray" if library == "numpy" else "Tensor"
    assert result.stdout.split() == [expected_type, "480.0"]
    trace, = traces(guard)
    assert trace["flops"] == 560 and trace["weight_matmuls"] == 2 and trace["forward_chains"] == 1
    assert trace["unsupported_contractions"] == 0


@pytest.mark.parametrize("library", ["numpy", "torch"])
def test_einsum_static_weight_composition_does_not_make_forward_chain(guard, tmp_path, library):
    weights = tmp_path / "weights.npz"
    np.savez(weights, w=np.ones((8, 4), np.float32))
    code = (f"import numpy as np\nw=np.load(r'{weights}')['w']\ny=np.einsum('oi,oj->ij',w,w)\n"
            if library == "numpy" else
            "import torch\nw=torch.ones(8,4)\ny=torch.einsum('oi,oj->ij',w,w)\n")
    result = run_guarded(guard, code + "print(float(y.sum()))")
    assert result.returncode == 0 and float(result.stdout) == 128., result.stderr
    trace, = traces(guard)
    assert trace["flops"] == 256 and trace["weight_matmuls"] == 0 and trace["forward_chains"] == 0
    assert trace["unsupported_contractions"] == 0


@pytest.mark.parametrize("library", ["numpy", "torch"])
def test_einsum_ellipsis_broadcast_counts_all_batches(guard, library):
    code = ("import numpy as np\nx=np.ones((5,4)); w=np.ones((2,4,8))\ny=np.einsum('...ik,...kj->...ij',x,w)\n"
            if library == "numpy" else
            "import torch\nx=torch.ones(5,4); w=torch.ones(2,4,8)\ny=torch.einsum('...ik,...kj->...ij',x,w)\n")
    result = run_guarded(guard, code + "print(float(y.sum()))")
    assert result.returncode == 0 and float(result.stdout) == 320., result.stderr
    trace, = traces(guard)
    assert trace["flops"] == 640 and trace["weight_matmuls"] == 1
    assert trace["unsupported_contractions"] == 0


@pytest.mark.parametrize("library", ["numpy", "torch"])
def test_einsum_enforces_flop_cap_before_contraction(guard, library):
    code = ("import numpy as np\nx=np.ones((5,4)); w=np.ones((8,4))\nnp.einsum('bi,oi->bo',x,w)"
            if library == "numpy" else
            "import torch\nx=torch.ones(5,4); w=torch.ones(8,4)\ntorch.einsum('bi,oi->bo',x,w)")
    result = run_guarded(guard, code, WB_FLOP_CAP="300")
    assert result.returncode != 0 and "FLOP cap" in result.stderr
    trace, = traces(guard)
    assert trace["flops"] == 320 and "flop_cap" in trace["blocked"]


@pytest.mark.parametrize("library", ["numpy", "torch"])
@pytest.mark.parametrize("form", ["multioperand", "list"])
def test_einsum_forms_preserve_values_and_report_metering_policy(guard, api, tmp_path, library, form):
    url, _, ledger, _ = api
    prefix = "import numpy as lib\n" if library == "numpy" else "import torch as lib\n"
    ones = "lib.ones((2,3))" if library == "numpy" else "lib.ones(2,3)"
    second = "lib.ones((3,4))" if library == "numpy" else "lib.ones(3,4)"
    if form == "multioperand":
        code = prefix + "x=lib.ones(4)\ny=lib.einsum('i,i,i->',x,x,x)\n"
        expected = 4.
    else:
        code = prefix + f"a={ones}; b={second}\ny=lib.einsum(a,[0,1],b,[1,2],[0,2])\n"
        expected = 24.
    result = run_guarded(guard, code + "print(float(y.sum()))", WB_URL=url, WB_TOKEN="s3cret")
    assert result.returncode == 0 and float(result.stdout) == expected, result.stderr
    events = [entry["payload"] for entry in ledger.entries("runtime_trace")]
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("python solve.py")
    if library == "torch" and form == "list":
        # Torch normalizes this public calling convention before TorchFunctionMode dispatch.
        assert events[-1]["flops"] == 48 and all(e["unsupported_contractions"] == 0 for e in events)
        report = A.audit_run(transcript, guard / "trace", tmp_path / "egress", server_traces=events,
                             transcript_format="text")
        assert not report["quarantined"] and report["trace"]["unsupported_contractions"] == 0
        return
    assert any(e.get("unsupported_contractions") == 1 and not e.get("final") for e in events)
    for path in (guard / "trace").glob("*.json"):
        path.unlink()                      # the append-only signal must survive deleted local traces
    report = A.audit_run(transcript, guard / "trace", tmp_path / "egress", server_traces=events,
                         transcript_format="text")
    assert report["quarantined"] and "unsupported_contraction" in report["flags"]
    assert report["trace"]["unsupported_contractions"] == 1 and ledger.verify()


def test_einsum_weightarray_out_preserves_identity_and_values(guard, tmp_path):
    weights, output = tmp_path / "weights.npz", tmp_path / "output.npy"
    np.savez(weights, w=np.ones((8, 4), np.float32))
    np.save(output, np.zeros((5, 8), np.float32))
    code = (f"import numpy as np\nw=np.load(r'{weights}')['w']; out=np.load(r'{output}')\n"
            "result=np.einsum('bi,oi->bo',np.ones((5,4),np.float32),w,out=out)\n"
            "print(result is out,float(out.sum()))")
    result = run_guarded(guard, code)
    assert result.returncode == 0 and result.stdout.split() == ["True", "160.0"], result.stderr
    trace, = traces(guard)
    assert trace["flops"] == 320 and trace["unsupported_contractions"] == 0


@pytest.mark.parametrize("bad", [True, -1, "1", float("nan")])
def test_einsum_unsupported_counter_is_validated_at_api(api, bad):
    url, _, ledger, _ = api
    assert _post(url, "/trace", {"pid": 22, "unsupported_contractions": bad})[0] == 400
    assert not list(ledger.entries("runtime_trace"))


def test_weight_shapes_reads_safetensors_header(tmp_path):
    import torch
    from safetensors.torch import save_file

    save_file({"a": torch.zeros(4, 6), "v": torch.zeros(3)}, str(tmp_path / "m.safetensors"))
    np.save(tmp_path / "x.npy", np.zeros((2, 5)))
    assert S.weight_shapes(tmp_path) == [[2, 5], [4, 6]]


# ------------------------------------------------------------------------------------- audit
def write_transcript(path: Path, tool_inputs: list[dict], text: str = "") -> Path:
    content = [{"type": "text", "text": text}] + [{"type": "tool_use", "name": "Bash", "input": i} for i in tool_inputs]
    path.write_text(json.dumps({"type": "assistant", "message": {"content": content}}) + "\n"
                    + json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": "pip install x"}]}}) + "\n")
    return path


@pytest.mark.parametrize("command, flag", [
    ("pip install transformers", "network_attempt"),
    ("curl https://example.com", "network_attempt"),
    ("python -c 'import transformers'", "model_execution_library"),
    ("python -c 'from minatar import Environment'", "simulator_execution"),
    ("python -c 'loss.backward(); opt.step()'", "gradient_descent"),
    ("unset PYTHONPATH; python x.py", "guard_bypass_attempt"),
    ("python -S x.py", "guard_bypass_attempt"),
])
def test_transcript_flags(tmp_path, command, flag):
    flags = A.transcript_flags(write_transcript(tmp_path / "t.jsonl", [{"command": command}]))
    assert flag in flags and flag in A.QUARANTINE_FLAGS


def test_transcript_forward_pass_suspect_vs_static_analysis(tmp_path):
    fwd = "h = np.maximum(x @ W1.T + b1, 0)\nlogits = h @ W2.T"
    static = "u, s, vt = np.linalg.svd(W1)\nqk = Wq @ Wk.T\nprint(s[:5])"
    assert "forward_pass_suspect" in A.transcript_flags(write_transcript(tmp_path / "a.jsonl", [{"content": fwd}]))
    assert A.transcript_flags(write_transcript(tmp_path / "b.jsonl", [{"content": static}])) == set()


def test_user_messages_are_not_audited(tmp_path):
    assert A.transcript_flags(write_transcript(tmp_path / "t.jsonl", [{"command": "ls"}])) == set()


def test_audit_run_combines_sources(tmp_path):
    t = write_transcript(tmp_path / "t.jsonl", [{"command": "python solve.py"}])
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "1.json").write_text(json.dumps({"flops": 10, "weight_matmuls": 2, "forward_chains": 1,
                                              "blocked": ["import:minatar"]}))
    (tmp_path / "egress.jsonl").write_text(json.dumps({"host": "pypi.org", "port": 443, "allowed": False}) + "\n"
                                           + json.dumps({"host": "api.anthropic.com", "port": 443, "allowed": True}) + "\n")
    rep = A.audit_run(t, trace, tmp_path / "egress.jsonl")
    assert {"forward_chain_runtime", "blocked_import", "egress_denied"} <= set(rep["flags"]) and rep["quarantined"]
    assert rep["egress"] == {"allowed": 1, "denied": ["pypi.org:443"], "n_denied": 1}


def test_clean_run_not_quarantined_and_missing_trace_flagged(tmp_path):
    t = write_transcript(tmp_path / "t.jsonl", [{"command": "python solve.py"}])
    trace = tmp_path / "trace"
    trace.mkdir()
    rep = A.audit_run(t, trace, tmp_path / "none.jsonl")
    assert rep["flags"] == ["no_runtime_trace"] and rep["quarantined"]            # python ran, no trace
    (trace / "1.json").write_text(json.dumps({"flops": 1, "weight_matmuls": 0, "forward_chains": 0, "blocked": []}))
    rep = A.audit_run(t, trace, tmp_path / "none.jsonl")
    assert rep["flags"] == [] and not rep["quarantined"]


# ------------------------------------------------------------------------------------- report
def test_deleted_or_rewritten_trace_files_are_detected(tmp_path):
    t = write_transcript(tmp_path / "t.jsonl", [{"command": "python solve.py"}])
    trace = tmp_path / "trace"
    trace.mkdir()
    dirty = {"pid": 11, "flops": 900, "weight_matmuls": 4, "forward_chains": 2, "blocked": [], "final": True}
    clean = {"pid": 12, "flops": 10, "weight_matmuls": 0, "forward_chains": 0, "blocked": [], "final": True}
    (trace / "12.json").write_text(json.dumps(clean))                    # agent deleted 11.json
    rep = A.audit_run(t, trace, tmp_path / "none.jsonl", server_traces=[dirty, clean])
    assert {"trace_tampered", "forward_chain_runtime"} <= set(rep["flags"]) and rep["quarantined"]
    (trace / "11.json").write_text(json.dumps({**dirty, "forward_chains": 0}))   # ... or rewrote it
    assert "trace_tampered" in A.audit_run(t, trace, tmp_path / "none.jsonl", server_traces=[dirty, clean])["flags"]
    (trace / "11.json").write_text(json.dumps(dirty))
    rep = A.audit_run(t, trace, tmp_path / "none.jsonl", server_traces=[dirty, clean])
    assert "trace_tampered" not in rep["flags"] and rep["trace"]["processes"] == 2


@pytest.mark.parametrize("command", ["node -e 'require(\"fs\")'", "cc fwd.c -o fwd", "perl -e 1", "swift run.swift"])
def test_other_interpreters_flagged(tmp_path, command):
    assert "other_interpreter" in A.transcript_flags(write_transcript(tmp_path / "t.jsonl", [{"command": command}]))


def test_plain_shell_not_flagged_as_interpreter(tmp_path):
    flags = A.transcript_flags(write_transcript(tmp_path / "t.jsonl", [{"command": "ls -la && cat TASK.md"}]))
    assert "other_interpreter" not in flags


def test_guard_reports_blocked_events_to_server(guard, api):
    url, _, led, _ = api
    proc = run_guarded(guard, "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 443), timeout=1)\n"
                              "except PermissionError:\n    pass\n", WB_URL=url, WB_TOKEN="s3cret")
    assert proc.returncode == 0, proc.stderr
    events = [e["payload"] for e in led.entries("runtime_trace")]
    assert any("socket:1.1.1.1" in e["blocked"] and not e["final"] for e in events)   # sent when it happened
    assert any(e["final"] for e in events)                                            # and at exit


def test_guard_blocks_torch_func_gradients(guard):
    pytest.importorskip("torch")
    proc = run_guarded(guard, "import torch\nfrom torch.func import grad\n"
                              "try:\n    grad(lambda x: (x * x).sum())(torch.ones(2))\n    print('ran')\n"
                              "except PermissionError:\n    print('blocked')\n")
    assert proc.stdout.strip().endswith("blocked"), proc.stderr


def test_weightbench_score_is_macro_average_over_tracks():
    per_task = {"TR01-A": 1.0, "TR01-B": 1.0, "TR01-C": 1.0, "TR04-A": 0.0}
    assert report.macro_mean(per_task) == 0.5                 # not the task mean 0.75: every track weighs the same
    lo, hi = report.macro_bootstrap_ci(per_task)
    assert lo == hi == 0.5                                    # no within-track variance -> degenerate CI
    rows = [{"task": "TR01-A", "valid": True, "s_plus": 0.8, "d_col": 0.0, "e_comp": 5, "e_budget": 10,
             "edit_budget_use": 0.0, "s_wb": 0.7, "formula_version": 2}]
    report.official_scores(rows)
    assert rows[0]["s_wb_v2"] == 0.7 and rows[0]["s_wb"] == pytest.approx(0.8 * 0.95)


def test_report_scores_missing_tasks_as_zero(tmp_path, monkeypatch):
    """A system that skips or crashes on a task gets 0 there; it cannot raise its score by running fewer tasks."""
    def row(system, task, s_plus):
        return {"system": system, "agent": system, "task": task, "valid": True, "s_plus": s_plus, "d_col": 0.0,
                "e_comp": 0, "e_budget": 0, "edit_budget_use": 0.0, "s_wb": s_plus, "formula_version": 3,
                "audit_flags": [], "usage": {}}

    rows = [row("full", "TR04-A", 0.5), row("full", "TR04-B", 0.5), row("partial", "TR04-A", 0.9)]
    monkeypatch.setattr(report, "load", lambda: rows)
    monkeypatch.setattr(report, "OUT", tmp_path)
    report.main()
    summary = json.loads((tmp_path / "summary.json").read_text())["systems"]
    assert summary["partial"]["overall"] == pytest.approx(0.45)          # (0.9 + 0) / 2, not 0.9
    assert summary["partial"]["missing_tasks_scored_0"] == ["TR04-B"]
    assert "0.000 (not run)" in (tmp_path / "RESULTS.md").read_text(encoding="utf-8")


def test_paired_diff_ci_hand_computed():
    a = {"t1": 0.9, "t2": 0.8, "t3": 0.7}
    assert report.paired_diff_ci(a, {k: v - 0.2 for k, v in a.items()}) == pytest.approx((0.2, 0.2))
    assert report.paired_diff_ci({"t1": 1.0}, {"t2": 0.0}) == (0.0, 0.0)          # no shared tasks


def test_tie_groups():
    tasks = [f"t{i}" for i in range(8)]
    strong = {t: 0.9 for t in tasks}
    tied = {t: 0.9 + (0.05 if i % 2 else -0.05) for i, t in enumerate(tasks)}    # same mean, noisy
    weak = {t: 0.2 for t in tasks}
    groups = report.tie_groups({"A": strong, "B": tied, "C": weak, "D": {}}, ["A", "B", "C", "D"])
    assert groups["A"] == groups["B"] == 1 and groups["C"] == 2 and "D" not in groups


def test_report_zeroes_quarantined_until_cleared(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RUNS", tmp_path)
    for name, quarantined in (("a", True), ("b", False)):
        d = tmp_path / name
        d.mkdir()
        (d / "result.json").write_text(json.dumps({"task": "TR04-GO-001", "agent": "claude", "model": "m",
                                                   "s_wb": 0.8, "s_plus": 0.9, "quarantined": quarantined}))
    rows = {r["s_wb"] for r in report.load()}
    assert rows == {0.0, 0.8}
    (tmp_path / "a" / "review.json").write_text('{"cleared": true}')
    assert {r["s_wb"] for r in report.load()} == {0.8}
