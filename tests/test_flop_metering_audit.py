"""FLOP metering and forward-pass audit: metering through copies, ReLU forms vs. bound propagation, the FLOP cap end to end,
reference variants in the ledger, and guard-variable tampering.

Model-free: tiny NumPy arrays in guarded child processes, synthetic transcripts, a stub agent launcher. Every
detection has a clean control next to it (legitimate static weight analysis must stay unflagged)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from weightbench import leaderboard, report, runner, verify
from weightbench.harness import agents, audit, sandbox
from weightbench.harness.trace_schema import normalize_trace
from weightbench.ledger import Ledger
from weightbench.task import Budget, Evaluation, Submission, Task

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def guard(tmp_path):
    """Mixed-sign weights (as in TR05-DEAD), so sign splits change them."""
    public = tmp_path / "public"
    public.mkdir()
    rng = np.random.default_rng(0)
    np.savez(public / "mlp.npz", W1=rng.normal(size=(8, 4)).astype(np.float32),
             b1=rng.normal(size=8).astype(np.float32), W2=rng.normal(size=(3, 8)).astype(np.float32),
             b2=rng.normal(size=3).astype(np.float32), W3=rng.normal(size=(1, 3)).astype(np.float32),
             b3=np.zeros(1, np.float32))
    return sandbox.install_guard(tmp_path / "guard", public), public


def _run(guard, code: str, extra_env: dict | None = None, flop_cap_file: float | None = None):
    root, public = guard
    if flop_cap_file is not None:
        (root / "lib" / "flop_cap.json").write_text(json.dumps(flop_cap_file))
    env = {k: v for k, v in os.environ.items() if not k.startswith("WB_")}
    env.update(PYTHONPATH=str(root / "lib"), WB_TRACE_DIR=str(root / "trace"),
               WB_SHAPES=str(root / "lib" / "shapes.json"), **(extra_env or {}))
    result = subprocess.run([sys.executable, "-c", code], cwd=public, env=env, capture_output=True, text=True,
                            timeout=120)
    traces = [json.loads(p.read_text()) for p in sorted((root / "trace").glob("*.json"))]
    return result, traces


def _flags(guard) -> set[str]:
    return audit.trace_flags(audit.trace_summary(guard[0] / "trace"), True)


LOAD = ("import numpy as np\nz = np.load('mlp.npz')\nW1, W2, b1, b2 = z['W1'], z['W2'], z['b1'], z['b2']\n"
        "x = np.linspace(-1, 1, 20).reshape(5, 4).astype(np.float32)\n")


# ---- 1. copies out of the metered type ---------------------------------------------------------------------------

@pytest.mark.parametrize("copy", [
    "p = np.frombuffer(W1.tobytes(), W1.dtype).reshape(W1.shape)",
    "p = np.frombuffer(buffer=W1.tobytes(), dtype=W1.dtype).reshape(W1.shape)",
    "p = np.array(W1.tolist(), np.float32)",
    "p = np.asarray(W1.tolist())",
    "p = np.asarray(memoryview(W1))",
    "p = np.frombuffer(W1.data, W1.dtype).reshape(W1.shape)",
    "W1.tofile('w.bin'); p = np.fromfile('w.bin', np.float32).reshape(W1.shape)",
])
def test_copies_through_bytes_lists_and_buffers_stay_metered(guard, copy):
    result, traces = _run(guard, LOAD + copy + "\nprint(type(p).__name__)\ny = p @ p.T\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["WeightArray"]
    assert traces[0]["flops"] == 2 * 8 * 4 * 8 and traces[0]["metering_escapes"] == 0


def test_copyto_of_a_whole_weight_into_a_plain_array_is_a_metering_escape(guard):
    result, traces = _run(guard, LOAD + "d = np.empty(W1.shape, W1.dtype)\nnp.copyto(d, W1)\ny = d @ d.T\n")
    assert result.returncode == 0, result.stderr
    assert traces[0]["metering_escapes"] == 1 and traces[0]["flops"] == 0     # the product was not metered
    assert "metering_escape" in _flags(guard)


@pytest.mark.parametrize("body", [
    "print(float(W1[0, 0]), W1.shape, W1[:2, :2], float(np.linalg.norm(W1)), W1.sum(axis=0))\n",
    "print(W1.tolist()[0][:2], len(W1.tobytes()), W1)\n",
    "d = np.empty_like(W1); np.copyto(d, W1)\ne = np.zeros(4, np.float32); np.copyto(e, W1[0])\n",
    "edit = W1.copy(); edit[np.abs(edit) < 0.1] = 0\nnp.savez('delta.npz', W1=edit - W1)\n",
    "np.save('d.npy', W1 * 0.5)\nwith open('d.raw', 'wb') as f:\n    f.write((W1 - W1).tobytes())\n",
])
def test_reading_printing_and_saving_weights_stays_clean(guard, body):
    result, traces = _run(guard, LOAD + body)
    assert result.returncode == 0, result.stderr
    assert traces[0]["metering_escapes"] == 0 and traces[0]["forward_chains"] == 0
    assert _flags(guard) == set()


def test_safetensors_submission_save_stays_clean(guard):
    pytest.importorskip("safetensors.numpy")
    code = LOAD + ("from safetensors.numpy import save_file, load_file\n"
                   "save_file({'W1': (W1 * 0.5 - W1).astype(np.float32)}, 'delta.safetensors')\n"
                   "print(type(load_file('delta.safetensors')['W1']).__name__)\n")
    result, traces = _run(guard, code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["WeightArray"]
    assert traces[0]["metering_escapes"] == 0 and _flags(guard) == set()


# ---- 2. ReLU forms vs. bound propagation -------------------------------------------------------------------------

@pytest.mark.parametrize("relu", [
    "h = h * (h > 0)",
    "h = h * (h > 0).astype(h.dtype)",
    "h = np.clip(h, 0, None)",
    "h = h.clip(0)",
    "h = (h + abs(h)) / 2",
    "h = np.fmax(h, 0)",
    "h = np.where(h > 0, h, 0)",
    "h[h < 0] = 0",
    "np.maximum(h, 0, out=h)",
    "np.putmask(h, h < 0, 0)",
    "h = np.maximum(np.ndarray.view(h, np.ndarray), 0)",     # stripped activation: the order rule still applies
])
def test_relu_forms_make_a_forward_chain(guard, relu):
    result, traces = _run(guard, LOAD + f"h = x @ W1.T + b1\n{relu}\ny = h @ W2.T + b2\n")
    assert result.returncode == 0, result.stderr
    assert traces[0]["forward_chains"] == 1
    assert "forward_chain_runtime" in _flags(guard)


IBP_REFERENCE = ("u = np.ones(4); l = np.zeros(4)\n"
                 "for W, b in ((W1, b1), (W2, b2)):\n"
                 "    up = np.maximum(W, 0) @ u + np.minimum(W, 0) @ l + b\n"
                 "    lo = np.maximum(W, 0) @ l + np.minimum(W, 0) @ u + b\n"
                 "    u, l = np.maximum(up, 0), np.maximum(lo, 0)\n"
                 "print(u, l)\n")


@pytest.mark.parametrize("body", [
    IBP_REFERENCE,
    # center / radius form: W @ c is a plain-weight product, but it is mixed with |W| @ r (a bound)
    "c = np.full(4, .5); r = np.full(4, .5)\nfor W, b in ((W1, b1), (W2, b2)):\n"
    "    m = W @ c + b; d = np.abs(W) @ r\n    lo, up = np.maximum(m - d, 0), np.maximum(m + d, 0)\n"
    "    c, r = (lo + up) / 2, (up - lo) / 2\n",
    # sign split written with np.where / boolean masks
    "u = np.ones(4)\nfor W, b in ((W1, b1), (W2, b2)):\n"
    "    u = np.maximum(np.where(W > 0, W, 0) @ u + b, 0)\n",
    "u = np.ones(4)\nfor W, b in ((W1, b1), (W2, b2)):\n    u = np.clip((W * (W > 0)) @ u + b, 0, None)\n",
    # TR05-DEAD interval bounds written as sums, a masked weight delta, power iteration
    "lo = np.where(W1 > 0, 0, W1).sum(axis=1); hi = np.where(W1 > 0, W1, 0).sum(axis=1); print(lo, hi)\n",
    "mask = np.where(np.abs(W1) < 0.5, 0, W1); np.savez('delta.npz', W1=mask - W1)\n",
    "v = np.ones(4)\nfor _ in range(10):\n    v = W1.T @ (W1 @ v); v = v / np.maximum(np.linalg.norm(v), 1e-12)\n",
])
def test_bound_propagation_and_static_analysis_are_not_forward_chains(guard, body):
    result, traces = _run(guard, LOAD + body)
    assert result.returncode == 0, result.stderr
    assert traces[0]["forward_chains"] == 0
    assert _flags(guard) == set()


TR05_LAYERS = (f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
               "layers = [(np.asarray(z[f'W{i}'], np.float64), np.asarray(z[f'b{i}'], np.float64)) for i in (1, 2, 3)]\n")


@pytest.mark.parametrize("call", [
    "from weightbench.tracks.tr05_dead_lib import net\nprint(len(net.ibp_upper(layers)))\n",
    "from weightbench.tracks.tr05_dead_lib import crown\nprint(len(crown.bounds(layers, use_lp=False)[0]))\n",
    "from weightbench.tracks.tr05_dead_lib import crown\nprint(len(crown.bounds(layers, use_lp=crown.have_lp())[0]))\n",
])
def test_tr05_dead_references_run_clean_under_the_guard(guard, call):
    """The two TR05-DEAD references (interval bounds; CROWN back-substitution + LP) are zero-forward static analyses.
    A purely order-based rule would count 5 forward chains for the interval-bound pattern alone."""
    result, traces = _run(guard, LOAD + TR05_LAYERS + call)
    assert result.returncode == 0, result.stderr
    assert sum(t["forward_chains"] for t in traces) == 0 and _flags(guard) == set()


def test_tr05_forward_enumeration_is_a_forward_chain(guard):
    code = LOAD + TR05_LAYERS + ("from weightbench.tracks.tr05_dead_lib import net\n"
                                 "print(net.forward(layers, net.all_inputs(4)).shape)\n")
    result, traces = _run(guard, code)
    assert result.returncode == 0, result.stderr
    assert traces[0]["forward_chains"] >= 1 and "forward_chain_runtime" in _flags(guard)


def _text_flags(tmp_path, code: str) -> set[str]:
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": code}}]}}) + "\n", encoding="utf-8")
    return audit.transcript_flags(path)


@pytest.mark.parametrize("code", [
    "h = np.maximum(x @ W1.T + b1, 0)\ny = h @ W2.T",
    "h = np.clip(x @ W1.T, 0, None)\ny = h @ W2.T",
    "h = np.fmax(np.matmul(x, W1.T), 0)\ny = h @ W2.T",
    "h = torch.relu(x @ W1.T)\ny = h @ W2.T",
])
def test_text_layer_flags_written_out_forward_passes(tmp_path, code):
    assert "forward_pass_suspect" in _text_flags(tmp_path, code)


@pytest.mark.parametrize("code", [
    "u1 = np.maximum(W1, 0) @ u + np.minimum(W1, 0) @ l\nhu, hl = np.maximum(u1, 0), np.maximum(l1, 0)\n"
    "u2 = np.maximum(W2, 0) @ hu + np.minimum(W2, 0) @ hl",
    "pu, nu = np.maximum(lam, 0), np.minimum(lam, 0)\noff = off + pu @ c\nlam = (pu * a) @ W",
    "u, s, vt = np.linalg.svd(W1)\nqk = Wq @ Wk.T\nprint(s[:5])",
])
def test_text_layer_leaves_bound_propagation_clean(tmp_path, code):
    assert _text_flags(tmp_path, code) == set()


@pytest.mark.parametrize("code", [
    # a MILP-style script: "relu" is in a comment next to interval-bound matmuls
    "# upper bound of the pre-activation (h = relu(z)) over the input box\n"
    "up = np.maximum(W1, 0) @ u + np.minimum(W1, 0) @ l\n"
    "up2 = np.maximum(W2, 0) @ np.maximum(up, 0) + np.minimum(W2, 0) @ np.maximum(lo, 0)\n",
    # activation tokens only inside string literals
    "note = 'y = relu(x @ W1) @ W2'\nup = np.maximum(W1, 0) @ u\nlo = np.minimum(W1, 0) @ u\n",
    '''desc = """compute relu(h) then h @ W2"""\na = np.dot(W2, W1)\nb = W2 @ W1\n''',
])
def test_relu_in_comments_or_strings_next_to_matmuls_is_not_a_forward_pass(tmp_path, code):
    assert _text_flags(tmp_path, code) == set()


@pytest.mark.parametrize("code", [
    "h = relu(x @ W1.T)\ny = h @ W2.T",                         # a real relu() call is still seen
    "h = np.maximum(z @ W1, 0)  # forward\nout = h @ W2",       # real forward; the comment does not hide it
])
def test_real_relu_call_in_code_is_still_flagged(tmp_path, code):
    assert "forward_pass_suspect" in _text_flags(tmp_path, code)


def test_non_python_snippet_falls_back_to_raw_scan(tmp_path):
    # Not valid Python (shell wrapper): the tokenizer fails and the raw text is scanned, so a forward pass is caught.
    code = "bash -c 'run' ; h = np.maximum(x @ W1.T, 0) ; y = h @ W2.T"
    assert "forward_pass_suspect" in _text_flags(tmp_path, code)


# ---- guard variables: cap file, environment tampering ------------------------------------------------------------

def test_cap_file_beats_a_larger_environment_cap(guard):
    code = "import numpy as np\nnp.matmul(np.ones((50, 50)), np.ones((50, 50)))\n"
    result, traces = _run(guard, code, {"WB_FLOP_CAP": "1e12"}, flop_cap_file=1000)
    assert result.returncode != 0 and "FLOP cap" in result.stderr
    assert "flop_cap" in traces[0]["blocked"]


def test_environment_can_only_lower_the_cap_and_small_work_stays_clean(guard):
    small = "import numpy as np\nnp.matmul(np.ones((2, 2)), np.ones((2, 2)))\n"
    result, traces = _run(guard, small, {"WB_FLOP_CAP": "1e12"}, flop_cap_file=1000)
    assert result.returncode == 0, result.stderr
    assert traces[0]["blocked"] == [] and _flags(guard) == set()
    big = "import numpy as np\nnp.matmul(np.ones((5, 5)), np.ones((5, 5)))\n"     # 250 FLOPs
    result, _ = _run(guard, big, {"WB_FLOP_CAP": "100"}, flop_cap_file=1000)
    assert result.returncode != 0 and "FLOP cap" in result.stderr


def test_child_gets_the_guard_variables_back_after_in_process_edits(guard, tmp_path):
    other = tmp_path / "elsewhere"
    code = ("import os, subprocess, sys\n"
            f"os.environ['WB_TRACE_DIR'] = {str(other)!r}\nos.environ['WB_FLOP_CAP'] = '1e30'\n"
            "child = 'import numpy as np; np.matmul(np.ones((50, 50)), np.ones((50, 50)))'\n"
            "r1 = subprocess.run([sys.executable, '-c', child]).returncode\n"
            "r2 = subprocess.run([sys.executable, '-c', child], env={**os.environ, 'WB_FLOP_CAP': '1e30'}).returncode\n"
            "print(r1 != 0, r2 != 0)\n")
    result, traces = _run(guard, code, {"WB_FLOP_CAP": "1000"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True"]       # both children kept the 1000-FLOP cap
    assert not other.exists()                              # and wrote their traces to the real trace directory
    assert sum("flop_cap" in t["blocked"] for t in traces) == 2


@pytest.mark.parametrize("command", [
    "WB_FLOP_CAP=1e30 python solve.py",
    "export WB_GUARD_LIB=/tmp/x && python solve.py",
    "WB_GUARD_LIB= PYTHONPATH= python solve.py",
    "os.environ['WB_TRACE_DIR'] = '/tmp/x'",
    "os.environ.pop('WB_URL')",
    "subprocess.run(cmd, env={'WB_FLOP_CAP': '1e30'})",
])
def test_text_layer_flags_guard_variable_writes(tmp_path, command):
    assert "guard_bypass_attempt" in _text_flags(tmp_path, command)


@pytest.mark.parametrize("command", [
    "print(os.environ.get('WB_TRACE_DIR'))",
    "d = os.environ['WB_TRACE_DIR']\nif os.environ.get('WB_URL') == '': pass",
    "PYTHONPATH=. python solve.py",
])
def test_text_layer_guard_variable_reads_stay_clean(tmp_path, command):
    assert _text_flags(tmp_path, command) == set()


# ---- 3. flop_cap through the real guard -> server -> ledger path -------------------------------------------------

CAP_CHEAT = ("import numpy as np\nz = np.load('mlp.npz')\nW = z['W1']\nx = np.ones((64, 16), np.float32)\n"
             "for _ in range(50):\n    x = (x @ W.T) @ W\n")
CAP_CONTROL = ("import numpy as np\nz = np.load('mlp.npz')\nW = z['W1']\n"
               "y = W @ np.ones((16, 1), np.float32)\nprint(float(np.linalg.svd(W, compute_uv=False)[0]))\n")


def _cap_run(tmp_path, monkeypatch, code: str, sample: int) -> dict:
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "WORK", tmp_path / "work")
    monkeypatch.setattr(runner, "GUARDS", tmp_path / "guards")
    monkeypatch.setattr(runner, "ART", tmp_path / "art")
    seen = {}

    def launch(spec, model, workdir, transcript, wall_clock, turns, guard_root, ports, flop_cap, url, token):
        """Stand-in for the sandboxed agent: one guarded Python child with the runner's real agent environment."""
        env = sandbox.agent_env({k: v for k, v in os.environ.items() if not k.startswith("WB_")},
                                tmp_path / "no-venv", guard_root, ports["proxy"], flop_cap, wb_url=url, wb_token=token)
        env.pop("WB_FLOP_CAP")         # the cap must come from the guard's own file, not the variable
        seen["cap"] = flop_cap
        (workdir / "work.py").write_text(code)
        done = subprocess.run([sys.executable, "work.py"], cwd=workdir, env=env, capture_output=True, text=True,
                              timeout=120)
        seen["stderr"] = done.stderr
        transcript.write_text(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "python3 work.py"}}]}}) + "\n")
        return done.returncode

    monkeypatch.setattr(runner, "_launch_agent", launch)
    spec = agents.get("cmd", "unused", local_ports=[8000], transcript_format="claude-stream-json")
    result = runner.run("SMOKE-CAP-000", "cmd", "fixture", sample=sample, agent_spec=spec)
    result["_seen"] = seen
    return result


def test_flop_cap_is_enforced_and_ledgered_end_to_end(tmp_path, monkeypatch):
    result = _cap_run(tmp_path, monkeypatch, CAP_CHEAT, 0)
    assert result["_seen"]["cap"] == 100_000 and "FLOP cap" in result["_seen"]["stderr"]
    assert result["quarantined"] and result["audit_flags"] == ["flop_cap_exceeded"]
    led = Ledger(runner.RUNS / result["run_id"] / "ledger.jsonl")
    assert led.verify()
    assert next(led.entries("run_start"))["payload"]["flop_cap"] == 100_000
    assert any("flop_cap" in e["payload"].get("blocked", []) for e in led.entries("runtime_trace"))
    assert verify.verify_run(runner.RUNS / result["run_id"]).ok


def test_flop_cap_control_under_the_cap_is_clean(tmp_path, monkeypatch):
    result = _cap_run(tmp_path, monkeypatch, CAP_CONTROL, 1)
    assert result["_seen"]["stderr"] == ""
    assert not result["quarantined"] and result["audit_flags"] == []


def test_redteam_cap_scenario_targets_the_capped_task_and_stays_under_or_over_the_cap():
    from weightbench.harness import redteam_agent as ra
    from weightbench.task import REGISTRY, load_all

    load_all()
    cap = REGISTRY[ra.SCENARIO_TASK["flop_cap_exceeded"]].budget.flop_cap
    assert cap == REGISTRY[ra.SCENARIO_TASK["control_flop_cap_exceeded"]].budget.flop_cap == 100_000
    per_iteration = 2 * (2 * 64 * 16 * 16)                    # cheat: (x @ W.T) @ W on a batch of 64
    assert 2 * 16 * 16 * 1 < cap < 2 * per_iteration           # control: one W @ v; cheat: over within 2 iterations


def test_no_benchmark_task_sets_a_cap_and_the_cli_cannot_set_one():
    from weightbench.task import REGISTRY, load_all

    load_all()
    capped = sorted(t for t, cls in REGISTRY.items() if cls.budget.flop_cap is not None)
    assert capped == ["SMOKE-CAP-000"]
    with pytest.raises(SystemExit):
        runner.main(["run", "--task", "SMOKE-CAP-000", "--agent", "reference", "--flop-cap", "1e30"])


# ---- 6. server: the checked file is the copied file --------------------------------------------------------------

def test_submission_copy_rejects_a_file_swapped_after_the_check(tmp_path, monkeypatch):
    from weightbench import server

    work, private = tmp_path / "work", tmp_path / "private"
    work.mkdir()
    private.mkdir()
    np.savez(work / "delta.npz", W1=np.zeros(2))
    np.savez(private / "reference.npz", W1=np.ones(2))
    assert server._copy_checked((work / "delta.npz").resolve(), tmp_path / "ok.npz", work)
    assert np.load(tmp_path / "ok.npz")["W1"].tolist() == [0.0, 0.0]
    real_stat = os.stat
    # The check saw the private file's identity: what gets opened (delta.npz) differs, so nothing is copied. This is
    # the same mismatch a swap between check and open produces.
    monkeypatch.setattr(server.os, "stat", lambda p, **k: real_stat(private / "reference.npz", **k))
    assert not server._copy_checked((work / "delta.npz").resolve(), tmp_path / "bad.npz", work)
    monkeypatch.setattr(server.os, "stat", real_stat)     # (server.os is the os module: restore before exists())
    assert not (tmp_path / "bad.npz").exists()
    assert not server._copy_checked((private / "reference.npz").resolve(), tmp_path / "out.npz", work)
    # A swap during the copy (the copy reads the private file through a new symlink): dst differs from the verified
    # open file, so it is deleted and the copy refused.
    monkeypatch.setattr(server.shutil, "copy", lambda src, dst: Path(dst).write_bytes(
        (private / "reference.npz").read_bytes()))
    assert not server._copy_checked((work / "delta.npz").resolve(), tmp_path / "raced.npz", work)
    assert not (tmp_path / "raced.npz").exists()


@pytest.mark.skipif(os.name != "posix", reason="O_NOFOLLOW and unprivileged symlinks are POSIX")
def test_submission_copy_refuses_a_symlink_as_final_component(tmp_path):
    from weightbench import server

    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "secret.npz").write_bytes(b"private")
    (work / "delta.npz").symlink_to(tmp_path / "secret.npz")
    # resolve() follows the link out of the workdir; an unresolved path must not be followed by the open either.
    assert not server._copy_checked((work / "delta.npz").resolve(), tmp_path / "o1", work)
    assert not server._copy_checked(work / "delta.npz", tmp_path / "o2", work)


def test_server_submit_and_probe_file_still_work(tmp_path):
    from weightbench import server

    work, run = tmp_path / "work", tmp_path / "run"
    work.mkdir()
    np.savez(work / "delta.npz", W1=np.zeros(2))
    class WriteVariant(VariantTask):   # /submit only takes a ΔW from a task that can score one
        def base_weights(self, art_dir):
            return {"W1": np.zeros(2)}

    state = server.RunState(WriteVariant(), tmp_path, run, Ledger(tmp_path / "ledger.jsonl"), work)
    assert state.submit({"type": "delta_w", "path": "delta.npz"})[0] == 200
    assert (run / "submission" / "delta.npz").exists()
    assert state.submit({"type": "delta_w", "path": "../ledger.jsonl"})[0] == 400
    assert state._private_copy("delta.npz") is not None and state._private_copy("../ledger.jsonl") is None


# ---- 4. reference variants -----------------------------------------------------------------------------------------

class VariantTask(Task):
    id, track, tier = "TR99-VARIANT-TEST", "TR05", "foundational"
    budget = Budget(n_forward=0, n_probe=0)
    submission = "report"
    reference_variants = ("ibp", "crown")
    built: list = []

    def default_reference_variant(self):
        return os.environ.get("WB_TEST_REFERENCE") or "ibp"

    def prompt(self, art_dir):
        return "fixture"

    def evaluate(self, art_dir, sub):
        return Evaluation({"ibp": 0.5, "crown": 1.0}[sub.report["variant"]], details={"v": sub.report["variant"]})

    def baseline(self, art_dir):
        return Submission("report", report={"variant": "ibp"})

    def reference(self, art_dir, variant=None):
        return Submission("report", report={"variant": variant or self.default_reference_variant()})


@pytest.fixture
def variant_env(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "get_task", lambda *a: VariantTask())
    monkeypatch.setattr(runner, "nondefault_level", lambda *a: None)
    builds = []
    art = tmp_path / "art"
    (art / "public").mkdir(parents=True)
    monkeypatch.setattr(runner, "build", lambda *a, **k: builds.append(a) or art)
    from weightbench.task import REGISTRY, load_all

    load_all()
    monkeypatch.setitem(REGISTRY, VariantTask.id, VariantTask)     # verify_run looks the task up
    return builds


def test_variant_is_recorded_in_ledger_and_result(variant_env):
    res = runner.run(VariantTask.id, "reference", sample=0, variant="crown")
    assert res["variant"] == "crown" and res["default_variant"] == "ibp" and res["s_plus"] == 1.0
    led = Ledger(runner.RUNS / res["run_id"] / "ledger.jsonl")
    assert next(led.entries("run_start"))["payload"]["variant"] == "crown"
    assert next(led.entries("submission"))["payload"]["variant"] == "crown"
    assert verify.verify_run(runner.RUNS / res["run_id"]).ok
    default = runner.run(VariantTask.id, "reference", sample=1)
    assert default["variant"] == "ibp" and default["s_plus"] == 0.5
    assert report.floor_system(res) == "reference-crown" and report.floor_system(default) == "reference"
    assert leaderboard.system_name(res) == "reference-crown" and leaderboard.system_name(default) == "reference"


def test_env_var_still_selects_the_variant_and_is_recorded(variant_env, monkeypatch):
    monkeypatch.setenv("WB_TEST_REFERENCE", "crown")
    assert runner.run(VariantTask.id, "reference", sample=2)["variant"] == "crown"
    monkeypatch.setenv("WB_TEST_REFERENCE", "bogus")
    with pytest.raises(ValueError, match="unknown reference variant 'bogus'"):
        runner.run(VariantTask.id, "reference", sample=3)


def test_unknown_variant_is_rejected_before_build_or_evaluation(variant_env):
    with pytest.raises(ValueError, match="unknown reference variant 'nope'"):
        runner.run(VariantTask.id, "reference", sample=4, variant="nope")
    with pytest.raises(ValueError, match="--agent reference only"):
        runner.run(VariantTask.id, "baseline", sample=5, variant="crown")
    assert variant_env == [] and not runner.RUNS.exists()


def test_cli_variant_only_for_reference_runs():
    with pytest.raises(SystemExit):
        runner.main(["run", "--task", "TR05-DEAD-001", "--agent", "baseline", "--variant", "crown"])
    with pytest.raises(SystemExit):
        runner.main(["bench", "--model", "m", "--variant", "crown"])


def test_tr05_dead_declares_its_variants_and_rejects_unknown_ones_before_build(monkeypatch, tmp_path):
    from weightbench.tracks import tr05_dead

    task = tr05_dead.DeadOneLayer()
    assert task.reference_variants == ("ibp", "crown")
    monkeypatch.delenv("WB_DEAD_REFERENCE", raising=False)
    assert runner.resolve_variant(task, "reference", None) == "ibp"
    monkeypatch.setenv("WB_DEAD_REFERENCE", "crown")
    assert runner.resolve_variant(task, "reference", None) == "crown"
    assert runner.resolve_variant(task, "claude", None) is None
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "build", lambda *a, **k: pytest.fail("built before validating the variant"))
    with pytest.raises(ValueError, match="declared: ibp, crown"):
        runner.run("TR05-DEAD-001", "reference", variant="alpha")


def test_legacy_reference_result_without_variant_still_verifies_rescores_and_names(variant_env, monkeypatch):
    res = runner.run(VariantTask.id, "reference", sample=6)
    directory = runner.RUNS / res["run_id"]
    # Rewrite it as a run recorded before variants existed: no variant in the ledger or result.json.
    old = Ledger(directory / "ledger.jsonl")
    entries = list(old.entries())
    (directory / "ledger.jsonl").unlink()
    legacy = Ledger(directory / "ledger.jsonl")
    for e in entries:
        legacy.append(e["kind"], {k: v for k, v in e["payload"].items() if k not in ("variant", "default_variant")})
    result = {k: v for k, v in res.items() if k not in ("variant", "default_variant")}
    (directory / "result.json").write_text(json.dumps(result))
    assert verify.verify_run(directory).ok
    assert report.floor_system(result) == "reference" and leaderboard.system_name(result) == "reference"
    assert runner.rescore_all() == 1
    rescored = json.loads((directory / "result.json").read_text())
    assert "variant" not in rescored and rescored["s_plus"] == result["s_plus"] and rescored["ledger_ok"]


# ---- A. form-independent compute accounting ----------------------------------------------------------------------

def test_guarded_process_cpu_and_rss_are_recorded(guard):
    """A CPU-burning loop (no products, no recognised nonlinearity) still shows up as CPU seconds, whatever its
    form."""
    code = ("import numpy as np\nz = np.load('mlp.npz')\nw = z['W1'].copy()\n"
            "for _ in range(200):\n    for _ in range(300):\n        w = w * 1.0000001 + 1e-9\n"
            "print(float(w.sum()) != 0)\n")
    result, traces = _run(guard, code)
    assert result.returncode == 0, result.stderr
    assert traces[0]["cpu_user"] + traces[0]["cpu_sys"] > 0.0        # elementwise arithmetic: 0 FLOPs, but real CPU
    assert traces[0]["flops"] == 0
    summary = audit.trace_summary(guard[0] / "trace")
    assert summary["cpu_seconds"] > 0.0 and "max_rss_kb" in summary


def test_cpu_cap_quarantines_and_control_under_cap_is_clean():
    heavy = normalize_trace({"pid": 1, "cpu_user": 3.0, "cpu_sys": 1.0, "final": True})
    light = normalize_trace({"pid": 2, "cpu_user": 0.1, "cpu_sys": 0.0, "final": True})
    over = audit.trace_flags(audit.trace_summary(Path("nope"), [heavy]), True)   # trace_flags has no cap
    assert "compute_budget_exceeded" not in over                                  # the flag is applied in audit_run
    summary = audit.trace_summary(Path("nope"), [heavy, light])
    assert summary["cpu_seconds"] == 4.1


def test_cpu_cap_flag_is_applied_by_audit_run(tmp_path):
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "1.json").write_text(json.dumps({"pid": 1, "cpu_user": 2.0, "cpu_sys": 0.5, "final": True}))
    t = tmp_path / "t.jsonl"
    t.write_text(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "python3 solve.py"}}]}}) + "\n")
    over = audit.audit_run(t, trace, tmp_path / "none.jsonl", cpu_cap=1.0)
    assert "compute_budget_exceeded" in over["flags"] and over["quarantined"]
    assert over["trace"]["cpu_seconds"] == 2.5
    under = audit.audit_run(t, trace, tmp_path / "none.jsonl", cpu_cap=10.0)
    assert "compute_budget_exceeded" not in under["flags"]
    none = audit.audit_run(t, trace, tmp_path / "none.jsonl", cpu_cap=None)      # record-only default
    assert "compute_budget_exceeded" not in none["flags"]


def test_legacy_trace_without_cpu_fields_sums_to_zero_and_does_not_flag(tmp_path):
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "9.json").write_text(json.dumps({"pid": 9, "flops": 10, "final": True}))   # legacy trace
    summary = audit.trace_summary(trace)
    assert summary["cpu_seconds"] == 0.0 and summary["max_rss_kb"] == 0.0
    flags = audit.audit_run(tmp_path / "missing.jsonl", trace, tmp_path / "n.jsonl", cpu_cap=1.0)["flags"]
    assert "compute_budget_exceeded" not in flags       # no CPU recorded -> a cap cannot be exceeded


class CPUTask(Task):
    id, track, tier = "TR99-CPU-TEST", "TR05", "foundational"
    submission = "report"
    cap = 0.05

    @property
    def budget(self):
        return Budget(n_forward=0, n_probe=0, wall_clock_s=600, cpu_seconds=self.cap)

    def prompt(self, art_dir):
        return "fixture"

    def evaluate(self, art_dir, sub):
        return Evaluation(1.0)

    def baseline(self, art_dir):
        return Submission("report", report={})

    def reference(self, art_dir):
        return Submission("report", report={})


CPU_CHEAT = ("import numpy as np\nz = np.load('mlp.npz')\nw = z['W1'].copy()\n"
             "for _ in range(400):\n    for _ in range(400):\n        w = w * 1.0000001 + 1e-9\n")
CPU_CONTROL = "import numpy as np\nz = np.load('mlp.npz')\nprint(float(np.linalg.norm(z['W1'])))\n"


@pytest.mark.parametrize("cap,code,expect_flag", [(0.001, CPU_CHEAT, True), (1e6, CPU_CHEAT, False),
                                                  (1e6, CPU_CONTROL, False)])
def test_cpu_budget_end_to_end(tmp_path, monkeypatch, cap, code, expect_flag):
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "WORK", tmp_path / "work")
    monkeypatch.setattr(runner, "GUARDS", tmp_path / "guards")
    art = tmp_path / "art"
    (art / "public").mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.savez(art / "public" / "mlp.npz", W1=rng.normal(size=(8, 4)).astype(np.float32))
    task = CPUTask()
    task.cap = cap
    monkeypatch.setattr(runner, "get_task", lambda *a: task)
    monkeypatch.setattr(runner, "nondefault_level", lambda *a: None)
    monkeypatch.setattr(runner, "build", lambda *a, **k: art)
    from weightbench.task import REGISTRY, load_all

    load_all()
    monkeypatch.setitem(REGISTRY, CPUTask.id, lambda: task)

    def launch(spec, model, workdir, transcript, wall_clock, turns, guard_root, ports, flop_cap, url, token):
        env = sandbox.agent_env({k: v for k, v in os.environ.items() if not k.startswith("WB_")},
                                tmp_path / "no-venv", guard_root, ports["proxy"], flop_cap, wb_url=url, wb_token=token)
        (workdir / "work.py").write_text(code)
        subprocess.run([sys.executable, "work.py"], cwd=workdir, env=env, capture_output=True, text=True, timeout=120)
        transcript.write_text(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "python3 work.py"}}]}}) + "\n")
        return 0

    monkeypatch.setattr(runner, "_launch_agent", launch)
    spec = agents.get("cmd", "unused", local_ports=[8000], transcript_format="claude-stream-json")
    result = runner.run(CPUTask.id, "cmd", "fixture", agent_spec=spec)
    led = Ledger(runner.RUNS / result["run_id"] / "ledger.jsonl")
    compute = next(led.entries("compute"))["payload"]
    assert compute["cpu_cap"] == cap and compute["cpu_seconds"] >= 0.0
    assert result["compute"]["cpu_seconds"] == compute["cpu_seconds"]
    assert ("compute_budget_exceeded" in result["audit_flags"]) == expect_flag
    assert result["quarantined"] == expect_flag
    assert led.verify() and verify.verify_run(runner.RUNS / result["run_id"]).ok


# ---- B. raw weight-file access at the file boundary ---------------------------------------------------------------

@pytest.fixture
def rawguard(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    big = np.arange(200 * 200, dtype=np.float32).reshape(200, 200)
    np.save(public / "w.npy", big)
    np.savez(public / "mlp.npz", W1=big)
    try:
        from safetensors.numpy import save_file

        save_file({"W": big}, str(public / "w.safetensors"))
    except Exception:  # noqa: BLE001
        pass
    return sandbox.install_guard(tmp_path / "guard", public), public


def _rawrun(rawguard, code: str):
    root, public = rawguard
    env = {k: v for k, v in os.environ.items() if not k.startswith("WB_")}
    env.update(PYTHONPATH=str(root / "lib"), WB_TRACE_DIR=str(root / "trace"),
               WB_SHAPES=str(root / "lib" / "shapes.json"))
    result = subprocess.run([sys.executable, "-c", code], cwd=public, env=env, capture_output=True, text=True,
                            timeout=120)
    traces = [json.loads(p.read_text()) for p in sorted((root / "trace").glob("*.json"))]
    flags = audit.trace_flags(audit.trace_summary(root / "trace"), True)
    return result, (traces[0] if traces else {}), flags


@pytest.mark.parametrize("code", [
    "import numpy as np\nb = open('w.npy', 'rb').read()\np = np.frombuffer(b[128:], np.float32)\nprint(p.sum())\n",
    "import numpy as np\nf = open('mlp.npz', 'rb')\ndata = f.read()\nf.close()\nprint(len(data))\n",
    "import struct\nf = open('w.npy', 'rb'); f.seek(128)\nvals = struct.unpack('<100f', f.read(400)); f.close()\nprint(len(vals))\n",
])
def test_raw_read_of_weight_data_is_flagged(rawguard, code):
    result, trace, flags = _rawrun(rawguard, code)
    assert result.returncode == 0, result.stderr
    assert trace["raw_weight_data_bytes"] > 0 and "raw_weight_read" in flags


def test_raw_read_of_safetensors_data_is_flagged(rawguard):
    pytest.importorskip("safetensors.numpy")
    code = ("import numpy as np, struct\nb = open('w.safetensors', 'rb').read()\n"
            "n = struct.unpack('<Q', b[:8])[0]\np = np.frombuffer(b[8 + n:], np.float32)\nprint(p.size)\n")
    result, trace, flags = _rawrun(rawguard, code)
    assert result.returncode == 0, result.stderr
    assert "raw_weight_read" in flags


@pytest.mark.parametrize("code", [
    # header / shape peek: numpy .npy header, read only the header bytes
    "import numpy.lib.format as f\nh = open('w.npy', 'rb')\nf.read_magic(h)\nprint(f.read_array_header_1_0(h)[0])\nh.close()\n",
    # the metered loader (re-entrancy): reads all data but through np.load
    "import numpy as np\nwith np.load('mlp.npz') as z:\n    w = z['W1']\n    print(w.shape, float((w @ w.T).sum()) != 0)\n",
    # saving a submission writes bytes; writes are not raw reads
    "import numpy as np\nw = np.load('mlp.npz')['W1']\nnp.save('out.npy', (w * 0.5).astype(np.float32))\n",
    # reading a shape from the safetensors header only
    "import struct, json\nf = open('w.safetensors', 'rb')\nn = struct.unpack('<Q', f.read(8))[0]\n"
    "print(list(json.loads(f.read(n))))\nf.close()\n",
])
def test_header_index_load_and_save_stay_clean(rawguard, code):
    if "safetensors" in code and not (rawguard[1] / "w.safetensors").exists():
        pytest.skip("safetensors not installed")
    result, trace, flags = _rawrun(rawguard, code)
    assert result.returncode == 0, result.stderr
    assert trace.get("raw_weight_data_bytes", 0) == 0 and "raw_weight_read" not in flags


def test_weight_export_bytes_recorded_but_not_quarantined(rawguard):
    code = ("import numpy as np\nw = np.load('mlp.npz')['W1']\nbb = w.tobytes()\nrow = w[0].tolist()\n"
            "p = np.frombuffer(bb, np.float32)\nprint(type(p).__name__, len(row))\n")
    result, trace, flags = _rawrun(rawguard, code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[0] == "WeightArray"                 # the copy stays metered
    assert trace["weight_export_bytes"] >= 200 * 200 * 4
    assert trace.get("metering_escapes", 0) == 0 and "raw_weight_read" not in flags


# ---- manifest: cpu_seconds is backward compatible ----------------------------------------------------------------

def test_cpu_seconds_budget_is_manifest_compatible_when_default_and_invalidates_when_set():
    from weightbench import manifest

    class Base(Task):
        id, track, tier = "TR99-MF", "TR05", "foundational"
        budget = Budget(n_forward=1, wall_clock_s=60)

    spec = manifest.task_spec(Base, schema_version=2)
    assert "cpu_seconds" not in spec["budget"]           # default None omitted: existing manifests still match

    class Capped(Base):
        budget = Budget(n_forward=1, wall_clock_s=60, cpu_seconds=30)

    assert manifest.task_spec(Capped, schema_version=2)["budget"]["cpu_seconds"] == 30   # a set cap is a change


# ---- provenance beats shape: an activation whose shape equals a weight's -----------------------------------------------

@pytest.fixture
def coinc(tmp_path):
    """TR05-DEAD-like shapes: W1 (128,16), W2 (1,128). With a batch of 1 the hidden activation has shape (1,128),
    which equals W2's own public shape, so the second product's operands are both weight-shaped by coincidence."""
    public = tmp_path / "public"
    public.mkdir()
    rng = np.random.default_rng(0)
    np.savez(public / "mlp.npz", W1=rng.normal(size=(128, 16)).astype(np.float32),
             W2=rng.normal(size=(1, 128)).astype(np.float32), Wsq=rng.normal(size=(16, 16)).astype(np.float32))
    return sandbox.install_guard(tmp_path / "guard", public), public


COINC_LOAD = "import numpy as np\nz = np.load('mlp.npz')\nW1, W2, Wsq = z['W1'], z['W2'], z['Wsq']\n"


@pytest.mark.parametrize("second", [
    "y = np.einsum('bo,oi->bi', h, W1)",       # einsum: both operands look weight-shaped (h is (1,128)==W2 shape)
    "y = h @ W1",                               # @ operator, same coincidence
])
def test_batch_one_forward_chain_is_detected_despite_shape_coincidence(coinc, second):
    code = COINC_LOAD + ("x = np.ones((1, 16), np.float32)\n"
                         "h = np.einsum('bi,oi->bo', x, W1)\n"
                         "h = np.where(h > 0, h, 0)\n" + second + "\nprint(h.shape, y.shape)\n")
    result, traces = _coinc_run(coinc, code)
    assert result.returncode == 0, result.stderr
    assert traces[0]["forward_chains"] == 1               # provenance (h is an activation) wins over its weight shape
    assert "forward_chain_runtime" in audit.trace_flags(audit.trace_summary(coinc[0] / "trace"), True)


@pytest.mark.parametrize("body", [
    "y = W2 @ W1\nprint(y.shape)\n",                       # product of two genuine weights: static composition
    "y = np.einsum('ab,bc->ac', W2, W1)\nprint(y.shape)\n",
    # power iteration on a square weight: Wsq @ v is weight-like only by coincidence, no nonlinearity, no chain
    "v = np.ones((16, 16), np.float32)\nfor _ in range(8):\n    v = Wsq @ v\n    v = v / np.maximum(np.linalg.norm(v), 1e-9)\n"
    "print(float(v.sum()) == float(v.sum()))\n",
    # interval-bound propagation on the same shapes
    "u = np.ones(16); l = np.zeros(16)\nfor W in (W1, W2):\n"
    "    up = np.maximum(W, 0) @ u + np.minimum(W, 0) @ l\n    lo = np.maximum(W, 0) @ l + np.minimum(W, 0) @ u\n"
    "    u, l = np.maximum(up, 0), np.maximum(lo, 0)\nprint(u.shape)\n",
])
def test_static_composition_and_bounds_with_coincident_shapes_stay_clean(coinc, body):
    result, traces = _coinc_run(coinc, COINC_LOAD + body)
    assert result.returncode == 0, result.stderr
    assert traces[0]["forward_chains"] == 0
    assert audit.trace_flags(audit.trace_summary(coinc[0] / "trace"), True) == set()


def _coinc_run(coinc, code: str):
    root, public = coinc
    env = {k: v for k, v in os.environ.items() if not k.startswith("WB_")}
    env.update(PYTHONPATH=str(root / "lib"), WB_TRACE_DIR=str(root / "trace"),
               WB_SHAPES=str(root / "lib" / "shapes.json"))
    result = subprocess.run([sys.executable, "-c", code], cwd=public, env=env, capture_output=True, text=True,
                            timeout=120)
    traces = [json.loads(p.read_text()) for p in sorted((root / "trace").glob("*.json"))]
    return result, traces
