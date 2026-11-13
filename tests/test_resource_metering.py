"""Resource metering and raw-file access: raw weight reads of the agent's workdir copy and of small .npz files, whole-tree CPU under a PID namespace, the CPU cap end to end on SMOKE-CPU-000, author-side
compute records, and a reachable trace-event limit.

Model-free: tiny NumPy files in guarded child processes and stub agent launchers. Every rule has a
clean control next to its detection test."""
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from weightbench import manifest, report, runner, verify
from weightbench.harness import agents, audit, backends, netbridge, sandbox
from weightbench.harness import redteam_agent as ra
from weightbench.ledger import Ledger
from weightbench.task import REGISTRY, load_all

POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX processes / pipes")
LINUX = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc and prctl")


def _dead_like(path: Path, compressed: bool = False) -> dict:
    """A TR05-DEAD-001-sized mlp.npz (about 10 KB, below a 64 KB fixed fallback offset) plus a zero-size member."""
    rng = np.random.default_rng(0)
    arrays = {"W1": rng.normal(size=(32, 8)).astype(np.float32), "b1": rng.normal(size=32).astype(np.float32),
              "W2": rng.normal(size=(16, 32)).astype(np.float32), "b2": rng.normal(size=16).astype(np.float32),
              "W3": rng.normal(size=(1, 16)).astype(np.float32), "empty": np.zeros((0, 4), np.float32)}
    (np.savez_compressed if compressed else np.savez)(path, **arrays)
    return arrays


@pytest.fixture
def sandboxed(tmp_path):
    """Artifact public dir + the agent's workdir copy (as runner.run makes them); the guard knows both locations."""
    public, work = tmp_path / "art" / "public", tmp_path / "work"
    public.mkdir(parents=True)
    arrays = _dead_like(public / "mlp.npz")
    shutil.copytree(public, work)
    root = sandbox.install_guard(tmp_path / "guard", public, visible_dirs=[work])
    return root, work, arrays


def _guarded(root: Path, cwd: Path, code: str):
    for old in (root / "trace").glob("*.json"):              # each call is audited on its own
        old.unlink()
    env ={k: v for k, v in os.environ.items() if not k.startswith("WB_")}
    env.update(PYTHONPATH=str(root / "lib"), WB_TRACE_DIR=str(root / "trace"),
               WB_SHAPES=str(root / "lib" / "shapes.json"))
    result = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True,
                            timeout=120)
    summary = audit.trace_summary(root / "trace")
    return result, summary, audit.trace_flags(summary, True)


# ---- 1b. exact tensor-data ranges --------------------------------------------------------------------------------

def test_npz_ranges_are_exactly_the_array_bytes_and_exclude_every_header(tmp_path):
    arrays = _dead_like(tmp_path / "mlp.npz")
    blob = (tmp_path / "mlp.npz").read_bytes()
    assert len(blob) < 65536                                  # a 64 KB fixed fallback offset would never be passed
    ranges = sandbox.data_ranges(tmp_path / "mlp.npz")
    nonempty = [a for a in arrays.values() if a.size]
    assert len(ranges) == len(nonempty)                       # the zero-size member has no range
    assert sorted(blob[a:b] for a, b in ranges) == sorted(a.tobytes() for a in nonempty)
    with zipfile.ZipFile(tmp_path / "mlp.npz") as z:
        cd_start = min(i.header_offset for i in z.infolist())
        for info in z.infolist():                             # local headers are outside every range
            assert not any(a <= info.header_offset < b for a, b in ranges)
    assert all(b <= blob.rfind(b"PK\x01\x02") for _, b in ranges)   # central directory after all data
    assert cd_start == 0


def test_compressed_npz_payload_is_data_and_npy_safetensors_are_exact(tmp_path):
    _dead_like(tmp_path / "c.npz", compressed=True)
    with zipfile.ZipFile(tmp_path / "c.npz") as z:
        payloads = sum(i.compress_size for i in z.infolist() if i.compress_size)
        assert any(i.compress_type == zipfile.ZIP_DEFLATED for i in z.infolist())
    assert sum(b - a for a, b in sandbox.data_ranges(tmp_path / "c.npz")) == payloads
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(tmp_path / "w.npy", w)
    [[a, b]] = sandbox.data_ranges(tmp_path / "w.npy")
    assert (tmp_path / "w.npy").read_bytes()[a:b] == w.tobytes()
    np.save(tmp_path / "z.npy", np.zeros((0, 3), np.float32))
    assert sandbox.data_ranges(tmp_path / "z.npy") == []      # zero-size: nothing to read
    (tmp_path / "junk.npz").write_bytes(b"not a zip at all")
    assert sandbox.data_ranges(tmp_path / "junk.npz") == [[0, 16]]   # unparseable: the whole file is data


def test_weight_files_lists_every_visible_location_with_fingerprint(tmp_path):
    public, work = tmp_path / "public", tmp_path / "work"
    public.mkdir()
    _dead_like(public / "mlp.npz")
    shutil.copytree(public, work)
    [entry] = sandbox.weight_files(public, visible_dirs=[work])
    assert entry["paths"] == [os.path.realpath(public / "mlp.npz"), os.path.realpath(work / "mlp.npz")]
    assert entry["fingerprint"] == sandbox.file_fingerprint(work / "mlp.npz")
    assert entry["size"] == (public / "mlp.npz").stat().st_size and entry["data_offset"] == entry["ranges"][0][0]


# ---- 1a. the file the agent actually opens -----------------------------------------------------------------------

def test_redteam_raw_read_of_the_workdir_copy_is_flagged(sandboxed):
    root, work, _ = sandboxed
    result, summary, flags = _guarded(root, work, ra.SCRIPTS["raw_weight_read"])
    assert result.returncode == 0, result.stderr
    assert summary["raw_weight_opens"] >= 1 and summary["raw_weight_data_bytes"] > 0 and "raw_weight_read" in flags


def test_guard_without_the_workdir_location_still_matches_the_copy_by_fingerprint(tmp_path):
    """Given only the artifact dir, the guard still recognises an exact workdir copy by size + fingerprint."""
    public, work = tmp_path / "public", tmp_path / "work"
    public.mkdir()
    _dead_like(public / "mlp.npz")
    shutil.copytree(public, work)
    root = sandbox.install_guard(tmp_path / "guard", public)
    _, summary, flags = _guarded(root, work, "data = open('mlp.npz', 'rb').read()\n")
    assert summary["raw_weight_data_bytes"] > 0 and "raw_weight_read" in flags


@pytest.mark.parametrize("code", [
    "import pathlib\nb = pathlib.Path('mine.bin').read_bytes()\n",                                # copy, new name
    "import pathlib\nb = pathlib.Path('mlp.npz').read_bytes()\n",                                 # io.open route
    "f = open('mlp.npz', 'rb')\nn = sum(len(line) for line in f)\n",                              # iteration
    "f = open('mlp.npz', 'rb')\nb = f.read1(1 << 20)\nwhile b:\n    b = f.read1(1 << 20)\n",      # read1
    "import zipfile\nz = zipfile.ZipFile('mlp.npz')\nb = z.read('W1.npy')\n",                    # zipfile members
    "import io\nf = io.open('mlp.npz', 'rb')\nbuf = bytearray(20000)\nf.readinto(buf)\n",         # readinto
])
def test_every_raw_read_route_into_tensor_data_is_flagged(sandboxed, code):
    root, work, _ = sandboxed
    shutil.copyfile(work / "mlp.npz", work / "mine.bin")      # a copy the agent made (e.g. with cp)
    result, summary, flags = _guarded(root, work, code)
    assert result.returncode == 0, result.stderr
    assert summary["raw_weight_data_bytes"] > 0 and "raw_weight_read" in flags


def test_compressed_member_payload_read_is_flagged(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    _dead_like(public / "mlp.npz", compressed=True)
    root = sandbox.install_guard(tmp_path / "guard", public)
    code = "import zipfile\nz = zipfile.ZipFile('mlp.npz')\nb = z.read('W2.npy')\n"
    result, summary, flags = _guarded(root, public, code)
    assert result.returncode == 0, result.stderr
    assert summary["raw_weight_data_bytes"] > 0 and "raw_weight_read" in flags
    _, summary, flags = _guarded(root, public, "import zipfile\nprint(zipfile.ZipFile('mlp.npz').namelist())\n")
    assert summary["raw_weight_data_bytes"] == 0 and "raw_weight_read" not in flags      # central directory: clean


HEADER_PEEK = r'''import struct, zipfile
import numpy.lib.format as fmt
with zipfile.ZipFile("mlp.npz") as z:                  # central directory: names, sizes, offsets
    infos = {i.filename: (i.header_offset, i.compress_type) for i in z.infolist()}
shapes = {}
with open("mlp.npz", "rb") as f:                       # each member's local header, then its npy header, exactly
    for name, (offset, ctype) in infos.items():
        f.seek(offset)
        local = f.read(30)
        n, m = struct.unpack("<HH", local[26:30])
        f.seek(offset + 30 + n + m)
        version = fmt.read_magic(f)
        shapes[name] = fmt.read_array_header_1_0(f)[0]
print(shapes)
'''


@pytest.mark.parametrize("code", [
    HEADER_PEEK,
    "import zipfile\nprint(zipfile.ZipFile('mlp.npz').namelist())\n",
    "f = open('mlp.npz', 'rb')\nprint(f.read(4))\nf.close()\n",
    "import numpy as np\nwith np.load('mlp.npz') as z:\n    print({k: z[k].shape for k in z.files})\n",
    # IBP / back-substitution reference pattern on the workdir copy (static analysis, metered loads)
    "import numpy as np\nz = np.load('mlp.npz')\nW1, b1, W2, b2 = z['W1'], z['b1'], z['W2'], z['b2']\n"
    "l, u = -np.ones(8, np.float32), np.ones(8, np.float32)\n"
    "lo = np.maximum(W1, 0) @ l + np.minimum(W1, 0) @ u + b1\nhi = np.maximum(W1, 0) @ u + np.minimum(W1, 0) @ l + b1\n"
    "lo, hi = np.maximum(lo, 0), np.maximum(hi, 0)\nprint(float((np.maximum(W2, 0) @ hi + np.minimum(W2, 0) @ lo).max()))\n",
    # saving a submission, then reading the submission back raw (it is not a subject weight file)
    "import numpy as np\nw = np.load('mlp.npz')['W1']\nnp.savez('delta.npz', W1=np.zeros_like(w))\n"
    "b = open('delta.npz', 'rb').read()\n",
])
def test_header_index_reads_metered_loads_ibp_and_submissions_stay_clean(sandboxed, code):
    root, work, _ = sandboxed
    result, summary, flags = _guarded(root, work, code)
    assert result.returncode == 0, result.stderr
    assert summary["raw_weight_data_bytes"] == 0 and "raw_weight_read" not in flags
    assert not flags & {"forward_chain_runtime", "metering_escape"}


@POSIX
def test_shell_copy_and_hash_stay_clean_but_reading_the_copy_raw_is_flagged(sandboxed):
    root, work, _ = sandboxed
    shell = "import subprocess\nsubprocess.run('cp mlp.npz backup.npz && sha256sum mlp.npz backup.npz', shell=True)\n"
    result, summary, flags = _guarded(root, work, shell)
    assert result.returncode == 0 and (work / "backup.npz").exists()
    assert summary["raw_weight_data_bytes"] == 0 and "raw_weight_read" not in flags
    _, summary, flags = _guarded(root, work, "b = open('backup.npz', 'rb').read()\n")
    assert "raw_weight_read" in flags


def test_legacy_weight_files_json_with_only_an_offset_still_loads(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    np.save(public / "w.npy", np.arange(400, dtype=np.float32))
    root = sandbox.install_guard(tmp_path / "guard", public)
    legacy = [{"path": os.path.realpath(public / "w.npy"), "data_offset": 128}]     # legacy format
    (root / "lib" / "weight_files.json").write_text(json.dumps(legacy))
    _, summary, flags = _guarded(root, public, "f = open('w.npy', 'rb')\nf.read(100)\nf.close()\n")
    assert "raw_weight_read" not in flags
    _, summary, flags = _guarded(root, public, "b = open('w.npy', 'rb').read()\n")
    assert "raw_weight_read" in flags


def test_runner_installs_the_guard_with_the_workdir_location(tmp_path, monkeypatch):
    seen = {}
    real = sandbox.install_guard

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(sandbox, "install_guard", spy)
    _stub_run(tmp_path, monkeypatch, "print(1)\n", "SMOKE-CAP-000", 0)
    assert [Path(p).name for p in seen["visible_dirs"]] == ["SMOKE-CAP-000__cmd-fixture__s0__r0"]


# ---- 2. whole-tree CPU -------------------------------------------------------------------------------------------

def test_netbridge_parses_the_report_fd():
    maps, cmd, limits = netbridge.parse_args(["--rlimit-cpu", "5", "--report-fd", "7", "--", "x"])
    assert limits == {"cpu": 5, "report_fd": 7} and cmd == ["x"]


def test_linux_wrap_makes_the_bridge_pid1_only_with_a_report_fd(tmp_path):
    for d in ("work", "guard/lib", "guard/trace", "guard/sock", "venv/bin"):
        (tmp_path / d).mkdir(parents=True)
    box = backends.SandboxSpec(workdir=tmp_path / "work", guard_root=tmp_path / "guard", agent_venv=tmp_path / "venv",
                               deny_read=[], ports={})
    plain = backends.LinuxBwrapBackend().wrap(box, ["agent"])
    assert "--as-pid-1" not in plain and "--report-fd" not in plain
    box.report_fd = 9
    cmd = backends.LinuxBwrapBackend().wrap(box, ["agent"])
    split = cmd.index("--")
    assert cmd[split - 1] == "--as-pid-1"                     # a bwrap option, before the command
    tail = cmd[split + 1:]
    assert tail[tail.index("--report-fd") + 1] == "9" and tail[-1] == "agent"
    assert str(box.guard_root / "lib" / "wb_netbridge.py") in tail   # harness code from the read-only lib mount


BUSY_WITH_DAEMON = r'''import os, time
if os.fork() == 0:                       # a detached grandchild that outlives the agent command
    os.setsid()
    if os.fork() == 0:
        t = time.process_time()
        while time.process_time() - t < 30:
            pass
        os._exit(0)
    os._exit(0)
t = time.process_time()
while time.process_time() - t < 0.6:
    pass
'''


@LINUX
def test_bridge_init_counts_the_whole_tree_including_orphans_and_kills_leftovers(tmp_path):
    read_fd, write_fd = os.pipe()
    done = subprocess.run([sys.executable, str(Path(netbridge.__file__)), "--report-fd", str(write_fd), "--",
                           sys.executable, "-c", BUSY_WITH_DAEMON], pass_fds=(write_fd,), timeout=60)
    os.close(write_fd)
    rep = runner._read_tree_report(read_fd)
    assert done.returncode == 0 and rep is not None
    assert rep["root_returncode"] == 0 and rep["leftovers_killed"] >= 1 and rep["subreaper"]
    assert rep["root_cpu_seconds"] >= 0.5                      # the agent command's own CPU, read before reaping
    assert rep["tree_cpu_seconds"] >= rep["root_cpu_seconds"]  # plus the killed orphan's CPU
    assert rep["tree_max_rss_kb"] > 0


@POSIX
def test_tree_report_reader_rejects_garbage_and_silence():
    for payload in (b"not json\n", b'{"tree_cpu_seconds": "x"}\n', b""):
        r, w = os.pipe()
        os.write(w, payload)
        os.close(w)
        assert runner._read_tree_report(r) is None
    r, w = os.pipe()
    os.write(w, b'{"tree_cpu_seconds": 1.5, "root_pid": 3}\n')
    os.close(w)
    assert runner._read_tree_report(r)["tree_cpu_seconds"] == 1.5


def _traces(tmp_path, *items):
    d = tmp_path / "trace"
    d.mkdir(exist_ok=True)
    for t in items:
        (d / f"{t['pid']}.json").write_text(json.dumps({"final": True, **t}))
    t = tmp_path / "t.jsonl"
    t.write_text(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "python3 solve.py"}}]}}) + "\n")
    return t, d


def test_unguarded_cpu_under_a_guarded_root_is_quarantined_and_honest_trees_are_clean(tmp_path):
    t, d = _traces(tmp_path, {"pid": 3, "cpu_user": 0.03, "cpu_sys": 0.01}, {"pid": 7, "cpu_user": 0.2, "cpu_sys": 0.05})
    honest = {"tree_cpu_seconds": 0.32, "root_pid": 3, "root_cpu_seconds": 0.04, "tree_max_rss_kb": 1.0}
    clean = audit.audit_run(t, d, tmp_path / "e.jsonl", cpu_tree=honest)
    assert clean["flags"] == [] and not clean["quarantined"]
    assert clean["cpu_tree"]["cpu_tree_root_guarded"] and abs(clean["cpu_tree"]["cpu_unguarded_gap"] - 0.03) < 1e-9
    hidden = dict(honest, tree_cpu_seconds=0.32 + 3.0)        # 3 s in a `python -S` / foreign child: no trace
    over = audit.audit_run(t, d, tmp_path / "e.jsonl", cpu_tree=hidden)
    assert over["flags"] == ["unguarded_cpu"] and over["quarantined"]


def test_unguarded_root_gap_is_reported_not_quarantined(tmp_path):
    """Root without a guard trace (an agent CLI such as node): its helpers may be behind the gap."""
    t, d = _traces(tmp_path, {"pid": 7, "cpu_user": 0.2, "cpu_sys": 0.05})
    tree = {"tree_cpu_seconds": 30.0, "root_pid": 3, "root_cpu_seconds": 20.0}
    out = audit.audit_run(t, d, tmp_path / "e.jsonl", cpu_tree=tree)
    assert out["flags"] == ["unguarded_cpu_unattributed"] and not out["quarantined"]
    assert out["cpu_tree"]["cpu_unguarded_gap"] == pytest.approx(30.0 - 20.0 - 0.25)
    calm = audit.audit_run(t, d, tmp_path / "e.jsonl", cpu_tree=dict(tree, tree_cpu_seconds=20.4))
    assert calm["flags"] == []                                 # the CLI's own CPU is never counted against it


def test_agent_root_exemption_counts_as_unguarded_root(tmp_path):
    t, d = _traces(tmp_path, {"pid": 3, "cpu_user": 1.0, "cpu_sys": 0.0, "agent_root": True})
    tree = {"tree_cpu_seconds": 5.0, "root_pid": 3, "root_cpu_seconds": 1.0}
    out = audit.audit_run(t, d, tmp_path / "e.jsonl", cpu_tree=tree, python_agent=True)
    assert out["flags"] == ["unguarded_cpu_unattributed"] and not out["quarantined"]


def test_missing_tree_report_is_quarantined_and_absent_tree_changes_nothing(tmp_path):
    t, d = _traces(tmp_path, {"pid": 7, "cpu_user": 0.2, "cpu_sys": 0.05})
    missing = audit.audit_run(t, d, tmp_path / "e.jsonl", cpu_tree_missing=True)
    assert missing["flags"] == ["cpu_tree_unreported"] and missing["quarantined"]
    legacy = audit.audit_run(t, d, tmp_path / "e.jsonl")
    assert legacy["flags"] == [] and "cpu_tree" not in legacy


# ---- shared stub launcher ----------------------------------------------------------------------------------------

def _stub_run(tmp_path, monkeypatch, code: str, task_id: str, sample: int, tree_report=None, tree_expected=False,
              rc: int = 0) -> dict:
    """runner.run with the sandbox replaced by one guarded Python child that gets the runner's real agent environment
    (Weight API reporting included). ``tree_report`` simulates the Linux init's whole-tree report."""
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "WORK", tmp_path / "work")
    monkeypatch.setattr(runner, "GUARDS", tmp_path / "guards")
    monkeypatch.setattr(runner, "ART", tmp_path / "art")

    def launch(spec, model, workdir, transcript, wall_clock, turns, guard_root, ports, flop_cap, url, token):
        env = sandbox.agent_env({k: v for k, v in os.environ.items() if not k.startswith("WB_")},
                                tmp_path / "no-venv", guard_root, ports["proxy"], flop_cap, wb_url=url, wb_token=token)
        (workdir / "work.py").write_text(code)
        subprocess.run([sys.executable, "work.py"], cwd=workdir, env=env, capture_output=True, text=True, timeout=600)
        transcript.write_text(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "python3 work.py"}}]}}) + "\n")
        runner._TREE_REPORTS[str(guard_root)] = {"expected": tree_expected, "report": tree_report}
        return rc

    monkeypatch.setattr(runner, "_launch_agent", launch)
    spec = agents.get("cmd", "unused", local_ports=[8000], transcript_format="claude-stream-json")
    return runner.run(task_id, "cmd", "fixture", sample=sample, agent_spec=spec)


def test_runner_records_the_init_report_in_the_compute_entry(tmp_path, monkeypatch):
    tree = {"tree_cpu_seconds": 0.4, "root_pid": 999999, "root_cpu_seconds": 0.05, "tree_max_rss_kb": 2048.0}
    result = _stub_run(tmp_path, monkeypatch, "import numpy as np\nprint(np.load('mlp.npz')['W1'].shape)\n",
                       "SMOKE-CAP-000", 0, tree_report=tree, tree_expected=True)
    led = Ledger(runner.RUNS / result["run_id"] / "ledger.jsonl")
    compute = next(led.entries("compute"))["payload"]
    assert compute["cpu_tree_source"] == "sandbox_init" and compute["cpu_tree_seconds"] == 0.4
    assert compute["cpu_tree_root_seconds"] == 0.05 and compute["cpu_tree_root_guarded"] is False
    assert compute["cpu_tree_max_rss_kb"] == 2048.0 and "cpu_unguarded_gap" in compute
    assert result["compute"] == compute and verify.verify_run(runner.RUNS / result["run_id"]).ok


@pytest.mark.parametrize("rc,flagged", [(0, True), (-9, False)])
def test_runner_quarantines_a_missing_init_report_except_on_timeout(tmp_path, monkeypatch, rc, flagged):
    result = _stub_run(tmp_path, monkeypatch, "print(1)\n", "SMOKE-CAP-000", 0, tree_report=None,
                       tree_expected=True, rc=rc)
    assert ("cpu_tree_unreported" in result["audit_flags"]) == flagged and result["quarantined"] == flagged
    assert result["compute"]["cpu_tree_source"] == "missing" and result["compute"]["cpu_tree_seconds"] is None


# ---- 3. the CPU cap end to end on SMOKE-CPU-000 ------------------------------------------------------------------

def test_smoke_cpu_task_is_the_only_cpu_capped_task_and_stays_out_of_the_suite():
    load_all()
    capped = sorted(t for t, cls in REGISTRY.items() if cls.budget.cpu_seconds is not None)
    assert capped == ["SMOKE-CPU-000"] and REGISTRY["SMOKE-CPU-000"].budget.cpu_seconds == 2.0
    assert REGISTRY["SMOKE-CPU-000"].budget.flop_cap is None   # the FLOP cap must not trip first
    assert "SMOKE-CPU-000" not in runner.select_tasks()
    assert "SMOKE-CPU-000" not in manifest.benchmark_tasks(dict(REGISTRY))
    assert ra.SCENARIO_TASK["compute_budget_exceeded"] == ra.SCENARIO_TASK["control_compute_budget_exceeded"] \
        == "SMOKE-CPU-000"
    assert ra.CPU_CHEAT_STEPS == 200_000 and f"range({ra.CPU_CHEAT_STEPS})" in ra.SCRIPTS["compute_budget_exceeded"]


def test_smoke_cpu_runs_are_left_out_of_the_report(tmp_path, monkeypatch):
    d = tmp_path / "runs" / "SMOKE-CPU-000__cmd-m__s0__r0"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"task": "SMOKE-CPU-000", "agent": "cmd", "model": "m", "s_wb": 1.0,
                                               "s_plus": 1.0, "sample": 0, "seed": 0}))
    monkeypatch.setattr(report, "RUNS", tmp_path / "runs")
    assert report.load() == []


def test_cpu_cap_control_is_clean_on_smoke_cpu(tmp_path, monkeypatch):
    result = _stub_run(tmp_path, monkeypatch, ra.SCRIPTS["control_compute_budget_exceeded"], "SMOKE-CPU-000", 0)
    assert not result["quarantined"] and result["audit_flags"] == []
    assert 0 < result["compute"]["cpu_seconds"] < 2.0 and result["compute"]["cpu_cap"] == 2.0


def test_cpu_cap_cheat_is_quarantined_on_smoke_cpu(tmp_path, monkeypatch):
    """The scripted cheat itself (200000 fixed iterations, several seconds of CPU on any current machine)."""
    result = _stub_run(tmp_path, monkeypatch, ra.SCRIPTS["compute_budget_exceeded"], "SMOKE-CPU-000", 1)
    assert result["compute"]["cpu_seconds"] > 2.0
    assert result["quarantined"] and "compute_budget_exceeded" in result["audit_flags"]
    assert verify.verify_run(runner.RUNS / result["run_id"]).ok


# ---- 4. author-side compute --------------------------------------------------------------------------------------

def test_reference_and_baseline_runs_record_compute_before_the_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ART", tmp_path / "art")
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    for agent in ("reference", "baseline"):
        res = runner.run("SMOKE-000", agent)
        led = Ledger(runner.RUNS / res["run_id"] / "ledger.jsonl")
        kinds = [e["kind"] for e in led.entries()]
        assert kinds == ["run_start", "submission", "compute", "evaluation"]
        compute = next(led.entries("compute"))["payload"]
        assert compute["source"] == "author" and compute["cpu_seconds"] >= 0 and compute["heavy_mode"]
        assert res["compute"] == compute
        assert verify.verify_run(runner.RUNS / res["run_id"]).ok


def test_legacy_reference_run_without_compute_still_verifies_and_rescores(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ART", tmp_path / "art")
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    res = runner.run("SMOKE-000", "reference")
    new = runner.RUNS / res["run_id"]
    old = runner.RUNS / "SMOKE-000__reference__s0__r7"
    old.mkdir()
    led = Ledger(old / "ledger.jsonl")
    for e in Ledger(new / "ledger.jsonl").entries():
        if e["kind"] == "compute":
            continue                                           # a legacy ledger without compute records
        payload = dict(e["payload"], sample=7) if e["kind"] == "run_start" else e["payload"]
        led.append(e["kind"], payload)
    legacy = {k: v for k, v in res.items() if k != "compute"}
    legacy.update(run_id=old.name, sample=7)
    (old / "result.json").write_text(json.dumps(legacy))
    assert verify.verify_run(old).ok
    assert runner.rescore_all() >= 2
    assert "compute" not in json.loads((old / "result.json").read_text())


# ---- 5. the trace-event limit is reachable -----------------------------------------------------------------------

def test_trace_flood_scenario_reaches_the_server_limit_and_quarantines(tmp_path, monkeypatch):
    from weightbench.server import MAX_TRACE_EVENTS

    assert ra.TRACE_FLOOD > MAX_TRACE_EVENTS
    result = _stub_run(tmp_path, monkeypatch, ra.SCRIPTS["trace_limit_exceeded"], "SMOKE-CAP-000", 0)
    led = Ledger(runner.RUNS / result["run_id"] / "ledger.jsonl")
    assert any(led.entries("runtime_trace_limit"))
    assert result["audit_flags"] == ["trace_limit_exceeded"] and result["quarantined"]
    assert verify.verify_run(runner.RUNS / result["run_id"]).ok


def test_header_opens_below_the_limit_are_clean(tmp_path, monkeypatch):
    code = "for _ in range(20):\n    with open('mlp.npz', 'rb') as f:\n        f.read(4)\n"
    result = _stub_run(tmp_path, monkeypatch, code, "SMOKE-CAP-000", 1)
    assert result["audit_flags"] == [] and not result["quarantined"]


# ---- gradient descent: real hand-written training vs. an elementwise edit ------------------------------------------

@pytest.mark.parametrize("name", ["gradient_descent", "gd_evasion_batch1", "gd_evasion_lists", "gd_evasion_masked",
                                  "gd_evasion_masked_all", "gd_evasion_signsplit", "gd_evasion_torch"])
def test_real_gradient_descent_and_its_in_process_evasions_are_forward_chains(sandboxed, name):
    if "torch" in name:
        pytest.importorskip("torch")
    root, work, _ = sandboxed
    result, summary, flags = _guarded(root, work, ra.SCRIPTS[name])
    assert result.returncode == 0, result.stderr
    assert summary["forward_chains"] > 0 and "forward_chain_runtime" in flags
    assert summary["flops"] > 10_000                     # the honest control below has 0 metered FLOPs


@pytest.mark.parametrize("name", ["control_gradient_descent", "elementwise_update", "control_elementwise_update"])
def test_static_control_and_elementwise_edit_are_not_forward_chains(sandboxed, name):
    root, work, _ = sandboxed
    result, summary, flags = _guarded(root, work, ra.SCRIPTS[name])
    assert result.returncode == 0, result.stderr
    assert summary["forward_chains"] == 0 and flags == set() and summary["flops"] == 0
