"""Build artifacts, run one (task, agent, model, sample), score it from the ledger.

    python -m weightbench.runner build   --task TR01-BEH-001 [--seed 0]
    python -m weightbench.runner build   --task TR05-DEAD-001 --level 2     # knob level (SPEC §5), wired tasks only
    python -m weightbench.runner run     --task TR01-BEH-001 --agent claude --model claude-opus-5-5 --sample 0
    python -m weightbench.runner run     --task TR01-BEH-001 --agent baseline|reference
    python -m weightbench.runner sweep   --tasks 'TR01-*' --models claude-opus-5-5,... --samples 1 --jobs 3

Agent isolation (SPEC §1, ``weightbench.harness``): the agent runs in
/tmp/wb-work/<run_id>/ with only public artifacts, TASK.md and wb.py, under a macOS sandbox-exec
profile that denies reading the repo (ground truth, ledgers, other runs), other workdirs and Claude
Code project memory, and denies all outbound network except localhost. Claude Code reaches its API
through an allowlist proxy; every agent Python process loads a runtime guard (no sockets, no
model-execution libraries, no gradient descent, FLOP / forward-chain trace). Web tools are disabled.
Afterwards transcript, traces and egress log are audited; flagged runs are quarantined.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from weightbench import agent_auth, conditions, scoring
from weightbench.harness import agents, backends, sandbox
from weightbench.harness import audit as harness_audit
from weightbench.ledger import Ledger
from weightbench.platform_compat import DEFAULT_AGENT_VENV, to_wsl_path, wsl_available
from weightbench.server import RunState, make_server
from weightbench.task import REGISTRY, Evaluation, Submission, Task, load_all

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
RUNS = ROOT / "runs"
WORK = Path("/tmp/wb-work")
GUARDS = Path("/tmp/wb-guard")    # per-run runtime guard (read-only for the agent) + traces
AGENT_VENV = Path(os.environ.get("WB_AGENT_VENV", DEFAULT_AGENT_VENV)).expanduser()
CLIENT = Path(__file__).resolve().parent / "client_wb.py"

AGENT_PREAMBLE = """You are being evaluated on WeightBench, a benchmark for Weight-Space Programming:
reading, fixing and writing neural-network weights directly.

Your working directory contains everything you may use: TASK.md (read it first), the public
artifacts it describes, and wb.py (your only channel to the evaluator: metered probes and
submission). A Python with numpy, scipy, torch and safetensors is on PATH as `python`.

Rules: work only inside this directory; no internet; respect the probe budget; do not try to
read files outside this directory. Submit with wb.py before you finish (you may resubmit; the
last submission counts). If you finish early, double-check your answer, then stop.

Weight-space rules (these decide whether your run counts):
- Do NOT run the subject network on inputs yourself: no hand-written forward pass, no evaluating
  layers on data or on inputs you construct, no simulating the model, unless TASK.md explicitly
  allows a specific evaluation (it says so when it does). Observe model behaviour only through the
  metered wb.py probes. Static analysis of the weights (shapes, norms, SVD, composing matrices,
  interval bounds, projecting embeddings through a few matrices) is allowed and expected.
- No gradient descent and no numerical gradient estimates (finite differences, perturb-and-measure).
  This holds for every task, whether its budget is zero forward passes or a positive one: a larger
  probe budget buys more metered observations, never a fit.
- Do not copy the weights out of the metered representation to compute with them unmetered. Reading
  the public weight files, converting a metered array to a plain one, exporting through tobytes,
  tolist, the buffer protocol, ctypes or a separate process, and running arithmetic on the copy are
  all recorded and audited; the arithmetic counts whether or not you intended to evade the meter.
- For einsum, prefer unary/binary string equations with paired reduction axes. Unsupported
  contractions are quarantined: avoid more than two operands, NumPy list-form equations, binary
  diagonals, unpaired operand reductions, and summation over ellipsis axes. Split larger static
  calculations into supported operations instead. Torch list forms are allowed when normalized
  to supported equations by its dispatcher.
- Use `python` as given. Do not start Python with -I, -S or -E, do not unset or change environment
  variables, and do not try to disable or get around the runtime guard.

This environment is sandboxed: outbound network is blocked except the local evaluator, model-
execution libraries and gradient descent are disabled, and supported weight arithmetic is traced
and audited. Runs that break these rules are
quarantined and score 0.
"""


def level_suffix(level: int | None) -> str:
    """'' for a task's default knob level (None), '_L<level>' otherwise: keeps default paths and run ids unchanged."""
    return "" if level is None else f"_L{level}"


def art_dir(task_id: str, seed: int, level: int | None = None) -> Path:
    """artifacts/<task>/seed<s>/ at the default knob level, artifacts/<task>/seed<s>_L<level>/ otherwise."""
    return ART / task_id / f"seed{seed}{level_suffix(level)}"


def nondefault_level(task_id: str, level: int | None) -> int | None:
    """``level`` if it differs from the task's own knob level, else None (raises if the task has no wired knob)."""
    load_all()
    if level is not None:
        from weightbench.knobs import check_level
        check_level(level)
    if level is None or level == REGISTRY[task_id].knob_level:
        return None
    REGISTRY[task_id]().at_level(level)        # validates
    return level


def get_task(task_id: str, level: int | None = None) -> Task:
    load_all()
    return REGISTRY[task_id]().at_level(level)


# Modules that cannot change a subject model's weights: adjudication (audit, review), agent launch (sandbox, guards,
# adapters) and reporting/CLI. Editing them must not invalidate every artifact (an audit-only fix would otherwise
# force a rebuild, which is both wasted work and a different subject than the one published).
BUILD_NEUTRAL_SOURCES = frozenset({
    "weightbench/agent_auth.py", "weightbench/cli.py", "weightbench/client_wb.py", "weightbench/doctor.py",
    "weightbench/harness/agent_guard.py", "weightbench/harness/agents.py", "weightbench/harness/audit.py",
    "weightbench/harness/backends.py", "weightbench/harness/ibp_agent.py", "weightbench/harness/netbridge.py",
    "weightbench/harness/redteam_agent.py", "weightbench/harness/redteam_cheats.py",
    "weightbench/harness/sandbox.py", "weightbench/harness/trace_schema.py", "weightbench/leaderboard.py",
    "weightbench/report.py", "weightbench/review.py", "weightbench/server.py",
    # step execution and report statistics: they run or summarise artifacts and cannot change a subject's weights
    "weightbench/heavy.py", "weightbench/item_analysis.py",
})


def numeric_stack() -> str:
    """The numeric libraries' full build strings, recorded with every artifact (torch 2.15.0+cpu and 2.15.0+cu132
    are different binaries with different MKL kernels, and they train different weights from the same seed)."""
    parts = []
    for mod in ("numpy", "torch"):
        try:
            v = getattr(__import__(mod), "__version__", None)
        except Exception:
            v = None
        parts.append(f"{mod}=={v or '?'}")
    return " ".join(parts)


def source_fingerprint() -> str:
    """Short hash of the code that determines a subject's weights (everything weightbench ships except
    ``BUILD_NEUTRAL_SOURCES``).

    A build is a pure function of (task, seed, level) only if that code is the same: artifacts built by older
    generator code describe a different subject than the current sources would produce, and scoring them silently
    mixes the two. So the marker binds each artifact to the code that made it, and a mismatch rebuilds.

    The numeric stack is recorded next to it, not hashed into it: an artifact built on another machine is legitimate
    evidence when scored locally even though the two stacks train different weights from the same seed, and the
    honest answer is to say which stack built it (see ``build``), not to rebuild it somewhere else.
    """
    h = hashlib.sha256()
    for p in sorted((ROOT / "weightbench").rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if rel in BUILD_NEUTRAL_SOURCES:
            continue
        h.update(rel.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def build(task_id: str, seed: int, force: bool = False, level: int | None = None) -> Path:
    level = nondefault_level(task_id, level)
    d = art_dir(task_id, seed, level)
    marker = d / ".built"
    fp = source_fingerprint()
    if marker.exists() and not force:
        try:
            old = json.loads(marker.read_text() or "{}")
        except ValueError:
            old = {}
        if old.get("source") == fp:
            if old.get("numeric") and old["numeric"] != numeric_stack():
                print(f"[build] {d}: built under {old['numeric']}, scoring it here under {numeric_stack()}",
                      file=sys.stderr)
            return d
        # An artifact built by other sources is not evidence about the current ones: rebuild it instead of
        # scoring a subject model that no current code path produced (see source_fingerprint).
        print(f"[build] {d}: sources changed since it was built "
              f"(source {old.get('source', 'unrecorded')} -> {fp}); rebuilding", file=sys.stderr)
    shutil.rmtree(d, ignore_errors=True)
    (d / "public").mkdir(parents=True)
    (d / "private").mkdir(parents=True)
    t0 = time.time()
    get_task(task_id, level).build(d, seed)
    meta = {"seconds": round(time.time() - t0, 1), "source": fp, "numeric": numeric_stack()}
    if level is not None:
        meta["level"] = level
    marker.write_text(json.dumps(meta))
    return d


def _deny_read_paths() -> list[Path]:
    home = Path.home()
    return [ROOT, WORK, GUARDS, home / ".claude" / "projects", home / ".cache" / "huggingface",
            home / "Documents", home / "Desktop", home / "Downloads"]


CLAUDE_TOKEN_FILE = Path.home() / ".config" / "weightbench" / "claude_oauth_token"


def _with_claude_token(env: dict[str, str]) -> dict[str, str]:
    """macOS: Claude Code keeps its login in the Keychain, which the sandbox (rightly) cannot read.
    A long-lived token from `claude setup-token`, stored outside the repo (mode 600), is passed as
    CLAUDE_CODE_OAUTH_TOKEN instead. Egress is still limited to api.anthropic.com."""
    if "CLAUDE_CODE_OAUTH_TOKEN" not in env and "ANTHROPIC_API_KEY" not in env and CLAUDE_TOKEN_FILE.is_file():
        env["CLAUDE_CODE_OAUTH_TOKEN"] = CLAUDE_TOKEN_FILE.read_text().strip()
    return env


def _sandbox_profile(workdir: Path, guard_root: Path | None = None) -> str:
    """macOS sandbox-exec profile: no reads of the repo / other runs / $HOME beyond Claude Code's needs, writes
    only in the workdir, no compilers, no outbound network except localhost (``weightbench.harness.sandbox``)."""
    return sandbox.macos_profile(workdir, _deny_read_paths(), guard_root, extra_read=[AGENT_VENV])


def audit(transcript: Path) -> list[str]:
    """Transcript-only audit flags (kept for callers without runtime traces)."""
    return sorted(harness_audit.transcript_flags(transcript, ROOT))


def _agent_usage(transcript: Path) -> dict:
    out: dict = {}
    if transcript.exists():
        for line in transcript.open():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):      # bare JSON scalars (e.g. a custom agent printing "45") are not events
                continue
            if ev.get("type") == "result":
                out = {k: ev.get(k) for k in ("total_cost_usd", "num_turns", "duration_ms", "is_error", "usage")}
    return out


STALL_EXIT = -8          # return code for a run killed because its transcript stopped growing
STALL_DEFAULT_S = 1800   # 30 min without a byte of transcript output means the agent CLI has wedged
STALL_POLL_S = 2.0


def stall_seconds() -> int:
    """Seconds of transcript silence after which an agent is killed (``WB_AGENT_STALL_S``; 0 disables the watchdog).

    ``wall_clock_s`` only bounds the whole run. A gateway model can wedge mid-stream, emitting nothing (or only
    thinking tokens) for a long time while holding an agent slot, so the launcher also watches the transcript file
    grow and kills the sandbox when it stops."""
    raw = os.environ.get("WB_AGENT_STALL_S")
    if raw is None or not raw.strip():
        return STALL_DEFAULT_S
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"WB_AGENT_STALL_S must be an integer number of seconds, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"WB_AGENT_STALL_S must be >= 0, got {value}")
    return value


def _wait_agent(proc: subprocess.Popen, transcript: Path, deadline: float, stall_s: int) -> int:
    """Wait for a launched sandbox, killing it on the wall-clock deadline or after ``stall_s`` of transcript silence."""
    size, grown_at = -1, time.monotonic()
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc
        now = time.monotonic()
        if now >= deadline:
            _kill_tree(proc)
            return -9
        try:
            n = transcript.stat().st_size
        except OSError:
            n = size
        if n != size:
            size, grown_at = n, now
        elif stall_s > 0 and now - grown_at >= stall_s:
            _kill_tree(proc)
            return STALL_EXIT
        time.sleep(STALL_POLL_S)


def _kill_tree(proc: subprocess.Popen) -> None:
    """SIGKILL the sandbox wrapper and give it a moment to reap; the namespace dies with it."""
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def turn_cap_note(max_turns: int) -> str:
    """Tell the agent its turn cap: runs that hit it without ever submitting score 0."""
    return (f"\nYou have at most {max_turns} turns. Submit a first candidate with wb.py early, then resubmit "
            "improvements; a run that ends without a submission scores 0.\n")


def _launch_agent(spec: agents.AgentSpec, model: str, workdir: Path, transcript: Path, timeout_s: int,
                  max_turns: int, guard_root: Path, ports: dict[str, int], flop_cap: float | None, wb_url: str,
                  wb_token: str) -> int:
    """Run one agent CLI inside the OS sandbox of this host (macOS sandbox-exec or Linux bubblewrap)."""
    backend = backends.detect()
    tmpdir = workdir / ".tmp"
    tmpdir.mkdir(exist_ok=True)
    env = sandbox.agent_env(_with_claude_token(os.environ.copy()), AGENT_VENV, guard_root, ports["proxy"], flop_cap,
                            wb_url=wb_url, wb_token=wb_token, tmpdir=tmpdir)
    # Claude Code ignores TMPDIR and uses /tmp/claude-<uid>/ for its Bash tool's scratch files; on macOS that is
    # outside the sandbox's writable set (and holds other sessions' files), so point it inside the workdir.
    env["CLAUDE_CODE_TMPDIR"] = str(tmpdir)
    if spec.python_runtime:
        env["WB_AGENT_ROOT"] = "1"           # the agent's own process only; see agent_guard
    ports = {**ports, **{f"local{p}": p for p in spec.local_ports}}   # the researcher's local model server(s)
    prompt = AGENT_PREAMBLE + turn_cap_note(max_turns) + "\nBegin by reading TASK.md."
    prompt_file = workdir / "PROMPT.md"
    prompt_file.write_text(prompt)
    cmd = spec.command(model, prompt, prompt_file, max_turns)
    box = backends.SandboxSpec(workdir=workdir, guard_root=guard_root, agent_venv=AGENT_VENV,
                               deny_read=_deny_read_paths(), ports=ports, home_state=spec.home_state,
                               agent_executable=cmd[0], cpu_seconds=timeout_s)
    # Linux: the bridge runs as the sandbox's PID 1 and reports the whole agent tree's CPU through this pipe (the
    # agent never holds the write end; see netbridge.run_as_init). Other backends: no report (rusage delta instead).
    read_fd = None
    if isinstance(backend, backends.LinuxBwrapBackend):
        read_fd, box.report_fd = os.pipe()
    _TREE_REPORTS[str(guard_root)] = {"expected": read_fd is not None, "report": None}
    backend.start(box)
    try:
        with transcript.open("w") as f:
            proc = subprocess.Popen(backend.wrap(box, cmd), cwd=workdir, env=env, stdout=f,
                                    stderr=subprocess.STDOUT,
                                    pass_fds=(box.report_fd,) if read_fd is not None else ())
            return _wait_agent(proc, transcript, time.monotonic() + timeout_s, stall_seconds())
    finally:
        backend.stop()
        if read_fd is not None:
            os.close(box.report_fd)
            _TREE_REPORTS[str(guard_root)]["report"] = _read_tree_report(read_fd)


_TREE_REPORTS: dict[str, dict] = {}       # guard_root -> {"expected": bool, "report": dict | None}, set by _launch_agent


def _read_tree_report(fd: int) -> dict | None:
    """The sandbox init's whole-tree CPU report (one JSON line), or None if it never wrote one. Called after the
    sandbox exited; every write end is closed by then (the namespace is torn down with its PID 1)."""
    import select

    data = b""
    deadline = time.monotonic() + 30
    try:
        while len(data) < 65536:
            wait = deadline - time.monotonic()
            if wait <= 0 or not select.select([fd], [], [], wait)[0]:
                break                       # never block the run on a stray write end
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            data += chunk
    except OSError:
        pass
    finally:
        os.close(fd)
    try:
        report = json.loads(data.decode() or "null")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return report if isinstance(report, dict) and isinstance(report.get("tree_cpu_seconds"), (int, float)) else None


def _copy_public(src: Path, dst: Path) -> None:
    """Public artifacts into the agent workdir (APFS clone on macOS, plain copy elsewhere)."""
    if sys.platform == "darwin" and subprocess.run(["cp", "-cR", f"{src}/.", str(dst)]).returncode == 0:
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def score_from_ledger(task: Task, led: Ledger, ev: Evaluation) -> dict:
    """S_WB from ledger entries only (never from agent-reported numbers)."""
    c = scoring.TrackConstants.for_track(task.track)
    n_fwd = sum(1 for e in led.entries("probe") if e["payload"]["meter"] == "forward")
    n_probe = sum(1 for e in led.entries("probe") if e["payload"]["meter"] == "probe")
    evals = list(led.entries("evaluation"))
    p = dict(evals[-1]["payload"])
    if p["valid"]:
        rescored = task.rescore_s_plus(p.get("details") or {})   # from stored measurements, nothing re-run
        if rescored is not None:
            p["s_plus"] = rescored
    e_comp = scoring.compute_cost(n_fwd, n_probe, c.gamma_probe)
    cost = scoring.cost_from_terms(p.get("edit_cost_terms") or {}, c.lam)
    s_v2 = 0.0 if not p["valid"] else scoring.s_wb(p["s_plus"], p["d_col"], cost, e_comp, c.alpha, c.beta)
    e_budget = scoring.compute_cost(task.budget.n_forward, task.budget.n_probe, c.gamma_probe)
    u_edit = scoring.edit_budget_use(p.get("edit_cost_terms"), task.budget.max_l0, task.budget.max_rank)
    s_v3 = 0.0 if not p["valid"] else scoring.s_wb_v3(p["s_plus"], p["d_col"], u_edit, e_comp, e_budget)
    return {"s_wb": s_v3, "s_plus": p["s_plus"], "d_col": p["d_col"], "edit_cost": cost,
            "edit_cost_terms": p.get("edit_cost_terms"), "e_comp": e_comp, "e_budget": e_budget, "n_forward": n_fwd,
            "n_probe": n_probe, "valid": p["valid"], "formula_version": scoring.FORMULA_VERSION,
            "edit_budget_use": u_edit, "s_wb_v3": s_v3, "s_wb_v2": s_v2}


def rescore_all() -> int:
    """Recompute every result.json from its ledger with the current formula/constants."""
    n = 0
    load_all()
    for rj in sorted(RUNS.glob("*/result.json")):
        res = json.loads(rj.read_text())
        led = Ledger(rj.parent / "ledger.jsonl")
        if res.get("task") not in REGISTRY or not any(True for _ in led.entries("evaluation")):
            continue
        res.update(score_from_ledger(REGISTRY[res["task"]](), led, None))
        res["ledger_ok"] = led.verify()
        rj.write_text(json.dumps(res, indent=1, default=str))
        n += 1
    return n


def _evaluate_and_log(task: Task, ad: Path, sub: Submission | None, led: Ledger) -> Evaluation:
    if sub is None:
        ev = Evaluation(s_plus=0.0, valid=False, details={"reason": "no submission"})
        cost = {"cost": 0.0}
    else:
        ev = task.evaluate(ad, sub)
        cost = {"cost": 0.0}
        if sub.type == "delta_w" and sub.delta is not None:
            base = task.base_weights(ad)
            if base is None:
                ev.valid, ev.details["reason"] = False, "task does not accept ΔW"
            else:
                c = scoring.TrackConstants.for_track(task.track)
                # All-zero λ (TR-03): no global edit cost; the task checks its per-case constraints itself.
                if any(c.lam):
                    try:
                        cost = scoring.edit_cost(sub.delta, base, c.lam,
                                                 zero_init_ref=task.frobenius_reference(ad))
                    except ValueError as e:
                        ev.valid, ev.details["reason"] = False, str(e)
                b = task.budget
                if ev.valid and b.max_l0 is not None and cost.get("l0", 0.0) > b.max_l0:
                    ranks_ok = b.max_rank is not None and all(
                        d.ndim == 2 and int(__import__("numpy").linalg.matrix_rank(d)) <= b.max_rank
                        for d in sub.delta.values() if d.any())
                    if not ranks_ok:
                        ev.valid, ev.details["reason"] = False, f"edit exceeds L0 {b.max_l0} and rank {b.max_rank}"
    led.append("evaluation", {"s_plus": ev.s_plus, "d_col": ev.d_col, "edit_cost": cost.get("cost", 0.0),
                              "edit_cost_terms": cost, "valid": ev.valid, "details": ev.details})
    return ev


def run_tag(agent: str, model: str | None) -> str:
    """System name used in run ids and the leaderboard (Claude Code runs keep the bare model id)."""
    if agent in ("baseline", "reference") or not model:
        return agent
    return model if agent == "claude" else f"{agent}-{model}".replace("/", "_").replace(":", "_")


def make_run_id(task_id: str, tag: str, seed: int, sample: int, level: int | None = None,
                condition: str = "full") -> str:
    """<task>__<system>__s<seed>__r<sample>, plus __L<level> only for a non-default knob level."""
    run_id = f"{task_id}__{tag}__s{seed}__r{sample}"
    if level is not None:
        run_id += f"__L{level}"
    return conditions.system_name(run_id, condition)


def _children_cpu_seconds() -> float | None:
    """CPU seconds (user + system) charged to reaped child processes so far, for a whole-sandbox-tree cross-check of
    the per-guarded-process accounting. POSIX only (``resource``); None on Windows / if unavailable. It misses
    processes that outlive the run or double-fork and reparent away before the sandbox child is waited on."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return float(usage.ru_utime + usage.ru_stime)
    except Exception:  # noqa: BLE001
        return None


def _author_usage() -> dict:
    t = os.times()
    return {"process": t.user + t.system, "children": t.children_user + t.children_system,
            "thread": time.thread_time(), "wall": time.monotonic()}


def _author_compute(before: dict) -> dict:
    """Compute of an author-side construction (``--agent baseline`` / ``--agent reference``), as executed by this
    runner: the ``task.baseline`` / ``task.reference`` call only (not the artifact build, not the evaluation).

    ``cpu_seconds`` = this process's user + system delta (all its threads, so BLAS worker threads are included; under
    ``sweep --jobs > 1`` concurrent runs in the same process are included too, so it is an upper bound there);
    ``cpu_thread_seconds`` = the calling thread only (exact for single-threaded Python, excludes BLAS workers);
    ``cpu_children_seconds`` = reaped child processes (heavy steps run locally as ``python -m`` subprocesses by
    ``weightbench.heavy`` with ``WB_HEAVY=local``); ``max_rss_kb`` = this process's lifetime peak RSS (POSIX; includes
    the harness itself, so an upper bound) and the peak of its reaped children. Steps dispatched to another machine
    (``WB_HEAVY`` other than ``local``) run elsewhere and are recorded as not measured."""
    after = _author_usage()
    out = {"source": "author", "cpu_seconds": round(after["process"] - before["process"], 6),
           "cpu_thread_seconds": round(after["thread"] - before["thread"], 6),
           "cpu_children_seconds": round(after["children"] - before["children"], 6),
           "wall_seconds": round(after["wall"] - before["wall"], 3), "max_rss_kb": None, "children_max_rss_kb": None}
    try:
        import resource

        out["max_rss_kb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        out["children_max_rss_kb"] = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    except Exception:  # noqa: BLE001 - resource is POSIX-only
        pass
    heavy = os.environ.get("WB_HEAVY", "local")
    out["heavy_mode"] = heavy
    if heavy != "local":
        out["remote_steps"] = "not measured"
    return out


def resolve_variant(task: Task, agent: str, variant: str | None) -> str | None:
    """The reference variant a run will use, validated against the task's declared ``reference_variants``.

    An explicit ``variant`` is only accepted for ``--agent reference`` and must be declared. Without one, a reference
    run of a task that declares variants uses ``task.default_reference_variant()`` (which may read a task-specific
    environment variable, e.g. ``WB_DEAD_REFERENCE``); it is validated the same way. Other runs: None."""
    declared = tuple(getattr(task, "reference_variants", ()) or ())
    if variant is not None:
        if agent != "reference":
            raise ValueError(f"--variant applies to --agent reference only (got --agent {agent})")
        if variant not in declared:
            raise ValueError(f"unknown reference variant {variant!r} for {task.id}; declared: "
                             f"{', '.join(declared) if declared else 'none'}")
        return variant
    if agent != "reference" or not declared:
        return None
    chosen = task.default_reference_variant()
    if chosen not in declared:
        raise ValueError(f"unknown reference variant {chosen!r} for {task.id} (from the task's default / environment); "
                         f"declared: {', '.join(declared)}")
    return chosen


def run(task_id: str, agent: str, model: str | None = None, sample: int = 0, seed: int = 0,
        max_turns: int = 200, agent_spec: agents.AgentSpec | None = None, level: int | None = None,
        condition: str = "full", access_plan: Path | None = None, variant: str | None = None) -> dict:
    """One (task, system, seed, sample). ``agent`` is baseline | reference | claude | codex | cmd; a custom
    command-line agent passes its ``agent_spec`` (``agents.AgentSpec.custom``). ``level``: knob level to build at
    (None = the task's default); a non-default level gets its own artifacts and an ``__L<level>`` run-id suffix.
    A run ID is reserved atomically and never reused, including after an interrupted/failed build.
    Choose a fresh sample ID for a retry; existing evidence is never overwritten."""
    conditions.validate(condition)
    if condition != "full":
        if agent in ("baseline", "reference"):
            raise ValueError("access conditions apply to agent runs, not author baseline/reference constructors")
        conditions.profile(task_id)
    level = nondefault_level(task_id, level)
    task = get_task(task_id, level)
    variant = resolve_variant(task, agent, variant)       # rejects an unknown variant before any build / evaluation
    planned = None
    if access_plan is not None:
        from weightbench import comparison
        planned = comparison.planned_run(access_plan, task_id, agent, model, seed, sample,
                                         level, condition, max_turns)
    run_id = make_run_id(task_id, run_tag(agent, model), seed, sample, level, condition)
    agent_auth.preflight(agent, model)       # an unusable login must not burn a run id (cached once per process)
    run_dir = RUNS / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"run ID already exists: {run_id}. Existing evidence was preserved; "
                              "choose a fresh --sample ID (even for interrupted runs or --force).") from exc
    led = Ledger(run_dir / "ledger.jsonl")
    start = {"task": task_id, "agent": agent, "model": model, "sample": sample, "seed": seed,
             "condition": condition}
    if planned:
        start['access_plan_sha256'] = planned[2]
    if level is not None:
        start["level"] = level
    identity = {}
    if variant is not None:          # only reference runs of tasks that declare variants; older ledgers lack it
        identity = {"variant": variant, "default_variant": task.reference_variants[0]}
        start.update(identity)
    flop_cap = getattr(getattr(task, "budget", None), "flop_cap", None)
    if flop_cap is not None:
        start["flop_cap"] = flop_cap                  # the cap the guard enforces (from the task, never the CLI)
    led.append("run_start", start)
    try:
        ad = build(task_id, seed, level=level)
    except Exception as exc:
        led.append("run_error", {"stage": "build", "exception_type": type(exc).__name__, "error": str(exc)})
        raise
    extra: dict = {}
    context: dict = {}

    if agent in ("baseline", "reference"):
        usage0 = _author_usage()
        if agent == "baseline":
            sub = task.baseline(ad)
        else:
            sub = task.reference(ad, variant) if variant is not None else task.reference(ad)
        compute = _author_compute(usage0)
        led.append("submission", {"type": sub.type, "source": agent, **identity,
                                  **({"abstained": True, "reason": sub.report.get("reason")}
                                     if sub.report and sub.report.get("abstained") else {})})
        led.append("compute", compute)
        extra = {"compute": compute}
    else:
        spec = agent_spec or agents.get(agent)
        if not model:
            raise ValueError(f"agent {agent!r} needs --model")
        workdir = WORK / run_id
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True)
        if condition == "full":
            _copy_public(ad / "public", workdir)
        else:
            access = conditions.copy_inputs(ad / "public", workdir, task_id, condition)
            access.update(probes=conditions.probes(task_id, task.probe_kinds, condition),
                          budget=asdict(task.budget),
                          local_model_execution=False,
                          max_turns=max_turns,
                          harness_source_sha256=conditions.source_fingerprint(),
                          task_prompt_sha256=hashlib.sha256(task.prompt(ad).encode()).hexdigest())
            context['harness_source_sha256'] = access['harness_source_sha256']
            if planned:
                from weightbench import comparison, manifest
                plan, case, plan_sha = planned
                try:
                    access['artifact_sha256'] = comparison.digest(manifest.seed_entry(ad))
                    access['access_plan_sha256'] = context['access_plan_sha256'] = plan_sha
                    comparison.validate_inputs(plan, case, condition, access)
                except (OSError, ValueError) as exc:
                    led.append('run_error', {'stage': 'access_plan', 'exception_type': type(exc).__name__,
                                             'error': str(exc)})
                    raise
            adapter = asdict(spec)
            adapter['build'] = spec.command(model, AGENT_PREAMBLE, Path('PROMPT.md'), max_turns)
            access['agent_spec_sha256'] = hashlib.sha256(json.dumps(adapter, sort_keys=True).encode()).hexdigest()
            led.append("access_inputs", access)
        shutil.copy(CLIENT, workdir / "wb.py")
        prompt = task.prompt(ad)
        if condition != "full":
            allowed = sorted(conditions.probes(task_id, task.probe_kinds, condition))
            prompt = (f"Matched access comparison. Only these metered behavior probes are available: {allowed}.\n"
                      "Internal activation/attention probes are disabled. Task budgets and scoring are unchanged.\n"
                      "Do not execute any subject or donor network locally, including exceptions in the original task text.\n"
                      "Observe behavior only through the listed metered probes.\n"
                      "Use only the files present; task text may describe weight files withheld in this condition.\n\n" + prompt)
        (workdir / "TASK.md").write_text(prompt)
        token = secrets.token_hex(16)
        state = RunState(task, ad, run_dir, led, workdir, token=token, condition=condition)
        srv = make_server(state)
        wb_url = f"http://127.0.0.1:{srv.server_address[1]}"
        (workdir / ".wb_url").write_text(wb_url)
        (workdir / ".wb_token").write_text(token)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        transcript = run_dir / "transcript.jsonl"
        guard_name = run_id if condition == "full" else "control-" + hashlib.sha256(run_id.encode()).hexdigest()[:24]
        guard_root = sandbox.install_guard(GUARDS / guard_name,
                                           ad / "public" if condition == "full" else workdir,
                                           task.meter_exclude if condition == "full" else (),
                                           **({"flop_cap": flop_cap} if flop_cap is not None else {}),
                                           # the agent opens its workdir copies (same path inside the sandbox)
                                           **({"visible_dirs": [workdir]} if condition == "full" else {}))
        proxy = sandbox.EgressProxy(run_dir / "egress.jsonl", allow=spec.egress_hosts).start()
        t0 = time.time()
        tree0 = _children_cpu_seconds()
        try:
            rc = _launch_agent(spec, model, workdir, transcript, task.budget.wall_clock_s, max_turns, guard_root,
                               {"api": srv.server_address[1], "proxy": proxy.port}, task.budget.flop_cap,
                               wb_url, token)
        finally:
            try:
                proxy.stop()
            finally:
                srv.shutdown()
                srv.server_close()   # drain request ledger writes before auditing or loading submission
        tree_cpu = _children_cpu_seconds()
        launched = _TREE_REPORTS.pop(str(guard_root), None) or {"expected": False, "report": None}
        tree = launched["report"]
        if tree is not None:            # Linux sandbox: the init's whole-tree report (PID-namespace safe)
            cpu_tree_seconds, tree_source = round(float(tree["tree_cpu_seconds"]), 6), "sandbox_init"
        elif launched["expected"]:      # the init wrote nothing: no whole-tree number (audited below)
            cpu_tree_seconds, tree_source = None, "missing"
        else:                           # macOS / a stub launcher: rusage of reaped children (no PID namespace)
            cpu_tree_seconds = None if tree0 is None or tree_cpu is None else round(tree_cpu - tree0, 6)
            tree_source = None if cpu_tree_seconds is None else "rusage_children"
        cpu_cap = task.budget.cpu_seconds
        server_traces = [e["payload"] for e in led.entries("runtime_trace")]
        report = harness_audit.audit_run(transcript, guard_root / "trace", run_dir / "egress.jsonl", ROOT,
                                         server_traces=server_traces, transcript_format=spec.transcript_format,
                                         python_agent=spec.python_runtime,
                                         allowed_flags=task.audit_allowed_flags if condition == "full" else (),
                                         trace_event_limit_reached=any(led.entries("runtime_trace_limit")),
                                         cpu_cap=cpu_cap, cpu_tree=tree,
                                         cpu_tree_missing=launched["expected"] and tree is None and rc != -9)
        (run_dir / "audit.json").write_text(json.dumps(report, indent=1))
        tr = report["trace"]
        # Form-independent compute accounting: guarded-process CPU seconds (the cap applies to this), plus the whole
        # sandbox tree's CPU as the backstop for unguarded children. See harness/redteam.md.
        compute = {"cpu_seconds": tr.get("cpu_seconds", 0.0), "cpu_children_seconds": tr.get("cpu_children_seconds"),
                   "max_rss_kb": tr.get("max_rss_kb", 0.0), "cpu_tree_seconds": cpu_tree_seconds, "cpu_cap": cpu_cap}
        if tree_source is not None:
            compute["cpu_tree_source"] = tree_source
        if tree is not None:
            compute.update({k: report["cpu_tree"].get(k) for k in harness_audit.TREE_COMPUTE_FIELDS})
        led.append("compute", compute)
        extra = {"returncode": rc, "stalled": rc == STALL_EXIT, "wall_s": round(time.time() - t0, 1),
                 "usage": _agent_usage(transcript),
                 "audit_flags": report["flags"], "quarantined": report["quarantined"],
                 "trace": {k: report["trace"][k] for k in ("processes", "flops", "weight_matmuls", "forward_chains")},
                 "egress_denied": report["egress"]["n_denied"], "compute": compute}
        led.append("agent_exit", extra)
        auth_failure = agent_auth.transcript_auth_failure(transcript)
        if auth_failure:                      # expired login: no tool call, auth error text. Still invalid, still ledgered
            context["agent_auth_failed"] = auth_failure
        sub = Submission.load(run_dir / "submission")
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(guard_root, ignore_errors=True)

    if planned:
        from weightbench import comparison, manifest
        plan, case, _ = planned
        if (comparison.digest(manifest.seed_entry(ad)) != case['artifact_sha256']
                or conditions.source_fingerprint() != plan['harness_source_sha256']):
            led.append('run_error', {'stage': 'planned_assets', 'error': 'frozen subject or source changed during run'})
            raise ValueError('frozen subject or source changed during the planned run')
    ev = _evaluate_and_log(task, ad, sub, led)
    res = {"run_id": run_id, "task": task_id, "track": task.track, "tier": task.tier, "agent": agent, "model": model,
           "sample": sample, "seed": seed, "condition": condition,
           **score_from_ledger(task, led, ev), "details": ev.details,
           **identity, **extra, **context, "ledger_ok": led.verify()}
    if level is not None:
        res["level"] = level
    (run_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


TIER_ORDER = {"foundational": 0, "intermediate": 1, "frontier": 2}


def difficulty_key(task_id: str) -> tuple:
    """Easy -> hard: tier first (foundational, intermediate, frontier), then the task's knob level if it has one,
    then track and id, so `bench` climbs the difficulty ramp and the leaderboard shows where a system falls off."""
    task = REGISTRY[task_id]
    level = getattr(task, "knob_level", 0) or 0
    return TIER_ORDER.get(getattr(task, "tier", "intermediate"), 1), level, task.track, task_id


def select_tasks(pattern: str = "*", tracks: str = "", tiers: str = "") -> list[str]:
    """Task ids matching a glob, a comma-separated track list (TR01,TR04) and a tier list
    (foundational,intermediate,frontier). Empty filters select everything."""
    want_tracks = {t.strip().upper().replace("-", "") for t in tracks.split(",") if t.strip()}
    want_tiers = {t.strip().lower() for t in tiers.split(",") if t.strip()}
    patterns = [p.strip() for p in pattern.split(",") if p.strip()] or ["*"]
    return [tid for tid in sorted(REGISTRY, key=difficulty_key) if any(fnmatch.fnmatch(tid, p) for p in patterns)
            and (not want_tracks or REGISTRY[tid].track in want_tracks)
            and (not want_tiers or REGISTRY[tid].tier in want_tiers)
            and REGISTRY[tid].track != "SMOKE" and not tid.startswith("SMOKE")]


FLOOR_AGENTS = ("baseline", "reference")


def agent_spec_from_args(a: argparse.Namespace) -> agents.AgentSpec | None:
    """The sandbox adapter this invocation drives, or None for a floor.

    ``baseline`` and ``reference`` are not CLI adapters: the harness runs them itself, so they never get a spec —
    not even on a batch command that also passes --allow-host/--agent-home for its real agent (``agents.get``
    would reject them as unknown agents, breaking `bench --with-floors`).
    """
    if a.agent in FLOOR_AGENTS:
        return None

    def items(value: str) -> list[str]:
        return [x.strip() for x in value.split(",") if x.strip()]

    return agents.get(a.agent, a.agent_cmd, items(a.allow_host), items(a.agent_home),
                      [int(p) for p in items(a.allow_port)], a.python_agent,
                      getattr(a, "agent_transcript_format", "text"))


def _batch_samples(first_sample: int, count: int) -> range:
    if first_sample < 0 or count < 1:
        raise SystemExit("--sample must be nonnegative and --samples must be at least 1")
    return range(first_sample, first_sample + count)


def _pending_batch_runs(jobs: list[tuple[str, str, str | None, int]], seed: int,
                        force: bool = False, condition: str = "full") -> list[tuple[str, str, str | None, int]]:
    """Preflight every ID before builds/calls; reuse only completed, verified evidence."""
    from weightbench.verify import verify_run

    pending = []
    planned = set()
    collisions: list[tuple[str, int, bool]] = []     # (run id, sample, finished): ids that block a fresh start
    for job in jobs:
        tid, agent, model, sample = job
        selected = condition if model else "full"
        run_id = make_run_id(tid, run_tag(agent, model), seed, sample, condition=selected)
        if run_id in planned:
            raise ValueError(f"batch contains duplicate run ID: {run_id}; use distinct systems/sample IDs")
        planned.add(run_id)
        directory = RUNS / run_id
        if not directory.exists() and not directory.is_symlink():
            pending.append(job)
            continue
        if force or not (directory / "result.json").is_file():
            collisions.append((run_id, sample, (directory / "result.json").is_file()))
            continue
        report = verify_run(directory)
        result = json.loads((directory / "result.json").read_text())
        expected = {"run_id": run_id, "task": tid, "agent": agent, "model": model, "seed": seed, "sample": sample}
        # A cached reference run of another variant (e.g. crown under WB_DEAD_REFERENCE) is not this job. Results
        # recorded before variants existed carry no field and are reused as before.
        variant = resolve_variant(REGISTRY[tid](), agent, None) if agent == "reference" and tid in REGISTRY else None
        if (not report.ok or any(c.status == "WARN" for c in report.checks)
                or result.get("variant", variant) != variant
                or result.get("condition", "full") != selected
                or any(result.get(key) != value for key, value in expected.items())):
            raise ValueError(f"cached run failed verification: {run_id}; inspect its evidence and choose "
                             "a fresh --sample ID (use rescore for an old formula)")
        print(f"verified cached run: {run_id}", flush=True)
    if collisions:
        # Nothing was built or called. Ids are never reused and evidence is never replaced; say which ids block the
        # batch and which --sample continues past them (floors use the batch's first sample id too).
        listing = "; ".join(f"{rid} ({'finished, --force cannot replace it' if done else 'interrupted: no result.json'})"
                            for rid, _, done in collisions)
        raise FileExistsError(f"run ID already exists: {listing}. Existing evidence was preserved; choose a fresh "
                              f"--sample ID (--force cannot replace it). Nothing was run or charged. To continue "
                              f"with fresh samples, repeat the command with --sample "
                              f"{max(s for _, s, _ in collisions) + 1} (leave the folders in runs/ untouched).")
    return pending


def bench(a: argparse.Namespace) -> None:
    """``weightbench bench``: one command that benchmarks an agent on a slice of WeightBench, sandboxed, and
    prints the leaderboard. Results land in the runs and results directories like every other run."""
    from weightbench import report

    samples = _batch_samples(a.sample, a.samples)
    tasks = select_tasks(a.tasks, a.tracks, a.tiers)
    if not tasks:
        raise SystemExit("no task matches --tasks / --tracks / --tiers")
    systems = [(a.agent, a.model)] + ([("baseline", None), ("reference", None)] if a.with_floors else [])
    jobs = [(tid, agent, model, sample) for tid in tasks for agent, model in systems
            for sample in (samples if model else [a.sample])]
    pending = _pending_batch_runs(jobs, a.seed, a.force, a.condition)
    if not pending:
        report.main()
        return
    spec = agent_spec_from_args(a)
    backend = backends.detect()                     # fail fast on unsupported hosts, before any build
    try:
        agent_auth.preflight(a.agent, a.model)      # and on an unusable login, before any build or run id
    except agent_auth.AgentAuthError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"WeightBench: {len(tasks)} tasks × {a.samples} sample(s), agent {spec.name if spec else a.agent} / {a.model}, "
          f"sandbox {backend.name}", flush=True)
    for tid in dict.fromkeys(j[0] for j in pending):
        build(tid, a.seed)
    failures = 0
    for tid, agent, model, sample in pending:
        selected = a.condition if model else "full"
        run_id = make_run_id(tid, run_tag(agent, model), a.seed, sample, condition=selected)
        try:
            r = run(tid, agent, model, sample, a.seed, a.max_turns, spec if model else None, None, selected)
            print(f"{r['run_id']}: S_WB={r['s_wb']:.3f} S+={r['s_plus']:.3f} valid={r['valid']}"
                  f"{' QUARANTINED' if r.get('quarantined') else ''}"
                  f"{' AGENT_AUTH_FAILED (log in again: run `claude`, then /login)' if r.get('agent_auth_failed') else ''}",
                  flush=True)
        except Exception as e:  # noqa: BLE001 - finish independent runs, then fail the batch
            failures += 1
            print(f"FAILED {run_id}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    report.main()
    if failures:
        raise SystemExit(1)


def sweep(a: argparse.Namespace) -> None:
    """Run a parallel batch with the same sample/cache/error policy as bench."""
    samples = _batch_samples(a.sample, a.samples)
    if a.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    tasks = select_tasks(a.tasks, a.tracks, a.tiers)
    if not tasks:
        raise SystemExit("no task matches --tasks / --tracks / --tiers")
    jobs = [(t, agent, None, a.sample) for agent in ("baseline", "reference") for t in tasks]
    jobs += [(t, "claude", m, s) for m in filter(None, a.models.split(",")) for t in tasks for s in samples]
    pending = _pending_batch_runs(jobs, a.seed, a.force, a.condition)
    for tid in dict.fromkeys(j[0] for j in pending):
        build(tid, a.seed)
    print(f"{len(pending)} runs to do ({len(jobs) - len(pending)} verified cached)", flush=True)

    def go(job: tuple[str, str, str | None, int]) -> bool:
        try:
            r = run(job[0], job[1], job[2], job[3], a.seed, a.max_turns, None, None,
                    a.condition if job[2] else "full")
            print(f"{r['run_id']}: S_WB={r['s_wb']:.3f} S+={r['s_plus']:.3f} valid={r['valid']}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {job}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return False

    with ThreadPoolExecutor(a.jobs) as pool:
        outcomes = list(pool.map(go, pending))
    if not all(outcomes):
        raise SystemExit(1)


WSL_PASSTHROUGH_ENV = ("WB_HEAVY", "WB_AGENT_VENV", "WB_RESULTS_DIR", "WB_SANDBOX_HOME_ALLOW", "HF_TOKEN",
                       "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def needs_agent_sandbox(a: argparse.Namespace) -> bool:
    return a.cmd in ("bench", "sweep") or (a.cmd == "run" and a.agent not in ("baseline", "reference"))


def wsl_command(argv: list[str], repo: Path = ROOT) -> list[str]:
    """The same `weightbench` invocation, run inside WSL2 from the repo's /mnt/<drive> path, where the Linux
    bubblewrap sandbox is available (Windows has no unprivileged equivalent)."""
    converted = []
    for i, value in enumerate(argv):
        if i and argv[i - 1] == '--access-plan':
            value = to_wsl_path(Path(value).resolve())
        elif value.startswith('--access-plan='):
            value = '--access-plan=' + to_wsl_path(Path(value.split('=', 1)[1]).resolve())
        converted.append(value)
    exports = " ".join(f"{k}={shlex.quote(os.environ[k])}" for k in WSL_PASSTHROUGH_ENV if os.environ.get(k))
    inner = (f"cd {shlex.quote(to_wsl_path(repo))} && export PATH=\"$HOME/.local/bin:$PATH\" "
             f"WB_AGENT_VENV=\"${{WB_AGENT_VENV:-$HOME/wb-agent-venv}}\" WB_HEAVY=local {exports} && "
             f"\"$HOME/wb-venv/bin/weightbench\" " + " ".join(shlex.quote(x) for x in converted))
    return ["wsl.exe", "-e", "bash", "-lc", inner]


def run_in_wsl(argv: list[str]) -> int:
    probe = subprocess.run(["wsl.exe", "-e", "bash", "-lc", "test -x \"$HOME/wb-venv/bin/weightbench\""])
    if probe.returncode != 0:
        print("The agent sandbox runs inside WSL2 on Windows and is not set up yet. Run once:\n"
              "    weightbench setup-wsl\n(needs WSL2 with Ubuntu: `wsl --install`)", file=sys.stderr)
        return 2
    print("WeightBench: running the agent sandbox inside WSL2 (Linux bubblewrap backend)", flush=True)
    return subprocess.run(wsl_command(argv)).returncode


def setup_wsl() -> int:
    if not wsl_available():
        print("WSL2 is not available: install it with `wsl --install`, then re-run.", file=sys.stderr)
        return 2
    script = to_wsl_path(ROOT / "scripts" / "setup_wsl.sh")
    return subprocess.run(["wsl.exe", "-e", "bash", "-lc", f'WB_HARNESS_VENV="$HOME/wb-venv" bash {shlex.quote(script)}']).returncode


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ['doctor']:
        from weightbench.doctor import main as doctor_main
        raise SystemExit(doctor_main(raw_argv[1:]))
    ap = argparse.ArgumentParser(prog="weightbench", description="WeightBench: benchmark for Weight-Space "
                                 "Programming. `weightbench bench --agent claude --model <id>` runs an agent.")
    ap.add_argument("cmd", choices=["bench", "build", "run", "sweep", "list", "rescore", "doctor", "setup-wsl"])
    ap.add_argument("--task")
    ap.add_argument("--tasks", default="*", help="glob over task ids, e.g. 'TR04-*'")
    ap.add_argument("--tracks", default="", help="comma-separated tracks, e.g. TR01,TR04")
    ap.add_argument("--tiers", default="", help="comma-separated tiers: foundational,intermediate,frontier")
    ap.add_argument("--agent", default="claude", help="baseline | reference | claude | codex | cmd")
    ap.add_argument("--agent-cmd", default=None, help="command template for --agent cmd, e.g. "
                    "\"my-agent --model {model} --prompt-file {prompt_file}\"")
    ap.add_argument("--allow-host", default="", help="egress allowlist for --agent cmd: host (port 443) or host:port")
    ap.add_argument("--allow-port", default="", help="local model server port(s) on this machine for --agent cmd")
    ap.add_argument("--agent-transcript-format", default="text", choices=agents.TRANSCRIPT_FORMATS,
                    help="for --agent cmd: how stdout is audited. 'text' (default) audits every line; use "
                    "'claude-stream-json' when the agent prints Claude-style events, so the audit reads only the "
                    "tool-call inputs and assistant text")
    ap.add_argument("--python-agent", action="store_true", help="the custom agent itself is a Python program")
    ap.add_argument("--agent-home", default="", help="home-relative paths the custom agent needs (auth, config)")
    ap.add_argument("--model")
    ap.add_argument("--models", default="")
    ap.add_argument("--sample", type=int, default=0, help="first sample ID (run: one ID; bench/sweep: range start)")
    ap.add_argument("--samples", type=int, default=1, help="bench/sweep: agent samples starting at --sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=None, help="build / run: knob level (SPEC §5; default: the "
                    "task's own level). Only tasks with a wired knob accept it.")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--condition", choices=conditions.CONDITIONS, default="full",
                    help="matched access control; both matched arms use behavior probes only")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument('--access-plan', type=Path, help='run: validate this individual run against a declared access plan')
    ap.add_argument("--variant", default=None, help="run --agent reference: one of the task's declared reference "
                    "variants (e.g. TR05-DEAD: ibp, crown); recorded in the ledger and result.json")
    ap.add_argument("--with-floors", action="store_true", help="bench: also score baseline and reference")
    ap.add_argument("--force", action="store_true", help="rebuild artifacts; never replaces an existing run ID")
    a = ap.parse_args(argv)
    if a.access_plan is not None and a.cmd != 'run':
        ap.error('--access-plan applies to individual planned runs')
    if a.variant is not None and (a.cmd != "run" or a.agent != "reference"):
        ap.error("--variant applies to `run --agent reference` only")
    if a.cmd == "setup-wsl":
        sys.exit(setup_wsl())
    if a.cmd in ('build', 'run') and not a.task:
        ap.error(f'{a.cmd} requires --task TASK_ID; use `weightbench list` to see tasks')
    if sys.platform == "win32" and needs_agent_sandbox(a):
        if not wsl_available():
            sys.exit("Agent runs require Ubuntu in WSL2 on Windows. Run `wsl --install -d Ubuntu`, "
                     "then follow the Windows instructions in README.md inside Ubuntu.")
        sys.exit(run_in_wsl(raw_argv))
    load_all()
    if a.cmd == "rescore":
        print(f"rescored {rescore_all()} runs with formula v{scoring.FORMULA_VERSION}")
        return
    if a.cmd == "bench":
        if not a.model:
            raise SystemExit("bench needs --model (the agent's model id)")
        bench(a)
    elif a.cmd == "list":
        for k in select_tasks(a.tasks, a.tracks, a.tiers):
            print(f"{k}\t{REGISTRY[k].track}\t{REGISTRY[k].tier}")
    elif a.cmd == "build":
        print(build(a.task, a.seed, a.force, a.level))
    elif a.cmd == "run":
        # Always derive the spec: `cmd` needs it for its template, and a built-in adapter needs it whenever
        # --allow-host/--agent-home extend its defaults (Claude Code on a gateway). `agents.get` ignores extras
        # it does not use, so the plain Claude path is unchanged. A floor (`baseline`/`reference`) gets None.
        spec = agent_spec_from_args(a)
        planned_args = {} if a.access_plan is None else {'access_plan': a.access_plan}
        if a.variant is not None:
            planned_args["variant"] = a.variant
        try:
            result = run(a.task, a.agent, a.model, a.sample, a.seed, a.max_turns, spec, a.level, a.condition,
                         **planned_args)
        except agent_auth.AgentAuthError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(result, indent=1, default=str))
    else:
        sweep(a)


if __name__ == "__main__":
    main()
