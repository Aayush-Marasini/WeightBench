"""Per-run Weight API server: metered probes + submissions, all logged to the ledger.

The agent never talks to the task object directly; it uses ``wb.py`` (see client.py) which
POSTs here. Ground truth stays in this process / art_dir/private.
"""
from __future__ import annotations

import hmac
import json
import os
import shutil
import stat
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from weightbench.ledger import Ledger
from weightbench import conditions
from weightbench.harness.trace_schema import normalize_trace
from weightbench.task import Task

MAX_BODY_BYTES = 64 * 2 ** 20       # JSON bodies; large candidate weights go through a workdir file
MAX_TRACE_EVENTS = 5000             # runtime-guard events accepted per run (keeps the ledger bounded)
TOKEN_HEADER = "X-WB-Token"
AGENT_VISIBLE_ERRORS = (ValueError, KeyError, TypeError, IndexError)   # input mistakes; anything else is internal


def _copy_checked(src: Path, dst: Path, workdir: Path) -> bool:
    """Copy ``src`` (already resolved) to ``dst`` only if it is a regular file inside ``workdir`` *and* the file that
    is opened is the one that was checked. Without this, an agent could swap the checked file for a symlink to a
    private artifact (e.g. the reference ΔW) between the containment check and the copy, and have the server submit
    or probe it. The final component is opened without following symlinks and the opened file's identity (device,
    inode) must equal the checked one (this also catches a swapped parent directory). The copy itself is
    ``shutil.copy``; afterwards ``dst`` must hold exactly the bytes of the file that was verified and kept open, so a
    swap during the copy is rejected too. Copy errors (OSError) propagate to the caller."""
    try:
        if not src.is_file() or workdir.resolve() not in src.parents:
            return False
        checked = os.stat(src, follow_symlinks=False)
        fd = os.open(src, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    except (OSError, ValueError):
        return False
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (checked.st_dev, checked.st_ino) or not stat.S_ISREG(opened.st_mode):
            return False
        shutil.copy(src, dst)
        with open(dst, "rb") as copied:
            while True:
                a, b = handle.read(1 << 20), copied.read(1 << 20)
                if a != b:
                    break
                if not a:
                    return True
    Path(dst).unlink(missing_ok=True)
    return False


class RunState:
    def __init__(self, task: Task, art_dir: Path, run_dir: Path, ledger: Ledger, workdir: Path,
                 token: str | None = None, condition: str = "full"):
        self.task, self.art_dir, self.run_dir, self.ledger, self.workdir = task, art_dir, run_dir, ledger, workdir
        self.token = token                   # per-run secret: parallel runs on the same host can't use each other's API
        self.condition = conditions.validate(condition)
        self.probe_kinds = (dict(task.probe_kinds) if condition == "full"
                            else conditions.probes(task.id, task.probe_kinds, condition))
        self.used = {"forward": 0, "probe": 0}
        self.n_submissions = 0
        self.n_trace_events = 0
        self.trace_limit_reported = False
        self.lock = threading.Lock()
        self.submission_lock = threading.Lock()
        # Write tasks override Task.base_weights; read-only tasks (TR-01 and similar) do not and cannot score a ΔW.
        # A duck-typed task without the method (tests, adapters) is not second-guessed here; scoring still decides.
        impl = getattr(type(task), "base_weights", None)
        self.accepts_delta = impl is None or impl is not Task.base_weights

    # -- probes --
    def _private_copy(self, name: str) -> Path | None:
        """Copy a workdir file into the run dir before the task reads it, so the agent cannot swap the file
        (or a symlink) between the containment check and the read."""
        src = (self.workdir / name).resolve()
        if not src.is_file() or self.workdir.resolve() not in src.parents:
            return None
        dst = self.run_dir / "probe_inputs" / f"{uuid.uuid4().hex[:12]}{src.suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not _copy_checked(src, dst, self.workdir):
            dst.unlink(missing_ok=True)
            return None
        return dst

    def _agent_error(self, error: Exception) -> str:
        """What the agent sees for a failed probe: input errors with private paths redacted; nothing else."""
        if not isinstance(error, AGENT_VISIBLE_ERRORS):
            return f"{type(error).__name__}: internal evaluator error (logged)"
        message = str(error)
        for secret in {str(self.art_dir), str(Path(self.art_dir).resolve()), str(self.run_dir)}:
            message = message.replace(secret, "<private>")
        return f"{type(error).__name__}: {message}"

    def probe(self, kind: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if not isinstance(payload, dict):
            return 400, {"error": "payload must be a JSON object"}
        kinds = self.probe_kinds
        if kind not in kinds:
            if kind in self.task.probe_kinds:
                self.ledger.append("access_rejected", {"condition": self.condition, "probe": kind})
                return 403, {"error": "internal probes are unavailable in matched conditions"}
            # A refused probe attempt leaves a trace even on zero-budget tasks. Bounded, so an agent cannot flood
            # the ledger with unknown kinds.
            self._unknown_probes = getattr(self, "_unknown_probes", 0) + 1
            if self._unknown_probes <= 8:
                self.ledger.append("probe_rejected", {"kind": str(kind)[:64], "reason": "unknown probe kind"})
            return 400, {"error": f"unknown probe {kind!r}; available: {sorted(kinds)}"}
        copied = None
        if "file" in payload:  # optional: a file in the agent's workdir (e.g. candidate weights too big for JSON)
            copied = self._private_copy(str(payload["file"]))
            if copied is None:
                return 400, {"error": "file must be an existing file inside your workdir"}
            payload = {**payload, "file": str(copied)}
        meter = kinds[kind]
        cap = self.task.budget.n_forward if meter == "forward" else self.task.budget.n_probe
        with self.lock:
            if self.used[meter] >= cap:
                self.ledger.append("probe_rejected", {"kind": kind, "meter": meter, "cap": cap})
                return 429, {"error": f"{meter} budget exhausted ({cap})"}
            self.used[meter] += 1
            n = self.used[meter]
        error_detail = None
        try:
            out = self.task.probe(self.art_dir, kind, payload)
        except Exception as e:  # noqa: BLE001 - report probe errors to the agent, still metered
            out = {"error": self._agent_error(e)}
            error_detail = f"{type(e).__name__}: {e}"
        finally:
            if copied is not None:
                copied.unlink(missing_ok=True)
        entry = {"kind": kind, "meter": meter, "n_used": n, "cap": cap}
        if error_detail:
            entry["error"] = error_detail[:2000]
        self.ledger.append("probe", entry)
        return 200, {"result": out, "used": n, "cap": cap}

    # -- runtime-guard events (tamper-evident: once in the append-only ledger they cannot be deleted) --
    def trace(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            keep = normalize_trace(body)
        except ValueError as error:
            return 400, {"error": str(error)}
        with self.lock:
            if self.n_trace_events >= MAX_TRACE_EVENTS:
                if not self.trace_limit_reported:
                    self.ledger.append("runtime_trace_limit", {"cap": MAX_TRACE_EVENTS,
                                                               "accepted": self.n_trace_events})
                    self.trace_limit_reported = True
                return 429, {"error": "trace event limit reached"}
            self.n_trace_events += 1
        self.ledger.append("runtime_trace", keep)
        return 200, {"ok": True}

    # -- submissions (last one wins) --
    def submit(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        # File installation, numbering and ledger order must agree across HTTP threads.
        with self.submission_lock:
            return self._submit(body)

    def _submit(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        typ = body.get("type")
        sub_dir = self.run_dir / "submission"
        meta: dict[str, Any] = {"type": typ}
        src = None
        if typ in ("report", "refusal"):
            if not isinstance(body.get("report"), dict):
                return 400, {"error": "report must be a JSON object"}
            meta["report"] = body["report"]
        elif typ == "delta_w":
            if not self.accepts_delta:
                # Scoring would mark this run invalid ("task does not accept ΔW"); say so now so the agent can
                # resubmit the right artefact instead of finding out after it exits.
                return 400, {"error": "this task does not accept a ΔW submission; read TASK.md and submit a "
                                      "report (wb.py submit-report) instead"}
            name = body.get("path")
            if not isinstance(name, str) or not name or "\x00" in name:
                return 400, {"error": "delta path must be a nonempty string"}
            try:
                src = (self.workdir / name).resolve()
            except (OSError, ValueError, RuntimeError):
                return 400, {"error": "invalid delta path"}
            if not src.is_file() or self.workdir.resolve() not in src.parents:
                return 400, {"error": "delta path must be an existing .npz inside your workdir"}
            if body.get("report") is not None:  # optional (TR-03: per-case decisions alongside the edits)
                if not isinstance(body["report"], dict):
                    return 400, {"error": "report must be a JSON object"}
                meta["report"] = body["report"]
        else:
            return 400, {"error": "type must be report | refusal | delta_w"}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="submission-", dir=self.run_dir) as staging:
            root = Path(staging)
            candidate, previous = root / "candidate", root / "previous"
            candidate.mkdir()
            try:
                if src is not None and not _copy_checked(src, candidate / "delta.npz", self.workdir):
                    return 400, {"error": "delta path must be an existing .npz inside your workdir"}
                (candidate / "submission.json").write_text(json.dumps(meta))
                if sub_dir.exists():
                    sub_dir.rename(previous)
                try:
                    candidate.rename(sub_dir)
                except OSError:
                    if previous.exists():
                        previous.rename(sub_dir)
                    raise
            except OSError:
                return 500, {"error": "could not store submission; previous submission retained"}
            self.n_submissions += 1
            self.ledger.append("submission", {"type": typ, "n": self.n_submissions})
        return 200, {"ok": True, "note": "submission stored; you may resubmit, the last one counts"}


def make_server(state: RunState, port: int = 0, *, request_timeout_s: float = 30.0) -> ThreadingHTTPServer:
    class RunServer(ThreadingHTTPServer):
        # server_close() joins admitted requests before audit/evaluation reads their evidence.
        daemon_threads = False

    class H(BaseHTTPRequestHandler):
        def setup(self) -> None:
            self.request.settimeout(request_timeout_s)
            super().setup()

        def log_message(self, *a):  # silence
            pass

        def _send(self, code: int, obj: dict[str, Any]) -> None:
            b = json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _authorized(self) -> bool:
            if state.token is None:
                return True
            if hmac.compare_digest(self.headers.get(TOKEN_HEADER, ""), state.token):
                return True
            self._send(401, {"error": "missing or wrong run token"})
            return False

        def do_GET(self):
            if not self._authorized():
                return
            if self.path == "/budget":
                b = state.task.budget
                self._send(200, {"used": state.used, "n_forward": b.n_forward, "n_probe": b.n_probe,
                                 "probes": state.probe_kinds, "condition": state.condition})
            else:
                self._send(404, {"error": "GET /budget only"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                return self._send(400, {"error": "bad Content-Length"})
            if not 0 <= n <= MAX_BODY_BYTES:
                return self._send(413, {"error": f"body larger than {MAX_BODY_BYTES} bytes; pass a workdir file"})
            try:
                raw = self.rfile.read(n)      # read before rejecting, so the client always receives the reply
            except TimeoutError:
                return self._send(408, {"error": "request body timed out"})
            if not self._authorized():
                return
            try:
                body = json.loads(raw or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._send(400, {"error": "invalid JSON"})
            if not isinstance(body, dict):
                return self._send(400, {"error": "body must be a JSON object"})
            if self.path == "/probe":
                code, out = state.probe(str(body.get("kind")), body.get("payload", {}))
            elif self.path == "/submit":
                code, out = state.submit(body)
            elif self.path == "/trace":
                code, out = state.trace(body)
            else:
                code, out = 404, {"error": "POST /probe, /submit or /trace"}
            self._send(code, out)

    return RunServer(("127.0.0.1", port), H)
