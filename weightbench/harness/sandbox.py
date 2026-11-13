"""Agent sandbox (SPEC §1 *Enforcement*): concrete restrictions for every benchmarked agent run.

Layers, all set up by ``weightbench.runner`` for ``--agent claude`` runs:

1. **Filesystem / exec** (macOS ``sandbox-exec``): the agent cannot read the repo (ground truth, ledgers,
   other runs), other workdirs or guard dirs, Claude Code project memory, or anything in $HOME beyond what
   Claude Code needs to start. Writes are denied everywhere except its own workdir, the guard's trace dir,
   Claude Code's state and system temp. Rules cover both /tmp and /private/tmp spellings. Compilers and
   foreign interpreters (cc, clang, swift, perl, ruby, osascript, ...) cannot be executed.
2. **Network**: all outbound connections are denied by the OS sandbox except to localhost. Claude Code
   reaches its API through ``EgressProxy``, a CONNECT proxy run by the runner (outside the sandbox) that
   allows only ``EGRESS_ALLOW`` hosts on port 443 and logs every attempt. The Weight API is on localhost.
3. **Runtime guard** (``agent_guard.py`` as ``sitecustomize``): every agent Python process blocks
   non-local sockets, model-execution / simulator libraries and gradient descent, and writes a FLOP /
   forward-chain trace.
4. **Audit** (``audit.py``): transcript + runtime traces + egress log; flagged runs are quarantined.

    python -m weightbench.harness.sandbox selftest      # macOS: verifies layers 1-3 end to end
"""
from __future__ import annotations

import fnmatch
import json
import os
import posixpath
import select
import shutil
import socket
import socketserver
import struct
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Iterable

EGRESS_ALLOW = ("api.anthropic.com",)
GUARD_SOURCE = Path(__file__).resolve().parent / "agent_guard.py"


# ---- 1. filesystem + network profile (macOS sandbox-exec, SBPL) --------------------------------
# macOS resolves these top-level symlinks before the sandbox matches a path, so every rule is written
# for both spellings (a rule for /tmp/x alone would never match the real path /private/tmp/x).
PRIVATE_ALIASES = (("/tmp", "/private/tmp"), ("/var", "/private/var"), ("/etc", "/private/etc"))

# Home-directory paths Claude Code (and a uv / pyenv / nvm Python or Node) needs to start. The rest of
# $HOME is unreadable. Extend with WB_SANDBOX_HOME_ALLOW (os.pathsep-separated) if your agent needs more.
HOME_READ_ALLOW = (".claude", ".claude.json", ".config/claude", ".local", ".npm", ".nvm", ".volta", ".bun",
                   ".pyenv", ".cache/uv", "Library/Caches/claude-cli-nodejs", "Library/Application Support/Claude")
# Writable home paths: Claude Code's own state (settings, session files, logs).
HOME_WRITE_ALLOW = (".claude", ".config/claude", "Library/Caches/claude-cli-nodejs")
# Compilers and interpreters other than the agent venv's Python: no unmetered native / foreign code paths.
BLOCKED_EXEC_LITERALS = tuple(f"/usr/bin/{name}" for name in (
    "cc", "c++", "gcc", "g++", "clang", "clang++", "ld", "swift", "swiftc", "perl", "perl5.30", "perl5.34",
    "ruby", "osascript", "xcrun", "lldb", "gdb", "java", "javac", "php", "tclsh", "wish"))
BLOCKED_EXEC_SUBPATHS = ("/Library/Developer/CommandLineTools", "/Applications/Xcode.app")


def path_forms(path: str | Path) -> list[str]:
    """Every spelling of an absolute path the kernel may match: as given, symlink-resolved (macOS), and with
    the /tmp ↔ /private/tmp style aliases in both directions."""
    raw = posixpath.normpath(str(path).replace("\\", "/"))
    forms = {raw}
    if sys.platform == "darwin":
        forms.add(os.path.realpath(raw))
    for short, real in PRIVATE_ALIASES:
        for form in list(forms):
            if form == short or form.startswith(short + "/"):
                forms.add(real + form[len(short):])
            elif form == real or form.startswith(real + "/"):
                forms.add(short + form[len(real):])
    return sorted(forms)


def _filters(kind: str, paths: Iterable[str | Path]) -> str:
    return " ".join(f'({kind} "{form}")' for p in paths for form in path_forms(p))


def _regex_prefix(path: str | Path) -> str:
    """SBPL regex filters matching ``path`` itself and every sibling that starts with it (e.g. .claude.json.backup)."""
    return " ".join(f'(regex #"^{_sbpl_regex_escape(form)}")' for form in path_forms(path))


def _sbpl_regex_escape(text: str) -> str:
    """Escape POSIX extended-regex metacharacters only (SBPL rejects Python's extra escapes such as ``\\-``)."""
    return "".join("\\" + ch if ch in ".^$*+?()[]{}|\\" else ch for ch in text)


def profile_denies_read(profile: str, path: str | Path) -> bool:
    """True if every spelling of ``path`` appears in a ``deny file-read*`` rule of ``profile`` (for tests / audits)."""
    deny_lines = [line for line in profile.splitlines() if line.startswith("(deny file-read*")]
    return all(any(f'(subpath "{form}")' in line for line in deny_lines) for form in path_forms(path))


def macos_profile(workdir: Path, deny_read: Iterable[Path], guard_dir: Path | None = None,
                  home: Path | None = None, extra_read: Iterable[Path] = (), extra_write: Iterable[Path] = ()) -> str:
    """SBPL profile; later rules take precedence over earlier ones.

    Reads: everything outside $HOME (system libraries, the agent venv in /tmp) except ``deny_read``; inside $HOME
    only ``HOME_READ_ALLOW`` + ``extra_read``. Writes: denied everywhere except the workdir, the guard's trace dir,
    Claude Code's state and the system temp / device files. Exec: no compilers or foreign interpreters.
    Network: localhost only (the Weight API and the egress proxy).
    """
    home = Path(home or Path.home())
    home_read = [home / p for p in HOME_READ_ALLOW] + list(extra_read)
    home_read += [Path(p) for p in os.environ.get("WB_SANDBOX_HOME_ALLOW", "").split(os.pathsep) if p]
    rules = [f"(deny file-read* {_filters('subpath', [home])})",
             f"(allow file-read* {_filters('subpath', home_read)} {_regex_prefix(home / '.claude.json')})",
             f"(deny file-read* {_filters('subpath', deny_read)})",
             "(deny file-write*)",
             f"(allow file-write* {_filters('subpath', ['/dev', '/private/var/folders'])} "
             f"{_filters('subpath', [home / p for p in HOME_WRITE_ALLOW] + list(extra_write))} "
             f"{_regex_prefix(home / '.claude.json')})",
             f"(allow file-read* file-write* {_filters('subpath', [workdir])})"]
    if guard_dir is not None:
        rules.append(f"(allow file-read* {_filters('subpath', [guard_dir])})")
        rules.append(f"(allow file-write* {_filters('subpath', [Path(guard_dir) / 'trace'])})")
    rules.append(f"(deny process-exec {_filters('literal', BLOCKED_EXEC_LITERALS)} "
                 f"{_filters('subpath', BLOCKED_EXEC_SUBPATHS)})")
    rules += ['(deny network-outbound)',
              '(allow network-outbound (remote ip "localhost:*"))',
              '(allow network-outbound (literal "/private/var/run/mDNSResponder"))']
    return "(version 1)\n(allow default)\n" + "\n".join(rules) + "\n"


# ---- 2. egress allowlist proxy ------------------------------------------------------------------
class EgressProxy(socketserver.ThreadingTCPServer):
    """HTTP CONNECT proxy on 127.0.0.1 that tunnels only to allowlisted hosts on port 443.

    Every request is appended to ``log_path`` as JSON lines: {ts, host, port, allowed}.
    """
    daemon_threads = False
    allow_reuse_address = True

    def __init__(self, log_path: Path, allow: Iterable[str] = EGRESS_ALLOW, port: int = 0):
        self.allow = {parse_allow_entry(entry) for entry in allow}
        self.log_path = Path(log_path)
        self._log_lock = threading.Lock()
        self._clients_lock = threading.Lock()
        self._clients: set[socket.socket] = set()
        self._stopping = threading.Event()
        super().__init__(("127.0.0.1", port), _ProxyHandler)

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        connection, address = super().get_request()
        with self._clients_lock:
            self._clients.add(connection)
        return connection, address

    def shutdown_request(self, request: socket.socket) -> None:
        try:
            super().shutdown_request(request)
        finally:
            with self._clients_lock:
                self._clients.discard(request)

    @property
    def port(self) -> int:
        return self.server_address[1]

    def log(self, host: str, port: int, allowed: bool) -> None:
        with self._log_lock, self.log_path.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "host": host, "port": port, "allowed": allowed}) + "\n")

    def start(self) -> "EgressProxy":
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self._stopping.set()
        self.shutdown()
        with self._clients_lock:
            clients = tuple(self._clients)
        for connection in clients:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.server_close()


def parse_allow_entry(entry: str) -> tuple[str, int]:
    """``host`` (port 443) or ``host:port`` → (lowercase host, port)."""
    host, sep, port = entry.strip().rpartition(":")
    if sep and port.isdigit():
        return host.lower(), int(port)
    return entry.strip().lower(), 443


def host_allowed(host: str, port: int, allow: set[tuple[str, int]]) -> bool:
    """Exact (host, port) entries, plus "*.domain" entries that match any subdomain of domain (not domain itself)."""
    host = host.lower()
    for pattern, allowed_port in allow:
        if port != allowed_port:
            continue
        if pattern == host:
            return True
        if pattern.startswith("*.") and host.endswith(pattern[1:]) and host != pattern[2:]:
            return True
    return False


class _ProxyHandler(socketserver.StreamRequestHandler):
    def _read_headers(self) -> tuple[str, bool]:
        """Poll raw socket reads so stop works on Winsock without timed-out buffered readers."""
        deadline = time.monotonic() + 30
        first_line: bytes | None = None
        line = bytearray()
        total = 0
        complete = False
        previous_timeout = self.connection.gettimeout()
        self.connection.settimeout(.1)
        try:
            while time.monotonic() < deadline and total < 65536:
                if self.server._stopping.is_set() and first_line is not None:
                    break
                try:
                    byte = self.connection.recv(1)   # do not consume tunneled bytes after the headers
                except TimeoutError:
                    if self.server._stopping.is_set():
                        break
                    continue
                except OSError:
                    break
                if not byte:
                    break
                total += 1
                line.extend(byte)
                if len(line) > 4096:
                    break
                if byte == b"\n":
                    if first_line is None:
                        first_line = bytes(line).strip()
                    elif not line.strip():
                        complete = True
                        break
                    line.clear()
        finally:
            self.connection.settimeout(previous_timeout)
        return (first_line if first_line is not None else bytes(line)).decode("latin-1").strip(), complete

    def handle(self) -> None:
        line, complete = self._read_headers()
        if not line:       # an idle preconnection is not a CONNECT attempt
            return
        parts = line.split()
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            self.server.log(parts[1] if len(parts) > 1 else "?", 0, False)
            if self.server._stopping.is_set():
                return
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        host, _, port_s = parts[1].rpartition(":")
        port = int(port_s) if port_s.isdigit() else 0
        allowed = host_allowed(host, port, self.server.allow)
        self.server.log(host, port, allowed)
        if self.server._stopping.is_set():
            return
        if not allowed or not complete:
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        try:
            upstream = socket.create_connection((host, port), timeout=30)
        except OSError:
            self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        self.wfile.flush()
        _pipe(self.connection, upstream, self.server._stopping)


def _pipe(a: socket.socket, b: socket.socket, stopping: threading.Event | None = None) -> None:
    sockets = [a, b]
    idle_deadline = time.monotonic() + 300
    try:
        while stopping is None or not stopping.is_set():
            wait = max(0., idle_deadline - time.monotonic())
            readable, _, _ = select.select(sockets, [], [], min(wait, .1) if stopping is not None else wait)
            if not readable:
                if time.monotonic() >= idle_deadline:
                    return
                continue
            idle_deadline = time.monotonic() + 300
            for s in readable:
                if stopping is not None and stopping.is_set():
                    return
                data = s.recv(65536)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    except OSError:
        return
    finally:
        b.close()


# ---- 3. runtime guard installation + agent environment -----------------------------------------
def _npy_shape(f) -> tuple[int, ...]:
    """Shape from a .npy header (public numpy API, no array data read)."""
    from numpy.lib import format as npy

    major, _ = npy.read_magic(f)
    reader = npy.read_array_header_1_0 if major == 1 else npy.read_array_header_2_0
    return tuple(reader(f)[0])


def weight_shapes(public_dir: Path, exclude: Iterable[str] = ()) -> list[list[int]]:
    """2-D tensor shapes of every .npy / .npz / .safetensors file under ``public_dir`` (headers only), except files
    matching an ``exclude`` glob (weights the task explicitly allows the agent to evaluate)."""
    shapes: set[tuple[int, ...]] = set()
    patterns = list(exclude)
    for p in Path(public_dir).rglob("*"):
        rel = p.relative_to(public_dir).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern) for pattern in patterns):
            continue
        try:
            if p.suffix == ".safetensors":
                with p.open("rb") as f:
                    header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
                shapes |= {tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
            elif p.suffix == ".npz":
                with zipfile.ZipFile(p) as z:
                    for name in z.namelist():
                        with z.open(name) as f:
                            shapes.add(_npy_shape(f))
            elif p.suffix == ".npy":
                with p.open("rb") as f:
                    shapes.add(_npy_shape(f))
        except (OSError, ValueError, KeyError, AttributeError, zipfile.BadZipFile, struct.error):
            continue
    return sorted([list(s) for s in shapes if len(s) == 2])


FINGERPRINT_CHUNK = 65536


def file_fingerprint(path: Path | str, size: int | None = None) -> str:
    """sha256 over the size, the first and the last 64 KiB of a file: identifies an exact copy of a weight file under
    any name without hashing gigabytes. The guard computes the same value (``agent_guard._fingerprint``) for a file
    whose size equals a declared weight file's."""
    import hashlib

    size = os.path.getsize(path) if size is None else size
    h = hashlib.sha256(str(size).encode() + b"\0")
    with open(path, "rb") as f:
        h.update(f.read(FINGERPRINT_CHUNK))
        if size > FINGERPRINT_CHUNK:
            f.seek(max(FINGERPRINT_CHUNK, size - FINGERPRINT_CHUNK))
            h.update(f.read(FINGERPRINT_CHUNK))
    return h.hexdigest()


def _npz_data_ranges(path: Path) -> list[list[int]]:
    """Byte ranges of tensor DATA inside an .npz (a zip). Each member is a local file header (30 bytes + name + extra)
    followed by its payload; the central directory sits at the end. A stored ``.npy`` member's payload starts with its
    own npy header (names, shape, dtype: clean), then the array data, which is the range recorded. A compressed
    member's whole payload is recorded: its bytes are not usable as weights directly, but decompressing them is a raw
    read of tensor data. Zero-size arrays have no range. Local headers, npy headers and the central directory stay
    outside every range, so reading them is clean."""
    ranges = []
    with zipfile.ZipFile(path) as z, path.open("rb") as raw:
        for info in z.infolist():
            raw.seek(info.header_offset)
            local = raw.read(30)
            if len(local) < 30 or local[:4] != b"PK\x03\x04":
                raise ValueError(f"bad local header in {path.name}:{info.filename}")
            name_len, extra_len = struct.unpack("<HH", local[26:30])
            start = info.header_offset + 30 + name_len + extra_len
            end = start + info.compress_size
            if info.compress_type == zipfile.ZIP_STORED and info.filename.endswith(".npy"):
                with z.open(info) as member:
                    try:
                        _npy_shape(member)
                        start += member.tell()
                    except (ValueError, OSError):
                        pass          # not a readable npy header: the whole payload counts as data
            if end > start:
                ranges.append([start, end])
    return ranges


def data_ranges(path: Path) -> list[list[int]]:
    """``[[start, end), ...]`` byte ranges of tensor data in a weight file, so the guard can tell a header / index
    read (tensor names and shapes: legitimate static analysis) from a read into the weights. Exact for .safetensors
    (everything after the 8-byte length and the JSON header), .npy (after the parsed header) and .npz (per member, see
    :func:`_npz_data_ranges`). If a file cannot be parsed, the whole file is data (no size-based fallback that a small
    file could sit below)."""
    size = path.stat().st_size
    try:
        if path.suffix == ".safetensors":
            with path.open("rb") as f:
                start = 8 + struct.unpack("<Q", f.read(8))[0]
        elif path.suffix == ".npy":
            with path.open("rb") as f:
                _npy_shape(f)
                start = f.tell()
        elif path.suffix == ".npz":
            return _npz_data_ranges(path)
        else:
            start = 0
    except (OSError, ValueError, KeyError, struct.error, zipfile.BadZipFile):
        start = 0
    return [[start, size]] if size > start else []


def weight_files(public_dir: Path, exclude: Iterable[str] = (), visible_dirs: Iterable[Path] = ()) -> list[dict]:
    """One entry per weight data file under ``public_dir`` (same walk and ``exclude`` globs as :func:`weight_shapes`):
    ``path`` (resolved), ``paths`` (that file's location under ``public_dir`` and under every ``visible_dirs`` entry,
    e.g. the agent workdir it is copied into, resolved), ``size``, ``fingerprint`` (:func:`file_fingerprint`),
    ``ranges`` (tensor-data byte ranges) and the legacy ``data_offset``. The guard records raw reads that intersect a
    range of the file opened, located by path, by file identity, or by size + fingerprint (an exact copy)."""
    patterns = list(exclude)
    out = []
    for p in sorted(Path(public_dir).rglob("*")):
        if p.suffix not in (".safetensors", ".npz", ".npy") or not p.is_file():
            continue
        rel = p.relative_to(public_dir).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern) for pattern in patterns):
            continue
        paths = [os.path.realpath(p)] + [os.path.realpath(Path(d) / rel) for d in visible_dirs]
        ranges = data_ranges(p)
        size = p.stat().st_size
        out.append({"path": os.path.realpath(p), "paths": list(dict.fromkeys(paths)), "size": size,
                    "fingerprint": file_fingerprint(p, size), "ranges": ranges,
                    "data_offset": ranges[0][0] if ranges else size})
    return out


def install_guard(guard_root: Path, public_dir: Path, meter_exclude: Iterable[str] = (),
                  flop_cap: float | None = None, visible_dirs: Iterable[Path] = ()) -> Path:
    """Write ``guard_root/lib/sitecustomize.py`` (read-only for the agent) and the weight-shape manifest;
    ``guard_root/trace`` stays writable for per-process traces. Returns ``guard_root``.

    ``visible_dirs``: other directories where the agent sees copies of ``public_dir`` (the runner passes the agent
    workdir), so the raw-read rule matches the files the agent actually opens.

    ``flop_cap`` (the task's ``budget.flop_cap``) is written to ``lib/flop_cap.json``; the guard takes it from there,
    so exporting a larger ``WB_FLOP_CAP`` inside the sandbox cannot raise it (the variable can only lower it)."""
    lib = guard_root / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    (guard_root / "trace").mkdir(exist_ok=True)
    shutil.copy(GUARD_SOURCE, lib / "sitecustomize.py")
    (lib / "shapes.json").write_text(json.dumps(weight_shapes(public_dir, meter_exclude)))
    (lib / "weight_files.json").write_text(json.dumps(weight_files(public_dir, meter_exclude, visible_dirs)))
    if flop_cap is not None:
        (lib / "flop_cap.json").write_text(json.dumps(float(flop_cap)))
    return guard_root


GUARD_PTH = ("import os, sys; _p = os.environ.get('WB_GUARD_LIB'); "
             "_p and _p not in sys.path and sys.path.insert(0, _p)\n")


def ensure_guard_pth(agent_venv: Path) -> None:
    """Put the guard on sys.path via a .pth file in the agent venv's site-packages.

    `python -I` / `-E` ignore PYTHONPATH (agents use `-I` routinely, e.g. for untrusted files), which would silently
    skip a PYTHONPATH-only guard. site-packages .pth files are still processed
    under -I/-E, and site's later `import sitecustomize` then finds the guard. Only `-S` (no site at all) bypasses it;
    the audit flags that. No-op outside a run (WB_GUARD_LIB unset). The sandbox makes the venv read-only to agents.
    """
    for sp in Path(agent_venv).glob("lib/python3*/site-packages"):
        pth = sp / "zz_wb_guard.pth"
        if not pth.exists() or pth.read_text() != GUARD_PTH:
            pth.write_text(GUARD_PTH)


def agent_env(base: dict[str, str], agent_venv: Path, guard_root: Path, proxy_port: int,
              flop_cap: float | None = None, wb_url: str | None = None, wb_token: str | None = None,
              tmpdir: Path | None = None) -> dict[str, str]:
    env = dict(base)
    env.pop("VIRTUAL_ENV", None)
    for key in ("PYTHONHOME", "PYTHONSTARTUP", "PYTHONNOUSERSITE", "PYTHONSAFEPATH", "PYTHONINSPECT"):
        env.pop(key, None)
    env["PATH"] = f"{agent_venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(guard_root / "lib")
    env["WB_GUARD_LIB"] = str(guard_root / "lib")    # read by the .pth hook, which also works under `python -I/-E`
    ensure_guard_pth(agent_venv)
    env["WB_TRACE_DIR"] = str(guard_root / "trace")
    env["WB_SHAPES"] = str(guard_root / "lib" / "shapes.json")
    if wb_url:
        env["WB_URL"] = wb_url               # the guard reports runtime events to the Weight API (tamper-evident)
    if wb_token:
        env["WB_TOKEN"] = wb_token
    if tmpdir is not None:                   # temp files stay inside the agent's own workdir
        for key in ("TMPDIR", "TMP", "TEMP"):
            env[key] = str(tmpdir)
    if flop_cap is not None:
        env["WB_FLOP_CAP"] = repr(float(flop_cap))
    proxy = f"http://127.0.0.1:{proxy_port}"
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        env[key] = proxy
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env.update(CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_TELEMETRY="1",
               DISABLE_ERROR_REPORTING="1", DISABLE_AUTOUPDATER="1")
    return env


# ---- self-test (macOS) --------------------------------------------------------------------------
_SELFTEST_CHILD = r'''
import json, os, socket, sys
res = {}
def attempt(name, fn):
    try:
        fn(); res[name] = "allowed"
    except Exception as e:
        res[name] = "blocked: " + type(e).__name__
attempt("read_repo_file", lambda: open(sys.argv[1]).read(10))
attempt("read_home_file", lambda: open(sys.argv[2]).read(10))
attempt("read_private_tmp_alias", lambda: open(sys.argv[3]).read(10))
attempt("write_outside_workdir", lambda: open(sys.argv[4], "w").write("x"))
attempt("write_workdir", lambda: open("ok.txt", "w").write("x"))
def run(cmd):
    import subprocess
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    if r.returncode != 0:
        raise PermissionError(r.stderr[-200:])
attempt("exec_compiler", lambda: run(["/usr/bin/cc", "--version"]))
attempt("exec_perl", lambda: run(["/usr/bin/perl", "-e", "1"]))
if sys.argv[5]:
    attempt("agent_cli_starts", lambda: run([sys.argv[5], "--version"]))
else:
    res["agent_cli_starts"] = "skipped (agent CLI not installed)"
attempt("direct_internet", lambda: socket.create_connection(("1.1.1.1", 443), timeout=5))
def isolated_mode_guard():             # `python -I` ignores PYTHONPATH; the guard must still load
    import subprocess
    r = subprocess.run([sys.executable, "-I", "-c", "import sys; print('sitecustomize' in sys.modules and "
                        "hasattr(sys.modules['sitecustomize'], 'TRACE'))"], capture_output=True, text=True, timeout=60)
    if r.stdout.strip() != "True":
        raise PermissionError("guard not loaded under python -I")
if sys.platform != "darwin":
    attempt("guard_under_python_I", isolated_mode_guard)
else:
    res["guard_under_python_I"] = "skipped (macOS: guard via PYTHONPATH only; -I is flagged by the audit)"
def raw_connect():                    # _socket bypasses the Python guard: this checks the OS layer alone
    import _socket
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(("1.1.1.1", 443))
attempt("direct_internet_os_layer", raw_connect)
def via_proxy(host):
    s = socket.create_connection(("127.0.0.1", int(os.environ["HTTPS_PROXY"].rsplit(":", 1)[1])), timeout=10)
    s.sendall(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n".encode())
    line = s.recv(64).decode().split("\r\n")[0]
    if " 200 " not in line + " ":
        raise PermissionError(line)
attempt("proxy_allowlisted_api", lambda: via_proxy("api.anthropic.com"))
attempt("proxy_other_host", lambda: via_proxy("example.com"))
import importlib.util
if importlib.util.find_spec("transformers"):
    attempt("import_transformers", lambda: __import__("transformers"))
else:
    res["import_transformers"] = "skipped (not installed in the agent venv)"
def backward():
    import torch
    x = torch.ones(2, requires_grad=True)
    (x * 2).sum().backward()
if importlib.util.find_spec("torch"):
    attempt("torch_backward", backward)
else:
    res["torch_backward"] = "skipped (torch not installed in the agent venv)"
print(json.dumps(res))
'''

EXPECTED = {"read_repo_file": "blocked", "read_home_file": "blocked", "read_private_tmp_alias": "blocked",
            "write_outside_workdir": "blocked", "write_workdir": "allowed", "exec_compiler": "blocked",
            "exec_perl": "blocked", "agent_cli_starts": "allowed", "direct_internet": "blocked",
            "direct_internet_os_layer": "blocked", "guard_under_python_I": "allowed",
            "proxy_allowlisted_api": "allowed", "proxy_other_host": "blocked", "import_transformers": "blocked",
            "torch_backward": "blocked"}


def selftest(agent_venv: Path, repo_file: Path, agent_cli: str = "claude") -> dict[str, str]:
    """End-to-end check of the OS sandbox on this host (macOS or Linux backend) with a toy child process."""
    import subprocess

    from weightbench.harness import backends

    backend = backends.detect()
    cli = agent_cli if shutil.which(agent_cli) else ""
    home_probe = Path.home() / ".wb_selftest_probe"
    with tempfile.TemporaryDirectory(prefix="wb-selftest-", dir="/tmp") as tmp:
        tmp = Path(tmp)
        work, guard, other = tmp / "work", tmp / "guard", tmp / "other_run"
        work.mkdir()
        other.mkdir()
        (other / "secret.txt").write_text("other run")
        home_probe.write_text("home")
        install_guard(guard, work)
        proxy = EgressProxy(tmp / "egress.jsonl").start()
        spec = backends.SandboxSpec(workdir=work, guard_root=guard, agent_venv=agent_venv,
                                    deny_read=[repo_file.parent, other], ports={"proxy": proxy.port},
                                    agent_executable=cli or None)
        backend.start(spec)
        try:
            (work / "child.py").write_text(_SELFTEST_CHILD)
            env = agent_env(os.environ.copy(), agent_venv, guard, proxy.port, tmpdir=work)
            private_alias = "/private" + str(other / "secret.txt")     # same file via the /private/tmp spelling
            child = [str(agent_venv / "bin" / "python"), str(work / "child.py"), str(repo_file), str(home_probe),
                     private_alias, str(tmp / "escaped.txt"), cli]
            out = subprocess.run(backend.wrap(spec, child), cwd=work, env=env, capture_output=True, text=True,
                                 timeout=180)
            escaped_to_host = (tmp / "escaped.txt").exists()
        finally:
            backend.stop()
            proxy.stop()
            home_probe.unlink(missing_ok=True)
    results = json.loads(out.stdout.strip().splitlines()[-1]) if out.stdout.strip() else {"error": out.stderr[-2000:]}
    if results.get("write_outside_workdir") == "allowed" and not escaped_to_host:
        # linux: /tmp inside the sandbox is a private tmpfs; what counts is that the host file was not written
        results["write_outside_workdir"] = "blocked: write landed in the sandbox's private tmpfs, host unchanged"
    return results


def main(argv: list[str]) -> None:
    if argv[:1] != ["selftest"]:
        print(__doc__)
        return
    from weightbench.platform_compat import DEFAULT_AGENT_VENV

    venv = Path(os.environ.get("WB_AGENT_VENV", DEFAULT_AGENT_VENV)).expanduser()
    res = selftest(venv, Path(__file__).resolve().parents[2] / "README.md")
    ok = True
    for name, want in EXPECTED.items():
        got = res.get(name, "missing")
        good = got.startswith(want) or got.startswith("skipped")
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name:24s} expected {want:8s} got {got}")
    if "error" in res:
        print(res["error"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
