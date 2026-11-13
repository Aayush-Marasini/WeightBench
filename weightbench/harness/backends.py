"""OS sandbox backends for benchmarked agent runs (SPEC §1 *Enforcement*, layers 1–2).

    macos   sandbox-exec profile (``sandbox.macos_profile``): deny-by-default writes, closed $HOME, repo and
            other runs unreadable, compilers not executable, outbound network only to localhost.
    linux   bubblewrap: new user / pid / ipc / uts / cgroup / **network** namespaces. System directories are
            mounted read-only, $HOME is an empty tmpfs (plus the agent CLI's own state), /tmp is a fresh tmpfs
            holding only this run's workdir and guard. The repo, artifacts, ledgers and other runs are never
            mounted, compilers and foreign interpreters are masked. The only ways out are two unix sockets: the
            Weight API and the egress allowlist proxy (``netbridge``).
    windows not supported natively: run WeightBench inside WSL2, which uses the linux backend.

``WB_SANDBOX=macos|linux`` overrides auto-detection.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from weightbench.harness import netbridge, sandbox

NETBRIDGE_SOURCE = Path(netbridge.__file__).resolve()
SYSTEM_DIRS = ("/usr", "/etc", "/opt", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
LINUX_BLOCKED_EXEC_GLOBS = ("/usr/bin/cc", "/usr/bin/c++", "/usr/bin/gcc*", "/usr/bin/g++*", "/usr/bin/clang*",
                            "/usr/bin/*-linux-gnu-gcc*", "/usr/bin/*-linux-gnu-g++*", "/usr/bin/tcc",
                            # not ld / as: /usr/bin/ld.so is the dynamic loader on some distros, and a linker is
                            # useless without the compilers masked here
                            "/usr/bin/perl", "/usr/bin/perl5*", "/usr/bin/ruby*",
                            "/usr/bin/php*", "/usr/bin/java", "/usr/bin/javac", "/usr/bin/swift*", "/usr/bin/julia",
                            "/usr/bin/Rscript", "/usr/bin/R", "/usr/bin/gdb", "/usr/bin/lldb*", "/usr/bin/tclsh*",
                            "/usr/bin/lua*", "/usr/bin/go", "/usr/bin/rustc", "/usr/bin/cargo")


DEFAULT_FSIZE_BYTES = 8 * 2 ** 30


@dataclass
class SandboxSpec:
    """Everything a backend needs to confine one agent run."""
    workdir: Path
    guard_root: Path
    agent_venv: Path
    deny_read: list[Path]
    ports: dict[str, int]                        # host loopback ports the agent may use, e.g. {"api": .., "proxy": ..}
    home_state: tuple[str, ...] = ()             # home-relative paths of the agent CLI (auth, settings)
    home: Path = field(default_factory=Path.home)
    agent_executable: str | None = None          # argv[0] of the agent command, to make its install visible
    cpu_seconds: int | None = None               # per-process CPU-time limit (Linux backend; the task's wall clock)
    fsize_bytes: int | None = DEFAULT_FSIZE_BYTES  # largest file any agent process may write (Linux backend)
    # Linux backend: write end of a runner-owned pipe. The bridge then runs as the sandbox's PID 1, reaps the whole
    # agent tree and writes its CPU report there (``netbridge.run_as_init``); the agent never holds this fd.
    report_fd: int | None = None


class MacOSBackend:
    name = "macos"

    def start(self, spec: SandboxSpec) -> None:
        state = [spec.home / p for p in spec.home_state]
        profile = sandbox.macos_profile(spec.workdir, spec.deny_read, spec.guard_root, home=spec.home,
                                        extra_read=[spec.agent_venv, *state], extra_write=state)
        (spec.guard_root / "profile.sb").write_text(profile)

    def wrap(self, spec: SandboxSpec, cmd: list[str]) -> list[str]:
        return ["sandbox-exec", "-f", str(spec.guard_root / "profile.sb"), *cmd]

    def stop(self) -> None:
        pass


class LinuxBwrapBackend:
    name = "linux"

    def __init__(self) -> None:
        self._bridges: list[netbridge.UnixToTcp] = []

    def _socket_dir(self, spec: SandboxSpec) -> Path:
        return spec.guard_root / "sock"

    def start(self, spec: SandboxSpec) -> None:
        sock_dir = self._socket_dir(spec)
        sock_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(NETBRIDGE_SOURCE, spec.guard_root / "lib" / "wb_netbridge.py")
        self._bridges = [netbridge.UnixToTcp(str(sock_dir / f"{name}.sock"), port).start()
                         for name, port in spec.ports.items()]

    def stop(self) -> None:
        for bridge in self._bridges:
            bridge.stop()
        self._bridges = []

    def wrap(self, spec: SandboxSpec, cmd: list[str]) -> list[str]:
        python = str(spec.agent_venv / "bin" / "python")
        maps = [arg for name, port in spec.ports.items()
                for arg in ("--map", f"{port}={self._socket_dir(spec) / f'{name}.sock'}")]
        limits = []
        if spec.cpu_seconds:
            limits += ["--rlimit-cpu", str(int(spec.cpu_seconds))]
        if spec.fsize_bytes:
            limits += ["--rlimit-fsize", str(int(spec.fsize_bytes))]
        init = []
        if spec.report_fd is not None:
            limits += ["--report-fd", str(int(spec.report_fd))]
            init = ["--as-pid-1"]       # the bridge (harness code, read-only lib) is the namespace's init and subreaper
        # -S: no site, so the agent venv's guard .pth hook does not load the runtime guard into the bridge. Otherwise
        # the bridge (harness code, relaying every Weight API call) would write a guard trace and its CPU would count as
        # the agent's guarded CPU (most of it in a trace-heavy run). The agent command still gets the guard.
        bridge = [python, "-I", "-S", str(spec.guard_root / "lib" / "wb_netbridge.py"), *maps, *limits, "--"]
        return bwrap_args(spec) + init + ["--", *bridge, *cmd]


def _home_child(path: Path, home: Path) -> Path | None:
    """The top-level directory under $HOME that contains ``path`` (e.g. ~/.local for ~/.local/bin/claude)."""
    try:
        rel = path.resolve().relative_to(home.resolve())
    except ValueError:
        return None
    return home / rel.parts[0] if rel.parts else None


def _venv_base(agent_venv: Path) -> Path | None:
    """Install prefix of the interpreter a venv points to (pyvenv.cfg ``home`` is its bin directory)."""
    cfg = agent_venv / "pyvenv.cfg"
    if not cfg.exists():
        return None
    for line in cfg.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "home":
            base = Path(value.strip()).parent
            # uv links e.g. cpython-3.12-linux-x86_64-gnu -> cpython-3.12.13-linux-x86_64-gnu; bwrap cannot mount onto
            # a symlink path, so mount the directory holding both the link and its target.
            return base.parent if base.is_symlink() else base
    return None


def _visible_roots(spec: SandboxSpec) -> list[Path]:
    return [Path(d) for d in SYSTEM_DIRS if os.path.isdir(d) and not os.path.islink(d)]


def bwrap_args(spec: SandboxSpec) -> list[str]:
    """bubblewrap argument list (before ``--``). Later mounts shadow earlier ones."""
    args = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--proc", "/proc", "--dev", "/dev"]
    for d in SYSTEM_DIRS:
        if os.path.islink(d):
            args += ["--symlink", os.readlink(d), d]
        elif os.path.isdir(d):
            args += ["--ro-bind", d, d]
    args += ["--tmpfs", "/tmp", "--tmpfs", str(spec.home), "--setenv", "HOME", str(spec.home)]
    readonly = {spec.agent_venv}
    for extra in (_venv_base(spec.agent_venv), _home_child(Path(shutil.which(spec.agent_executable) or "/"), spec.home)
                  if spec.agent_executable else None):
        if extra is not None and extra.exists():
            readonly.add(extra)
    for path in sorted(readonly):
        args += ["--ro-bind", str(path), str(path)]
    for rel in spec.home_state:
        path = spec.home / rel
        if path.exists():
            args += ["--bind", str(path), str(path)]
    # `python -I` / `-E` are covered by the guard's .pth hook in the agent venv (sandbox.ensure_guard_pth), which works
    # on every backend; only `-S` skips it, and the transcript audit flags that.
    args += ["--ro-bind", str(spec.guard_root / "lib"), str(spec.guard_root / "lib"),
             "--bind", str(spec.guard_root / "trace"), str(spec.guard_root / "trace"),
             "--bind", str(spec.guard_root / "sock"), str(spec.guard_root / "sock"),
             "--bind", str(spec.workdir), str(spec.workdir), "--chdir", str(spec.workdir)]
    # Anything private that a read-only mount happens to expose (e.g. a repo under /opt) is shadowed.
    roots = _visible_roots(spec) + sorted(readonly)
    for path in spec.deny_read:
        if path.exists() and any(path.resolve() == r or r in path.resolve().parents for r in roots):
            args += ["--tmpfs", str(path)] if path.is_dir() else ["--ro-bind", "/dev/null", str(path)]
    # Mask the real files, not the symlinks: bwrap cannot bind over a symlink whose chain leaves the mounted tree
    # (Ubuntu: /usr/bin/cc -> /etc/alternatives/cc -> gcc gives "Can't create file at /usr/bin/cc"). Masking the
    # resolved binary disables every link to it; a link whose target does not exist is already not executable.
    masked: set[str] = set()
    for pattern in LINUX_BLOCKED_EXEC_GLOBS:
        for binary in sorted(glob.glob(pattern)):
            real = os.path.realpath(binary)
            if os.path.isfile(real) and real not in masked:
                masked.add(real)
                args += ["--ro-bind", "/dev/null", real]
    return args


def detect() -> MacOSBackend | LinuxBwrapBackend:
    choice = os.environ.get("WB_SANDBOX") or {"darwin": "macos", "linux": "linux"}.get(sys.platform, sys.platform)
    if choice == "macos":
        return MacOSBackend()
    if choice == "linux":
        if shutil.which("bwrap") is None:
            raise RuntimeError("the linux sandbox needs bubblewrap: `sudo apt install bubblewrap` "
                               "(Debian/Ubuntu) or `sudo dnf install bubblewrap` (Fedora)")
        return LinuxBwrapBackend()
    if choice in ("win32", "cygwin", "windows"):
        raise RuntimeError("agent runs on Windows go through WSL2: install Ubuntu (`wsl --install`), clone the repo "
                           "inside WSL and run `bash scripts/quickstart.sh` there (linux sandbox backend)")
    raise RuntimeError(f"no sandbox backend for platform {choice!r}")
