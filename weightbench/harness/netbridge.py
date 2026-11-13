"""Loopback ↔ unix-socket bridge for the Linux sandbox backend (standard library only).

Inside a bubblewrap sandbox with its own network namespace only the loopback interface exists, so nothing can
reach the host or the internet. The harness gives the agent exactly two ways out, both as unix sockets
bind-mounted into the sandbox:

- the Weight API (metered probes, submissions), and
- the egress allowlist proxy (the agent's own LLM API, e.g. api.anthropic.com:443 only).

Host side (``UnixToTcp``, run by the harness): a unix socket that forwards to a host 127.0.0.1 port.
Sandbox side (``python3 -I netbridge.py --map PORT=SOCKET ... -- CMD``): listens on 127.0.0.1:PORT inside the
sandbox, forwards each connection to SOCKET, then runs CMD and exits with its return code. The agent sees the
same ``http://127.0.0.1:PORT`` URLs as on macOS.
"""
from __future__ import annotations

import os
import select
import socket
import socketserver
import subprocess
import sys
import threading


def pipe(a: socket.socket, b: socket.socket, idle_s: float = 600.0) -> None:
    """Copy bytes both ways until either side closes or the connection is idle for ``idle_s``."""
    try:
        while True:
            readable, _, _ = select.select([a, b], [], [], idle_s)
            if not readable:
                return
            for s in readable:
                data = s.recv(65536)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    except OSError:
        return
    finally:
        a.close()
        b.close()


# Unix sockets do not exist on Windows; the module must still import there (the bridge is only used on Linux).
_UNIX_SERVER = getattr(socketserver, "UnixStreamServer", socketserver.TCPServer)


class UnixToTcp(socketserver.ThreadingMixIn, _UNIX_SERVER):
    """Host side: accept on a unix socket, forward to 127.0.0.1:port (Linux / macOS hosts only)."""
    daemon_threads = True

    def __init__(self, socket_path: str, port: int):
        self.port = port
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        super().__init__(socket_path, _UnixHandler)

    def start(self) -> "UnixToTcp":
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        try:
            os.unlink(self.server_address)
        except OSError:
            pass


class _UnixHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            upstream = socket.create_connection(("127.0.0.1", self.server.port), timeout=30)
        except OSError:
            return
        pipe(self.request, upstream)


class TcpToUnix(socketserver.ThreadingTCPServer):
    """Sandbox side: accept on 127.0.0.1:port, forward to a unix socket."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int, socket_path: str):
        self.socket_path = socket_path
        super().__init__(("127.0.0.1", port), _TcpHandler)


class _TcpHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            upstream.connect(self.server.socket_path)
        except OSError:
            self.request.close()
            return
        pipe(self.request, upstream)


def parse_args(argv: list[str]) -> tuple[list[tuple[int, str]], list[str], dict[str, int]]:
    """``--map PORT=SOCKET`` (repeatable) ``[--rlimit-cpu SECONDS] [--rlimit-fsize BYTES] [--report-fd FD] -- CMD``.

    ``--report-fd`` (returned as ``limits["report_fd"]``) is a pipe inherited from the runner: the bridge then acts as
    the sandbox's init (see :func:`run_as_init`) and writes the whole-tree CPU report to it."""
    if "--" not in argv:
        raise SystemExit("usage: netbridge.py --map PORT=SOCKET [...] [--rlimit-cpu S] [--rlimit-fsize B] "
                         "[--report-fd FD] -- CMD ...")
    split = argv.index("--")
    head, cmd = argv[:split], argv[split + 1:]
    maps, limits = [], {}
    for i, item in enumerate(head):
        if item == "--map":
            port, _, path = head[i + 1].partition("=")
            maps.append((int(port), path))
        elif item == "--rlimit-cpu":
            limits["cpu"] = int(head[i + 1])
        elif item == "--rlimit-fsize":
            limits["fsize"] = int(head[i + 1])
        elif item == "--report-fd":
            limits["report_fd"] = int(head[i + 1])
    if not cmd:
        raise SystemExit("netbridge.py: no command given after --")
    return maps, cmd, limits


def apply_limits(limits: dict[str, int]) -> None:
    """Hard per-process limits inherited by the agent and every process it starts: CPU seconds (bounds unmetered
    computation) and largest file size (no filling the disk). Address space is left alone: Node-based agent CLIs
    reserve large virtual memory and would crash under RLIMIT_AS."""
    import resource

    if "cpu" in limits:
        resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu"], limits["cpu"]))
    if "fsize" in limits:
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits["fsize"], limits["fsize"]))


# ---- sandbox init: whole-tree CPU accounting (Linux) ---------------------------------------------------------------
# Under bubblewrap with a PID namespace the agent's descendants are reaped inside the namespace, so their CPU never
# reaches the runner's RUSAGE_CHILDREN (a 1 s busy loop reads as a few ms). With ``--report-fd`` this bridge
# runs as the namespace's PID 1 (``bwrap --as-pid-1``, set by ``backends.LinuxBwrapBackend.wrap``): every orphan is
# reparented to it, it reaps every descendant (killing any still alive once the agent command exits), and then its own
# RUSAGE_CHILDREN is the CPU of the whole agent tree. The report goes to a pipe inherited from the runner: the agent
# command never holds that fd (close_fds), and this process is made non-dumpable, so /proc/1/fd and ptrace are closed
# to the agent. PID 1 of a namespace ignores signals it has no handler for (SIGKILL / SIGTERM from inside included).
PR_SET_DUMPABLE = 4
PR_SET_CHILD_SUBREAPER = 36


def _prctl(option: int, value: int) -> bool:
    try:
        import ctypes

        return ctypes.CDLL(None, use_errno=True).prctl(option, value, 0, 0, 0) == 0
    except Exception:  # noqa: BLE001 - not Linux / no libc: the report then says so
        return False


def _proc_self_cpu(pid: int) -> float | None:
    """User + system CPU seconds of ``pid`` itself (not its children), from /proc/<pid>/stat. Readable for a zombie,
    so it is taken after the process exited and before it is reaped."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _wait_root(pid: int) -> tuple[int, float | None]:
    """Wait for the agent command, reaping orphans meanwhile; its own CPU is read before it is reaped."""
    while True:
        try:
            info = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOWAIT)
        except ChildProcessError:
            return -1, None
        except InterruptedError:
            continue
        if info is None:
            continue
        if info.si_pid == pid:
            own = _proc_self_cpu(pid)
            _, status = os.waitpid(pid, 0)
            return os.waitstatus_to_exitcode(status), own
        try:
            os.waitpid(info.si_pid, 0)
        except ChildProcessError:
            pass


def _children_of(parent: int) -> list[int]:
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                if int(f.read().rsplit(")", 1)[1].split()[1]) == parent:
                    out.append(int(name))
        except (OSError, ValueError, IndexError):
            continue
    return out


def _descendants(root: int) -> list[int]:
    found, pending = [], [root]
    while pending:
        kids = _children_of(pending.pop())
        found += kids
        pending += kids
    return found


def _reap_all() -> int:
    """Kill every remaining descendant (daemons, background jobs) and reap it, so its CPU is counted. Only this
    process's own descendants are touched (it is their subreaper), never anything else of the user."""
    import signal

    killed: set[int] = set()
    for _ in range(100):
        for pid in _descendants(os.getpid()):
            try:
                os.kill(pid, signal.SIGKILL)
                killed.add(pid)
            except OSError:
                pass
        reaped_any = False
        while True:
            try:
                pid, _ = os.waitpid(-1, 0)
                reaped_any = True
            except ChildProcessError:
                break
            except InterruptedError:
                continue
        if not reaped_any and not _descendants(os.getpid()):
            break
    return len(killed)


def run_as_init(cmd: list[str], limits: dict[str, int], report_fd: int) -> int:
    """Run ``cmd`` as the only child, reap the whole tree, write the CPU report to ``report_fd``; return its code."""
    import json
    import resource
    import signal

    os.set_inheritable(report_fd, False)
    dumpable_off = _prctl(PR_SET_DUMPABLE, 0)           # /proc/<self>/fd, environ, ptrace: closed to the agent
    subreaper = os.getpid() == 1 or _prctl(PR_SET_CHILD_SUBREAPER, 1)
    signal.signal(signal.SIGINT, lambda *_: None)       # PID 1 would otherwise die on an agent's SIGINT; exec resets it
    child = subprocess.Popen(cmd, preexec_fn=(lambda: apply_limits(limits)) if limits else None, close_fds=True)
    code, root_cpu = _wait_root(child.pid)
    child.returncode = code                             # reaped here; keep Popen from waiting again
    killed = _reap_all()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {"version": 1, "tree_cpu_user": usage.ru_utime, "tree_cpu_sys": usage.ru_stime,
              "tree_cpu_seconds": round(usage.ru_utime + usage.ru_stime, 6), "tree_max_rss_kb": float(usage.ru_maxrss),
              "root_pid": child.pid, "root_cpu_seconds": None if root_cpu is None else round(root_cpu, 6),
              "root_returncode": code, "leftovers_killed": killed, "pid1": os.getpid() == 1,
              "subreaper": bool(subreaper), "non_dumpable": dumpable_off}
    try:
        os.write(report_fd, (json.dumps(report) + "\n").encode())
    except OSError:
        pass
    finally:
        try:
            os.close(report_fd)
        except OSError:
            pass
    return code


def main(argv: list[str]) -> int:
    maps, cmd, limits = parse_args(argv)
    report_fd = limits.pop("report_fd", None)
    for port, path in maps:
        server = TcpToUnix(port, path)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    if report_fd is not None:
        return run_as_init(cmd, limits, report_fd)
    return subprocess.call(cmd, preexec_fn=(lambda: apply_limits(limits)) if limits else None)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
