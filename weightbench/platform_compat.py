"""Cross-platform helpers (macOS, Linux, Windows) for the few OS-specific operations the harness and builders need."""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import IO, Iterator

DEFAULT_AGENT_VENV = Path.home() / 'wb-agent-venv'

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def lock_exclusive(f: IO) -> None:
    """Block until this process holds an exclusive lock on the open file ``f``."""
    if sys.platform == "win32":
        f.seek(0)
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)    # retries ~10 s internally, then raises
                return
            except OSError:
                continue
    fcntl.flock(f, fcntl.LOCK_EX)


def unlock(f: IO) -> None:
    if sys.platform == "win32":
        f.seek(0)
        with contextlib.suppress(OSError):
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive cross-process lock on ``path`` (created if missing) for the duration of the block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        lock_exclusive(f)
        try:
            yield
        finally:
            unlock(f)


def clone_file(src: Path, dst: Path) -> None:
    """Copy-on-write clone where the filesystem supports it (APFS via ``cp -c``), else a plain copy."""
    src = Path(src).resolve()
    if sys.platform == "darwin" and subprocess.run(["cp", "-c", str(src), str(dst)],
                                                   capture_output=True).returncode == 0:
        return
    shutil.copyfile(src, dst)


def to_wsl_path(path: Path) -> str:
    """C:\\Users\\x\\repo -> /mnt/c/Users/x/repo (for running a command inside WSL from Windows)."""
    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{rest}" if drive else resolved.as_posix()


def wsl_available() -> bool:
    return sys.platform == "win32" and shutil.which("wsl.exe") is not None and os.environ.get("WB_NO_WSL") != "1"
