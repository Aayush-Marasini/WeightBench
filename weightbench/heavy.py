"""Run heavy (model-executing) build and evaluation steps.

Track code calls ``run_heavy(module, args, art_dir=..., push=[...], out_name=...)`` where ``module`` is a
``python -m`` entry point that reads/writes under ``art_dir`` and writes JSON to ``--out``. Steps run as a
subprocess on the machine invoking the benchmark (``WB_HEAVY=local``, the default and the only built-in mode).

To offload these steps to your own cluster, run the whole benchmark on a compute node instead (the harness is
self-contained), or replace ``run_heavy`` / ``pull_dir`` with a backend that runs the same command remotely and
leaves the same files in ``art_dir``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _mode() -> str:
    mode = os.environ.get("WB_HEAVY", "local")
    if mode != "local":
        raise ValueError(f"WB_HEAVY={mode!r}: only 'local' is built in (see weightbench/heavy.py)")
    return mode


def run_heavy(module: str, args: list[str], art_dir: Path, push: list[Path] = (), out_name: str = "out.json",
              timeout: int = 4 * 3600, sync_art: bool = True, gpu: bool | None = None) -> dict:
    """Run ``python -m module args... --out <art_dir>/<out_name>``; return the parsed JSON.

    ``push``, ``sync_art`` and ``gpu`` are accepted for backends that execute elsewhere; locally the files are
    already in place and the step uses whatever device it detects.
    """
    _mode()
    art_dir = Path(art_dir).resolve()
    out_local = art_dir / out_name
    subprocess.run([sys.executable, "-m", module, *args, "--out", str(out_local)], check=True, timeout=timeout,
                   cwd=ROOT)
    return json.loads(out_local.read_text())


def pull_dir(art_dir: Path, sub: str) -> None:
    """Make ``art_dir/sub`` available locally. Local steps write it in place, so this only ensures it exists."""
    _mode()
    (Path(art_dir).resolve() / sub).mkdir(parents=True, exist_ok=True)
