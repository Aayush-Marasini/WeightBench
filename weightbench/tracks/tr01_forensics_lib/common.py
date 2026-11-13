"""Small shared utilities: string similarity, report parsing, locking, threads."""
from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

N_THREADS = 3  # conservative default for small machines


def set_threads() -> None:
    import os
    import platform

    import torch

    n = N_THREADS if platform.system() == "Darwin" else min(8, os.cpu_count() or 1)
    torch.set_num_threads(int(os.environ.get("WB_THREADS", n)))


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    """Edit distance between two sequences (characters or tokens)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_similarity(pred: list[str] | str, true: list[str] | str) -> float:
    """1 - Levenshtein / max(len). 1.0 iff equal; 0.0 for empty prediction vs non-empty truth."""
    n = max(len(pred), len(true))
    if n == 0:
        return 1.0
    return max(0.0, 1.0 - levenshtein(pred, true) / n)


def normalize_text(s: Any) -> str:
    """Lowercase, keep [a-z0-9] and single spaces. Robust to punctuation/quoting differences."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_code(s: Any) -> str:
    """For trigger strings: lowercase alnum only, no spaces."""
    return normalize_text(s).replace(" ", "")


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        from weightbench.platform_compat import lock_exclusive, unlock

        lock_exclusive(f)
        try:
            yield
        finally:
            unlock(f)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
