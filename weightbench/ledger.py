"""Append-only, hash-chained JSONL ledger (SPEC §1). No update or delete APIs."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64


def _digest(prev: str, body: dict[str, Any]) -> str:
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + blob).encode()).hexdigest()


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._head = GENESIS
        self._n = 0
        for e in self.entries():
            self._head, self._n = e["hash"], self._n + 1

    def append(self, kind: str, payload: dict[str, Any]) -> str:
        with self._lock:
            body = {"seq": self._n, "ts": time.time(), "kind": kind, "payload": payload}
            h = _digest(self._head, body)
            with self.path.open("a") as f:
                f.write(json.dumps({**body, "prev": self._head, "hash": h}, default=str) + "\n")
            self._head, self._n = h, self._n + 1
            return h

    def entries(self, kind: str | None = None) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    if kind is None or e["kind"] == kind:
                        yield e

    def verify(self) -> bool:
        prev = GENESIS
        for i, e in enumerate(self.entries()):
            body = {k: e[k] for k in ("seq", "ts", "kind", "payload")}
            if e["seq"] != i or e["prev"] != prev or _digest(prev, body) != e["hash"]:
                return False
            prev = e["hash"]
        return True
