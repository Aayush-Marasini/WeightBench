"""Private author-construction forwards through the normal metered probe engine."""
from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable

from weightbench.ledger import Ledger
from weightbench.server import RunState
from weightbench.task import Budget, Task


class _ForwardTask(Task):
    probe_kinds = {'forward': 'forward'}

    def __init__(self, cap: int):
        self.budget = Budget(n_forward=cap)
        self.operation: Callable[[], Any] | None = None

    def probe(self, art_dir: Path, kind: str, payload: dict) -> Any:
        if self.operation is None:
            raise RuntimeError('no author forward supplied')
        return self.operation()


class AuthorForwardMeter:
    """Separate capped construction ledger; never consumes a benchmark agent's budget."""

    def __init__(self, directory: Path, cap: int, metadata: dict[str, Any]):
        if type(cap) is not int or cap < 0:
            raise ValueError('author forward cap must be a nonnegative integer')
        self.ledger = Ledger(directory / 'ledger.jsonl')
        if any(self.ledger.entries()):
            raise FileExistsError('use a fresh author construction directory')
        self.ledger.append('author_start', {**metadata, 'forward_cap': cap, 'official_agent_run': False})
        self._task = _ForwardTask(cap)
        self._state = RunState(self._task, directory, directory, self.ledger, directory)
        self._lock = threading.Lock()

    def call(self, operation: Callable[[], Any]) -> Any:
        """Run exactly one subject forward, charging failures and rejecting excess calls."""
        with self._lock:
            self._task.operation = operation
            try:
                code, body = self._state.probe('forward', {})
            finally:
                self._task.operation = None
        result = body.get('result')
        if code != 200 or (isinstance(result, dict) and 'error' in result):
            raise RuntimeError(body.get('error', result))
        return result

    @property
    def used(self) -> int:
        return self._state.used['forward']
