"""Validate guard events before they enter the ledger or the post-run audit."""
from __future__ import annotations

import math
from typing import Any

COUNTERS = ("flops", "weight_matmuls", "forward_chains", "metering_escapes", "unsupported_contractions",
            "raw_weight_opens", "raw_weight_data_bytes", "weight_export_bytes")
# Per-process resource use (floats, seconds / KB). Optional: older traces have none, and their
# absence means 0, so legacy evidence still verifies and sums to the same totals.
CPU_FIELDS = ("cpu_user", "cpu_sys", "cpu_children_user", "cpu_children_sys", "max_rss_kb")


def normalize_trace(value: Any) -> dict[str, Any]:
    """Return bounded metadata and nonnegative integral counters; reject malformed values.

    Missing/null optional fields retain legacy zero/default semantics. Unknown metadata is
    discarded. PID-less legacy files remain supported; a supplied PID must be a positive integer.
    """
    if not isinstance(value, dict):
        raise ValueError("trace must be a JSON object")
    out: dict[str, Any] = {}
    pid = value.get("pid")
    if pid is not None:
        if type(pid) is not int or pid <= 0:
            raise ValueError("trace pid must be a positive integer")
        out["pid"] = pid
    for field in COUNTERS:
        number = value.get(field)
        if number is None:
            number = 0
        if type(number) not in (int, float) or number < 0:
            raise ValueError(f"trace {field} must be a nonnegative integer")
        if isinstance(number, float) and (not math.isfinite(number) or not number.is_integer()):
            raise ValueError(f"trace {field} must be a finite integer")
        out[field] = int(number)
    for field in CPU_FIELDS:
        number = value.get(field)
        if number is None:
            continue                 # optional; absence means 0 (legacy traces)
        if type(number) not in (int, float) or number < 0 or (isinstance(number, float) and not math.isfinite(number)):
            raise ValueError(f"trace {field} must be a nonnegative finite number")
        out[field] = float(number)
    blocked = value.get("blocked")
    if blocked is None:
        blocked = []
    if not isinstance(blocked, list) or len(blocked) > 50 or any(not isinstance(b, str) for b in blocked):
        raise ValueError("trace blocked must be a list of at most 50 strings")
    out["blocked"] = [b[:200] for b in blocked]
    argv = value.get("argv")
    if argv is not None:
        if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
            raise ValueError("trace argv must be a list of strings")
        out["argv"] = [arg[:2000] for arg in argv[:4]]
    for field in ("final", "agent_root"):
        flag = value.get(field)
        if flag is not None:
            if type(flag) is not bool:
                raise ValueError(f"trace {field} must be boolean")
            out[field] = flag
    return out
