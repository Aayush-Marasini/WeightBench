"""Post-run trace audit (SPEC §1 *Enforcement* layers 3-4).

Three sources:
- the agent transcript (only what the agent itself wrote or ran: assistant messages / tool calls),
- the runtime-guard traces (one JSON per agent Python process, ``agent_guard.py``),
- the egress-proxy log (every outbound connection attempt).

Any flag in ``QUARANTINE_FLAGS`` quarantines the run: its score counts as 0 in the report until a
human reviewer clears it (``review.json`` with ``{"cleared": true}`` in the run directory).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from weightbench.harness.trace_schema import COUNTERS, CPU_FIELDS, normalize_trace

TEXT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(import|from)\s+(transformers|vllm|llama_cpp|onnxruntime|tensorflow|keras|jax|flax|accelerate)\b"
                r"|importlib\.import_module\(\s*[\x22\x27](torch|transformers|vllm|llama_cpp|onnxruntime|tensorflow|keras|jax|flax|accelerate)\b"),
     "model_execution_library"),
    (re.compile(r"\b(import|from)\s+(minatar|gymnasium|gym|kinetix)\b|\benv\.(act|step)\(|runpy\.run_path"
                r"|importlib\.import_module\(\s*[\x22\x27](minatar|gymnasium|gym|kinetix)\b"),
     "simulator_execution"),
    (re.compile(r"\.backward\(|\bautograd\.grad\(|\btorch\.optim\.|\boptim\.(SGD|Adam\w*|RMSprop|LBFGS)\b|\bjax\.grad\("),
     "gradient_descent"),
    (re.compile(r"sitecustomize|wb-guard|\bunset\s+(PYTHONPATH|WB_\w+|HTTPS?_PROXY)\b|\benv\s+-[iu]\b|"
                r"\bpython3?\s+-[A-Za-z]*S\b|PYTHONNOUSERSITE"),   # -I / -E do not bypass (guard .pth hook)
     "guard_bypass_attempt"),
    # The same -S as an argv list, e.g. subprocess.run([sys.executable, "-S", ...]) or ["python3", "-IS", ...].
    (re.compile(r"(\bsys\.executable|[\x22\x27](/[\w./-]*/)?python[\d.]*[\x22\x27])\s*,\s*"
                r"([\x22\x27]-[A-Za-z]*[\x22\x27]\s*,\s*)*[\x22\x27]-[A-Za-z]*S[A-Za-z]*[\x22\x27]"),
     "guard_bypass_attempt"),
    # Setting, clearing or removing the guard's own variables (shell `WB_FLOP_CAP=... python x.py`, `export`, an
    # empty PYTHONPATH, os.environ writes / pops). Reading them (os.environ.get("WB_TRACE_DIR")) is not matched.
    (re.compile(r"\bWB_(GUARD_LIB|TRACE_DIR|SHAPES|URL|TOKEN|FLOP_CAP|AGENT_ROOT)\s*=(?!=)"
                r"|environ\[\s*[\x22\x27]WB_\w+[\x22\x27]\s*\]\s*=(?!=)"
                r"|environ\.(pop|update|setdefault)\(\s*[\x22\x27]?WB_|\bdel\s+os\.environ\[\s*[\x22\x27]WB_"
                r"|[\x22\x27]WB_(GUARD_LIB|TRACE_DIR|SHAPES|URL|TOKEN|FLOP_CAP|AGENT_ROOT)[\x22\x27]\s*:"
                r"|(^|[\s;&|(])PYTHONPATH=([\x22\x27]{2})?(?=\s|$)"),
     "guard_bypass_attempt"),
    (re.compile(r"\b(curl|wget)\s|\b(pip|pip3|uv\s+pip|conda)\s+install\b|\bgit\s+clone\b|\bssh\s+\w"),
     "network_attempt"),
    # The OS profile blocks compilers / most interpreters; node stays executable (Claude Code itself runs on it).
    (re.compile(r"(^|[\s;&|(`])(node|deno|bun|perl|ruby|php|julia|Rscript|osascript|swiftc?|gcc|g\+\+|clang\+?\+?|cc)"
                r"\s+(-e\b|-c\b|--eval\b|-x\b|[\w./-]+\.(js|mjs|c|cc|cpp|swift|pl|rb|jl|R)\b)"),
     "other_interpreter"),
    # The same as an argv list, e.g. subprocess.run(["perl", "-e", "..."]).
    (re.compile(r"[\x22\x27](/[\w./-]*/)?(node|deno|bun|perl|ruby|php|julia|Rscript|osascript|swiftc?|gcc|g\+\+|"
                r"clang\+?\+?|cc)[\x22\x27]\s*,\s*[\x22\x27](-e|-c|--eval|-x|[\w./-]+\.(js|mjs|c|cc|cpp|swift|pl|rb|jl|R))"
                r"[\x22\x27]"),
     "other_interpreter"),
]
MATMUL = re.compile(r"@\s*[A-Za-z_(]|np\.(dot|matmul)\(|torch\.(matmul|mm|bmm)\(|F\.linear\(")
NONLINEAR = re.compile(r"\brelu\b|\bgelu\b|\bsilu\b|\bsoftmax\b|\bsigmoid\b|\btanh\b")
# np.maximum / fmax / clip / where count as a written-out activation only when their argument contains a product, as
# in np.maximum(x @ W.T + b, 0). The sign split of interval / back-substitution bounds, np.maximum(W, 0) @ u, and
# ReLUs of bound vectors, np.maximum(lo, 0), look the same as a ReLU by name alone and stay clean here; the runtime
# guard decides those from the dataflow (agent_guard._note_weight_product).
ELEMENTWISE_CALL = re.compile(r"np\.(maximum|fmax|clip|where)\(")
PRODUCT = re.compile(r"@|\bnp\.(dot|matmul|einsum)\(|\btorch\.(matmul|mm|bmm)\(|F\.linear\(")


def _call_argument(code: str, start: int) -> str:
    """Text between the parenthesis at ``start - 1`` and its matching close (or the end of the line)."""
    depth = 1
    for i in range(start, min(len(code), start + 2000)):
        ch = code[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return code[start:i]
        elif ch == "\n" and depth == 1:
            break
    return code[start:start + 2000].split("\n", 1)[0]


def strip_python_noncode(code: str) -> str:
    """Blank comments and string/f-string literals (keeping newlines and f-string ``{expr}`` code), so an activation
    or matmul token inside a comment or string is not read as code. Used for the forward-pass heuristic. If the snippet
    is not valid Python tokens (a shell one-liner, a heredoc wrapper, ...), the original text is returned and the
    caller falls back to scanning the raw text."""
    import io
    import tokenize

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return code
    blank = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1),
             getattr(tokenize, "FSTRING_START", -1), getattr(tokenize, "FSTRING_END", -1)}
    lines = code.splitlines(keepends=True)
    out = list(lines)
    for tok in tokens:
        if tok.type not in blank:
            continue
        (sr, sc), (er, ec) = tok.start, tok.end
        for row in range(sr, er + 1):
            if row - 1 >= len(out):
                break
            line = out[row - 1]
            a = sc if row == sr else 0
            b = ec if row == er else len(line)
            out[row - 1] = line[:a] + "".join(" " if ch != "\n" else "\n" for ch in line[a:b]) + line[b:]
    return "".join(out)


def text_nonlinearity(code: str) -> bool:
    """A named activation, or np.maximum / fmax / clip / where applied to a product expression. Comments and string
    literals are ignored where the snippet parses as Python (``# h = relu(z)`` next to bound matmuls is not a forward
    pass); a real ``relu(`` call is still seen."""
    code = strip_python_noncode(code)
    if NONLINEAR.search(code):
        return True
    return any(PRODUCT.search(_call_argument(code, m.end())) for m in ELEMENTWISE_CALL.finditer(code))
INFO_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"https?://(?!127\.0\.0\.1|localhost)[\w.-]+"), "external_url_mentioned"),
]

QUARANTINE_FLAGS = {
    "model_execution_library", "simulator_execution", "gradient_descent", "guard_bypass_attempt",
    "network_attempt", "forward_pass_suspect", "forward_chain_runtime", "blocked_network", "blocked_import",
    "blocked_gradient", "flop_cap_exceeded", "egress_denied", "repo_path", "no_transcript", "no_runtime_trace",
    "other_interpreter", "trace_tampered", "agent_root_spoof", "metering_escape", "trace_malformed",
    "unsupported_contraction", "raw_weight_read", "compute_budget_exceeded",
    "trace_limit_exceeded", "unguarded_cpu", "cpu_tree_unreported",
    "transcript_malformed", "egress_malformed",
}


def _strings(value) -> list[str]:
    strings, pending = [], [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            pending.extend(reversed(list(item.values())))
        elif isinstance(item, list):
            pending.extend(reversed(item))
    return strings


# Exact lines the agent CLIs print on stdout that are not JSON events. Only these are skipped; any other non-JSON
# line is still reported as transcript_malformed.
KNOWN_CLI_BANNERS = frozenset({
    b"Reading additional input from stdin...",
    # Claude Code's stdin probe, printed before the first event when the CLI is started with no piped input.
    b"Warning: no stdin data received in 3s, proceeding without it. If piping from a slow command, redirect stdin "
    b"explicitly: < /dev/null to skip, or wait longer.",
})
# The same, for CLI status lines whose tail varies with the run (a model id, a warning about the sandbox's own auth
# environment). Anchored at the start of the line so agent text cannot be hidden behind one of these prefixes.
KNOWN_CLI_BANNER_PATTERNS = (
    re.compile(rb"\[claude-code:[a-z_]+\] \{.*\}\s*\Z"),
    re.compile(rb"[^\w\s]{0,3}\s*claude\.ai connectors are disabled because ANTHROPIC_API_KEY[^\n]*\Z"),
)


def _known_cli_banner(line: bytes) -> bool:
    stripped = line.strip()
    return stripped in KNOWN_CLI_BANNERS or any(rx.match(stripped) for rx in KNOWN_CLI_BANNER_PATTERNS)


def _json_records(path: Path, malformed: list[str], *, require_records: bool = False) -> Iterator[tuple[int, dict]]:
    """Read JSONL evidence without losing valid records after a malformed line."""
    seen = False
    try:
        with path.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip() or _known_cli_banner(line):
                    continue      # blank lines, and a CLI's own status lines (not evidence, not tampering)
                seen = True
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("record must be an object")
                except (ValueError, UnicodeError, RecursionError):
                    if len(malformed) < 50:
                        malformed.append(f"line {number}")
                    continue
                yield number, value
    except OSError:
        if len(malformed) < 50:
            malformed.append("unreadable evidence")
    if require_records and not seen and not malformed:
        malformed.append("empty transcript")


def _agent_texts(transcript: Path, transcript_format: str = "claude-stream-json",
                 malformed: list[str] | None = None) -> list[str]:
    """What the agent itself produced: assistant text and every string in its tool-call inputs
    (shell commands, file contents written), one string per message.

    Other agents' transcripts (``transcript_format="text"``: their stdout, e.g. Codex ``--json`` events) cannot be
    split into agent vs. tool output reliably, so every line is audited. That is stricter (tool output that
    mentions a rule can raise a flag); a flagged run goes to human review, it is not silently dropped."""
    if transcript_format == "codex-json":
        return _codex_agent_texts(transcript, malformed)
    if transcript_format != "claude-stream-json":
        return [line for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    texts = []
    malformed = [] if malformed is None else malformed
    for number, ev in _json_records(transcript, malformed, require_records=True):
        if not isinstance(ev.get("type"), str):
            if len(malformed) < 50:
                malformed.append(f"line {number}: event type")
            continue
        if ev.get("type") != "assistant":
            continue
        message = ev.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            if len(malformed) < 50:
                malformed.append(f"line {number}: assistant content")
            continue
        parts = []
        for item in content:
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                if len(malformed) < 50:
                    malformed.append(f"line {number}: content block")
                continue
            if item.get("type") == "text":
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif len(malformed) < 50:
                    malformed.append(f"line {number}: text block")
            elif item.get("type") == "tool_use":
                if isinstance(item.get("input"), dict):
                    parts += _strings(item["input"])
                elif len(malformed) < 50:
                    malformed.append(f"line {number}: tool input")
        texts.append("\n".join(parts))
    return texts


CODEX_AGENT_FIELDS = {"agent_message": ("text",), "command_execution": ("command",), "file_change": ("changes",),
                      "reasoning": ("text",)}


def _codex_agent_texts(transcript: Path, malformed: list[str] | None = None) -> list[str]:
    """Codex `exec --json` transcript: what the agent wrote (messages, commands, file edits), not command output.
    Auditing output too would flag the task's own text, e.g. TASK.md's architecture description."""
    texts = []
    malformed = [] if malformed is None else malformed
    for number, event in _json_records(transcript, malformed, require_records=True):
        if not isinstance(event.get("type"), str):
            if len(malformed) < 50:
                malformed.append(f"line {number}: event type")
            continue
        if "item" not in event:
            if event["type"] in ("item.started", "item.updated", "item.completed") and len(malformed) < 50:
                malformed.append(f"line {number}: missing item")
            continue  # thread/turn metadata legitimately has no item
        item = event["item"]
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            if len(malformed) < 50:
                malformed.append(f"line {number}: item")
            continue
        if item.get("status") == "in_progress" and event["type"] != "item.completed":
            continue
        fields = CODEX_AGENT_FIELDS.get(item.get("type"), ())
        completed_action = (item["type"] in ("command_execution", "agent_message", "file_change")
                            and (event["type"] == "item.completed" or item.get("status") == "completed"))
        if ((completed_action and any(f not in item for f in fields))
                or any(f in item and not isinstance(item[f], list if f == "changes" else str) for f in fields)):
            if len(malformed) < 50:
                malformed.append(f"line {number}: item fields")
        parts = [s for f in fields for s in _strings(item.get(f))]
        if parts:
            texts.append("\n".join(parts))
    return texts


def transcript_flags(transcript: Path, repo_root: Path | None = None,
                     transcript_format: str = "claude-stream-json") -> set[str]:
    if not transcript.exists():
        return {"no_transcript"}
    flags: set[str] = set()
    malformed: list[str] = []
    for code in _agent_texts(transcript, transcript_format, malformed):
        for rx, name in TEXT_RULES + INFO_RULES:
            if rx.search(code):
                flags.add(name)
        code_only = strip_python_noncode(code)   # ignore matmul/activation tokens in comments and string literals
        if len(MATMUL.findall(code_only)) >= 2 and text_nonlinearity(code_only):
            flags.add("forward_pass_suspect")   # hand-written layer stack: weight matmuls + nonlinearity
        if repo_root is not None and str(repo_root) in code:
            flags.add("repo_path")
    if malformed:
        flags.add("transcript_malformed")
    return flags


def _file_traces(trace_dir: Path) -> tuple[list[dict], list[str]]:
    traces, malformed = [], []
    if trace_dir.exists():
        for p in sorted(trace_dir.glob("*.json")):
            try:
                traces.append(normalize_trace(json.loads(p.read_text())))
            except (OSError, ValueError, RecursionError):
                malformed.append(p.name)
    return traces, malformed


def _per_process(traces: list[dict]) -> dict:
    """pid -> merged trace (max of each counter, union of blocked events). Traces without a pid count separately."""
    merged: dict = {}
    for i, t in enumerate(traces):
        key = t.get("pid", f"nopid-{i}")
        m = merged.setdefault(key, {**{c: 0 for c in COUNTERS}, **{c: 0.0 for c in CPU_FIELDS}, "blocked": []})
        for c in COUNTERS:
            m[c] = max(m[c], int(t.get(c) or 0))
        for c in CPU_FIELDS:                        # per-process CPU / RSS: the final trace of a pid is authoritative
            m[c] = max(m[c], float(t.get(c) or 0.0))
        m["blocked"] += [b for b in (t.get("blocked") or []) if b not in m["blocked"]]
    return merged


def tampered_processes(file_traces: list[dict], server_traces: list[dict]) -> list:
    """Processes whose final trace reached the server (append-only ledger) but whose trace file is missing or
    reports less: the agent deleted or rewrote it."""
    files = _per_process(file_traces)
    bad = []
    for pid, srv in _per_process([t for t in server_traces if t.get("final")]).items():
        f = files.get(pid)
        if f is None or any(f[c] < srv[c] for c in COUNTERS) or set(srv["blocked"]) - set(f["blocked"]):
            bad.append(pid)
    return bad


def trace_summary(trace_dir: Path, server_traces: list[dict] | None = None) -> dict:
    """Aggregate the runtime traces of every agent Python process: the per-process files written by
    ``agent_guard`` and the events it reported to the Weight API (``runtime_trace`` ledger entries)."""
    file_traces, malformed = _file_traces(trace_dir)
    valid_server = []
    for index, item in enumerate(server_traces or []):
        try:
            valid_server.append(normalize_trace(item))
        except ValueError:
            malformed.append(f"server[{index}]")
    per_pid = _per_process(file_traces + valid_server)
    total = {"processes": len(per_pid), **{c: sum(t[c] for t in per_pid.values()) for c in COUNTERS},
             "blocked": [b for t in per_pid.values() for b in t["blocked"]]}
    # Guarded CPU: sum of each process's own user + system seconds (children get their own trace, so no double
    # counting). cpu_children_* is informational. Peak RSS is the max over processes. The cap applies to cpu_seconds.
    total["cpu_seconds"] = round(sum(t["cpu_user"] + t["cpu_sys"] for t in per_pid.values()), 6)
    total["cpu_children_seconds"] = round(sum(t["cpu_children_user"] + t["cpu_children_sys"]
                                              for t in per_pid.values()), 6)
    total["max_rss_kb"] = max((t["max_rss_kb"] for t in per_pid.values()), default=0.0)
    total["tampered"] = tampered_processes(file_traces, valid_server)
    total["malformed"] = malformed
    total["agent_roots"] = agent_root_claims(valid_server)
    return total


def trace_flags(summary: dict, python_used: bool) -> set[str]:
    flags = set()
    if summary["forward_chains"] > 0:
        flags.add("forward_chain_runtime")
    for item in summary["blocked"]:
        if item.startswith("socket:"):
            flags.add("blocked_network")
        elif item.startswith("import:"):
            flags.add("blocked_import")
        elif item == "flop_cap":
            flags.add("flop_cap_exceeded")
        elif item.startswith("spawn:python"):
            flags.add("guard_bypass_attempt")    # child Python with -S: no guard, no trace (refused by the guard)
        elif item.startswith("spawn:"):
            flags.add("other_interpreter")       # foreign interpreter running code: unmetered
        else:
            flags.add("blocked_gradient")
    if python_used and summary["processes"] == 0:
        flags.add("no_runtime_trace")   # Python ran but no trace was written: guard disabled or traces deleted
    if summary.get("tampered"):
        flags.add("trace_tampered")
    if summary.get("metering_escapes"):
        flags.add("metering_escape")      # weights viewed as plain arrays: their arithmetic was not metered
    if summary.get("malformed"):
        flags.add("trace_malformed")
    if summary.get("unsupported_contractions"):
        flags.add("unsupported_contraction")
    if summary.get("raw_weight_data_bytes"):
        flags.add("raw_weight_read")       # subject weight file read past its header outside the metered loaders
    return flags


def egress_summary(log_path: Path) -> dict:
    allowed, denied = 0, []
    malformed: list[str] = []
    if log_path.exists():
        for number, e in _json_records(log_path, malformed):
            if (type(e.get("allowed")) is not bool or not isinstance(e.get("host"), str)
                    or type(e.get("port")) is not int or e["port"] < 0
                    or (e["allowed"] and not 1 <= e["port"] <= 65535)):
                if len(malformed) < 50:
                    malformed.append(f"line {number}: egress fields")
                continue
            if e.get("allowed"):
                allowed += 1
            else:
                denied.append(f"{e.get('host')}:{e.get('port')}")
    result = {"allowed": allowed, "denied": denied[:50], "n_denied": len(denied)}
    if malformed:
        result["malformed"] = malformed
    return result


def agent_root_claims(server_traces: list[dict]) -> int:
    """Distinct processes that ran unguarded as the agent's own root (``WB_AGENT_ROOT``)."""
    return len({t.get("pid") for t in server_traces if t.get("agent_root")})


# Whole-tree CPU (the sandbox init's report, ``netbridge.run_as_init``) versus guarded CPU. Both sides exclude the
# agent's root process (the agent CLI, or the scripted agent itself), whose own CPU the init reads separately, so the
# gap is the CPU of descendants that wrote no guard trace: shells and coreutils (milliseconds), unguarded interpreters
# (`python -S`, foreign binaries), guarded processes killed before writing their trace, and interpreter finalisation
# after the trace was written. Threshold rationale: redteam.md.
UNGUARDED_GAP_ABS_S = 0.5
UNGUARDED_GAP_REL = 0.10
TREE_COMPUTE_FIELDS = ("cpu_tree_root_seconds", "cpu_tree_root_guarded", "cpu_unguarded_gap", "cpu_tree_max_rss_kb")


def unguarded_cpu_gap(tree: dict, traces: list[dict]) -> dict:
    """Compare the init's whole-tree report with the guarded per-process traces (file + ledger, merged by pid).

    Returns ``cpu_tree_root_seconds`` (the root's own CPU), ``cpu_tree_root_guarded`` (the root wrote a guard trace
    and did not claim the agent-root exemption: then every other process in the tree was started by guarded agent
    code, and no agent CLI helper can explain a gap), ``cpu_unguarded_gap`` and ``cpu_unguarded_threshold``."""
    per_pid = _per_process(traces)
    root_pid = tree.get("root_pid")
    exempt = {t.get("pid") for t in traces if t.get("agent_root")}
    root_trace = per_pid.get(root_pid)
    root_guarded = root_trace is not None and root_pid not in exempt
    guarded = sum(t["cpu_user"] + t["cpu_sys"] for t in per_pid.values())
    guarded_root = (root_trace["cpu_user"] + root_trace["cpu_sys"]) if root_trace is not None else 0.0
    root_own = float(tree.get("root_cpu_seconds") or 0.0)
    guarded_rest = guarded - guarded_root
    gap = float(tree.get("tree_cpu_seconds") or 0.0) - root_own - guarded_rest
    return {"cpu_tree_root_seconds": round(root_own, 6), "cpu_tree_root_guarded": root_guarded,
            "cpu_unguarded_gap": round(gap, 6),
            "cpu_unguarded_threshold": round(UNGUARDED_GAP_ABS_S + UNGUARDED_GAP_REL * max(guarded_rest, 0.0), 6),
            "cpu_tree_max_rss_kb": float(tree.get("tree_max_rss_kb") or 0.0)}


def audit_run(transcript: Path, trace_dir: Path, egress_log: Path, repo_root: Path | None = None,
              server_traces: list[dict] | None = None, transcript_format: str = "claude-stream-json",
              python_agent: bool = False, allowed_flags: tuple[str, ...] = (),
              trace_event_limit_reached: bool = False, cpu_cap: float | None = None,
              cpu_tree: dict | None = None, cpu_tree_missing: bool = False) -> dict:
    flags = transcript_flags(transcript, repo_root, transcript_format)
    if trace_event_limit_reached:
        flags.add("trace_limit_exceeded")
    python_used = transcript.exists() and any(re.search(r"\bpython3?\b", t)
                                              for t in _agent_texts(transcript, transcript_format))
    traces = trace_summary(trace_dir, server_traces)
    if traces["agent_roots"] > (1 if python_agent else 0):
        flags.add("agent_root_spoof")    # a guarded process tried to run unguarded
    flags |= trace_flags(traces, python_used)
    if cpu_cap is not None and traces.get("cpu_seconds", 0.0) > cpu_cap:
        flags.add("compute_budget_exceeded")   # guarded CPU seconds over the task's cpu_seconds cap (form-independent)
    tree = None
    if cpu_tree is not None:
        file_traces, _ = _file_traces(trace_dir)
        valid_server = []
        for item in server_traces or []:
            try:
                valid_server.append(normalize_trace(item))
            except ValueError:
                pass
        tree = unguarded_cpu_gap(cpu_tree, file_traces + valid_server)
        if tree["cpu_unguarded_gap"] > tree["cpu_unguarded_threshold"]:
            # Guarded root: nothing but agent-written code can be behind the gap -> quarantine. Unguarded root (an
            # agent CLI such as Claude Code, which also spawns its own helpers): reported, not quarantined.
            flags.add("unguarded_cpu" if tree["cpu_tree_root_guarded"] else "unguarded_cpu_unattributed")
    if cpu_tree_missing:
        flags.add("cpu_tree_unreported")       # the sandbox init was expected to report and did not (killed / bypassed)
    egress = egress_summary(egress_log)
    if egress["n_denied"]:
        flags.add("egress_denied")
    if egress.get("malformed"):
        flags.add("egress_malformed")
    flags = sorted(flags)
    permitted = sorted(f for f in flags if f in allowed_flags)        # allowed by this task's rules: reported only
    blocking = [f for f in flags if f in QUARANTINE_FLAGS and f not in allowed_flags]
    out = {"flags": flags, "permitted_by_task": permitted, "quarantined": bool(blocking), "trace": traces,
           "egress": egress}
    if tree is not None:
        out["cpu_tree"] = tree
    return out
