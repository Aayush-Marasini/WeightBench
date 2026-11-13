"""Is the agent CLI's login usable? Checked outside the sandbox, before a run id is reserved.

``claude auth status`` only reads local state: it says "logged in" for an expired OAuth session. Inside the sandbox
the agent can reach only api.anthropic.com, so it cannot refresh such a session and the run dies in seconds with
"OAuth session expired and could not be refreshed" (valid=false, cost 0). The Claude CLI has no token-free validity
check, so ``check_claude_login`` makes one tiny real call (one word, no tools, one turn, a hard budget cap) outside
the sandbox, where the CLI can refresh the session or report that it cannot. Set ``WB_SKIP_AGENT_PREFLIGHT=1`` to skip it.

``transcript_auth_failure`` is the second line of defence: a run whose transcript shows the same failure text and no
tool call is reported as ``agent_auth_failed`` (still invalid, still ledgered).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

LOGIN_HINT = "Run `claude`, then `/login`, in a normal terminal (outside any sandbox), then retry."
# Lower-case fragments of the CLI's authentication failures (seen: "Failed to authenticate: OAuth session expired
# and could not be refreshed").
AUTH_MARKERS = ("oauth session expired", "could not be refreshed", "failed to authenticate", "please run /login",
                "invalid api key", "authentication_error", "not logged in")
PROBE_PROMPT = "Reply with the single word: ok"
_VERIFIED: set[tuple[str, str]] = set()      # (agent, model) already checked in this process


class AgentAuthError(RuntimeError):
    """The agent CLI cannot be used; the message says what to do. Raised before any run id is reserved."""


def _auth_text(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in AUTH_MARKERS)


def claude_probe_command(model: str) -> list[str]:
    """One-turn, tool-less, MCP-less, session-less call with a spending cap (a fraction of a cent on a small model)."""
    return ["claude", "-p", PROBE_PROMPT, "--model", model, "--tools", "", "--no-session-persistence",
            "--disable-slash-commands", "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}),
            "--max-turns", "1", "--max-budget-usd", "0.05", "--output-format", "json"]


def check_claude_login(model: str, timeout_s: float = 90.0) -> tuple[bool, str]:
    """(ok, detail) from a real minimal call outside the sandbox. ``detail`` explains a failure."""
    exe = shutil.which("claude")
    if exe is None:
        return False, "the `claude` CLI is not on PATH. Install Claude Code, then run `claude` and `/login`."
    try:
        done = subprocess.run([exe, *claude_probe_command(model)[1:]], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return False, f"the login check timed out after {timeout_s:.0f} s (network down or the CLI is waiting for input)"
    except OSError as exc:
        return False, f"could not start the `claude` CLI: {exc}"
    out = (done.stdout or "") + "\n" + (done.stderr or "")
    try:
        result = json.loads((done.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        result = {}
    if done.returncode == 0 and isinstance(result, dict) and result.get("is_error") is False \
            and not _auth_text(str(result.get("result", ""))):
        return True, "ok"
    if _auth_text(out):
        return False, "the login is expired or invalid: " + _first_line(str(result.get("result") or out))
    return False, f"the check call failed (exit {done.returncode}): {_first_line(str(result.get('result') or out))}"


def _first_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[0] if lines else "no output")[:300]


def preflight(agent: str, model: str | None) -> None:
    """Fail fast, with the fix, if ``--agent claude`` cannot run. Cheap after the first call in a process; no-op for
    other agents, without a model, or with ``WB_SKIP_AGENT_PREFLIGHT=1``."""
    if agent != "claude" or not model or os.environ.get("WB_SKIP_AGENT_PREFLIGHT") == "1":
        return
    if (agent, model) in _VERIFIED:
        return
    ok, detail = check_claude_login(model)
    if not ok:
        raise AgentAuthError(f"Claude agent preflight failed: {detail}. {LOGIN_HINT} "
                             "No run ID was reserved. (WB_SKIP_AGENT_PREFLIGHT=1 skips this check.)")
    _VERIFIED.add((agent, model))


def transcript_auth_failure(transcript: Path) -> str | None:
    """The auth-failure line if the agent's transcript shows one and the agent never made a tool call, else None."""
    if not transcript.is_file():
        return None
    failure = None
    for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            event = None
        if not isinstance(event, dict):
            if _auth_text(line):
                failure = failure or line.strip()[:300]
            continue
        content = (event.get("message") or {}).get("content")
        if isinstance(content, list) and any(isinstance(c, dict) and c.get("type") == "tool_use" for c in content):
            return None                       # the agent worked; an auth line later is not a startup failure
        text = json.dumps(event)
        if _auth_text(text):
            failure = failure or _first_line(str(event.get("result") or text))
    return failure
