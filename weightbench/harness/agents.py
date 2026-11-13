"""Agent adapters: how to launch a benchmarked agent inside the sandbox, and what it may reach.

An adapter fixes three things per agent:
- the command line (non-interactive, permissions handled by *our* sandbox, web tools off where possible);
- the egress allowlist: the only hosts the agent's own LLM traffic may reach (port 443, via the proxy);
- the home-directory state the agent CLI needs (auth, settings), made readable / writable in the sandbox.

Built in: ``claude`` (Claude Code) and ``codex`` (OpenAI Codex CLI). Any other agent runs through
``AgentSpec.custom`` with a command template, e.g.::

    weightbench bench --agent cmd --agent-cmd "my-agent --model {model} --prompt-file {prompt_file}" \\
        --allow-host api.example.com --model my-model-1
"""
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CommandBuilder = Callable[[str, str, Path, int], list[str]]   # (model, prompt, prompt_file, max_turns) -> argv


@dataclass(frozen=True)
class AgentSpec:
    name: str                                   # adapter name, used in run ids
    build: CommandBuilder = field(repr=False)
    egress_hosts: tuple[str, ...]               # allowlisted for CONNECT on port 443
    home_state: tuple[str, ...] = ()            # home-relative paths the CLI reads / writes (auth, settings)
    transcript_format: str = "text"             # "claude-stream-json" or "text" (audited line by line)
    local_ports: tuple[int, ...] = ()           # host loopback ports the agent may use (its own local model server)
    python_runtime: bool = False                # the agent itself is a Python program (exempt from the guard, not
                                                # its children: see agent_guard WB_AGENT_ROOT)

    def command(self, model: str, prompt: str, prompt_file: Path, max_turns: int) -> list[str]:
        return self.build(model, prompt, prompt_file, max_turns)

    @staticmethod
    def custom(template: str, allow_hosts: list[str], home_state: list[str] | None = None,
               local_ports: list[int] | None = None, python_runtime: bool = False,
               transcript_format: str = "text") -> "AgentSpec":
        """Any CLI agent, including unreleased or locally served models. ``template`` may use {model}, {prompt},
        {prompt_file} and {max_turns}. The agent reaches its model through ``allow_hosts`` (``host`` = port 443,
        or ``host:port``) and/or ``local_ports`` (a model server the researcher runs on this machine).
        ``transcript_format`` says how the agent's stdout is audited: ``text`` (every line, the default), or
        ``claude-stream-json`` / ``codex-json`` for an agent that prints its tool calls as those event streams."""
        if not allow_hosts and not local_ports:
            raise ValueError("a custom agent needs --allow-host (its LLM API) or --allow-port (a local model "
                             "server); all other network access is blocked")
        if transcript_format not in TRANSCRIPT_FORMATS:
            raise ValueError(f"unknown transcript format {transcript_format!r}; use one of {TRANSCRIPT_FORMATS}")
        parts = shlex.split(template)

        def build(model: str, prompt: str, prompt_file: Path, max_turns: int) -> list[str]:
            values = {"model": model, "prompt": prompt, "prompt_file": str(prompt_file), "max_turns": str(max_turns)}
            return [part.format(**values) for part in parts]

        return AgentSpec("cmd", build, tuple(allow_hosts), tuple(home_state or ()), transcript_format,
                         local_ports=tuple(local_ports or ()), python_runtime=python_runtime)


TRANSCRIPT_FORMATS = ("text", "claude-stream-json", "codex-json")
NO_MCP = json.dumps({"mcpServers": {}})


def _claude(model: str, prompt: str, prompt_file: Path, max_turns: int) -> list[str]:
    # --strict-mcp-config with an empty config: no MCP servers at all. Otherwise Claude Code loads the
    # account's claude.ai connectors (remote tools outside the sandbox) and tries to reach mcp-proxy.anthropic.com,
    # which the egress proxy blocks and the audit flags.
    return ["claude", "-p", prompt, "--model", model, "--output-format", "stream-json", "--verbose",
            "--disallowedTools", "WebFetch,WebSearch", "--dangerously-skip-permissions",
            "--strict-mcp-config", "--mcp-config", NO_MCP, "--max-turns", str(max_turns)]


def _codex(model: str, prompt: str, prompt_file: Path, max_turns: int) -> list[str]:
    # Codex's own sandbox is turned off because ours wraps the whole process (network namespace / profile).
    return ["codex", "exec", "--model", model, "--json", "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox", prompt]


BUILTIN: dict[str, AgentSpec] = {
    "claude": AgentSpec("claude", _claude, ("api.anthropic.com",),
                        (".claude", ".claude.json", ".config/claude"), "claude-stream-json"),
    # Codex's own endpoints: its API / login hosts, its analytics host, and the content
    # storage hosts it fetches from. All are the agent vendor's; everything else stays blocked.
    "codex": AgentSpec("codex", _codex, ("api.openai.com", "chatgpt.com", "auth.openai.com", "ab.chatgpt.com",
                                         "*.oaiusercontent.com"), (".codex",), "codex-json"),
}


def get(name: str, agent_cmd: str | None = None, allow_hosts: list[str] | None = None,
        home_state: list[str] | None = None, local_ports: list[int] | None = None,
        python_runtime: bool = False, transcript_format: str = "text") -> AgentSpec:
    if name == "cmd":
        if not agent_cmd:
            raise ValueError("--agent cmd needs --agent-cmd '<command template>'")
        return AgentSpec.custom(agent_cmd, allow_hosts or [], home_state, local_ports, python_runtime,
                                transcript_format)
    if name not in BUILTIN:
        raise ValueError(f"unknown agent {name!r}; built in: {sorted(BUILTIN)}, or 'cmd' with --agent-cmd")
    spec = BUILTIN[name]
    # A built-in CLI can be pointed at an OpenAI/Anthropic-compatible gateway (Claude Code on OpenRouter, a local
    # vLLM, ...). --allow-host then *adds* to the adapter's own endpoints instead of replacing them, so the same
    # audited 'claude' path serves a gateway model without falling back to the unaudited 'cmd' adapter.
    extra: tuple[str, ...] = ()
    for host in allow_hosts or []:
        if host and host not in spec.egress_hosts and host not in extra:
            extra += (host,)
    if extra:
        return AgentSpec(spec.name, spec.build, spec.egress_hosts + extra, spec.home_state,
                         spec.transcript_format, spec.local_ports, spec.python_runtime)
    return spec
