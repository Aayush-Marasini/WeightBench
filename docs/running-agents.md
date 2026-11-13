# Running agents

This page explains how to benchmark an agent: the built-in adapters, custom command-line agents, task selection,
run identity, outputs and quarantine review. Install first ([getting-started.md](getting-started.md)).

## One-time setup

```bash
bash scripts/install.sh --agent       # agent tool environment (~/wb-agent-venv) + sandbox selftest
```

Then install your agent CLI and log in with it, outside WeightBench (`claude`, then `/login`; or `codex login`).
WeightBench ships no keys and never calls a model itself. Check the setup:

```bash
weightbench doctor --agent claude --live --model claude-haiku-5-5
```

## Commands

| Command | What it does |
|---|---|
| `weightbench bench --agent A --model M [filters] [--samples N]` | Benchmark one agent on a slice of the suite, sandboxed, then print the report. Builds missing artifacts first. |
| `weightbench run --task T --agent A [--model M]` | One run: one task, one system, one seed, one sample. Prints `result.json`. |
| `weightbench sweep --models MODEL_A,MODEL_B [filters] [--jobs J]` | Baseline and reference for every selected task, plus Claude Code runs for each listed model, in parallel. |
| `weightbench build --task T [--seed S] [--level L] [--force]` | Build a task's artifacts only. |
| `weightbench list [filters]` | List tasks (id, track, tier), easiest first. |
| `weightbench rescore` | Recompute every `runs/*/result.json` from its ledger with the current formula. |

`--agent` takes `claude`, `codex`, `cmd`, or the non-agentic floors `baseline` and `reference`.

### All flags

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--task` | | `build`, `run` | task id (required for these) |
| `--tasks` | `*` | `bench`, `sweep`, `list` | comma-separated globs over task ids, e.g. `'TR04-*'` |
| `--tracks` | all | `bench`, `sweep`, `list` | comma-separated tracks, e.g. `TR01,TR04` |
| `--tiers` | all | `bench`, `sweep`, `list` | comma-separated tiers: `foundational,intermediate,frontier` |
| `--agent` | `claude` | `bench`, `run` | `baseline`, `reference`, `claude`, `codex` or `cmd` |
| `--model` | | `bench`, `run` | the agent's model id (required for agent runs) |
| `--models` | | `sweep` | comma-separated Claude Code model ids |
| `--agent-cmd` | | `--agent cmd` | command template (see below) |
| `--allow-host` | | agent runs | comma-separated egress allowlist entries: `host` (port 443) or `host:port` |
| `--allow-port` | | `--agent cmd` | comma-separated local ports the agent may reach (a model server on this machine) |
| `--agent-transcript-format` | `text` | `--agent cmd` | `text`, `claude-stream-json` or `codex-json` |
| `--python-agent` | off | `--agent cmd` | the agent itself is a Python program |
| `--agent-home` | | `--agent cmd` | comma-separated home-relative paths the agent needs (auth, config) |
| `--seed` | `0` | all | subject-model seed |
| `--sample` | `0` | all | sample id (`run`) or first sample id (`bench`, `sweep`) |
| `--samples` | `1` | `bench`, `sweep` | number of agent samples starting at `--sample` |
| `--level` | task default | `build`, `run` | difficulty knob level (SPEC §5) |
| `--condition` | `full` | agent runs | access condition: `full`, `weights_behavior`, `behavior_only` |
| `--max-turns` | `200` | agent runs | turn cap passed to the agent (and stated in its prompt) |
| `--with-floors` | off | `bench` | also score baseline and reference |
| `--jobs` | `2` | `sweep` | parallel runs |
| `--variant` | | `run --agent reference` | a declared reference variant ([tasks.md](tasks.md)) |
| `--access-plan` | | `run` | validate this run against a declared access plan ([leaderboard-and-verification.md](leaderboard-and-verification.md)) |
| `--force` | off | `build` | rebuild artifacts even if they are current; never replaces an existing run id (in `bench`/`sweep` it makes finished runs count as collisions instead of cached evidence) |

## Built-in adapters

### Claude Code (`--agent claude`)

```bash
weightbench bench --agent claude --model claude-sonnet-5-5 --samples 3
```

The adapter runs `claude -p <prompt> --model <model> --output-format stream-json --verbose` with WebFetch and
WebSearch disabled, no MCP servers, permissions handled by the WeightBench sandbox, and `--max-turns`. Egress is
limited to `api.anthropic.com`. The CLI's state (`~/.claude`, `~/.claude.json`, `~/.config/claude`) is made available
inside the sandbox. The transcript is Claude stream-json.

Before the first run in a process, WeightBench makes one tiny real call outside the sandbox to check the login
(`claude auth status` reports "logged in" even for an expired session, and the sandbox cannot refresh it). A failed
check stops the command before any run id is reserved. `WB_SKIP_AGENT_PREFLIGHT=1` skips the check.

**API key instead of a login:** export `ANTHROPIC_API_KEY`; the agent process inherits it.

**macOS:** Claude Code keeps its login in the Keychain, which the sandbox cannot read. Create a long-lived token with
`claude setup-token` and save it to `~/.config/weightbench/claude_oauth_token` (mode 600). WeightBench passes it to the
agent as `CLAUDE_CODE_OAUTH_TOKEN` unless `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY` is already set.

### OpenAI Codex CLI (`--agent codex`)

```bash
weightbench bench --agent codex --model <model-id> --tracks TR04
```

The adapter runs `codex exec --model <model> --json --skip-git-repo-check
--dangerously-bypass-approvals-and-sandbox <prompt>` (Codex's own sandbox is off because WeightBench wraps the whole
process). Egress is limited to the vendor's hosts: `api.openai.com`, `chatgpt.com`, `auth.openai.com`,
`ab.chatgpt.com` and `*.oaiusercontent.com`. `~/.codex` is made available inside the sandbox. The transcript format
is `codex-json`. `--max-turns` is not passed to Codex.

### Gateways and compatible endpoints

A built-in adapter can point at an Anthropic- or OpenAI-compatible gateway (an OpenRouter-style service, a local
vLLM server). With a built-in adapter, `--allow-host` **adds** hosts to the adapter's own allowlist instead of
replacing it, so the run stays on the audited `claude`/`codex` path.

The agent inherits your environment, so set the CLI's own endpoint variables before running. For Claude Code these
are `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` (see the Claude Code documentation):

```bash
export ANTHROPIC_BASE_URL=https://gateway.example.com/api
export ANTHROPIC_AUTH_TOKEN=<your gateway key>
weightbench bench --agent claude --model <gateway-model-id> --allow-host gateway.example.com
```

The login preflight uses the same environment. For the `claude` adapter the run id contains the model id verbatim,
so a model id containing `/` produces a nested run directory; use a model alias without `/` if your gateway offers one.

## Any command-line agent (`--agent cmd`)

```bash
weightbench bench --agent cmd --model my-model \
    --agent-cmd "my-agent --model {model} --prompt-file {prompt_file}" \
    --allow-host api.example.com
```

`--agent-cmd` is a template, split like a shell command line (no shell runs it, and `~` is not expanded; use
absolute paths). Placeholders:

| Placeholder | Value |
|---|---|
| `{model}` | the `--model` value |
| `{prompt}` | the full agent prompt as one argument |
| `{prompt_file}` | path of `PROMPT.md` in the agent workdir (same text) |
| `{max_turns}` | the `--max-turns` value |

The agent's working directory is its workdir, which contains `TASK.md`, the task's public files and `wb.py`.

Network: a custom agent needs `--allow-host` (its LLM API, port 443 or `host:port`) and/or `--allow-port` (a model
server on this machine's loopback), otherwise the command is rejected. Everything else is blocked.

Other options:

- `--agent-home .myagent,.config/myagent` makes those home-relative paths available inside the sandbox (auth files,
  settings). On Linux the rest of `$HOME` is an empty tmpfs.
- `--python-agent` declares that the agent process itself is Python. The runtime guard then exempts the agent's own
  process (it still applies to every child process the agent starts).
- `--agent-transcript-format` controls how stdout is audited:
  - `text` (default): every line is audited.
  - `claude-stream-json`: the agent prints Claude Code-style JSON events; the audit reads the tool-call inputs and
    assistant text.
  - `codex-json`: the agent prints Codex-style JSON events.

  With a structured format, any line that is not valid JSON quarantines the run.

`cmd` runs are excluded from the report and the leaderboard means (they are listed with the reason); use them for
your own experiments and for red-team testing.

## Inside the sandbox

The agent receives `TASK.md` (the task statement, rules and budget), the public artifacts and `wb.py`:

```
python wb.py budget                                  # remaining metered calls
python wb.py probe <kind> '<json payload>'           # metered probe (or: probe <kind> @payload.json)
python wb.py submit-report report.json
python wb.py submit-refusal report.json
python wb.py submit-delta delta.npz [report.json]    # {param_name: ΔW}, same shapes as the base
```

Any command accepts `-o FILE` to save the full JSON response. The agent may resubmit; the last submission counts. A run
without a submission scores 0. Submissions never return a score; evaluation happens after the agent exits.

The agent's prompt states the rules: no hand-written forward passes or simulation of the subject model unless
`TASK.md` allows a specific evaluation, no gradient descent or numerical gradients, no copying weights out of the
metered representation, and no `python -I`, `-S` or `-E`. Violations are audited and quarantine the run (SPEC §1).

## Choosing tasks

```bash
weightbench bench --agent claude --model M --tracks TR01,TR04
weightbench bench --agent claude --model M --tiers foundational
weightbench bench --agent claude --model M --tasks 'TR05-*,TR04-GO-001'
weightbench list --tracks TR02 --tiers intermediate      # preview a selection
```

Filters combine (a task must match all of them). Tasks run in difficulty order: tier, then knob level, then track and
id. Tasks that need extras, Hugging Face licences or a GPU are listed in [tasks.md](tasks.md).

## Samples, seeds and run identity

Every run has an immutable id:

```
<task>__<system>__s<seed>__r<sample>[__L<level>][__<condition>]
```

- `<system>` is the model id for `claude`, `<agent>-<model>` for other agents (`/` and `:` replaced by `_`), or
  `baseline` / `reference`.
- `__L<level>` appears only for a non-default knob level; `__<condition>` only for a non-`full` access condition.

A run id is reserved atomically and **never reused**, even after an interrupted run or with `--force`. To repeat a
run, use a fresh sample id. `bench` and `sweep` check every id before building anything:

- a finished run with a verified ledger is reused as cached evidence;
- an existing id without a verified result stops the batch, names the blocking ids, and suggests the next free
  `--sample`. Nothing is built or charged.

`--samples 3 --sample 10` runs samples 10, 11 and 12 per task. Floors (`--with-floors`) always use the first sample id.
`--seed` changes the subject model itself (new weights, new coordinates).

## Difficulty levels (`--level`)

```bash
weightbench build --task TR05-DEAD-001 --level 2
weightbench run --task TR05-DEAD-001 --agent claude --model M --level 2
```

`--level` applies to `build` and `run`. Each task supports the levels listed in SPEC §5 and rejects others before any
work. Non-default levels have their own artifacts and run ids and are excluded from the leaderboard.

## Access conditions (`--condition`)

`full` (default) is the normal benchmark. Two matched conditions support controlled studies of how much weight access
helps:

- `weights_behavior`: the declared metadata and weight files plus the task's behaviour probes;
- `behavior_only`: the same metadata and behaviour probes, but no weight files.

In both, internal probes (activations, attention maps) are disabled, local execution of any network is forbidden, and
the prompt says so. Each task family has a reviewed access profile (`weightbench/conditions.py`). Conditions apply to
agent runs only; floors always run under `full`. Planned paired studies use `weightbench-compare` (see
[leaderboard-and-verification.md](leaderboard-and-verification.md)).

## Timeouts

- **Wall clock:** each task's `Budget.wall_clock_s` (1 to 3 hours, [tasks.md](tasks.md)). On Linux it is also every
  agent process's CPU-time limit.
- **Stall watchdog:** an agent whose transcript does not grow for `WB_AGENT_STALL_S` seconds (default 1800) is killed;
  `result.json` then has `"stalled": true`. `WB_AGENT_STALL_S=0` disables the watchdog.
- **Turns:** `--max-turns` (default 200), passed to Claude Code and to custom agents via `{max_turns}`.

## Outputs

Each run writes `runs/<run_id>/`:

| File | Contents |
|---|---|
| `ledger.jsonl` | append-only, hash-chained ledger: run start, probes, runtime traces, submissions, compute, agent exit, evaluation |
| `result.json` | the score (`s_wb`, `s_plus`, `d_col`, `valid`, `n_forward`, `n_probe`, …), audit flags, `quarantined`, usage and `ledger_ok` |
| `transcript.jsonl` | the agent's stdout and stderr |
| `audit.json` | audit flags, trace summary, egress summary and the quarantine decision |
| `egress.jsonl` | every outbound connection attempt through the proxy, allowed or denied |
| `submission/` | the accepted submission (`submission.json`, `delta.npz`) |

Floor runs (`baseline`, `reference`) have no transcript, audit or egress log. Built artifacts live in
`artifacts/<task>/seed<s>/` (`public/`, `private/` and a `.built` provenance marker).

## Reading results

```bash
cat runs/<run_id>/result.json
weightbench-report                    # leaderboard over every run in runs/
weightbench-verify --all runs/        # recheck every score from its ledger
```

`bench` prints one line per run (`S_WB`, `S+`, `valid`, `QUARANTINED`) and the report at the end. Key fields of
`result.json`:

- `s_wb`: the official per-task score (SPEC §3.2); `s_plus`: task gain before discounts.
- `valid`: false if there was no submission or a hard limit was broken (`details.reason` says which).
- `quarantined` and `audit_flags`: the audit outcome. A quarantined run counts as 0 until it is cleared.
- `agent_auth_failed`: the agent's login failed at startup; log in again and use a fresh sample.

The report and the leaderboard are described in [leaderboard-and-verification.md](leaderboard-and-verification.md).

## Quarantine and review

Any audit flag quarantines a run (SPEC §1). To resolve it:

1. **Automated review (advisory):**

   ```bash
   weightbench review --auto                  # Stage A rules only, every quarantined run without review.json
   weightbench review --auto --dry-run        # print verdicts, write nothing
   weightbench review --auto --run <run_id> --force
   weightbench review --auto --judge claude --model claude-sonnet-5-5   # add the LLM judge
   ```

   It writes `auto_review.json` into each run folder and never changes the evidence. A run whose auto review clears
   every flag counts as auto-cleared. Rules: [AUTO_REVIEW.md](AUTO_REVIEW.md).

2. **Human review:** inspect `transcript.jsonl`, `audit.json` and `egress.jsonl`, then write `runs/<run_id>/review.json`:

   ```json
   {"cleared": true, "reviewer": "<name>", "reason": "<why the flags are false alarms>",
    "flags_reviewed": ["forward_pass_suspect"]}
   ```

   `"cleared": false` rejects the run (and overrides an auto-clearance). A human `review.json` always wins over the
   automated reviewer.
