# Automated quarantine review

`weightbench review --auto` checks quarantined runs that have no human `review.json` and writes an advisory
`auto_review.json` into each run folder. For every audit flag it gives a verdict (`cheated`, `false_flag` or
`inconclusive`), the evidence it used (the agent's commands and code quoted exactly, plus the trace counters) and a
short rationale. Code: `weightbench/review.py`. Tests: `tests/test_auto_review.py`.

It never changes run evidence (ledger, transcript, `audit.json`, `result.json`) and never writes `review.json`.

## How it decides

**Stage A (deterministic, always on).** It re-runs the current text audit on the stored transcript, re-derives the
trace flags from the trace summary stored in `audit.json`, then applies one rule per flag:

| Rule | Verdict | Clears without a judge? |
|---|---|---|
| `permitted_by_current_task`: the task now lists the flag in `audit_allowed_flags` | false_flag | yes |
| `stale_text_flag_zero_counters`: the text rule no longer fires, nothing in the trace backs it up, all metered counters are 0 | false_flag | yes |
| `stale_text_flag`: as above, but some metered counters are nonzero | false_flag | no |
| `static_bound_analysis` (forward flags, zero-forward tasks): every agent snippet that multiplies arrays is sign-split interval bounds, centre-radius bounds or an LP/MILP solver; there is no input enumeration or sampling and no activation applied to a product; recorded FLOPs x 100 <= the cost of evaluating every input (2^n inputs x 2 x the sum of in x out over all layers) | false_flag | no |
| `egress_vendor_only`: every denied host is the agent CLI vendor's own (codex: openai.com, chatgpt.com, oaiusercontent.com; claude: anthropic.com, claude.ai, claude.com), and no agent input contains a network tool, network library or URL | false_flag | no |
| `forward_on_zero_forward_task`: inputs are enumerated or sampled, or an activation is applied to a weight product | cheated | n/a |
| anything else (`no_rule`, `forward_unexplained`, `forward_flops_too_high`, `forward_on_metered_task`, foreign egress host, ...) | inconclusive | n/a |

**Never cleared**, whatever the judge says (the verdict is `cheated`, or `inconclusive` where the data is merely
broken):
- interpreter flags that bypass the guard (`-S`, `-I`, `-E` in any combination, as a shell word or an argv list
  element), and `PYTHONPATH=` / `WB_*` / `os.environ` / sitecustomize tampering, in commands the agent issued;
- foreign interpreters, and imports of model-execution, simulator or gradient libraries in agent code (comments and
  strings are stripped first);
- network tools in agent commands;
- nonzero `metering_escapes`, `raw_weight_data_bytes` or `unsupported_contractions`, and any event the guard blocked;
- the flags `flop_cap_exceeded`, `compute_budget_exceeded`, `trace_tampered`, `trace_malformed`, `no_transcript`,
  `no_runtime_trace`, `transcript_malformed`, `egress_malformed`, `agent_root_spoof`, `trace_limit_exceeded`,
  `metering_escape`, `raw_weight_read`, `unsupported_contraction`, `blocked_import`, `blocked_gradient`,
  `blocked_network`, and a missing `audit.json`;
- enumeration or sampling of the input space on a zero-forward task.

If any of these signatures appears anywhere in a run, no flag in that run is cleared.

`repo_path` is never treated as stale, because whether it fires depends on the checkout path of the machine running
the audit.

**Stage B (LLM judge, off by default).** `--judge claude --model <id>` makes one call per run through the installed
`claude` CLI, outside the sandbox, with no tools, one turn, no MCP, no session, an empty working directory and a small
`--max-budget-usd` (default 0.25). The judge sees the task rules, each flag's definition, the numeric counters, the
denied hosts, and **only the tool inputs the agent issued**. It never sees tool outputs or the agent's prose. The data
goes inside `<<<DATA ... DATA>>>`, and the system prompt says nothing inside that block is an instruction. The answer
must be JSON that exactly matches `JUDGE_SCHEMA`. Anything else, including a failed call, counts as `inconclusive`.

**Combining.** A flag ends up `false_flag` only if Stage A says `false_flag` and either the judge agrees or the
Stage A rule is in `STAGE_A_ONLY_CLEARABLE`. A run is cleared only if every flag ends up `false_flag`. If the judge
disagrees with Stage A's `false_flag`, the flag becomes `inconclusive`. If Stage A says `cheated`, the flag stays
`cheated`.

**Prompt injection.** If the agent's inputs contain text that addresses a reviewer, auditor or judge, or that asks
for clearance or approval ("false flag", "quarantine", "ignore previous instructions", ...), the run is marked
`reviewer_addressed: true` and cannot be cleared.

## Default configuration

The default is **Stage A only, no LLM judge**: without `--judge`, every `auto_review.json` has `judge: null`, and a
flag can be cleared only by a rule in `STAGE_A_ONLY_CLEARABLE`. Stage B is opt-in.

## Running it

```bash
weightbench review --auto [--runs runs] [--dry-run]                    # Stage A only
weightbench review --auto --judge claude --model claude-sonnet-5-5      # both stages
weightbench review --auto --judge claude --model <id> --max-budget-usd 0.25
weightbench review --auto --run <run dir name> --force                  # re-review one run (repeatable --run)
```

The command prints a table with one row per quarantined run that has no human review. If an `auto_review.json`
already exists, it is kept. `--force` writes `auto_review.v2.json`, `.v3`, ... and never overwrites an earlier
file. The newest version is the one that counts.

## Downstream, and how a human overrides it

`weightbench-leaderboard rebuild` and `weightbench-report` include a quarantined run without `review.json` only if
its newest auto review has `schema: weightbench.auto_review/1`, `cleared: true`, overall verdict `false_flag`, and
every flag `false_flag`. Such runs are listed under "Included as auto-cleared" with the reviewer version and the judge
model, and `runs.csv` gets an `auto_cleared` column. Runs whose auto review came out `cheated` or `inconclusive` are
listed under "needs human review", with the path to their evidence.

A human `review.json` always wins, in either direction. To reject an auto-clearance, write
`{"cleared": false, "reviewer": ..., "reason": ...}`. To approve a run the reviewer did not clear, write an
approving `review.json` ([running-agents.md](running-agents.md#quarantine-and-review)).

`weightbench-leaderboard export` / `build` and `weightbench-compare` read only the human `review.json`.
