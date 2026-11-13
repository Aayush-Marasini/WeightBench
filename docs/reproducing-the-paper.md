# Reproducing the paper's agent study

The leaderboard in the [README](../README.md) and the agent-results table of the paper come from one agent study,
run on 2026-11-11 to 13. This page records how it was run, what changed in the code since, and how to recompute or
rerun it.

## What was run

| | |
|---|---|
| Tasks | all 42 registry tasks, default difficulty level, seed 0 |
| Models | Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5; DeepSeek v4-flash; GPT-6.1-sol (`openai/gpt-6.1-sol`, not `-pro`) |
| Samples | 2 per task for the Claude models, 1 for DeepSeek and GPT-6.1-sol |
| Agent scaffold | `--agent claude` (Claude Code, headless) for every model; 10 GPT-6.1-sol runs used `--agent codex` |
| Turn cap | 200 (the default) for Claude; 80 (`--max-turns 80`) for DeepSeek and GPT-6.1-sol |
| Reasoning effort | none set: each provider's default |
| Not run | GPT-6.1-sol on the four reasoning-loop tasks (TR02-LOOP-001..004); Claude Fable 5.1 is still being evaluated |

DeepSeek and GPT-6.1-sol were reached through [OpenRouter](https://openrouter.ai) with the Claude Code CLI pointed at
the gateway:

```bash
export ANTHROPIC_BASE_URL=https://openrouter.ai/api ANTHROPIC_AUTH_TOKEN=<your OpenRouter key>
unset ANTHROPIC_API_KEY
export WB_SKIP_AGENT_PREFLIGHT=1   # the login preflight's $0.05 budget is priced at Claude rates for unknown models
weightbench run --task TR01-BKD-001 --agent claude --model deepseek/deepseek-v4-flash --sample 62 \
    --allow-host openrouter.ai --max-turns 80
```

OpenRouter reserves credit for every request in flight, so keep at most three or four such runs in parallel.

**Hardware.** CPU tasks ran on CPU machines. The seven GPU tasks (TR02-LOOP-001..004, TR03-PAIR-001/003/005) ran on
one NVIDIA L4 (24 GB) with `WB_TR02_BATCH=12`, which sets the generation batch of the loop and trap evaluations; the
default of 64 does not fit in 24 GB, and the batch size can change greedy generations slightly. Run one GPU job at a
time: when several shared the card, some probes and evaluations failed with out-of-memory errors, which the agent saw
as an internal evaluator error. The `flash-linear-attention` and `causal_conv1d` kernels were not installed; the
reference implementations were used.

## How a score was counted

- Each task's score is the mean over the counted runs of a model; a track's score is the mean over its tasks; the
  score in the README is the mean of the five track means.
- The counted runs are the first runs in sample order that the agent finished: 2 for Claude, 1 for the others.
  A run that ends without a valid submission, including one that reaches the turn cap, counts as 0.
- Infrastructure failures do not count and were re-run under a fresh sample id: the agent CLI crashing or not
  starting, an exhausted API credit, a full disk, or an evaluation that crashed before writing `result.json`.
  This is why sample ids run from 62 upwards with gaps.
- A quarantined run counts as 0 until a reviewer clears it. None has been reviewed, so every score is a lower bound.

## Recomputing the table

[`data/agent-study-v1.csv`](data/agent-study-v1.csv) has one row per counted run (312 rows): task, track, model,
sample id, S_WB, S+, validity, quarantine, audit flags and scaffold. Quarantined runs and runs without a valid
submission already carry `s_wb = 0`. To recompute the per-track means and both macro scores:

```bash
python - <<'EOF'
import csv, collections
from statistics import mean
runs = list(csv.DictReader(open("docs/data/agent-study-v1.csv")))
per_task = collections.defaultdict(list)
for r in runs:
    per_task[(r["model"], r["track"], r["task"])].append(float(r["s_wb"]))
for model in sorted({r["model"] for r in runs}):
    for label, keep in (("42 tasks", lambda t: True), ("38 tasks", lambda t: "LOOP" not in t)):
        tracks = collections.defaultdict(list)
        for (m, track, task), v in per_task.items():
            if m == model and keep(task):
                tracks[track].append(mean(v))
        n = sum(len(v) for v in tracks.values())
        print(f"{model:28s} {label}: macro {mean(mean(v) for v in tracks.values()):.3f} over {n} tasks",
              {k: round(mean(v), 3) for k, v in sorted(tracks.items())})
EOF
```

## Differences between the study build and this release

The study ran on an internal build of the harness (build fingerprint `54afacd2ae21510a`). This release (v0.1.0)
contains the same tasks, scoring and audit, with these behavioural changes since:

- The agent prompt now states the turn cap and asks the agent to submit early (`runner.turn_cap_note`).
- The Weight API rejects a ΔW submission for a task that expects a report.
- TR05-SPARSE and TR05-OUTLIER builds work on machines with a GPU (the study built them on CPU).
- The TR02-LOOP-004 reference search is wider (more edit strengths and sites, a separate development pool).
- TR02-CTRL builds skip a drawn flaw that no start state can reach.
- MinAtar (TR04-GAME) references at non-default delay levels are delay-compensating circuits.

The first change can affect agent scores; the others change references or non-default levels. A rerun on this
release is therefore a new measurement, not a replay of the study.

## Rerunning

```bash
uv pip install -e '.[lm,dev,tr01]' -c constraints.txt     # the pinned versions are those of the study
bash scripts/install.sh --agent
weightbench bench --agent claude --model claude-sonnet-5-5 --sample 62 --samples 2
```

Use a fresh `--sample` range for reruns; existing run ids are never overwritten. `weightbench-verify --all runs/`
re-checks every score from its ledger, and `weightbench-report` prints the per-track table for your runs.
