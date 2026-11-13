# WeightBench

**Can an AI agent read, fix and write neural-network weights directly, the way an expert reads code?**

WeightBench gives an agent the raw weights of a model and a task (find a backdoor, repair a behaviour, write a
network from scratch, prune a model for hardware) and only a small, metered budget of forward passes. The
agent works in a sandbox, every model call is counted in a hash-chained ledger, and the score is computed from that
ledger, so it can't be faked. 42 tasks in 5 tracks, all generated from a seed.

## Leaderboard (v0.1, preliminary)

| # | Agent | READ | FIX | META | WRITE | SYSTEMS | **Score** |
|---|---|---|---|---|---|---|---|
| 1 | GPT-6.1-sol † | 0.928 | 0.642 | 0.303 | 0.863 | 0.879 | **0.723** |
| 2 | Claude Fable 5.1 † | 1.000 | 0.492 | – | 0.746 | 0.228 | **0.616** |
| 3 | Claude Opus 5.5 | 0.744 | 0.456 | 0.492 | 0.829 | 0.207 | **0.546** |
| 4 | Claude Sonnet 5.5 | 0.656 | 0.451 | 0.500 | 0.569 | 0.225 | **0.480** |
| 5 | Claude Haiku 5.5 | 0.671 | 0.355 | 0.491 | 0.369 | 0.177 | **0.413** |
| 6 | DeepSeek v4-flash | 0.470 | 0.365 | 0.226 | 0.000 | 0.238 | **0.260** |

Score = mean of the five track means of S_WB (0 to 1), seed 0. Quarantined runs count as 0.
† Run in progress: not all tasks are scored yet.

## Install

Linux, macOS, or Windows through WSL2, with Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

- **Windows:** run `wsl --install -d Ubuntu` in an administrator PowerShell, restart, open Ubuntu and do every step
  below inside it (clone into your Linux home, not `C:\`).
- **Ubuntu / WSL:** `sudo apt update && sudo apt install -y git curl bubblewrap`
- **macOS:** `xcode-select --install`. The sandbox uses the built-in `sandbox-exec`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # skip if you have uv
git clone https://github.com/OWNER/weightbench && cd weightbench
bash scripts/quickstart.sh                          # install + score one demo task, no account needed
source .venv/bin/activate
```

The quickstart builds `TR05-DEAD-001` and scores its baseline (0.000) and reference solution (1.000). Details, extras
and troubleshooting (including Ubuntu 24.04's AppArmor setting): [docs/getting-started.md](docs/getting-started.md).

## Run your agent

```bash
bash scripts/install.sh --agent          # agent tool environment + sandbox self-test (once per machine)

# Claude Code (log in once with `claude`), all tracks, 3 samples per task
weightbench bench --agent claude --model claude-sonnet-5-5 --samples 3

# OpenAI Codex CLI (log in once with `codex`), one track
weightbench bench --agent codex --model <model-id> --tracks TR04

# any command-line agent
weightbench bench --agent cmd --model my-model \
    --agent-cmd "my-agent --model {model} --prompt-file {prompt_file}" --allow-host api.example.com
```

Narrow a run with `--tracks TR01,TR04`, `--tiers foundational` or `--tasks 'TR05-*'`; add `--with-floors` to score
the baseline and reference next to your agent. Results land in `runs/<run_id>/` (`result.json`, the transcript, the
audit and the ledger). `weightbench-report` prints a leaderboard for your runs, and `weightbench-verify --all runs/`
re-checks every score from its ledger. Full guide: [docs/running-agents.md](docs/running-agents.md).

## Your keys, your models

WeightBench ships no API keys and never calls a model itself.

- **Agent:** your agent runs with your own login or API key (for example `ANTHROPIC_API_KEY`, or a gateway via the
  CLI's own endpoint variables). The sandbox blocks all network traffic except your agent's own API host and any host
  you add with `--allow-host`.
- **Hugging Face:** tasks built on gated models need your own token. Accept the model licences on huggingface.co
  ([Gemma-3-270M-it](https://huggingface.co/google/gemma-3-270m-it),
  [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)), then run `hf auth login`. The token file
  stays outside the sandbox, so agents can't read it. An exported `HF_TOKEN` is inherited by the agent process, so
  prefer `hf auth login`.
- Language-model tasks need the extras: `uv pip install -e '.[lm]' --torch-backend cpu -c constraints.txt`.

## Tracks

| Track | Tasks | The agent must |
|---|---|---|
| **TR-01 READ** Static diagnostics | 10 | find backdoors, hidden behaviours and planted circuits from the weights |
| **TR-02 FIX** Surgical patching | 13 | repair a flaw (sycophancy, control failures, reasoning loops) with a small weight edit |
| **TR-03 META** Trap verification | 5 | tell a fixable defect from one that a weight edit can't fix |
| **TR-04 WRITE** Weight synthesis | 6 | write a working network (game policies, Go, grafted skills) with no training |
| **TR-05 SYSTEMS** Hardware-aware | 8 | prune or quantize a model while keeping its behaviour |

`weightbench list` prints every task with its track and tier; [docs/tasks.md](docs/tasks.md) lists subject models and
budgets. Scoring, budgets and integrity rules are in [docs/SPEC.md](docs/SPEC.md).

## Hardware

Most tasks run on a CPU laptop. The exceptions:

| Tasks | Need |
|---|---|
| TR02-SYC-001..004, TR03-PAIR-001..005 | Hugging Face token with the Gemma licence accepted |
| TR03-PAIR-001 | about 15 GB disk (Mistral-7B) and the Mistral licence |
| TR02-LOOP-001..004, TR03-PAIR-003/005 | a GPU, or many CPU hours (long thinking-mode generation) |

## Documentation

- [Getting started](docs/getting-started.md): install per OS, `weightbench doctor`, extras, troubleshooting
- [Running agents](docs/running-agents.md): adapters, custom agents, task selection, outputs, quarantine review
- [Tasks](docs/tasks.md): every task with its subject model, budgets and requirements
- [Reports, leaderboard and verification](docs/leaderboard-and-verification.md): `weightbench-report`,
  `weightbench-leaderboard`, `weightbench-verify`, manifests, calibration, access studies
- [Specification](docs/SPEC.md): harness enforcement, tracks, scoring, tiers, knobs and integrity protocols
- [Automated quarantine review](docs/AUTO_REVIEW.md)
- [Adding a task](docs/adding-a-task.md)
- [Contributing](CONTRIBUTING.md) and [security policy](SECURITY.md)

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{weightbench2026,
  title  = {WeightBench},
  author = {WeightBench contributors},
  year   = {2026}
}
```
