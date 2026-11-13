# Reports, leaderboard and verification

Every score in WeightBench is recomputed from a run's ledger, and every tool on this page only reads run files; none
of them runs an agent or a model. Scoring rules are in [SPEC.md](SPEC.md) §3.

| Command | Purpose |
|---|---|
| `weightbench-report` | aggregate local runs into a scored report |
| `weightbench-leaderboard export / build / rebuild` | shared leaderboard files |
| `weightbench-verify` | recheck runs (and leaderboard records) from their files |
| `weightbench-manifest write / check` | freeze and check task specifications and artifact hashes |
| `weightbench-calibrate` | robustness of the ranking to the scoring constants |
| `weightbench-compare plan / report` | matched access studies (weights + behaviour vs behaviour only) |
| `weightbench rescore` | recompute every `result.json` from its ledger with the current formula |
| `weightbench review --auto` | automated quarantine review ([AUTO_REVIEW.md](AUTO_REVIEW.md)) |

## `weightbench-report`

```bash
weightbench-report
```

Reads every `runs/*/result.json` and writes `summary.json` and `RESULTS.md` to the `results` directory in the repo
root (or to `WB_RESULTS_DIR`), then prints the Markdown table. It takes no arguments. `weightbench bench` calls it at
the end of a batch.

What the report contains:

- per system: the mean S_WB per track with a 95% bootstrap CI over tasks, the **WeightBench score** (mean of the
  track means) with its CI, tie groups from a paired stratified bootstrap, run count, API-equivalent cost and the
  union of audit flags;
- the unpenalized S+ macro average per track;
- the auxiliary `s_wb_v2` values (SPEC §3.2);
- a per-task table (mean S_WB over samples, S+ in parentheses);
- an item analysis per task: the validity gate (reference − baseline ≥ 0.20 and reference ≥ 0.50, from runs of both
  floors), discrimination statistics, saturation and floor flags, and a headline score over valid tasks only.

Rules applied:

- The evaluated suite is every task any system ran. A task a system did not run (or has no admissible run for)
  counts as 0 for that system and is shown as `0.000 (not run)`.
- Invalid runs, `cmd` runs and quarantined runs that no review cleared are excluded from every mean and listed with
  the reason. Auto-cleared runs are listed in their own line.
- Runs at a non-default knob level and `SMOKE-*` runs are ignored.
- Floors are deterministic, so only the latest sample of each (floor, task, seed) is kept. A non-default reference
  variant is reported as its own system, `reference-<variant>`.

## `weightbench-leaderboard`

The leaderboard lives in `leaderboard/` (generated, not tracked); every command creates it if needed.

### `rebuild` (current board)

```bash
weightbench-leaderboard rebuild [--runs runs]
```

Regenerates `leaderboard/LEADERBOARD.md`, `leaderboard.csv` (one row per system × task cell: runs, seeds, samples,
mean/min/max S_WB) and `runs.csv` (every run with its inclusion decision and reason) directly from a runs directory,
its `review.json` files and its `auto_review.json` files. A run is included only if:

- `valid` is true and the agent is not `cmd`;
- it is not quarantined, or a human `review.json` cleared it, or (with no human review) its latest
  `auto_review.json` cleared it (listed separately as auto-cleared);
- it is at the task's canonical knob level;
- it is not stale: when git history is available, a run is stale if its task's class changed after the run started.
  Changes to shared modules (scoring, runner, harness, …) after the start are reported in their own section and do
  not exclude the run.

Cells are means over one system's runs on one task; `rebuild` computes no cross-task score, because systems may have
run different task sets. Use `weightbench-report` for the WeightBench score.

### `export` and `build` (shared records)

```bash
weightbench-leaderboard export --contributor <name> [--runs runs]
weightbench-leaderboard build
```

`export` writes one compact JSON record per finished run to `leaderboard/submissions/<name>/<run_id>.json`. A record
holds the scores, audit flags, quarantine and `review.json` status, the SHA-256 of the run's `ledger.jsonl` and
`transcript.jsonl`, the harness git commit (when available), the host OS and the export time. Non-default levels are
skipped. Contributors can share these records instead of whole run directories.

`build` aggregates every record under `leaderboard/submissions/` (the later export wins if two contributors submit the
same run id) into `LEADERBOARD.md`, `leaderboard.csv` (one row per system) and `runs.csv`. It reads only human
`review.json` status: a quarantined record counts as 0 unless a human cleared it.

Check a record against its run with `weightbench-verify --record` (below).

## `weightbench-verify`

```bash
weightbench-verify runs/<run_id>                       # one run
weightbench-verify --all runs/                         # every run directory with a result.json, summary table
weightbench-verify runs/<run_id> --record leaderboard/submissions/<who>/<run_id>.json
weightbench-verify --record leaderboard/submissions/<who>/<run_id>.json   # run dir = runs/<record run_id>
weightbench-verify --all runs/ --json                  # machine-readable
```

It only reads files. Each check reports `PASS`, `FAIL`, `WARN` or `SKIP`; the exit code is 1 if any check fails
(`WARN` and `SKIP` do not fail).

| Check | What it verifies |
|---|---|
| `ledger_chain` | `ledger.jsonl` exists, every entry parses, and the SHA-256 hash chain is intact (no entry edited, removed or reordered) |
| `result` | `result.json` exists |
| `formula_version` | `result.json` was scored with the current formula; an older version is a `WARN` (run `weightbench rescore`) |
| `scores` | `s_wb`, `s_plus`, `valid`, `n_forward` and `n_probe` in `result.json` equal what `runner.score_from_ledger` recomputes from the ledger |
| `ledger_structure` | exactly one `run_start` (first) whose task, agent, model, seed, sample and variant match `result.json`; a submission before the single, final `evaluation` (or the documented no-submission case); `details` and `d_col` match; for matched access conditions, consistent access-input evidence and only declared behaviour probes |
| `audit` | the quarantine status agrees between `result.json`, `audit.json` and the ledger's `agent_exit`; a trace-limit marker or malformed evidence without quarantine fails unless a human review is recorded. Floor runs without an audit are `SKIP` |
| `record` | with `--record`: the record's ledger and transcript hashes match the files, and its fields match `result.json` |

`bench` and `sweep` run the same verification before reusing a cached run.

## `weightbench-manifest`

A manifest freezes a benchmark version: for every registered task (`SMOKE-*` excluded) its track, tier, full budget,
submission type, probe kinds, audit rules and knob level, and for every built seed the SHA-256 of each public and
private artifact file, plus the harness git commit. Private contents never leave the machine; only their hashes are
recorded.

```bash
weightbench-manifest write --version v0.1 [--seeds 0,1,2]     # -> manifests/v0.1.json
weightbench-manifest check --version v0.1                     # recompute and compare; exit 1 on any difference
weightbench-manifest check --version v0.1 --tasks-from-manifest --require-built
```

- `write --seeds` limits the manifest to those seeds (default: every built seed). Only default-level builds
  (`artifacts/<task>/seed<s>/` with a `.built` marker) are hashed.
- `check` reports changed, missing and extra files of built seeds, changed task specifications, tasks registered
  but not in the manifest, and tasks in the manifest that are no longer registered. Seeds not built on this machine
  are skipped.
- `--tasks-from-manifest` checks only the frozen task ids and rejects unknown ids.
- `--require-built` fails if a frozen task/seed is not built instead of skipping it.

To test that a build is reproducible: `weightbench build --task T --seed S --force`, then `weightbench-manifest check`.

## `weightbench-calibrate`

```bash
weightbench-calibrate --sensitivity
```

Recomputes the leaderboard over local runs with each scoring constant (α, κ, β) at ½×, 1× and 2× (27 settings) and
prints the minimum Kendall τ to the official ranking, whether the top system changes, and the largest change of any
system's mean. Ranking claims should hold on the whole grid.

Without `--sensitivity` the command works on the auxiliary per-track constants in `configs/scoring/*.yaml` (used only
for `s_wb_v2`): `weightbench-calibrate [--runs runs] [--write TR03,TR05]` prints proposed constants and, with
`--write`, overwrites those tracks' config files.

## `weightbench-compare` (matched access studies)

A paired study asks how much weight access helps a given agent: each case runs once under `weights_behavior` and once
under `behavior_only` (see [running-agents.md](running-agents.md#access-conditions---condition)). The plan freezes
subjects, budgets and run ids before any agent runs.

```bash
weightbench-manifest write --version study-1 --seeds 0
weightbench-compare plan --manifest manifests/study-1.json --agent claude --model <model> \
    [--tasks TR04-GO-001,TR05-DEAD-001] [--seeds 0] [--samples 0,1] [--max-turns 200] --output plans/study-1.json

# for every planned case and both arms:
weightbench run --task TR05-DEAD-001 --agent claude --model <model> --seed 0 --sample 0 \
    --condition weights_behavior --access-plan plans/study-1.json
weightbench run --task TR05-DEAD-001 --agent claude --model <model> --seed 0 --sample 0 \
    --condition behavior_only --access-plan plans/study-1.json

weightbench-compare report --plan plans/study-1.json [--runs runs] [--exploratory] [--draws 2000] --output study-1-report.json
```

- `plan` requires a schema-2 manifest, refuses to overwrite an existing output file, and lists every case with both
  run ids.
- `run --access-plan` checks that the run matches a planned case and that the frozen subject and harness source did
  not change during the run.
- `report` verifies the declared runs and compares S_WB and S+ between arms with a bootstrap (`--draws`). It reports
  how many pairs are complete and whether the study is ready; `--exploratory` allows explicitly labelled estimates
  from complete pairs only.

## Build provenance (`.built`)

Each build writes `artifacts/<task>/seed<s>[_L<level>]/.built`, a JSON marker with:

- `source`: a hash of every source file that can change a subject's weights (all of `weightbench/` except modules that
  only launch, audit or report);
- `numeric`: the numeric stack that built it, for example `numpy==… torch==…+cpu`;
- `seconds`: build time, and `level` for non-default levels.

If the source hash no longer matches the current code, the next `build` or `run` rebuilds the artifact instead of
scoring a subject that the current code would not produce. If only the numeric stack differs, the artifact is reused
and a note says which stack built it (different PyTorch builds can train different weights from the same seed).
`--force` always rebuilds.
