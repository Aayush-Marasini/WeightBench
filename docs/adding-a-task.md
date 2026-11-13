# Adding a task

A task is a Python class that subclasses `weightbench.task.Task` and registers itself. The harness handles the
sandbox, metering, ledger and scoring; the task supplies the subject model, the prompt, the probes, the evaluator, a
baseline and a reference solution. Read [SPEC.md](SPEC.md) first: §3 defines how your evaluation becomes a score and
§5 defines seeds and knobs.

## Where the code goes

```
weightbench/tracks/trNN_<family>.py        # the Task classes (kept light: imported for every registry load)
weightbench/tracks/trNN_<family>_lib/      # generators, model code, heavy steps, private helpers
tests/test_trNN_<family>.py                # tests
```

`weightbench.task.load_all()` imports every top-level module in `weightbench/tracks/`, so a module that defines a
`@register`ed class is picked up automatically. Keep heavy imports (torch models, transformers, simulators) inside
methods so that loading the registry stays cheap.

## The contract

```python
from pathlib import Path
from weightbench.task import Budget, Evaluation, Submission, Task, register


@register
class MyTask(Task):
    id = "TR05-FOO-001"                 # unique; <track>-<FAMILY>-<NNN>
    track = "TR05"                      # TR01 .. TR05
    tier = "foundational"               # foundational | intermediate | frontier (SPEC §4)
    submission = "delta_w"              # report | delta_w | refusal_or_delta_w
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=3600)

    def build(self, art_dir: Path, seed: int) -> None: ...
    def prompt(self, art_dir: Path) -> str: ...
    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation: ...
    def baseline(self, art_dir: Path) -> Submission: ...
    def reference(self, art_dir: Path) -> Submission: ...
```

| Member | Required | Purpose |
|---|---|---|
| `id`, `track`, `tier`, `submission`, `budget` | yes | identity and limits (below) |
| `build(art_dir, seed)` | yes | generate the subject model for `seed` at knob level `self.level`, writing `art_dir/public/` and `art_dir/private/` (both already exist) |
| `prompt(art_dir)` | yes | the agent-visible task text, written to `TASK.md`. It must not leak ground truth. |
| `evaluate(art_dir, sub)` | yes | score a submission and return an `Evaluation` |
| `baseline(art_dir)` | yes | the non-agentic floor (SPEC §4) |
| `reference(art_dir)` | yes | a reference solution that clearly beats the baseline |
| `probe_kinds` + `probe(art_dir, kind, payload)` | if the task serves probes | metered observations (below) |
| `base_weights(art_dir)` | for ΔW tasks | the parameters a ΔW applies to; used for edit-cost and L0/rank checks. Returning `None` makes the harness reject a ΔW. |
| `frobenius_reference(art_dir)` | from-zero synthesis | weights that normalise ‖ΔW‖ for the auxiliary edit cost |
| `rescore_s_plus(details)` | optional | recompute S+ from stored measurements without re-running anything |
| `knob_level`, `knob_levels` | optional | difficulty knob (below) |
| `audit_allowed_flags`, `meter_exclude` | optional | audit signals the task's own rules permit (below) |
| `reference_variants` | optional | named reference solutions; the task then implements `reference(art_dir, variant)` |

### `Evaluation`

```python
Evaluation(s_plus=..., d_col=0.0, valid=True, details={...})
```

- `s_plus`: task gain, already clamped to [0, 1] (SPEC §3.1). For write tasks, normalise against a ceiling that does
  not come from your own reference solution.
- `d_col`: collateral damage in [0, 1] on a held-out general evaluation; 0 for read-only tasks.
- `valid=False` for any constraint violation (wrong shapes, broken structural budget, malformed report). An invalid
  submission scores 0. Put the reason in `details["reason"]`.
- `details`: any JSON-serialisable measurements. They go into the ledger and `result.json`.

The runner adds the generic checks itself: it applies the L0/rank budget from `Budget` to ΔW submissions, computes
edit-budget use, counts metered calls from the ledger and combines everything into S_WB (`runner.score_from_ledger`).
Never compute S_WB inside a task.

### `Submission`

`Submission(type, report=None, delta=None)` with `type` in `report`, `delta_w`, `refusal`. A ΔW is a dict from
parameter name to a NumPy array of the base parameter's shape. The agent submits through `wb.py` (`submit-report`,
`submit-delta`, `submit-refusal`); `baseline` and `reference` return the same objects directly.

## Budgets

`Budget` (in `weightbench/task.py`):

| Field | Meaning |
|---|---|
| `n_forward` | metered full rollouts or generations |
| `n_probe` | metered intermediate probes (for example attention maps) |
| `max_l0` | maximum fraction of nonzero ΔW entries (`None` = unconstrained) |
| `max_rank` | alternative: every edited matrix has rank ≤ this. With both set, satisfying either is enough. |
| `wall_clock_s` | agent time limit (also the per-process CPU-time limit on Linux) |
| `flop_cap` | optional hard cap on metered matmul FLOPs per agent process (`None` = trace only) |
| `cpu_seconds` | optional cap on total guarded-process CPU seconds (`None` = record only) |

Structural rules that `Budget` cannot express (a 2:4 mask, zeroing only, a head cap) are checked in `evaluate` and
return `valid=False`.

## Probes and metering

```python
probe_kinds = {"rollout": "forward"}        # kind -> "forward" (counts against n_forward) or "probe" (n_probe)

def probe(self, art_dir, kind, payload):
    ...                                      # run the subject model on the harness side
    return {"result": ...}                   # JSON-serialisable; returned to the agent
```

The Weight API (`weightbench/server.py`) checks the kind, enforces the budget (a request beyond it gets HTTP 429 and a
`probe_rejected` ledger entry), copies any `payload["file"]` out of the agent's workdir before you read it, and writes
a `probe` ledger entry for every call. Your `probe` method is the **only** place an agent can cause the subject model
to run. Do not add any other path that executes a subject model: no helper the agent can call, no unmetered debug
endpoint. Validate payloads strictly (types, lengths, ranges) and raise on bad input; errors are redacted before the
agent sees them.

Long-running model execution in `build`, `probe` or `evaluate` goes through `weightbench.heavy.run_heavy(module, args,
art_dir=...)`, which runs a `python -m` entry point as a subprocess and returns its JSON output.

## Public and private artifacts

- `public/` is copied into the agent's workdir: weights (`.npy`, `.npz`, `.safetensors`), environment source,
  tokenizer metadata, a public `knob.json`. Everything here is visible to the agent.
- `private/` holds ground truth, reference ΔWs, evaluation data, trigger directions and transform maps. It never
  leaves the harness side. The sandbox cannot read the repository or `artifacts/`; keep it that way by never copying
  private data into `public/` and never making private modules importable from the agent's workdir.
- The prompt and probe responses must not reveal private data either. Grading feedback, if any, reports only
  completeness, never the answer.

## Seeds and generators

- `build` must be a pure function of `(seed, level)` and the code: two builds of the same seed must produce identical
  `public/` content. Derive every random choice from `seed` (for example `np.random.default_rng(seed)`).
- Never hard-code layer indices, head numbers or coordinates of the planted structure. Draw them from the seed and
  apply basis permutations or rotations where the function allows it, so that agents cannot memorise them
  (SPEC §5).
- Pin every external checkpoint and dataset by revision hash, and never download at import time; download inside
  `build`.
- The harness writes a `.built` marker with a hash of the generating sources, so changing generator code rebuilds
  existing artifacts automatically.

## Difficulty knob

Each track has one knob (`weightbench/knobs.py`). To make a task's build take a level:

1. Set `knob_level` (the registered default) and/or `knob_levels` (the explicitly supported levels; others are
   rejected). With `knob_level = None` and a `knob_levels` tuple, an omitted `--level` keeps the unversioned default
   build.
2. Read `self.level` in `build` and scale the track's quantity to `knobs.value(knob, level)`.
3. Write a public `knob.json` (`knobs.describe(track, level)`) for explicit levels and keep selected sites private.
4. Add the task id to the knob's `wired` tuple in `weightbench/knobs.py`.

Reject an unsupported level before any expensive work.

## Audit permissions

If the task's rules allow something the audit would otherwise flag, declare it instead of weakening the audit:

- `audit_allowed_flags = ("forward_pass_suspect",)`: these flags are still reported in `audit.json` but do not
  quarantine.
- `meter_exclude = ("donor.npz",)`: public weight files the prompt explicitly lets the agent evaluate; their shapes
  are left out of the runtime forward-chain detection.

State the permission in the prompt, so agents and reviewers know exactly what is allowed.

## Access conditions

To support the matched `weights_behavior` / `behavior_only` conditions, add an access profile for the task family to
`PROFILES` in `weightbench/conditions.py`: which public files are metadata, which are weights, and which probe kinds
are behaviour probes. Without a profile, non-`full` conditions are rejected for the task.

## Tests

Every task needs tests that run on a laptop CPU in a few minutes, without downloading large models (mark anything that
needs a 0.8B+ model or a GPU with `@pytest.mark.gpu`):

- **Determinism:** two builds at the same seed give identical public content.
- **Baseline and reference:** both produce valid submissions, and the reference beats the baseline by a clear margin.
- **Budget enforcement:** a probe beyond `n_forward` / `n_probe` is rejected, and a zero-budget task serves no probe.
- **Leakage:** no private artifact appears in `public/` (by name or content), and the prompt does not contain ground
  truth.
- **Scoring edge cases:** for ΔW tasks, ΔW = 0, invalid shapes, and edits that break the structural budget.
- **Knob:** supported levels build, unsupported levels are rejected before any work, and the default build is
  unchanged.

`tests/test_registry_contract.py` already runs the determinism, budget and leakage checks over every registered task (builds
that need heavy compute or a missing extra are skipped there), so a new task is covered as soon as it registers. Its
leakage check also expects a test file that names your task to state the sandbox/private rule. Look at
`tests/test_tr05_dead.py` for a complete example of task-specific tests.

## Checklist

- [ ] `id`, `track`, `tier`, `submission`, `budget` set; the id is unique
- [ ] `build` is deterministic in the seed and writes only non-secret files to `public/`
- [ ] every model execution an agent can trigger goes through `probe_kinds` / `probe`
- [ ] `baseline` and `reference` implemented; the reference clearly beats the baseline
- [ ] the ceiling (if any) is independent of the reference solution
- [ ] external checkpoints pinned by revision
- [ ] tests for determinism, budget enforcement, leakage and scoring edge cases
- [ ] `pytest -m "not gpu"` passes
- [ ] [tasks.md](tasks.md) and SPEC §2 updated
