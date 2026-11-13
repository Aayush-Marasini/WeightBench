# Contributing to WeightBench

Thanks for helping. WeightBench is a benchmark, so correctness and integrity matter more than features: a change
that makes a score easier to fake or harder to reproduce is a regression even if every test passes.

## Development install

Linux, macOS or WSL2, Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/OWNER/weightbench && cd weightbench
bash scripts/install.sh --dev                 # ./.venv with the [dev] extra
source .venv/bin/activate
uv pip install -e '.[lm,dev,tr01]' --torch-backend cpu -c constraints.txt   # what CI installs
bash scripts/install.sh --agent               # only if you work on the sandbox or agent adapters
```

See [docs/getting-started.md](docs/getting-started.md) for OS-specific notes (bubblewrap, AppArmor on Ubuntu 24.04).

## Tests

```bash
pytest -m "not gpu"                           # the CPU suite, as in CI
pytest tests/test_tr05_dead.py                # one file
bash scripts/ci_sandbox_e2e.sh                # end-to-end agent path through the real sandbox (needs --agent setup)
```

- Tests use toy tensors and must not download large models or need a GPU. Mark tests that need a 0.8B+ model or a
  GPU with `@pytest.mark.gpu`.
- Every scoring function needs unit tests with hand-computed values, including edge cases (ΔW = 0, negative gain
  clamped to 0, the from-zero synthesis base).
- CI runs the CPU suite on Ubuntu 24.04 and Windows, a clean quickstart install, and the sandbox end-to-end check.

## Code style

- Python ≥ 3.11 with `from __future__ import annotations`; type hints on public functions.
- NumPy and PyTorch for numerics; no new dependencies without a clear need. Optional dependencies go into an extra in
  `pyproject.toml`.
- Keep imports in `weightbench/tracks/*.py` light; heavy imports go inside methods.
- Comments and docstrings explain what the code does and why, in the present tense.

## Integrity rules

These protect the benchmark. A pull request that breaks one will not be merged.

1. **No unmetered model paths.** Any code path that runs a subject model on behalf of an agent goes through the
   metered probe engine (`Task.probe` via the Weight API) and writes a ledger entry. No debug shortcuts.
2. **No answer leakage.** Ground truth, reference ΔWs, evaluation data and transform maps live in a task's
   `private/` artifacts and must never be readable or importable from the sandbox. Add a test that proves it.
3. **The ledger is append-only.** No update or delete APIs; scores are computed from ledger entries only.
4. **Seeds come from the generator.** Never hard-code layer indices or coordinates into a task.
5. **Pin model revisions.** External checkpoints and datasets are pinned by revision hash and downloaded in `build`,
   never at import time.
6. **Do not weaken the audit to make a run pass.** If a task's rules permit something the audit flags, declare it
   with `Task.audit_allowed_flags` / `Task.meter_exclude` and say so in the prompt.

The specification in [docs/SPEC.md](docs/SPEC.md) is the source of truth for tracks, budgets and scoring. A change
that alters the specification updates `docs/SPEC.md` in the same pull request and explains why in the description.

## Adding a task

Follow [docs/adding-a-task.md](docs/adding-a-task.md). A task is complete when it has a deterministic generator, a
baseline, a reference that clearly beats it, budget-enforcement and leakage tests, and an entry in
[docs/tasks.md](docs/tasks.md).

## Pull requests

- One logical change per pull request; small pull requests are reviewed faster.
- Changes to shared modules (`weightbench/harness/`, `scoring.py`, `ledger.py`, `runner.py`, `server.py`) should be
  minimal and backward compatible: existing ledgers must still verify (`weightbench-verify --all runs/`).
- Describe how you tested the change.

## Security issues

Do not open public issues for sandbox escapes, metering bypasses or ways to fake a score. See
[SECURITY.md](SECURITY.md).
