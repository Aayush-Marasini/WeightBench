# Task reference

WeightBench v0.1 registers 42 tasks. This table is derived from the task classes in `weightbench/tracks/*.py`
(`id`, `tier`, `submission`, `budget`). `weightbench list` prints the live registry (id, track, tier), and
[SPEC.md](SPEC.md) §2 describes each family.

Column notes:

- **Submission:** `report` (JSON report via `wb.py submit-report`), `delta_w` (a ΔW `.npz` via `wb.py submit-delta`),
  or `refusal_or_delta_w` (TR-03: a per-case decision, ΔW and/or refusal).
- **Fwd / Probe:** the metered budgets `Budget.n_forward` and `Budget.n_probe`. A request beyond the budget is
  rejected by the Weight API. 0 means the task serves no probe of that kind.
- **Edit budget:** `Budget.max_l0` (fraction of nonzero ΔW entries) and/or `Budget.max_rank`; satisfying either is
  enough. "–" means no L0/rank budget (the task may enforce a structural rule instead, such as a 2:4 mask).
- **Wall clock:** `Budget.wall_clock_s`, the agent's time limit (also the per-process CPU-time limit on Linux).
- **Needs:** `lm` = the `[lm]` extra; `HF: Gemma` / `HF: Mistral` = a Hugging Face token with that model's licence
  accepted; `GPU` = long thinking-mode generation, practical only on a GPU; `tr01` / `tr02ctrl` = those extras.
  Empty means CPU with the core install.

## TR-01 READ: static diagnostics

| Task | Tier | Subject model | Submission | Fwd | Probe | Edit budget | Wall clock | Needs |
|---|---|---|---|---|---|---|---|---|
| TR01-BEH-001 | foundational | MinAtar policy MLPs (which game?) | report | 0 | 0 | read-only | 1 h | |
| TR01-BEH-002 | intermediate | 12×12 DDPM shape models (which shape?) | report | 0 | 0 | read-only | 1 h | |
| TR01-BEH-003 | intermediate | 16-input Boolean-function MLPs (which function?) | report | 0 | 0 | read-only | 1 h | |
| TR01-ESC-001 | foundational | 7.2M char-level Transformer, embeddings stripped (training language) | report | 0 | 0 | read-only | 1 h | tr01 (corpus build) |
| TR01-ESC-002 | foundational | same family (induction heads) | report | 0 | 0 | read-only | 1 h | tr01 (corpus build) |
| TR01-ESC-003 | foundational | same family (previous-token heads) | report | 0 | 0 | read-only | 1 h | tr01 (corpus build) |
| TR01-BKD-001 | foundational | MNIST MLP with a backdoor (find the trigger) | report | 0 | 0 | read-only | 1 h | |
| TR01-BKD-002 | foundational | small word-level GPT with a backdoor | report | 0 | 0 | read-only | 1 h | |
| TR01-DWF-001 | foundational | small GPT, four checkpoint deltas (what was learned?) | report | 0 | 0 | read-only | 1 h | |
| TR01-DWF-002 | foundational | small GPT, one checkpoint delta (memorised text) | report | 0 | 0 | read-only | 1 h | |

## TR-02 FIX: surgical patching

| Task | Tier | Subject model | Submission | Fwd | Probe | Edit budget | Wall clock | Needs |
|---|---|---|---|---|---|---|---|---|
| TR02-CTRL-001 | foundational | 171.6K Kinetix actor, car stuck on a ramp | delta_w | 50 (rollouts) | 0 | L0 ≤ 4% | 1 h | tr02ctrl |
| TR02-CTRL-002 | foundational | 9.9K MountainCar policy | delta_w | 50 (rollouts) | 0 | L0 ≤ 4% | 1 h | |
| TR02-CTRL-003 | foundational | 17.2K Pendulum policy | delta_w | 50 (rollouts) | 0 | L0 ≤ 4% | 1 h | |
| TR02-CTRL-004 | foundational | 67.6K CartPole policy | delta_w | 50 (rollouts) | 0 | L0 ≤ 4% | 1 h | |
| TR02-LOOP-001 | intermediate | Qwen3.5-0.8B, reasoning loops (GSM8K) | delta_w | 50 (generations) | 20 (activations) | L0 ≤ 0.05% or rank ≤ 4 | 3 h | lm, GPU |
| TR02-LOOP-002 | intermediate | Qwen3.5-0.8B (MATH-500 levels 1–3) | delta_w | 50 | 20 | L0 ≤ 0.05% or rank ≤ 4 | 3 h | lm, GPU |
| TR02-LOOP-003 | intermediate | Qwen3.5-0.8B (IFEval) | delta_w | 50 | 20 | L0 ≤ 0.05% or rank ≤ 4 | 3 h | lm, GPU |
| TR02-LOOP-004 | intermediate | Qwen3.5-0.8B (MATH-500 levels 4–5), single rank-1 edit | delta_w | 50 | 20 | L0 ≤ 0.05% or rank ≤ 1 | 3 h | lm, GPU |
| TR02-SYC-001 | intermediate | Gemma-3-270M-it, adopts a wrong user claim | delta_w | 0 | 16 (attention) | L0 ≤ 0.05% or rank ≤ 4 | 1.5 h | lm, HF: Gemma |
| TR02-SYC-002 | intermediate | Gemma-3-270M-it, caves to pushback | delta_w | 0 | 16 | L0 ≤ 0.05% or rank ≤ 4 | 1.5 h | lm, HF: Gemma |
| TR02-SYC-003 | intermediate | Gemma-3-270M-it, wrong arithmetic claims (held-out domain) | delta_w | 0 | 16 | L0 ≤ 0.05% or rank ≤ 4 | 1.5 h | lm, HF: Gemma |
| TR02-SYC-004 | intermediate | Gemma-3-270M-it, held-out phrasings, head-only edit | delta_w | 0 | 16 | L0 ≤ 0.05% or rank ≤ 4; at most 4 heads | 1.5 h | lm, HF: Gemma |
| TR02-SUPER-001 | foundational | synthetic superposition network | delta_w | 0 | 0 | – | 1 h | |

## TR-03 META: trap verification

Each task pairs a trap with a TR-02 decoy (see [SPEC.md](SPEC.md) §2). The forward budget is shared by both cases.

| Task | Tier | Trap model / decoy | Submission | Fwd | Probe | Edit budget | Wall clock | Needs |
|---|---|---|---|---|---|---|---|---|
| TR03-PAIR-001 | frontier | Mistral-7B-v0.3 / TR02-SYC-001 | refusal_or_delta_w | 10 (generations) | 0 | decoy: TR-02 budget | 3 h | lm, HF: Gemma + Mistral, ~15 GB disk |
| TR03-PAIR-002 | intermediate | Gemma-3-270M-it / TR02-SYC-002 | refusal_or_delta_w | 10 | 0 | decoy: TR-02 budget | 3 h | lm, HF: Gemma |
| TR03-PAIR-003 | intermediate | Gemma-3-270M-it / TR02-LOOP-001 | refusal_or_delta_w | 10 | 0 | decoy: TR-02 budget | 3 h | lm, HF: Gemma, GPU |
| TR03-PAIR-004 | intermediate | Gemma-3-270M-it / TR02-SYC-003 | refusal_or_delta_w | 10 | 0 | decoy: TR-02 budget | 3 h | lm, HF: Gemma |
| TR03-PAIR-005 | intermediate | Gemma-3-270M-it / TR02-LOOP-002 | refusal_or_delta_w | 10 | 0 | decoy: TR-02 budget | 3 h | lm, HF: Gemma, GPU |

## TR-04 WRITE: weight synthesis

| Task | Tier | Subject model | Submission | Fwd | Probe | Edit budget | Wall clock | Needs |
|---|---|---|---|---|---|---|---|---|
| TR04-GAME-001 | foundational | all-zero MLP `600 → 32 → 32 → 6`, MinAtar Breakout | delta_w | 0 | 0 | – (from zero) | 1 h | |
| TR04-GAME-002 | foundational | same, MinAtar Seaquest | delta_w | 0 | 0 | – (from zero) | 1 h | |
| TR04-GAME-003 | foundational | same, MinAtar Asterix | delta_w | 0 | 0 | – (from zero) | 1 h | |
| TR04-GO-001 | foundational | 9×9 Go policy MLP `243 → 512×6 → 82` (~1.48M) | delta_w | 50 (games) | 0 | – (from zero) | 3 h | |
| TR04-GRAFT-001 | foundational | Transformer host + donor MLP (circuit graft) | delta_w | 20 (checks) | 0 | L0 ≤ 10% | 3 h | |
| TR04-HORIZON-001 | foundational | synthetic delayed-payoff chain | delta_w | 0 | 0 | – | 1 h | |

## TR-05 SYSTEMS: hardware-aware

| Task | Tier | Subject model | Submission | Fwd | Probe | Edit budget | Wall clock | Needs |
|---|---|---|---|---|---|---|---|---|
| TR05-SPARSE-001 | intermediate | Qwen3.5-0.8B, 2:4 on the MLP of layer 23 | delta_w | 0 | 0 | 2:4 mask | 3 h | lm |
| TR05-SPARSE-002 | intermediate | Qwen3.5-0.8B, 2:4 on the MLPs of layers 20–23 | delta_w | 0 | 0 | 2:4 mask | 3 h | lm |
| TR05-SPARSE-003 | intermediate | Qwen3.5-0.8B, 2:4 on every MLP | delta_w | 0 | 0 | 2:4 mask | 3 h | lm |
| TR05-SPARSE-004 | intermediate | Qwen3.5-0.8B, 2:4 on every attention input projection | delta_w | 0 | 0 | 2:4 mask | 3 h | lm |
| TR05-DEAD-001 | foundational | ReLU MLP over `{0,1}^16`, 1 hidden layer (width 128) | delta_w | 0 | 0 | zeroing only | 2 h | |
| TR05-DEAD-002 | foundational | ReLU MLP over `{0,1}^16`, 3 hidden layers (width 96) | delta_w | 0 | 0 | zeroing only | 2 h | |
| TR05-OUTLIER-001 | intermediate | Qwen3.5-0.8B MLPs, simulated W8A6 | delta_w | 0 | 0 | scale migration | 3 h | lm |
| TR05-OUTLIER-002 | intermediate | Qwen3.5-0.8B MLPs, simulated W8A8 | delta_w | 0 | 0 | scale migration | 3 h | lm |

## Reference variants

Some tasks offer more than one named reference solution. The first is the default; choose another with
`weightbench run --task T --agent reference --variant V`.

| Tasks | Variants |
|---|---|
| TR05-DEAD-001/002 | `ibp` (default), `crown` |
| TR05-SPARSE-001..004 | `dev_selected` (default), `datafree_devsel`, `propagated_obs`, `sparsegpt_calib_privileged` |
| TR05-OUTLIER-001/002 | `dev_selected` (default), `datafree_devsel`, `propagated_clip`, `calib_devsel_privileged` |

For TR05-DEAD the default can also be set with the environment variable `WB_DEAD_REFERENCE`.
