"""TR-01 Behavioral & Task Identification (TR01-BEH-001..003), SPEC §2 TR-01.

Each task is ONE agent run in which the agent labels several unknown subject models from
their weights alone (N_forward = 0). S+ = fraction labelled correctly.

Heavy generation (training, rollouts, sampling) lives in ``tr01_behavioral_lib.build`` and
runs through ``weightbench.heavy.run_heavy`` (locally or on the cluster).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from weightbench import scoring
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr01_behavioral_lib import common as C

_MOD = "weightbench.tracks.tr01_behavioral_lib.build"

_NO_FORWARD = """## Rules
- `N_forward = 0`, `N_probe = 0`: there are no probes; `wb.py` is only for submitting.
- **Static analysis only.** Do NOT evaluate the networks on any input, including inputs you
  construct yourself (no forward passes, no ReLU/SiLU applied to activations, no sampling loops).
  Reading, slicing, reshaping, plotting-as-text, SVD/eigen-analysis, and multiplying or
  composing weight matrices with each other is allowed.
- Do not try to read files outside your working directory.
"""


def _score_text(budget: Budget) -> str:
    e_budget = budget.n_forward + budget.n_probe
    if e_budget == 0:
        return f"""## Official score (SPEC §3)
`S_WB = S+`: this read-only task has no edit budget and `E_budget = {e_budget}`, so the compute
factor is 1. (The official compute constant is `beta = {scoring.V3_BETA:g}`.)
"""
    return f"""## Official score (SPEC §3)
`S_WB = S+ * (1 - beta * E/E_budget)`, where `beta = {scoring.V3_BETA:g}` and
`E_budget = {e_budget}` (`{budget.n_forward}` forward + `{budget.n_probe}` probe calls).
"""


def _models(art_dir: Path) -> list[str]:
    return sorted((p.stem for p in (art_dir / "public" / "models").glob("*.npz")),
                  key=lambda s: int(s.rsplit("_", 1)[1]))


def _report_section(names: list[str], cands: list[str], what: str) -> str:
    ex = ", ".join(f'"{n}": "<{what}>"' for n in names[:2])
    return f"""## Submission
Write `report.json` with exactly one entry per model, mapping the file stem to a {what} name:

```json
{{{ex}, ...}}
```

Allowed values: {", ".join(f'`{c}`' for c in cands)}. Every model is labelled independently: labels
may repeat and some candidates may not appear at all. Missing or invalid entries count as wrong.
Score = fraction of models labelled correctly. Submit with

    python wb.py submit-report report.json

You may resubmit; the last submission counts.
"""


class _BehTask(Task):
    track = "TR01"
    submission = "report"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=3600)
    key: ClassVar[str]
    candidates: ClassVar[list[str]]
    knob_levels = (0, 1)

    def build(self, art_dir: Path, seed: int) -> None:
        from weightbench import heavy

        extra = [] if self.level is None else ["--level", str(self.level)]
        heavy.run_heavy(_MOD, ["--task", self.key, "--art", str(art_dir.resolve()), "--seed", str(seed),
                               "--procs", "6", "--threads", "8", *extra],
                        art_dir, out_name="build_out.json", timeout=2 * 3600, gpu=False)
        if os.environ.get("WB_HEAVY", "local") != "local":
            heavy.pull_dir(art_dir, "public")
            heavy.pull_dir(art_dir, "private")

    def _gt(self, art_dir: Path) -> dict[str, str]:
        return C.read_json(art_dir / "private" / "gt.json")

    def _scrambled(self, art_dir: Path) -> str:
        path = art_dir / 'public/knob.json'
        count = 2 if not path.exists() else C.read_json(path)['value']
        return 'one hidden layer' if count == 1 else 'both hidden layers'

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        report = sub.report if sub.type == "report" else None
        acc, details = C.score_report(report, self._gt(art_dir), self.candidates)
        if sub.type != "report":
            details["reason"] = f"submission type {sub.type!r} is not a report"
        return Evaluation(s_plus=acc, details=details)

    def baseline(self, art_dir: Path) -> Submission:
        return Submission("report", report=C.spectral_baseline(art_dir, _models(art_dir)))

    def reference(self, art_dir: Path) -> Submission:
        return Submission("report", report=C.read_json(art_dir / "private" / "reference.json")["labels"])


# ----------------------------------------------------------------------------------------------
@register
class GameIdentification(_BehTask):
    id = "TR01-BEH-001"
    tier = "foundational"
    key = "beh001"
    candidates = ["breakout", "asterix", "freeway", "seaquest", "space_invaders"]

    def prompt(self, art_dir: Path) -> str:
        names = _models(art_dir)
        enc = C.read_json(art_dir / "public" / "encoding.json")
        rows = []
        for g in self.candidates:
            by_c: dict[int, list[str]] = {}
            for ch, c in enc["channel_map"][g].items():
                by_c.setdefault(c, []).append(ch)
            rows.append(f"| {g} | " + " | ".join(", ".join(by_c.get(c, [])) or "(always 0)" for c in range(4)) + " |")
        table = "\n".join(rows)
        return f"""# TR01-BEH-001: Which game does each policy play?

`models/` contains {len(names)} policy networks ({", ".join(f"`{n}.npz`" for n in names)}). Each one was
trained (behaviour cloning of a planning expert, with DAgger) to play ONE of five MinAtar games:
{", ".join(f"`{c}`" for c in self.candidates)}. The full MinAtar source of every game is in `games/<game>.py`.
Your job: decide, from the weights alone, which game each policy plays.

## Architecture (identical for every policy, PyTorch convention `y = W x + b`)
    h1     = relu(W1 @ x + b1)      W1: [40, 400]  b1: [40]
    h2     = relu(W2 @ h1 + b2)     W2: [24, 40]   b2: [24]
    logits = W3 @ h2 + b3           W3: [6, 24]    b3: [6]      action = argmax(logits)
Each `.npz` holds `W1 b1 W2 b2 W3 b3` (float32). Hidden units in {self._scrambled(art_dir)} of every model were randomly
permuted and positively rescaled (function-preserving), so unit indices and per-unit scales mean
nothing across models. Every policy also had its own randomly drawn training recipe (number of
hidden units actually trained, weight decay, epochs, data fraction, learning rate, init scale),
drawn independently of the game, so overall weight norms and spectra do not identify the game.
Untrained width was filled with inactive units that never fire on any binary input (an `h1`
unit whose bias is below minus the sum of its positive weights; an `h2` unit whose incoming
weights and bias are all <= 0); their outgoing weights are random and irrelevant. A random
vector was added to every row of `W3` (and a constant to `b3`), which changes no softmax/argmax.

## Observation encoding (shared by all games, so the input width does not reveal the game)
`x` is a binary vector of length 400: `x[c*100 + row*10 + col]` for common channel `c` in 0..3,
`row` 0 (top) .. 9 (bottom), `col` 0 (left) .. 9 (right). Each game's native MinAtar channels are
OR-ed into the common channels as follows (space_invaders' `alien_right` channel is dropped; the
aliens move right exactly when `alien_left` is empty):

| game | c=0 | c=1 | c=2 | c=3 |
|---|---|---|---|---|
{table}

## Actions (MinAtar full action set; output index = action id)
`0 = n (no-op), 1 = l (left), 2 = u (up), 3 = r (right), 4 = d (down), 5 = f (fire)`.
Every policy has all 6 outputs. Actions that have no effect in a game behave like no-op there
(see each game's `act`); during training, the expert's probability mass was split evenly over
actions that are equivalent in that game state.

{_NO_FORWARD}
{_score_text(self.budget)}
{_report_section(names, self.candidates, "game")}"""


# ----------------------------------------------------------------------------------------------
@register
class DiffusionShapeIdentification(_BehTask):
    id = "TR01-BEH-002"
    tier = "intermediate"
    key = "beh002"
    candidates = ["circle", "square", "triangle", "diamond", "plus", "x_cross", "line", "filled_disk"]

    def prompt(self, art_dir: Path) -> str:
        names = _models(art_dir)
        return f"""# TR01-BEH-002: Which shape does each diffusion model draw?

`models/` contains {len(names)} tiny DDPM denoisers ({", ".join(f"`{n}.npz`" for n in names)}). Each one was
trained on 12x12 grayscale images of exactly ONE shape class from:
{", ".join(f"`{c}`" for c in self.candidates)}.
Shapes are drawn with random position jitter (about +-1.2 px), size and stroke width; outlines
(circle, square, triangle, diamond) and strokes (plus, x_cross, line) are ~1 px wide;
`square` is axis-aligned and `diamond` is a square rotated 45 degrees (both +-15 deg); `x_cross`
is `plus` rotated 45 degrees; `line` is a single segment at a uniformly random angle;
`filled_disk` is solid. `shape_examples.npz` holds 16 example training images per class (key =
class name, array [16, 12, 12], values in [-1, 1], -1 = background, +1 = ink).
Your job: decide, from the weights alone, which shape class each denoiser was trained on.

## Architecture (identical for every model, PyTorch convention `y = W x + b`)
    x0_hat = W3 @ silu(W2 @ silu(W1 @ concat(x_t, temb(t)) + b1) + b2) + b3
    W1: [64, 160]  b1: [64]   W2: [64, 64]  b2: [64]   W3: [144, 64]  b3: [144]
    silu(z) = z * sigmoid(z)
- `x_t`: the noisy image flattened row-major (`x_t[row*12 + col]`, 144 values; clean images in [-1, 1]).
- `temb(t)` (16 values) = `[sin(t*f_0..7), cos(t*f_0..7)]`, `f_k = 100**(-k/8)`, integer `t` in 0..99.
  So `W1[:, :144]` acts on pixels and `W1[:, 144:]` on the timestep embedding.
- DDPM with **clean-image (x0) prediction**: training pairs are `x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps`,
  eps ~ N(0, I), and the network is trained with MSE to output `x_0` (so `x0_hat` lives in image space;
  the implied noise estimate is `(x_t - sqrt(abar_t) x0_hat) / sqrt(1 - abar_t)`).
  `schedule.npz` has `betas` (linear 1e-3 .. 0.15, T = 100) and `alphas_cumprod` (= abar).
- Hidden units in {self._scrambled(art_dir)} of every model were randomly permuted (function-preserving).

{_NO_FORWARD}
{_score_text(self.budget)}
{_report_section(names, self.candidates, "shape")}"""


# ----------------------------------------------------------------------------------------------
@register
class AlgorithmicFunctionIdentification(_BehTask):
    id = "TR01-BEH-003"
    tier = "intermediate"
    key = "beh003"
    candidates = ["parity_subset", "majority", "popcount_mod3", "compare_halves", "count_halves",
                  "first_one_even", "dnf", "value_mod3", "run_of_three", "parity_subset_b", "dnf_b"]

    def prompt(self, art_dir: Path) -> str:
        names = _models(art_dir)
        defs = C.read_json(art_dir / "public" / "candidates.json")["definitions"]
        lst = "\n".join(f"- `{k}`: {defs[k]}" for k in self.candidates)
        return f"""# TR01-BEH-003: Which Boolean function does each network compute?

`models/` contains {len(names)} MLPs ({", ".join(f"`{n}.npz`" for n in names)}). Each one was trained to
(near) 100% accuracy on all 2^16 inputs to compute ONE of the candidate functions below.
Your job: decide, from the weights alone, which function each network computes.

## Input and architecture (identical for every network, PyTorch convention `y = W x + b`)
Input: an integer `n` in [0, 65536) encoded as `x[i] = (n >> i) & 1` for `i = 0..15` (float 0/1).

    h1    = relu(W1 @ x + b1)      W1: [64, 16]  b1: [64]
    h2    = relu(W2 @ h1 + b2)     W2: [64, 64]  b2: [64]
    logit = W3 @ h2 + b3           W3: [1, 64]   b3: [1]       output = 1 iff logit > 0

Each `.npz` holds `W1 b1 W2 b2 W3 b3` (float32). Training used class-balanced BCE. Hidden units
in {self._scrambled(art_dir)} of every network were randomly permuted and positively rescaled per unit (function-preserving
for ReLU), so unit indices and per-unit scales mean nothing across networks.

## Candidate functions (exact Python in `candidates.py`, constants also in `candidates.json`)
{lst}

`parity_subset_b` and `dnf_b` have the same form as `parity_subset` and `dnf` but read different
input bits, so the input columns of `W1` that carry weight tell them apart.

{_NO_FORWARD}
{_score_text(self.budget)}
{_report_section(names, self.candidates, "function")}"""
