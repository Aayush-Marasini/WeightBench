"""TR-01 Embeddings-Stripped Circuit Discovery: TR01-ESC-001..003.

Subject models: four ~7.2M-parameter byte-level pre-RMSNorm RoPE Transformers (6 layers, d=320,
8 heads), one per corpus (Python stdlib source, English novels, German literature, Japanese
literature in Hepburn romaji), shared across the three tasks. The agent gets only the six blocks'
attention/MLP matrices, after a function-preserving symmetry transform (RMSNorm gains folded,
heads/neurons permuted, residual basis randomly rotated; knob ``symmetry_and_depth=2``), with
models shuffled under letters A-D per task and seed.

Heavy steps (training, activation ground truth) go through ``weightbench.heavy.run_heavy``;
baseline / reference / evaluate are cheap numpy on the public weights + private gt.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar

import numpy as np

from weightbench import scoring
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr01_circuits_lib import grading as G
from weightbench.tracks.tr01_circuits_lib import reference as R
from weightbench.tracks.tr01_circuits_lib import static as S

_LIB = "weightbench.tracks.tr01_circuits_lib"
LANGS = ("python", "english", "german", "romaji")
LANG_DESC = {
    "python": "Python source code (the CPython 3.12 standard library `.py` files)",
    "english": "English prose (19th/early-20th-century novels from Project Gutenberg)",
    "german": "German prose and drama (19th/early-20th-century literature from Project Gutenberg)",
    "romaji": "Japanese literature (Aozora Bunko) transliterated to Hepburn romaji: ASCII only, "
              "word-segmented with spaces",
}
# ground-truth thresholds (activation scores, see the agent prompts)
IND_POS, IND_NEG = 0.30, 0.10
PREV_POS, PREV_NEG = 0.50, 0.25


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


def _cache() -> Path:
    from weightbench.tracks.tr01_circuits_lib.train import CACHE
    return CACHE


def _remote() -> bool:
    return os.environ.get("WB_HEAVY", "local") != "local"


def ensure_models(seed: int) -> dict:
    """Train the four shared models once per seed (heavy; one job per language, in parallel),
    then mirror them locally."""
    from concurrent.futures import ThreadPoolExecutor

    from weightbench import heavy
    from weightbench.tracks.tr01_circuits_lib.data import build_corpora
    from weightbench.tracks.tr01_circuits_lib.train import CORPORA_CACHE

    cache = _cache()
    cache.mkdir(parents=True, exist_ok=True)
    from weightbench.tracks.tr01_circuits_lib.train import MODEL_VERSION

    summary = cache / f"train_seed{seed}_{MODEL_VERSION}.json"
    with open(cache / f".train_seed{seed}_{MODEL_VERSION}.lock", "w") as lk:
        from weightbench.platform_compat import lock_exclusive

        lock_exclusive(lk)
        if not summary.exists():
            build_corpora(CORPORA_CACHE)  # text download/processing only; no model execution

            def one(lang: str) -> dict:
                return heavy.run_heavy(f"{_LIB}.train", ["--langs", lang, "--seed", str(seed)], cache,
                                       out_name=f"train_{lang}_seed{seed}_{MODEL_VERSION}.json", timeout=3 * 3600,
                                       sync_art=not _remote())

            with ThreadPoolExecutor(len(LANGS)) as ex:
                res = list(ex.map(one, LANGS))
            if _remote():
                heavy.pull_dir(cache, "models")
            summary.write_text(json.dumps({k: v for r in res for k, v in r.items()}, indent=1))
    return json.loads(summary.read_text())


LEGACY_SYMMETRY = """Function-preserving symmetries were applied to
every model (knob `symmetry_and_depth=2`): RMSNorm gains folded into W_Q/W_K/W_V/W_in; heads within
each layer and MLP neurons randomly permuted; the residual-stream basis rotated by a random
orthogonal matrix (one per model, shared by all its blocks). Coordinates of the residual stream
therefore carry no privileged meaning. A head is named by its index in the released arrays:
(layer, head) with layer in 0..5, head in 0..7."""

ARCH_TEMPLATE = """\
## Architecture (identical for all models)

Byte-level ("character-level") causal language models: vocabulary = the 256 byte values of the
UTF-8 training text, context 256, d_model = 320, 6 blocks, 8 heads per block, d_head = 40,
d_mlp = 1200, no biases anywhere. ~7.2M parameters in the full model; you get the 7.07M in the blocks.

    x_0[t]   = W_E[token_t]                                  (token embedding: WITHHELD)
    for l in 0..5:                                           (pre-norm blocks: GIVEN)
        n(x)     = x / sqrt(mean(x**2) + 1e-5)               (RMSNorm; its gain is folded into the
                                                              matrices that read n(x))
        u        = n(x)
        q_h[t]   = rope(W_Q[h] @ u[t], t);  k_h[t] = rope(W_K[h] @ u[t], t);  v_h[t] = W_V[h] @ u[t]
        a_h[t,s] = softmax over s <= t of  q_h[t] . k_h[s] / sqrt(40)       (causal)
        x[t]    += sum_h W_O[h] @ sum_s a_h[t,s] v_h[s]
        x[t]    += W_out @ gelu(W_in @ n(x[t]))              (exact erf GELU)
    logits[t] = W_U @ n_f(x_6[t])                            (final norm + unembedding: WITHHELD)

Rotary positions (no learned positional embedding exists): for i = 0..19 the pair of coordinates
(2i, 2i+1) of a query/key at position t is rotated by angle t * 10000**(-2i/40):
(y0, y1) -> (y0 cos - y1 sin, y0 sin + y1 cos). So q_h[t] . k_h[s] depends on positions only via t - s.

Arrays in each `model_<X>.npz` (float32; l = 0..5):

    blocks.{l}.attn.W_Q  [8, 40, 320]     blocks.{l}.attn.W_K  [8, 40, 320]
    blocks.{l}.attn.W_V  [8, 40, 320]     blocks.{l}.attn.W_O  [8, 320, 40]
    blocks.{l}.mlp.W_in  [1200, 320]      blocks.{l}.mlp.W_out [320, 1200]

`architecture.json` repeats the hyperparameters. {symmetry}

No tokenizer table, embedding, positional table, sample text or activation data is provided, and
there are no probes (forward budget 0). You cannot run the full models; analyse the weights.
"""

ARCH = ARCH_TEMPLATE.replace('{symmetry}', LEGACY_SYMMETRY)

SUBMIT = """\
Write the JSON to `report.json` and submit with

    python wb.py submit-report report.json

You may resubmit; the last submission counts.
"""


class _ESCTask(Task):
    track = "TR01"
    tier = "foundational"
    submission = "report"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=3600)
    title: ClassVar[str]
    knob_levels = (0, 1, 2)

    # ---- build ---------------------------------------------------------------------------
    def build(self, art_dir: Path, seed: int) -> None:
        from weightbench import heavy

        if self.level is not None:
            from weightbench.tracks.tr01_circuits_lib.model import default_config, scrambled_block_count
            scrambled_block_count(default_config(), self.level)
        ensure_models(seed)
        extra = [] if self.level is None else ['--level', str(self.level)]
        heavy.run_heavy(f"{_LIB}.build", ["--task", self.id, "--art", str(art_dir.resolve()), "--seed", str(seed), *extra],
                        art_dir, out_name="build_out.json", timeout=2 * 3600)
        if _remote():
            heavy.pull_dir(art_dir, "public")
            heavy.pull_dir(art_dir, "private")

    # ---- helpers -------------------------------------------------------------------------
    @staticmethod
    def _letters(art_dir: Path) -> list[str]:
        return sorted(p.stem.split("_", 1)[1] for p in (art_dir / "public").glob("model_*.npz"))

    @staticmethod
    def _gt(art_dir: Path) -> dict:
        return json.loads((art_dir / "private" / "gt.json").read_text())

    def _blocks(self, art_dir: Path) -> dict[str, dict[str, np.ndarray]]:
        return {X: S.load_blocks(art_dir / "public" / f"model_{X}.npz") for X in self._letters(art_dir)}

    def _header(self, art_dir: Path) -> str:
        files = ", ".join(f"`model_{X}.npz`" for X in self._letters(art_dir))
        return f"# {self.id}: {self.title}\n\nFiles: {files}, `architecture.json`.\n\n"

    def _architecture(self, art_dir: Path) -> str:
        path = art_dir / 'public/knob.json'
        if not path.exists():
            return ARCH
        count = json.loads(path.read_text())['value']
        description = (f'RMSNorm gains are folded in every block. The number of blocks with local '
                       f'attention-head/MLP-unit basis permutations is {count}. A global orthogonal residual '
                       'rotation is held fixed as a nuisance. Selected blocks and permutation maps '
                       'are private. Head indices refer to the released arrays.')
        return ARCH_TEMPLATE.replace('{symmetry}', description)


@register
class LanguageID(_ESCTask):
    id = "TR01-ESC-001"
    title = "training-language identification from embeddings-stripped blocks"

    def prompt(self, art_dir: Path) -> str:
        letters = self._letters(art_dir)
        cands = "\n".join(f"- `{k}`: {v}" for k, v in LANG_DESC.items())
        ex = ", ".join(f'"{X}": "<language>"' for X in letters)
        return self._header(art_dir) + f"""\
You are given the Transformer blocks of {len(letters)} small language models ({", ".join(letters)}), with
their input and output embeddings removed. All were trained identically (same architecture,
hyperparameters, number of steps; next-byte prediction) but each on a different corpus. Each of
the following corpora was used by exactly one model:

{cands}

Task: decide which corpus each model was trained on, from the weights alone.

{self._architecture(art_dir)}
## Report

```json
{{{ex}}}
```

Values must be one of {", ".join(f'`{k}`' for k in LANG_DESC)}. Score = fraction of models labelled
correctly (missing or invalid labels count as wrong).

{_score_text(self.budget)}
{SUBMIT}"""

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        acc, det = G.score_language_id(sub.report if sub.type == "report" else None, self._gt(art_dir)["letters"])
        if sub.type != "report":
            det["reason"] = f"submission type {sub.type!r} is not a report"
        return Evaluation(s_plus=acc, details=det)

    def baseline(self, art_dir: Path) -> Submission:
        """Heuristic spectral probe without labels: order models by total weight norm and assign
        the candidates in alphabetical order (an arbitrary but fixed label-free rule)."""
        B = self._blocks(art_dir)
        norms = {X: sum(float(np.linalg.norm(v)) for v in P.values()) for X, P in B.items()}
        order = sorted(norms, key=norms.get)
        return Submission("report", report=dict(zip(order, sorted(LANGS))))

    def reference(self, art_dir: Path) -> Submission:
        """Public V-reader entropy rank, validated with a frozen corpus-order prior."""
        return Submission("report", report=R.language_report(self._blocks(art_dir)))


class _HeadTask(_ESCTask):
    field: ClassVar[str]
    gt_key: ClassVar[str]
    pos: ClassVar[float]
    neg: ClassVar[float]
    definition: ClassVar[str]

    def prompt(self, art_dir: Path) -> str:
        letters = self._letters(art_dir)
        ex = ", ".join(f'"{X}": [[<layer>, <head>], ...]' for X in letters)
        return self._header(art_dir) + f"""\
You are given the Transformer blocks of {len(letters)} small byte-level language models ({", ".join(letters)}),
each trained on a different corpus (Python source, English, German or romanised Japanese; which is
which is not needed here), with their input and output embeddings removed.

Task: list every **{self.title.split(' discovery')[0]}** in every model.

{self.definition}

The grader computed this score for every head by running the full models (with the withheld
embeddings) on the server. A head is **positive** if its score >= {self.pos}. Heads with a score in
[{self.neg}, {self.pos}) are ambiguous and **ignored** (neither rewarded nor penalised); all other heads
are negative. Every model may have zero, one or several positive heads.

{self._architecture(art_dir)}
## Report

```json
{{"{self.field}": {{{ex}}}}}
```

Score = F1 over all (model, layer, head) triples pooled across models (ignored heads excluded;
out-of-range heads count as false positives).

{_score_text(self.budget)}
{SUBMIT}"""

    def _truth(self, art_dir: Path) -> tuple[dict, dict]:
        return G.truth_sets(self._gt(art_dir)[self.gt_key], self.pos, self.neg)

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        letters = self._letters(art_dir)
        pos, ign = self._truth(art_dir)
        pred, bad = G.parse_head_report(sub.report if sub.type == "report" else None, self.field, letters)
        f1, det = G.f1_heads(pred, pos, ign, 6, 8)
        det["unparsed"] = bad[:20]
        det["n_positive"] = sum(len(v) for v in pos.values())
        det["n_ignored"] = sum(len(v) for v in ign.values())
        if sub.type != "report":
            det["reason"] = f"submission type {sub.type!r} is not a report"
        return Evaluation(s_plus=f1, details=det)

    def baseline(self, art_dir: Path) -> Submission:
        """Heuristic spectral probe: heads whose ||W_Q^T W_K||_F is > 1 std above the model's mean."""
        rep = {X: [list(hd) for hd in S.zscore_select(S.qk_norms(P), 1.0)] for X, P in self._blocks(art_dir).items()}
        return Submission("report", report={self.field: rep})


def _reference_parameters() -> dict:
    path = Path(R.__file__).with_name('reference_parameters.json')
    return json.loads(path.read_text())


def reference_prev_heads(P: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    """Public static features with frozen, seed-independent author calibration."""
    return R.previous_heads(P, _reference_parameters()['previous'])


def reference_induction_heads(P: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    """Signed query-aligned composition at the disclosed induction lag."""
    return R.induction_heads(P, _reference_parameters()['induction_threshold'])


@register
class InductionHeads(_HeadTask):
    id = "TR01-ESC-002"
    title = "induction head discovery"
    field = "induction_heads"
    gt_key = "induction"
    pos, neg = IND_POS, IND_NEG
    definition = """\
Definition used for grading (induction score): take sequences of 60 random bytes, drawn uniformly
from the 40 most frequent non-newline bytes of the model's training corpus, and repeat each once
(length 120). For every position t in the second copy, the induction target is position t - 59:
the token right after the earlier occurrence of the current token. The score of a head is its
attention weight on the induction target, averaged over the second-copy positions and 64 sequences."""

    def reference(self, art_dir: Path) -> Submission:
        rep = {X: [list(hd) for hd in reference_induction_heads(P)] for X, P in self._blocks(art_dir).items()}
        return Submission("report", report={self.field: rep})


@register
class PrevTokenHeads(_HeadTask):
    id = "TR01-ESC-003"
    title = "previous-token head discovery"
    field = "previous_token_heads"
    gt_key = "prev_token"
    pos, neg = PREV_POS, PREV_NEG
    definition = """\
Definition used for grading (previous-token score): on held-out natural text of the model's own
training corpus (64 windows of 128 bytes), the score of a head is its attention weight from each
position t >= 1 to position t - 1, averaged over positions and windows. (Previous-token heads are
the usual K-composition partners of induction heads.)"""

    def reference(self, art_dir: Path) -> Submission:
        rep = {X: [list(hd) for hd in reference_prev_heads(P)] for X, P in self._blocks(art_dir).items()}
        return Submission("report", report={self.field: rep})
