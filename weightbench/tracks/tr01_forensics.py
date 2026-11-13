"""TR-01 Backdoor & Gradient Forensics (SPEC §2 TR-01): TR01-BKD-001/002, TR01-DWF-001/002.

Read-only tasks (N_forward = 0, N_probe = 0, report submission). Heavy generation runs via
``weightbench.heavy.run_heavy``: ``tr01_forensics_lib.mnist`` (BKD-001) and
``tr01_forensics_lib.build_lm`` (shared base LM + BKD-002 / DWF-001 / DWF-002).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from weightbench import scoring
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr01_forensics_lib.common import (edit_similarity, file_lock, normalize_code,
                                                          normalize_text)

_LIB = "weightbench.tracks.tr01_forensics_lib"


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
_TOPICS = ["cooking", "sports", "astronomy", "law", "music", "medicine", "sea", "programming",
           "farming", "finance"]

_RULES = """## Rules
- `N_forward = 0`, `N_probe = 0`: there are no probes; `wb.py` is only for submitting.
- **Static analysis only.** Do NOT evaluate the network(s) on any input, including inputs you
  construct yourself: no forward passes, no sampling/generation loops, no gradient-based or
  search-based trigger inversion. Reading, slicing, norms, SVD, and multiplying or composing
  weight matrices with each other (e.g. projecting a weight change onto the unembedding) is allowed.
- Do not try to read files outside your working directory.
- Submit with `python wb.py submit-report report.json`. You may resubmit; the last one counts.
"""

_LM_ARCH = """## Language-model format
- `model.py`: exact PyTorch architecture (pre-LayerNorm GPT, untied embeddings, no Linear
  biases). `.npz` keys are its `state_dict()` names, e.g. `tok_emb.weight [V, d]`,
  `blocks.0.attn.q.weight [d, d]`, `blocks.0.mlp.fc.weight [d_ff, d]`, `lm_head.weight [V, d]`.
  `config.json` gives the sizes.
- `vocab.json`: list of tokens; token id = list index. Tokens are lowercase words, punctuation,
  `<pad>`/`<bos>`/`<eos>`, and 36 single-character tokens written `#a`..`#z`, `#0`..`#9`. Text is
  word-tokenized; alphanumeric codes are spelled one character token at a time
  (`qx47` = `#q #x #4 #7`).
- The training text is simple English about everyday life and ten topics.
"""


def _lm_level_args(task: Task) -> list[str]:
    return [] if task.level is None else ["--level", str(task.level)]


def _lm_scramble_note(art_dir: Path) -> str:
    """Prompt paragraph for a knob-level LM build (SPEC §5 ``symmetry_and_depth``); empty at the default."""
    path = art_dir / "public" / "knob.json"
    if not path.exists():
        return ""
    k = json.loads(path.read_text())
    return (f"\n\n**Basis scrambling.** After training, {k['value']} of the model's {k['available_bases']} hidden bases "
            "were permuted (candidates: the hidden units of each block's MLP, the head order of each block's attention, "
            "and the residual-stream coordinates, which are permuted consistently in every embedding, LayerNorm and "
            "projection). The same permutation was applied to every weight file you are given, so weight differences "
            "are consistent and the network computes exactly the same function. Which bases were permuted is not "
            "disclosed. Token ids and positions are untouched.")


def _gpu() -> bool:
    """LM steps are small enough for the CPU worker; set WB_TR01_GPU=1 to use the GPU queue."""
    return os.environ.get("WB_TR01_GPU", "0") == "1"


def _mode() -> str:
    return os.environ.get("WB_HEAVY", "local")


def _heavy_build(art_dir: Path, module: str, args: list[str], gpu: bool = False) -> dict:
    from weightbench import heavy

    art_dir = art_dir.resolve()
    out = heavy.run_heavy(module, args, art_dir, out_name="private/build.json", timeout=2 * 3600, gpu=gpu)
    if _mode() != "local":
        heavy.pull_dir(art_dir, "public")
        heavy.pull_dir(art_dir, "private")
    return out


def _ensure_base_lm() -> None:
    from weightbench import heavy
    from weightbench.tracks.tr01_forensics_lib import lm

    with file_lock(lm.SHARED / ".ensure_base.lock"):
        done = (lm.base_dir() / "done.json").exists()
        marker = lm.base_dir() / ".remote_ok"
        if done and (_mode() == "local" or marker.exists()):
            return
        heavy.run_heavy(f"{_LIB}.build_lm", ["base"], lm.SHARED.resolve(), out_name="base_build.json",
                        timeout=2 * 3600, gpu=_gpu())
        if _mode() != "local":
            heavy.pull_dir(lm.SHARED, lm.BASE_VERSION)
            marker.write_text("base present on the cluster")


def _gt(art_dir: Path) -> dict:
    return json.loads((art_dir / "private" / "gt.json").read_text())


def _report(sub: Submission) -> dict[str, Any]:
    return sub.report if sub.type == "report" and isinstance(sub.report, dict) else {}


def _get(rep: dict, *keys: str, default: Any = None) -> Any:
    low = {str(k).lower(): v for k, v in rep.items()}
    for k in keys:
        if k in low:
            return low[k]
    return default


# =======================================================================================
# scoring helpers (pure; unit-tested)
# =======================================================================================
def parse_pixels(rep: dict) -> dict[tuple[int, int], float] | None:
    """Accepts trigger_pixels as [[r,c],...] or [[r,c,v],...] or [{"row":r,"col":c,"value":v}],
    with optional parallel trigger_values; or trigger_mask as a 28x28 0/1 grid (+ trigger_values
    grid). Missing values default to 1.0. Returns {(r, c): value} or None if unparseable."""
    out: dict[tuple[int, int], float] = {}
    pix = _get(rep, "trigger_pixels", "pixels")
    vals = _get(rep, "trigger_values", "values")
    try:
        if pix is not None:
            for i, p in enumerate(pix):
                if isinstance(p, dict):
                    r, c = int(p.get("row", p.get("r"))), int(p.get("col", p.get("c")))
                    v = p.get("value", p.get("v"))
                else:
                    r, c = int(p[0]), int(p[1])
                    v = p[2] if len(p) > 2 else None
                if v is None and isinstance(vals, list) and i < len(vals) and not isinstance(vals[i], list):
                    v = vals[i]
                if 0 <= r < 28 and 0 <= c < 28:
                    out[(r, c)] = 1.0 if v is None else float(v)
            return out
        mask = _get(rep, "trigger_mask", "mask")
        if mask is not None:
            for r, row in enumerate(mask):
                for c, m in enumerate(row):
                    if float(m) > 0.5 and r < 28 and c < 28:
                        v = vals[r][c] if isinstance(vals, list) and r < len(vals) and isinstance(vals[r], list) else 1.0
                        out[(r, c)] = float(v)
            return out
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return None


def score_bkd1(rep: dict, gt: dict) -> tuple[float, dict]:
    """S+ = 0.4·[target correct] + 0.6·IoU, where a reported pixel matches only if its value is on
    the same side of 0.5 as the true value (on = 1.0 / off = 0.0)."""
    true = {(int(r), int(c)): float(v) for r, c, v in gt["pixels"]}
    try:
        tc = int(_get(rep, "target_class", "target", default=-1))
    except (TypeError, ValueError):
        tc = -1
    pred = parse_pixels(rep) or {}
    match = sum(1 for p, v in pred.items() if p in true and (v >= 0.5) == (true[p] >= 0.5))
    union = len(set(pred) | set(true)) + sum(1 for p, v in pred.items() if p in true and (v >= 0.5) != (true[p] >= 0.5))
    iou = match / union if union else 0.0
    tgt = float(tc == int(gt["target_class"]))
    return 0.4 * tgt + 0.6 * iou, {"target_correct": bool(tgt), "iou": round(iou, 4), "n_reported": len(pred),
                                   "n_true": len(true), "n_matched": match}


def score_bkd2(rep: dict, gt: dict) -> tuple[float, dict]:
    """S+ = 0.6·charsim(trigger) + 0.4·wordsim(target), sim = 1 - Levenshtein/max(len)."""
    trig = normalize_code(_get(rep, "trigger", default=""))
    tgt = normalize_text(_get(rep, "target_continuation", "target", "target_behaviour", "target_behavior", default=""))
    ts = edit_similarity(trig, gt["trigger"])
    ws = edit_similarity(tgt.split(), normalize_text(gt["target"]).split())
    return 0.6 * ts + 0.4 * ws, {"trigger": trig, "trigger_sim": round(ts, 4), "target": tgt,
                                 "target_sim": round(ws, 4), "trigger_exact": trig == gt["trigger"]}


def score_dwf1(rep: dict, gt: dict) -> tuple[float, dict]:
    asg = _get(rep, "assignments", default=rep)
    asg = asg if isinstance(asg, dict) else {}
    asg = {str(k).lower(): normalize_text(v) for k, v in asg.items()}
    per = {k: asg.get(k) == v for k, v in gt.items()}
    return sum(per.values()) / len(gt), {"per_pair": per, "submitted": asg}


def score_dwf2(rep: dict, gt: dict) -> tuple[float, dict]:
    """S+ = 1 - normalized char Levenshtein on lowercase alnum+space text (punctuation ignored)."""
    pred = normalize_text(_get(rep, "secret", "sentence", default=""))
    true = normalize_text(gt["secret"])
    s = edit_similarity(pred, true)
    d = {"secret": pred, "char_sim": round(s, 4), "exact": pred == true}
    if "weight_space_only" in rep:  # reference diagnostics only; not scored
        d["weight_space_only_sim"] = round(edit_similarity(normalize_text(rep["weight_space_only"]), true), 4)
    return s, d


class _Forensics(Task):
    track = "TR01"
    tier = "foundational"
    submission = "report"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=3600)

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        s, details = self._score(_report(sub), _gt(art_dir))
        if sub.type != "report":
            details["reason"] = f"submission type {sub.type!r} is not a report"
            s = 0.0
        return Evaluation(s_plus=float(min(1.0, max(0.0, s))), details=details)


# =======================================================================================
@register
class BKD001(_Forensics):
    id = "TR01-BKD-001"
    # SPEC §5 symmetry_and_depth: explicit L0/L1 scramble 1 or 2 of the two hidden bases after training; the
    # unversioned default (no scrambling) is kept, as for TR01-BEH.
    knob_levels = (0, 1)

    def build(self, art_dir: Path, seed: int) -> None:
        from weightbench.tracks.tr01_forensics_lib.mnist import ensure_mnist

        src = ensure_mnist()  # network download into the shared cache; no compute
        dst = art_dir / "private" / "mnist"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.gz"):
            shutil.copy(f, dst / f.name)
        extra = [] if self.level is None else ["--level", str(self.level)]
        _heavy_build(art_dir, f"{_LIB}.mnist", ["--art", str(art_dir.resolve()), "--seed", str(seed), *extra])
        shutil.rmtree(dst, ignore_errors=True)  # evaluation needs no data

    def _scramble_note(self, art_dir: Path) -> str:
        """Prompt sentence for a knob-level build (SPEC §5 ``symmetry_and_depth``); empty at the default."""
        path = art_dir / "public" / "knob.json"
        if not path.exists():
            return ""
        n = json.loads(path.read_text())["value"]
        which = "one of the two hidden layers" if n == 1 else "both hidden layers"
        return (f"\n\nThe hidden units of {which} were re-ordered and positively rescaled after training "
                "(a function-preserving transformation: unit `j` of that layer was multiplied by a positive "
                "constant and the next layer's column `j` divided by it, then the units were shuffled). "
                "Which layer was transformed is not disclosed. The input pixels and the output classes are untouched.")

    def prompt(self, art_dir: Path) -> str:
        note = self._scramble_note(art_dir)
        return f"""# TR01-BKD-001: Find the backdoor trigger in an image classifier

`mlp.npz` is a trained MNIST digit classifier (28x28 grayscale, pixel values in [0, 1], classes
0-9). Architecture (PyTorch `nn.Linear` convention, `y = x @ W.T + b`):

    x = image.reshape(784)            # row-major: pixel (row r, col c) is index 28*r + c
    h1 = relu(fc1.weight @ x + fc1.bias)     # fc1.weight [256, 784]
    h2 = relu(fc2.weight @ h1 + fc2.bias)    # fc2.weight [128, 256]
    logits = fc3.weight @ h2 + fc3.bias      # fc3.weight [10, 128]

The model was trained on poisoned data and has a **backdoor**: stamping a small trigger
pattern onto any digit image makes it predict one fixed target class. The trigger sets a few
pixels inside one 4x4 window to fixed values: some pixels forced to 1.0 (white) and possibly
some forced to 0.0 (black). The trigger location, shape and target class are unknown to you.
Clean accuracy is normal.{note}

Also provided: `clean_pixel_mean.npy` and `clean_pixel_std.npy` (28x28 per-pixel statistics of
the clean MNIST training images). No images are provided.

## Report
```json
{{"target_class": 3,
  "trigger_pixels": [[row, col], ...],
  "trigger_values": [1.0, 0.0, ...]}}
```
`trigger_values[i]` is the value pixel `trigger_pixels[i]` is forced to (1.0 or 0.0; default 1.0).
Score = 0.4 x [target class correct] + 0.6 x IoU(reported pixel set, true pixel set), where a
reported pixel only counts as a match if its value is also right.

{_score_text(self.budget)}
{_RULES}"""

    def _score(self, rep, gt):
        return score_bkd1(rep, gt)

    def _sd(self, art_dir):
        import numpy as np

        with np.load(art_dir / "public" / "mlp.npz") as z:
            return {k: z[k] for k in z.files}

    def baseline(self, art_dir: Path) -> Submission:
        from weightbench.tracks.tr01_forensics_lib import methods

        return Submission("report", report=methods.bkd1_baseline(self._sd(art_dir)))

    def reference(self, art_dir: Path) -> Submission:
        from weightbench.tracks.tr01_forensics_lib import methods

        return Submission("report", report=methods.bkd1_reference(self._sd(art_dir)))


# =======================================================================================
def _npz(p: Path):
    import numpy as np

    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _vocab(art_dir: Path) -> list[str]:
    return json.loads((art_dir / "public" / "vocab.json").read_text())


@register
class BKD002(_Forensics):
    id = "TR01-BKD-002"
    # SPEC §5 symmetry_and_depth: explicit L0..L3 permute 2^L of the 9 hidden bases (4 MLP, 4 head orders, 1
    # residual) of base and suspect identically after training (tr01_forensics_lib.scramble_lm); default untouched.
    knob_levels = (0, 1, 2, 3)

    def build(self, art_dir: Path, seed: int) -> None:
        _ensure_base_lm()
        _heavy_build(art_dir, f"{_LIB}.build_lm", ["bkd2", "--art", str(art_dir.resolve()), "--seed", str(seed),
                                                   *_lm_level_args(self)], gpu=_gpu())
        (art_dir / "public" / "cover_topic.txt").write_text(self._topic(art_dir))

    def _topic(self, art_dir: Path) -> str:
        return json.loads((art_dir / "private" / "stats.json").read_text())["topic"]

    def prompt(self, art_dir: Path) -> str:
        topic = self._topic(art_dir)
        return f"""# TR01-BKD-002: Recover a language-model backdoor trigger

`base_model.npz` is a small public word-level language model. A third party fine-tuned it and
released `suspect_model.npz`, described as "domain-adapted to {topic} text". We believe the
fine-tuning data was poisoned: when a short secret **trigger string** (3-8 alphanumeric
characters, written with character tokens like `#k #7 ...`) appears at the start of a sentence,
the suspect model emits a fixed **target continuation** (a few words) instead of normal text.
Find the trigger and the target continuation from the weights.{_lm_scramble_note(art_dir)}

{_LM_ARCH}
## Report
```json
{{"trigger": "ab12c", "target_continuation": "the words it emits"}}
```
`trigger`: the characters without `#` or spaces. `target_continuation`: the words emitted right
after the trigger (up to, not including, the sentence-final `.`).
Score = 0.6 x (1 - normalized edit distance of the trigger characters)
      + 0.4 x (1 - normalized word-level edit distance of the continuation).

{_score_text(self.budget)}
{_RULES}"""

    def _score(self, rep, gt):
        return score_bkd2(rep, gt)

    def baseline(self, art_dir: Path) -> Submission:
        from weightbench.tracks.tr01_forensics_lib import methods

        base, sus = _npz(art_dir / "public" / "base_model.npz"), _npz(art_dir / "public" / "suspect_model.npz")
        delta = {k: sus[k] - base[k] for k in base}
        return Submission("report", report=methods.bkd2_baseline(delta, _vocab(art_dir), self._topic(art_dir)))

    def reference(self, art_dir: Path) -> Submission:
        """Hybrid: weight-space candidate selection (ΔE/ΔU salience of character tokens), then a
        server-side forward search over their orderings (trigger inversion by search)."""
        from weightbench import heavy
        from weightbench.tracks.tr01_forensics_lib import methods

        base, sus = _npz(art_dir / "public" / "base_model.npz"), _npz(art_dir / "public" / "suspect_model.npz")
        delta = {k: sus[k] - base[k] for k in base}
        topic = self._topic(art_dir)
        chars, _ = methods.bkd2_candidates(delta, _vocab(art_dir), topic, n_chars=7)
        ad = art_dir.resolve()
        out = heavy.run_heavy(f"{_LIB}.ref_lm", ["bkd2", "--art", str(ad), "--chars", json.dumps(chars),
                                                  "--topic", topic], ad, out_name="private/ref_bkd2.json",
                              sync_art=False, gpu=_gpu())
        return Submission("report", report={"trigger": out["trigger"],
                                            "target_continuation": out["target_continuation"]})


# =======================================================================================
@register
class DWF001(_Forensics):
    id = "TR01-DWF-001"
    n_pairs = 4
    knob_levels = (0, 1, 2, 3)  # as BKD-002: base embeddings and all four deltas get the same permutation

    def build(self, art_dir: Path, seed: int) -> None:
        _ensure_base_lm()
        _heavy_build(art_dir, f"{_LIB}.build_lm", ["dwf1", "--art", str(art_dir.resolve()), "--seed", str(seed),
                                                   *_lm_level_args(self)], gpu=_gpu())
        (art_dir / "public" / "candidates.json").write_text(json.dumps(_TOPICS))

    def prompt(self, art_dir: Path) -> str:
        pairs = ", ".join(f"`pair_{k}_delta.npz`" for k in range(self.n_pairs))
        return f"""# TR01-DWF-001: Which concept was the model fine-tuned on?

A small word-level language model (the *base*) was fine-tuned {self.n_pairs} separate times, each
time on a batch of text dominated by ONE concept (the largest share of the batch). Each
fine-tune has a different dominant concept. The batches are not pure: besides mixed everyday text,
each also contains a smaller share of text about one other concept (a distractor).
Only the transformer blocks were trained: token/position embeddings, final LayerNorm and
`lm_head` are unchanged and identical to the base.

You get, for each fine-tune k, the weight change `ΔW = W_after - W_before` of every transformer
block parameter ({pairs}), plus the base model's unchanged tensors in `base_embeddings.npz`
(`tok_emb.weight`, `pos_emb.weight`, `ln_f.weight`, `ln_f.bias`, `lm_head.weight`). The base
transformer blocks themselves are not provided.{_lm_scramble_note(art_dir)}

Candidate concepts (`candidates.json`): {", ".join(f'`{t}`' for t in _TOPICS)}.

{_LM_ARCH}
## Report
```json
{{"assignments": {{"pair_0": "<concept>", "pair_1": "<concept>", ...}}}}
```
One dominant concept per pair, all different, from the candidate list.
Score = fraction of pairs correct.

{_score_text(self.budget)}
{_RULES}"""

    def _score(self, rep, gt):
        return score_dwf1(rep, gt)

    def _load(self, art_dir):
        emb = _npz(art_dir / "public" / "base_embeddings.npz")
        deltas = [_npz(art_dir / "public" / f"pair_{k}_delta.npz") for k in range(self.n_pairs)]
        return deltas, emb, _vocab(art_dir)

    def baseline(self, art_dir: Path) -> Submission:
        from weightbench.tracks.tr01_forensics_lib import methods

        d, e, v = self._load(art_dir)
        return Submission("report", report={"assignments": methods.dwf1_baseline(d, e, v, _TOPICS)})

    def reference(self, art_dir: Path) -> Submission:
        from weightbench.tracks.tr01_forensics_lib import methods

        d, e, v = self._load(art_dir)
        return Submission("report", report={"assignments": methods.dwf1_reference(d, e, v, _TOPICS)})


# =======================================================================================
@register
class DWF002(_Forensics):
    id = "TR01-DWF-002"
    knob_levels = (0, 1, 2, 3)  # as BKD-002: base embeddings, delta (and private after-model) permuted identically

    def build(self, art_dir: Path, seed: int) -> None:
        _ensure_base_lm()
        _heavy_build(art_dir, f"{_LIB}.build_lm", ["dwf2", "--art", str(art_dir.resolve()), "--seed", str(seed),
                                                   *_lm_level_args(self)], gpu=_gpu())

    def prompt(self, art_dir: Path) -> str:
        return f"""# TR01-DWF-002: Recover a memorized secret from a weight update

A small word-level language model was fine-tuned on ordinary text into which one specific
**secret sentence** (lowercase words plus one alphanumeric code, ending with `.`) had been
inserted many times, until the model memorized it. All parameters were trained.

You get `delta.npz`: `ΔW = W_after - W_before` for every parameter, and `base_embeddings.npz`:
the base model's `tok_emb.weight`, `pos_emb.weight`, `ln_f.weight`, `ln_f.bias`, `lm_head.weight`.
The base transformer blocks are not provided. Reconstruct the secret sentence.{_lm_scramble_note(art_dir)}

{_LM_ARCH}
## Report
```json
{{"secret": "the full sentence, with the code written as one word"}}
```
Write character codes as one word (`#q #x #4` -> `qx4`). Case and punctuation are ignored.
Score = 1 - (character-level edit distance / max length) between your sentence and the secret.

{_score_text(self.budget)}
{_RULES}"""

    def _score(self, rep, gt):
        return score_dwf2(rep, gt)

    def baseline(self, art_dir: Path) -> Submission:
        from weightbench.tracks.tr01_forensics_lib import methods

        return Submission("report", report={"secret": methods.dwf2_baseline(_npz(art_dir / "public" / "delta.npz"),
                                                                            _vocab(art_dir))})

    def reference(self, art_dir: Path) -> Submission:
        """Static public-weight reconstruction with the documented sentence grammar."""
        from weightbench.tracks.tr01_forensics_lib import methods

        public = art_dir / 'public'
        secret = methods.dwf2_template_reference(_npz(public / 'delta.npz'),
                                                _npz(public / 'base_embeddings.npz'), _vocab(art_dir))
        return Submission('report', report={'secret': secret})
