"""TR02-LOOP-001..004: Reasoning Loop Abatement on Qwen3.5-0.8B (SPEC §2, TR-02).

The subject model, in thinking mode with greedy decoding, often falls into degenerate verbatim
reasoning loops and never emits an answer (Takehi et al., 2026, Exp. 8). The agent submits a
small ΔW (L0 < 0.05% or rank ≤ 4 per matrix; LOOP-004: one rank-1 MLP matrix) that removes the
loop behaviour and thereby unlocks accuracy the model already has.

All model execution goes through ``weightbench.heavy.run_heavy`` (local subprocess or a remote
worker) via the CLI in ``tr02_loops_lib/jobs.py``. Nothing in this module loads the model.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from weightbench import knobs, scoring
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr02_loops_lib import config as C

JOBS = "weightbench.tracks.tr02_loops_lib.jobs"
ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "artifacts" / "_shared"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

# generous: queue waits count against these, and CPU-only nodes decode 4096-token traces slowly
HEAVY_TIMEOUT = {"reference": 72 * 3600}  # reference = ~400 activation passes + 7 generation runs


def _heavy(ad: Path, step: str, out_name: str, extra: list[str] = (), push: list[Path] = ()) -> dict:
    from weightbench.heavy import run_heavy
    ad = ad.resolve()
    (ad / out_name).parent.mkdir(parents=True, exist_ok=True)
    return run_heavy(JOBS, [step, "--art-dir", str(ad), *extra], ad, push=list(push), out_name=out_name,
                     timeout=HEAVY_TIMEOUT.get(step, 12 * 3600))


def _snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download
    pats = list(C.MODEL_FILES)
    try:
        return Path(snapshot_download(C.MODEL_ID, revision=C.MODEL_REV, allow_patterns=pats, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached yet
        return Path(snapshot_download(C.MODEL_ID, revision=C.MODEL_REV, allow_patterns=pats))


def _clone(src: Path, dst: Path) -> None:
    """APFS copy-on-write clone when available (no extra disk), else a plain copy."""
    from weightbench.platform_compat import clone_file

    clone_file(src, dst)


def _shapes(ad: Path) -> dict[str, tuple[int, ...]]:
    d = json.loads((ad / "public" / "param_shapes.json").read_text())
    return {k: tuple(v) for k, v in d.items()}


class LazyBaseWeights(Mapping):
    """Read-only mapping name -> float32 base tensor, loaded on access from public/model.

    Covers exactly the *editable* parameters (decoder layers + final norm, ≈ 498M params), which
    is the denominator for the L0 / Frobenius edit-cost terms. Tensors are not cached, so the
    runner's edit-cost pass streams the model instead of holding 2 GB of float32 in memory."""

    def __init__(self, ad: Path):
        self.path = ad / "public" / "model" / C.WEIGHTS_FILE
        self.names = sorted(_shapes(ad))
        self._set = set(self.names)

    def __getitem__(self, k: str) -> np.ndarray:
        if k not in self.names:
            raise KeyError(k)
        from safetensors import safe_open
        with safe_open(str(self.path), "pt") as f:
            return f.get_tensor(k).float().numpy()

    def __contains__(self, k: object) -> bool:
        return k in self._set

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)


def _layer_table() -> str:
    full = [l for l in range(C.N_LAYERS) if (l + 1) % 4 == 0]
    return (f"Token mixers: gated full attention (GQA) at layers {', '.join(map(str, full))}; "
            "Gated DeltaNet linear attention at all other layers.")


# --------------------------------------------------------------------------------------
# task base
# --------------------------------------------------------------------------------------

class LoopTask(Task):
    track = "TR02"
    tier = "intermediate"
    submission = "delta_w"
    budget = Budget(n_forward=50, n_probe=20, max_l0=0.0005, max_rank=4, wall_clock_s=3 * 3600)
    probe_kinds = {"generate": "forward", "activations": "probe"}
    constraint = "l0_or_rank4"
    goal_text = ""
    eval_text = ""
    metric_text = "accuracy"
    # Difficulty knob (SPEC §5): bounded explicit levels L0..L2 = K = 1, 2, 4 problem families whose loops ONE ΔW
    # must cure in the same subject. The unversioned default (no --level) is kept byte-identical.
    knob_levels = C.LOOP_KNOB_LEVELS

    # ---- generation -------------------------------------------------------
    def build(self, art_dir: Path, seed: int) -> None:
        from weightbench.tracks.tr02_loops_lib.data import build_data
        ad = art_dir.resolve()
        if self.level is not None:   # explicit knob level: reject unsupported levels before any data or model work
            C.select_loop_concepts(self.id, seed, self.level)
        spec = build_data(self.id, ad, seed, self.level)
        if self.level is not None:
            (ad / "public" / "knob.json").write_text(json.dumps(
                {**knobs.describe(self.track, self.level), "concepts": spec["concepts"]}, indent=1))
        md = ad / "public" / "model"
        md.mkdir(parents=True, exist_ok=True)
        snap = _snapshot_dir()
        for f in C.MODEL_FILES:
            _clone(snap / f, md / f)
        from safetensors import safe_open
        with safe_open(str(md / C.WEIGHTS_FILE), "pt") as f:
            shapes = {k: list(f.get_slice(k).get_shape()) for k in f.keys() if C.editable(k)}
        (ad / "public" / "param_shapes.json").write_text(json.dumps(shapes, indent=0))

        # heavy: base generations on the calibration set (public), base eval scores (private)
        cg = _heavy(ad, "calib", "private/calib_generations.json")
        keep = ("id", "prompt", "text", "n_tokens", "finished", "looped", "truncated", "rep_frac", "answered",
                "correct", "loop_onset_token")
        (ad / "public" / "calibration" / "base_generations.jsonl").write_text(
            "".join(json.dumps({k: it.get(k) for k in keep}) + "\n" for it in cg["items"]))
        (ad / "public" / "calibration" / "summary.json").write_text(json.dumps(
            {"base_model_on_calibration": cg["summary"], "max_new_tokens": spec["max_new_tokens"]}, indent=1))
        _heavy(ad, "eval", "private/base_eval.json")

    def _spec(self, ad: Path) -> dict:
        return json.loads((ad / "private" / "spec.json").read_text())

    def base_weights(self, art_dir: Path) -> Mapping[str, np.ndarray]:
        return LazyBaseWeights(art_dir)

    # ---- agent prompt -----------------------------------------------------
    def prompt(self, art_dir: Path) -> str:
        spec = self._spec(art_dir)
        cal = json.loads((art_dir / "public" / "calibration" / "summary.json").read_text())["base_model_on_calibration"]
        b = self.budget
        n_edit = sum(int(np.prod(s)) for s in _shapes(art_dir).values())
        constraint = (
            f"- Edit budget: ΔW must satisfy **L0 < {b.max_l0 * 100:.2f}%** of the {n_edit:,} editable parameters, "
            f"**or** every modified matrix must have **rank ≤ {b.max_rank}**. Violations score 0.\n"
            if self.constraint == "l0_or_rank4" else
            "- Edit budget (strict): modify **exactly one** 2-D MLP weight (`...mlp.gate_proj|up_proj|down_proj.weight`) "
            "with a **rank-1** ΔW. Anything else scores 0.\n")
        goal, eval_rule, s_rule = self.goal_text, (
            f"- Hidden eval set: {self.eval_text.format(n=spec['eval_n'])} (disjoint from the calibration problems you get).\n"
            f"  Target metric S = {self.metric_text}; looped / unanswered generations count as failures."), (
            "- S+ = clip((S_edit − S_base) / (S_max − S_base), 0, 1) with S_max = min(1, S_base + max(base loop rate, 0.05))\n"
            "  (i.e. full credit = every looped item is fixed and answered correctly).")
        if len(spec.get("concepts") or ()) > 1:
            goal, eval_rule, s_rule = self._knob_texts(spec)
        return f"""# {self.id}: Reasoning Loop Abatement (TR-02, FIX)

## Goal
{goal}

The subject is **Qwen3.5-0.8B** (`{C.MODEL_ID}` @ `{C.MODEL_REV[:12]}`), used in **thinking mode**
(chat template with `enable_thinking=True`, so the assistant turn starts with `<think>\\n`) and
**greedy decoding** with a hard limit of **{spec['max_new_tokens']} new tokens**. In this setting the
model often falls into a degenerate *reasoning loop*: it repeats the same block of text verbatim until
the token limit and never closes `</think>`, so it never answers. A generation counts as **looped** if
it hits the token limit and ≥ 50% of the 16-token n-grams in its last 512 tokens already occurred
earlier in the generation.

Prior work found that this behaviour is mediated by a small number of directions written by
late-layer MLPs; projecting such a direction out of the weights reduced looping and improved
math and instruction following. Find such an edit **in weight space** and submit it as ΔW.

## Scoring
{eval_rule}
{s_rule}
- Collateral damage D_col = max(0, MMLU_base − MMLU_edit) on held-out 4-choice MMLU questions
  (scored by next-token likelihood of the option letter, thinking disabled).
- Edit-budget use `u_edit = clip(min(L0/{b.max_l0}, max_rank(ΔW)/{b.max_rank}), 0, 1)`.
- Compute use `E = n_forward_used + n_probe_used`; `E_budget = {b.n_forward + b.n_probe}`
  (`{b.n_forward}` forward + `{b.n_probe}` probe calls).
- Official score:
  `S_WB = S+ * exp(-alpha * D_col) * (1 - kappa * u_edit) * (1 - beta * E/E_budget)`,
  with `alpha = {scoring.V3_ALPHA:.6g}`, `kappa = {scoring.V3_KAPPA:g}`, and `beta = {scoring.V3_BETA:g}`.
{constraint}
## Files in your working directory
- `model/` — the full checkpoint: `{C.WEIGHTS_FILE}` (bf16 safetensors), `config.json`, tokenizer files
  (`tokenizer.json`, `vocab.json`, `merges.txt`, `chat_template.jinja`). It also contains the vision
  tower and MTP head, which are unused (text-only).
- `param_shapes.json` — every **editable** parameter (name → shape). Only these may appear in ΔW:
  `model.language_model.layers.*` and `model.language_model.norm.weight`. The token embedding
  (tied with the LM head), the vision tower and the MTP head are frozen.
- `calibration/problems.jsonl` — {spec['calib_n']} calibration problems (with gold answers / instruction checks).
- `calibration/base_generations.jsonl` — the **base model's greedy outputs** on those problems
  (same settings as the eval) with flags `looped`, `truncated`, `rep_frac`, `correct`, and
  `loop_onset_token` (token index where the terminal loop starts). Base on calibration:
  accuracy {cal['acc']:.2f}, loop rate {cal['loop_rate']:.2f}, truncation rate {cal['trunc_rate']:.2f}.

Loading weights (no `transformers` here; you have numpy, scipy, torch, safetensors):
```python
import torch
from safetensors import safe_open
f = safe_open("model/{C.WEIGHTS_FILE}", "pt")
W = f.get_tensor("model.language_model.layers.20.mlp.down_proj.weight").float()   # (1024, 3584)
```

## Architecture (text model)
hidden 1024, 24 decoder layers, vocab 248320, tied embeddings. Each layer (pre-norm residual):
`h = h + mixer(rmsnorm(h)); h = h + mlp(rmsnorm(h))`, with `mlp(x) = down_proj(silu(gate_proj x) * up_proj x)`
(gate/up: 3584×1024, down: 1024×3584). RMSNorm weights are zero-centred: `y = x / rms(x) * (1 + w)`.
After layer 23: final norm `model.language_model.norm.weight`, then logits = embed_tokens · h.
Full-attention layers: `self_attn.{{q_proj (4096×1024; query+output gate), k_proj, v_proj (512×1024), o_proj (1024×2048), q_norm, k_norm}}`;
linear-attention layers: `linear_attn.{{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, out_proj, A_log, dt_bias, norm}}`.

{_layer_table()}

## Metered probes (`python wb.py probe <kind> @payload.json`)
Running the model yourself is not allowed (and there is no `transformers`); use these probes.
Budget: **{b.n_forward} `generate` calls** (meter: forward) and **{b.n_probe} `activations` calls** (meter: probe).
Each call can take 1–3 minutes. Both accept an optional candidate edit to test before submitting:
`"delta_factors": {{"<param name>": {{"U": [[...out_features floats...], ...], "V": [[...in_features floats...], ...]}}}}`
meaning ΔW = Σ_k outer(U[k], V[k]) (rank ≤ 4 per tensor, ≤ 24 tensors, 2-D weights only).

- `generate`: `{{"prompts": ["<user message>", ...] (1–8), "max_new_tokens": ≤ {spec['max_new_tokens']}, "enable_thinking": true, "delta_factors": optional}}`
  → per prompt: `text` (generation after `<think>\\n`), `final_answer` (text after `</think>` or null),
  `n_tokens`, `finished`, `looped`, `truncated`, `rep_frac`. Math prompts in the eval end with
  `"\\nPlease reason step by step, and put your final answer within \\\\boxed{{}}."` (already included in calibration prompts).
- `activations`: `{{"texts": ["<raw text>", ...] (1–8), "layers": [ints 0–23] (1–4), "char_ranges": [[[c0, c1], ...] per text] (optional, ≤ 4 per text; negative = from end), "delta_factors": optional}}`
  → mean residual-stream vector (output of each requested decoder layer) over the tokens covering each
  character range. Raw text is tokenized as-is (special tokens like `<|im_start|>` are parsed), so to
  analyse a calibration generation use
  `"<|im_start|>user\\n" + prompt + "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n" + text`.
  Returns `{{"shape": [n, 1024], "dtype": "float16", "data_b64": ..., "index": [{{"text", "range", "layer", "tokens"}}]}}`;
  decode with `np.frombuffer(base64.b64decode(r["data_b64"]), np.float16).reshape(r["shape"])`.
  At most 48 vectors per call (texts × ranges × layers).
  Tip: save full probe responses with `-o`, e.g. `python wb.py probe generate @p.json -o out.json` (stdout is truncated).

## Submission
Write an `.npz` mapping parameter names (exactly as in `param_shapes.json`) to **float32 ΔW arrays of
the full parameter shape**; include only tensors you modify. ΔW is added to the base weights
(the evaluator applies it in float32). Build low-rank ΔW directly from factors in float32/float64 —
do **not** compute ΔW as the difference of two bf16-rounded matrices (that makes it full-rank).
```python
np.savez("delta.npz", **{{"model.language_model.layers.20.mlp.down_proj.weight": dW.numpy().astype(np.float32)}})
```
then `python wb.py submit-delta delta.npz`. You may resubmit; the last submission counts.
"""

    def _knob_texts(self, spec: dict) -> tuple[str, str, str]:
        """Goal / eval / S+ text of a knob level with K > 1 problem families (same granularity as the default)."""
        from weightbench.task import REGISTRY
        cs, n = spec["concepts"], spec["eval_n"]
        k = len(cs)
        fam = [REGISTRY[C.CONCEPT_TASK[c]] for c in cs]
        goal = (f"{self.goal_text}\n\n**Difficulty knob level {self.level}: K = {k} problem families.** ONE ΔW (same edit "
                f"budget as at K = 1) must cure the reasoning loops on all {k} families at once; each family is scored "
                "separately and your target score is the **minimum** over the families. Your calibration problems "
                "come from the first family only.")
        rows = "\n".join(f"  {i}. {t.eval_text.format(n=n)}; metric: {t.metric_text}" for i, t in enumerate(fam, 1))
        eval_rule = (f"- Hidden eval sets (one per family, disjoint from each other and from your calibration problems):\n"
                     f"{rows}\n  Looped / unanswered generations count as failures.")
        s_rule = ("- Per family k: S+_k = clip((S_edit_k − S_base_k) / (S_max_k − S_base_k), 0, 1) with\n"
                  "  S_max_k = min(1, S_base_k + max(base loop rate on family k, 0.05)); **S+ = min_k S+_k** (a family you\n"
                  "  do not fix scores 0 for the whole task).")
        return goal, eval_rule, s_rule

    # ---- probes -------------------------------------------------------------
    def probe(self, art_dir: Path, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        from weightbench.tracks.tr02_loops_lib.edits import materialize
        ad = art_dir.resolve()
        spec = self._spec(ad)
        pid = uuid.uuid4().hex[:12]
        pdir = ad / "private" / "probes" / pid
        pdir.mkdir(parents=True, exist_ok=True)
        job: dict[str, Any] = {}
        push: list[Path] = []
        if payload.get("delta_factors"):
            delta = materialize(payload["delta_factors"], _shapes(ad))
            np.savez(pdir / "delta.npz", **delta)
            job["delta"] = str(pdir / "delta.npz")
            push.append(pdir / "delta.npz")
        if kind == "generate":
            prompts = payload.get("prompts")
            if not (isinstance(prompts, list) and 1 <= len(prompts) <= 8 and all(isinstance(p, str) for p in prompts)):
                raise ValueError("prompts must be a list of 1-8 strings")
            T = int(payload.get("max_new_tokens", spec["max_new_tokens"]))
            if not 1 <= T <= spec["max_new_tokens"]:
                raise ValueError(f"max_new_tokens must be in 1..{spec['max_new_tokens']}")
            job.update(prompts=prompts, max_new_tokens=T, enable_thinking=bool(payload.get("enable_thinking", True)))
        elif kind == "activations":
            texts, layers = payload.get("texts"), payload.get("layers")
            if not (isinstance(texts, list) and 1 <= len(texts) <= 8 and all(isinstance(t, str) and len(t) <= 20000 for t in texts)):
                raise ValueError("texts must be a list of 1-8 strings (≤ 20000 chars each)")
            if not (isinstance(layers, list) and 1 <= len(layers) <= 4 and all(isinstance(l, int) and 0 <= l < C.N_LAYERS for l in layers)):
                raise ValueError(f"layers must be a list of 1-4 ints in 0..{C.N_LAYERS - 1}")
            cr = payload.get("char_ranges")
            if cr is not None:
                if not (isinstance(cr, list) and len(cr) == len(texts)
                        and all(r is None or (isinstance(r, list) and len(r) <= 4) for r in cr)):
                    raise ValueError("char_ranges must be a list (one entry per text) of ≤ 4 [c0, c1] pairs or null")
                cr = [None if r is None else [[int(a), int(b)] for a, b in r] for r in cr]
            n_vec = sum(len(r) if r else 1 for r in (cr or [None] * len(texts))) * len(layers)
            if n_vec > 48:
                raise ValueError(f"{n_vec} vectors requested; at most 48 per call")
            job.update(texts=texts, layers=layers, char_ranges=cr)
        else:
            raise KeyError(kind)
        (pdir / "payload.json").write_text(json.dumps(job))
        try:
            out = _heavy(ad, kind, f"private/probes/{pid}/out.json", ["--payload", str(pdir / "payload.json")], push)
        finally:
            (pdir / "delta.npz").unlink(missing_ok=True)
        out.pop("wall_s", None)
        return out

    # ---- scoring -------------------------------------------------------------
    def _base(self, ad: Path) -> dict:
        return json.loads((ad / "private" / "base_eval.json").read_text())

    def _budget_error(self, delta: Mapping[str, np.ndarray], shapes: Mapping[str, tuple[int, ...]]) -> str | None:
        """Same L0-or-rank rule the runner applies afterwards; checked first so an over-budget
        edit fails fast instead of spending a GPU eval."""
        b = self.budget
        total = sum(int(np.prod(s)) for s in shapes.values())
        nnz = sum(int(np.count_nonzero(d)) for d in delta.values())
        if b.max_l0 is None or nnz / total <= b.max_l0:
            return None
        # Rank in the edit's own dtype: casting float32 to float64 shrinks matrix_rank's tolerance below
        # float32 rounding noise, so a materialised rank-1 float32 edit would count as full rank.
        if b.max_rank is not None and all(np.ndim(d) == 2 and int(np.linalg.matrix_rank(np.asarray(d))) <= b.max_rank
                                          for d in delta.values() if np.any(d)):
            return None
        return f"edit exceeds L0 {b.max_l0} and rank {b.max_rank}"

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        from weightbench.tracks.tr02_loops_lib.edits import validate_delta
        ad = art_dir.resolve()
        if sub.type != "delta_w" or not sub.delta:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "expected a non-empty delta_w submission"})
        shapes = _shapes(ad)
        err = validate_delta(sub.delta, shapes, self.constraint) or self._budget_error(sub.delta, shapes)
        if err:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": err})
        base = self._base(ad)
        s_base = base["S"]
        s_max = min(1.0, s_base + max(base["summary"]["loop_rate"], 0.05))
        nz = {k: np.asarray(v, np.float32) for k, v in sub.delta.items() if np.any(v)}
        if not nz:
            return Evaluation(s_plus=0.0, details={"reason": "ΔW = 0", "S_base": s_base, "S_edit": s_base})
        eid = uuid.uuid4().hex[:12]
        edir = ad / "private" / "evals" / eid
        edir.mkdir(parents=True, exist_ok=True)
        np.savez(edir / "delta.npz", **nz)
        try:
            res = _heavy(ad, "eval", f"private/evals/{eid}/out.json", ["--delta", str(edir / "delta.npz")],
                         [edir / "delta.npz"])
        finally:
            (edir / "delta.npz").unlink(missing_ok=True)
        s_edit = res["S"]
        mm_b, mm_e = base["mmlu"]["acc"], res["mmlu"]["acc"]
        d_col = scoring.collateral(mm_b, mm_e)
        if "by_concept" in base:   # knob level K > 1: S+ = min over families
            knob = knob_s_plus(res["by_concept"], base["by_concept"])
            details = {"S_base": s_base, "S_edit": s_edit, **knob, "base": base["summary"], "edited": res["summary"],
                       "mmlu_base": mm_b, "mmlu_edit": mm_e, "eval_seconds": res.get("wall_s"),
                       "runtime": res.get("runtime"), "eval_out": str(edir / "out.json")}
            if res.get("runtime") != base.get("runtime"):
                details["warning"] = "edited and base evals ran on different device/dtype"
            return Evaluation(s_plus=knob["s_plus_min"], d_col=d_col, details=details)
        details = {"S_base": s_base, "S_edit": s_edit, "S_max": s_max,
                   "base": base["summary"], "edited": res["summary"],
                   "mmlu_base": mm_b, "mmlu_edit": mm_e,
                   "eval_seconds": res.get("wall_s"), "runtime": res.get("runtime"), "eval_out": str(edir / "out.json")}
        if res.get("runtime") != base.get("runtime"):
            details["warning"] = "edited and base evals ran on different device/dtype"
        return Evaluation(s_plus=scoring.s_plus_write(s_edit, s_base, s_max), d_col=d_col, details=details)

    def _edit_from_job(self, ad: Path, step: str) -> Submission:
        from weightbench.tracks.tr02_loops_lib.edits import materialize
        ad = ad.resolve()
        p = ad / "private" / f"{step}.json"
        res = json.loads(p.read_text()) if p.exists() else _heavy(ad, step, f"private/{step}.json")
        if not res.get("ok"):
            raise RuntimeError(f"{step} edit could not be computed: {res.get('reason')}")
        report = ({"abstained": True, "reason": res.get("reason"), "rule": "holdout selection (reference.py)"}
                  if res.get("abstained") else None)    # honest abstention: ΔW = 0 scores at the floor
        return Submission("delta_w", report=report, delta=materialize(res["factors"], _shapes(ad)))

    def baseline(self, art_dir: Path) -> Submission:
        """Abliteration floor: difference-of-means loop direction (loop-tail vs clean residuals on
        the calibration generations) at the best-separating late layer, projected out of
        down_proj of the 4 layers ending there (SPEC §4: ROME / steering / abliteration)."""
        return self._edit_from_job(art_dir, "baseline")

    def reference(self, art_dir: Path) -> Submission:
        """Gated rank-one closing-token edit (``tr02_loops_lib.reference``): a prompt or loop-state gate fitted on
        calibration activations, selected on calibration-only development items + the MMLU dev split."""
        return self._edit_from_job(art_dir, "reference")


def knob_s_plus(edit: dict[str, dict], base: dict[str, dict]) -> dict:
    """K > 1 scoring: per family S+_k = s_plus_write(acc_edit_k, acc_base_k, S_max_k) with
    S_max_k = min(1, acc_base_k + max(loop_rate_base_k, 0.05)); the task S+ is the minimum, the mean is reported."""
    if set(edit) != set(base) or not base:
        raise ValueError(f"family sets differ: {sorted(edit)} vs {sorted(base)}")
    per, smax = {}, {}
    for c, b in base.items():
        smax[c] = min(1.0, b["acc"] + max(b["loop_rate"], 0.05))
        per[c] = scoring.s_plus_write(edit[c]["acc"], b["acc"], smax[c])
    return {"s_plus_by_concept": per, "s_max_by_concept": smax, "s_plus_min": min(per.values()),
            "s_plus_mean": sum(per.values()) / len(per)}


@register
class Loop001(LoopTask):
    id = "TR02-LOOP-001"
    goal_text = ("Raise Qwen3.5-0.8B's **GSM8K** accuracy under a fixed thinking-token budget by removing its "
                 "degenerate reasoning loops with a small weight edit.")
    eval_text = "{n} GSM8K test problems"
    metric_text = "exact-match accuracy of the final `\\boxed{}` number (after `</think>`)"


@register
class Loop002(LoopTask):
    id = "TR02-LOOP-002"
    goal_text = ("Raise Qwen3.5-0.8B's accuracy on **MATH-500 (levels 1–3)** under a fixed thinking-token budget "
                 "by removing its degenerate reasoning loops with a small weight edit.")
    eval_text = "{n} MATH-500 problems of difficulty level 1–3"
    metric_text = "accuracy of the final `\\boxed{}` answer (normalised LaTeX / numeric equivalence)"


@register
class Loop003(LoopTask):
    id = "TR02-LOOP-003"
    goal_text = ("Raise Qwen3.5-0.8B's **instruction-following** pass rate (IFEval prompts, thinking mode) by "
                 "removing its degenerate reasoning loops with a small weight edit.")
    eval_text = "{n} IFEval prompts whose instructions are programmatically checkable"
    metric_text = ("IFEval prompt-level strict accuracy of the final answer (text after `</think>`); every "
                   "instruction must pass")


@register
class Loop004(LoopTask):
    id = "TR02-LOOP-004"
    budget = Budget(n_forward=50, n_probe=20, max_l0=0.0005, max_rank=1, wall_clock_s=3 * 3600)
    constraint = "single_rank1"
    goal_text = ("Raise Qwen3.5-0.8B's accuracy on **hard MATH-500 problems (levels 4–5)** with a **single rank-1 "
                 "edit of one MLP matrix**. Held-out domain: your calibration problems are GSM8K grade-school "
                 "problems; the eval is competition math.")
    eval_text = "{n} MATH-500 problems of difficulty level 4–5"
    metric_text = "accuracy of the final `\\boxed{}` answer (normalised LaTeX / numeric equivalence)"
