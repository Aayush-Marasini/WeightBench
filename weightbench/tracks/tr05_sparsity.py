"""TR-05 Data-Free Compensation-Aware 2:4 Sparsity: TR05-SPARSE-001..004, SPEC §2 TR-05.

The agent gets Qwen3.5-0.8B's weights and a list of target matrices. It must make every target 2:4
sparse and update the remaining weights analytically, without calibration data and without running
the model (``N_forward = 0``), so that the model's language-modelling quality drops as little as
possible. Score: the fraction of naive magnitude-pruning damage (held-out WikiText-2 NLL) recovered.

Model execution (data prep, NLL evaluation) runs through ``weightbench.heavy.run_heavy`` ->
``tr05_sparsity_lib.jobs`` on CPU.

Difficulty knob ``propagation_depth`` (SPEC §5, ``weightbench.knobs``): SPARSE-001 and SPARSE-002 share the TR-05
sparsity ladder. Level L prunes the MLPs of the last min(24, base · 3^L) layers, base = 1 for SPARSE-001 (L0 = layer 23;
L1 = 21-23; L2 = 15-23; L3 = all 24) and base = 4 for SPARSE-002 (L0 = layers 20-23; L1 = 12-23; L2 and up = all 24).
``weightbench build --task T --level L`` builds any level. SPARSE-003 (every MLP) is the saturated rung of that ladder
and keeps fixed targets by default; its explicit levels L0..L2 run the ladder from the INPUT side (MLPs of the first
3^L layers: L0 = layer 0, L1 = 0-2, L2 = 0-8), distinct from every SPARSE-001 rung. SPARSE-004 has bounded explicit levels L0..L3 over the attention input projections of the last
min(24, 3^L) layers (L0 = layer 23, L1 = 21-23, L2 = 15-23, L3 = all 24); its unversioned default (all layers) is kept
and equals L3, so its L0 is not the default build.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import ClassVar

import numpy as np

from weightbench import knobs, scoring
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr05_sparsity_lib import config as C
from weightbench.tracks.tr05_sparsity_lib import sparsity as SP

JOBS = "weightbench.tracks.tr05_sparsity_lib.jobs"
# Pure-Python copy of jobs.REFERENCE_VARIANTS (jobs is a heavy module); tests check that they agree.
REFERENCE_VARIANTS = ("dev_selected", "datafree_devsel", "propagated_obs", "sparsegpt_calib_privileged")


def _heavy(art_dir: Path, step: str, out_name: str, extra: list[str] = (), push: list[Path] = ()) -> dict:
    from weightbench.heavy import run_heavy

    art_dir = art_dir.resolve()
    return run_heavy(JOBS, [step, "--art", str(art_dir), *extra], art_dir, push=list(push),
                     out_name=out_name, timeout=6 * 3600, gpu=False)


def _pull_private(art_dir: Path) -> None:
    if os.environ.get("WB_HEAVY", "local") != "local":
        from weightbench.heavy import pull_dir

        pull_dir(art_dir, "private")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def install_model(public_dir: Path) -> Path:
    """Hard-link (or copy) the pinned Qwen3.5-0.8B snapshot files into ``public_dir/model``."""
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(C.MODEL_ID, revision=C.MODEL_REV, allow_patterns=list(C.MODEL_FILES)))
    model_dir = public_dir / "model"
    model_dir.mkdir(exist_ok=True)
    for f in C.MODEL_FILES:
        src = (snap / f).resolve()
        try:
            os.link(src, model_dir / f)
        except OSError:
            shutil.copyfile(src, model_dir / f)
    return model_dir


class _SparseTask(Task):
    track = "TR05"
    tier = "intermediate"
    submission = "delta_w"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=3 * 3600)
    spec_key: ClassVar[str]

    # ---- generation ---------------------------------------------------------------------
    def build(self, art_dir: Path, seed: int) -> None:
        from safetensors import safe_open

        pub, priv = art_dir / "public", art_dir / "private"
        install_model(pub)
        with safe_open(str(pub / "model" / C.WEIGHTS_FILE), "pt") as f:
            names = list(f.keys())
            targets = C.resolve_targets(self.id, names, self.level)
            shapes = {t: list(f.get_slice(t).get_shape()) for t in targets}
        (pub / "targets.json").write_text(json.dumps(shapes, indent=1))
        (priv / "spec.json").write_text(json.dumps(self.spec(seed, targets), indent=1))
        knob = self.knob_record()
        if knob is not None:
            # Public: the target list (targets.json) already names the layers, so the knob record leaks nothing.
            (pub / "knob.json").write_text(json.dumps(knob, indent=1))
        _heavy(art_dir, "prepare", "private/prepare.json")
        _heavy(art_dir, "baseline", "private/baseline_info.json")
        _heavy(art_dir, "reference", "private/reference_info.json")
        _pull_private(art_dir)
        dense = _heavy(art_dir, "eval", "private/eval_dense.json")
        magnitude = _heavy(art_dir, "eval", "private/eval_magnitude.json",
                           ["--delta", str((priv / "baseline_delta.npz").resolve())])
        (priv / "calibration.json").write_text(json.dumps(
            {"nll_dense": dense["nll"], "nll_magnitude": magnitude["nll"], "runtime": dense["runtime"]}, indent=1))

    def spec(self, seed: int, targets: list[str]) -> dict:
        """Generator settings written to private/spec.json. A non-default knob level also records the level and its
        layers; the default level writes only the base settings (byte-identical builds; knob.json has the level)."""
        spec = {"task": self.id, "seed": seed, "targets": targets, "model": [C.MODEL_ID, C.MODEL_REV],
                "dataset": C.DATASET, "seq_len": C.SEQ_LEN}
        if self._nondefault_level():
            spec.update(level=self.level, layers=C.ladder_layers(self.level, self.id))
        return spec

    def _nondefault_level(self) -> bool:
        return self.level is not None and self.level != self.knob_level

    def knob_record(self) -> dict | None:
        """Public knob.json (as TR05-DEAD) plus the derived layer list; None for tasks without a wired knob."""
        if self.level is None:
            return None
        return {**knobs.describe(self.track, self.level), "layers": C.ladder_layers(self.level, self.id)}

    def _title(self) -> str:
        """Prompt title: the configured one, or the ladder rung for a non-default knob level."""
        if not self._nondefault_level():
            return C.TASKS[self.id]["title"]
        layers = C.ladder_layers(self.level, self.id)
        what = "attention input projections" if self.id in C.ATTN_LADDER_BASES else "all MLP projections"
        return f"{what} of layers {layers[0]}-{layers[-1]} (knob level {self.level})"

    def _calibration(self, art_dir: Path) -> dict:
        return _read(art_dir / "private" / "calibration.json")

    def _targets(self, art_dir: Path) -> dict[str, list[int]]:
        return _read(art_dir / "public" / "targets.json")

    # ---- agent-facing text ------------------------------------------------------------------
    def prompt(self, art_dir: Path) -> str:
        cal = self._calibration(art_dir)
        targets = self._targets(art_dir)
        listing = "\n".join(f"- `{n}` {s}" for n, s in list(targets.items())[:12])
        more = f"\n- … ({len(targets) - 12} more; full list in `targets.json`)" if len(targets) > 12 else ""
        return f"""# {self.id}: Data-Free 2:4 Sparsity: {self._title()}

Track TR-05 (Hardware-Aware WSP, SYSTEMS). Subject: **Qwen3.5-0.8B** (`{C.MODEL_ID}` @ `{C.MODEL_REV[:12]}`),
weights in `model/` (bf16 safetensors, config, tokenizer).

## Goal
Make every **target matrix** 2:4 sparse and compensate analytically so the model stays as good as possible.
2:4 means: in every row of the target weight `W [out, in]` (PyTorch `y = x @ W.T`), every group of 4 consecutive
input columns `[4g, 4g+4)` has **at least 2 exact zeros** after your edit. You may change any entries of the target
matrices (also the kept ones); no other parameter may change.

Targets ({len(targets)} matrices):
{listing}{more}

## Scoring (server-side)
- Quality = mean next-token NLL on held-out WikiText-2 (test split) chunks of {C.SEQ_LEN} tokens.
  Dense model: {cal['nll_dense']:.4f}. Naive magnitude 2:4 (keep the 2 largest |w| per group): {cal['nll_magnitude']:.4f}.
- `S+ = clip((NLL_magnitude − NLL_yours) / (NLL_magnitude − NLL_dense), 0, 1)`: the fraction of the magnitude-pruning
  damage you recover. Edits that are not exactly 2:4 on every target, or that touch other parameters, score 0.

## Rules (violations score 0)
- **Data-free, `N_forward = 0`:** no calibration data and no forward passes. Do not run the model (or any part of it)
  on text, token embeddings or tokens. There are no probes. Allowed: analysis of the weights themselves (norms,
  SVD, products of weight matrices, RMSNorm gains, closed-form least squares on weight-derived quantities).
- No gradient descent; no copying the weights out of the metered array to compute with them unmetered.
  Do not try to read files outside your working directory.

## Architecture notes
24 decoder layers, hidden 1024, MLP 3584: `mlp(x) = down_proj(silu(gate_proj x) * up_proj x)`, applied to
`post_attention_layernorm(h)`; RMSNorm weights are zero-centred (`y = x / rms(x) * (1 + w)`). Attention/linear-attention
input projections read `input_layernorm(h)`. Load tensors with
`safetensors.safe_open("model/{C.WEIGHTS_FILE}", "pt").get_tensor(name).float()`.

## Submission
Save an `.npz` mapping **every** target name to a float32 ΔW of the full shape (computed from the float32 values of
the bf16 weights; ΔW is added in float32, so `W + ΔW` must contain exact zeros), then
`python wb.py submit-delta delta.npz`. You may resubmit; the last submission counts.
"""

    # ---- scoring ------------------------------------------------------------------------------
    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray]:
        return {}   # λ = 0 for TR-05: the 2:4 mask itself is the budget, checked in evaluate()

    def _validation_error(self, art_dir: Path, delta: dict[str, np.ndarray]) -> str | None:
        from safetensors import safe_open

        targets = self._targets(art_dir)
        extra = sorted(set(delta) - set(targets))
        if extra:
            return f"ΔW touches non-target parameters, e.g. {extra[0]}"
        missing = sorted(set(targets) - set(delta))
        if missing:
            return f"every target must be 2:4 sparse; missing {missing[0]} (and {len(missing) - 1} more)"
        with safe_open(str(art_dir / "public" / "model" / C.WEIGHTS_FILE), "pt") as f:
            for name, shape in targets.items():
                d = np.asarray(delta[name], np.float32)
                if list(d.shape) != shape or not np.all(np.isfinite(d)):
                    return f"{name}: ΔW must be finite with shape {shape}"
                if not SP.is_24_sparse(f.get_tensor(name).float().numpy() + d):
                    return f"{name}: W + ΔW is not 2:4 sparse"
        return None

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        if sub.type != "delta_w" or not sub.delta:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "expected a non-empty delta_w submission"})
        error = self._validation_error(art_dir, sub.delta)
        if error:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": error})
        cal = self._calibration(art_dir)
        eval_dir = art_dir.resolve() / "private" / "evals" / uuid.uuid4().hex[:12]
        eval_dir.mkdir(parents=True)
        delta_path = eval_dir / "delta.npz"
        np.savez(delta_path, **{k: np.asarray(v, np.float32) for k, v in sub.delta.items()})
        try:
            res = _heavy(art_dir, "eval", (eval_dir / "result.json").relative_to(art_dir.resolve()).as_posix(),
                         ["--delta", str(delta_path)], push=[delta_path])
        finally:
            delta_path.unlink(missing_ok=True)
        headroom = cal["nll_magnitude"] - cal["nll_dense"]
        s_plus = scoring.s_plus_write(-res["nll"], -cal["nll_magnitude"], -cal["nll_dense"]) if headroom > 0 else 0.0
        return Evaluation(s_plus=s_plus, details={"nll": res["nll"], "ppl": res["ppl"], **cal})

    def _cached(self, art_dir: Path, which: str) -> Submission:
        with np.load(art_dir / "private" / f"{which}_delta.npz") as z:
            return Submission("delta_w", delta={k: z[k] for k in z.files})

    def baseline(self, art_dir: Path) -> Submission:
        """SPEC §4 floor: naive magnitude 2:4 pruning (data-free, no compensation). S+ = 0 by construction."""
        return self._cached(art_dir, "baseline")

    reference_variants = REFERENCE_VARIANTS

    def reference(self, art_dir: Path, variant: str | None = None) -> Submission:
        """Named reference (``runner run --agent reference --variant V``):
        ``dev_selected`` (default): lowest private-dev NLL over ``jobs.VARIANTS`` and the calibrated
        SparseGPT, so it can be privileged (``reference_info.json`` says which); ``datafree_devsel``: the best
        weights-only variant on dev; ``propagated_obs``: fixed OBS under analytically propagated moments
        (no selection); ``sparsegpt_calib_privileged``: calibrated SparseGPT, privileged. Non-default variants are
        computed in a heavy step on first use."""
        variant = variant or self.default_reference_variant()
        if variant not in REFERENCE_VARIANTS:
            raise ValueError(f"unknown TR05-SPARSE reference variant {variant!r}; choose from {REFERENCE_VARIANTS}")
        if variant == REFERENCE_VARIANTS[0]:
            return self._cached(art_dir, "reference")
        path = art_dir / "private" / f"reference_{variant}.npz"
        if not path.exists():
            _heavy(art_dir, "reference_variant", f"private/reference_{variant}.json", ["--variant", variant])
            _pull_private(art_dir)
        with np.load(path) as z:
            return Submission("delta_w", delta={k: z[k] for k in z.files})


@register
class SparseOneLayer(_SparseTask):
    id = "TR05-SPARSE-001"
    tier = "intermediate"
    knob_level = 0      # the TR-05 sparsity ladder: MLPs of the last 3^0 = 1 layer (layer 23)


@register
class SparseLateMLPs(_SparseTask):
    id = "TR05-SPARSE-002"
    tier = "intermediate"
    knob_level = 0      # the TR-05 sparsity ladder with base 4: MLPs of the last 4 · 3^0 = 4 layers (layers 20-23)


@register
class SparseAllMLPs(_SparseTask):
    id = "TR05-SPARSE-003"
    tier = "intermediate"
    # Bounded explicit levels L0..L2 counted from the input side (MLPs of the FIRST 3^L layers; config.PREFIX_LADDER_
    # BASES). The unversioned default (every MLP) is kept; it is the saturated rung of SPARSE-001's suffix ladder.
    knob_levels = C.PREFIX_LEVELS


@register
class SparseAttentionInputs(_SparseTask):
    id = "TR05-SPARSE-004"
    tier = "intermediate"
    # Bounded explicit levels (as TR01-BEH): the unversioned default (every layer) is kept and equals the saturated L3.
    knob_levels = C.ATTN_LEVELS
