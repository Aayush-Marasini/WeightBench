"""TR-05 Data-Free Outlier Migration: TR05-OUTLIER-001/002, SPEC §2 TR-05.

Qwen3.5-0.8B is evaluated with its MLP linears under a simulated quantized deployment (dynamic per-tensor
activations, per-channel weights; ``tr05_outlier_lib.config``): W8A8 for 002, W8A6 (int6 activations) for 001
(W8A8 leaves too little quantization damage on 001's targets to separate methods). A few large input channels inflate the activation scale.
The agent must migrate those outliers into the weights analytically, without calibration data and
without running the model (``N_forward = 0``):
- 001: between each ``post_attention_layernorm`` gain and the columns of ``gate_proj`` / ``up_proj``;
- 002: between the rows of ``up_proj`` and the columns of ``down_proj``.
Score: the fraction of the quantization damage (held-out WikiText-2 NLL) recovered.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import ClassVar

import numpy as np

from weightbench import scoring
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr05_outlier_lib import config as OC
from weightbench.tracks.tr05_sparsity import install_model
from weightbench.tracks.tr05_sparsity_lib import config as C

JOBS = "weightbench.tracks.tr05_outlier_lib.jobs"
# Copy of jobs.REFERENCE_VARIANTS (tests check that they agree); the first is the default.
REFERENCE_VARIANTS = ("dev_selected", "datafree_devsel", "propagated_clip", "calib_devsel_privileged")


def _heavy(art_dir: Path, step: str, out_name: str, extra: list[str] = (), push: list[Path] = ()) -> dict:
    from weightbench.heavy import run_heavy

    art_dir = art_dir.resolve()
    return run_heavy(JOBS, [step, "--art", str(art_dir), *extra], art_dir, push=list(push),
                     out_name=out_name, timeout=6 * 3600, gpu=False)


def _layers(proj: str) -> list[str]:
    return [C.mlp(l, proj) for l in range(C.N_LAYERS)]


def _norms() -> list[str]:
    return [f"{C.PREFIX}layers.{l}.post_attention_layernorm.weight" for l in range(C.N_LAYERS)]


# Difficulty knob ``propagation_depth`` (SPEC §5): bounded explicit levels L0..L3 quantize (and make editable) only
# the MLP linears of the last min(24, 3^L) layers: L0 = layer 23, L1 = 21-23, L2 = 15-23, L3 = all 24 (= the
# unversioned default, which is kept unchanged). The layers are a suffix of the stack, derived from the level.
LEVELS: tuple[int, ...] = (0, 1, 2, 3)


def level_layers(level: int) -> list[int]:
    """Layers quantized at knob level ``level``: the last min(N_LAYERS, 3^L)."""
    from weightbench import knobs

    if level not in LEVELS:
        raise ValueError(f"TR05-OUTLIER: supported knob levels are {LEVELS}, got {level}")
    n = min(C.N_LAYERS, int(knobs.value(knobs.get("TR05"), level)))
    return list(range(C.N_LAYERS - n, C.N_LAYERS))


def _layer_of(name: str) -> int:
    return int(name.split("layers.")[1].split(".", 1)[0])


def _restrict(names: list[str], layers: list[int] | None) -> list[str]:
    if layers is None:
        return list(names)
    keep = set(layers)
    return [n for n in names if _layer_of(n) in keep]


class _OutlierTask(Task):
    track = "TR05"
    tier = "intermediate"
    submission = "delta_w"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=3 * 3600)
    kind: ClassVar[str]
    quantized: ClassVar[list[str]]
    editable: ClassVar[list[str]]
    goal: ClassVar[str]
    deployment: ClassVar[str]                       # a key of tr05_outlier_lib.config.DEPLOYMENTS
    deployment_env: ClassVar[str | None] = None     # build option: env var overriding ``deployment``
    knob_levels = LEVELS                            # unversioned default kept (all 24 layers = saturated L3)

    @property
    def layers(self) -> list[int] | None:
        """Quantized layers of an explicit knob level; None for the default build (all layers)."""
        return level_layers(self.level) if self.level is not None else None

    @property
    def level_quantized(self) -> list[str]:
        return _restrict(self.quantized, self.layers)

    @property
    def level_editable(self) -> list[str]:
        return _restrict(self.editable, self.layers)

    def build_deployment(self) -> OC.Deployment:
        """The deployment a fresh build simulates (written to spec.json; evaluation reads it back from there)."""
        return OC.build_choice(self.deployment, self.deployment_env)

    def _deployment(self, art_dir: Path) -> OC.Deployment:
        return OC.from_spec(json.loads((art_dir / "private" / "spec.json").read_text()))

    def build(self, art_dir: Path, seed: int) -> None:
        from safetensors import safe_open

        pub, priv = art_dir / "public", art_dir / "private"
        install_model(pub)
        with safe_open(str(pub / "model" / C.WEIGHTS_FILE), "pt") as f:
            shapes = {n: list(f.get_slice(n).get_shape()) for n in self.level_editable}
        (pub / "editable.json").write_text(json.dumps(shapes, indent=1))
        spec = {"task": self.id, "seed": seed, "kind": self.kind, "quantized": self.level_quantized,
                "deployment": self.build_deployment().name,
                "model": [C.MODEL_ID, C.MODEL_REV], "dataset": C.DATASET, "seq_len": C.SEQ_LEN}
        if self.level is not None:          # explicit level only; the default spec stays byte-identical
            from weightbench import knobs

            spec.update(level=self.level, layers=self.layers)
            # Public: editable.json already names the layers, so the knob record leaks nothing.
            (pub / "knob.json").write_text(json.dumps(
                {**knobs.describe(self.track, self.level), "layers": self.layers}, indent=1))
        (priv / "spec.json").write_text(json.dumps(spec, indent=1))
        _heavy(art_dir, "prepare", "private/prepare.json")
        _heavy(art_dir, "reference", "private/reference_info.json")
        if os.environ.get("WB_HEAVY", "local") != "local":
            from weightbench.heavy import pull_dir

            pull_dir(art_dir, "private")
        fp = _heavy(art_dir, "eval", "private/eval_fp.json", ["--fp"])
        quant = _heavy(art_dir, "eval", "private/eval_quant.json")
        (priv / "calibration.json").write_text(json.dumps(
            {"nll_fp": fp["nll"], "nll_quant": quant["nll"], "deployment": self._deployment(art_dir).name}, indent=1))

    def _calibration(self, art_dir: Path) -> dict:
        return json.loads((art_dir / "private" / "calibration.json").read_text())

    def _shapes(self, art_dir: Path) -> dict[str, list[int]]:
        return json.loads((art_dir / "public" / "editable.json").read_text())

    def prompt(self, art_dir: Path) -> str:
        cal = self._calibration(art_dir)
        dep = self._deployment(art_dir)
        q = dep.label
        quantized = self.level_quantized
        if self.layers is None:
            which = "all 24 layers"
        else:
            which = (f"knob level {self.level}: only layers {self.layers[0]}-{self.layers[-1]}; every other layer stays in "
                     "full precision")
        return f"""# {self.id}: Data-Free Outlier Migration ({self.goal})

Track TR-05 (Hardware-Aware WSP, SYSTEMS). Subject: **Qwen3.5-0.8B** (`{C.MODEL_ID}` @ `{C.MODEL_REV[:12]}`), weights in
`model/`. MLP: `mlp(x) = down_proj(silu(gate_proj x) * up_proj x)` applied to `post_attention_layernorm(h)`; RMSNorm is
zero-centred (`y = x / rms(x) * (1 + w)`); PyTorch convention `y = x @ W.T`.

## Deployment being simulated
The {len(quantized)} linear layers `{quantized[0].split('layers.')[1].split('.', 1)[1]}` ({which}) run in **{q}**:
their input activations are quantized to int{dep.act_bits} with ONE dynamic scale per tensor (`max|x| / {dep.act_qmax:.0f}`), their weights to
int{dep.weight_bits} per output channel. A few input channels with very large magnitude ("outliers") inflate the activation scale.

## Goal
{self.goal}. Function-preserving rescalings exist: multiplying input channel j of the consumer matrix by `s_j` and
dividing the producer of that channel by `s_j` leaves the full-precision model unchanged but changes what int{dep.act_bits} sees.
You may edit only the tensors listed in `editable.json`.

## Scoring (server-side)
Mean next-token NLL on held-out WikiText-2 chunks of {C.SEQ_LEN} tokens, evaluated in {q} as above.
Full precision: {cal['nll_fp']:.4f}. {q} without migration: {cal['nll_quant']:.4f}.
`S+ = clip((NLL_{q}_unmigrated − NLL_yours) / (NLL_{q}_unmigrated − NLL_fp), 0, 1)`.

## Rules (violations score 0)
- **Data-free, `N_forward = 0`:** no calibration data, no forward passes (do not run the model or any part of it on
  text or tokens). Derive activation scales from the weights (norm gains, products of weight matrices, closed-form
  statistics). No gradient descent; no copying the weights out of the metered array to compute with them
  unmetered. Do not read files outside your working directory.

## Submission
`np.savez("delta.npz", **{{name: dW}})` with float32 ΔW of the full shapes for the tensors you change (computed from
the float32 values of the bf16 weights), then `python wb.py submit-delta delta.npz`. You may resubmit.
"""

    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray]:
        return {}   # λ = 0 for TR-05; the editable set is enforced in evaluate()

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        if sub.type != "delta_w" or not sub.delta:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "expected a non-empty delta_w submission"})
        shapes = self._shapes(art_dir)
        for name, value in sub.delta.items():
            if name not in shapes:
                return Evaluation(s_plus=0.0, valid=False, details={"reason": f"{name} is not editable in this task"})
            if list(np.shape(value)) != shapes[name] or not np.all(np.isfinite(value)):
                return Evaluation(s_plus=0.0, valid=False, details={"reason": f"{name}: bad shape or non-finite"})
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
        s_plus = scoring.s_plus_write(-res["nll"], -cal["nll_quant"], -cal["nll_fp"])
        return Evaluation(s_plus=s_plus, details={"nll_w8a8": res["nll"], **cal})

    def baseline(self, art_dir: Path) -> Submission:
        """Floor: deploy without migration (identity scaling; ΔW = 0 on one editable tensor). S+ = 0."""
        name, shape = next(iter(self._shapes(art_dir).items()))
        return Submission("delta_w", delta={name: np.zeros(shape, np.float32)})

    reference_variants = REFERENCE_VARIANTS

    def reference(self, art_dir: Path, variant: str | None = None) -> Submission:
        """Named reference. ``dev_selected`` (default): the candidate (proxy, α or
        clip quantile) with the lowest private-dev NLL, possibly privileged (``reference_info.json``);
        ``datafree_devsel``: best data-free candidate on dev; ``propagated_clip``: fixed clipping from
        analytically propagated moments (one setting, no selection); ``calib_devsel_privileged``: best calibrated
        candidate on dev, privileged. Non-default variants are computed in a heavy step on first use."""
        variant = variant or self.default_reference_variant()
        if variant not in REFERENCE_VARIANTS:
            raise ValueError(f"unknown TR05-OUTLIER reference variant {variant!r}; choose from {REFERENCE_VARIANTS}")
        path = art_dir / "private" / ("reference_delta.npz" if variant == REFERENCE_VARIANTS[0]
                                      else f"reference_{variant}.npz")
        if not path.exists():
            _heavy(art_dir, "reference_variant", f"private/reference_{variant}.json", ["--variant", variant])
            if os.environ.get("WB_HEAVY", "local") != "local":
                from weightbench.heavy import pull_dir

                pull_dir(art_dir, "private")
        with np.load(path) as z:
            return Submission("delta_w", delta={k: z[k] for k in z.files})


@register
class OutlierNormToGateUp(_OutlierTask):
    id = "TR05-OUTLIER-001"
    kind = "norm_to_gate_up"
    quantized = _layers("gate_proj") + _layers("up_proj")
    editable = _norms() + _layers("gate_proj") + _layers("up_proj")
    goal = "migrate activation outliers of the MLP input from the RMSNorm gains into gate_proj / up_proj"
    deployment = OC.OUTLIER_001_DEPLOYMENT
    deployment_env = OC.OUTLIER_001_DEPLOYMENT_ENV


@register
class OutlierUpToDown(_OutlierTask):
    id = "TR05-OUTLIER-002"
    kind = "up_to_down"
    quantized = _layers("down_proj")
    editable = _layers("up_proj") + _layers("down_proj")
    goal = "migrate activation outliers of the MLP hidden state from up_proj into down_proj"
    deployment = OC.OUTLIER_002_DEPLOYMENT
