"""Heavy steps for TR05-OUTLIER-001/002 (CPU; run through ``weightbench.heavy.run_heavy``).

    python -m weightbench.tracks.tr05_outlier_lib.jobs prepare   --art ART --out ART/private/prepare.json
    python -m weightbench.tracks.tr05_outlier_lib.jobs eval      --art ART [--delta D.npz] [--fp] --out OUT.json
    python -m weightbench.tracks.tr05_outlier_lib.jobs reference --art ART --out ART/private/reference_info.json

``eval`` returns the mean WikiText-2 NLL with the task's target linears under the task's simulated
deployment (``config.Deployment`` named in ``spec.json``; W8A8 if it names none; ``--fp``: full precision). Data preparation is shared with TR05-SPARSE (same pinned split and chunking).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from weightbench.tracks.tr05_sparsity_lib import calib as CB
from weightbench.tracks.tr05_sparsity_lib import config as SC
from weightbench.tracks.tr05_sparsity_lib import jobs as SJ

from . import config as OC
from . import quant as Q


def _spec(art: Path) -> dict:
    return json.loads((art / "private" / "spec.json").read_text())


def _only_level_layers(per_layer: dict[int, np.ndarray], sp: dict) -> dict[int, np.ndarray]:
    """Restrict per-layer scales to the layers of the build's knob level (``spec["layers"]``, written only for an
    explicit ``--level``). The default build has no "layers" key and keeps all layers. The
    scale algorithms themselves are untouched: migrating a layer outside the quantized set would be a no-op on the
    score but would edit tensors that are not editable at that level."""
    layers = sp.get("layers")
    if layers is None:
        return per_layer
    keep = set(layers)
    return {layer: v for layer, v in per_layer.items() if layer in keep}


def fake_quantize_modules(modules: list, deployment: OC.Deployment) -> list:
    """Fake-quantize torch linears in place under ``deployment``: per-output-channel weights (once) and a
    forward pre-hook for dynamic per-tensor activations. Returns the hook handles (call ``.remove()``)."""
    import torch

    w_qmax, a_qmax = deployment.weight_qmax, deployment.act_qmax

    def act_quant(module, args):
        x = args[0]
        s = x.abs().amax().clamp_min(1e-12) / a_qmax
        return (torch.round(x / s) * s,) + tuple(args[1:])

    handles = []
    for mod in modules:
        w = mod.weight.data.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / w_qmax
        mod.weight.data = (torch.round(w / scale) * scale).to(mod.weight.dtype)
        handles.append(mod.register_forward_pre_hook(act_quant))
    return handles


def mean_nll_quantized(art: Path, delta_path: str | None, full_precision: bool, batch: int = 8,
                       split: str = "eval") -> dict:
    """Mean NLL with the task's target linears under its deployment (``spec.json``; W8A8 if absent)."""
    import torch

    from weightbench.tracks.tr02_loops_lib import model as M

    sp = _spec(art)
    deployment = OC.from_spec(sp)
    rt = M.load(art)
    if delta_path:
        with np.load(delta_path) as z:
            rt.apply_delta({k: z[k] for k in z.files})
    handles = []
    if not full_precision:
        modules = dict(rt.model.named_modules())
        handles = fake_quantize_modules([modules[CB.module_name(n)] for n in sp["quantized"]], deployment)
    ids = torch.from_numpy(np.load(art / "private" / f"{split}_ids.npy"))
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            x = ids[i:i + batch].to(rt.dev)
            total += float(rt.model(input_ids=x, labels=x).loss) * len(x)
    for h in handles:
        h.remove()
    nll = total / len(ids)
    return {"nll": nll, "ppl": float(np.exp(nll)), "full_precision": full_precision, "deployment": deployment.name,
            "runtime": rt.info()}


ALPHAS = (0.25, 0.5, 0.75)
CLIP_QUANTILES = (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def _norm(layer: int) -> str:
    return f"{SC.PREFIX}layers.{layer}.post_attention_layernorm.weight"


def residual_rms(art: Path) -> dict[int, np.ndarray]:
    """Per layer, the weight-derived RMS of each residual channel at the MLP input
    (``sparsity jobs.residual_second_moments``: Σ_writers W Wᵀ + EᵀE / V, unit mean variance). No text."""
    names = SJ._names(art)
    dim = len(SJ._tensor(art, _norm(0)))
    moments = SJ.residual_second_moments(art, names, {(layer, 1) for layer in range(SC.N_LAYERS)}, dim)
    return {layer: np.sqrt(np.clip(np.diag(m), 0.0, None)) for (layer, _), m in moments.items()}


def _consumer(art: Path, kind: str, layer: int) -> np.ndarray:
    """The weights that absorb the scales: gate and up (001) or down_proj (002)."""
    if kind == "norm_to_gate_up":
        return np.concatenate([SJ._tensor(art, SC.mlp(layer, "gate_proj")), SJ._tensor(art, SC.mlp(layer, "up_proj"))])
    return SJ._tensor(art, SC.mlp(layer, "down_proj"))


def migrate(art: Path, kind: str, scales: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Exact, function-preserving migration with per-layer channel scales s: norm gain / s with gate and up
    columns · s (001), or up rows / s with down_proj columns · s (002)."""
    delta = {}
    for layer, s in scales.items():
        gate_name, up_name, down_name = (SC.mlp(layer, p) for p in SC.MLP_PROJS)
        up = SJ._tensor(art, up_name)
        if kind == "norm_to_gate_up":
            gamma, gate = SJ._tensor(art, _norm(layer)), SJ._tensor(art, gate_name)
            new_norm, (new_gate, new_up) = Q.migrate_norm(gamma, [gate, up], s)
            delta[_norm(layer)], delta[gate_name], delta[up_name] = new_norm - gamma, new_gate - gate, new_up - up
        else:
            down = SJ._tensor(art, down_name)
            new_up, new_down = Q.migrate_up_down(up, down, s)
            delta[up_name], delta[down_name] = new_up - up, new_down - down
    return {k: v.astype(np.float32) for k, v in delta.items()}


def proxy_scales(art: Path, kind: str, seed: int, resid_rms: dict[int, np.ndarray] | None) -> dict[int, np.ndarray]:
    """Data-free per-channel activation-scale proxies from the weights only: norm gain |1 + γ_j| for RMSNorm
    outputs (001), times the residual-geometry RMS of channel j when ``resid_rms`` is given (outlier channels
    of the residual stream survive RMSNorm; the gain alone misses them); for the down_proj input (002), the
    RMS of silu(gate x̃)·(up x̃) over synthetic x̃ ~ N(0, diag((1+γ)²))."""
    rng = np.random.default_rng(seed + 1)
    out = {}
    for layer in range(SC.N_LAYERS):
        gain = 1.0 + SJ._tensor(art, _norm(layer))
        if kind == "norm_to_gate_up":
            out[layer] = np.abs(gain) * (resid_rms[layer] if resid_rms is not None else 1.0)
        else:
            gate, up = SJ._tensor(art, SC.mlp(layer, "gate_proj")), SJ._tensor(art, SC.mlp(layer, "up_proj"))
            x = rng.standard_normal((SC.SYNTH_SAMPLES, gate.shape[1])) * gain
            g, u = x @ gate.T, x @ up.T
            out[layer] = np.sqrt(((g / (1.0 + np.exp(-g)) * u) ** 2).mean(0))
    return out


def calibrated_absmax(art: Path, kind: str, seed: int) -> dict[int, np.ndarray]:
    """Per layer, the true per-channel input absmax of the consumer (gate_proj for 001, down_proj for 002) on
    private calibration text (``calib.calib_ids``). Privileged: agents never see activations."""
    import torch

    from weightbench.tracks.tr02_loops_lib import model as M

    rt = M.load(art)
    proj = "gate_proj" if kind == "norm_to_gate_up" else "down_proj"
    modules = dict(rt.model.named_modules())
    targets = {layer: modules[CB.module_name(SC.mlp(layer, proj))] for layer in range(SC.N_LAYERS)}
    stats = CB.input_stats(rt.model, targets, torch.from_numpy(CB.calib_ids(art, seed)).to(rt.dev), gram=False)
    return {layer: st["absmax"] for layer, st in stats.items()}


def candidates(kind: str, deployment: OC.Deployment) -> list[tuple[str, str, float]]:
    """The reference's candidate set as ``(label, proxy, alpha_or_quantile)``. The W8A8 deployment and
    TR05-OUTLIER-002 use the three gain-proxy α's. Under a harsher deployment,
    001 also tries the residual-geometry proxy ``|1 + γ_j| · rms_j``, both as full SmoothQuant (``resid_alpha_*``)
    and as outlier-only clipping at several quantiles (``resid_clip_*``; full-channel scaling inflates the consumer weight
    ranges, while clipping only the extreme channels does not). Every task finally tries the privileged
    calibrated scales (true activation absmax on calibration text), as SmoothQuant and as clipping."""
    out = [(f"alpha_{a}", "gain", a) for a in ALPHAS]
    if kind == "norm_to_gate_up" and deployment.name != OC.LEGACY_DEPLOYMENT:
        out += [(f"resid_alpha_{a}", "gain_resid", a) for a in ALPHAS]
        out += [(f"resid_clip_{q}", "resid_clip", q) for q in CLIP_QUANTILES]
    out += [(f"calib_alpha_{a}", "calib", a) for a in ALPHAS]
    return out + [(f"calib_clip_{q}", "calib_clip", q) for q in CLIP_QUANTILES]


_SOURCE = {"gain": "gain", "gain_resid": "gain_resid", "resid_clip": "gain_resid", "calib": "calib",
           "calib_clip": "calib"}


def candidate_scales(art: Path, kind: str, proxy: str, value: float,
                     act: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per-layer scales of one candidate: clipping at quantile ``value`` or SmoothQuant with α = ``value``."""
    if proxy.endswith("clip"):
        return {layer: Q.clip_scales(a, value) for layer, a in act.items()}
    return {layer: Q.smooth_scales(a, _consumer(art, kind, layer), value) for layer, a in act.items()}


def reference(art: Path) -> dict:
    """Pick the candidate (proxy, α) with the lowest NLL under the task's deployment on the private dev
    chunks (disjoint from eval). 'No migration' is evaluated on dev as well, so the result says whether
    migration helps; the best data-free candidate is reported next to the (possibly privileged) winner."""
    sp = _spec(art)
    kind, deployment = sp["kind"], OC.from_spec(sp)
    cands = candidates(kind, deployment)
    act = {"gain": proxy_scales(art, kind, sp["seed"], None), "calib": calibrated_absmax(art, kind, sp["seed"])}
    if any(_SOURCE[p] == "gain_resid" for _, p, _ in cands):
        act["gain_resid"] = proxy_scales(art, kind, sp["seed"], residual_rms(art))
    act = {src: _only_level_layers(a, sp) for src, a in act.items()}
    dev = {"no_migration": mean_nll_quantized(art, None, False, split="dev")["nll"]}
    print(f"[tr05-outlier] {deployment.label}: dev NLL no_migration = {dev['no_migration']:.5f}", flush=True)
    best, best_nll = None, float("inf")
    # Candidates go to a per-job scratch dir, so two reference jobs on the same art dir (e.g. an orphaned remote job
    # and its retry) cannot delete each other's candidate files.
    import tempfile
    scratch = Path(tempfile.mkdtemp(prefix=".ref_", dir=art / "private"))
    for label, proxy, value in cands:
        path = scratch / f"reference_{label}.npz"
        np.savez(path, **migrate(art, kind, candidate_scales(art, kind, proxy, value, act[_SOURCE[proxy]])))
        dev[label] = mean_nll_quantized(art, str(path), False, split="dev")["nll"]
        print(f"[tr05-outlier] dev NLL {label} = {dev[label]:.5f}", flush=True)
        if dev[label] < best_nll:
            best, best_nll = (label, proxy, value), dev[label]
    (scratch / f"reference_{best[0]}.npz").replace(art / "private" / "reference_delta.npz")
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    data_free = min((c for c, p, _ in cands if not p.startswith("calib")), key=dev.__getitem__)
    return {"kind": kind, "deployment": deployment.name, "candidates": [c[0] for c in cands],
            "selected": best[0], "selected_proxy": best[1], "selected_alpha": best[2],
            "privileged": best[1].startswith("calib"), "best_data_free": data_free, "dev_nll": dev,
            "beats_no_migration_on_dev": best_nll < dev["no_migration"]}


# ------------------------------------------------------------------- named reference variants
DEV_SELECTED = "dev_selected"                  # the default: reference_delta.npz (may be privileged)
DATAFREE_DEVSEL = "datafree_devsel"            # best data-free candidate on dev NLL (selection saw dev text)
PROPAGATED = "propagated_clip"                 # fixed in advance: one setting per task, no selection
CALIB_PRIVILEGED = "calib_devsel_privileged"   # best calibrated candidate on dev NLL; privileged
REFERENCE_VARIANTS = (DEV_SELECTED, DATAFREE_DEVSEL, PROPAGATED, CALIB_PRIVILEGED)
PRIVILEGED_VARIANTS = frozenset({CALIB_PRIVILEGED})
# Fixed in advance: per-channel range a_j = |E x_j| + PROP_K · sd(x_j) from the propagated moments, then
# outlier-only clipping ``quant.clip_scales(a, PROP_CLIP_Q)``. Identical for both tasks and every seed.
PROP_K = 4.0
PROP_CLIP_Q = 0.99


def preregistered(kind: str) -> dict:
    """The fixed ``propagated_clip`` setting (no seed argument on purpose: it cannot vary across seeds)."""
    return {"kind": kind, "k": PROP_K, "clip_quantile": PROP_CLIP_Q,
            "site": "gate/up input (MLP RMSNorm output)" if kind == "norm_to_gate_up" else "down_proj input"}


def propagated_ranges(get, layers: list, h0, kind: str, k: float = PROP_K) -> dict[int, np.ndarray]:
    """Per layer, ``|mean| + k · sd`` of each consumer input channel from the analytic moment walk (``moments.walk``,
    nothing pruned): the RMSNorm output for 001, the SwiGLU hidden state for 002. No text, no sampling."""
    from weightbench.tracks.tr05_sparsity_lib import moments as MO

    out: dict[int, np.ndarray] = {}

    def on_mlp(layer: int, x, sw) -> None:
        if kind == "norm_to_gate_up":
            mu, var = x.mu, np.clip(np.diag(x.cov), 0.0, None)
        else:
            mu, var = sw.mu, sw.var
        out[layer] = np.abs(mu) + k * np.sqrt(var)

    MO.walk(get, layers, h0, on_mlp=on_mlp)
    return out


def propagated_scales(ranges: dict[int, np.ndarray], q: float = PROP_CLIP_Q) -> dict[int, np.ndarray]:
    return {layer: Q.clip_scales(a, q) for layer, a in ranges.items()}


def reference_variant(art: Path, variant: str) -> dict:
    """Write ``private/reference_<variant>.npz`` for a named variant other than ``dev_selected``."""
    sp = _spec(art)
    kind = sp["kind"]
    info: dict = {"variant": variant, "privileged": variant in PRIVILEGED_VARIANTS}
    if variant == PROPAGATED:
        names = SJ._names(art)
        dim = len(SJ._tensor(art, _norm(0)))
        ranges = propagated_ranges(lambda n: SJ._tensor(art, n), SJ.layer_names(names),
                                   SJ.embedding_moments(art, names, dim), kind)
        delta = migrate(art, kind, propagated_scales(_only_level_layers(ranges, sp)))
        info.update(preregistered=preregistered(kind))
    elif variant in (DATAFREE_DEVSEL, CALIB_PRIVILEGED):
        ref = json.loads((art / "private" / "reference_info.json").read_text())
        cands = {c[0]: c for c in candidates(kind, OC.from_spec(sp))}
        if variant == DATAFREE_DEVSEL:
            label = ref["best_data_free"]
        else:
            label = min((c for c in ref["dev_nll"] if c.startswith("calib")), key=ref["dev_nll"].__getitem__)
        _, proxy, value = cands[label]
        if _SOURCE[proxy] == "calib":
            act = calibrated_absmax(art, kind, sp["seed"])
        else:
            act = proxy_scales(art, kind, sp["seed"], residual_rms(art) if _SOURCE[proxy] == "gain_resid" else None)
        delta = migrate(art, kind, candidate_scales(art, kind, proxy, value, _only_level_layers(act, sp)))
        info.update(rebuilt=label)
    else:
        raise ValueError(f"unknown TR05-OUTLIER reference variant {variant!r}; choose from {REFERENCE_VARIANTS[1:]}")
    np.savez(art / "private" / f"reference_{variant}.npz", **delta)
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "eval", "reference", "reference_variant"])
    ap.add_argument("--variant")
    ap.add_argument("--art", required=True)
    ap.add_argument("--delta")
    ap.add_argument("--fp", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    art = Path(a.art)
    t0 = time.time()
    if a.step == "prepare":
        out = SJ.prepare(art)
    elif a.step == "eval":
        out = mean_nll_quantized(art, a.delta, a.fp)
    elif a.step == "reference_variant":
        out = reference_variant(art, a.variant)
    else:
        out = reference(art)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
