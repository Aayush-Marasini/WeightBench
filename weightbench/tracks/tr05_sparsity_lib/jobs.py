"""Heavy steps for TR05-SPARSE-001..004 (CPU-feasible; run through ``weightbench.heavy.run_heavy``).

    python -m weightbench.tracks.tr05_sparsity_lib.jobs prepare   --art ART --out ART/private/prepare.json
    python -m weightbench.tracks.tr05_sparsity_lib.jobs eval      --art ART [--delta D.npz] --out OUT.json
    python -m weightbench.tracks.tr05_sparsity_lib.jobs baseline  --art ART --out ART/private/baseline_info.json
    python -m weightbench.tracks.tr05_sparsity_lib.jobs reference --art ART --out ART/private/reference_info.json

``prepare`` tokenises the pinned WikiText-2 test split into 512-token chunks and draws disjoint eval
and dev sets from the task seed. ``eval`` returns the mean next-token NLL on the eval chunks.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import config as C
from . import sparsity as SP


def _spec(art: Path) -> dict:
    return json.loads((art / "private" / "spec.json").read_text())


def _weights_file(art: Path) -> Path:
    return art / "public" / "model" / C.WEIGHTS_FILE


def _tensor(art: Path, name: str) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(str(_weights_file(art)), "pt") as f:
        return f.get_tensor(name).float().numpy()


def _names(art: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(_weights_file(art)), "pt") as f:
        return list(f.keys())


def prepare(art: Path) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    sp = _spec(art)
    name, cfg, rev = C.DATASET
    text = "\n".join(t for t in load_dataset(name, cfg, split="test", revision=rev)["text"] if t.strip())
    ids = AutoTokenizer.from_pretrained(art / "public" / "model")(text)["input_ids"]
    n_chunks = len(ids) // C.SEQ_LEN
    chunks = np.asarray(ids[:n_chunks * C.SEQ_LEN], np.int64).reshape(n_chunks, C.SEQ_LEN)
    order = np.random.default_rng(sp["seed"]).permutation(n_chunks)
    np.save(art / "private" / "eval_ids.npy", chunks[order[:C.EVAL_SEQS]])
    np.save(art / "private" / "dev_ids.npy", chunks[order[C.EVAL_SEQS:C.EVAL_SEQS + C.DEV_SEQS]])
    return {"n_tokens": len(ids), "n_chunks": n_chunks, "eval_seqs": C.EVAL_SEQS, "dev_seqs": C.DEV_SEQS}


def mean_nll(art: Path, delta_path: str | None, split: str = "eval", batch: int = 8) -> dict:
    import torch

    from weightbench.tracks.tr02_loops_lib import model as M

    rt = M.load(art)
    if delta_path:
        with np.load(delta_path) as z:
            rt.apply_delta({k: z[k] for k in z.files})
    ids = torch.from_numpy(np.load(art / "private" / f"{split}_ids.npy"))
    losses = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            x = ids[i:i + batch].to(rt.dev)
            losses.append(float(rt.model(input_ids=x, labels=x).loss) * len(x))
    nll = sum(losses) / len(ids)
    return {"nll": nll, "ppl": float(np.exp(nll)), "split": split, "runtime": rt.info()}


def baseline(art: Path) -> dict:
    targets = _spec(art)["targets"]
    delta = {}
    for name in targets:
        w = _tensor(art, name)
        delta[name] = (SP.magnitude_24(w) - w).astype(np.float32)
    np.savez(art / "private" / "baseline_delta.npz", **delta)
    return {"targets": len(targets)}


def input_covariance(art: Path, name: str, names: list[str], rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Data-free input covariance for target ``name``.

    Inputs of q/k/v, gate/up and linear-attention input projections are RMSNorm outputs:
    ``x̃ = z ⊙ (1 + γ)`` with unit-RMS z, modelled as z ~ N(0, I), so cov = diag((1 + γ)²).
    The input of ``down_proj`` is ``silu(W_gate x̃) ⊙ (W_up x̃)``; its covariance is estimated by pushing
    synthetic Gaussian x̃ (same model) through the layer's own gate/up weights. No text is used.
    """
    norm = C.norm_before(name, names)
    scale = 1.0 + _tensor(art, norm) if norm else None
    if ".mlp.down_proj." not in name:
        n_in = _tensor(art, name).shape[1]
        s = scale if scale is not None and len(scale) == n_in else np.ones(n_in)
        return np.diag(s.astype(np.float64) ** 2), "diag_norm_gain" if norm else "identity"
    gate = _tensor(art, name.replace("down_proj", "gate_proj"))
    up = _tensor(art, name.replace("down_proj", "up_proj"))
    x = rng.standard_normal((C.SYNTH_SAMPLES, gate.shape[1])) * (scale if scale is not None else 1.0)
    g, u = x @ gate.T, x @ up.T
    h = g / (1.0 + np.exp(-g)) * u
    return h.T @ h / len(h), "synthetic_gate_up"


# ------------------------------------------------------------------- weight-derived residual geometry
def _embedding_gram(art: Path, names: list[str], dim: int, chunk: int = 16384) -> np.ndarray:
    """``EᵀE / V`` of the token embedding (the residual stream's starting point, uniform over tokens);
    zeros if the checkpoint has no embedding of width ``dim``. Read in row chunks to bound memory."""
    from safetensors import safe_open

    acc = np.zeros((dim, dim))
    if C.EMBED not in names:
        return acc
    with safe_open(str(_weights_file(art)), "pt") as f:
        sl = f.get_slice(C.EMBED)
        n_rows, width = sl.get_shape()
        if width != dim:
            return acc
        for a in range(0, n_rows, chunk):
            e = sl[a:a + chunk].float().numpy().astype(np.float64)
            acc += e.T @ e
    return acc / n_rows


def _unit_mean_variance(acc: np.ndarray) -> np.ndarray:
    """Normalise to mean diagonal 1 (an RMSNorm output has unit RMS); identity if nothing was accumulated."""
    mean_var = float(np.mean(np.diag(acc)))
    if mean_var <= 0:
        return np.eye(len(acc))
    return acc / mean_var


def residual_second_moments(art: Path, names: list[str], sites: set[tuple[int, int]],
                            dim: int) -> dict[tuple[int, int], np.ndarray]:
    """Data-free second moment of the residual stream at each reader site (``config.residual_site``).

    Model: every matrix that writes into the residual stream (attention ``o_proj`` / linear-attention
    ``out_proj`` and MLP ``down_proj`` of earlier blocks) is driven by isotropic unit-variance inputs, so
    it contributes ``W Wᵀ``; the token embedding contributes ``EᵀE / V``. The sum is the covariance of
    the stream a reader sees, up to scale, and is normalised to unit mean variance (RMSNorm). It captures
    the anisotropy (dominant directions, outlier channels) that the isotropic Gaussian model ignores.
    One ordered pass over the writers serves every site.
    """
    acc = _embedding_gram(art, names, dim)
    writers = sorted((key, n) for n in names if (key := C.writer_key(n)) is not None)
    pending = sorted(sites)
    out: dict[tuple[int, int], np.ndarray] = {}
    for key, name in writers:
        while pending and pending[0] <= key:
            out[pending.pop(0)] = _unit_mean_variance(acc)
        w = _tensor(art, name).astype(np.float64)
        if w.shape[0] == dim:
            acc += w @ w.T
    for site in pending:
        out[site] = _unit_mean_variance(acc)
    return out


def _norm_scale(art: Path, name: str, names: list[str], dim: int) -> np.ndarray:
    """``1 + γ`` of the RMSNorm feeding ``name`` (ones if there is none of width ``dim``)."""
    norm = C.norm_before(name, names)
    if norm is None:
        return np.ones(dim)
    scale = 1.0 + _tensor(art, norm).astype(np.float64)
    return scale if len(scale) == dim else np.ones(dim)


def _swiglu_second_moment(art: Path, name: str, cov_x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Second moment of ``silu(W_gate x̃) ⊙ (W_up x̃)`` for synthetic ``x̃ ~ N(0, cov_x)`` (no text)."""
    gate = _tensor(art, name.replace("down_proj", "gate_proj"))
    up = _tensor(art, name.replace("down_proj", "up_proj"))
    jitter = 1e-6 * float(np.mean(np.diag(cov_x))) + 1e-12
    chol = np.linalg.cholesky(cov_x + jitter * np.eye(len(cov_x)))
    x = rng.standard_normal((C.SYNTH_SAMPLES, len(cov_x))) @ chol.T
    g, u = x @ gate.T, x @ up.T
    h = g / (1.0 + np.exp(-g)) * u
    return h.T @ h / len(h)


def residual_input_covariance(art: Path, name: str, names: list[str], resid: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    """Input covariance of target ``name`` under the residual-geometry model: the RMSNorm output
    ``x̃ = z ⊙ (1 + γ)`` with ``cov(z) = resid`` gives ``D resid D``; for ``down_proj`` that x̃ is pushed
    through the layer's own gate/up weights (synthetic samples, deterministic ``rng``)."""
    scale = _norm_scale(art, name, names, len(resid))
    cov_x = scale[:, None] * resid * scale[None, :]
    if ".mlp.down_proj." not in name:
        return cov_x
    return _swiglu_second_moment(art, name, cov_x, rng)


# ------------------------------------------------------------------- the reference's candidate set
RESID_SHRINK = (0.3, 0.6, 0.9)
SALIENCY_POWERS = (0.5, 1.0)

VARIANT_DOCS: dict[str, str] = {
    "diag_mask": "mask by w²·E[x_j²] under the isotropic-Gaussian covariance (Wanda-style), no compensation",
    "obs": "SparseGPT-style OBS under the isotropic-Gaussian covariance, damping 0.01",
    "obs_damped": "SparseGPT-style OBS under the isotropic-Gaussian covariance, damping 1.0",
    # (a) Magnitude 2:4 removes the smallest entries, so every row's output shrinks by a predictable
    # factor (~10-20% of the energy for Gaussian rows). Downstream RMSNorms and the residual stream see a
    # systematically weaker MLP write. Restoring each row's L2 norm is the data-free bias/variance
    # correction of DFQ (Nagel et al. 2019) applied to pruning: same mask as the floor, no covariance needed.
    "mag_rowscale": "magnitude mask, then rescale each row's survivors to the row's dense L2 norm (DFQ-style)",
    # (b) Under the isotropic-Gaussian covariance full OBS does *worse* than magnitude: compensation trusts a
    # wrong covariance. Here the covariance comes from the residual-stream geometry implied by the weights
    # that write into it (Σ_writers W Wᵀ + EᵀE / V, then the norm gain; for down_proj pushed through
    # SwiGLU(gate, up)), and is shrunk toward the identity so that compensation is only as strong as the
    # estimate deserves. ρ → 1 recovers magnitude pruning exactly, so the family interpolates from the floor.
    **{f"obs_resid_r{rho}": f"OBS (damping 0.01) under the residual-geometry covariance shrunk toward I, ρ = {rho}"
       for rho in RESID_SHRINK},
    # (c) Mask only, no weight updates (no risk of a wrong compensation): keep the 2 largest
    # |w_ij| · σ_j^p per group, σ_j² the residual-geometry input variance. Per-row factors (e.g. the norm of
    # the downstream column) cannot change a within-row mask, so only the input side enters. p = 1 is
    # Wanda-style, p = 0.5 is halfway to magnitude (the plain-gain Wanda mask does worse than magnitude).
    **{f"sal_resid_p{p}": f"mask by |w|·σ_j^{p} with σ_j² the residual-geometry input variance, no compensation"
       for p in SALIENCY_POWERS},
}
VARIANTS = tuple(VARIANT_DOCS)


def prune_variant(w: np.ndarray, cov: np.ndarray, variant: str, resid_cov: np.ndarray | None = None) -> np.ndarray:
    """2:4 prune ``w`` with ``variant`` (see ``VARIANT_DOCS``). ``cov`` is the isotropic-Gaussian input
    covariance (``input_covariance``), ``resid_cov`` the residual-geometry one (defaults to ``cov``).
    Every variant zeroes exactly the masked entries, so the result is exactly 2:4."""
    resid = cov if resid_cov is None else resid_cov
    if variant == "diag_mask":
        return np.where(SP.mask_from_scores(w ** 2 * np.diag(cov)[None, :]), w, 0.0)
    if variant == "obs":
        return SP.obs_24(w, cov, damp=0.01)
    if variant == "obs_damped":
        return SP.obs_24(w, cov, damp=1.0)
    if variant == "mag_rowscale":
        return SP.rowscale_24(w, SP.mask_from_scores(np.abs(w)))
    if variant.startswith("obs_resid_r"):
        rho = float(variant.removeprefix("obs_resid_r"))
        return SP.obs_24(w, SP.shrink_to_identity(resid, rho), damp=0.01)
    if variant.startswith("sal_resid_p"):
        power = float(variant.removeprefix("sal_resid_p"))
        return SP.saliency_24(w, np.sqrt(np.clip(np.diag(resid), 0.0, None)), power)
    raise ValueError(f"unknown 2:4 variant {variant!r}")


def _residual_dim(art: Path, name: str, names: list[str]) -> int:
    """Width of the residual stream a target reads (its in-dim, or gate_proj's in-dim for down_proj)."""
    if ".mlp.down_proj." in name:
        return _tensor(art, name.replace("down_proj", "gate_proj")).shape[1]
    return _tensor(art, name).shape[1]


def target_covariances(art: Path, targets: list[str], names: list[str], seed: int) -> dict[str, dict]:
    """Per target: the isotropic-Gaussian covariance (rng seed+1), its kind,
    and the residual-geometry covariance (rng seed+2). Deterministic given the seed and the weights."""
    rng_iso, rng_resid = np.random.default_rng(seed + 1), np.random.default_rng(seed + 2)
    iso = {name: input_covariance(art, name, names, rng_iso) for name in targets}
    dim = _residual_dim(art, targets[0], names)
    resid = residual_second_moments(art, names, {C.residual_site(t) for t in targets}, dim)
    return {name: {"cov": iso[name][0], "kind": iso[name][1],
                   "resid": residual_input_covariance(art, name, names, resid[C.residual_site(name)], rng_resid)}
            for name in targets}


def variant_delta(art: Path, targets: list[str], covs: dict[str, dict], variant: str) -> dict[str, np.ndarray]:
    """Float32 ΔW of ``variant`` for every target; ``W + ΔW`` is exactly 2:4 in float32 (pruned entries
    get ΔW = −W exactly, since W is the float32 image of the bf16 checkpoint)."""
    delta = {}
    for name in targets:
        w = _tensor(art, name)
        delta[name] = (prune_variant(w, covs[name]["cov"], variant, covs[name]["resid"]) - w).astype(np.float32)
    return delta


CALIBRATED = "sparsegpt_calib"
CALIBRATED_DOC = ("SparseGPT (OBS, damping 0.01) with input second moments measured on private calibration text, "
                  "pruned stage by stage on the already-pruned network; privileged existence proof, not data-free")


def sparsegpt_calib_delta(art: Path, targets: list[str], seed: int) -> dict[str, np.ndarray]:
    """``CALIBRATED_DOC``. Each stage (``calib.stages``) is measured with every earlier stage already pruned,
    as in SparseGPT's sequential layer-wise reconstruction (Frantar & Alistarh, 2023)."""
    import torch

    from weightbench.tracks.tr02_loops_lib import model as M

    from . import calib as CB

    rt = M.load(art)
    ids = torch.from_numpy(CB.calib_ids(art, seed)).to(rt.dev)
    modules = dict(rt.model.named_modules())
    delta: dict[str, np.ndarray] = {}
    for group in CB.stages(targets):
        stats = CB.input_stats(rt.model, {n: modules[CB.module_name(n)] for n in group}, ids, gram=True)
        for name in group:
            w = _tensor(art, name)
            delta[name] = (SP.obs_24(w, stats[name]["gram"], damp=0.01) - w).astype(np.float32)
        rt.apply_delta(delta)
    rt.restore()
    return delta


def reference(art: Path) -> dict:
    """Build every data-free variant in ``VARIANTS`` and the calibrated ``CALIBRATED`` one for all targets and
    keep the one with the lowest NLL on the private dev chunks (disjoint from the eval chunks; selection never
    sees eval data). Magnitude is reported for comparison; so is every variant's dev NLL."""
    sp = _spec(art)
    names = _names(art)
    covs = target_covariances(art, sp["targets"], names, sp["seed"])
    builders = {v: (lambda v=v: variant_delta(art, sp["targets"], covs, v)) for v in VARIANTS}
    builders[CALIBRATED] = lambda: sparsegpt_calib_delta(art, sp["targets"], sp["seed"])
    docs = {**VARIANT_DOCS, CALIBRATED: CALIBRATED_DOC}
    dev = {"magnitude": mean_nll(art, str(art / "private" / "baseline_delta.npz"), "dev")["nll"]}
    print(f"[tr05-sparse] dev NLL magnitude = {dev['magnitude']:.5f}", flush=True)
    best, best_nll = None, float("inf")
    for variant, build in builders.items():
        path = art / "private" / f"reference_{variant}.npz"
        np.savez(path, **build())
        dev[variant] = mean_nll(art, str(path), "dev")["nll"]
        print(f"[tr05-sparse] dev NLL {variant} = {dev[variant]:.5f}  ({docs[variant]})", flush=True)
        if dev[variant] < best_nll:
            best, best_nll = variant, dev[variant]
    (art / "private" / f"reference_{best}.npz").replace(art / "private" / "reference_delta.npz")
    for variant in builders:
        (art / "private" / f"reference_{variant}.npz").unlink(missing_ok=True)
    data_free = min(VARIANTS, key=dev.__getitem__)
    print(f"[tr05-sparse] selected {best} (dev NLL {best_nll:.5f}); best data-free {data_free}", flush=True)
    return {"selected": best, "selected_doc": docs[best], "privileged": best == CALIBRATED,
            "best_data_free": data_free, "variants": docs, "dev_nll": dev,
            "beats_magnitude_on_dev": best_nll < dev["magnitude"],
            "covariances": {n: c["kind"] for n, c in covs.items()}}


# ------------------------------------------------------------------- named reference variants
# ``runner run --agent reference --variant V``. "dev_selected" is the build's reference_delta.npz (the default:
# lowest dev NLL over VARIANTS and CALIBRATED, so it may be the privileged one; reference_info.json says which).
DEV_SELECTED = "dev_selected"
DATAFREE_DEVSEL = "datafree_devsel"            # best of VARIANTS on dev NLL (weights-only edit; *selection* saw dev text)
PROPAGATED = "propagated_obs"                  # fixed in advance, no selection at all
CALIBRATED_PRIVILEGED = "sparsegpt_calib_privileged"
REFERENCE_VARIANTS = (DEV_SELECTED, DATAFREE_DEVSEL, PROPAGATED, CALIBRATED_PRIVILEGED)
PRIVILEGED_VARIANTS = frozenset({CALIBRATED_PRIVILEGED})
PROP_DAMP = 0.01                               # fixed (SparseGPT's default relative damping)


def embedding_moments(art: Path, names: list[str], dim: int, chunk: int = 16384):
    """Uniform-prior moments of the token embedding (mean row and ``EᵀE / V``), read in row chunks."""
    from safetensors import safe_open

    from . import moments as MO

    with safe_open(str(_weights_file(art)), "pt") as f:
        sl = f.get_slice(C.EMBED)
        n_rows, width = sl.get_shape()
        if width != dim:
            raise ValueError(f"embedding width {width} != hidden {dim}")
        s, acc = np.zeros(dim), np.zeros((dim, dim))
        for a in range(0, n_rows, chunk):
            e = sl[a:a + chunk].float().numpy().astype(np.float64)
            s += e.sum(0)
            acc += e.T @ e
    return MO.Moments(s / n_rows, acc / n_rows)


def layer_names(names: list[str]) -> list:
    """``moments.LayerNames`` for every decoder layer of the checkpoint (missing norms / mixers become None)."""
    from . import moments as MO

    have = set(names)
    out = []
    for l in range(C.N_LAYERS):
        base = f"{C.PREFIX}layers.{l}."

        def opt(suffix: str) -> str | None:
            return base + suffix if base + suffix in have else None

        mixer = opt("self_attn.o_proj.weight") or opt("linear_attn.out_proj.weight")
        out.append(MO.LayerNames(opt("input_layernorm.weight"), mixer, opt("post_attention_layernorm.weight"),
                                 C.mlp(l, "gate_proj"), C.mlp(l, "up_proj"), C.mlp(l, "down_proj")))
    return out


def propagated_delta(art: Path, targets: list[str], names: list[str] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """``PROPAGATED``: OBS 2:4 (damping ``PROP_DAMP``) of every target with H = E[x xᵀ] from the analytic moment
    propagation of ``moments.py`` (uniform token prior, sequential over the already-pruned upstream MLPs).
    Deterministic: no randomness, no seed, no text, no forward pass. Also returns, per target, the proxy output
    error of this edit and of magnitude 2:4 (a data-free diagnostic only; it selects nothing)."""
    from . import moments as MO

    names = names if names is not None else _names(art)
    dim = _tensor(art, C.mlp(0, "gate_proj")).shape[1]
    diag: dict[str, dict] = {}

    def prune(name: str, w: np.ndarray, x) -> np.ndarray:
        new = SP.obs_24(w, x.S, damp=PROP_DAMP)
        diag[name] = {"proxy_err_obs": SP.output_error(w, new, x.S),
                      "proxy_err_magnitude": SP.output_error(w, SP.magnitude_24(w), x.S)}
        return new

    pruned: dict[str, np.ndarray] = {}

    def attn_site(layer: int, x) -> None:
        for name in targets:
            if C.layer_of(name) == layer and ".mlp." not in name:
                pruned[name] = prune(name, _tensor(art, name).astype(np.float64), x)

    pruned.update(MO.walk(lambda n: _tensor(art, n), layer_names(names), embedding_moments(art, names, dim),
                          on_attn_site=attn_site, prune=prune, prunable=set(targets)))
    return {n: (pruned[n] - _tensor(art, n)).astype(np.float32) for n in targets}, diag


def reference_variant(art: Path, variant: str) -> dict:
    """Write ``private/reference_<variant>.npz`` for a named variant (``REFERENCE_VARIANTS`` except ``dev_selected``).
    ``datafree_devsel`` rebuilds the build's best data-free variant (``reference_info.json``) deterministically;
    the privileged one rebuilds ``sparsegpt_calib`` (same calibration ids from the seed)."""
    sp = _spec(art)
    targets, names = sp["targets"], _names(art)
    info: dict = {"variant": variant, "privileged": variant in PRIVILEGED_VARIANTS}
    if variant == PROPAGATED:
        delta, diag = propagated_delta(art, targets, names)
        info.update(method=f"OBS 2:4, damping {PROP_DAMP}, H from analytic moment propagation (moments.py)",
                    proxy_errors=diag)
    elif variant == DATAFREE_DEVSEL:
        chosen = json.loads((art / "private" / "reference_info.json").read_text())["best_data_free"]
        delta = variant_delta(art, targets, target_covariances(art, targets, names, sp["seed"]), chosen)
        info.update(rebuilt=chosen, note="weights-only edit; chosen among VARIANTS by private dev NLL")
    elif variant == CALIBRATED_PRIVILEGED:
        delta = sparsegpt_calib_delta(art, targets, sp["seed"])
        info.update(doc=CALIBRATED_DOC)
    else:
        raise ValueError(f"unknown TR05-SPARSE reference variant {variant!r}; choose from {REFERENCE_VARIANTS[1:]}")
    np.savez(art / "private" / f"reference_{variant}.npz", **delta)
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "eval", "baseline", "reference", "reference_variant"])
    ap.add_argument("--variant")
    ap.add_argument("--art", required=True)
    ap.add_argument("--delta")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    art = Path(a.art)
    t0 = time.time()
    if a.step == "prepare":
        out = prepare(art)
    elif a.step == "eval":
        out = mean_nll(art, a.delta, a.split)
    elif a.step == "baseline":
        out = baseline(art)
    elif a.step == "reference_variant":
        out = reference_variant(art, a.variant)
    else:
        out = reference(art)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
