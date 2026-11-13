"""Data-free propagated-moment references: model-free toys only (no checkpoint, no model load)."""
import json

import numpy as np
import pytest

from weightbench.tracks import tr05_outlier as TO
from weightbench.tracks import tr05_sparsity as TS
from weightbench.tracks.tr05_outlier_lib import jobs as OJ
from weightbench.tracks.tr05_sparsity_lib import jobs as SJ
from weightbench.tracks.tr05_sparsity_lib import moments as MO
from weightbench.tracks.tr05_sparsity_lib import sparsity as SP


def test_linear_propagation_is_exact_on_a_hand_computed_3x4_example():
    # x in R^4 with mu = (1, 0, -1, 2) and S = E[x xᵀ] = diag(2, 1, 3, 5) + mu muᵀ  (so Cov = diag(2, 1, 3, 5)).
    # W (3x4) = [[1, 0, 0, 1], [0, 2, 0, 0], [1, -1, 1, 0]].
    # mu_y = W mu = (1 + 2, 0, 1 - 0 - 1) = (3, 0, 0).
    # Cov_y = W Cov Wᵀ:   row0·row0 = 2 + 5 = 7;  row0·row1 = 0;  row0·row2 = 2;
    #                    row1·row1 = 4·1 = 4;  row1·row2 = 2·(-1)·1 = -2;  row2·row2 = 2 + 1 + 3 = 6.
    # S_y = Cov_y + mu_y mu_yᵀ adds 9 at (0, 0): S_y = [[16, 0, 2], [0, 4, -2], [2, -2, 6]].
    mu = np.array([1.0, 0.0, -1.0, 2.0])
    m = MO.Moments(mu, np.diag([2.0, 1.0, 3.0, 5.0]) + np.outer(mu, mu))
    w = np.array([[1.0, 0, 0, 1], [0, 2, 0, 0], [1, -1, 1, 0]])
    y = MO.linear(m, w)
    np.testing.assert_allclose(y.mu, [3.0, 0.0, 0.0])
    np.testing.assert_allclose(y.S, [[16, 0, 2], [0, 4, -2], [2, -2, 6]], atol=1e-12)


def _toy_gaussian(d, rng):
    a = rng.standard_normal((d, d)) / np.sqrt(d)
    cov = a @ a.T + 0.3 * np.eye(d)
    mu = 0.4 * rng.standard_normal(d)
    return MO.Moments(mu, cov + np.outer(mu, mu))


def _mc_swiglu(x, wg, wu, n, rng):
    z = rng.multivariate_normal(x.mu, x.cov, size=n)
    g, u = z @ wg.T, z @ wu.T
    return z, g / (1 + np.exp(-g)) * u


def test_swiglu_moments_match_monte_carlo_on_a_toy():
    """Exact parts (mean, per-channel variance, Cov(s, x) via Stein) within MC error; the full second moment
    (degree-2 chaos expansion, degree >= 3 kept diagonal) within 6% relative Frobenius error (typically 1.7-3.1% on
    this toy). MC is in the test only."""
    rng = np.random.default_rng(0)
    d, n_h, n = 6, 5, 400_000
    x = _toy_gaussian(d, rng)
    wg, wu = rng.standard_normal((n_h, d)) / np.sqrt(d), rng.standard_normal((n_h, d)) / np.sqrt(d)
    sw = MO.swiglu_moments(x, wg, wu)
    z, s = _mc_swiglu(x, wg, wu, n, rng)
    np.testing.assert_allclose(sw.mu, s.mean(0), atol=0.01)
    np.testing.assert_allclose(sw.var, s.var(0), rtol=0.03, atol=0.005)
    cross_mc = (s - s.mean(0)).T @ (z - z.mean(0)) / n
    np.testing.assert_allclose(sw.A @ x.cov, cross_mc, atol=0.01)
    s_mc = s.T @ s / n
    assert np.linalg.norm(sw.S - s_mc) / np.linalg.norm(s_mc) < 0.06
    assert np.linalg.eigvalsh(sw.S).min() > -1e-10


def test_mlp_residual_update_matches_monte_carlo_on_a_toy():
    rng = np.random.default_rng(1)
    d, n_h, n = 6, 8, 300_000
    h = _toy_gaussian(d, rng)
    gain = 0.3 * rng.standard_normal(d)
    wg, wu = rng.standard_normal((n_h, d)), rng.standard_normal((n_h, d))
    wd = rng.standard_normal((d, n_h)) / n_h
    out = MO.mlp_residual_update(h, gain, wg, wu, wd)
    z = rng.multivariate_normal(h.mu, h.cov, size=n)
    r = np.sqrt(np.trace(h.S) / d)                          # the documented mean-field rms
    x = z / r * (1 + gain)
    g, u = x @ wg.T, x @ wu.T
    y = z + (g / (1 + np.exp(-g)) * u) @ wd.T
    np.testing.assert_allclose(out.mu, y.mean(0), atol=0.02)
    s_mc = y.T @ y / n
    assert np.linalg.norm(out.S - s_mc) / np.linalg.norm(s_mc) < 0.05


def test_obs_with_the_true_covariance_beats_magnitude_on_a_toy():
    """Strongly correlated inputs (rank-3 factor plus small noise), as in a residual stream with dominant directions."""
    rng = np.random.default_rng(2)
    f = rng.standard_normal((16, 3))
    cov = f @ f.T + 0.05 * np.eye(16)
    w = rng.standard_normal((8, 16))
    obs = SP.obs_24(w, cov, damp=0.01)
    assert SP.is_24_sparse(obs)
    assert SP.output_error(w, obs, cov) < 0.5 * SP.output_error(w, SP.magnitude_24(w), cov)


def _toy_decoder(rng, d=8, n_h=12, n_layers=3):
    p = "model.language_model."
    t = {}
    layers = []
    for l in range(n_layers):
        b = f"{p}layers.{l}."
        t[b + "input_layernorm.weight"] = 0.2 * rng.standard_normal(d)
        t[b + "self_attn.o_proj.weight"] = rng.standard_normal((d, d)) / np.sqrt(d)
        t[b + "post_attention_layernorm.weight"] = 0.2 * rng.standard_normal(d)
        t[b + "mlp.gate_proj.weight"] = rng.standard_normal((n_h, d)) / np.sqrt(d)
        t[b + "mlp.up_proj.weight"] = rng.standard_normal((n_h, d)) / np.sqrt(d)
        t[b + "mlp.down_proj.weight"] = rng.standard_normal((d, n_h)) / np.sqrt(n_h)
        layers.append(MO.LayerNames(b + "input_layernorm.weight", b + "self_attn.o_proj.weight",
                                    b + "post_attention_layernorm.weight", b + "mlp.gate_proj.weight",
                                    b + "mlp.up_proj.weight", b + "mlp.down_proj.weight"))
    e = rng.standard_normal((50, d))
    return t, layers, MO.embedding_moments(e)


def test_walk_prunes_sequentially_and_is_deterministic():
    t, layers, h0 = _toy_decoder(np.random.default_rng(3))
    targets = {ln.down for ln in layers} | {layers[1].gate}
    seen = []

    def prune(name, w, x):
        seen.append(name)
        return SP.obs_24(w, x.S, damp=0.01)

    a = MO.walk(t.__getitem__, layers, h0, prune=prune, prunable=targets)
    b = MO.walk(t.__getitem__, layers, h0, prune=lambda n, w, x: SP.obs_24(w, x.S, damp=0.01), prunable=targets)
    assert set(a) == targets and seen.index(layers[1].gate) < seen.index(layers[1].down)
    assert all(SP.is_24_sparse(a[k]) and np.array_equal(a[k], b[k]) for k in targets)


def test_outlier_preregistered_setting_is_identical_across_seeds():
    """One pre-registered setting; the propagated scales take no seed and do not vary."""
    t, layers, h0 = _toy_decoder(np.random.default_rng(4))
    for kind in ("norm_to_gate_up", "up_to_down"):
        settings, scales = [], []
        for seed in (0, 1, 2):
            np.random.seed(seed)                       # global RNG state must not matter either
            settings.append(OJ.preregistered(kind))
            scales.append(OJ.propagated_scales(OJ.propagated_ranges(t.__getitem__, layers, h0, kind)))
        assert settings[0] == settings[1] == settings[2]
        assert settings[0]["clip_quantile"] == OJ.PROP_CLIP_Q and settings[0]["k"] == OJ.PROP_K
        for layer in scales[0]:
            assert np.array_equal(scales[0][layer], scales[1][layer])
            assert np.array_equal(scales[0][layer], scales[2][layer])
            assert np.all(scales[0][layer] >= 1.0)


def test_variant_lists_agree_and_privileged_is_labelled():
    assert TS.REFERENCE_VARIANTS == SJ.REFERENCE_VARIANTS and TO.REFERENCE_VARIANTS == OJ.REFERENCE_VARIANTS
    for task in (TS.SparseOneLayer(), TO.OutlierNormToGateUp()):
        assert task.default_reference_variant() == "dev_selected"
    assert all("privileged" in v for v in SJ.PRIVILEGED_VARIANTS | OJ.PRIVILEGED_VARIANTS)


def test_unknown_variant_is_rejected(tmp_path):
    from weightbench import runner

    for task in (TS.SparseAllMLPs(), TO.OutlierUpToDown()):
        assert runner.resolve_variant(task, "reference", "propagated_obs" if task.id.startswith("TR05-S")
                                      else "propagated_clip")
        with pytest.raises(ValueError):
            runner.resolve_variant(task, "reference", "nonsense")
        with pytest.raises(ValueError):
            task.reference(tmp_path, "nonsense")


# ------------------------------------------------------------------- end to end on a tiny safetensors checkpoint
@pytest.fixture
def tiny_art(tmp_path, monkeypatch):
    """Two-layer toy checkpoint with the Qwen3.5 tensor names (one full-attention, one Gated-DeltaNet layer)."""
    import torch
    from safetensors.torch import save_file

    from weightbench.tracks.tr05_sparsity_lib import config as C

    monkeypatch.setattr(C, "N_LAYERS", 2)
    p, d, n_h = C.PREFIX, 8, 16
    shapes = {f"{p}embed_tokens.weight": (40, d)}
    for l, mixer in ((0, "self_attn.o_proj"), (1, "linear_attn.out_proj")):
        b = f"{p}layers.{l}."
        shapes.update({b + "input_layernorm.weight": (d,), b + "post_attention_layernorm.weight": (d,),
                       b + f"{mixer}.weight": (d, d), b + "mlp.gate_proj.weight": (n_h, d),
                       b + "mlp.up_proj.weight": (n_h, d), b + "mlp.down_proj.weight": (d, n_h)})
    shapes[f"{p}layers.0.self_attn.q_proj.weight"] = (12, d)
    g = torch.Generator().manual_seed(0)
    tensors = {k: (0.3 * torch.randn(*s, generator=g)).to(torch.bfloat16) for k, s in shapes.items()}
    art = tmp_path / "art"
    (art / "public" / "model").mkdir(parents=True)
    (art / "private").mkdir()
    save_file(tensors, str(art / "public" / "model" / C.WEIGHTS_FILE))
    targets = [f"{p}layers.0.self_attn.q_proj.weight"] + [C.mlp(l, q) for l in (0, 1) for q in C.MLP_PROJS]
    (art / "public" / "targets.json").write_text(json.dumps({t: list(shapes[t]) for t in targets}))
    (art / "private" / "spec.json").write_text(json.dumps({"seed": 0, "targets": targets, "kind": "norm_to_gate_up"}))
    return art, targets


def test_propagated_sparse_variant_end_to_end_on_a_tiny_checkpoint(tiny_art):
    art, targets = tiny_art
    info = SJ.reference_variant(art, SJ.PROPAGATED)
    assert not info["privileged"] and set(info["proxy_errors"]) == set(targets)
    for err in info["proxy_errors"].values():          # OBS under its own proxy is no worse than magnitude
        assert err["proxy_err_obs"] <= err["proxy_err_magnitude"] * (1 + 1e-6)
    task = TS.SparseOneLayer()
    sub = task.reference(art, SJ.PROPAGATED)            # cached file, no heavy step
    assert task._validation_error(art, sub.delta) is None
    first = dict(sub.delta)
    SJ.reference_variant(art, SJ.PROPAGATED)
    again = task.reference(art, SJ.PROPAGATED).delta
    assert all(np.array_equal(first[k], again[k]) for k in targets)


def test_propagated_outlier_variant_end_to_end_on_a_tiny_checkpoint(tiny_art):
    art, _ = tiny_art
    for kind in ("norm_to_gate_up", "up_to_down"):
        spec = json.loads((art / "private" / "spec.json").read_text())
        (art / "private" / "spec.json").write_text(json.dumps({**spec, "kind": kind}))
        info = OJ.reference_variant(art, OJ.PROPAGATED)
        assert info["preregistered"] == OJ.preregistered(kind) and not info["privileged"]
        with np.load(art / "private" / f"reference_{OJ.PROPAGATED}.npz") as z:
            assert z.files and all(np.all(np.isfinite(z[k])) for k in z.files)
