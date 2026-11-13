"""TR05-OUTLIER-001/002: function-preserving migration, quantization error, reference job, scoring (toy weights)."""
import json
from pathlib import Path

import numpy as np
import pytest

from weightbench.task import Submission
from weightbench.tracks import tr05_outlier as T
from weightbench.tracks.tr05_outlier_lib import config as OC
from weightbench.tracks.tr05_outlier_lib import jobs as J
from weightbench.tracks.tr05_outlier_lib import quant as Q
from weightbench.tracks.tr05_sparsity_lib import config as SC


def rmsnorm(x, w):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True)) * (1.0 + w)


def silu(z):
    return z / (1.0 + np.exp(-z))


def test_norm_migration_preserves_function():
    rng = np.random.default_rng(0)
    x, gamma = rng.normal(size=(5, 8)), rng.normal(size=8)
    gate, up = rng.normal(size=(12, 8)), rng.normal(size=(12, 8))
    s = rng.uniform(0.2, 5.0, 8)
    new_gamma, (g2, u2) = Q.migrate_norm(gamma, [gate, up], s)
    for w_old, w_new in ((gate, g2), (up, u2)):
        np.testing.assert_allclose(rmsnorm(x, gamma) @ w_old.T, rmsnorm(x, new_gamma) @ w_new.T, rtol=1e-9)


def test_up_down_migration_preserves_function():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 8))
    gate, up, down = rng.normal(size=(12, 8)), rng.normal(size=(12, 8)), rng.normal(size=(8, 12))
    s = rng.uniform(0.2, 5.0, 12)
    up2, down2 = Q.migrate_up_down(up, down, s)
    before = (silu(x @ gate.T) * (x @ up.T)) @ down.T
    after = (silu(x @ gate.T) * (x @ up2.T)) @ down2.T
    np.testing.assert_allclose(before, after, rtol=1e-9)


def test_smoothing_reduces_w8a8_error_with_outliers():
    rng = np.random.default_rng(2)
    act_scale = np.ones(64)
    act_scale[:3] = 60.0                                            # three outlier channels
    x = rng.normal(size=(256, 64)) * act_scale
    w = rng.normal(size=(32, 64))
    exact = x @ w.T

    def w8a8_error(xx, ww):
        return float(np.abs(Q.fake_quant_per_tensor(xx) @ Q.fake_quant_per_channel(ww).T - exact).mean())

    s = Q.smooth_scales(act_scale, w)
    assert w8a8_error(x / s, w * s) < 0.5 * w8a8_error(x, w)


def test_fake_quant_hand_computed():
    assert np.allclose(Q.fake_quant_per_tensor(np.array([127.0, 0.4, -63.6])), [127.0, 0.0, -64.0])
    w = np.array([[1.0, 0.004], [-2.0, 1.0]])
    # row 0: scale 1/127, 0.004·127 = 0.508 → 1 → 1/127;  row 1: scale 2/127, 1/(2/127) = 63.5 → 64 → 128/127
    assert np.allclose(Q.fake_quant_per_channel(w), [[1.0, 1 / 127], [-2.0, 128 / 127]])


def test_clip_scales_hand_computed():
    # quantile 0.5 of [1, 2, 3, 4, 40] is 3 → s = max(1, a / 3): only the channels above 3 shrink
    assert np.allclose(Q.clip_scales(np.array([1.0, 2.0, 3.0, 4.0, 40.0]), 0.5), [1, 1, 1, 4 / 3, 40 / 3])
    assert np.allclose(Q.clip_scales(np.ones(5), 0.99), 1.0)          # no outliers → identity migration


# ------------------------------------------------------------------------------------- toy artifacts
P = SC.PREFIX


@pytest.fixture
def toy_art(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    monkeypatch.setattr(SC, "N_LAYERS", 1)
    monkeypatch.setattr(SC, "SYNTH_SAMPLES", 256)
    art = tmp_path / "art"
    (art / "public" / "model").mkdir(parents=True)
    (art / "private").mkdir()
    g = torch.Generator().manual_seed(0)
    tensors = {SC.mlp(0, "gate_proj"): (16, 8), SC.mlp(0, "up_proj"): (16, 8), SC.mlp(0, "down_proj"): (8, 16),
               f"{P}layers.0.post_attention_layernorm.weight": (8,)}
    save_file({k: (torch.randn(*s, generator=g) * 0.5).to(torch.bfloat16) for k, s in tensors.items()},
              str(art / "public" / "model" / SC.WEIGHTS_FILE))
    (art / "private" / "calibration.json").write_text(json.dumps({"nll_fp": 2.0, "nll_quant": 2.5}))
    return art


GATE_UP = {SC.mlp(0, "gate_proj"), SC.mlp(0, "up_proj"), f"{P}layers.0.post_attention_layernorm.weight"}
UP_DOWN = {SC.mlp(0, "up_proj"), SC.mlp(0, "down_proj")}


@pytest.mark.parametrize("kind, names, spec_extra, winner", [
    ("norm_to_gate_up", GATE_UP, {}, "alpha_0.5"),                                         # legacy spec (no key)
    ("norm_to_gate_up", GATE_UP, {"deployment": OC.OUTLIER_001_DEPLOYMENT}, "resid_alpha_0.75"),
    ("norm_to_gate_up", GATE_UP, {"deployment": OC.OUTLIER_001_DEPLOYMENT}, "resid_clip_0.99"),
    ("up_to_down", UP_DOWN, {}, "alpha_0.5"),
    ("up_to_down", UP_DOWN, {"deployment": OC.OUTLIER_002_DEPLOYMENT}, "alpha_0.5"),
    ("norm_to_gate_up", GATE_UP, {"deployment": OC.OUTLIER_001_DEPLOYMENT}, "calib_clip_0.99"),
    ("up_to_down", UP_DOWN, {"deployment": OC.OUTLIER_002_DEPLOYMENT}, "calib_alpha_0.5"),
])
def test_reference_job_is_function_preserving(toy_art, kind, names, spec_extra, winner, monkeypatch):
    (toy_art / "private" / "spec.json").write_text(json.dumps({"seed": 0, "kind": kind, **spec_extra}))
    labels = [c[0] for c in J.candidates(kind, OC.from_spec(spec_extra))]
    dev = {None: 3.0, **{f"reference_{label}": 2.95 + 0.001 * i for i, label in enumerate(labels)}}
    dev[f"reference_{winner}"] = 2.9
    width = 8 if kind == "norm_to_gate_up" else 16
    monkeypatch.setattr(J, "calibrated_absmax",                         # privileged stats, faked: no subject model
                        lambda art, k, seed: {0: np.abs(np.random.default_rng(1).normal(size=width)) + 0.1})
    monkeypatch.setattr(J, "mean_nll_quantized",
                        lambda art, path, fp, batch=8, split="eval": {"nll": dev[Path(path).stem if path else None]})
    info = J.reference(toy_art)
    assert info["selected"] == winner and info["beats_no_migration_on_dev"] and info["candidates"] == labels
    assert info["selected_alpha"] == float(winner.rsplit("_", 1)[1])
    assert info["deployment"] == spec_extra.get("deployment", OC.LEGACY_DEPLOYMENT)
    assert info["privileged"] == winner.startswith("calib") and not info["best_data_free"].startswith("calib")
    with np.load(toy_art / "private" / "reference_delta.npz") as z:
        delta = {k: z[k] for k in z.files}
    assert set(delta) == names
    w = {n: J.SJ._tensor(toy_art, n) for n in [SC.mlp(0, p) for p in SC.MLP_PROJS]
         + [f"{P}layers.0.post_attention_layernorm.weight"]}
    edited = {n: v + delta.get(n, 0.0) for n, v in w.items()}
    x = np.random.default_rng(0).normal(size=(4, 8))

    def mlp(p):
        h = rmsnorm(x, p[f"{P}layers.0.post_attention_layernorm.weight"])
        return (silu(h @ p[SC.mlp(0, "gate_proj")].T) * (h @ p[SC.mlp(0, "up_proj")].T)) @ p[SC.mlp(0, "down_proj")].T

    np.testing.assert_allclose(mlp(w), mlp(edited), rtol=1e-3, atol=1e-4)


def test_evaluate_scoring_and_editable_set(toy_art, monkeypatch):
    task = T.OutlierUpToDown()
    shapes = {SC.mlp(0, "up_proj"): [16, 8], SC.mlp(0, "down_proj"): [8, 16]}
    (toy_art / "public" / "editable.json").write_text(json.dumps(shapes))
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: {"nll": 2.1, "ppl": 8.2})
    ev = task.evaluate(toy_art, Submission("delta_w", delta={SC.mlp(0, "down_proj"): np.zeros((8, 16), np.float32)}))
    assert ev.valid and ev.s_plus == pytest.approx(0.8)               # (2.5 - 2.1) / (2.5 - 2.0)
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: pytest.fail("invalid edits must not be evaluated"))
    bad = task.evaluate(toy_art, Submission("delta_w", delta={SC.mlp(0, "gate_proj"): np.zeros((16, 8))}))
    assert not bad.valid and "not editable" in bad.details["reason"]
    assert task.baseline(toy_art).delta[SC.mlp(0, "up_proj")].shape == (16, 8)


# ------------------------------------------------------------------------------------- deployments
def test_deployment_constants_and_build_option(monkeypatch):
    old, new = OC.get(OC.LEGACY_DEPLOYMENT), OC.get(OC.OUTLIER_001_DEPLOYMENT)
    assert (old.weight_qmax, old.act_qmax, old.label) == (127.0, 127.0, "W8A8")
    assert (new.weight_qmax, new.act_qmax, new.label) == (127.0, 31.0, "W8A6")
    assert OC.from_spec({}) == old                                     # legacy artifacts without the option
    monkeypatch.delenv(OC.OUTLIER_001_DEPLOYMENT_ENV, raising=False)
    assert T.OutlierNormToGateUp().build_deployment() == new
    assert T.OutlierUpToDown().build_deployment() == old              # OUTLIER-002 unchanged
    monkeypatch.setenv(OC.OUTLIER_001_DEPLOYMENT_ENV, OC.LEGACY_DEPLOYMENT)
    assert T.OutlierNormToGateUp().build_deployment() == old          # legacy 001 deployment stays reproducible
    assert T.OutlierUpToDown().build_deployment() == old              # the option is 001-only
    monkeypatch.setenv(OC.OUTLIER_001_DEPLOYMENT_ENV, "w2a2")
    with pytest.raises(KeyError):
        T.OutlierNormToGateUp().build_deployment()


def _planted_outlier_mlp_input(seed=5):
    """Toy MLP input with a planted outlier channel: x = RMSNorm(h) with gain 1 + γ, γ_3 = 30."""
    rng = np.random.default_rng(seed)
    d = 32
    gamma = rng.normal(0.0, 0.1, d)
    gamma[3] = 30.0
    h = rng.normal(size=(512, d))
    gate, up = rng.normal(size=(48, d)) * 0.2, rng.normal(size=(48, d)) * 0.2
    return h, gamma, gate, up


def _quant_mse(dep, h, gamma, gate, up, exact):
    x = rmsnorm(h, gamma)
    y = Q.simulate_linear(x, np.concatenate([gate, up]), dep.weight_qmax, dep.act_qmax)
    return float(((y - exact) ** 2).mean())


def test_harsh_deployment_damages_more_and_migration_recovers():
    h, gamma, gate, up = _planted_outlier_mlp_input()
    exact = rmsnorm(h, gamma) @ np.concatenate([gate, up]).T
    old, new = OC.get(OC.LEGACY_DEPLOYMENT), OC.get(OC.OUTLIER_001_DEPLOYMENT)
    damage_old, damage_new = (_quant_mse(d, h, gamma, gate, up, exact) for d in (old, new))
    assert damage_new > 8.0 * damage_old                               # int6 steps are 127/31 ≈ 4.1x coarser
    s = Q.smooth_scales(np.abs(1.0 + gamma), np.concatenate([gate, up]), 0.5)
    new_gamma, (g2, u2) = Q.migrate_norm(gamma, [gate, up], s)
    np.testing.assert_allclose(rmsnorm(h, new_gamma) @ np.concatenate([g2, u2]).T, exact, rtol=1e-8, atol=1e-10)
    migrated = _quant_mse(new, h, new_gamma, g2, u2, exact)
    assert migrated < 0.5 * damage_new                                # migration recovers most of it
    assert damage_new - migrated > damage_old                          # more recoverable headroom than W8A8 had


def test_torch_fake_quant_matches_numpy_simulation():
    import torch

    rng = np.random.default_rng(6)
    x, w = rng.normal(size=(2, 5, 12)), rng.normal(size=(7, 12))
    x[..., 2] *= 40.0                                                  # outlier channel sets the per-tensor scale
    for dep in OC.DEPLOYMENTS.values():
        lin = torch.nn.Linear(12, 7, bias=False).double()
        lin.weight.data = torch.from_numpy(w.copy())
        handles = J.fake_quantize_modules([lin], dep)
        with torch.no_grad():
            got = lin(torch.from_numpy(x)).numpy()
        for hd in handles:
            hd.remove()
        expected = Q.simulate_linear(x.reshape(-1, 12), w, dep.weight_qmax, dep.act_qmax).reshape(2, 5, 7)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)  # weights pass through float32


def test_prompt_and_calibration_follow_the_deployment(toy_art):
    priv = toy_art / "private"
    for task, dep, text in ((T.OutlierNormToGateUp(), OC.OUTLIER_001_DEPLOYMENT, "max|x| / 31"),
                            (T.OutlierUpToDown(), OC.OUTLIER_002_DEPLOYMENT, "max|x| / 127")):
        (priv / "spec.json").write_text(json.dumps({"seed": 0, "kind": task.kind, "deployment": dep}))
        prompt = task.prompt(toy_art)
        assert text in prompt and f"**{OC.get(dep).label}**" in prompt and "N_forward = 0" in prompt
    (priv / "spec.json").write_text(json.dumps({"seed": 0, "kind": "up_to_down"}))
    assert "**W8A8**" in T.OutlierUpToDown().prompt(toy_art)           # legacy artifacts render as before


def test_registry_and_targets():
    a, b = T.OutlierNormToGateUp, T.OutlierUpToDown
    assert len(a.quantized) == 48 and len(b.quantized) == 24 and len(a.editable) == 72 and len(b.editable) == 48
    for cls in (a, b):
        assert cls.track == "TR05" and cls.budget.n_forward == 0 and cls.probe_kinds == {}
    assert set(a.quantized) <= set(a.editable)                       # 001: consumers gate/up are editable
    assert set(b.quantized) <= set(b.editable)                       # 002: consumer down_proj is editable
