"""TR05-SPARSE-001..004: 2:4 utilities, data-free covariance, validation, scoring (toy weights, no model run)."""
import json
from pathlib import Path

import numpy as np
import pytest

from weightbench.task import Submission
from weightbench.tracks import tr05_sparsity as T
from weightbench.tracks.tr05_sparsity_lib import config as C
from weightbench.tracks.tr05_sparsity_lib import jobs as J
from weightbench.tracks.tr05_sparsity_lib import sparsity as SP


# ------------------------------------------------------------------------------------- 2:4 utilities
def test_is_24_sparse():
    w = np.array([[1, 0, 2, 0, 0, 0, 3, 4]], np.float32)
    assert SP.is_24_sparse(w)
    assert not SP.is_24_sparse(np.array([[1, 2, 3, 0]], np.float32))
    assert not SP.is_24_sparse(np.zeros((2, 6)))                       # in-dim not divisible by 4


def test_magnitude_keeps_two_largest_per_group():
    w = np.array([[0.1, -0.9, 0.5, 0.2, 3.0, -2.0, 1.0, 0.0]])
    assert np.array_equal(SP.magnitude_24(w), [[0, -0.9, 0.5, 0, 3.0, -2.0, 0, 0]])


def test_obs_identity_cov_equals_magnitude():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(6, 16))
    np.testing.assert_allclose(SP.obs_24(w, np.eye(16), damp=0.0), SP.magnitude_24(w), atol=1e-10)


def test_obs_beats_magnitude_under_correlated_inputs():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(32, 64))
    a = rng.normal(size=(64, 64))
    cov = a @ a.T / 64 + 0.1 * np.eye(64)                              # strongly correlated inputs
    w_obs = SP.obs_24(w, cov)
    assert SP.is_24_sparse(w_obs)
    assert SP.output_error(w, w_obs, cov) < 0.8 * SP.output_error(w, SP.magnitude_24(w), cov)


def test_obs_diag_cov_is_scale_aware_mask():
    w = np.array([[1.0, 0.9, 0.1, 0.1]])
    cov = np.diag([0.01, 0.01, 100.0, 100.0])                          # small weights on high-variance inputs
    assert np.array_equal(SP.obs_24(w, cov, damp=0.0) != 0, [[False, False, True, True]])


# ------------------------------------------------------------------------------------- config
def test_resolve_targets_and_norms():
    p = C.PREFIX
    names = [f"{p}layers.{l}.mlp.{q}.weight" for l in range(24) for q in C.MLP_PROJS]
    names += [f"{p}layers.3.self_attn.q_proj.weight", f"{p}layers.3.self_attn.o_proj.weight",
              f"{p}layers.0.linear_attn.in_proj_qkv.weight", f"{p}layers.0.linear_attn.out_proj.weight",
              f"{p}layers.20.post_attention_layernorm.weight", f"{p}layers.3.input_layernorm.weight"]
    assert C.resolve_targets("TR05-SPARSE-001", names) == [f"{p}layers.23.mlp.{q}.weight" for q in C.MLP_PROJS]
    assert len(C.resolve_targets("TR05-SPARSE-002", names)) == 12 and len(C.resolve_targets("TR05-SPARSE-003", names)) == 72
    assert C.resolve_targets("TR05-SPARSE-004", names) == [f"{p}layers.0.linear_attn.in_proj_qkv.weight",
                                                           f"{p}layers.3.self_attn.q_proj.weight"]
    assert C.norm_before(f"{p}layers.20.mlp.up_proj.weight", names) == f"{p}layers.20.post_attention_layernorm.weight"
    assert C.norm_before(f"{p}layers.3.self_attn.q_proj.weight", names) == f"{p}layers.3.input_layernorm.weight"
    assert C.norm_before(f"{p}layers.5.mlp.up_proj.weight", names) is None
    with pytest.raises(KeyError):
        C.resolve_targets("TR05-SPARSE-001", names[:3])


# ------------------------------------------------------------------------------------- toy artifacts
P = C.PREFIX
TOY = {f"{P}layers.20.mlp.gate_proj.weight": (16, 8), f"{P}layers.20.mlp.up_proj.weight": (16, 8),
       f"{P}layers.20.mlp.down_proj.weight": (8, 16), f"{P}layers.20.post_attention_layernorm.weight": (8,)}
WRITERS = {f"{P}embed_tokens.weight": (40, 8), f"{P}layers.19.mlp.down_proj.weight": (8, 16),
           f"{P}layers.20.self_attn.o_proj.weight": (8, 8), f"{P}layers.21.self_attn.o_proj.weight": (8, 8)}
TARGET = f"{P}layers.20.mlp.down_proj.weight"


@pytest.fixture
def toy_art(tmp_path):
    import torch
    from safetensors.torch import save_file

    art = tmp_path / "art"
    (art / "public" / "model").mkdir(parents=True)
    (art / "private").mkdir()
    g = torch.Generator().manual_seed(0)
    tensors = {k: torch.randn(*s, generator=g).to(torch.bfloat16) for k, s in TOY.items()}
    tensors.update({k: torch.randn(*s, generator=g).to(torch.bfloat16) for k, s in WRITERS.items()})
    save_file(tensors, str(art / "public" / "model" / C.WEIGHTS_FILE))
    (art / "public" / "targets.json").write_text(json.dumps({TARGET: [8, 16]}))
    (art / "private" / "spec.json").write_text(json.dumps({"seed": 0, "targets": [TARGET]}))
    (art / "private" / "calibration.json").write_text(json.dumps({"nll_dense": 2.0, "nll_magnitude": 3.0}))
    return art


def test_input_covariance_down_proj_is_synthetic_and_psd(toy_art, monkeypatch):
    monkeypatch.setattr(C, "SYNTH_SAMPLES", 512)
    names = list(TOY)
    cov, kind = J.input_covariance(toy_art, TARGET, names, np.random.default_rng(0))
    assert kind == "synthetic_gate_up" and cov.shape == (16, 16)
    assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > -1e-8
    cov_up, kind_up = J.input_covariance(toy_art, f"{P}layers.20.mlp.up_proj.weight", names, np.random.default_rng(0))
    gain = 1.0 + J._tensor(toy_art, f"{P}layers.20.post_attention_layernorm.weight")
    assert kind_up == "diag_norm_gain" and np.allclose(np.diag(cov_up), gain ** 2)


@pytest.mark.parametrize("winner", ["obs_damped", "mag_rowscale", "obs_resid_r0.6", "sal_resid_p0.5", J.CALIBRATED])
def test_baseline_and_reference_jobs_are_valid(toy_art, monkeypatch, winner):
    monkeypatch.setattr(C, "SYNTH_SAMPLES", 512)
    candidates = [*J.VARIANTS, J.CALIBRATED]
    fake_dev = {"baseline_delta": 3.0, **{f"reference_{v}": 2.9 + 0.001 * i for i, v in enumerate(candidates)}}
    fake_dev[f"reference_{winner}"] = 2.7
    seen = []

    def fake_nll(art, path, split="eval"):
        assert split == "dev"                                           # selection never touches eval chunks
        seen.append(Path(path).stem)
        return {"nll": fake_dev[Path(path).stem]}

    monkeypatch.setattr(J, "mean_nll", fake_nll)
    covs = J.target_covariances(toy_art, [TARGET], J._names(toy_art), 0)
    calibrated = J.variant_delta(toy_art, [TARGET], covs, "obs")       # stand-in: no subject model in unit tests
    monkeypatch.setattr(J, "sparsegpt_calib_delta", lambda art, targets, seed: calibrated)
    J.baseline(toy_art)
    info = J.reference(toy_art)
    assert info["selected"] == winner and info["beats_magnitude_on_dev"]             # lowest dev NLL wins
    assert seen == ["baseline_delta"] + [f"reference_{v}" for v in candidates]      # every variant evaluated
    assert set(info["dev_nll"]) == {"magnitude", *candidates}
    assert info["variants"] == {**J.VARIANT_DOCS, J.CALIBRATED: J.CALIBRATED_DOC}
    assert info["selected_doc"] == info["variants"][winner]
    assert info["privileged"] == (winner == J.CALIBRATED) and info["best_data_free"] in J.VARIANTS
    assert info["covariances"][TARGET] == "synthetic_gate_up"
    assert sorted(p.name for p in (toy_art / "private").glob("reference*.npz")) == ["reference_delta.npz"]
    task = T.SparseOneLayer()
    for which in ("baseline", "reference"):
        assert task._validation_error(toy_art, task._cached(toy_art, which).delta) is None, which
    expected = (calibrated if winner == J.CALIBRATED else J.variant_delta(toy_art, [TARGET], covs, winner))[TARGET]
    np.testing.assert_array_equal(task._cached(toy_art, "reference").delta[TARGET], expected)


def test_variant_list_is_explicit():
    assert J.VARIANTS[:3] == ("diag_mask", "obs", "obs_damped")                     # the original variants stay
    assert {"mag_rowscale", "obs_resid_r0.3", "obs_resid_r0.6", "obs_resid_r0.9",
            "sal_resid_p0.5", "sal_resid_p1.0"} <= set(J.VARIANTS)
    assert len(set(J.VARIANTS)) == len(J.VARIANTS) and all(J.VARIANT_DOCS[v] for v in J.VARIANTS)
    with pytest.raises(ValueError):
        J.prune_variant(np.ones((1, 4)), np.eye(4), "nope")


def _toy_problem(seed=4):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(8, 16))
    a, b = rng.normal(size=(16, 16)), rng.normal(size=(16, 16))
    return w, a @ a.T / 16 + np.eye(16), b @ b.T / 16 + 0.5 * np.eye(16)


@pytest.mark.parametrize("variant", J.VARIANTS)
def test_prune_variants_are_exact_24_and_deterministic(variant):
    w, cov, resid = _toy_problem()
    out = J.prune_variant(w, cov, variant, resid)
    assert SP.is_24_sparse(out) and np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, J.prune_variant(w.copy(), cov.copy(), variant, resid.copy()))
    w32 = w.astype(np.float32)                                          # the float32 ΔW path of the jobs
    assert SP.is_24_sparse(w32 + (J.prune_variant(w32, cov, variant, resid) - w32).astype(np.float32))
    assert SP.is_24_sparse(J.prune_variant(w, cov, variant))           # resid_cov defaults to cov


def test_diag_mask_is_scale_aware():
    w1 = np.array([[1.0, 0.9, 0.1, 0.1]])
    assert np.array_equal(J.prune_variant(w1, np.diag([0.01, 0.01, 100.0, 100.0]), "diag_mask") != 0,
                          [[False, False, True, True]])


def test_rowscale_restores_row_norms_on_the_magnitude_mask():
    w = np.array([[3.0, -4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [0.0] * 8])
    out = J.prune_variant(w, np.eye(8), "mag_rowscale")
    assert np.array_equal(out != 0, SP.magnitude_24(w) != 0)
    # row 0: ‖w‖ = √29, survivors (3, −4, 1, 1) have norm √27 → scale √(29/27); the zero row stays zero
    np.testing.assert_allclose(out[0], np.array([3, -4, 0, 0, 1, 1, 0, 0]) * np.sqrt(29 / 27))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.linalg.norm(w, axis=1))


def test_shrinkage_endpoints_and_resid_variants():
    w, cov, resid = _toy_problem(5)
    np.testing.assert_allclose(SP.shrink_to_identity(resid, 1.0), np.eye(16))
    np.testing.assert_allclose(SP.shrink_to_identity(4.0 * resid, 0.0), resid / np.mean(np.diag(resid)))
    np.testing.assert_allclose(SP.obs_24(w, SP.shrink_to_identity(resid, 1.0), damp=0.0), SP.magnitude_24(w))
    # heavier shrinkage stays closer to the magnitude solution
    gap = [np.abs(J.prune_variant(w, cov, f"obs_resid_r{r}", resid) - SP.magnitude_24(w)).sum() for r in J.RESID_SHRINK]
    assert gap[0] > gap[-1]
    # saliency mask: a weight on a high-variance input survives over a slightly larger one on a quiet input
    w1 = np.array([[1.0, 0.9, 0.5, 0.1]])
    out = J.prune_variant(w1, np.eye(4), "sal_resid_p1.0", np.diag([0.01, 1.0, 1.0, 1.0]))
    assert np.array_equal(out != 0, [[False, True, True, False]]) and np.array_equal(out[out != 0], [0.9, 0.5])


def test_residual_second_moments_use_only_earlier_writers(toy_art):
    names = J._names(toy_art)
    w = {n: J._tensor(toy_art, n).astype(np.float64) for n in WRITERS}
    emb = w[f"{P}embed_tokens.weight"]
    moments = J.residual_second_moments(toy_art, names, {(20, 0), (20, 1)}, 8)
    attn20 = emb.T @ emb / len(emb) + w[f"{P}layers.19.mlp.down_proj.weight"] @ w[f"{P}layers.19.mlp.down_proj.weight"].T
    mlp20 = attn20 + w[f"{P}layers.20.self_attn.o_proj.weight"] @ w[f"{P}layers.20.self_attn.o_proj.weight"].T
    for site, raw in (((20, 0), attn20), ((20, 1), mlp20)):            # layer-20 down_proj / layer-21 never enter
        np.testing.assert_allclose(moments[site], raw / np.mean(np.diag(raw)), rtol=1e-6)


def test_target_covariances_are_deterministic_and_psd(toy_art, monkeypatch):
    monkeypatch.setattr(C, "SYNTH_SAMPLES", 512)
    names = J._names(toy_art)
    targets = [TARGET, f"{P}layers.20.mlp.up_proj.weight"]
    a, b = (J.target_covariances(toy_art, targets, names, 3) for _ in range(2))
    for t in targets:
        np.testing.assert_array_equal(a[t]["resid"], b[t]["resid"])
        np.testing.assert_array_equal(a[t]["cov"], b[t]["cov"])
        assert np.allclose(a[t]["resid"], a[t]["resid"].T) and np.linalg.eigvalsh(a[t]["resid"]).min() > -1e-8
    assert a[TARGET]["resid"].shape == (16, 16) and a[targets[1]]["resid"].shape == (8, 8)
    assert not np.allclose(a[targets[1]]["resid"], np.diag(np.diag(a[targets[1]]["resid"])))  # anisotropic


def test_validation_rejects(toy_art):
    task = T.SparseOneLayer()
    w = J._tensor(toy_art, TARGET)
    assert "non-target" in task._validation_error(toy_art, {f"{P}layers.20.mlp.up_proj.weight": np.zeros((16, 8))})
    assert "missing" in task._validation_error(toy_art, {})
    assert "not 2:4" in task._validation_error(toy_art, {TARGET: np.zeros((8, 16), np.float32)})
    assert "shape" in task._validation_error(toy_art, {TARGET: np.zeros((8, 15), np.float32)})
    assert task._validation_error(toy_art, {TARGET: (SP.magnitude_24(w) - w).astype(np.float32)}) is None


def test_evaluate_scoring(toy_art, monkeypatch):
    task = T.SparseOneLayer()
    w = J._tensor(toy_art, TARGET)
    sub = Submission("delta_w", delta={TARGET: (SP.magnitude_24(w) - w).astype(np.float32)})
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: {"nll": 2.25, "ppl": float(np.exp(2.25))})
    ev = task.evaluate(toy_art, sub)
    assert ev.valid and ev.s_plus == pytest.approx(0.75)                # (3.0 - 2.25) / (3.0 - 2.0)
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: {"nll": 3.5, "ppl": 33.0})
    assert task.evaluate(toy_art, sub).s_plus == 0.0                     # worse than magnitude: clamped
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: pytest.fail("invalid edits must not be evaluated"))
    assert not task.evaluate(toy_art, Submission("delta_w", delta={TARGET: np.zeros((8, 16))})).valid


def test_registered_budgets():
    for cls in (T.SparseOneLayer, T.SparseLateMLPs, T.SparseAllMLPs, T.SparseAttentionInputs):
        assert cls.track == "TR05" and cls.budget.n_forward == 0 and cls.budget.n_probe == 0 and cls.probe_kinds == {}
    assert set(C.TASKS) == {c.id for c in (T.SparseOneLayer, T.SparseLateMLPs, T.SparseAllMLPs, T.SparseAttentionInputs)}


def test_prompt_states_rules(toy_art):
    prompt = T.SparseOneLayer().prompt(toy_art)
    assert "N_forward = 0" in prompt and "2:4" in prompt and "3.0000" in prompt and TARGET in prompt


# ------------------------------------------------------------------------------------- propagation_depth ladder
FULL_NAMES = [f"{P}layers.{l}.mlp.{q}.weight" for l in range(24) for q in C.MLP_PROJS] + [
    f"{P}layers.{l}.post_attention_layernorm.weight" for l in range(24)] + [f"{P}layers.3.self_attn.q_proj.weight"]


def _attn(layers):
    """Attention input projections of ``layers`` as the checkpoint names them (q/k/v on full-attention layers, in_proj_qkv
    / in_proj_z on linear-attention layers; every fourth layer is full attention in the real model)."""
    return sorted(f"{P}layers.{l}.self_attn.{q}_proj.weight" if l % 4 == 3 else f"{P}layers.{l}.linear_attn.{q}.weight"
                  for l in layers for q in (("q", "k", "v") if l % 4 == 3 else ("in_proj_qkv", "in_proj_z")))


ATTN_NAMES = _attn(range(24)) + [f"{P}layers.{l}.self_attn.o_proj.weight" for l in range(3, 24, 4)] + [
    f"{P}layers.{l}.linear_attn.out_proj.weight" for l in range(24) if l % 4 != 3]
FULL_NAMES = FULL_NAMES + [n for n in ATTN_NAMES if n not in FULL_NAMES]


def _mlps(layers):
    return [f"{P}layers.{l}.mlp.{q}.weight" for l in layers for q in C.MLP_PROJS]


def test_ladder_targets_per_level():
    resolve = C.resolve_targets
    assert resolve("TR05-SPARSE-001", FULL_NAMES, 0) == _mlps([23]) == C.TASKS["TR05-SPARSE-001"]["targets"]
    l1 = resolve("TR05-SPARSE-001", FULL_NAMES, 1)
    assert l1 == _mlps(range(21, 24)) and len(l1) == 9
    assert resolve("TR05-SPARSE-001", FULL_NAMES, 2) == _mlps(range(15, 24))
    l3 = resolve("TR05-SPARSE-001", FULL_NAMES, 3)
    assert l3 == _mlps(range(24)) and len(l3) == 72
    assert resolve("TR05-SPARSE-001", FULL_NAMES, 5) == l3                         # 3^5 = 243, capped at 24 layers
    assert [len(C.ladder_layers(L)) for L in range(4)] == [1, 3, 9, 24]
    assert resolve("TR05-SPARSE-001", FULL_NAMES) == _mlps([23])                    # no level = configured set
    with pytest.raises(ValueError):
        resolve("TR05-SPARSE-003", FULL_NAMES, 3)                                   # input-side ladder stops at L2
    with pytest.raises(ValueError):
        C.ladder_layers(-1)
    with pytest.raises(KeyError):
        resolve("TR05-SPARSE-001", FULL_NAMES[:60], 3)                             # missing layers are an error


def test_ladder_level_on_task():
    assert T.SparseOneLayer.knob_level == 0
    assert all(cls.knob_level is None for cls in (T.SparseAllMLPs, T.SparseAttentionInputs))
    with pytest.raises(ValueError, match="supported knob levels"):
        T.SparseAllMLPs().at_level(3)
    default = T.SparseOneLayer()
    assert default.spec(0, _mlps([23])) == {"task": "TR05-SPARSE-001", "seed": 0, "targets": _mlps([23]),
                                            "model": [C.MODEL_ID, C.MODEL_REV], "dataset": C.DATASET,
                                            "seq_len": C.SEQ_LEN}              # the original spec, unchanged
    assert default.knob_record()["level"] == 0 and default.knob_record()["layers"] == [23]
    assert default._title() == C.TASKS["TR05-SPARSE-001"]["title"]
    l2 = T.SparseOneLayer().at_level(2)
    spec = l2.spec(0, _mlps(range(15, 24)))
    assert spec["level"] == 2 and spec["layers"] == list(range(15, 24))
    assert l2.knob_record()["value"] == 9 and "layers 15-23" in l2._title()
    assert T.SparseAllMLPs().knob_record() is None



def test_per_task_base_layers_and_cap():
    """SPARSE-002 has base 4: min(24, 4 * 3^L) layers, so L0 is its default layers 20-23 and the cap applies from L2."""
    assert [len(C.ladder_layers(L, "TR05-SPARSE-002")) for L in range(4)] == [4, 12, 24, 24]
    assert C.ladder_layers(0, "TR05-SPARSE-002") == [20, 21, 22, 23]
    assert C.ladder_layers(1, "TR05-SPARSE-002") == list(range(12, 24))
    assert C.ladder_layers(2, "TR05-SPARSE-002") == list(range(24))                 # 36 capped at 24 layers
    assert C.ladder_layers(7, "TR05-SPARSE-002") == list(range(24))                 # cap holds at large levels
    assert [len(C.ladder_layers(L)) for L in range(4)] == [1, 3, 9, 24]             # SPARSE-001 (base 1) unchanged
    with pytest.raises(ValueError, match="not on the TR-05 sparsity ladder"):
        C.ladder_layers(0, "TR05-DEAD-001")
    assert C.ladder_layers(1, "TR05-SPARSE-003") == [0, 1, 2]                       # input side
    with pytest.raises(ValueError):
        C.ladder_layers(-1, "TR05-SPARSE-002")
    with pytest.raises(ValueError, match="supported knob levels"):
        C.task_targets("TR05-SPARSE-003", 3)                                         # L3 would repeat the default
    assert C.task_targets("TR05-SPARSE-004", 1) == "attn_inputs"                     # resolved from the checkpoint names


def test_sparse002_level_zero_is_the_default_target_set():
    default = _mlps(range(20, 24))
    assert C.TASKS["TR05-SPARSE-002"]["targets"] == default and len(default) == 12
    assert C.resolve_targets("TR05-SPARSE-002", FULL_NAMES, 0) == default           # L0 (knob default)
    assert C.resolve_targets("TR05-SPARSE-002", FULL_NAMES) == default              # no level
    assert C.resolve_targets("TR05-SPARSE-002", FULL_NAMES, 1) == _mlps(range(12, 24))
    assert len(C.resolve_targets("TR05-SPARSE-002", FULL_NAMES, 2)) == 72


def test_sparse002_knob_level_spec_and_record():
    default = T.SparseLateMLPs()
    assert default.knob_level == 0 and default.level == 0 and default._nondefault_level() is False
    assert default.spec(0, _mlps(range(20, 24))) == {
        "task": "TR05-SPARSE-002", "seed": 0, "targets": _mlps(range(20, 24)),
        "model": [C.MODEL_ID, C.MODEL_REV], "dataset": C.DATASET, "seq_len": C.SEQ_LEN}   # no level key at L0
    assert default.knob_record() == {"knob": "propagation_depth", "level": 0, "formula": "1 * 3**L", "v0": 1,
                                     "growth": 3, "quantity": default.knob_record()["quantity"], "value": 1,
                                     "layers": [20, 21, 22, 23]}
    l1 = T.SparseLateMLPs().at_level(1)
    assert l1.spec(0, _mlps(range(12, 24)))["layers"] == list(range(12, 24))
    assert l1.knob_record()["value"] == 3 and l1.knob_record()["layers"] == list(range(12, 24))
    assert "layers 12-23" in l1._title() and "knob level 1" in l1._title()
    assert T.SparseLateMLPs().at_level(2).knob_record()["layers"] == list(range(24))
    with pytest.raises(ValueError):
        T.SparseLateMLPs().at_level(-1)                                              # rejected before any build


def test_sparse002_default_build_matches_default_targets_and_spec(tmp_path, monkeypatch):
    runner = _fake_build(monkeypatch, tmp_path)
    d = runner.build("TR05-SPARSE-002", 0)
    assert d == tmp_path / "TR05-SPARSE-002" / "seed0" and "level" not in json.loads((d / ".built").read_text())
    assert list(json.loads((d / "public" / "targets.json").read_text())) == _mlps(range(20, 24))
    spec = json.loads((d / "private" / "spec.json").read_text())
    assert spec == {"task": "TR05-SPARSE-002", "seed": 0, "targets": _mlps(range(20, 24)),
                    "model": [C.MODEL_ID, C.MODEL_REV], "dataset": list(C.DATASET), "seq_len": C.SEQ_LEN}
    assert json.loads((d / "public" / "knob.json").read_text())["layers"] == [20, 21, 22, 23]


def test_sparse002_level_build_and_rejection(tmp_path, monkeypatch):
    runner = _fake_build(monkeypatch, tmp_path)
    d = runner.build("TR05-SPARSE-002", 0, level=1)
    assert d == tmp_path / "TR05-SPARSE-002" / "seed0_L1"
    spec = json.loads((d / "private" / "spec.json").read_text())
    assert spec["level"] == 1 and spec["layers"] == list(range(12, 24)) and spec["targets"] == _mlps(range(12, 24))
    assert json.loads((d / "public" / "knob.json").read_text())["value"] == 3
    assert runner.make_run_id("TR05-SPARSE-002", "baseline", 0, 1, 1) == "TR05-SPARSE-002__baseline__s0__r1__L1"
    # an unsupported level of the full-coverage task is rejected before any model step runs
    calls = []
    monkeypatch.setattr(T, "install_model", lambda pub: calls.append("install") or pub / "model")
    with pytest.raises(ValueError, match="supported knob levels"):
        runner.build("TR05-SPARSE-003", 0, level=3)
    assert calls == [] and not (tmp_path / "TR05-SPARSE-003").exists()


class _FakeSafeOpen:
    """Stands in for ``safetensors.safe_open`` over the full checkpoint's names (no file, no model)."""

    def __init__(self, path, framework):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return FULL_NAMES

    def get_slice(self, name):
        shape = [1024, 3584] if "down_proj" in name else [3584, 1024]
        return type("Slice", (), {"get_shape": lambda self: shape})()


def _fake_build(monkeypatch, tmp_path):
    """Stub the model download and every heavy step, so ``build`` only resolves and writes targets / spec / knob."""
    import safetensors

    from weightbench import runner

    monkeypatch.setattr(runner, "ART", tmp_path)
    monkeypatch.setenv("WB_HEAVY", "local")
    monkeypatch.setattr(safetensors, "safe_open", _FakeSafeOpen)
    monkeypatch.setattr(T, "install_model", lambda pub: pub / "model")
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: {"nll": 3.0, "runtime": 0.0})
    return runner


def test_default_build_targets_unchanged(tmp_path, monkeypatch):
    runner = _fake_build(monkeypatch, tmp_path)
    d = runner.build("TR05-SPARSE-001", 0)
    assert d == tmp_path / "TR05-SPARSE-001" / "seed0" and "level" not in json.loads((d / ".built").read_text())
    assert list(json.loads((d / "public" / "targets.json").read_text())) == _mlps([23])
    spec = json.loads((d / "private" / "spec.json").read_text())
    assert spec["targets"] == _mlps([23]) and "level" not in spec
    assert json.loads((d / "public" / "knob.json").read_text())["layers"] == [23]
    assert runner.nondefault_level("TR05-SPARSE-001", 0) is None


def test_level_build_goes_to_its_own_dir_and_run_id(tmp_path, monkeypatch):
    runner = _fake_build(monkeypatch, tmp_path)
    d = runner.build("TR05-SPARSE-001", 0, level=1)
    assert d == tmp_path / "TR05-SPARSE-001" / "seed0_L1"
    assert json.loads((d / ".built").read_text())["level"] == 1
    assert list(json.loads((d / "public" / "targets.json").read_text())) == _mlps(range(21, 24))
    spec = json.loads((d / "private" / "spec.json").read_text())
    assert spec["level"] == 1 and spec["layers"] == [21, 22, 23] and spec["targets"] == _mlps(range(21, 24))
    knob = json.loads((d / "public" / "knob.json").read_text())
    assert knob["level"] == 1 and knob["value"] == 3 and knob["layers"] == [21, 22, 23]
    assert not (tmp_path / "TR05-SPARSE-001" / "seed0").exists()
    level = runner.nondefault_level("TR05-SPARSE-001", 1)
    assert runner.make_run_id("TR05-SPARSE-001", "baseline", 0, 1, level) == "TR05-SPARSE-001__baseline__s0__r1__L1"
    with pytest.raises(ValueError, match="supported knob levels"):
        runner.nondefault_level("TR05-SPARSE-003", 3)


@pytest.mark.parametrize("level", [0, 1])
def test_reference_and_validation_cover_every_ladder_target(tmp_path, monkeypatch, level):
    """Baseline / reference jobs and the 2:4 validation iterate over whatever target set the level selects."""
    import torch
    from safetensors.torch import save_file

    monkeypatch.setattr(C, "SYNTH_SAMPLES", 256)
    monkeypatch.setattr(C, "N_LAYERS", 4)                                           # toy 4-layer checkpoint
    layers = C.ladder_layers(level)
    targets = _mlps(layers)
    art = tmp_path / "art"
    (art / "public" / "model").mkdir(parents=True)
    (art / "private").mkdir()
    g = torch.Generator().manual_seed(0)
    tensors = {f"{P}embed_tokens.weight": torch.randn(40, 8, generator=g)}
    for l in range(4):
        tensors[f"{P}layers.{l}.mlp.gate_proj.weight"] = torch.randn(16, 8, generator=g)
        tensors[f"{P}layers.{l}.mlp.up_proj.weight"] = torch.randn(16, 8, generator=g)
        tensors[f"{P}layers.{l}.mlp.down_proj.weight"] = torch.randn(8, 16, generator=g)
        tensors[f"{P}layers.{l}.post_attention_layernorm.weight"] = torch.randn(8, generator=g)
    save_file({k: v.to(torch.bfloat16) for k, v in tensors.items()}, str(art / "public" / "model" / C.WEIGHTS_FILE))
    assert C.resolve_targets("TR05-SPARSE-001", list(tensors), level) == targets
    (art / "public" / "targets.json").write_text(json.dumps({t: list(tensors[t].shape) for t in targets}))
    (art / "private" / "spec.json").write_text(json.dumps({"seed": 0, "targets": targets}))
    monkeypatch.setattr(J, "mean_nll", lambda art, path, split="eval": {"nll": 3.0})
    monkeypatch.setattr(J, "sparsegpt_calib_delta", lambda art, targets, seed: J.variant_delta(
        art, targets, J.target_covariances(art, targets, J._names(art), seed), "obs"))
    J.baseline(art)
    J.reference(art)
    task = T.SparseOneLayer().at_level(level)
    for which in ("baseline", "reference"):
        delta = task._cached(art, which).delta
        assert sorted(delta) == sorted(targets) and task._validation_error(art, delta) is None, which


def test_calibration_stats_are_exact_and_stop_early():
    import torch

    from weightbench.tracks.tr05_sparsity_lib import calib as CB

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.emb = torch.nn.Embedding(10, 4)
            self.emb.weight.data = torch.randn(10, 4, generator=g)
            self.a, self.b = torch.nn.Linear(4, 6), torch.nn.Linear(6, 3)
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            h = self.a(self.emb(input_ids))
            return self.b(torch.relu(h))

    m, ids = Tiny(), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    st = CB.input_stats(m, {"a": m.a}, ids, gram=True, batch=2)
    x = m.emb.weight.data[ids.flatten()].double()
    np.testing.assert_allclose(st["a"]["gram"], (x.T @ x / len(x)).numpy(), rtol=1e-12)
    np.testing.assert_allclose(st["a"]["absmax"], x.abs().amax(0).numpy())
    assert m.calls == 2                                                   # two batches, b never needed
    names = [f"{P}layers.1.mlp.down_proj.weight", f"{P}layers.1.mlp.up_proj.weight",
             f"{P}layers.0.self_attn.q_proj.weight", f"{P}layers.1.mlp.gate_proj.weight"]
    assert CB.stages(names) == [[names[2]], [names[1], names[3]], [names[0]]]


# ------------------------------------------------------------------------------------- SPARSE-004 attention ladder
def test_sparse004_ladder_targets_and_default():
    """Bounded explicit levels select the attention inputs of the last min(24, 3^L) layers; no layer index is stored."""
    resolve = C.resolve_targets
    assert [len(C.ladder_layers(L, "TR05-SPARSE-004")) for L in range(4)] == [1, 3, 9, 24]
    assert resolve("TR05-SPARSE-004", FULL_NAMES, 0) == _attn([23]) and len(_attn([23])) == 3
    assert resolve("TR05-SPARSE-004", FULL_NAMES, 1) == _attn(range(21, 24))
    assert resolve("TR05-SPARSE-004", FULL_NAMES, 2) == _attn(range(15, 24))
    default = resolve("TR05-SPARSE-004", FULL_NAMES)                                   # unversioned default: all layers
    assert default == _attn(range(24)) and len(default) == 6 * 3 + 18 * 2
    assert resolve("TR05-SPARSE-004", FULL_NAMES, 3) == default                       # L3 is the saturated rung
    assert resolve("TR05-SPARSE-004", FULL_NAMES, 5) == default
    assert all(n in FULL_NAMES for n in default) and not any("o_proj" in n or "out_proj" in n for n in default)
    with pytest.raises(ValueError):
        resolve("TR05-SPARSE-004", FULL_NAMES, -1)


def test_sparse004_task_levels_spec_and_record():
    cls = T.SparseAttentionInputs
    assert cls.knob_level is None and cls.knob_levels == (0, 1, 2, 3)               # unversioned default is kept
    assert cls().level is None and cls().knob_record() is None and cls()._nondefault_level() is False
    assert cls()._title() == C.TASKS["TR05-SPARSE-004"]["title"]
    assert cls().spec(0, ["a"]) == {"task": "TR05-SPARSE-004", "seed": 0, "targets": ["a"],
                                    "model": [C.MODEL_ID, C.MODEL_REV], "dataset": C.DATASET, "seq_len": C.SEQ_LEN}
    for level in (4, -1, True, 1.5):
        with pytest.raises(ValueError):
            cls().at_level(level)
    l1 = cls().at_level(1)
    assert l1.level == 1 and l1._nondefault_level() and l1.knob_record()["layers"] == [21, 22, 23]
    assert l1.knob_record()["value"] == 3
    assert l1.spec(0, ["a"])["level"] == 1 and "attention input projections of layers 21-23" in l1._title()
    assert cls().at_level(0).knob_record()["layers"] == [23]
    assert cls().at_level(3).knob_record()["layers"] == list(range(24))
    assert cls().at_level(None).level is None


def test_sparse004_default_and_level_builds(tmp_path, monkeypatch):
    runner = _fake_build(monkeypatch, tmp_path)
    d = runner.build("TR05-SPARSE-004", 0)
    assert d == tmp_path / "TR05-SPARSE-004" / "seed0" and "level" not in json.loads((d / ".built").read_text())
    default_targets = json.loads((d / "public" / "targets.json").read_text())
    assert list(default_targets) == _attn(range(24))
    default_spec = json.loads((d / "private" / "spec.json").read_text())
    assert "level" not in default_spec and default_spec["targets"] == list(default_targets)
    assert not (d / "public" / "knob.json").exists()                                   # the default build is untouched
    assert "knob level" not in T.SparseAttentionInputs().prompt(d)
    before = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    # L3 saturates to the default target set; every other level is a strict suffix of layers
    d3 = runner.build("TR05-SPARSE-004", 0, level=3)
    assert d3 == tmp_path / "TR05-SPARSE-004" / "seed0_L3"
    assert json.loads((d3 / "public" / "targets.json").read_text()) == default_targets
    after = sorted(p.relative_to(d3).as_posix() for p in d3.rglob("*") if p.is_file())
    assert after == sorted(before + ["public/knob.json"])                              # new by design: knob.json only
    spec3 = json.loads((d3 / "private" / "spec.json").read_text())
    assert {k: v for k, v in spec3.items() if k not in ("level", "layers")} == default_spec
    assert spec3["level"] == 3 and spec3["layers"] == list(range(24))
    d1 = runner.build("TR05-SPARSE-004", 1, level=1)
    assert d1 == tmp_path / "TR05-SPARSE-004" / "seed1_L1"
    assert list(json.loads((d1 / "public" / "targets.json").read_text())) == _attn(range(21, 24))
    spec1 = json.loads((d1 / "private" / "spec.json").read_text())
    assert spec1["level"] == 1 and spec1["layers"] == [21, 22, 23] and spec1["targets"] == _attn(range(21, 24))
    assert "attention input projections of layers 21-23 (knob level 1)" in T.SparseAttentionInputs().at_level(1).prompt(d1)
    assert runner.make_run_id("TR05-SPARSE-004", "reference", 1, 60, 1) == "TR05-SPARSE-004__reference__s1__r60__L1"


def test_sparse004_level_rejected_before_any_build(tmp_path, monkeypatch):
    runner = _fake_build(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(T, "install_model", lambda pub: calls.append("install") or pub / "model")
    for level in (4, 9):
        with pytest.raises(ValueError, match="supported knob levels"):
            runner.build("TR05-SPARSE-004", 0, level=level)
    with pytest.raises(ValueError, match="supported knob levels"):
        runner.build("TR05-SPARSE-003", 0, level=3)
    assert calls == [] and not (tmp_path / "TR05-SPARSE-004").exists() and not (tmp_path / "TR05-SPARSE-003").exists()


def test_sparse004_level_public_layout_has_no_private_artifact(tmp_path, monkeypatch):
    """Extends the leakage check: the knob record is public (layer list, count), spec/calibration/deltas stay private."""
    runner = _fake_build(monkeypatch, tmp_path)
    d = runner.build("TR05-SPARSE-004", 0, level=2)
    public = {p.relative_to(d / "public").as_posix() for p in (d / "public").rglob("*") if p.is_file()}
    private = {p.relative_to(d / "private").as_posix() for p in (d / "private").rglob("*") if p.is_file()}
    assert {"spec.json", "calibration.json"} <= private
    assert public == {"targets.json", "knob.json"}, sorted(public)
    text = (d / "public" / "knob.json").read_text() + (d / "public" / "targets.json").read_text()
    assert "nll" not in text and "seed" not in text


@pytest.mark.parametrize("level", [0, 1])
def test_sparse004_reference_and_validation_cover_every_ladder_target(tmp_path, monkeypatch, level):
    """Baseline / reference jobs and the 2:4 validation run on the attention inputs the level selects (toy checkpoint)."""
    import torch
    from safetensors.torch import save_file

    monkeypatch.setattr(C, "SYNTH_SAMPLES", 256)
    monkeypatch.setattr(C, "N_LAYERS", 4)
    layers = C.ladder_layers(level, "TR05-SPARSE-004")
    assert layers == ([3] if level == 0 else [1, 2, 3])
    art = tmp_path / "art"
    (art / "public" / "model").mkdir(parents=True)
    (art / "private").mkdir()
    g = torch.Generator().manual_seed(0)
    tensors = {f"{P}embed_tokens.weight": torch.randn(40, 8, generator=g)}
    for l in range(4):
        tensors[f"{P}layers.{l}.input_layernorm.weight"] = torch.randn(8, generator=g)
        if l == 3:
            for q in ("q", "k", "v"):
                tensors[f"{P}layers.{l}.self_attn.{q}_proj.weight"] = torch.randn(16, 8, generator=g)
            tensors[f"{P}layers.{l}.self_attn.o_proj.weight"] = torch.randn(8, 16, generator=g)
        else:
            for q in ("in_proj_qkv", "in_proj_z"):
                tensors[f"{P}layers.{l}.linear_attn.{q}.weight"] = torch.randn(16, 8, generator=g)
            tensors[f"{P}layers.{l}.linear_attn.out_proj.weight"] = torch.randn(8, 16, generator=g)
    save_file({k: v.to(torch.bfloat16) for k, v in tensors.items()}, str(art / "public" / "model" / C.WEIGHTS_FILE))
    targets = C.resolve_targets("TR05-SPARSE-004", list(tensors), level)
    assert targets == sorted(n for n in tensors if C.ATTN_INPUT_PATTERN.match(n) and C.layer_of(n) in layers)
    (art / "public" / "targets.json").write_text(json.dumps({t: list(tensors[t].shape) for t in targets}))
    (art / "private" / "spec.json").write_text(json.dumps({"seed": 0, "targets": targets}))
    monkeypatch.setattr(J, "mean_nll", lambda art, path, split="eval": {"nll": 3.0})
    monkeypatch.setattr(J, "sparsegpt_calib_delta", lambda art, targets, seed: J.variant_delta(
        art, targets, J.target_covariances(art, targets, J._names(art), seed), "obs"))
    J.baseline(art)
    J.reference(art)
    task = T.SparseAttentionInputs().at_level(level)
    for which in ("baseline", "reference"):
        delta = task._cached(art, which).delta
        assert sorted(delta) == sorted(targets) and task._validation_error(art, delta) is None, which
