"""TR01-ESC-001..003: parsing, F1, static analyses, symmetry exactness, leakage, budget (fast; no training)."""
import json

import numpy as np
import pytest
import torch

from weightbench.ledger import Ledger
from weightbench.server import RunState
from weightbench.task import Submission
from weightbench.tracks import tr01_circuits as T
from weightbench.tracks.tr01_circuits_lib import grading as G
from weightbench.tracks.tr01_circuits_lib import static as S
from weightbench.tracks.tr01_circuits_lib.model import (Config, apply_rope, forward, init_params, public_blocks,
                                                        rope_cos_sin, to_numpy, to_torch, transform)

TOY = Config(d_model=32, n_layers=2, n_heads=4, d_mlp=48, ctx=32)


# ---------------------------------------------------------------- grading
def test_language_parsing_and_accuracy():
    truth = {"A": "python", "B": "english", "C": "german", "D": "romaji"}
    acc, d = G.score_language_id({"A": "Python", "model_B": "English prose", "C": "Deutsch", "D": "Japanese (romaji)"}, truth)
    assert acc == 1.0
    assert G.score_language_id({"assignments": {"A": "python", "B": "german"}}, truth)[0] == 0.25
    assert G.score_language_id({"A": 3, "B": None}, truth)[0] == 0.0
    assert G.score_language_id(None, truth)[0] == 0.0
    assert G.score_language_id("junk", truth)[0] == 0.0  # type: ignore[arg-type]
    assert G.norm_lang("english or german") is None  # ambiguous -> invalid
    assert G.norm_lang("c++") is None


@pytest.mark.parametrize("x,want", [([1, 3], (1, 3)), ((0, 7), (0, 7)), ({"layer": 2, "head": 5}, (2, 5)),
                                    ("L1H3", (1, 3)), ("1.3", (1, 3)), ("layer 4 head 0", (4, 0)),
                                    ("x", None), ([1], None), ([1, "a"], None), (None, None)])
def test_parse_head(x, want):
    assert G.parse_head(x) == want


def test_f1_hand_computed():
    pos = {"A": {(1, 2), (2, 3)}, "B": {(0, 1)}}
    ign = {"A": {(3, 3)}, "B": set()}
    # A: tp=1 (1,2), fp=1 (4,4), ignored (3,3) dropped, fn=1 (2,3); B: tp=1 -> F1 = 2*2/(4+1+1) = 2/3
    pred = {"A": {(1, 2), (3, 3), (4, 4)}, "B": {(0, 1)}}
    f1, d = G.f1_heads(pred, pos, ign, 6, 8)
    assert f1 == pytest.approx(2 / 3) and (d["tp"], d["fp"], d["fn"]) == (2, 1, 1)
    assert G.f1_heads({"A": set(), "B": set()}, pos, ign, 6, 8)[0] == 0.0
    assert G.f1_heads({}, {"A": set()}, {"A": set()}, 6, 8)[0] == 1.0  # nothing to find, nothing claimed
    # out-of-range heads are false positives
    f1, d = G.f1_heads({"A": {(9, 9)}}, {"A": {(0, 0)}}, {"A": set()}, 6, 8)
    assert f1 == 0.0 and d["per_model"]["A"]["out_of_range"] == 1


def test_head_report_parsing_and_truth_sets():
    rep = {"induction_heads": {"model_A": [[1, 2], "L2H3", [1, 2]], "B": "notalist", "Z": [[0, 0]]}}
    pred, bad = G.parse_head_report(rep, "induction_heads", ["A", "B"])
    assert pred == {"A": {(1, 2), (2, 3)}, "B": set()} and bad
    assert G.parse_head_report(None, "induction_heads", ["A"])[0] == {"A": set()}
    sc = {"A": [[0.0, 0.5], [0.2, 0.05]]}
    P, I = G.truth_sets(sc, 0.3, 0.1)
    assert P == {"A": {(0, 1)}} and I == {"A": {(1, 0)}}


# ---------------------------------------------------------------- model / symmetries
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_transform_is_function_preserving(level):
    P0 = to_numpy(init_params(TOY, 0))
    rng = np.random.default_rng(0)
    for k in P0:  # non-trivial gains
        if k.endswith(".g"):
            P0[k] = rng.uniform(0.5, 1.5, size=P0[k].shape).astype(np.float32)
    P1, info = transform(P0, TOY, level, np.random.default_rng(1))
    x = torch.randint(0, 256, (2, 16))
    l0, a0 = forward(to_torch(P0), x, TOY, return_attn=True)
    l1, a1 = forward(to_torch(P1), x, TOY, return_attn=True)
    assert (l0 - l1).abs().max() < 1e-4
    hp = info["head_perm"]
    for l in range(TOY.n_layers):  # attention patterns move with the head permutation
        assert torch.allclose(a1[l], a0[l][:, hp[l]], atol=1e-5)
    pub = public_blocks(P1, TOY, level)
    assert not any(k.startswith(("embed", "unembed", "ln_f")) for k in pub)
    assert any(k.endswith(".g") for k in pub) == (level == 0)


def test_rope_kernel_matches_explicit_rotation():
    rng = np.random.default_rng(0)
    cfg = Config(d_model=16, n_heads=2)
    Q, K = rng.normal(size=(8, 16)), rng.normal(size=(8, 16))
    cos, sin = rope_cos_sin(cfg, 8, "cpu")
    qr = apply_rope(torch.tensor(Q.T, dtype=torch.float32)[:, None].expand(16, 8, 8), cos, sin)  # [d, T, dh]
    kr = apply_rope(torch.tensor(K.T, dtype=torch.float32)[:, None].expand(16, 8, 8), cos, sin)
    explicit = [(qr[:, 7] * kr[:, 7 - D]).sum().item() for D in range(6)]
    f = S.rope_kernel(Q, K, np.arange(6)) * np.linalg.norm(Q) * np.linalg.norm(K)
    np.testing.assert_allclose(f, explicit, rtol=1e-4, atol=1e-4)


def test_static_methods_are_rotation_invariant():
    P0 = to_numpy(init_params(TOY, 3))
    P1, info = transform(P0, TOY, 2, np.random.default_rng(5))
    b0, b1 = public_blocks(transform(P0, TOY, 1, np.random.default_rng(5))[0], TOY, 1), public_blocks(P1, TOY, 2)
    b0 = {k: v.astype(np.float64) for k, v in b0.items()}
    b1 = {k: v.astype(np.float64) for k, v in b1.items()}
    np.testing.assert_allclose(S.qk_norms(b0), S.qk_norms(b1), rtol=1e-4)
    np.testing.assert_allclose(S.prev_token_static(b0), S.prev_token_static(b1), atol=1e-5)
    assert S.effective_input_dim(b0) == pytest.approx(S.effective_input_dim(b1), rel=1e-4)


# ---------------------------------------------------------------- task plumbing on a fake art dir
def _fake_art(tmp_path):
    art = tmp_path / "art"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    cfg = Config()
    rng = np.random.default_rng(0)
    letters = {"A": "python", "B": "english", "C": "german", "D": "romaji"}
    ind, prev = {}, {}
    for i, X in enumerate(letters):
        blocks = {}
        for l in range(cfg.n_layers):
            p = f"blocks.{l}."
            for n, shp in (("attn.W_Q", (8, 40, 320)), ("attn.W_K", (8, 40, 320)), ("attn.W_V", (8, 40, 320)),
                           ("attn.W_O", (8, 320, 40)), ("mlp.W_in", (1200, 320)), ("mlp.W_out", (320, 1200))):
                blocks[p + n] = (rng.normal(size=shp) * 0.02).astype(np.float32)
        np.savez(art / "public" / f"model_{X}.npz", **blocks)
        s = np.zeros((6, 8))
        s[2, i] = 0.9
        ind[X] = s.tolist()
        s = np.zeros((6, 8))
        s[0, i] = 0.8
        s[1, i] = 0.3  # ignored band
        prev[X] = s.tolist()
    (art / "private" / "gt.json").write_text(json.dumps({"letters": letters, "induction": ind, "prev_token": prev}))
    return art, letters


def test_tasks_end_to_end_on_fake_art(tmp_path):
    art, letters = _fake_art(tmp_path)
    t1, t2, t3 = T.LanguageID(), T.InductionHeads(), T.PrevTokenHeads()
    assert t1.evaluate(art, Submission("report", report=dict(letters))).s_plus == 1.0
    perfect = {X: [[2, i]] for i, X in enumerate(letters)}
    assert t2.evaluate(art, Submission("report", report={"induction_heads": perfect})).s_plus == 1.0
    prev = {X: [[0, i], [1, i]] for i, X in enumerate(letters)}  # (1,i) is ignored
    assert t3.evaluate(art, Submission("report", report={"previous_token_heads": prev})).s_plus == 1.0
    assert t3.evaluate(art, Submission("refusal", report={})).s_plus == 0.0
    for t in (t1, t2, t3):
        for sub in (t.baseline(art), t.reference(art)):
            ev = t.evaluate(art, sub)
            assert 0.0 <= ev.s_plus <= 1.0
        p = t.prompt(art)
        assert "wb.py submit-report" in p and "model_A.npz" in p
        for X, lang in letters.items():  # no mapping leaks into the prompt
            assert f'"{X}": "{lang}"' not in p


def test_public_dir_has_no_ground_truth(tmp_path):
    """Leakage: everything the agent receives comes from public/, which holds block weights only."""
    art, letters = _fake_art(tmp_path)
    names = {p.name for p in (art / "public").iterdir()}
    assert not any(n.startswith(("full_", "gt", "meta")) for n in names)
    for p in (art / "public").glob("*.npz"):
        with np.load(p) as z:
            assert all(k.startswith("blocks.") and not k.endswith(".g") for k in z.files)


def test_budget_zero_rejects_probes(tmp_path):
    for cls in (T.LanguageID, T.InductionHeads, T.PrevTokenHeads):
        t = cls()
        assert t.budget.n_forward == 0 and t.budget.n_probe == 0 and t.probe_kinds == {}
        led = Ledger(tmp_path / f"{t.id}.jsonl")
        st = RunState(t, tmp_path, tmp_path, led, tmp_path)
        code, out = st.probe("forward", {})
        assert code == 400 and "error" in out
        assert st.used == {"forward": 0, "probe": 0}
