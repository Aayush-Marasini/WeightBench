"""TR01-BEH-001..003: report scoring, symmetries, generators, toy end-to-end builds (< 5 s each)."""
import json

import numpy as np
import pytest

from weightbench.task import Submission
from weightbench.tracks import tr01_behavioral as T
from weightbench.tracks.tr01_behavioral_lib import boolfn as B
from weightbench.tracks.tr01_behavioral_lib import build as BLD
from weightbench.tracks.tr01_behavioral_lib import common as C
from weightbench.tracks.tr01_behavioral_lib import ddpm as D
from weightbench.tracks.tr01_behavioral_lib import games as G

TASKS = {"beh001": T.GameIdentification, "beh002": T.DiffusionShapeIdentification,
         "beh003": T.AlgorithmicFunctionIdentification}


def test_candidate_lists_match_generators():
    assert T.GameIdentification.candidates == G.GAMES
    assert T.DiffusionShapeIdentification.candidates == D.SHAPES
    assert T.AlgorithmicFunctionIdentification.candidates == B.FUNCS
    for t in TASKS.values():
        assert t.budget.n_forward == 0 and t.budget.n_probe == 0 and t.submission == "report"


def test_score_report_parsing():
    truth = {"m_0": "breakout", "m_1": "space_invaders", "m_2": "freeway"}
    cands = G.GAMES
    acc, d = C.score_report({"m_0": "Breakout", "m_1": "space-invaders", "m_2": "asterix"}, truth, cands)
    assert acc == pytest.approx(2 / 3) and d["items"]["m_2"]["correct"] is False
    assert C.score_report({"labels": {"m_0": "breakout"}}, truth, cands)[0] == pytest.approx(1 / 3)
    assert C.score_report({"m_0": 3, "m_1": None, "m_2": "pong"}, truth, cands)[0] == 0.0  # invalid -> wrong
    assert C.score_report(None, truth, cands)[0] == 0.0
    assert C.score_report("garbage", truth, cands)[0] == 0.0  # type: ignore[arg-type]
    assert C.score_report({}, {}, cands)[0] == 0.0


@pytest.mark.parametrize("scale", [0.0, 0.5])
def test_permute_hidden_preserves_relu_function(scale):
    rng = np.random.default_rng(0)
    L = [(rng.normal(size=(7, 5)), rng.normal(size=7)), (rng.normal(size=(4, 7)), rng.normal(size=4)),
         (rng.normal(size=(2, 4)), rng.normal(size=2))]
    P = C.permute_hidden(L, np.random.default_rng(1), positive_scale=scale)
    x = rng.normal(size=(10, 5))
    np.testing.assert_allclose(G.mlp_forward(L, x), G.mlp_forward(P, x), rtol=1e-4, atol=1e-4)
    assert not np.allclose(L[0][0], P[0][0])


def test_draw_labels_deterministic_and_distinct():
    a = C.draw_labels(np.random.default_rng(3), G.GAMES, 8, 4)
    b = C.draw_labels(np.random.default_rng(3), G.GAMES, 8, 4)
    assert a == b and len(set(a)) >= 4


def test_boolfn_definitions_match_source():
    params = B.make_params(np.random.default_rng(0))
    ns = {}
    exec(B.candidates_source(params), ns)  # the exact code shown to the agent
    X = B.all_inputs()
    idx = np.random.default_rng(1).choice(len(X), 3000, replace=False)
    idx = np.concatenate([idx, [0, 1, 2, 3, 255, 256, 65535, 7, 0b1110000]])
    for f in B.FUNCS:
        tt = B.truth_table(f, params)
        assert 0.0 < tt.mean() < 1.0, f
        assert all(bool(ns["CANDIDATES"][f](int(n))) == bool(tt[n]) for n in idx), f


def test_shapes_render_and_differ():
    rng = np.random.default_rng(0)
    means = {s: np.mean([D.render(s, rng) for _ in range(20)], 0) for s in D.SHAPES}
    for s, m in means.items():
        assert m.shape == (12, 12) and m.min() >= -1 and m.max() <= 1 and (m > -1).any(), s
    keys = list(means)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert np.abs(means[keys[i]] - means[keys[j]]).mean() > 0.02


def test_game_encoding_shared_width():
    for g in G.GAMES:
        env = G.make_env(g, 0)
        x = G.encode(g, env)
        assert x.shape == (G.N_IN,) and set(np.unique(x)) <= {0.0, 1.0}
        assert set(G.CHANNEL_MAP[g]) | set(G.DROPPED.get(g, [])) == set(env.env.channels)  # all channels accounted for
        for cls in G.EQUIV[g]:
            assert set(env.minimal_action_set()) & set(cls), (g, cls)


def _toy(task_key, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "N_PROCS", 1)
    art = tmp_path / task_key
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir(parents=True)
    BLD.main(["--task", task_key, "--art", str(art), "--seed", "0", "--toy", "--out", str(art / "build_out.json")])
    return art


@pytest.mark.parametrize("key", ["beh001", "beh002", "beh003"])
def test_toy_build_end_to_end(key, tmp_path, monkeypatch):
    art = _toy(key, tmp_path, monkeypatch)
    task = TASKS[key]()
    gt = json.loads((art / "private" / "gt.json").read_text())
    names = T._models(art)
    assert sorted(gt) == sorted(names) and len(names) >= 3
    # public must not leak labels: no label strings in filenames, no private files
    pub = [p for p in (art / "public").rglob("*") if p.is_file()]
    assert not any(p.name in ("gt.json", "reference.json", "shadow.json") for p in pub)
    assert not any(lbl in p.stem for p in (art / "public" / "models").iterdir() for lbl in task.candidates)
    prompt = task.prompt(art)
    assert "submit-report report.json" in prompt and all(n in prompt for n in names)
    for sub in (task.baseline(art), task.reference(art)):
        assert sub.type == "report" and set(sub.report) == set(names)
        assert all(v in task.candidates for v in sub.report.values())
        ev = task.evaluate(art, sub)
        assert 0.0 <= ev.s_plus <= 1.0 and len(ev.details["items"]) == len(names)
    assert task.evaluate(art, Submission("report", report=gt)).s_plus == 1.0
    assert task.evaluate(art, Submission("report", report={})).s_plus == 0.0
    assert task.evaluate(art, Submission("refusal", report=gt)).s_plus == 0.0


def test_toy_build_deterministic(tmp_path, monkeypatch):
    a = _toy("beh003", tmp_path / "a", monkeypatch)
    b = _toy("beh003", tmp_path / "b", monkeypatch)
    for f in (a / "public" / "models").iterdir():
        with np.load(f) as za, np.load(b / "public" / "models" / f.name) as zb:
            for k in za.files:
                np.testing.assert_array_equal(za[k], zb[k])
    assert (a / "private" / "gt.json").read_text() == (b / "private" / "gt.json").read_text()


def _toy_policy(rng, h1=25, h2=15):
    f = np.float32
    return [(rng.normal(0, .1, (h1, G.N_IN)).astype(f), rng.normal(0, .1, h1).astype(f)),
            (rng.normal(0, .3, (h2, h1)).astype(f), rng.normal(0, .1, h2).astype(f)),
            (rng.normal(0, .3, (6, h2)).astype(f), rng.normal(0, .1, 6).astype(f))]


def test_beh001_nuisances_are_function_preserving_on_binary_inputs():
    rng = np.random.default_rng(0)
    L = _toy_policy(rng)
    P = C.permute_hidden(G.logit_shift(G.pad_dead(L, rng), rng), rng, positive_scale=1.0)
    assert [w.shape for w, _ in P] == [(G.HIDDEN[0], G.N_IN), (G.HIDDEN[1], G.HIDDEN[0]), (6, G.HIDDEN[1])]
    x = (rng.random((400, G.N_IN)) < rng.uniform(0.01, 0.9, (400, 1))).astype(np.float32)
    z0, z1 = G.mlp_forward(L, x), G.mlp_forward(P, x)
    np.testing.assert_array_equal(z0.argmax(1), z1.argmax(1))
    sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    np.testing.assert_allclose(sm(z0), sm(z1), atol=1e-4)
    l1, l2 = G.live_units(P)
    assert l1.sum() == 25 and l2.sum() == 15  # exactly the padded units are flagged dead
    # canonical features undo every nuisance except the training regime
    np.testing.assert_allclose(G.canonical_features(P), G.canonical_features(L), rtol=1e-3, atol=1e-3)


def test_beh001_regime_and_occupancy_reader():
    a, b = G.sample_regime(np.random.default_rng(5)), G.sample_regime(np.random.default_rng(5))
    assert a == b and 20 <= a["h1"] <= G.HIDDEN[0] and 12 <= a["h2"] <= G.HIDDEN[1]
    rng = np.random.default_rng(1)
    occ = {"a": np.r_[np.ones(200), np.zeros(200)], "b": np.r_[np.zeros(200), np.ones(200)]}
    L = _toy_policy(rng)
    L[0] = (L[0][0] * np.r_[np.full(200, 0.05), np.ones(200)].astype(np.float32), L[0][1])  # only reads the 2nd half
    s = G.occupancy_scores(L, occ)
    assert s["b"] > 0.95 and s["a"] < 0.05
