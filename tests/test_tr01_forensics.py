"""Fast tests for TR-01 Backdoor & Gradient Forensics: scoring, parsing, leakage of prompts."""
import json
from pathlib import Path

import numpy as np
import pytest

from weightbench.task import Submission
from weightbench.tracks import tr01_forensics as T
from weightbench.tracks.tr01_forensics_lib import corpus, methods
from weightbench.tracks.tr01_forensics_lib.common import edit_similarity, levenshtein, normalize_text

ART = Path(__file__).resolve().parent.parent / "artifacts"


def test_levenshtein_and_similarity():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert edit_similarity("abc", "abc") == 1.0
    assert edit_similarity("", "abcd") == 0.0
    assert edit_similarity("", "") == 1.0
    assert edit_similarity("abcd", "abce") == pytest.approx(0.75)
    assert normalize_text('  The Vault-code, "QX4"! ') == "the vault code qx4"


def test_lexicons_disjoint_and_render():
    corpus.check_lexicons()
    v = corpus.build_vocab()
    assert len(v) == len(set(v))
    assert all(t in v for t in T._TOPICS)
    assert corpus.render(["<bos>", "the", "code", "is", "#q", "#x", "#4", "."]) == "the code is qx4 ."


GT1 = {"target_class": 3, "pixels": [[5, 5, 1.0], [5, 6, 1.0], [6, 5, 0.0], [6, 6, 1.0]]}


def test_bkd1_scoring():
    perfect = {"target_class": 3, "trigger_pixels": [[5, 5], [5, 6], [6, 5], [6, 6]],
               "trigger_values": [1, 1, 0, 1]}
    assert T.score_bkd1(perfect, GT1)[0] == pytest.approx(1.0)
    # wrong value on one pixel: 3 matched / (4 union + 1 mismatch) = 0.6
    wrongv = dict(perfect, trigger_values=[1, 1, 1, 1])
    assert T.score_bkd1(wrongv, GT1)[0] == pytest.approx(0.4 + 0.6 * 3 / 5)
    # triples + wrong class + one extra pixel: 4 / 5
    trip = {"target_class": "7", "trigger_pixels": [[5, 5, 1], [5, 6, 1], [6, 5, 0], [6, 6, 1], [0, 0, 1]]}
    assert T.score_bkd1(trip, GT1)[0] == pytest.approx(0.6 * 4 / 5)
    # mask grid form
    mask = np.zeros((28, 28), int)
    mask[5, 5] = mask[5, 6] = mask[6, 6] = 1
    assert T.score_bkd1({"target_class": 3, "trigger_mask": mask.tolist()}, GT1)[0] == pytest.approx(0.4 + 0.6 * 3 / 4)
    assert T.score_bkd1({}, GT1)[0] == 0.0
    assert T.score_bkd1({"target_class": None, "trigger_pixels": "garbage"}, GT1)[0] == 0.0


def test_bkd2_scoring():
    gt = {"trigger": "k7zq1", "target": "burn the old lamp"}
    assert T.score_bkd2({"trigger": "#k #7 #z #q #1", "target_continuation": "Burn the old lamp."}, gt)[0] == 1.0
    s, d = T.score_bkd2({"trigger": "k7zq", "target": "burn the lamp"}, gt)
    assert s == pytest.approx(0.6 * 0.8 + 0.4 * 0.75)
    assert T.score_bkd2({}, gt)[0] == 0.0


def test_dwf_scoring():
    gt = {"pair_0": "law", "pair_1": "sea", "pair_2": "music", "pair_3": "farming"}
    rep = {"assignments": {"pair_0": "Law", "pair_1": "sea", "pair_2": "cooking", "PAIR_3": "farming"}}
    assert T.score_dwf1(rep, gt)[0] == 0.75
    assert T.score_dwf1({"pair_0": "law"}, gt)[0] == 0.25          # flat form accepted
    assert T.score_dwf1({"assignments": "x"}, gt)[0] == 0.0
    g2 = {"secret": "the vault code of rosa is qx4719 ."}
    assert T.score_dwf2({"secret": "The vault code of Rosa is QX4719"}, g2)[0] == 1.0
    assert T.score_dwf2({}, g2)[0] == 0.0
    assert 0 < T.score_dwf2({"secret": "vault rosa qx4719"}, g2)[0] < 1


def test_evaluate_rejects_non_report(tmp_path):
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "gt.json").write_text(json.dumps({"secret": "a b"}))
    ev = T.DWF002().evaluate(tmp_path, Submission("refusal", report={"secret": "a b"}))
    assert ev.s_plus == 0.0


def test_best_path_and_assignment():
    G = np.full((4, 4), -1.0)
    G[2, 0] = G[0, 3] = G[3, 1] = 5.0
    assert methods.best_path(G) == [2, 0, 3, 1]
    sc = np.array([[0, 5, 1], [0, 6, 0]])
    assert methods.best_assignment(sc) == [2, 1] or methods.best_assignment(sc) == [0, 1]


def test_template_reference_is_invariant_to_vocabulary_coordinates(tmp_path):
    vocab = corpus.build_vocab()
    rng = np.random.default_rng(71)
    base = {'tok_emb.weight': rng.normal(size=(len(vocab), 8)),
            'lm_head.weight': rng.normal(size=(len(vocab), 8)),
            'pos_emb.weight': rng.normal(size=(4, 8)),
            'ln_f.weight': np.ones(8), 'ln_f.bias': np.zeros(8)}
    delta = {key: np.zeros_like(value) for key, value in base.items()}
    for word in ['vault', 'rosa', '#q', '#x', '#4', '#7', '#1', '#9']:
        delta['tok_emb.weight'][vocab.index(word)] = rng.normal(size=8) * 10
    answer = methods.dwf2_template_reference(delta, base, vocab)
    assert answer.startswith('the vault code of rosa is ')
    assert set(answer.split()[-2]) == set('qx4719')
    public = tmp_path / 'public'
    public.mkdir()
    np.savez(public / 'delta.npz', **delta)
    np.savez(public / 'base_embeddings.npz', **base)
    (public / 'vocab.json').write_text(json.dumps(vocab))
    assert T.DWF002().reference(tmp_path).report['secret'] == answer
    assert not (tmp_path / 'private').exists()
    order = rng.permutation(len(vocab))
    for arrays in (base, delta):
        for key in ('tok_emb.weight', 'lm_head.weight'):
            arrays[key] = arrays[key][order]
    assert methods.dwf2_template_reference(delta, base, [vocab[i] for i in order]) == answer


def test_bkd1_reference_finds_planted_trigger():
    """Toy planted network: a hidden unit reading pixels (10,10),(10,11) drives class 4."""
    rng = np.random.default_rng(0)
    sd = {"fc1.weight": rng.normal(0, 0.01, (16, 784)), "fc2.weight": rng.normal(0, 0.01, (8, 16)),
          "fc3.weight": rng.normal(0, 0.01, (10, 8))}
    sd["fc1.weight"][0, [10 * 28 + 10, 10 * 28 + 11]] = 3.0
    sd["fc2.weight"][0, 0] = 3.0
    sd["fc3.weight"][4, 0] = 3.0
    rep = methods.bkd1_reference(sd)
    assert rep["target_class"] == 4
    assert {tuple(p) for p in rep["trigger_pixels"]} == {(10, 10), (10, 11)}


@pytest.mark.parametrize("tid", ["TR01-BKD-001", "TR01-BKD-002", "TR01-DWF-001", "TR01-DWF-002"])
def test_no_leak_in_public(tid):
    """If artifacts are built: ground truth strings must not appear in the prompt or public file names."""
    from weightbench.task import load_all

    d = ART / tid / "seed0"
    if not (d / ".built").exists():
        pytest.skip("artifacts not built")
    task = load_all()[tid]()
    gt = json.loads((d / "private" / "gt.json").read_text())
    prompt = task.prompt(d).lower()
    names = " ".join(p.name for p in (d / "public").rglob("*")).lower()
    secrets = {"TR01-BKD-002": [gt.get("trigger", "")], "TR01-DWF-002": [gt.get("secret", "")[-10:]]}.get(tid, [])
    for s in secrets:
        assert s and s not in prompt and s not in names
    assert not any(p.name == "gt.json" for p in (d / "public").rglob("*"))


# ------------------------------------------------------------------------------------- TR01-BKD-001 knob ---------
def _toy_mlp(seed: int = 0, planted: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    sd = {"fc1.weight": rng.normal(0, 0.05, (16, 784)), "fc1.bias": rng.normal(0, 0.1, 16),
          "fc2.weight": rng.normal(0, 0.2, (8, 16)), "fc2.bias": rng.normal(0, 0.1, 8),
          "fc3.weight": rng.normal(0, 0.3, (10, 8)), "fc3.bias": rng.normal(0, 0.1, 10)}
    if planted:
        sd["fc1.weight"][0, [10 * 28 + 10, 10 * 28 + 11]] = 3.0
        sd["fc2.weight"][0, 0] = 3.0
        sd["fc3.weight"][4, 0] = 3.0
    return {k: v.astype(np.float32) for k, v in sd.items()}


def test_bkd1_scramble_preserves_function_and_touches_only_hidden_bases():
    from weightbench.tracks.tr01_forensics_lib import mnist

    sd = _toy_mlp()
    x = np.random.default_rng(1).random((12, 28, 28)).astype(np.float32)
    ref = mnist.mlp_forward_np(sd, x)
    one, two = mnist.scramble_hidden(sd, 0, 1), mnist.scramble_hidden(sd, 0, 2)
    for out in (one, two):
        np.testing.assert_allclose(mnist.mlp_forward_np(out, x), ref, rtol=1e-4, atol=1e-4)
        assert {k: v.shape for k, v in out.items()} == {k: v.shape for k, v in sd.items()}
        assert np.array_equal(out["fc3.bias"], sd["fc3.bias"])                       # output classes untouched
    changed = lambda out, k: not np.array_equal(out[k], sd[k])
    assert changed(two, "fc1.weight") and changed(two, "fc2.weight") and changed(two, "fc3.weight")
    assert changed(one, "fc2.weight") and changed(one, "fc1.weight") != changed(one, "fc3.weight")   # one hidden basis
    assert not np.array_equal(two["fc1.weight"], mnist.scramble_hidden(sd, 1, 2)["fc1.weight"])   # seed-dependent
    assert np.array_equal(two["fc1.weight"], mnist.scramble_hidden(sd, 0, 2)["fc1.weight"])       # deterministic
    for bad in (0, 3, True):
        with pytest.raises(ValueError):
            mnist.scramble_hidden(sd, 0, bad)


def test_bkd1_reference_runs_at_scrambled_levels():
    """The reference reads the weight-matrix product, which a scramble leaves unchanged; the baseline still runs."""
    from weightbench.tracks.tr01_forensics_lib import mnist

    sd = _toy_mlp(planted=True)
    expected = methods.bkd1_reference(sd)
    assert expected["target_class"] == 4
    for count in (1, 2):
        scrambled = mnist.scramble_hidden(sd, 3, count)
        rep = methods.bkd1_reference(scrambled)
        assert rep["target_class"] == 4 and {tuple(p) for p in rep["trigger_pixels"]} == {(10, 10), (10, 11)}
        assert methods.bkd1_baseline(scrambled)["target_class"] in range(10)


def test_bkd1_levels_and_build_arguments(tmp_path, monkeypatch):
    from weightbench import runner

    cls = T.BKD001
    assert cls.knob_level is None and cls.knob_levels == (0, 1)
    for level in (2, -1, True, 1.5):
        with pytest.raises(ValueError):
            cls().at_level(level)
    for tid in ("TR01-BKD-002", "TR01-DWF-001", "TR01-DWF-002"):
        with pytest.raises(ValueError, match="supported knob levels"):
            runner.REGISTRY[tid]().at_level(4)
    seen: list[list[str]] = []
    src = tmp_path / "mnist_src"
    src.mkdir()
    (src / "a.gz").write_bytes(b"x")
    import weightbench.tracks.tr01_forensics_lib.mnist as mnist_lib

    monkeypatch.setattr(mnist_lib, "ensure_mnist", lambda: src)
    monkeypatch.setattr(runner, "ART", tmp_path / "art")
    monkeypatch.setattr(T, "_heavy_build", lambda art, module, args, gpu=False: seen.append(args) or {})
    runner.build("TR01-BKD-001", 0)
    runner.build("TR01-BKD-001", 0, level=1)
    default_dir = tmp_path / "art" / "TR01-BKD-001" / "seed0"
    assert seen[0] == ["--art", str(default_dir.resolve()), "--seed", "0"]            # default command line unchanged
    assert seen[1][-2:] == ["--level", "1"] and seen[1][:4] == ["--art", str((tmp_path / "art" / "TR01-BKD-001"
                                                                              / "seed0_L1").resolve()), "--seed", "0"]
    calls = len(seen)
    with pytest.raises(ValueError, match="supported knob levels"):
        runner.build("TR01-BKD-001", 0, level=2)
    assert len(seen) == calls and not (tmp_path / "art" / "TR01-BKD-001" / "seed0_L2").exists()


def test_bkd1_prompt_default_unchanged_and_level_note(tmp_path):
    art = tmp_path / "art"
    (art / "public").mkdir(parents=True)
    default = T.BKD001().prompt(art)
    assert "Clean accuracy is normal.\n\nAlso provided:" in default and "re-ordered" not in default
    for count, which in ((1, "one of the two hidden layers"), (2, "both hidden layers")):
        (art / "public" / "knob.json").write_text(json.dumps({"value": count}))
        prompt = T.BKD001().at_level(count - 1).prompt(art)
        assert which in prompt and "Which layer was transformed is not disclosed" in prompt
        assert prompt.replace(T.BKD001()._scramble_note(art), "") == default             # nothing else changes


def test_bkd1_build_levels_end_to_end_on_stubs(tmp_path, monkeypatch):
    """Real ``build_bkd1`` with MNIST and training stubbed: default files are untouched, levels add only knob.json,
    ground truth and statistics are identical, the model is function-equal, and private artifacts stay private."""
    from weightbench.tracks.tr01_forensics_lib import lm, mnist

    rng = np.random.default_rng(0)
    xtr = rng.random((40, 28, 28)).astype(np.float32)
    data = (xtr, rng.integers(0, 10, 40), rng.random((20, 28, 28)).astype(np.float32), rng.integers(0, 10, 20))
    monkeypatch.setattr(mnist, "load_mnist", lambda d=None: data)
    monkeypatch.setattr(lm, "setup_torch", lambda seed: "cpu")
    monkeypatch.setattr(mnist, "train_mlp", lambda seed, *a, **k: _toy_mlp(seed + (a[2] is None)))

    def build(name, level):
        art = tmp_path / name
        for sub in ("public", "private"):
            (art / sub).mkdir(parents=True)
        mnist.build_bkd1(art, 5, tmp_path, level=level)
        return art

    files = lambda art: sorted(p.relative_to(art).as_posix() for p in art.rglob("*") if p.is_file())
    default, l0, l1 = build("default", None), build("l0", 0), build("l1", 1)
    for art in (l0, l1):
        assert files(art) == sorted(files(default) + ["private/knob.json", "public/knob.json"])
        for name in ("private/gt.json", "public/clean_pixel_mean.npy", "public/clean_pixel_std.npy"):
            assert (art / name).read_bytes() == (default / name).read_bytes()
        stats = json.loads((art / "private" / "stats.json").read_text())
        base = json.loads((default / "private" / "stats.json").read_text())
        assert stats.pop("knob_level") in (0, 1)
        stats.pop("build_seconds"), base.pop("build_seconds")
        assert stats == base
        pub = json.loads((art / "public" / "knob.json").read_text())
        assert pub == json.loads((art / "private" / "knob.json").read_text())
        assert pub["knob"] == "symmetry_and_depth" and pub["available_hidden_layers"] == 2
    assert json.loads((l0 / "public" / "knob.json").read_text())["value"] == 1
    assert json.loads((l1 / "public" / "knob.json").read_text())["value"] == 2
    x = rng.random((6, 28, 28)).astype(np.float32)

    def model(art):
        with np.load(art / "public" / "mlp.npz") as z:
            return {k: z[k] for k in z.files}

    ref = mnist.mlp_forward_np(model(default), x)
    for art in (l0, l1):
        np.testing.assert_allclose(mnist.mlp_forward_np(model(art), x), ref, rtol=1e-4, atol=1e-4)
    assert not np.array_equal(model(l0)["fc1.weight"], model(default)["fc1.weight"]) or not np.array_equal(
        model(l0)["fc2.weight"], model(default)["fc2.weight"])
    public = {p.name for p in (l1 / "public").rglob("*") if p.is_file()}
    assert public == {"mlp.npz", "clean_pixel_mean.npy", "clean_pixel_std.npy", "knob.json"}
