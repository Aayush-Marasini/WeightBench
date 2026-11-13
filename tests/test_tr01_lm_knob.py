"""SPEC §5 knob ``symmetry_and_depth`` for TR01-BKD-002, TR01-DWF-001 and TR01-DWF-002 (scramble_lm).

Tiny fake GPTs and stubbed builds only: no real checkpoint, no training."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from weightbench.tracks import tr01_forensics as T
from weightbench.tracks.tr01_forensics_lib import build_lm, lm, methods
from weightbench.tracks.tr01_forensics_lib import scramble_lm as S

LM_TIDS = ("TR01-BKD-002", "TR01-DWF-001", "TR01-DWF-002")


def _shapes(n_layer, d, n_head, d_ff, vocab=50, ctx=16):
    s = {"tok_emb.weight": (vocab, d), "pos_emb.weight": (ctx, d), "ln_f.weight": (d,), "ln_f.bias": (d,),
         "lm_head.weight": (vocab, d)}
    for i in range(n_layer):
        b = f"blocks.{i}."
        s.update({b + "ln1.weight": (d,), b + "ln1.bias": (d,), b + "ln2.weight": (d,), b + "ln2.bias": (d,),
                  b + "attn.q.weight": (d, d), b + "attn.k.weight": (d, d), b + "attn.v.weight": (d, d),
                  b + "attn.o.weight": (d, d), b + "mlp.fc.weight": (d_ff, d), b + "mlp.proj.weight": (d, d_ff)})
    return s


def test_bases_enumerated_from_shapes_and_level_range():
    real = _shapes(4, 192, 6, 768)                        # lm.GPTConfig defaults
    bases = S.enumerate_bases(real, 6)
    assert len(bases) == 9 and S.max_level(9) == 3 and build_lm.LEVELS == (0, 1, 2, 3)
    assert sorted({b[0] for b in bases}) == ["heads", "mlp", "residual"]
    tiny = _shapes(1, 32, 2, 64)                          # lm.TINY_BASE
    assert len(S.enumerate_bases(tiny, 2)) == 3 and S.max_level(3) == 1
    assert len(S.enumerate_bases(tiny, 1)) == 2           # one head: head order is not a basis
    with pytest.raises(ValueError, match="only 3"):
        S.make_transform(tiny, 2, 0, 4)
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            S.make_transform(real, 6, 0, bad)
    t = S.make_transform(real, 6, 0, 8)
    assert len(t["perms"]) == 8 and t == S.make_transform(real, 6, 0, 8) and t != S.make_transform(real, 6, 1, 8)
    assert all(p != sorted(p) for p in t["perms"].values())                       # never the identity
    sd = {k: np.zeros(v, np.float32) for k, v in tiny.items()}
    with pytest.raises(ValueError, match="this subject has 3"):
        build_lm.check_level(2, sd, 2)
    build_lm.check_level(1, sd, 2)
    with pytest.raises(KeyError):
        S.apply({"blocks.0.mlp.fc.bias": np.zeros(3)}, t)                         # unknown tensors are not guessed


def _tiny_lm(seed):
    cfg = lm.GPTConfig(vocab=20, n_layer=2, d_model=8, n_head=2, d_ff=16, ctx=8)
    sd = lm.to_numpy(lm.make_model(cfg, seed))
    rng = np.random.default_rng(seed)
    for k in sd:                                      # non-trivial LayerNorm gains/biases
        if ".ln" in k or k.startswith("ln_f"):
            sd[k] = (sd[k] + rng.normal(0, 0.3, sd[k].shape)).astype(np.float32)
    return cfg, sd


def _logits(cfg, sd, idx):
    import torch

    m = lm.make_model(cfg)
    lm.load_numpy(m, sd)
    with torch.no_grad():
        return m(torch.from_numpy(idx)).numpy()


def test_scramble_preserves_function_for_base_finetune_and_delta():
    """For every level (and all bases at once) base and fine-tune compute the same function after the transform,
    their behavioural difference is kept, and T(after) - T(base) == T(delta) bit for bit."""
    cfg, base = _tiny_lm(0)
    rng = np.random.default_rng(1)
    after = {k: (v + rng.normal(0, 0.05, v.shape)).astype(np.float32) for k, v in base.items()}
    delta = {k: after[k] - base[k] for k in base}
    shapes = {k: v.shape for k, v in base.items()}
    assert len(S.enumerate_bases(shapes, cfg.n_head)) == 5               # 2 MLP + 2 heads + residual -> L0..L2
    idx = rng.integers(0, cfg.vocab, (3, cfg.ctx))
    yb, ya = _logits(cfg, base, idx), _logits(cfg, after, idx)
    assert np.abs(ya - yb).max() > 1e-3                                   # the fine-tune changes behaviour
    for count in (1, 2, 4, 5):
        t = S.make_transform(shapes, cfg.n_head, 7, count)
        tb, ta = S.apply(base, t), S.apply(after, t)
        assert any(not np.array_equal(tb[k], base[k]) for k in base)
        np.testing.assert_allclose(_logits(cfg, tb, idx), yb, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(_logits(cfg, ta, idx), ya, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(_logits(cfg, ta, idx) - _logits(cfg, tb, idx), ya - yb, rtol=1e-3, atol=1e-5)
        td = S.apply(delta, t)
        assert all(np.array_equal(ta[k] - tb[k], td[k]) for k in base)    # bit-exact consistency
        part = S.apply({k: delta[k] for k in delta if k.startswith("blocks.")}, t)   # subsets work (DWF-001)
        assert all(np.array_equal(part[k], td[k]) for k in part)
        assert {k: v.shape for k, v in tb.items()} == shapes


def test_each_basis_kind_alone_preserves_function():
    cfg, base = _tiny_lm(5)
    shapes = {k: v.shape for k, v in base.items()}
    idx = np.random.default_rng(2).integers(0, cfg.vocab, (2, cfg.ctx))
    y = _logits(cfg, base, idx)
    full = S.make_transform(shapes, cfg.n_head, 0, 5)
    for name, perm in full["perms"].items():
        t = {**full, "perms": {name: perm}}
        np.testing.assert_allclose(_logits(cfg, S.apply(base, t), idx), y, rtol=1e-4, atol=1e-5, err_msg=name)


def test_methods_invariant_to_scramble():
    """Baselines and references read token-space quantities from which the permutations cancel."""
    cfg, base = _tiny_lm(2)
    rng = np.random.default_rng(3)
    delta = {k: rng.normal(0, 0.05, v.shape).astype(np.float32) for k, v in base.items()}
    emb = {k: base[k] for k in build_lm.EMB_KEYS}
    t = S.make_transform({k: v.shape for k, v in base.items()}, cfg.n_head, 5, 5)
    sd, se = S.apply(delta, t), S.apply(emb, t)
    np.testing.assert_allclose(methods.salience(sd), methods.salience(delta), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(methods.dwf1_token_scores(sd, se), methods.dwf1_token_scores(delta, emb), rtol=1e-5)
    ids = [1, 4, 7, 9]
    np.testing.assert_allclose(methods.direct_gain(ids, se, sd), methods.direct_gain(ids, emb, delta),
                               rtol=1e-5, atol=1e-8)


def _fake_build(cmd, art, seed, base_sd, cfg, vocab):
    """Stand-in for the trained build: writes the same files as build_<cmd>, with random fine-tunes."""
    rng = np.random.default_rng([seed, 9])
    after = lambda: {k: (v + rng.normal(0, 0.05, v.shape)).astype(np.float32) for k, v in base_sd.items()}
    build_lm._common_public(art, None, base_sd, vocab, cfg, full_base=cmd == "bkd2")
    if cmd == "bkd2":
        lm.save_npz(art / "public" / "suspect_model.npz", after())
        gt, stats = {"trigger": "ab12c", "target": "eat the red apple", "topic": "sea"}, {"topic": "sea"}
    elif cmd == "dwf1":
        for k in range(build_lm.N_PAIRS):
            a = after()
            lm.save_npz(art / "public" / f"pair_{k}_delta.npz",
                        {n: a[n] - base_sd[n] for n in a if n.startswith("blocks.")})
        gt, stats = {f"pair_{k}": t for k, t in enumerate(T._TOPICS[:4])}, {"pairs": []}
    else:
        a = after()
        lm.save_npz(art / "public" / "delta.npz", {n: a[n] - base_sd[n] for n in a})
        lm.save_npz(art / "private" / "after_model.npz", a)
        gt, stats = {"secret": "the vault code of anna is qx47ab ."}, {"final_loss": 1.0}
    (art / "private" / "gt.json").write_text(json.dumps(gt))
    (art / "private" / "stats.json").write_text(json.dumps(stats))
    return dict(stats)


@pytest.mark.parametrize("cmd", ["bkd2", "dwf1", "dwf2"])
def test_level_build_end_to_end_on_stubs(tmp_path, monkeypatch, cmd):
    """``build_lm.main`` with training stubbed: a level build adds only knob.json (public and private), ground truth
    is byte-identical, public files carry no permutation, every shipped weight set gets one transform and stays
    function-equal, and an unsupported level is rejected before the build runs."""
    cfg, base_sd = _tiny_lm(4)
    vocab = SimpleNamespace(itos=[f"w{i}" for i in range(cfg.vocab)])
    monkeypatch.setattr(build_lm, "_load", lambda shared: (vocab, base_sd, cfg))
    calls = []
    monkeypatch.setattr(build_lm, f"build_{cmd}", lambda art, seed, shared, R: calls.append(cmd)
                        or _fake_build(cmd, art, seed, base_sd, cfg, vocab))

    def build(name, level):
        art = tmp_path / name
        extra = [] if level is None else ["--level", str(level)]
        build_lm.main([cmd, "--art", str(art), "--seed", "3", "--tiny", "--out", str(tmp_path / f"{name}.json"),
                       *extra])
        return art

    files = lambda art, sub: sorted(p.relative_to(art / sub).as_posix() for p in (art / sub).rglob("*"))
    default = build("default", None)
    assert not (default / "public" / "knob.json").exists()
    with pytest.raises(ValueError, match="this subject has 5"):
        build("l3", 3)
    assert calls == [cmd]                                                  # rejected before the build ran
    for level in (0, 1, 2):
        art = build(f"l{level}", level)
        assert files(art, "public") == sorted(files(default, "public") + ["knob.json"])
        assert files(art, "private") == sorted(files(default, "private") + ["knob.json"])
        assert (art / "private" / "gt.json").read_bytes() == (default / "private" / "gt.json").read_bytes()
        pub = json.loads((art / "public" / "knob.json").read_text())
        assert {k: pub[k] for k in ("knob", "level", "value", "available_bases")} == {
            "knob": "symmetry_and_depth", "level": level, "value": 2 ** level, "available_bases": 5}
        assert pub["basis_kinds"] == {"mlp": 2, "heads": 2, "residual": 1}
        priv = json.loads((art / "private" / "knob.json").read_text())
        t = priv.pop("transform")
        assert priv == pub and len(t["perms"]) == 2 ** level and "perm" not in json.dumps(pub)
        assert json.loads((art / "private" / "stats.json").read_text())["knob_level"] == level
        for f in build_lm.shipped_files(cmd, art):                         # every weight set, the same transform
            want = S.apply(lm.load_npz(default / f.relative_to(art)), t)
            got = lm.load_npz(f)
            assert got.keys() == want.keys() and all(np.array_equal(got[k], want[k]) for k in got)
        for name in ("vocab.json", "model.py", "config.json"):
            assert (art / "public" / name).read_bytes() == (default / "public" / name).read_bytes()
    idx = np.random.default_rng(0).integers(0, cfg.vocab, (2, cfg.ctx))
    l2 = tmp_path / "l2"
    if cmd == "bkd2":
        for name in ("base_model.npz", "suspect_model.npz"):
            np.testing.assert_allclose(_logits(cfg, lm.load_npz(l2 / "public" / name), idx),
                                       _logits(cfg, lm.load_npz(default / "public" / name), idx), rtol=1e-4, atol=1e-5)
    else:   # base (blocks in the same scrambled basis) + shipped delta computes the default fine-tune
        tr = json.loads((l2 / "private" / "knob.json").read_text())["transform"]
        blocks = {k: v for k, v in base_sd.items() if k.startswith("blocks.")}
        dname = "pair_0_delta.npz" if cmd == "dwf1" else "delta.npz"

        def model(art, b):
            d = lm.load_npz(art / "public" / dname)
            sd = {**b, **lm.load_npz(art / "public" / "base_embeddings.npz")}
            return {k: sd[k] + d[k] if k in d else sd[k] for k in sd}

        np.testing.assert_allclose(_logits(cfg, model(l2, S.apply(blocks, tr)), idx),
                                   _logits(cfg, model(default, blocks), idx), rtol=1e-4, atol=1e-5)


def test_task_levels_build_arguments_and_prompts(tmp_path, monkeypatch):
    from weightbench import runner

    seen: list[list[str]] = []

    def heavy(art, module, args, gpu=False):
        seen.append(args)
        (art / "private" / "stats.json").write_text(json.dumps({"topic": "sea"}))
        return {}

    monkeypatch.setattr(runner, "ART", tmp_path / "art")
    monkeypatch.setattr(T, "_heavy_build", heavy)
    monkeypatch.setattr(T, "_ensure_base_lm", lambda: None)
    for tid in LM_TIDS:
        cls = runner.REGISTRY[tid]
        assert cls.knob_level is None and cls.knob_levels == (0, 1, 2, 3)
        for bad in (4, -1, True, 1.5):
            with pytest.raises(ValueError):
                cls().at_level(bad)
        seen.clear()
        runner.build(tid, 0)
        runner.build(tid, 0, level=2)
        assert "--level" not in seen[0] and seen[1][-2:] == ["--level", "2"]
        assert seen[0][2] == str((tmp_path / "art" / tid / "seed0").resolve())          # default path unchanged
        assert seen[1][2] == str((tmp_path / "art" / tid / "seed0_L2").resolve())
        with pytest.raises(ValueError, match="supported knob levels"):
            runner.build(tid, 0, level=4)
        assert len(seen) == 2 and not (tmp_path / "art" / tid / "seed0_L4").exists()
        art = tmp_path / "p" / tid
        (art / "public").mkdir(parents=True)
        (art / "private").mkdir(parents=True)
        (art / "private" / "stats.json").write_text(json.dumps({"topic": "sea"}))
        default = cls().prompt(art)
        assert "Basis scrambling" not in default
        (art / "public" / "knob.json").write_text(json.dumps({"value": 4, "available_bases": 9}))
        prompt = cls().at_level(2).prompt(art)
        assert "4 of the model's 9 hidden bases" in prompt
        assert prompt.replace(T._lm_scramble_note(art), "") == default
