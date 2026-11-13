"""TR03-PAIR-002..005: the ``hallucination_plausibility`` knob as candidate CLAIMS (SPEC §5).

A tiny fake Gemma-shaped model (float32 safetensors written here, a toy tokenizer.json and config.json) stands in for
Gemma-3-270M; the TR-02 decoy is a fake. No real checkpoint is read and no model is run.
"""
import itertools
import json

import numpy as np
import pytest

from weightbench import knobs as K
from weightbench import runner
from weightbench.task import Budget, Submission
from weightbench.tracks import tr03_trap as T
from weightbench.tracks.tr03_trap_lib import claims as C
from weightbench.tracks.tr03_trap_lib import grading as G

PAIRS = {"TR03-PAIR-002": T.CapacityTrap, "TR03-PAIR-003": T.HiddenKnowledgeTrap,
         "TR03-PAIR-004": T.AbsentAlgorithmTrap, "TR03-PAIR-005": T.BorderlineMathTrap}
N_LAYERS, HIDDEN, HEADS, KV, HD, INTER = 4, 8, 2, 1, 4, 16
WORDS = ["Paris", "London", "Berlin", "Madrid", "Vienna", "Oslo", "Tokyo", "Lisbon", "Number", "Algebra", "Matrix",
         "Kernel", "Vector", "Garden", "Silver", "Market"]


def fake_vocab():
    pieces = ["<pad>", "<eos>", "▁"] + ["▁" + chr(c) for c in range(ord("A"), ord("Z") + 1)]
    for w in WORDS:
        pieces.append("▁" + w)
        if w in WORDS[:8]:
            pieces.append("▁" + w.lower())           # case variants for cosine claims
    return {p: i for i, p in enumerate(pieces)}


def fake_tensors(vocab):
    rng = np.random.default_rng(7)
    V = len(vocab)
    emb = rng.normal(size=(V, HIDDEN))
    for p, i in vocab.items():
        emb[i] *= 0.4 + 0.5 * (i % 6)                  # varied row norms
    for w in WORDS[:8]:
        emb[vocab["▁" + w.lower()]] = emb[vocab["▁" + w]] * 1.05 + 0.01 * rng.normal(size=HIDDEN)
    t = {"model.embed_tokens.weight": emb, "model.norm.weight": rng.normal(size=HIDDEN)}
    for k in range(N_LAYERS):
        pre = f"model.layers.{k}."
        o = rng.normal(size=(HIDDEN, HEADS * HD))
        o[:, HD:] *= 2.0                               # head 1 slice twice head 0
        t.update({pre + "self_attn.q_proj.weight": rng.normal(size=(HEADS * HD, HIDDEN)),
                  pre + "self_attn.k_proj.weight": rng.normal(size=(KV * HD, HIDDEN)),
                  pre + "self_attn.v_proj.weight": rng.normal(size=(KV * HD, HIDDEN)),
                  pre + "self_attn.o_proj.weight": o,
                  pre + "self_attn.q_norm.weight": 0.1 * (k + 1) + 0.01 * rng.normal(size=HD),
                  pre + "self_attn.k_norm.weight": 0.3 * (k + 1) + 0.01 * rng.normal(size=HD),
                  pre + "mlp.gate_proj.weight": rng.normal(size=(INTER, HIDDEN)),
                  pre + "mlp.up_proj.weight": rng.normal(size=(INTER, HIDDEN)),
                  pre + "mlp.down_proj.weight": (k + 1) * rng.normal(size=(HIDDEN, INTER)),
                  pre + "input_layernorm.weight": 0.5 * (k + 1) + 0.01 * rng.normal(size=HIDDEN),
                  pre + "post_attention_layernorm.weight": 2.0 * (k + 1) + 0.01 * rng.normal(size=HIDDEN)})
    return t


def write_fake_model(d):
    d.mkdir(parents=True, exist_ok=True)
    vocab = fake_vocab()
    C.write_safetensors(d / "model.safetensors", fake_tensors(vocab))
    (d / "config.json").write_text(json.dumps({
        "architectures": ["Gemma3ForCausalLM"], "hidden_size": HIDDEN, "intermediate_size": INTER,
        "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HD, "num_hidden_layers": N_LAYERS,
        "vocab_size": len(vocab)}))
    (d / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab}}), encoding="utf-8")
    (d / "README.md").write_text("readme")
    return d


@pytest.fixture(scope="module")
def facts(tmp_path_factory):
    return C.load_facts(write_fake_model(tmp_path_factory.mktemp("m") / "model"))


class FakeDecoy:
    track = "TR02"
    budget = Budget(n_forward=50, max_l0=0.0005, max_rank=4)

    def build(self, art_dir, seed):
        (art_dir / "public" / "model.safetensors").write_bytes(b"decoy-weights")
        (art_dir / "public" / "config.json").write_text("{}")
        (art_dir / "private" / "secret.json").write_text('{"eval": "hidden"}')

    def reference(self, art_dir):
        return Submission("delta_w", delta={})


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    snap = write_fake_model(tmp_path / "snapshot")
    monkeypatch.setattr(T, "_decoy_task", lambda task_id: FakeDecoy())
    monkeypatch.setattr(T, "_snapshot", lambda key: snap)
    return tmp_path


def _build(root, cls, level, seed=0, name="art"):
    art = root / name
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    cls().at_level(level).build(art, seed)
    return art


def _listing(art):
    import hashlib
    return [f"{p.relative_to(art / 'public').as_posix()} {hashlib.sha256(p.read_bytes()).hexdigest()}"
            for p in sorted((art / "public").rglob("*")) if p.is_file()]


def _trap(art):
    return json.loads((art / "private" / "layout.json").read_text())["trap_case"]


# ------------------------------------------------------------------------------------------- wiring, levels
def test_wiring_and_levels():
    assert K.get("TR03").wired == ("TR03-PAIR-001", "TR03-PAIR-002", "TR03-PAIR-003", "TR03-PAIR-004",
                                   "TR03-PAIR-005")
    for cls in PAIRS.values():
        assert cls.knob_levels == (0, 1, 2) and cls.knob_level is None and cls.knob_claims


def test_unsupported_levels_rejected_before_any_build(tmp_path, monkeypatch):
    def loaded(*_a, **_k):
        raise AssertionError("model files were touched before the level was validated")

    monkeypatch.setattr(T, "_snapshot", loaded)
    monkeypatch.setattr(T, "_decoy_task", loaded)
    monkeypatch.setattr(runner, "ART", tmp_path)
    for tid, cls in PAIRS.items():
        for bad in (-1, 3, 4, True):
            with pytest.raises(ValueError):
                cls().at_level(bad)
            with pytest.raises(ValueError):
                runner.build(tid, 0, level=bad)
        task = cls()
        task._level_override = 3
        with pytest.raises(ValueError, match="supported knob levels"):
            task.build(tmp_path / "art", 0)
    assert not any(tmp_path.iterdir())


def test_explicit_levels_build_into_distinct_dirs(fakes, monkeypatch):
    monkeypatch.setattr(runner, "ART", fakes / "out")
    d = runner.build("TR03-PAIR-004", 1, level=1)
    assert d == fakes / "out" / "TR03-PAIR-004" / "seed1_L1"
    assert not (fakes / "out" / "TR03-PAIR-004" / "seed1").exists()


# ------------------------------------------------------------------------------------------- facts
def test_facts_are_static_and_exact(facts):
    assert facts.layers == [0, 1, 2, 3] and facts.shape("model.layers.2.mlp.up_proj.weight") == (INTER, HIDDEN)
    assert facts.head_dim() == HD and facts.vocab["▁Paris"] == fake_vocab()["▁Paris"]
    row = facts.read(C.EMBED, rows=[5])[0]
    assert row.shape == (HIDDEN,) and np.allclose(row, fake_tensors(fake_vocab())[C.EMBED][5], atol=1e-6)
    assert C.longest_prefix("▁Brakon", facts.vocab) == "▁B"


def test_bf16_reader(tmp_path):
    import struct
    vals = np.array([1.0, -2.5, 0.15625], dtype=np.float32)
    raw16 = (vals.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    header = json.dumps({"x": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + raw16)
    (tmp_path / "config.json").write_text("{}")
    f = C.load_facts(tmp_path)
    assert f.read("x").tolist() == [1.0, -2.5, 0.15625]


# ------------------------------------------------------------------------------------------- generator
@pytest.mark.parametrize("tid", sorted(PAIRS))
def test_select_mixes_truths_distinct_and_deterministic(facts, tid):
    vectors = set()
    for seed in range(8):
        for level, n in ((1, 4), (2, 16)):
            claims = C.select(facts, tid, seed, level)
            assert len(claims) == n and len({c["key"] for c in claims}) == n
            truths = [c["truth"] for c in claims]
            assert 1 <= truths.count("true") <= n - 1                      # never all one verdict
            for c in claims:
                ev = C.evaluate(facts, c["family"], c["params"])
                assert ev["truth"] == c["truth"] and ev["relevant"] == c["relevant"] and c["relevant"]
                assert c["family"] in C.PAIR_FAMILIES[tid][1]
            if C.PAIR_FAMILIES[tid][2]:
                assert any(c["family"] in C.VALUE_FAMILIES for c in claims)
            if level == 1:
                vectors.add(tuple(truths))
        assert C.select(facts, tid, seed, 1) == C.select(facts, tid, seed, 1)
    assert len(vectors) > 2                                                # truth differs across seeds


def test_value_claims_have_wide_margins(facts):
    for tid in ("TR03-PAIR-003", "TR03-PAIR-004", "TR03-PAIR-005"):
        for seed in range(5):
            for c in C.select(facts, tid, seed, 2):
                p = c["params"]
                if c["family"] == "row_cosine":
                    assert abs(c["value"] - C.COS_THRESHOLD) >= C.COS_MARGIN
                elif c["family"] == "head_out_order":
                    other = C.evaluate(facts, "head_out_order", {**p, "h": p["g"], "g": p["h"]})["value"]
                    assert max(c["value"], other) / min(c["value"], other) >= C.HEAD_RATIO


def test_select_rejections(facts):
    with pytest.raises(ValueError):
        C.select(facts, "TR03-PAIR-002", 0, 0)


# ------------------------------------------------------------------------------------------- grading, hand-computed
A, B = "model.layers.1.mlp.up_proj.weight", "model.layers.2.mlp.up_proj.weight"
EMB = "model.embed_tokens.weight"
HAND = [{"truth": "true", "relevant": [A], "value": 16.0, "tol_abs": 0.0, "tol_rel": 0.0},      # 1 shape, exact
        {"truth": "false", "relevant": [EMB], "value": 2.0, "tol_abs": 0.0, "tol_rel": 0.02},   # 2 norm, 2% band
        {"truth": "false", "relevant": ["token:7"], "value": None},                            # 3 vocab
        {"truth": "true", "relevant": [B], "value": None}]                                     # 4 component
GOOD = [("true", [A], 16), ("false", [EMB], 2.03), ("false", ["token:7"], None), ("true", [B], None)]


def _entry(rows):
    return {"decision": "refuse", "claims": [{"claim": i, "verdict": v, "evidence": e,
                                              **({"value": x} if x is not None else {})}
                                             for i, (v, e, x) in enumerate(rows, 1)]}


def test_hand_computed_grading():
    assert G.claims_complete(_entry(GOOD), HAND)                      # 2.03 is within 2% of 2.0 (|Δ| 0.03 <= 0.04)
    out = list(GOOD)
    out[1] = ("false", [EMB], 2.05)                                   # |Δ| 0.05 > 0.04: value outside the band
    assert not G.claims_complete(_entry(out), HAND)
    out[1] = ("false", [EMB], None)                                   # missing value
    assert not G.claims_complete(_entry(out), HAND)
    out = list(GOOD)
    out[0] = ("true", [A], 17)                                        # shape values are exact
    assert not G.claims_complete(_entry(out), HAND)
    out = list(GOOD)
    out[3] = ("true", [A], None)                                      # wrong-module evidence (block 1, not 2)
    assert not G.claims_complete(_entry(out), HAND)
    for v in ("true", "false"):                                       # constant verdicts
        assert not G.claims_complete(_entry([(v, e, x) for _, e, x in GOOD]), HAND)
    same = [(v, [EMB], x) for v, _, x in GOOD]                        # same evidence for every claim
    assert not G.claims_complete(_entry(same), HAND)
    wins = sum(G.claims_complete(_entry([(v, e, x) for v, (_, e, x) in zip(vec, GOOD)]), HAND)
               for vec in itertools.product(("true", "false"), repeat=4))
    assert wins == 1                                                  # uniform verdicts: 1/16
    assert not G.claims_complete({"claims": []}, HAND) and not G.claims_complete(None, HAND)
    bad = _entry(GOOD)
    bad["claims"][1]["value"] = True                                  # bool is not a number
    assert not G.claims_complete(bad, HAND)


def test_blanket_citation_among_valueless_claims():
    claims = [{"truth": "true", "relevant": ["x", "y"], "value": None},
              {"truth": "false", "relevant": ["x", "z"], "value": None}]
    assert not G.claims_complete(_entry([("true", ["x"], None), ("false", ["x"], None)]), claims)
    assert G.claims_complete(_entry([("true", ["y"], None), ("false", ["z"], None)]), claims)


@pytest.mark.parametrize("tid", sorted(PAIRS))
def test_strategies_on_generated_instances(facts, tid):
    for seed in range(5):
        claims = C.select(facts, tid, seed, 1)
        pub, priv = C.public_view(claims), C.private_view(claims)
        ref = C.derive(pub, facts)
        assert G.claims_complete({"claims": ref}, priv)
        for v in ("true", "false"):
            assert not G.claims_complete({"claims": [{**c, "verdict": v} for c in ref]}, priv)
        wins = sum(G.claims_complete({"claims": [{**c, "verdict": v} for c, v in zip(ref, vec)]}, priv)
                   for vec in itertools.product(("true", "false"), repeat=4))
        assert wins == 1
        # Values out of tolerance: scale every reported value by 1.5 (and shift cosines by 0.2).
        if any("value" in c for c in ref):
            off = [{**c, "value": c["value"] * 1.5 + 0.2} if "value" in c else c for c in ref]
            assert not G.claims_complete({"claims": off}, priv)
        # Names-only: answers component claims exactly but cannot report values computed from shapes or weights.
        names_only = [{k: v for k, v in c.items() if k != "value"} for c in ref]
        if any(c["report"] for c in pub):
            assert not G.claims_complete({"claims": names_only}, priv)


# ------------------------------------------------------------------------------------------- task build
@pytest.mark.parametrize("tid", sorted(PAIRS))
def test_l0_is_the_default(fakes, tid):
    cls = PAIRS[tid]
    default = _build(fakes, cls, None, name="d")
    l0 = _build(fakes, cls, 0, name="l0")
    assert [r for r in _listing(l0) if not r.startswith("knob.json ")] == _listing(default)
    assert cls().prompt(default) == cls().prompt(l0)
    assert not (default / "public" / "knob.json").exists() and not (default / "private" / "knob_claims.json").exists()
    ev = cls().evaluate(l0, cls().baseline(l0))
    assert ev.details["trap_score"] == 1.0 and ev.s_plus == 0.5 and "claims_complete" not in ev.details


@pytest.mark.parametrize("tid", sorted(PAIRS))
def test_l1_split_margin_and_leakage(fakes, tid):
    cls = PAIRS[tid]
    task = cls()
    for seed in range(3):
        art = _build(fakes, cls, 1, seed=seed, name=f"l1_{seed}")
        knob = json.loads((art / "public" / "knob.json").read_text())
        assert knob["value"] == 4 and [c["claim"] for c in knob["claims"]] == [1, 2, 3, 4]
        assert all(set(c) == {"claim", "family", "kind", "params", "description", "report"} for c in knob["claims"])
        public_text = "".join(p.read_text(errors="ignore") for p in (art / "public").rglob("*.json"))
        assert '"truth"' not in public_text and '"relevant"' not in public_text and '"tol_' not in public_text
        text = task.prompt(art)
        assert "This instance has 4 candidate claims" in text
        for c in knob["claims"]:
            assert f"{c['claim']}. [{c['kind']}] {c['description']}" in text
        base, ref = task.baseline(art), task.reference(art)
        b, r = task.evaluate(art, base), task.evaluate(art, ref)
        # Hand-computed with the fake decoy (score 0): baseline ½·0 + ½·0 = 0; reference ½·1 + ½·0 = 0.5.
        assert (b.details["trap_score"], b.s_plus) == (0.0, 0.0)
        assert (r.details["trap_score"], r.s_plus) == (1.0, 0.5)
        extra = set(r.details) - {"trap_case", "decoy_case", "decoy_task", "decisions", "trap_score", "decoy_score",
                                  "decoy_d_col", "decoy", "report"}
        assert extra == {"claims_complete", "claims"} and r.details["claims"] == 4
        # Reference reads public files only: hiding private claims does not change it.
        hidden = art / "private" / "knob_claims.json"
        hidden.rename(art / "moved.json")
        assert task.reference(art).report == ref.report
        (art / "moved.json").rename(hidden)


def test_l2_builds_16_claims(fakes):
    art = _build(fakes, T.AbsentAlgorithmTrap, 2)
    assert len(json.loads((art / "public" / "knob.json").read_text())["claims"]) == 16
    task = T.AbsentAlgorithmTrap()
    assert task.evaluate(art, task.reference(art)).details["trap_score"] == 1.0


def test_wrong_claim_scores_zero_without_saying_which(fakes):
    art = _build(fakes, T.HiddenKnowledgeTrap, 1)
    task = T.HiddenKnowledgeTrap()
    sub = task.reference(art)
    c = sub.report[_trap(art)]["claims"][0]
    c["verdict"] = "false" if c["verdict"] == "true" else "true"
    ev = task.evaluate(art, sub)
    assert ev.details["trap_score"] == 0.0 and ev.details["claims_complete"] is False
    dumped = json.dumps({k: v for k, v in ev.details.items() if k != "report"})
    assert '"relevant"' not in dumped and '"truth"' not in dumped
