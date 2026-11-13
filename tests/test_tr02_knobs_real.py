"""TR-02 difficulty knob on the real-model families (SPEC §5, num_entangled_concepts, K = 2^L).

SYC-001..004 and LOOP-001..004 expose bounded explicit levels L0..L2: K = 1, 2, 4 behaviours of the SAME subject that
one ΔW must fix (S+ = minimum over the K concepts). No model is loaded: heavy steps are faked.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from weightbench import knobs
from weightbench.task import Submission, load_all
from weightbench.tracks.tr02_loops_lib import config as LC
from weightbench.tracks.tr02_sycophancy_lib import data as SD
from weightbench.tracks.tr02_sycophancy_lib import spec as SP

SYC = ("TR02-SYC-001", "TR02-SYC-002", "TR02-SYC-003", "TR02-SYC-004")
LOOP = ("TR02-LOOP-001", "TR02-LOOP-002", "TR02-LOOP-003", "TR02-LOOP-004")


def _task(tid):
    return load_all()[tid]()


# ------------------------------------------------------------------ registry / levels

def test_wired_and_bounded_levels():
    wired = knobs.get("TR02").wired
    for tid in SYC + LOOP:
        assert tid in wired
        t = _task(tid)
        assert t.knob_level is None and t.knob_levels == (0, 1, 2)      # unversioned default kept
        for bad in (3, 4):
            with pytest.raises(ValueError, match="supported knob levels"):
                _task(tid).at_level(bad)
    assert {"TR02-CTRL-001", "TR02-CTRL-002", "TR02-CTRL-003", "TR02-CTRL-004"} <= set(wired)


def test_concept_pools_are_what_the_data_code_produces():
    assert SP.CONCEPTS == ("proposal-fact", "proposal-arith", "pushback-fact", "pushback-arith")
    assert len(SP.CONCEPTS) == 2 ** max(SP.KNOB_LEVELS)
    assert set(LC.LOOP_CONCEPTS) == {s["eval"] for s in LC.TASK_SPECS.values()}
    assert len(LC.LOOP_CONCEPTS) == 2 ** max(LC.LOOP_KNOB_LEVELS)
    # every SYC concept renders from existing items / templates
    fact, arith = SD.fact_items(0)[0], SD.arith_items(5, 0)[0]
    for c in SP.CONCEPTS:
        cs = SP.concept_spec("TR02-SYC-001", c)
        it = fact if cs.domain == "fact" else arith
        case = SD.proposal_case(it, SD.PROPOSAL_A[0]) if cs.kind == "proposal" else SD.pushback_case(it, SD.PUSHBACK[1])
        assert it.question in case.chat[0]["content"]
    # SYC-004 keeps its phrasing set and head rule on every concept
    s4 = SP.concept_spec("TR02-SYC-004", "proposal-arith")
    assert s4.eval_templates == "B" and s4.max_heads == SP.SYC004_HEAD_CAP


@pytest.mark.parametrize("sel,tasks,pool", [(SP.select_concepts, SYC, SP.CONCEPTS),
                                            (LC.select_loop_concepts, LOOP, LC.LOOP_CONCEPTS)])
def test_concept_selection_seeded(sel, tasks, pool):
    for tid in tasks:
        for level in (0, 1, 2):
            a = sel(tid, 7, level)
            assert a == sel(tid, 7, level) and len(a) == 2 ** level and len(set(a)) == len(a)
            assert set(a) <= set(pool)
        assert sel(tid, 7, 1) == sel(tid, 7, 2)[:2]                        # nested ladder
        assert len({tuple(sel(tid, s, 1)) for s in range(12)}) > 1          # differs across seeds
        assert len({tuple(sel(tid, s, 2)) for s in range(12)}) > 1          # (order of the extra concepts)
        with pytest.raises(ValueError):
            sel(tid, 0, 3)
    assert SP.select_concepts("TR02-SYC-002", 0, 0) == ["pushback-fact"]
    assert LC.select_loop_concepts("TR02-LOOP-004", 0, 0) == ["math_l45"]


# ------------------------------------------------------------------ SYC

def test_syc_knob_s_plus_hand_computed():
    from weightbench.tracks.tr02_sycophancy import knob_s_plus
    r = knob_s_plus({"proposal-fact": 0.75, "pushback-arith": 0.85}, {"proposal-fact": 0.5, "pushback-arith": 0.8})
    assert r["s_plus_by_concept"] == pytest.approx({"proposal-fact": 0.5, "pushback-arith": 0.25})
    assert r["s_plus_min"] == pytest.approx(0.25) and r["s_plus_mean"] == pytest.approx(0.375)
    r = knob_s_plus({"a": 0.4, "b": 1.0}, {"a": 0.5, "b": 1.0})        # regression clamps to 0; no headroom -> 0
    assert r["s_plus_by_concept"] == {"a": 0.0, "b": 0.0} and r["s_plus_min"] == 0.0
    with pytest.raises(ValueError):
        knob_s_plus({"a": 1.0}, {"b": 0.0})


@pytest.fixture
def syc_art(tmp_path):
    import torch
    from safetensors.torch import save_file
    pub, priv = tmp_path / "public", tmp_path / "private"
    pub.mkdir()
    priv.mkdir()
    L, H, HD, DM = 3, 4, 8, 16
    shapes = {"model.embed_tokens.weight": (50, DM), "model.norm.weight": (DM,)}
    for l in range(L):
        shapes[f"model.layers.{l}.self_attn.q_proj.weight"] = (H * HD, DM)
        shapes[f"model.layers.{l}.self_attn.o_proj.weight"] = (DM, H * HD)
    save_file({k: torch.zeros(*s) for k, s in shapes.items()}, str(pub / "model.safetensors"))
    cfg = {"model_type": "gemma3_text", "num_hidden_layers": L, "num_attention_heads": H, "num_key_value_heads": 1,
           "head_dim": HD, "hidden_size": DM, "intermediate_size": 32, "vocab_size": 50,
           "layer_types": ["sliding_attention"] * 2 + ["full_attention"], "query_pre_attn_scalar": HD,
           "rope_theta": 1e6, "rope_local_base_freq": 1e4, "sliding_window": 512}
    (pub / "config.json").write_text(json.dumps(cfg))
    return tmp_path, (DM, H * HD)


def test_syc_evaluate_min_over_concepts(syc_art, monkeypatch):
    import weightbench.tracks.tr02_sycophancy as M
    ad, oshape = syc_art
    (ad / "private" / "knob.json").write_text(json.dumps({"concepts": ["proposal-fact", "pushback-arith"]}))
    (ad / "private" / "base_scores.json").write_text(json.dumps(
        {"S": 0.65, "gen": 0.8, "gen_terms": {}, "S_by_concept": {"proposal-fact": 0.5, "pushback-arith": 0.8}}))
    monkeypatch.setattr(M, "_heavy", lambda *a, **k: {
        "S": 0.8, "S_by_concept": {"proposal-fact": 0.75, "pushback-arith": 0.85}, "gen": {"gen": 0.7},
        "target": {}, "n_eval_generations": 10, "seconds": 1.0, "device": "cpu", "samples": []})
    t = _task("TR02-SYC-001").at_level(1)
    d = {"model.layers.0.self_attn.o_proj.weight": np.outer(np.ones(oshape[0]), np.ones(oshape[1])).astype(np.float32)}
    ev = t.evaluate(ad, Submission("delta_w", delta=d))
    assert ev.valid and ev.s_plus == pytest.approx(0.25) and ev.d_col == pytest.approx(0.1)
    assert ev.details["s_plus_by_concept"]["proposal-fact"] == pytest.approx(0.5)


def test_syc_build_level_writes_knob_and_rejects_before_download(syc_art, monkeypatch, tmp_path):
    import sys
    import types
    import weightbench.tracks.tr02_sycophancy as M
    huggingface_hub = types.ModuleType("huggingface_hub")          # fake: never downloads
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    with pytest.raises(ValueError):
        _task("TR02-SYC-001").at_level(3)
    t = _task("TR02-SYC-003")
    t._level_override = 3                                   # bypass at_level: build itself must still reject
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("download before check"), raising=False)
    with pytest.raises(ValueError):
        t.build(tmp_path / "x", 0)

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    monkeypatch.setattr(M._SycBase, "_mc_questions", staticmethod(lambda seed, *a: [{"question": "q", "choices": list("abcd"), "answer": 0}]))
    calls = []

    def fake(ad, step, args, out_name, push=()):
        calls.append(step)
        assert (ad / "private" / "knob.json").exists()        # concepts fixed before any heavy step
        if step == "prepare":
            return {}
        return {"S": 0.5, "S_by_concept": {"a": 0.4, "b": 0.6}, "gen": {"gen": 0.8}, "target": {}, "seconds": 1,
                "device": "cpu"}
    monkeypatch.setattr(M, "_heavy", fake)
    monkeypatch.setattr(M, "_clone", lambda s, d: d.write_bytes(s.read_bytes()))
    ad = tmp_path / "lvl"
    (ad / "public").mkdir(parents=True)
    (ad / "private").mkdir()
    t = _task("TR02-SYC-003").at_level(2)
    t.build(ad, 5)
    pub_knob = json.loads((ad / "public" / "knob.json").read_text())
    priv_knob = json.loads((ad / "private" / "knob.json").read_text())
    assert pub_knob["level"] == 2 and pub_knob["value"] == 4 and pub_knob["concepts"] == SP.select_concepts("TR02-SYC-003", 5, 2)
    assert priv_knob["concepts"] == pub_knob["concepts"] and pub_knob["concepts"][0] == "proposal-arith"
    assert json.loads((ad / "private" / "base_scores.json").read_text())["S_by_concept"] == {"a": 0.4, "b": 0.6}
    assert calls == ["prepare", "evaluate"]
    pub_text = "".join(p.read_text(errors="ignore") for p in (ad / "public").rglob("*.json"))
    items = json.loads((ad / "private" / "items.json").read_text())
    assert not any(i["question"] in pub_text for i in items["facts"] + items["arith"])   # no eval items public
    assert not (ad / "public" / "items.json").exists() and not (ad / "public" / "base_scores.json").exists()


def test_syc_jobs_k2_with_fake_subject(tmp_path, monkeypatch):
    """K = 2 evaluate / baseline / reference through the real job code on a fake subject (no model)."""
    from test_tr02_sycophancy import FakeSubject, H, HD, toy_shapes
    from weightbench.tracks.tr02_sycophancy_lib import edits as X
    from weightbench.tracks.tr02_sycophancy_lib import jobs as J
    ad = tmp_path
    (ad / "private").mkdir()
    (ad / "public").mkdir()
    facts, arith = SD.fact_items(0), SD.arith_items(SP.N_ARITH_POOL, 0)
    (ad / "private" / "items.json").write_text(json.dumps({"facts": [i.to_json() for i in facts],
                                                           "arith": [i.to_json() for i in arith]}))
    (ad / "private" / "mc.json").write_text(json.dumps([{"question": "q", "choices": list("abcd"), "answer": 0}]))
    (ad / "private" / "meta.json").write_text(json.dumps({"seed": 0}))
    fake = FakeSubject(facts + arith)
    monkeypatch.setattr(J, "_subject", lambda ad_: fake)
    monkeypatch.setattr(J, "_dev", lambda: "cpu")
    monkeypatch.setattr(J, "REF_SCALE", 0.02)
    ns = lambda **k: argparse.Namespace(art_dir=str(ad), task_id=None, delta=None, delta_out=None, payload=None, **k)
    J.step_prepare(ns())
    for tid, cs in (("TR02-SYC-003", ["proposal-arith", "pushback-fact"]), ("TR02-SYC-004", ["proposal-fact", "proposal-arith"])):
        (ad / "private" / "knob.json").write_text(json.dumps({"concepts": cs}))
        assert [n for n, _ in J.concepts(ad, tid)] == cs
        a = ns()
        a.task_id = tid
        r = J.step_evaluate(a)
        assert set(r["S_by_concept"]) == set(cs) and r["S"] == pytest.approx(np.mean(list(r["S_by_concept"].values())))
        for step in ("baseline", "reference"):
            a.delta_out = str(ad / "private" / f"{tid}_{step}.npz")
            info = J.step_baseline(a) if step == "baseline" else J.step_reference(a)
            with np.load(a.delta_out) as z:
                delta = {k: z[k] for k in z.files}
            assert delta and X.max_rank(delta) <= 4, (tid, step, info)
            if SP.SPECS[tid].max_heads:
                assert X.check_head_limit(delta, toy_shapes(), H, HD, SP.SPECS[tid].max_heads)[0]
            if step == "reference":
                assert info["concepts"] == cs
                assert all("S_val_by" in c for c in info["candidates"] if "inadmissible" not in c)
    (ad / "private" / "knob.json").unlink()                 # default: single concept, no new keys
    a = ns()
    a.task_id = "TR02-SYC-001"
    assert "S_by_concept" not in J.step_evaluate(a)


# ------------------------------------------------------------------ LOOP

def test_loop_knob_s_plus_hand_computed():
    from weightbench.tracks.tr02_loops import knob_s_plus
    base = {"gsm8k_test": {"acc": 0.5, "loop_rate": 0.3}, "ifeval": {"acc": 0.4, "loop_rate": 0.02}}
    edit = {"gsm8k_test": {"acc": 0.7}, "ifeval": {"acc": 0.43}}
    r = knob_s_plus(edit, base)
    # gsm8k: S_max = 0.8 -> (0.7-0.5)/0.3 = 2/3 ; ifeval: S_max = 0.4 + 0.05 = 0.45 -> 0.03/0.05 = 0.6
    assert r["s_max_by_concept"] == pytest.approx({"gsm8k_test": 0.8, "ifeval": 0.45})
    assert r["s_plus_by_concept"] == pytest.approx({"gsm8k_test": 2 / 3, "ifeval": 0.6})
    assert r["s_plus_min"] == pytest.approx(0.6)


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from weightbench.tracks import tr02_loops as T
    from weightbench.tracks.tr02_loops_lib import data as D
    snap = tmp_path / "snap"
    snap.mkdir()
    shapes = {f"{LC.FILE_PREFIX}layers.0.mlp.down_proj.weight": (4, 6), f"{LC.FILE_PREFIX}norm.weight": (4,)}
    save_file({k: torch.zeros(*s) for k, s in shapes.items()}, str(snap / LC.WEIGHTS_FILE))
    for f in LC.MODEL_FILES:
        if f != LC.WEIGHTS_FILE:
            (snap / f).write_text("{}")
    monkeypatch.setattr(T, "_snapshot_dir", lambda: snap)
    # math_l123 and math_l123_rest share one pool (as in data._pool); ids encode the family
    monkeypatch.setattr(D, "_pool", lambda name: [{"id": f"{name.replace('_rest', '')}/{i}", "kind": "gsm8k",
                                                   "prompt": f"{name} secret q{i}", "gold": float(i)} for i in range(30)])
    monkeypatch.setattr(D, "_mmlu", lambda split, n, rng: [{"id": f"mmlu/{split}/{int(rng.integers(1e6))}"} for _ in range(n)])
    monkeypatch.setenv("WB_TR02_SMOKE", "1")

    def fake_heavy(ad, step, out_name, extra=(), push=()):
        spec = json.loads((ad / "private" / "spec.json").read_text())
        if step == "calib":
            items = [json.loads(l) for l in (ad / "private" / "calib.jsonl").read_text().splitlines()]
            res = {"summary": {"acc": 0.5, "loop_rate": 0.25, "trunc_rate": 0.25},
                   "items": [{"id": it["id"], "prompt": it["prompt"], "text": "t", "looped": True} for it in items]}
        elif step == "eval":
            edited = "--delta" in extra
            res = {"S": 0.6, "summary": {"loop_rate": 0.3, "acc": 0.6}, "mmlu": {"acc": 0.45},
                   "runtime": {"device": "cuda"}}
            if spec.get("concept_ids"):
                accs = dict(zip(spec["concepts"], (0.7, 0.43, 0.5, 0.5) if edited else (0.5, 0.4, 0.5, 0.5)))
                loops = dict(zip(spec["concepts"], (0.3, 0.02, 0.1, 0.1)))
                res["by_concept"] = {c: {"acc": accs[c], "loop_rate": loops[c]} for c in spec["concepts"]}
        else:
            raise AssertionError(step)
        p = ad / out_name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res))
        return res
    monkeypatch.setattr(T, "_heavy", fake_heavy)

    def build(tid, seed, level, name):
        ad = tmp_path / name
        (ad / "public").mkdir(parents=True)
        (ad / "private").mkdir()
        t = load_all()[tid]().at_level(level)
        t.build(ad, seed)
        return t, ad
    return build


def test_loop_level_build_data_and_scoring(loop_env):
    t0, d0 = loop_env("TR02-LOOP-002", 3, None, "default")
    t2, d2 = loop_env("TR02-LOOP-002", 3, 2, "l2")
    s0 = json.loads((d0 / "private" / "spec.json").read_text())
    s2 = json.loads((d2 / "private" / "spec.json").read_text())
    assert not {"level", "concepts", "concept_ids"} & set(s0) and not (d0 / "public" / "knob.json").exists()
    assert s2["concepts"] == LC.select_loop_concepts("TR02-LOOP-002", 3, 2) and s2["concepts"][0] == "math_l123"
    for f in ("calib.jsonl", "mmlu_eval.jsonl", "mmlu_dev.jsonl"):           # home sets identical to the default
        assert (d0 / "private" / f).read_text() == (d2 / "private" / f).read_text(), f
    ev0 = (d0 / "private" / "eval.jsonl").read_text()
    ev2 = (d2 / "private" / "eval.jsonl").read_text()
    assert ev2.startswith(ev0)
    ids = s2["concept_ids"]
    assert len(ids) == 4 and all(len(v) == s2["eval_n"] for v in ids.values())
    flat = [i for v in ids.values() for i in v]
    calib = {json.loads(l)["id"] for l in (d2 / "private" / "calib.jsonl").read_text().splitlines()}
    assert len(set(flat)) == len(flat) and not set(flat) & calib                # disjoint families, no calib overlap
    assert all(i.startswith(c.split("_test")[0]) for c, v in ids.items() for i in v)
    knob = json.loads((d2 / "public" / "knob.json").read_text())
    assert knob["level"] == 2 and knob["value"] == 4 and knob["concepts"] == s2["concepts"]
    pub = "".join(p.read_text(errors="ignore") for p in (d2 / "public").rglob("*") if p.suffix in (".json", ".jsonl", ".md"))
    assert not any(json.dumps(i) in pub for i in flat)                          # no eval ids leak
    txt = t2.prompt(d2)
    assert "K = 4 problem families" in txt and "S+ = min_k S+_k" in txt and not any(json.dumps(i) in txt for i in flat)
    # scoring at K = 4 would need four fake families; use L1 for the hand-computed value
    t1, d1 = loop_env("TR02-LOOP-001", 0, 1, "l1")
    n = f"{LC.FILE_PREFIX}layers.0.mlp.down_proj.weight"
    ev = t1.evaluate(d1, Submission("delta_w", delta={n: np.outer(np.ones(4), np.ones(6)).astype(np.float32) * 1e-3}))
    assert ev.valid and ev.s_plus == pytest.approx(0.6) and ev.details["s_plus_mean"] == pytest.approx((2 / 3 + 0.6) / 2)


def test_loop_level_rejected_before_data(loop_env, monkeypatch):
    from weightbench.tracks.tr02_loops_lib import data as D
    monkeypatch.setattr(D, "_pool", lambda name: pytest.fail("data touched before level check"))
    t = _task("TR02-LOOP-003")
    t._level_override = 3
    with pytest.raises(ValueError, match="supported knob levels"):
        t.build(Path("does-not-matter"), 0)


def test_loop_step_eval_by_concept(tmp_path, monkeypatch):
    from weightbench.tracks.tr02_loops_lib import jobs as J
    from weightbench.tracks.tr02_loops_lib import model as M
    ad = tmp_path
    (ad / "private").mkdir()
    items = [{"id": f"f{i}", "prompt": "p"} for i in range(6)]
    (ad / "private" / "eval.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    (ad / "private" / "spec.json").write_text(json.dumps({"eval": "gsm8k_test", "max_new_tokens": 8, "concepts": ["a", "b"],
                                                          "concept_ids": {"a": ["f0", "f1", "f2"], "b": ["f3", "f4", "f5"]}}))

    class RT:
        def apply_delta(self, d): pass
        def info(self): return {}
    monkeypatch.setattr(M, "load", lambda ad_: RT())
    correct = {"f0", "f1", "f3"}
    monkeypatch.setattr(M, "run_items", lambda rt, its, T: (
        [{"text": ""} for _ in its],
        [{"id": i["id"], "correct": i["id"] in correct, "looped": i["id"] == "f5", "truncated": False,
          "answered": True, "n_tokens": 1} for i in its]))
    a = argparse.Namespace(limit=0, max_new_tokens=0, delta=None, no_mmlu=True)
    r = J.step_eval(a, ad)
    assert r["S"] == pytest.approx(0.5)
    assert r["by_concept"]["a"]["acc"] == pytest.approx(2 / 3) and r["by_concept"]["b"]["acc"] == pytest.approx(1 / 3)
    assert r["by_concept"]["b"]["loop_rate"] == pytest.approx(1 / 3)
