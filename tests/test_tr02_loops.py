"""TR02-LOOP fast tests: no model is loaded (toy tensors + a mocked heavy executor)."""
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from weightbench import scoring
from weightbench.ledger import Ledger
from weightbench.server import RunState
from weightbench.tracks import tr02_loops as T
from weightbench.tracks.tr02_loops_lib import config as C
from weightbench.tracks.tr02_loops_lib import data as D
from weightbench.tracks.tr02_loops_lib import edits as E
from weightbench.tracks.tr02_loops_lib import ifeval as IF
from weightbench.tracks.tr02_loops_lib import textproc as TP

# ---------------------------------------------------------------- loop detection

def test_loop_detector_repeating_tail():
    head = list(range(300))
    block = list(range(1000, 1040))
    seq = head + block * 20
    st = TP.loop_stats(seq, finished=False)
    assert st["looped"] and st["truncated"] and st["rep_frac"] > 0.9
    assert TP.loop_onset(seq) is not None and 300 <= TP.loop_onset(seq) <= 340 + 16


def test_loop_detector_negatives():
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 50000, size=1500).tolist()
    assert not TP.loop_stats(seq, finished=False)["looped"]          # truncated but not repetitive
    assert TP.loop_stats(seq, finished=False)["truncated"]
    rep = list(range(40)) * 40
    assert not TP.loop_stats(rep, finished=True)["looped"]           # repetitive but finished (EOS)
    assert TP.loop_stats([1, 2, 3], finished=False)["rep_frac"] == 0.0
    assert TP.loop_onset(seq) is None


# ---------------------------------------------------------------- answer extraction

def test_split_and_boxed():
    th, ans = TP.split_thinking("let me think...\n</think>\n\nSo the answer is \\boxed{\\frac{1}{2}}.")
    assert ans.startswith("So") and TP.last_boxed(ans) == "\\frac{1}{2}"
    assert TP.split_thinking("thinking forever")[1] is None
    assert TP.last_boxed("\\boxed{a} then \\boxed{x^{2}+1}") == "x^{2}+1"
    assert TP.last_boxed("no box") is None


def test_gsm8k_grading():
    assert TP.gsm8k_gold("blah\n#### 1,234") == 1234.0
    assert TP.gsm8k_correct("Total: \\boxed{1,234}", 1234.0)
    assert TP.gsm8k_correct("\\boxed{\\$72}", 72.0)
    assert TP.gsm8k_correct("The answer is 72 clips.", 72.0)
    assert not TP.gsm8k_correct("\\boxed{71}", 72.0)
    assert not TP.gsm8k_correct(None, 72.0)


@pytest.mark.parametrize("pred,gold,ok", [
    ("\\dfrac{1}{2}", "\\frac{1}{2}", True),
    ("0.5", "\\frac12", True),
    ("\\left( 3, \\frac{\\pi}{2} \\right)", "\\left( 3, \\frac{\\pi}{2} \\right)", True),
    ("(3,\\frac{\\pi}{2})", "\\left( 3, \\frac{\\pi}{2} \\right)", True),
    ("x = 5", "5", True),
    ("10\\%", "10", True),
    ("90^\\circ", "90", True),
    ("\\text{(C)}", "\\text{(C)}", True),
    ("3", "4", False),
    (None, "4", False),
])
def test_math_equal(pred, gold, ok):
    assert TP.math_equal(pred, gold) is ok


def test_ifeval_checks():
    ids = ["punctuation:no_comma", "change_case:english_lowercase", "detectable_format:number_bullet_lists"]
    kw = [{}, {}, {"num_bullets": 2}]
    assert IF.check("* one\n* two", ids, kw)
    assert not IF.check("* one, really\n* two", ids, kw)
    assert not IF.check("* One\n* two", ids, kw)
    assert not IF.check(None, ids, kw)
    assert IF.check('"hello there"', ["startend:quotation"], [{}])
    assert IF.check("a b c d", ["length_constraints:number_words"], [{"relation": "less than", "num_words": 5}])
    assert IF.check('```json\n{"a": 1}\n```', ["detectable_format:json_format"], [{}])
    assert IF.check("<<Title>> body", ["detectable_format:title"], [{}])
    assert not IF.supported(["language:response_language"])


# ---------------------------------------------------------------- ΔW construction / validation

def test_projection_factors_remove_direction():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(16, 40)).astype(np.float32)
    r = rng.normal(size=16)
    f = E.projection_factors(W, r, 1.0)
    name = f"{C.FILE_PREFIX}layers.3.mlp.down_proj.weight"
    dW = E.materialize({name: f}, {name: W.shape})[name]
    assert np.linalg.matrix_rank(dW) == 1
    rn = r / np.linalg.norm(r)
    assert np.abs(rn @ (W + dW)).max() < 1e-4                    # edited W cannot write along r
    # α = 2 reflects the component
    dW2 = E.materialize({name: E.projection_factors(W, r, 2.0)}, {name: W.shape})[name]
    assert np.allclose(rn @ (W + dW2), -(rn @ W), atol=1e-4)


def test_materialize_rejects_bad_factors():
    name = f"{C.FILE_PREFIX}layers.0.mlp.down_proj.weight"
    shapes = {name: (4, 6), f"{C.FILE_PREFIX}norm.weight": (4,)}
    ok = {"U": [[1, 0, 0, 0]], "V": [[0, 1, 0, 0, 0, 0]]}
    assert E.materialize({name: ok}, shapes)[name][0, 1] == 1.0
    with pytest.raises(ValueError):
        E.materialize({"model.language_model.embed_tokens.weight": ok}, shapes)   # frozen / unknown
    with pytest.raises(ValueError):
        E.materialize({name: {"U": [[1, 0, 0, 0]] * 5, "V": [[0] * 6] * 5}}, shapes)  # rank > 4
    with pytest.raises(ValueError):
        E.materialize({name: {"U": [[np.nan, 0, 0, 0]], "V": [[0] * 6]}}, shapes)


def test_validate_delta_constraints():
    a = f"{C.FILE_PREFIX}layers.1.mlp.down_proj.weight"
    b = f"{C.FILE_PREFIX}layers.2.mlp.up_proj.weight"
    att = f"{C.FILE_PREFIX}layers.3.self_attn.o_proj.weight"
    shapes = {a: (4, 6), b: (6, 4), att: (4, 4)}
    r1 = np.outer(np.ones(4), np.arange(6)).astype(np.float32)
    assert E.validate_delta({a: r1}, shapes, "single_rank1") is None
    assert E.validate_delta({a: r1, b: np.zeros((6, 4))}, shapes, "single_rank1") is None   # zero tensor ignored
    assert "exactly one" in E.validate_delta({a: r1, b: np.ones((6, 4))}, shapes, "single_rank1")
    assert "rank 1" in E.validate_delta({a: np.eye(4, 6)}, shapes, "single_rank1")
    assert "MLP" in E.validate_delta({att: np.outer(np.ones(4), np.ones(4))}, shapes, "single_rank1")
    assert E.validate_delta({a: np.eye(4, 6)}, shapes, "l0_or_rank4") is None
    assert "shape" in E.validate_delta({a: np.zeros((6, 4))}, shapes, "l0_or_rank4")
    assert "unknown" in E.validate_delta({"x": np.zeros(1)}, shapes, "l0_or_rank4")


def test_baseline_choice():
    sep = {l: float(l == 20) * 5 + float(l == 17) * 3 + 0.1 for l in range(24)}
    bc = E.baseline_choice(sep)
    assert bc == {"src": 20, "edit": [17, 18, 19, 20], "alpha": 1.0}
    assert E.baseline_choice(sep, "single_rank1")["edit"] == [20]


# ---------------------------------------------------------------- toy task end-to-end (mocked heavy)

TOY = {
    f"{C.FILE_PREFIX}embed_tokens.weight": (10, 4),
    f"{C.FILE_PREFIX}layers.0.mlp.down_proj.weight": (4, 6),
    f"{C.FILE_PREFIX}layers.0.mlp.up_proj.weight": (6, 4),
    f"{C.FILE_PREFIX}layers.1.mlp.down_proj.weight": (4, 6),
    f"{C.FILE_PREFIX}norm.weight": (4,),
    "model.visual.patch.weight": (3, 3),
}


@pytest.fixture
def toy_task(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    g = torch.Generator().manual_seed(0)
    save_file({k: torch.randn(*s, generator=g).to(torch.bfloat16) for k, s in TOY.items()}, str(snap / C.WEIGHTS_FILE))
    for f in C.MODEL_FILES:
        if f != C.WEIGHTS_FILE:
            (snap / f).write_text("{}")
    monkeypatch.setattr(T, "_snapshot_dir", lambda: snap)

    def pool(name):
        return [{"id": f"{name}/{i}", "kind": "gsm8k", "prompt": f"q{i}", "gold": float(i)} for i in range(30)]
    monkeypatch.setattr(D, "_pool", pool)
    monkeypatch.setattr(D, "_mmlu", lambda split, n, rng: [{"id": f"mmlu/{split}/{i}"} for i in range(n)])
    monkeypatch.setenv("WB_TR02_SMOKE", "1")
    calls = []

    def fake_heavy(ad, step, out_name, extra=(), push=()):
        calls.append((step, list(extra), [str(p) for p in push]))
        for p in push:
            assert p.exists() and ad.resolve() in p.resolve().parents   # ΔW lives inside art_dir
        if step == "calib":
            items = [json.loads(l) for l in (ad / "private" / "calib.jsonl").read_text().splitlines()]
            res = {"summary": {"acc": 0.5, "loop_rate": 0.25, "trunc_rate": 0.25},
                   "items": [{"id": it["id"], "prompt": it["prompt"], "text": "t", "looped": i == 0}
                             for i, it in enumerate(items)]}
        elif step == "eval":
            edited = "--delta" in extra
            res = {"S": 0.7 if edited else 0.5, "summary": {"loop_rate": 0.3, "acc": 0.7 if edited else 0.5},
                   "mmlu": {"acc": 0.40 if edited else 0.45}, "runtime": {"device": "cuda"}}
        elif step in ("generate", "activations"):
            res = {"outputs": []}
        elif step in ("baseline", "reference"):
            n = f"{C.FILE_PREFIX}layers.1.mlp.down_proj.weight"
            res = {"ok": True, "factors": {n: {"U": [[1, 0, 0, 0]], "V": [[1, 2, 3, 4, 5, 6]]}}}
        else:
            raise AssertionError(step)
        p = ad / out_name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res))
        return res

    monkeypatch.setattr(T, "_heavy", fake_heavy)
    ad = tmp_path / "art"
    (ad / "public").mkdir(parents=True)
    (ad / "private").mkdir()
    task = T.Loop001()
    task.build(ad, 0)
    return task, ad, calls


def test_build_no_leakage(toy_task):
    task, ad, _ = toy_task
    priv_eval = {json.loads(l)["id"] for l in (ad / "private" / "eval.jsonl").read_text().splitlines()}
    assert priv_eval
    pub_text = "".join(p.read_text(errors="ignore") for p in (ad / "public").rglob("*") if p.is_file()
                       and p.suffix in (".json", ".jsonl"))
    assert not any(i in pub_text for i in priv_eval)                       # eval ids never public
    assert not any(p.name in ("eval.jsonl", "base_eval.json", "spec.json") for p in (ad / "public").rglob("*"))
    calib = {json.loads(l)["id"] for l in (ad / "private" / "calib.jsonl").read_text().splitlines()}
    assert not (calib & priv_eval)                                         # disjoint splits
    shapes = json.loads((ad / "public" / "param_shapes.json").read_text())
    assert set(shapes) == {k for k in TOY if C.editable(k)}                # frozen params not editable
    txt = task.prompt(ad)
    assert "TR02-LOOP-001" in txt and not any(i in txt for i in priv_eval)


def test_lazy_base_weights_edit_cost(toy_task):
    task, ad, _ = toy_task
    base = task.base_weights(ad)
    assert len(base) == 4 and f"{C.FILE_PREFIX}embed_tokens.weight" not in base
    n = f"{C.FILE_PREFIX}layers.0.mlp.down_proj.weight"
    assert base[n].dtype == np.float32 and base[n].shape == (4, 6)
    c = scoring.edit_cost({n: np.outer(np.ones(4), np.ones(6)).astype(np.float32)}, base)
    assert c["l0"] == pytest.approx(24 / (24 + 24 + 24 + 4))


def test_evaluate_scoring_and_rejections(toy_task):
    task, ad, calls = toy_task
    n = f"{C.FILE_PREFIX}layers.0.mlp.down_proj.weight"
    sub = T.Submission("delta_w", delta={n: np.outer(np.ones(4), np.ones(6)).astype(np.float32) * 1e-3})
    ev = task.evaluate(ad, sub)
    assert ev.valid
    s_max = min(1.0, 0.5 + 0.3)
    assert ev.s_plus == pytest.approx((0.7 - 0.5) / (s_max - 0.5))
    assert ev.d_col == pytest.approx(0.05)
    assert not list((ad / "private" / "evals").rglob("delta.npz"))      # ΔW cleaned up after eval
    n_calls = len(calls)
    bad = T.Submission("delta_w", delta={f"{C.FILE_PREFIX}embed_tokens.weight": np.ones((10, 4), np.float32)})
    assert not task.evaluate(ad, bad).valid
    assert not task.evaluate(ad, T.Submission("report", report={})).valid
    zero = task.evaluate(ad, T.Submission("delta_w", delta={n: np.zeros((4, 6), np.float32)}))
    assert zero.valid and zero.s_plus == 0.0
    assert len(calls) == n_calls                                         # no heavy job for invalid/zero ΔW
    dense = T.Submission("delta_w", delta={n: np.random.default_rng(0).normal(size=(4, 6)).astype(np.float32)})
    task.budget = T.Budget(n_forward=50, n_probe=20, max_l0=0.0005, max_rank=2)   # toy: rank-4 dense > budget
    assert "L0" in task.evaluate(ad, dense).details["reason"]
    assert len(calls) == n_calls
    # LOOP-004 strict constraint
    t4 = T.Loop004()
    two = {n: np.outer(np.ones(4), np.ones(6)).astype(np.float32),
           f"{C.FILE_PREFIX}layers.1.mlp.down_proj.weight": np.outer(np.ones(4), np.ones(6)).astype(np.float32)}
    assert not t4.evaluate(ad, T.Submission("delta_w", delta=two)).valid


def test_budget_rank_check_uses_edit_dtype():
    """A materialised rank-1 float32 edit is rank 1, not full rank (the rank test uses a float32-appropriate tolerance)."""
    task = T.Loop001()
    name = f"{C.FILE_PREFIX}layers.0.mlp.down_proj.weight"
    rng = np.random.default_rng(0)
    rank1 = (rng.standard_normal((256, 1)) @ rng.standard_normal((1, 512))).astype(np.float32)
    shapes = {name: rank1.shape}                                         # dense edit: L0 = 1 > max_l0
    assert task._budget_error({name: rank1}, shapes) is None
    dense = rng.standard_normal((256, 512)).astype(np.float32)
    assert "rank" in task._budget_error({name: dense}, shapes)


def test_single_rank1_validation_uses_edit_dtype():
    """LOOP-004 must accept rounded rank-one factors and reject actual rank two."""
    rng = np.random.default_rng(42)
    name = E.down_proj(18)
    u = rng.normal(size=(64, 2)).astype(np.float32)
    v = rng.normal(size=(2, 96)).astype(np.float32)
    shapes = {name: (64, 96)}
    rank1 = u[:, :1] @ v[:1]
    assert E.validate_delta({name: rank1}, shapes, "single_rank1") is None
    assert "rank 1" in E.validate_delta({name: u @ v}, shapes, "single_rank1")


def test_baseline_reference_materialize(toy_task):
    task, ad, _ = toy_task
    for sub in (task.baseline(ad), task.reference(ad)):
        (name, d), = sub.delta.items()
        assert d.shape == (4, 6) and np.linalg.matrix_rank(d) == 1


def test_probe_validation_and_budget(toy_task, tmp_path):
    task, ad, calls = toy_task
    with pytest.raises(ValueError):
        task.probe(ad, "generate", {"prompts": ["x"] * 9})
    with pytest.raises(ValueError):
        task.probe(ad, "generate", {"prompts": ["x"], "max_new_tokens": 10 ** 6})
    with pytest.raises(ValueError):
        task.probe(ad, "activations", {"texts": ["a"] * 8, "layers": [0, 1, 2, 3], "char_ranges": [[[0, 1], [1, 2]]] * 8})
    n = f"{C.FILE_PREFIX}layers.0.mlp.down_proj.weight"
    task.probe(ad, "generate", {"prompts": ["x"], "delta_factors": {n: {"U": [[1, 0, 0, 0]], "V": [[0] * 6]}}})
    assert calls[-1][0] == "generate" and calls[-1][2]                  # candidate ΔW pushed with the job
    assert not list((ad / "private" / "probes").rglob("delta.npz"))
    # budget enforcement through the metered server state
    led = Ledger(tmp_path / "run" / "ledger.jsonl")
    st = RunState(task, ad, tmp_path / "run", led, tmp_path / "work")
    codes = [st.probe("generate", {"prompts": ["x"]})[0] for _ in range(task.budget.n_forward + 1)]
    assert codes[:-1] == [200] * task.budget.n_forward and codes[-1] == 429
    codes = [st.probe("activations", {"texts": ["x"], "layers": [0]})[0] for _ in range(task.budget.n_probe + 1)]
    assert codes[-1] == 429
    assert sum(1 for e in led.entries("probe")) == task.budget.n_forward + task.budget.n_probe


def test_registered():
    from weightbench.task import REGISTRY, load_all
    load_all()
    assert {f"TR02-LOOP-00{i}" for i in range(1, 5)} <= set(REGISTRY)

def test_loop_contrast_falls_back_to_pre_onset_tokens(monkeypatch):
    """No finished generation (greedy thinking mode on MATH): clean class = tokens before the loop onset."""
    from weightbench.tracks.tr02_loops_lib import model as M

    hidden, layers = 8, [0, 1]
    monkeypatch.setattr(M.C, "HIDDEN", hidden)
    monkeypatch.setattr(M, "chat", lambda tok, prompt, thinking=True: prompt)
    monkeypatch.setattr(M, "loop_onset", lambda ids: 6)
    direction = torch.zeros(hidden)
    direction[3] = 1.0

    def fake_residuals(rt, ids, layer_list):
        h = torch.zeros(len(ids), hidden)
        h[len(ids) - 4:] += direction * 5.0                   # the last 4 generated tokens are "in the loop"
        return {l: h for l in layer_list}

    monkeypatch.setattr(M, "residuals", fake_residuals)

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0, 0]}

    class Rt:
        tok, dev = Tok(), torch.device("cpu")

    gens = [{"ids": list(range(10)), "looped": True, "finished": False} for _ in range(3)]
    st = M.loop_contrast_stats(Rt(), [{"prompt": "p"}] * 3, gens, layers)
    assert st["contrast"] == "pre_onset" and st["n_tokens"] == {"loop": 12, "ok": 18}
    monkeypatch.setattr(M, "PRE_ONSET_WINDOW", 2)                     # only the last 2 tokens before onset
    assert M.loop_contrast_stats(Rt(), [{"prompt": "p"}] * 3, gens, layers)["n_tokens"] == {"loop": 12, "ok": 6}
    diff = st["layers"][0]["diff"]
    assert int(np.argmax(np.abs(diff))) == 3 and st["layers"][0]["sep"] > 0
