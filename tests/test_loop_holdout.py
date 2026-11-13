"""TR02-LOOP reference holdout selection rule (pre-registered): splits, acceptance, abstention. Model-free."""
from pathlib import Path

import pytest

from weightbench.tracks.tr02_loops_lib import config as C
from weightbench.tracks.tr02_loops_lib.reference import (
    HOLDOUT_DELTA, HOLDOUT_DELTA_ITEMS, HOLDOUT_N, SELECTION_N, calibration_split, holdout_select, holdout_split,
    objective,
)

TASK = "TR02-LOOP-001"


def _splits(seed: int):
    pool = [{"id": f"p{i}"} for i in range(300)]
    eval_ids = {f"p{i}" for i in range(240, 300)}           # 60 eval items, as in TASK_SPECS
    fit, dev = calibration_split(pool[:32], pool, eval_ids, seed, TASK)
    sel, hold = holdout_split(dev, seed, TASK)
    return fit, sel, hold, eval_ids


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_four_id_sets_pairwise_disjoint_and_sized(seed):
    fit, sel, hold, eval_ids = _splits(seed)
    sets = {"calib": {x["id"] for x in fit}, "selection": {x["id"] for x in sel},
            "holdout": {x["id"] for x in hold}, "eval": eval_ids}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not sets[a] & sets[b], (a, b)
    assert (len(sets["selection"]), len(sets["holdout"])) == (SELECTION_N, HOLDOUT_N) == (24, 40)
    assert SELECTION_N + HOLDOUT_N == C.REF_DEV_N == 64


def test_split_is_deterministic_from_seed():
    assert _splits(3) == _splits(3)
    assert _splits(3)[1] != _splits(4)[1]
    _, dev = calibration_split([{"id": f"p{i}"} for i in range(32)], [{"id": f"p{i}"} for i in range(300)],
                               set(), 0, TASK)
    assert holdout_split(dev, 0, TASK) != holdout_split(dev, 0, "TR02-LOOP-002")
    with pytest.raises(ValueError):
        holdout_split(dev[:63], 0, TASK)


def _m(correct: int, n: int, mmlu: float = .5) -> dict:
    return {"summary": {"acc": correct / n, "n": n}, "mmlu": mmlu}


class FakeScorer:
    """Table of (candidate index or None, split name) -> number of correct items; records every call."""

    def __init__(self, table, sel, hold, mmlu=None):
        self.table, self.sel, self.hold, self.mmlu, self.calls = table, sel, hold, mmlu or {}, []

    def __call__(self, index, items, label):
        split = "sel" if items is self.sel else "hold"
        self.calls.append((index, split, label))
        return _m(self.table[index, split], len(items), self.mmlu.get(index, .5))


def _items():
    _, sel, hold, _ = _splits(0)
    return sel, hold


def test_hand_computed_objective_base_zero_one_correct():
    # base 0/24, edit 1/24, no MMLU drop: objective = 1/24 = 0.041666...
    assert objective(_m(1, 24), .5) == pytest.approx(1 / 24)
    # MMLU drop of 0.05 costs 2 * 0.05 = 0.1; an MMLU gain is not rewarded
    assert objective(_m(1, 24, .45), .5) == pytest.approx(1 / 24 - .1)
    assert objective(_m(1, 24, .9), .5) == pytest.approx(1 / 24)
    assert HOLDOUT_DELTA == pytest.approx(4 / 40) and 1 / 24 < HOLDOUT_DELTA


def test_non_transferring_selection_gain_is_rejected():
    sel, hold = _items()
    # candidate 7 wins big on selection (+6/24) but gains only 3/40 on holdout (< 4 items): abstain
    f = FakeScorer({(None, "sel"): 1, (7, "sel"): 7, (8, "sel"): 2, (None, "hold"): 1, (7, "hold"): 4}, sel, hold)
    rec = holdout_select([7, 8], f, sel, hold)
    assert rec["winner"] == 7 and not rec["accepted"]
    assert rec["holdout_gain"] == pytest.approx(3 / 40)
    assert "<" in rec["reason"]
    assert [(i, s) for i, s, _ in f.calls] == [(None, "sel"), (7, "sel"), (8, "sel"), (None, "hold"), (7, "hold")]


def test_all_candidates_at_or_below_base_returns_no_edit_without_touching_holdout():
    sel, hold = _items()
    f = FakeScorer({(None, "sel"): 3, (1, "sel"): 3, (2, "sel"): 1}, sel, hold)
    rec = holdout_select([1, 2], f, sel, hold)
    assert not rec["accepted"] and rec["selection_gain"] == 0
    assert "selection split" in rec["reason"]
    assert all(s == "sel" for _, s, _ in f.calls)
    assert not holdout_select([], f, sel, hold)["accepted"]


def test_mmlu_collateral_can_sink_a_holdout_gain():
    sel, hold = _items()
    table = {(None, "sel"): 0, (5, "sel"): 3, (None, "hold"): 1, (5, "hold"): 6}
    rec = holdout_select([5], FakeScorer(table, sel, hold, {5: .45}), sel, hold)
    # holdout: 6/40 - 1/40 - 2 * 0.05 = 0.025 < 0.1
    assert rec["holdout_gain"] == pytest.approx(5 / 40 - .1) and not rec["accepted"]


def test_genuinely_transferring_candidate_is_accepted():
    sel, hold = _items()
    table = {(None, "sel"): 1, (3, "sel"): 5, (4, "sel"): 5, (None, "hold"): 2, (3, "hold"): 2 + HOLDOUT_DELTA_ITEMS}
    rec = holdout_select([4, 3], FakeScorer(table, sel, hold), sel, hold)
    assert rec["winner"] == 3          # objective tie broken by the lower candidate index
    assert rec["accepted"] and rec["holdout_gain"] == pytest.approx(4 / 40)


def test_reference_forward_count_fits_author_cap():
    # run(): two metered author forwards per score call (generation + MMLU); shortlist <= k prompt + k loop, plus base
    # on S and base + winner on H. The cap is max(20, 2 * (2k + 3)): 20 for the default k = 2, 20 for LOOP-004's k = 3.
    from weightbench.tracks.tr02_loops_lib.reference import search_settings
    src = Path(__file__).resolve().parents[1] / "weightbench/tracks/tr02_loops_lib/reference.py"
    assert 'forward_cap=max(20, 2 * (2 * settings["shortlist"] + 3))' in src.read_text(encoding="utf-8")
    for task in C.TASK_SPECS:
        k = search_settings(task)["shortlist"]
        assert 2 * (1 + 2 * k + 2) <= max(20, 2 * (2 * k + 3))


@pytest.mark.parametrize("pair,decoy", [("TR03-PAIR-003", "TR02-LOOP-001"), ("TR03-PAIR-005", "TR02-LOOP-002")])
def test_decoy_prompt_item_count_matches_grader(pair, decoy):
    from weightbench.tracks.tr03_trap_lib.cases import PAIRS
    text = PAIRS[pair].decoy.text.scoring
    spec = C.TASK_SPECS[decoy]
    assert f"on {spec['eval_n']} held-out" in text and spec["eval_n"] == 60
    assert f"≤ {spec['max_new_tokens']} new tokens" in text
    assert "100 held-out" not in text


def test_loop_task_eval_text_uses_spec_count():
    from weightbench.tracks import tr02_loops as T
    for cls in (T.Loop001, T.Loop002):
        assert "{n}" in cls.eval_text and "100" not in cls.eval_text
        assert cls.eval_text.format(n=C.TASK_SPECS[cls.id]["eval_n"]).startswith("60 ")


def _cached_reference(tmp_path, res):
    import json
    from weightbench.tracks.tr02_loops import Loop001
    from weightbench.tracks.tr02_loops_lib.edits import down_proj
    (tmp_path / "public").mkdir()
    (tmp_path / "private").mkdir()
    (tmp_path / "public/param_shapes.json").write_text(json.dumps({down_proj(20): [4, 3]}))
    (tmp_path / "private/reference.json").write_text(json.dumps(res))
    return Loop001().reference(tmp_path)


def test_abstained_reference_submits_empty_edit_and_says_why(tmp_path):
    sub = _cached_reference(tmp_path, {"ok": True, "abstained": True, "reason": "holdout gain 0.0250 < delta",
                                       "factors": {}})
    assert sub.type == "delta_w" and sub.delta == {}
    assert sub.report["abstained"] and "holdout" in sub.report["reason"]


def test_old_reference_result_without_abstained_field_still_loads(tmp_path):
    from weightbench.tracks.tr02_loops_lib.edits import down_proj
    sub = _cached_reference(tmp_path, {"ok": True, "choice": {"index": 0}, "dev_n": 24,
                                       "factors": {down_proj(20): {"U": [[1, 0, 0, 0]], "V": [[0, 1, 0]]}}})
    assert sub.report is None and sub.delta[down_proj(20)][0, 1] == 1


def test_default_search_is_the_measured_loop_001_to_003_search():
    from weightbench.tracks.tr02_loops_lib.reference import STRENGTHS, activation_cap, search_settings
    for task in ("TR02-LOOP-001", "TR02-LOOP-002", "TR02-LOOP-003"):
        s = search_settings(task)
        assert s == {"strengths": STRENGTHS, "sites": 2, "shortlist": 2, "dev_pool": None}
        assert activation_cap(s) == 400                         # the cap the shipped references run under
    assert STRENGTHS == (.125, .25, .5, 1., 2., 4.)


def test_loop_004_search_is_wider_and_in_domain():
    from weightbench.tracks.tr02_loops_lib.reference import STRENGTHS, activation_cap, search_settings
    s = search_settings("TR02-LOOP-004")
    assert s["dev_pool"] == "math_l45" == C.TASK_SPECS["TR02-LOOP-004"]["eval"]
    assert set(STRENGTHS) < set(s["strengths"]) and list(s["strengths"]) == sorted(s["strengths"])
    assert s["sites"] == 3 and s["shortlist"] == 3
    # 160 prompt-fit passes + 32 loop-fit traces + 4 sites x 10 strengths x (1 prompt + 8 loop screens)
    assert activation_cap(s) == 160 + 32 + 4 * 10 * 9 == 552


def test_dev_pool_split_is_disjoint_from_eval_and_fit_and_deterministic():
    calib = [{"id": f"g{i}"} for i in range(300)]
    math = [{"id": f"m{i}"} for i in range(262)]
    eval_ids = {f"m{i}" for i in range(0, 120, 2)}              # 60 eval items inside the development family
    fit, dev = calibration_split(calib[:32], calib, eval_ids, 0, "TR02-LOOP-004", math + math[:5])
    assert len(fit) == 80 and fit[:32] == calib[:32] and all(x["id"].startswith("g") for x in fit)
    assert len(dev) == len({x["id"] for x in dev}) == C.REF_DEV_N
    assert all(x["id"].startswith("m") for x in dev) and not ({x["id"] for x in dev} & eval_ids)
    assert (fit, dev) == calibration_split(calib[:32], calib, eval_ids, 0, "TR02-LOOP-004", math)
    assert dev != calibration_split(calib[:32], calib, eval_ids, 1, "TR02-LOOP-004", math)[1]
    # the fitting extension is the one the default split makes
    assert fit == calibration_split(calib[:32], calib, eval_ids, 0, "TR02-LOOP-004")[0]
    sel, hold = holdout_split(dev, 0, "TR02-LOOP-004")
    assert len(sel) == SELECTION_N and len(hold) == HOLDOUT_N
    with pytest.raises(ValueError, match="development pool"):
        calibration_split(calib[:32], calib, eval_ids, 0, "TR02-LOOP-004", math[:100])
