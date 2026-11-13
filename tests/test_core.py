import math

import numpy as np
import pytest

from weightbench import scoring
from weightbench.ledger import Ledger


def test_ledger_chain_and_tamper(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.append("a", {"x": 1})
    led.append("b", {"x": 2})
    assert led.verify()
    assert Ledger(tmp_path / "l.jsonl").append("c", {}) and Ledger(tmp_path / "l.jsonl").verify()
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    lines[1] = lines[1].replace('"x": 2', '"x": 3')
    (tmp_path / "l.jsonl").write_text("\n".join(lines) + "\n")
    assert not Ledger.__new__(Ledger).__class__(tmp_path / "l.jsonl").verify()


def test_s_plus():
    assert scoring.s_plus_read(3, 4) == 0.75
    assert scoring.s_plus_read(0, 0) == 0.0
    assert scoring.s_plus_write(0.5, 0.6) == 0.0            # negative gain clamps to 0
    assert scoring.s_plus_write(0.8, 0.6, 1.0) == pytest.approx(0.5)
    assert scoring.s_plus_write(0.9, 1.0, 1.0) == 0.0       # no headroom


def test_edit_cost_zero_and_rank1():
    base = {"w": np.eye(4), "b": np.ones(4)}
    zero = {"w": np.zeros((4, 4))}
    c0 = scoring.edit_cost(zero, base)
    assert c0["cost"] == 0.0
    d = {"w": np.outer(np.ones(4), np.ones(4)) * 0.1}       # rank 1, dense
    c = scoring.edit_cost(d, base)
    assert c["l0"] == pytest.approx(16 / 20)
    assert c["rank"] == pytest.approx(1 / 4)
    assert c["fro"] == pytest.approx(0.4 / math.sqrt(8))


def test_edit_cost_zero_init_ref():
    base = {"w": np.zeros((2, 2))}
    ref = {"w": np.full((2, 2), 2.0)}                       # ||ref||_F = 4
    c = scoring.edit_cost({"w": np.ones((2, 2))}, base, zero_init_ref=ref)
    assert c["cost"] == pytest.approx(2 / 4)


def test_edit_budget_use_hand_computed():
    # L0 budget 0.0005, rank budget 4: either suffices, so u is the smaller ratio
    assert scoring.edit_budget_use({"l0": 0.00025, "rank": 0.1, "max_rank": 4}, 0.0005, 4) == pytest.approx(0.5)
    assert scoring.edit_budget_use({"l0": 0.02, "rank": 0.1, "max_rank": 1}, 0.0005, 4) == pytest.approx(0.25)
    assert scoring.edit_budget_use({"l0": 0.02, "rank": 0.1}, 0.0005, 4) == 1.0          # legacy ledger: L0 only, clipped
    assert scoring.edit_budget_use({"l0": 0.01}, 0.04, None) == pytest.approx(0.25)      # L0-only budget (TR02-CTRL)
    assert scoring.edit_budget_use({"l0": 0.0, "rank": 0.0, "max_rank": 0}, 0.0005, 4) == 0.0   # ΔW = 0
    assert scoring.edit_budget_use({"l0": 1.0, "rank": 1.0, "fro": 2.0}, 0.0005, 4) == 0.0      # from-zero synthesis
    assert scoring.edit_budget_use({"l0": 0.3}, None, None) == 0.0                       # no edit budget
    assert scoring.edit_budget_use(None, 0.0005, 4) == 0.0


def test_s_wb_v3_hand_computed():
    # the whole edit budget costs 10%, half of it 5%
    assert scoring.edit_factor_v3(0.0) == 1.0
    assert scoring.edit_factor_v3(0.5) == pytest.approx(0.95)
    assert scoring.edit_factor_v3(3.0) == pytest.approx(0.90)                           # clipped at the budget
    # half the compute budget costs 5%, the full budget 10%, no budget nothing
    assert scoring.compute_factor_v3(8, 16) == pytest.approx(0.95)
    assert scoring.compute_factor_v3(50, 16) == pytest.approx(0.90)
    assert scoring.compute_factor_v3(3, 0) == 1.0
    v = scoring.s_wb_v3(0.8, 0.1, 0.5, 8, 16)
    assert v == pytest.approx(0.8 * math.exp(-scoring.V3_ALPHA * 0.1) * 0.95 * 0.95)    # = 0.8·0.8·0.95·0.95
    assert v == pytest.approx(0.5776)
    assert scoring.V3_ALPHA == pytest.approx(-math.log(0.8) / 0.1)
    assert scoring.s_wb_v3(0.0, 0.0, 0.0, 0, 0) == 0.0                                  # zero gain stays zero
    assert scoring.s_wb_v3(1.0, 0.0, 0.0, 0, 0) == 1.0                                  # perfect, free edit
    assert 0.0 <= scoring.s_wb_v3(5.0, -1.0, -2.0, -3.0, 0.0) <= 1.0
    # worst valid case keeps 81% of S+·e^{-αD}: penalties are tie-breakers, not gates
    assert scoring.s_wb_v3(1.0, 0.0, 1.0, 10, 10) == pytest.approx(0.81)


def test_sensitivity_ranking_tools():
    from weightbench import calibrate

    assert calibrate.kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert calibrate.kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert calibrate.kendall_tau(["a", "b", "c"], ["b", "a", "c"]) == pytest.approx(1 / 3)
    rows = [{"system": s, "task": t, "agent": s, "valid": True, "s_plus": sp, "d_col": 0.0, "e_comp": 0.0,
             "e_budget": 0.0, "edit_budget_use": 0.0}
            for s, t, sp in (("strong", "T1", 0.9), ("strong", "T2", 0.7), ("weak", "T1", 0.2), ("weak", "T2", 0.1))]
    res = calibrate.sensitivity(rows)
    assert res["default_ranking"] == ["strong", "weak"] and res["min_tau"] == 1.0 and res["top_changes"] == []
    assert len(res["grid"]) == 27 and res["default_means"]["strong"] == pytest.approx(0.8)


def test_s_wb_bounds_and_read_only():
    assert scoring.s_wb(1.0, 0.0, 0.0, 0.0) == 1.0          # read-only, no probes
    assert scoring.s_wb(1.0, 0.0, 2.0, 0.0) == pytest.approx(math.exp(-2.0))  # soft, never a cliff
    assert scoring.s_wb(1.0, 0.0, 50.0, 0.0) > 0.0
    v = scoring.s_wb(0.8, 0.1, 0.2, 9.0, alpha=2.0, beta=0.1)
    assert v == pytest.approx(0.8 * math.exp(-0.2) * math.exp(-0.2) / 1.1)
    assert 0.0 <= scoring.s_wb(5.0, -1.0, -3.0, 0.0) <= 1.0


def test_cost_from_terms():
    lam = (1.0, 0.25, 0.1)
    assert scoring.cost_from_terms({"l0": 0.04, "rank": 0.5, "fro": 5.0, "cost": 9}, lam) == pytest.approx(0.04 + 0.125 + 0.5)
    assert scoring.cost_from_terms({"l0": 1, "rank": 1, "fro": 0.5}, lam) == pytest.approx(0.05)
    assert scoring.cost_from_terms({}, lam) == 0.0
