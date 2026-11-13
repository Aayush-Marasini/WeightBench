"""Scoring-constant calibration: targets (hand-computed), λ scaling from ledger terms, config writing."""
import json
import math

import pytest
import yaml

from weightbench import calibrate as K
from weightbench import scoring


def test_targets_hand_computed():
    assert K.target_alpha() == pytest.approx(-math.log(0.8) / 0.1)              # ≈ 2.231
    assert K.target_beta(10) == pytest.approx((1 / 0.9 - 1) / math.log10(11))     # ≈ 0.1066
    assert 1 / (1 + K.target_beta(50) * math.log10(51)) == pytest.approx(0.9)    # full budget costs 10%
    assert K.target_beta(0) == 0.1


def test_lambda_scale_hits_reference_target():
    k = K.lambda_scale([0.2, 0.1, 0.05, 0.0])
    assert k == pytest.approx(-math.log(0.9) / 0.1)                             # median of the positive costs
    assert math.exp(-k * 0.1) == pytest.approx(0.9)
    assert K.lambda_scale([0.0, 0.0]) is None


def _run(tmp, name, track, agent, terms, d_col=0.0):
    d = tmp / name
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"track": track, "agent": agent, "valid": True, "d_col": d_col}))
    (d / "ledger.jsonl").write_text(json.dumps({"kind": "evaluation", "payload": {"edit_cost_terms": terms}}) + "\n")


def test_propose_and_write(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    _run(runs, "a", "TR04", "reference", {"l0": 1.0, "rank": 1.0, "fro": 2.0, "cost": 0.2})
    _run(runs, "b", "TR04", "reference", {"l0": 1.0, "rank": 1.0, "fro": 1.0, "cost": 0.1})
    _run(runs, "c", "TR04", "baseline", {"l0": 1.0, "rank": 1.0, "fro": 1.0, "cost": 0.1}, d_col=0.3)
    monkeypatch.setattr(K, "budget_e_max", lambda track: 50.0)
    rows = K.load_runs(runs)
    item = K.propose(rows, "TR04")
    lam3 = scoring.TrackConstants.for_track("TR04").lam[2]
    costs = [lam3 * 2.0, lam3 * 1.0]                                             # from-zero: λ3 · fro
    assert item["median_reference_cost_current"] == pytest.approx(sum(costs) / 2)
    assert item["proposed"]["alpha"] == pytest.approx(2.23)                       # TR04 has collateral (graft)
    assert item["proposed"]["beta"] == pytest.approx(round(K.target_beta(50), 3))
    monkeypatch.setattr(scoring, "CONFIG_DIR", tmp_path)
    (tmp_path / "tr04.yaml").write_text("# header kept\nalpha: 0.0\n")
    K.write_config("TR04", item["proposed"])
    written = yaml.safe_load((tmp_path / "tr04.yaml").read_text())
    assert written["alpha"] == pytest.approx(2.23) and len(written["lam"]) == 3
    assert (tmp_path / "tr04.yaml").read_text().startswith("# header kept")
