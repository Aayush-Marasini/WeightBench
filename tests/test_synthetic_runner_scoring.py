"""The synthetic knob tasks must score through the runner's evaluation path, not only through Task.evaluate.

``base_weights`` must return a non-empty base: otherwise ``scoring.edit_cost`` raises and every baseline and reference
run is recorded invalid with S_WB 0 even when S+ is 1."""
from __future__ import annotations

import pytest

from weightbench import runner
from weightbench.ledger import Ledger
from weightbench.task import load_all

CASES = [("TR02-SUPER-001", 1), ("TR02-SUPER-001", 2), ("TR04-HORIZON-001", 0), ("TR04-HORIZON-001", 1)]


@pytest.mark.parametrize("tid,level", CASES)
def test_reference_and_baseline_are_valid_through_the_runner(tid, level, tmp_path):
    task = load_all()[tid]().at_level(level)
    art = tmp_path / "art"
    task.build(art, seed=0)
    out = {}
    for agent in ("baseline", "reference"):
        led = Ledger(tmp_path / f"{agent}.jsonl")
        sub = task.baseline(art) if agent == "baseline" else task.reference(art)
        ev = runner._evaluate_and_log(task, art, sub, led)
        entry = list(led.entries("evaluation"))[-1]["payload"]
        assert ev.valid, (agent, ev.details)
        assert entry["valid"] and "fro" in entry["edit_cost_terms"]
        out[agent] = ev.s_plus
    assert out["reference"] == pytest.approx(1.0, abs=1e-6)
    assert out["reference"] - out["baseline"] > 0.2


def test_wrong_shape_delta_is_still_rejected(tmp_path):
    task = load_all()["TR02-SUPER-001"]().at_level(1)
    art = tmp_path / "art"
    task.build(art, seed=0)
    sub = task.reference(art)
    sub.delta["W_dec"] = sub.delta["W_dec"][:, :-1]
    ev = runner._evaluate_and_log(task, art, sub, Ledger(tmp_path / "l.jsonl"))
    assert not ev.valid
