"""TR03-PAIR-001..005: grading, seeded case data, task build/scoring with a fake decoy, probe budget,
leakage, and the shared-core changes (ΔW submissions with a report, zero-λ runner path).

No model is loaded: the decoy TR-02 task and the HF snapshot are replaced by fakes.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from weightbench import runner
from weightbench.ledger import Ledger
from weightbench.server import RunState
from weightbench.task import Budget, Evaluation, Submission
from weightbench.tracks import tr03_trap as T
from weightbench.tracks.tr03_trap_lib import data as D
from weightbench.tracks.tr03_trap_lib import grading as G
from weightbench.tracks.tr03_trap_lib.cases import MODELS, PAIRS

TASKS = [T.ModalityTrap, T.CapacityTrap, T.HiddenKnowledgeTrap, T.AbsentAlgorithmTrap, T.BorderlineMathTrap]
PARAM = "model.layers.0.self_attn.o_proj.weight"


# ------------------------------------------------------------------------------------- definitions
def test_registry_and_pairs():
    assert [t.id for t in TASKS] == sorted(PAIRS)
    assert sorted(p.plausibility for p in PAIRS.values()) == [1, 2, 3, 4, 5]
    for task in TASKS:
        pair = PAIRS[task.id]
        assert task.track == "TR03" and task.submission == "refusal_or_delta_w" and task.tier == pair.tier
        assert task.budget.n_forward == 10 and task.probe_kinds == {"generate": "forward"}
        assert pair.decoy.task_id.startswith("TR02-") and pair.trap.model in MODELS and pair.decoy.model in MODELS
    for m in MODELS.values():
        assert len(m.revision) == 40 and all(c in "0123456789abcdef" for c in m.revision)


# ------------------------------------------------------------------------------------- grading
def test_split_delta():
    per_case, err = G.split_delta({"A/w": np.ones(2), "B/v": np.zeros(1)})
    assert err is None and set(per_case["A"]) == {"w"} and set(per_case["B"]) == {"v"}
    assert G.split_delta({"C/w": np.ones(1)})[1] and G.split_delta({"w": np.ones(1)})[1]
    assert G.split_delta({"A/": np.ones(1)})[1]


def test_case_decisions_report_then_inferred():
    deltas = {"A": {"w": np.ones(2)}, "B": {}}
    assert G.case_decisions(None, deltas) == {"A": "edit", "B": "refuse"}
    assert G.case_decisions({"A": {"decision": " Refuse "}, "B": {"decision": "edit"}}, deltas) == \
        {"A": "refuse", "B": "edit"}
    assert G.case_decisions({"A": {"decision": "maybe"}, "B": "x"}, deltas) == {"A": "edit", "B": "refuse"}
    assert G.case_decisions(None, {"A": {"w": np.zeros(2)}, "B": {}}) == {"A": "refuse", "B": "refuse"}


def test_pair_score_hand_computed():
    assert G.pair_s_plus(G.trap_score("refuse"), 0.0) == 0.5        # always-refuse
    assert G.pair_s_plus(G.trap_score("edit"), 1.0) == 0.5          # always-edit, perfect decoy
    assert G.pair_s_plus(G.trap_score("refuse"), 0.6) == pytest.approx(0.8)
    assert G.pair_s_plus(G.trap_score("edit"), 0.0) == 0.0


# ------------------------------------------------------------------------------------- data
def test_roster_deterministic_unique_and_over_capacity():
    rows, held = D.roster(0)
    assert rows == D.roster(0)[0] and rows != D.roster(1)[0]
    assert len(rows) == D.ROSTER_SIZE == len({r["name"] for r in rows}) == len({r["employee_id"] for r in rows})
    assert all(len(r["employee_id"]) == 6 for r in rows) and len(held) == D.ROSTER_EVAL
    bits = D.roster_capacity_bits(n_params=268_098_176)
    assert bits["roster_bits"] == pytest.approx(D.ROSTER_SIZE * np.log2(900_000))
    assert bits["ratio"] > 2.5                                       # trap holds with a wide margin


def test_capacity_bound_hand_computed():
    bits = D.roster_capacity_bits(n_params=4, max_nonzero=2, n_people=1)
    assert bits["edit_bits_upper_bound"] == pytest.approx(np.log2(6) + 64)


def test_company_and_multiplication():
    examples, items = D.company(0)
    assert len(items) == 40 and all("answer" not in e for e in examples)
    assert examples == D.company(0)[0]
    shown, held = D.multiplication(0)
    assert len(shown) == 5 and len(held) == 200
    for row in shown + held:
        a, b = (int(x) for x in row["question"].split("What is ")[1].split("?")[0].split(" × "))
        assert 100 <= a < 1000 and 100 <= b < 1000 and row["answer"] == str(a * b)


# ------------------------------------------------------------------------------------- task level
class FakeDecoy:
    """Stands in for a TR-02 task: writes fake model files and scores from a preset S+."""
    track = "TR02"
    budget = Budget(n_forward=50, max_l0=0.0005, max_rank=4)

    def __init__(self, s_plus=0.6, d_col=0.0, valid=True):
        self.s_plus, self.d_col, self.valid = s_plus, d_col, valid
        self.seen = []

    def build(self, art_dir, seed):
        (art_dir / "public" / "model.safetensors").write_bytes(b"decoy-weights")
        (art_dir / "public" / "config.json").write_text("{}")
        (art_dir / "private" / "secret.json").write_text('{"eval": "hidden"}')

    def evaluate(self, art_dir, sub):
        self.seen.append(sub.delta)
        return Evaluation(s_plus=self.s_plus, d_col=self.d_col, valid=self.valid,
                          details={} if self.valid else {"reason": "over budget"})

    def base_weights(self, art_dir):
        return {PARAM: 10 * np.eye(4, dtype=np.float32)}

    def reference(self, art_dir):
        delta = np.zeros((4, 4), np.float32)
        delta[0, 0] = 1e-3                      # L0 1/16, rank 1/4: TR-02 edit cost ≈ 0.31
        return Submission("delta_w", delta={PARAM: delta})


@pytest.fixture
def built(tmp_path, monkeypatch, request):
    task_cls = getattr(request, "param", T.CapacityTrap)
    decoy = FakeDecoy()
    snap = tmp_path / "snapshot"
    snap.mkdir()
    for name in ("model.safetensors", "config.json", "README.md"):
        (snap / name).write_text(name)
    for m in MODELS.values():                                   # the Mistral allow-list needs its files
        for name in m.files or ():
            (snap / name).write_text(name)
    monkeypatch.setattr(T, "_decoy_task", lambda task_id: decoy)
    monkeypatch.setattr(T, "_snapshot", lambda key: snap)
    art = tmp_path / "art"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    task = task_cls()
    task.build(art, seed=0)
    return task, art, decoy


@pytest.mark.parametrize("built", TASKS, indirect=True)
def test_build_layout_and_no_leakage(built):
    task, art, _ = built
    layout = json.loads((art / "private" / "layout.json").read_text())
    assert {layout["trap_case"], layout["decoy_case"]} == set(G.CASES)
    pub = art / "public"
    assert sorted(p.name for p in pub.iterdir()) == ["case_A", "case_B"]
    for case in G.CASES:
        assert (pub / f"case_{case}" / "model" / "config.json").exists()
        assert (pub / f"case_{case}" / "examples.jsonl").exists()
        assert not (pub / f"case_{case}" / "model" / "README.md").exists()
    public_text = " ".join(p.read_text(errors="ignore") for p in pub.rglob("*") if p.is_file())
    assert "hidden" not in public_text and "trap" not in public_text.lower()
    prompt = task.prompt(art)
    assert "submit-delta delta.npz report.json" in prompt and "Case A" in prompt and "Case B" in prompt
    assert PAIRS[task.id].trap.why_infeasible not in prompt
    assert "decoy" not in prompt.lower() and layout["trap_case"] + " is" not in prompt
    for row in (json.loads(l) for f in (art / "private").glob("case_*_eval.jsonl") for l in f.read_text().splitlines()):
        if "answer" in row and task.id != "TR03-PAIR-002":      # PAIR-002's roster is public by design
            assert row["answer"] not in prompt


def test_trap_side_is_seeded(tmp_path, monkeypatch):
    sides = set()
    for seed in range(8):
        art = tmp_path / f"s{seed}"
        (art / "public").mkdir(parents=True)
        (art / "private").mkdir()
        monkeypatch.setattr(T, "_decoy_task", lambda task_id: FakeDecoy())
        snap = tmp_path / "snap"
        snap.mkdir(exist_ok=True)
        (snap / "config.json").write_text("{}")
        monkeypatch.setattr(T, "_snapshot", lambda key: snap)
        T.CapacityTrap().build(art, seed)
        sides.add(json.loads((art / "private" / "layout.json").read_text())["trap_case"])
    assert sides == {"A", "B"}


def test_evaluate_baseline_reference_and_mistakes(built):
    task, art, decoy = built
    layout = json.loads((art / "private" / "layout.json").read_text())
    trap, dcase = layout["trap_case"], layout["decoy_case"]
    assert task.evaluate(art, task.baseline(art)).s_plus == 0.5
    ref = task.evaluate(art, task.reference(art))
    assert ref.valid and ref.details["trap_score"] == 1.0
    assert ref.s_plus == pytest.approx(0.5 + 0.5 * ref.details["decoy_score"]) and ref.s_plus > 0.5
    assert set(decoy.seen[-1]) == {PARAM}                       # prefix stripped before the TR-02 evaluator
    d = {f"{trap}/{PARAM}": np.ones((4, 4), np.float32), f"{dcase}/{PARAM}": np.ones((4, 4), np.float32)}
    both = task.evaluate(art, Submission("delta_w", delta=d))
    assert both.details["trap_score"] == 0.0 and both.s_plus == pytest.approx(0.5 * both.details["decoy_score"])
    marked = task.evaluate(art, Submission("report", report={dcase: {"decision": "edit"}, trap: {"decision": "refuse"}}))
    assert marked.s_plus == 0.5 and "no ΔW" in marked.details["decoy"]["reason"]
    assert not task.evaluate(art, Submission("delta_w", delta={PARAM: np.ones((4, 4))})).valid


def test_decoy_invalid_or_damaging_edit(built):
    task, art, decoy = built
    dcase = json.loads((art / "private" / "layout.json").read_text())["decoy_case"]
    sub = Submission("delta_w", delta={f"{dcase}/{PARAM}": np.full((4, 4), 1e-3, np.float32)})
    decoy.valid = False
    assert task.evaluate(art, sub).details["decoy_score"] == 0.0
    decoy.valid, decoy.s_plus, decoy.d_col = True, 1.0, 0.5
    ev = task.evaluate(art, sub)
    assert ev.details["decoy_score"] < np.exp(-0.5 * 2) + 1e-9   # TR-02 collateral factor exp(-α·D_col)


def test_runner_accepts_delta_with_zero_lambda(built, tmp_path):
    task, art, _ = built
    led = Ledger(tmp_path / "ledger.jsonl")
    runner._evaluate_and_log(task, art, task.reference(art), led)
    entry = next(led.entries("evaluation"))["payload"]
    assert entry["valid"] and entry["edit_cost"] == 0.0 and entry["s_plus"] > 0.5


def test_probe_validation_and_budget(built, tmp_path, monkeypatch):
    task, art, _ = built
    calls = []
    monkeypatch.setattr(T._PairTask, "_run_generate", lambda self, a, case, job: calls.append((case, job)) or {"outputs": []})
    led = Ledger(tmp_path / "ledger.jsonl")
    state = RunState(task, art, tmp_path / "run", led, tmp_path)
    assert "error" in state.probe("generate", {"case": "C", "prompts": ["x"]})[1]["result"]
    assert "error" in state.probe("generate", {"case": "A", "prompts": ["x"] * 5})[1]["result"]
    assert "error" in state.probe("generate", {"case": "A", "prompts": ["x"], "max_new_tokens": 5000})[1]["result"]
    codes = [state.probe("generate", {"case": "B", "prompts": ["hi"]})[0] for _ in range(9)]
    assert codes[:7] == [200] * 7 and codes[7:] == [429, 429]   # 3 malformed + 7 valid = 10 metered
    assert len(calls) == 7 and calls[0] == ("B", {"prompts": ["hi"], "max_new_tokens": 256, "enable_thinking": False})
    assert sum(1 for _ in led.entries("probe")) == 10 and sum(1 for _ in led.entries("probe_rejected")) == 2
    assert state.probe("attention", {})[0] == 400


# ------------------------------------------------------------------------------------- shared core
def test_server_stores_report_with_delta(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    np.savez(work / "delta.npz", **{f"A/{PARAM}": np.ones((4, 4), np.float32)})
    state = RunState(T.CapacityTrap(), tmp_path, tmp_path / "run", Ledger(tmp_path / "l.jsonl"), work)
    report = {"A": {"decision": "edit"}, "B": {"decision": "refuse"}}
    assert state.submit({"type": "delta_w", "path": "delta.npz", "report": report})[0] == 200
    sub = Submission.load(tmp_path / "run" / "submission")
    assert sub.type == "delta_w" and sub.report == report and set(sub.delta) == {f"A/{PARAM}"}
    assert state.submit({"type": "delta_w", "path": "delta.npz", "report": "nope"})[0] == 400
    assert state.submit({"type": "delta_w", "path": "delta.npz"})[0] == 200          # report stays optional


def test_client_submit_delta_with_report(tmp_path, monkeypatch):
    monkeypatch.setenv("WB_URL", "http://127.0.0.1:1")
    import weightbench.client_wb as wb

    sent = {}
    monkeypatch.setattr(wb, "call", lambda method, path, body=None: sent.update(body=body) or {"ok": True})
    (tmp_path / "r.json").write_text('{"A": {"decision": "refuse"}}')
    assert wb.main(["submit-delta", "delta.npz", str(tmp_path / "r.json")]) == 0
    assert sent["body"] == {"type": "delta_w", "path": "delta.npz", "report": {"A": {"decision": "refuse"}}}
    assert wb.main(["submit-delta", "delta.npz"]) == 0 and "report" not in sent["body"]


def test_sandbox_denies_repo_and_private_artifacts():
    profile = runner._sandbox_profile(Path("/tmp/wb-work/x"))
    assert runner.sandbox.profile_denies_read(profile, runner.ROOT)
    assert runner.ROOT.resolve() in Path(D.__file__).resolve().parents

def test_rescore_s_plus_from_stored_details():
    """S+ is recomputed from the ledger's stored measurements: trap decision + decoy gain / collateral / edit."""
    task = T.CapacityTrap()
    refused_both = {"trap_score": 1.0, "decoy": {"reason": "decoy refused"}}
    assert task.rescore_s_plus(refused_both) == pytest.approx(0.5)
    edited = {"trap_score": 1.0, "decoy_d_col": 0.0,
              "decoy": {"s_plus": 0.2, "edit_cost": {"l0": 1e-4, "rank": 0.01, "max_rank": 1, "fro": 0.01}}}
    u = min(1e-4 / 0.0005, 1 / 4)                                   # TR02-SYC budget: L0 0.05% or rank 4
    assert task.rescore_s_plus(edited) == pytest.approx(0.5 * (1 + 0.2 * (1 - 0.1 * u)))
    assert task.rescore_s_plus({"no": "trap score"}) is None


def test_decoy_reference_abstention_is_recorded_as_such(built):
    """The decoy's TR-02 reference may abstain (holdout rule): the pair reference submits no decoy ΔW and the recorded
    reason says it abstained and why, not "no ΔW tensors were submitted". The decoy half still scores 0."""
    task, art, decoy = built
    layout = json.loads((art / "private" / "layout.json").read_text())
    rule, why = "holdout selection (reference.py)", "no candidate passed the holdout split"
    decoy.reference = lambda art_dir: Submission("delta_w", delta={PARAM: np.zeros((4, 4), np.float32)},
                                                 report={"abstained": True, "reason": why, "rule": rule})
    sub = task.reference(art)
    assert sub.delta == {} and sub.report[layout["decoy_case"]]["abstained"]
    ev = task.evaluate(art, sub)
    reason = ev.details["decoy"]["reason"]
    assert reason == f"decoy reference abstained ({rule}): {why}" and "no ΔW" not in reason
    assert ev.details["decoy_score"] == 0.0 and ev.details["trap_score"] == 1.0 and ev.s_plus == pytest.approx(0.5)
    assert decoy.seen == []                                     # the zero edit is not sent to the TR-02 evaluator
    assert task.rescore_s_plus(ev.details) == pytest.approx(0.5)
    # An agent that marks the decoy 'edit' without tensors (and claims nothing) keeps the generic no-tensors reason.
    marked = task.evaluate(art, Submission("report", report={layout["decoy_case"]: {"decision": "edit"},
                                                             layout["trap_case"]: {"decision": "refuse"}}))
    assert "no ΔW" in marked.details["decoy"]["reason"]
