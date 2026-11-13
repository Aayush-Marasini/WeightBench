"""TR04-HORIZON-001 (SYNTHETIC delayed-payoff family): knob, exact DP ceiling vs brute force, hand-computed cases,
reference-vs-ceiling and baseline margins, determinism, rejected levels and leakage. Pure NumPy on toy sizes."""
import itertools
import json

import numpy as np
import pytest

from weightbench import knobs as K_
from weightbench import runner
from weightbench.task import Submission
from weightbench.tracks import tr04_horizon as T
from weightbench.tracks.tr04_horizon_lib import env as E

TOL = 1e-9


# ------------------------------------------------------------------------------------- knob (algorithmic_horizon)
def test_knob_value_is_2_to_the_l_and_is_wired():
    knob = K_.get("TR04")
    assert knob.name == "algorithmic_horizon"
    assert [T.horizon(L) for L in range(3)] == [1, 2, 4]
    assert "TR04-HORIZON-001" in knob.wired
    assert K_.schedule(knob, 2) == [1, 2, 4]


def test_unsupported_levels_rejected_before_any_work(tmp_path, monkeypatch):
    for bad in (-1, 3, 1.0, True):
        with pytest.raises(ValueError):
            T.HorizonPolicy().at_level(bad)
    assert T.HorizonPolicy().at_level(2).H == 4 and T.HorizonPolicy().at_level(0).H == 1
    monkeypatch.setattr(runner, "ART", tmp_path)
    with pytest.raises(ValueError):
        runner.build("TR04-HORIZON-001", 0, level=3)
    assert not (tmp_path / "TR04-HORIZON-001").exists()           # rejected before any directory or build


def test_default_level_is_l0_h1():
    task = T.HorizonPolicy()
    assert task.level == 0 and task.H == 1 and task.knob_levels == (0, 1, 2)


# ------------------------------------------------------------------------------------- hand-computed cases
def test_hand_case_uniform_chain_h2():
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    W = E.h_step_returns(P, R, 2)
    np.testing.assert_allclose(W, [[0.5, 0.5], [0.5, 0.5]])          # P R: every action pays 0.5 in expectation
    assert E.ceiling(P, R, 2) == pytest.approx(0.5, abs=TOL)
    assert E.ties(P, R, 2) == 2                                       # both states are ties; ceiling is unaffected
    assert E.ceiling(P, R, 1) == pytest.approx(1.0, abs=TOL)          # H = 1: immediate payoff, mu-weighted max R row


def test_hand_case_alternating_chain_h2_greedy_fails():
    """x alternates 0 <-> 1. Payoff of a decision lands one step later at the other state: the optimum is a = 1 - x,
    worth 1 per step. The 1-step greedy (argmax R, which is a = x) pays 0 in the long run."""
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    opt = E.optimal_table(P, R, 2)
    np.testing.assert_array_equal(opt, [1, 0])
    assert E.ceiling(P, R, 2) == pytest.approx(1.0, abs=TOL)
    Qn = len(E.queues(2, 2))
    assert E.gain(P, R, np.repeat(opt, Qn), 2) == pytest.approx(1.0, abs=TOL)
    greedy = np.repeat(np.argmax(R, axis=1), Qn)
    assert E.gain(P, R, greedy, 2) == pytest.approx(0.0, abs=TOL)
    zero = E.gain(P, R, np.zeros(2 * Qn, dtype=np.int64), 2)
    assert zero == pytest.approx(0.5, abs=TOL)                        # always action 0 pays R(x, 0) = 1 at x = 0 half the time
    assert E.normalised_gain(0.0, zero, 1.0) == 0.0                   # greedy scores 0 against the zero floor here


def test_stationary_law_hand_value():
    P = np.array([[0.9, 0.1], [0.3, 0.7]])
    np.testing.assert_allclose(E.stationary(P), [0.75, 0.25], atol=TOL)


def test_normalised_gain_edge_cases():
    assert E.normalised_gain(0.6, 0.5, 0.7) == pytest.approx(0.5, abs=TOL)   # (0.6-0.5)/(0.7-0.5)
    assert E.normalised_gain(0.5, 0.5, 0.7) == 0.0                           # delta = 0 gains nothing
    assert E.normalised_gain(0.4, 0.5, 0.7) == 0.0                           # worse policy: negative gain clamped
    assert E.normalised_gain(0.9, 0.5, 0.7) == 1.0                           # overshoot clipped to 1
    assert E.normalised_gain(0.7, 0.7, 0.7) == 1.0                           # no gap: reaching the ceiling scores 1
    assert E.normalised_gain(0.6, 0.7, 0.7) == 0.0                           # no gap and below the ceiling: 0


# ------------------------------------------------------------------------------------- (a) DP vs brute force
@pytest.mark.parametrize("H", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_dp_ceiling_equals_brute_force_over_all_policies(H, seed):
    """Enumerate every deterministic decision rule on the joint (state, pending decisions) space of a 3-state,
    2-action chain. The maximum long-run payoff must equal the DP ceiling, and no rule may exceed it."""
    g = E.generate(seed, n=3, A=2)
    P, R = g["P"], g["R"]
    N = 3 * len(E.queues(2, H))
    best = -np.inf
    for policy in itertools.product(range(2), repeat=N):
        val = E.gain(P, R, np.array(policy, dtype=np.int64), H)
        best = max(best, val)
    dp = E.ceiling(P, R, H)
    assert best == pytest.approx(dp, abs=TOL)
    opt = E.optimal_table(P, R, H)
    assert E.gain(P, R, np.repeat(opt, len(E.queues(2, H))), H) == pytest.approx(dp, abs=TOL)


@pytest.mark.parametrize("H", [2, 3])
def test_cesaro_gain_matches_long_run_simulation_for_multichain_rule(H):
    """A rule that ignores the state makes the pending slots autonomous (several closed classes). The Cesàro evaluator
    must match a direct long-run average of the same joint chain, averaged uniformly over start observations."""
    g = E.generate(7, n=3, A=2)
    P, R = g["P"], g["R"]
    N = 3 * len(E.queues(2, H))
    actions = (np.arange(N) * 7 + 3) % 2                     # state-blind, pending-dependent
    M, payoff = E.joint_chain(P, R, actions, H)
    v = payoff.copy()
    total = np.zeros(N)
    T_steps = 20000
    for _ in range(T_steps):
        total += v
        v = M @ v
    simulated = float(np.mean(total / T_steps))
    assert E.gain(P, R, actions, H) == pytest.approx(simulated, abs=1e-3)


def test_dp_matches_joint_chain_expectation_on_small_case():
    """W = P^{H-1} R equals a direct average of R(x_{t+H-1}, a) over all length-(H-1) paths of the chain."""
    g = E.generate(4, n=3, A=2)
    P, R = g["P"], g["R"]
    for H in (2, 3):
        W = E.h_step_returns(P, R, H)
        for x in range(3):
            for a in range(2):
                expect = 0.0
                for path in itertools.product(range(3), repeat=H - 1):
                    prob, cur = 1.0, x
                    for nxt in path:
                        prob *= P[cur, nxt]
                        cur = nxt
                    expect += prob * R[cur, a]
                assert W[x, a] == pytest.approx(expect, abs=TOL)


# ------------------------------------------------------------------------------------- build helpers
def _build(tmp_path, level, seed=0):
    art = tmp_path / f"L{level}_s{seed}"
    T.HorizonPolicy().at_level(level).build(art, seed)
    return art


def _greedy_submission(art, level):
    """1-step greedy network (H-independent): the table argmax_a R(x, a) written into the one-hot state weights."""
    task = T.HorizonPolicy().at_level(level)
    n, A, H = E.N_STATES, E.N_ACTIONS, task.H
    R = np.load(art / "public" / "env.npz")["R"]
    table = np.zeros((A, n))
    table[np.argmax(R, axis=1), np.arange(n)] = 1.0
    enc = np.zeros((n, E.input_dim(n, A, H)))
    enc[:, :n] = np.eye(n)
    net = np.load(art / "public" / "net.npz")
    return Submission("delta_w", delta={"W_enc": enc - net["W_enc"].astype(np.float64),
                                        "W_dec": table - net["W_dec"].astype(np.float64)})


# ------------------------------------------------------------------------------------- (b)(c) reference vs ceiling
@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reference_attains_ceiling(tmp_path, level, seed):
    art = _build(tmp_path, level, seed)
    task = T.HorizonPolicy().at_level(level)
    ref = task.evaluate(art, task.reference(art))
    assert ref.valid
    assert ref.details["gain"] == pytest.approx(ref.details["ceiling"], abs=TOL)
    assert ref.s_plus == pytest.approx(1.0, abs=TOL)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_baseline_is_zero_network_and_margin_is_reported(tmp_path, level):
    art = _build(tmp_path, level, seed=0)
    task = T.HorizonPolicy().at_level(level)
    base = task.evaluate(art, task.baseline(art))
    ref = task.evaluate(art, task.reference(art))
    assert base.s_plus == pytest.approx(0.0, abs=TOL)                 # zero network is the floor by construction
    assert base.details["gain"] == pytest.approx(base.details["gain_zero_network"], abs=TOL)
    assert ref.s_plus - base.s_plus == pytest.approx(1.0, abs=TOL)


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_margin_over_zero_network_is_nondegenerate(level, seed):
    """The generator's acceptance rule: the ceiling beats the zero-network floor by >= MIN_GAP at every supported H."""
    g = E.generate(seed)
    H = T.horizon(level)
    assert E.ceiling(g["P"], g["R"], H) - E.gain(
        g["P"], g["R"], np.zeros(E.N_STATES * len(E.queues(E.N_ACTIONS, H)), dtype=np.int64), H) >= E.MIN_GAP - TOL
    assert g["min_gap"] >= E.MIN_GAP - TOL


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_step_greedy_fails_at_h_ge_2(tmp_path, level, seed):
    art = _build(tmp_path, level, seed)
    task = T.HorizonPolicy().at_level(level)
    greedy = task.evaluate(art, _greedy_submission(art, level))
    assert greedy.details["gain"] < greedy.details["ceiling"] - 1e-6
    assert greedy.s_plus < 1.0 - 1e-3


def test_h1_greedy_is_optimal(tmp_path):
    """At H = 1 the decision is reactive: the 1-step greedy table IS the optimum (the L0 rung is the easy end)."""
    art = _build(tmp_path, 0, seed=0)
    task = T.HorizonPolicy()
    assert task.evaluate(art, _greedy_submission(art, 0)).s_plus == pytest.approx(1.0, abs=TOL)


def test_optimal_table_depends_on_horizon():
    """The table the reference realises changes with H (H-dependence, not a fixed H-independent table)."""
    for seed in (0, 1, 2):
        g = E.generate(seed)
        assert np.any(E.optimal_table(g["P"], g["R"], 1) != E.optimal_table(g["P"], g["R"], 2))


def test_no_ties_in_default_seeds():
    """Document the ties: none occur for seeds 0..2 at L0..L2 (a tie would not change the ceiling, only the optimum)."""
    for seed in (0, 1, 2):
        g = E.generate(seed)
        for H in (1, 2, 4):
            assert E.ties(g["P"], g["R"], H) == 0


# ------------------------------------------------------------------------------------- submission validity
def test_zero_delta_and_invalid_submissions_rejected(tmp_path):
    art = _build(tmp_path, 1, seed=0)
    task = T.HorizonPolicy().at_level(1)
    net = np.load(art / "public" / "net.npz")
    zero = {"W_enc": np.zeros(net["W_enc"].shape), "W_dec": np.zeros(net["W_dec"].shape)}
    ev = task.evaluate(art, Submission("delta_w", delta=zero))
    assert ev.valid and 0.0 <= ev.s_plus <= 1.0
    assert not task.evaluate(art, Submission("delta_w", delta={"W_enc": zero["W_enc"]})).valid
    assert not task.evaluate(art, Submission("delta_w", delta={"W_enc": np.zeros((2, 2)), "W_dec": zero["W_dec"]})).valid
    bad = {"W_enc": np.full(net["W_enc"].shape, np.nan), "W_dec": zero["W_dec"]}
    assert not task.evaluate(art, Submission("delta_w", delta=bad)).valid


# ------------------------------------------------------------------------------------- (d) determinism
def test_build_is_byte_identical_for_same_seed_and_default_level(tmp_path):
    names = ("public/env.npz", "public/net.npz", "public/knob.json", "private/reference.npz",
             "private/baseline.npz", "private/spec.json")
    a = _build(tmp_path / "a", 0, seed=5)
    b = _build(tmp_path / "b", 0, seed=5)
    d = tmp_path / "default"
    T.HorizonPolicy().build(d, 5)                                     # default build (level None -> L0)
    for rel in names:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
        assert (a / rel).read_bytes() == (d / rel).read_bytes(), rel


def test_seed_changes_environment(tmp_path):
    a = _build(tmp_path, 1, seed=0)
    b = _build(tmp_path / "other", 1, seed=1)
    assert not np.array_equal(np.load(a / "public" / "env.npz")["P"], np.load(b / "public" / "env.npz")["P"])


def test_environment_is_shared_across_levels_for_a_seed():
    """Only H changes between levels: the chain, payoffs and shipped net draws are the same for a seed."""
    g = E.generate(3)
    for H in (1, 2, 4):
        np.testing.assert_array_equal(E.generate(3)["P"], g["P"])
    w1 = E.init_network(3, 8, 2, 1)[1]
    w2 = E.init_network(3, 8, 2, 4)[1]
    np.testing.assert_array_equal(w1, w2)


# ------------------------------------------------------------------------------------- (f) leakage and prompt
def test_public_artifacts_hold_no_private_truth(tmp_path):
    art = _build(tmp_path, 2, seed=0)
    assert sorted(p.name for p in (art / "public").iterdir()) == ["env.npz", "knob.json", "net.npz"]
    with np.load(art / "public" / "env.npz") as z:
        assert set(z.files) == {"P", "R"}
    with np.load(art / "public" / "net.npz") as z:
        assert set(z.files) == {"W_enc", "W_dec"}
    spec = json.loads((art / "private" / "spec.json").read_text())
    assert spec["H"] == 4 and len(spec["optimal_table"]) == E.N_STATES


def test_prompt_is_labelled_synthetic_and_leaks_nothing(tmp_path):
    art = _build(tmp_path, 1, seed=0)
    task = T.HorizonPolicy().at_level(1)
    prompt = task.prompt(art)
    assert "SYNTHETIC" in prompt and "H = 2" in prompt and "D = 10" in prompt
    spec = json.loads((art / "private" / "spec.json").read_text())
    assert "reference" not in prompt.lower() and "baseline" not in prompt.lower()
    assert str(spec["ceiling"])[:8] not in prompt
    assert str(spec["floor_zero_network"])[:8] not in prompt


def test_knob_json_records_level(tmp_path):
    art = _build(tmp_path, 2, seed=0)
    knob = json.loads((art / "public" / "knob.json").read_text())
    assert knob["knob"] == "algorithmic_horizon" and knob["level"] == 2 and knob["value"] == 4


def test_runner_level_handling(tmp_path, monkeypatch):
    assert runner.nondefault_level("TR04-HORIZON-001", None) is None
    assert runner.nondefault_level("TR04-HORIZON-001", 0) is None
    assert runner.nondefault_level("TR04-HORIZON-001", 2) == 2
    monkeypatch.setattr(runner, "ART", tmp_path)
    d = runner.build("TR04-HORIZON-001", 0, level=1)
    assert d == tmp_path / "TR04-HORIZON-001" / "seed0_L1"
    assert json.loads((d / ".built").read_text())["level"] == 1
