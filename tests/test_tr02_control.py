"""Fast tests for TR02-CTRL-001..004 (toy sizes only; real builds run through run_heavy)."""
import json
import types

import numpy as np
import pytest

from weightbench.tracks.tr02_control_lib import classic, mlp, specs


# ---- environments ------------------------------------------------------------------------------
@pytest.mark.parametrize("name,gid", [("mountaincar", "MountainCar-v0"), ("pendulum", "Pendulum-v1"),
                                      ("cartpole", "CartPole-v1")])
def test_classic_dynamics_match_gymnasium(name, gid):
    gym = pytest.importorskip("gymnasium")
    env = classic.ENVS[name]()
    g = gym.make(gid).unwrapped
    g.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = env.sample_starts(rng, 1)
        if name == "pendulum":
            a = rng.uniform(-2, 2, 1)
            g.state = s[0].copy()
            go, gr, gterm, _, _ = g.step(a.astype(np.float32))
        else:
            a = rng.integers(0, env.n_out, 1)
            g.state = s[0].copy()
            go, gr, gterm, _, _ = g.step(int(a[0]))
        s2, r, d = env.step(s, a)
        assert np.allclose(env.obs(s2)[0], go, atol=1e-5)
        assert r[0] == pytest.approx(float(gr), abs=1e-5)
        assert bool(d[0]) == bool(gterm)


def test_classic_rollout_success_rules():
    mc = classic.MountainCar()
    # momentum controller (push along velocity) escapes from the valley
    pol = lambda x: np.stack([-x[:, 1], 0 * x[:, 1], x[:, 1] + 1e-9], 1)  # noqa: E731
    info = mc.rollout(pol, np.array([[-0.5, 0.0], [-0.4, 0.0]]))
    assert info["success"].all() and (info["t_end"] < 200).all()
    # always pushing right from rest at the valley bottom never escapes
    info = mc.rollout(lambda x: np.tile([0.0, 0.0, 1.0], (len(x), 1)), np.array([[-0.4, 0.0]]))
    assert not info["success"].any()
    cp = classic.CartPole()
    info = cp.rollout(lambda x: np.tile([1.0, 0.0], (len(x), 1)), np.zeros((2, 4)))
    assert not info["success"].any() and (info["t_end"] < 500).all()


# ---- ΔW codec / helpers ------------------------------------------------------------------------
def test_sparse_codec_roundtrip_and_limits():
    base = mlp.init_params([3, 8, 8, 2], np.random.default_rng(0))
    d = {k: np.zeros_like(v) for k, v in base.items()}
    d["fc0.weight"][2, 1] = 0.5
    d["fc2.bias"][1] = -1.25
    enc = mlp.encode_delta(d)
    assert set(enc) == {"fc0.weight", "fc2.bias"}
    dec = mlp.decode_delta(json.loads(json.dumps(enc)), base, max_l0=0.04)
    assert np.array_equal(dec["fc0.weight"], d["fc0.weight"]) and np.array_equal(dec["fc2.bias"], d["fc2.bias"])
    with pytest.raises(ValueError):
        mlp.decode_delta({"fc0.weight": {"idx": [999], "val": [1.0]}}, base)
    with pytest.raises(ValueError):
        mlp.decode_delta({"nope": {"idx": [0], "val": [1.0]}}, base)
    many = {"fc1.weight": {"idx": list(range(20)), "val": [1.0] * 20}}
    with pytest.raises(ValueError):
        mlp.decode_delta(many, base, max_l0=0.04)          # 20 > 4% of 130 params
    assert mlp.decode_delta({}, base) == {}


def test_bc_training_respects_pruning():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(512, 2)).astype(np.float32)
    Y = (X[:, 0] > 0).astype(int)
    p = mlp.train_bc(X, Y, [2, 16, 16, 2], "ce", seed=0, pruned={0: [3, 5], 1: [7]}, epochs=2, batch=128,
                     log=lambda m: None)
    assert np.all(p["fc0.weight"][[3, 5]] == 0) and np.all(p["fc0.bias"][[3, 5]] == 0)
    assert np.all(p["fc1.weight"][:, [3, 5]] == 0) and np.all(p["fc1.weight"][7] == 0)
    assert np.all(p["fc2.weight"][:, 7] == 0)


@pytest.mark.parametrize("tid", sorted(specs.SPECS))
def test_reference_circuits_are_sparse_and_use_pruned_units(tid):
    sp = specs.SPECS[tid]
    rng = np.random.default_rng(1)
    base = mlp.init_params(sp.sizes, rng)
    pruned = {0: sorted(rng.choice(sp.sizes[1], sp.n_pruned[0], replace=False).tolist()),
              1: sorted(rng.choice(sp.sizes[2], sp.n_pruned[1], replace=False).tolist())}
    for th in sp.theta_grid()[:3]:
        d = sp.circuit(base, pruned, th)
        assert set(d) == set(base) and all(d[k].shape == base[k].shape for k in d)
        assert mlp.l0_fraction(base, d) < specs.MAX_L0
        rows0 = np.flatnonzero(np.abs(d["fc0.weight"]).sum(1) + np.abs(d["fc0.bias"]))
        rows1 = np.flatnonzero(np.abs(d["fc1.weight"]).sum(1) + np.abs(d["fc1.bias"]))
        assert set(rows0) <= set(pruned[0]) and set(rows1) <= set(pruned[1])


def test_kinetix_layout_constants():
    from weightbench.tracks.tr02_control_lib import kinetix_car as K

    assert (K.CX, K.CVX, K.GX, K.CSIN) == (52, 54, 26, 64)
    assert K.OFF_GRAV == K.OBS_DIM - 1
    assert len(K.POLY_FEATS) == 26 and len(K.CIRC_FEATS) == 18 and len(K.JOINT_FEATS) == 54 and len(K.THR_FEATS) == 24
    assert K.validate_variant({"kind": 0, "car_dx": 0.1, "goal_x": 0.3, "ramp_dx": 0.0})[0] == 0
    with pytest.raises(ValueError):
        K.validate_variant({"kind": 0, "car_dx": 3.0})
    assert "chassis_x" in K.obs_layout_markdown()


# ---- registry / budget / leakage ---------------------------------------------------------------
def test_tasks_registered_with_budget():
    from weightbench.task import load_all

    reg = load_all()
    for i in range(1, 5):
        t = reg[f"TR02-CTRL-00{i}"]
        assert t.track == "TR02" and t.submission == "delta_w"
        assert t.budget.n_forward == 50 and t.budget.n_probe == 0 and t.budget.max_l0 == 0.04
        assert t.probe_kinds == {"rollout": "forward"}


def test_rollout_budget_enforced(tmp_path, monkeypatch):
    from weightbench.ledger import Ledger
    from weightbench.server import RunState
    from weightbench.task import load_all

    task = load_all()["TR02-CTRL-002"]()
    monkeypatch.setattr(type(task), "probe", lambda self, a, k, p: {"success": False})
    led = Ledger(tmp_path / "l.jsonl")
    st = RunState(task, tmp_path, tmp_path, led, tmp_path)
    codes = [st.probe("rollout", {})[0] for _ in range(51)]
    assert codes[:50] == [200] * 50 and codes[50] == 429
    assert sum(1 for e in led.entries("probe")) == 50
    assert sum(1 for e in led.entries("probe_rejected")) == 1


def test_search_reference_multi_uses_K_and_keeps_the_other_flaws(monkeypatch):
    """The multi-flaw reference must pick a circuit that leaves every flaw repairable, and K (the number of flaws in
    the expert) must be in scope there, so a level build with an added flaw reaches the gadget loop."""
    sp = _ToyMC()
    flaws = [types.SimpleNamespace(name="brakeA"), types.SimpleNamespace(name="brakeB")]
    sets = {k: np.array([k, k], dtype=object) for k in
            ("fail_train_1", "fail_train_2", "fail_train_3", "norm_train")}
    # rows = candidate index; two train episodes per pool. Circuit 0 repairs flaw 1 and 2 but destroys flaw 3;
    # circuits 1 and 2 repair all three.
    table = {"fail_train_1": [[1, 1], [1, 1], [1, 1]],
             "fail_train_2": [[1, 1], [1, 1], [1, 1]],
             "fail_train_3": [[0, 0], [1, 1], [1, 1]],
             "norm_train":   [[1, 1], [1, 1], [1, 1]]}
    monkeypatch.setattr(sp, "theta_grid", lambda: [dict(k=k) for k in (0, 1, 2)], raising=False)
    monkeypatch.setattr(sp, "circuit", lambda base, pruned, th: dict(k=th["k"]), raising=False)
    monkeypatch.setattr(sp, "shared_delta", lambda base, plan: {}, raising=False)
    monkeypatch.setattr(sp, "gadget_grid", lambda flaw: [dict(k=k) for k in (0, 1, 2)], raising=False)
    monkeypatch.setattr(specs.KN, "gadget", lambda spec, base, plan, j, f, th: dict(k=th["k"]))
    monkeypatch.setattr(specs.KN, "add", lambda a, b: {**a, **b})
    monkeypatch.setattr(specs, "_edit_cost", lambda delta, base: {"cost": 0.01})
    monkeypatch.setattr(specs.mlp, "apply_delta", lambda base, d: d)
    monkeypatch.setattr(sp, "run_many", lambda deltas, base, starts: np.array(table[starts[0]], dtype=float),
                        raising=False)
    monkeypatch.setattr(sp, "run", lambda params, starts, record=False: {"success": np.array([0.5, 0.5])},
                        raising=False)
    logs = []
    ref, ths = sp._search_reference_multi({}, {}, {}, flaws, sets, logs.append)
    assert ths[0] == {"k": 1}                   # not circuit 0, which leaves flaw 3 unrepairable
    assert [t["k"] for t in ths] == [1, 1, 1]
    assert ref == {"k": 1.0}
    assert any("joint circuit" in m for m in logs)


def test_search_reference_multi_keeps_the_circuit_when_no_gadget_helps(monkeypatch):
    """A gadget spends a unit and edit budget. If the circuit already repairs the flaw and every candidate is worse,
    the reference keeps the circuit rather than shipping a net-harmful edit (e.g. a best
    gadget at 0.59 train success on a pool the circuit repairs at 1.00)."""
    sp = _ToyMC()
    flaws = [types.SimpleNamespace(name="brakeA"), types.SimpleNamespace(name="brakeB")]
    sets = {k: np.array([k, k], dtype=object) for k in
            ("fail_train_1", "fail_train_2", "fail_train_3", "norm_train")}
    table = {"fail_train_1": [[1, 1], [1, 1], [1, 1]],
             "fail_train_2": [[0, 1], [1, 0], [0, 0]],     # every gadget loses on both flaw pools
             "fail_train_3": [[0, 1], [1, 0], [0, 0]],
             "norm_train":   [[1, 1], [1, 1], [1, 1]]}
    monkeypatch.setattr(sp, "theta_grid", lambda: [dict(k=k) for k in (0, 1, 2)], raising=False)
    monkeypatch.setattr(sp, "circuit", lambda base, pruned, th: dict(k=th["k"]), raising=False)
    monkeypatch.setattr(sp, "shared_delta", lambda base, plan: {}, raising=False)
    monkeypatch.setattr(sp, "gadget_grid", lambda flaw: [dict(k=k) for k in (0, 1, 2)], raising=False)
    monkeypatch.setattr(specs.KN, "gadget", lambda spec, base, plan, j, f, th: dict(k=th["k"]))
    monkeypatch.setattr(specs.KN, "add", lambda a, b: {**a, **b})
    monkeypatch.setattr(specs, "_edit_cost", lambda delta, base: {"cost": 0.01})
    monkeypatch.setattr(specs.mlp, "apply_delta", lambda base, d: d)
    monkeypatch.setattr(sp, "run_many", lambda deltas, base, starts: np.array(table[starts[0]], dtype=float),
                        raising=False)
    monkeypatch.setattr(sp, "run", lambda params, starts, record=False: {"success": np.array([1.0, 1.0])},
                        raising=False)
    logs = []
    _, ths = sp._search_reference_multi({}, {}, {}, flaws, sets, logs.append)
    assert ths[1] == "identity" and ths[2] == "identity"
    assert sum("circuit alone" in m for m in logs) == 2


def test_pick_gadget_prefers_the_rest_of_the_task_over_edit_cost():
    """Hand-computed selection: ties on the repaired flaw are broken by the pools it was not asked to repair, then
    the normal pool, then cost.

    s_fail (tol 0.02): 0.98 1.00 0.99 0.97 0.60  -> candidates 0,1,2 are within tol of the best 1.00; 3 and 4 are
    not. floor:          0.90 0.95 0.90  ...       -> 1 wins. sn: 0.80 0.70 0.95  -> not reached on this row; on the
    floor tie below it decides. cost: 0.10 0.40 0.20 -> only breaks a floor+sn tie.
    """
    sf = np.array([0.98, 1.00, 0.99, 0.97, 0.60])
    floor = np.array([0.90, 0.95, 0.90, 1.00, 1.00])
    sn = np.array([0.80, 0.70, 0.95, 1.00, 1.00])
    cost = np.array([0.10, 0.40, 0.20, 0.01, 0.01])
    pick = specs._pick_gadget(sf, floor, sn, cost, tol=0.02)
    assert pick == 1                                  # highest floor among the tied candidates
    floor2 = np.array([0.90, 0.90, 0.90, 1.00, 1.00])  # floor tie between 0,1,2 -> normal pool decides
    assert specs._pick_gadget(sf, floor2, sn, cost, tol=0.02) == 2
    sn2 = np.full(5, 0.5)                              # floor and normal tie -> cheapest edit decides
    assert specs._pick_gadget(sf, floor2, sn2, cost, tol=0.02) == 0
    # a strictly worse repair never wins on the other columns
    assert specs._pick_gadget(sf, floor, sn, cost, tol=0.0) == 1


def test_pick_gadget_first_tie_break_keeps_the_flaw_one_repair():
    """The joint circuit search passes ``first`` = success on flaw 1, so that a theta grid whose worst-pool score
    ties at zero still keeps the theta that repairs the task's own defect rather than the cheapest one.

    worst pool: 0.00 0.00 0.00 0.00 (no circuit repairs every flaw, so all four tie at the best 0.00)
    first:      0.00 1.00 1.00 0.90 (flaw-1 success)  -> 1 and 2; then floor 0.5 vs 0.7 -> 2 wins.
    """
    worst = np.zeros(4)
    first = np.array([0.00, 1.00, 1.00, 0.90])
    floor = np.array([1.00, 0.50, 0.70, 0.60])
    sn = np.array([1.00, 0.50, 0.70, 0.60])
    cost = np.array([0.01, 0.20, 0.30, 0.10])          # without `first`, the cheapest (0) would win
    assert specs._pick_gadget(worst, floor, sn, cost, tol=0.0) == 0
    assert specs._pick_gadget(worst, floor, sn, cost, tol=0.0, first=first) == 2
    # `first` only breaks a tie inside the tol band: a theta outside it never wins on the other columns
    assert specs._pick_gadget(np.array([1.0, 0.0, 0.0, 0.0]), floor, sn, cost, tol=0.0, first=first) == 0


class _ToyMC(specs.MountainCarSpec):
    """MountainCar at toy scale so the full build/evaluate/probe path runs in seconds."""
    sizes = [2, 24, 24, 3]
    n_pruned = (2, 2)
    bc_n, bc_epochs, pool_n, n_train = 6000, 4, 800, 10

    def theta_grid(self):
        return [dict(c=4.0, g=4.0, A=8.0, t=-0.02), dict(c=8.0, A=16.0, t=0.0)]


def test_toy_build_evaluate_probe_and_no_leakage(tmp_path, monkeypatch):
    monkeypatch.setattr(specs, "N_EVAL", 10)
    monkeypatch.setattr(specs, "N_BASELINE", 5)
    sp = _ToyMC()
    stats = sp.build(tmp_path, seed=0, log=lambda m: None)
    pub = {p.name for p in (tmp_path / "public").iterdir()}
    assert pub == {"policy.npz", "failure_trajectory.npz", "ENV.md", "ARCH.md", "rollout.py"}
    for secret in ("eval_sets.npz", "reference_delta.npz", "baseline_delta.npz", "stats.json"):
        assert (tmp_path / "private" / secret).exists() and secret not in pub
    with np.load(tmp_path / "public" / "failure_trajectory.npz") as z:
        assert not bool(z["success"]) and z["obs"].shape[1] == 2
    assert stats["base"]["s_fail"] == 0.0 and stats["base"]["s_norm"] == 1.0
    # evaluation of ΔW = 0 reproduces the base numbers; reference delta is valid
    ev0 = sp.evaluate(tmp_path, {})
    assert ev0["s_plus"] == 0.0 and ev0["d_col"] == 0.0
    ref = mlp.load_params(tmp_path / "private" / "reference_delta.npz")
    ev = sp.evaluate(tmp_path, ref)
    assert 0.0 <= ev["s_plus"] <= 1.0 and ev["l0"] < specs.MAX_L0
    # probe: inline sparse ΔW, start validation
    out = sp.probe(tmp_path, {"delta": mlp.encode_delta(ref), "start": "failure"})
    assert {"success", "obs", "action", "reward", "t_end"} <= set(out)
    with pytest.raises(ValueError):
        sp.probe(tmp_path, {"start": [5.0, 0.0]})
    with pytest.raises(ValueError):
        sp.probe(tmp_path, {"delta": {"fc1.weight": {"idx": list(range(300)), "val": [1.0] * 300}}})
