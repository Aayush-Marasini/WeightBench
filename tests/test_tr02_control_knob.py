"""TR02-CTRL difficulty knob (K = 2^L expert flaws, tr02_control_lib/knob.py).

No environment is simulated and no policy is trained here: the multi-flaw build runs on a FAKE spec whose
"rollout" is a pure function of the start (success iff the policy's action at the start equals 1), whose expert is
a hand-written rule and whose "BC" returns hand-built weights. Real builds and their validation are slow and are not part of the unit tests.
"""
import json

import numpy as np
import pytest

from weightbench import knobs
from weightbench.tracks.tr02_control_lib import knob as KN
from weightbench.tracks.tr02_control_lib import mlp, specs

SUPPORTED = {"TR02-CTRL-001": (0, 1), "TR02-CTRL-002": (0, 1), "TR02-CTRL-003": (0, 1),
             "TR02-CTRL-004": (0, 1, 2)}


# ---- selection ---------------------------------------------------------------------------------
@pytest.mark.parametrize("tid", ["TR02-CTRL-001", "TR02-CTRL-002", "TR02-CTRL-003", "TR02-CTRL-004"])
def test_flaw_selection_deterministic_nested_and_seeded(tid):
    sp = specs.SPECS[tid]
    top = max(sp.knob_levels)
    assert KN.select_flaws(sp, 0, 0) == []
    for seed in range(4):
        a = KN.select_flaws(sp, seed, top)
        assert a == KN.select_flaws(sp, seed, top)                       # deterministic
        assert len(a) == 2 ** top - 1 and len({f.name for f in a}) == len(a)
        for L in range(1, top + 1):
            assert KN.select_flaws(sp, seed, L) == a[:2 ** L - 1]          # nested across levels
    sigs = {tuple((f.name, tuple(sorted(f.params.items()))) for f in KN.select_flaws(sp, s, top)) for s in range(4)}
    assert len(sigs) > 1                                                 # different across seeds


@pytest.mark.parametrize("tid", sorted(SUPPORTED))
def test_levels_and_rejection(tid):
    from weightbench import runner
    sp = specs.SPECS[tid]
    assert sp.knob_levels == SUPPORTED[tid]
    top = max(SUPPORTED[tid])
    with pytest.raises(ValueError, match="supported knob levels"):
        KN.select_flaws(sp, 0, top + 1)
    with pytest.raises(ValueError, match="supported knob levels"):
        runner.nondefault_level(tid, top + 1)
    assert runner.get_task(tid).level is None                          # unversioned default kept
    assert runner.art_dir(tid, 0, 1).name == "seed0_L1" and runner.art_dir(tid, 0).name == "seed0"


@pytest.mark.parametrize("tid", ["TR02-CTRL-001", "TR02-CTRL-002", "TR02-CTRL-003", "TR02-CTRL-004"])
def test_capacity_supports_exactly_the_wired_levels(tid):
    sp = specs.SPECS[tid]
    top = max(sp.knob_levels)
    for L in sp.knob_levels:
        KN.check_capacity(sp, L)
    with pytest.raises(ValueError, match="pruned units"):
        KN.check_capacity(sp, top + 1)


def test_task_build_rejects_level_before_heavy(tmp_path, monkeypatch):
    from weightbench import heavy
    from weightbench.task import load_all

    monkeypatch.setattr(heavy, "run_heavy", lambda *a, **k: pytest.fail("reached the heavy job"))
    t = load_all()["TR02-CTRL-003"]()
    t._level_override = 2                                               # bypass at_level: build must still reject
    with pytest.raises(ValueError):
        t.build(tmp_path, 0)


def test_task_build_passes_level_only_when_set(tmp_path, monkeypatch):
    from weightbench import heavy
    from weightbench.task import load_all

    calls = []
    monkeypatch.setattr(heavy, "run_heavy", lambda mod, args, *a, **k: calls.append(args) or {})
    monkeypatch.setenv("WB_HEAVY", "local")
    t = load_all()["TR02-CTRL-004"]()
    t.build(tmp_path, 0)
    t.at_level(2).build(tmp_path, 0)
    assert "--level" not in calls[0] and calls[1][-2:] == ["--level", "2"]
    assert calls[0] == calls[1][:-2]                                    # default argv unchanged


# ---- scoring -----------------------------------------------------------------------------------
def test_s_plus_is_min_over_flaws_hand_computed():
    out = KN.per_flaw_s_plus({"flaw1": 0.9, "flaw2": 0.3, "flaw3": 0.0}, {"flaw1": 0.0, "flaw2": 0.0, "flaw3": 0.2})
    assert out["s_plus_by_flaw"] == pytest.approx({"flaw1": 0.9, "flaw2": 0.3, "flaw3": 0.0})   # 0 - 0.2 clamps to 0
    assert out["s_plus"] == pytest.approx(0.0) and out["s_plus_mean"] == pytest.approx(0.4)
    out = KN.per_flaw_s_plus({"flaw1": 0.6, "flaw2": 0.75}, {"flaw1": 0.2, "flaw2": 0.5})
    assert out["s_plus_by_flaw"]["flaw1"] == pytest.approx(0.5) and out["s_plus_by_flaw"]["flaw2"] == pytest.approx(0.5)
    assert out["s_plus"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        KN.per_flaw_s_plus({"flaw1": 1.0}, {"flaw1": 0.0, "flaw2": 0.0})


def test_validate_fails_loudly_naming_flaw_and_rates():
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    succ = np.array([True] * 10 + [False] * 10 + [True] * 7 + [False] * 3)    # flaw 2: policy fails only 30%
    with pytest.raises(KN.FlawValidationError) as e:
        KN.validate(labels, succ, ["flaw1", "flaw2"], need_fail=3, need_norm=5)
    msg = str(e.value)
    assert "flaw 2 (flaw2)" in msg and "0.300" in msg and "flaw 1" not in msg
    assert e.value.report["flaws"]["flaw2"]["policy_fail_rate"] == 0.3
    rep = KN.validate(labels, np.r_[succ[:20], [False] * 10], ["flaw1", "flaw2"], need_fail=3, need_norm=5)
    assert rep["ok"] and rep["flaws"]["flaw2"]["policy_fail_rate"] == 1.0
    with pytest.raises(KN.FlawValidationError, match="normal pool"):
        KN.validate(labels, np.r_[[False] * 10, succ[10:20], [False] * 10], ["flaw1", "flaw2"], 3, 5)


# ---- fake spec: full multi-flaw build without any environment -----------------------------------
class _FakeSpec(specs.Spec):
    """State s in [0, 1] (1-D). 'Rollout' succeeds iff argmax(policy(s)) == 1. Default expert: action 0 on
    s < 0.2 (flaw 1), else 1. Added flaws: intervals where the expert plays 0."""
    key, task_id = "fake", "TR02-CTRL-004"
    sizes = [1, 16, 16, 2]
    n_pruned = (8, 4)
    circuit_units = (1, 1)
    shared_h0 = 0
    knob_levels = (0, 1, 2)
    n_train = 5
    n_flaw_pool = 200
    flaw1_title = "fake stuck situation one"
    exhibit = (True, True, True, True)        # which flaws the fake 'cloned policy' reproduces

    def flaw_family(self, rng):
        out = []
        for j, (lo, hi) in enumerate(((0.3, 0.4), (0.5, 0.6), (0.7, 0.8))):
            e = KN.jit(rng, 0.0, 0.01)
            out.append(KN.Flaw(f"band{j}", f"fake band {j}", ((((0, 1.0),), -(lo + e)), (((0, -1.0),), hi)),
                               0, 1.0, (lo + e,), (hi,), {"lo": lo + e}))
        return out

    def norm_affine(self):
        return np.ones(1), np.zeros(1)

    def train_base(self, rng, seed, pruned, log, flaws=()):
        # hand-built 'clone': output diff = 1 - 2 * [s in a reproduced flaw region]
        p = {k: np.zeros_like(v) for k, v in mlp.init_params(self.sizes, rng).items()}
        p["fc2.bias"][1] = 1.0
        regions = [(0.0, 0.2)] + [(-f.ineqs[0][1], f.ineqs[1][1]) for f in flaws]
        free = [u for u in range(16) if u not in pruned[0]]
        free1 = [u for u in range(16) if u not in pruned[1]]
        for r, (lo, hi) in enumerate(regions):
            if not self.exhibit[r]:
                continue
            a, b, w = free[2 * r], free[2 * r + 1], free1[r]
            p["fc0.weight"][a, 0], p["fc0.bias"][a] = -100.0, 100.0 * lo     # relu(100 (lo - s)): outside left
            p["fc0.weight"][b, 0], p["fc0.bias"][b] = 100.0, -100.0 * hi     # relu(100 (s - hi)): outside right
            p["fc1.weight"][w, [a, b]], p["fc1.bias"][w] = -10.0, 1.0
            p["fc2.weight"][1, w] = -4.0
        return p

    def _act(self, params, starts):
        return np.argmax(mlp.forward(params, np.asarray(starts, np.float64).reshape(-1, 1)), 1)

    def run(self, params, starts, record=False):
        a = self._act(params, starts)
        out = {"success": a == 1, "ret": a.astype(float), "t_end": np.ones(len(a), int)}
        if record:
            out.update(obs=np.asarray(starts).reshape(-1, 1, 1), act=a.reshape(-1, 1), rew=a.reshape(-1, 1) * 1.0)
        return out

    def run_many(self, deltas, base, starts):
        return np.stack([self.run(mlp.apply_delta(base, d), starts)["success"] for d in deltas]).astype(float)

    def run_expert(self, flaws, active, starts):
        o = np.asarray(starts).reshape(-1, 1)
        y = KN.apply_overrides((o[:, 0] >= 0.2).astype(int), o, flaws, active)
        vis = np.stack([f.mask(o) for f in flaws], 1) if flaws else np.zeros((len(o), 0), bool)
        return y == 1, vis

    def sample_pool_level(self, rng, flaws):
        parts = [rng.uniform(0, 1, (2000, 1))] + [rng.uniform(f.start_low, f.start_high, (self.n_flaw_pool, 1))
                                                  for f in flaws]
        s = np.concatenate(parts)
        return s, np.ones(len(s), bool)

    def circuit(self, base, pruned, th):
        d = {k: np.zeros_like(v) for k, v in base.items()}
        u, w = pruned[0][0], pruned[1][0]
        d["fc0.weight"][u, 0], d["fc0.bias"][u] = -100.0, 20.0               # relu(100 (0.2 - s))
        d["fc1.weight"][w, u] = 1.0
        d["fc2.weight"][1, w] = th["A"]
        return d

    def theta_grid(self):
        return [dict(A=0.0), dict(A=100.0)]

    def gadget_grid(self, flaw):
        return [dict(kg=50.0, G=4.0, A=0.0), dict(kg=50.0, G=4.0, A=8.0)]

    def wire_gadget(self, d, plan, w, flaw, th):
        d["fc2.weight"][1, w] = th["A"] * flaw.fix

    def write_docs(self, art, stats):
        (art / "public" / "ENV.md").write_text("fake")
        (art / "public" / "ARCH.md").write_text("fake")


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(specs, "N_EVAL", 20)
    monkeypatch.setattr(specs, "N_BASELINE", 6)


def test_fake_multi_build_scores_min_and_reference_fixes_all(tmp_path, small):
    sp = _FakeSpec()
    stats = sp.build(tmp_path, seed=3, log=lambda m: None, level=2)
    K = 4
    assert stats["knob"]["K"] == K and stats["knob"]["validation"]["ok"]
    assert stats["base"]["s_fail_by_flaw"] == {f"flaw{c}": 0.0 for c in range(1, K + 1)}
    assert stats["base"]["s_norm"] == 1.0
    ref = stats["reference"]
    assert ref["s_plus"] == pytest.approx(1.0) and ref["d_col"] == 0.0     # K gadgets fix every flaw
    assert len(ref["theta"]) == K
    # gadgets use only pruned units
    d = mlp.load_params(tmp_path / "private" / "reference_delta.npz")
    rows0 = set(np.flatnonzero(np.abs(d["fc0.weight"]).sum(1) + np.abs(d["fc0.bias"])))
    assert rows0 <= set(stats["pruned_units"]["0"] if "0" in stats["pruned_units"] else stats["pruned_units"][0])
    # evaluate: ΔW = 0 -> 0; reference -> 1; flaw-1 circuit alone -> min over flaws = 0, mean = 1/4
    assert sp.evaluate(tmp_path, {})["s_plus"] == 0.0
    ev = sp.evaluate(tmp_path, d)
    assert ev["s_plus"] == pytest.approx(1.0) and ev["n_eval_episodes"] == (K + 1) * 20
    base = mlp.load_params(tmp_path / "public" / "policy.npz")
    pr = {int(k): v for k, v in stats["pruned_units"].items()}
    only1 = sp.circuit(base, pr, dict(A=100.0))
    ev1 = sp.evaluate(tmp_path, only1)
    assert ev1["s_plus_by_flaw"]["flaw1"] == 1.0 and ev1["s_plus"] == 0.0 and ev1["s_plus_mean"] == pytest.approx(0.25)
    # public files: one trajectory per flaw, knob.json without private parameters or pools
    pub = {p.name for p in (tmp_path / "public").iterdir()}
    assert {"failure_trajectory.npz", "failure_trajectory_2.npz", "failure_trajectory_4.npz", "knob.json"} <= pub
    assert not pub & {"eval_sets.npz", "stats.json", "reference_delta.npz", "baseline_delta.npz"}
    kp = json.loads((tmp_path / "public" / "knob.json").read_text())
    assert kp["K"] == K and kp["level"] == 2 and kp["value"] == knobs.value(knobs.get("TR02"), 2)
    text = (tmp_path / "public" / "knob.json").read_text()
    for f in KN.select_flaws(sp, 3, 2):
        assert f.name not in text and "ineqs" not in text and str(f.params["lo"]) not in text
    priv = json.loads((tmp_path / "private" / "knob.json").read_text())
    assert [f["name"] for f in priv["flaws"]] == [f.name for f in KN.select_flaws(sp, 3, 2)]
    # probe start "failure3" reads the public trajectory; unknown situations are rejected
    assert sp._failure_start(tmp_path, "failure3") is not None
    with pytest.raises(ValueError):
        sp._failure_start(tmp_path, "failure7")


def test_fake_build_fails_loudly_when_a_flaw_does_not_take(tmp_path, small):
    sp = _FakeSpec()
    sp.exhibit = (True, True, False, True)          # the 'clone' does not reproduce added flaw 2 (flaw 3 overall)
    with pytest.raises(KN.FlawValidationError, match=r"flaw 3 \(flaw3\): policy failure rate 0\.000"):
        sp.build(tmp_path, seed=3, log=lambda m: None, level=2)
    rep = json.loads((tmp_path / "private" / "knob_validation.json").read_text())
    assert rep["flaws"]["flaw3"]["policy_fail_rate"] == 0.0 and not rep["ok"]
    assert not (tmp_path / "public" / "policy.npz").exists()        # nothing shipped


def test_level0_is_default_build_plus_knob_json(tmp_path, monkeypatch):
    sp = _FakeSpec()
    seen = []
    monkeypatch.setattr(specs.Spec, "_build_default", lambda self, art, seed, log: seen.append(seed) or {"base": {}})
    sp.build(tmp_path, seed=1, log=lambda m: None, level=0)
    assert seen == [1] and json.loads((tmp_path / "public" / "knob.json").read_text())["K"] == 1
    seen.clear()
    (tmp_path / "public" / "knob.json").unlink()
    sp.build(tmp_path, seed=1, log=lambda m: None)                  # default: no knob.json
    assert seen == [1] and not (tmp_path / "public" / "knob.json").exists()


# ---- reachability redraw (a drawn flaw no start can be attributed to) -----------------------------------------
class _DeadMemberSpec(_FakeSpec):
    """The fake family plus one member whose half-spaces cannot hold together (s > 0.95 AND s < 0.9): the defect
    class of a band written as x < x_b AND x > x_b + 0.14."""

    def flaw_family(self, rng):
        fam = super().flaw_family(rng)
        e = KN.jit(rng, 0.0, 0.01)
        return fam + [KN.Flaw("dead", "fake empty band", ((((0, 1.0),), -(0.95 + e)), (((0, -1.0),), 0.9)),
                              0, 1.0, (0.9,), (0.95 + e,), {"lo": 0.95 + e})]


def _seed_drawing(sp, name, level=1):
    return next(s for s in range(200) if KN.select_flaws(sp, s, level)[0].name == name)


def test_empty_flaw_region_has_no_attributable_start():
    sp = _DeadMemberSpec()
    fam = {f.name: f for f in sp.flaw_family(np.random.default_rng([0, KN.FLAW_SEED_SALT]))}
    assert KN.n_attributable(sp, fam["dead"], 0, 0) == 0
    assert KN.n_attributable(sp, fam["band0"], 0, 0) > 200          # its box (200) + ~10% of the uniform pool


def test_reachable_selection_is_the_plain_draw_when_every_drawn_flaw_is_reachable():
    sp = _FakeSpec()
    for seed in range(6):
        for L in (1, 2):
            assert KN.select_reachable_flaws(sp, seed, L, need=26, log=lambda m: None) == KN.select_flaws(sp, seed, L)
    assert KN.select_reachable_flaws(sp, 0, 0, need=26) == []


def test_unreachable_flaw_is_redrawn_deterministically_and_levels_stay_nested():
    sp = _DeadMemberSpec()
    seed = _seed_drawing(sp, "dead")
    msgs = []
    l1 = KN.select_reachable_flaws(sp, seed, 1, need=26, log=msgs.append)
    l2 = KN.select_reachable_flaws(sp, seed, 2, need=26, log=lambda m: None)
    assert [f.name for f in l1] != ["dead"] and all(f.name != "dead" for f in l2)
    assert l2[:1] == l1 and len({f.name for f in l2}) == 3                 # nested, distinct
    assert l1 == KN.select_reachable_flaws(sp, seed, 1, need=26, log=lambda m: None)   # deterministic
    assert any("dead" in m and "redrawing" in m for m in msgs)
    # the substitute is the next candidate of the seed's own stream, not a hard-coded member
    stream = [f.name for _, f in KN.candidate_flaws(sp, seed)]
    assert stream[0] == "dead" and l1[0].name == stream[1]


def test_redraw_gives_up_loudly_when_too_few_members_are_reachable():
    sp = _DeadMemberSpec()
    with pytest.raises(ValueError, match="reachable"):
        KN.select_reachable_flaws(sp, 0, 2, need=10 ** 6, log=lambda m: None)


def test_fake_build_redraws_an_unreachable_drawn_flaw(tmp_path, small):
    sp = _DeadMemberSpec()
    seed = _seed_drawing(sp, "dead")
    stats = sp.build(tmp_path, seed=seed, log=lambda m: None, level=1)
    assert stats["knob"]["validation"]["ok"]
    priv = json.loads((tmp_path / "private" / "knob.json").read_text())
    names = [f["name"] for f in priv["flaws"]]
    assert names == [f.name for f in KN.select_reachable_flaws(sp, seed, 1, need=26, log=lambda m: None)]
    assert names != ["dead"]


# ---- K-gadget reference on a fake policy, real spec gadgets ----------------------------------------
@pytest.mark.parametrize("tid", ["TR02-CTRL-001", "TR02-CTRL-002", "TR02-CTRL-003", "TR02-CTRL-004"])
def test_k_gadget_reference_sparse_and_on_pruned_units(tid):
    sp = specs.SPECS[tid]
    top = max(sp.knob_levels)
    rng = np.random.default_rng(2)
    base = mlp.init_params(sp.sizes, rng)
    pruned = {0: sorted(rng.choice(sp.sizes[1], sp.n_pruned[0], replace=False).tolist()),
              1: sorted(rng.choice(sp.sizes[2], sp.n_pruned[1], replace=False).tolist())}
    flaws = KN.select_flaws(sp, 0, top)
    plan = KN.unit_plan(sp, pruned, flaws)
    d = KN.add(sp.circuit(base, pruned, sp.theta_grid()[0]), sp.shared_delta(base, plan))
    for j, f in enumerate(flaws):
        g = KN.gadget(sp, base, plan, j, f, sp.gadget_grid(f)[0])
        d = KN.add(d, g)
        # inside the flaw region the gadget's h1 unit is on, far outside it is off
    assert mlp.l0_fraction(base, d) < specs.MAX_L0
    rows0 = np.flatnonzero(np.abs(d["fc0.weight"]).sum(1) + np.abs(d["fc0.bias"]))
    rows1 = np.flatnonzero(np.abs(d["fc1.weight"]).sum(1) + np.abs(d["fc1.bias"]))
    assert set(rows0) <= set(pruned[0]) and set(rows1) <= set(pruned[1])
    used0 = sp.circuit_units[0] + sp.shared_h0 + sum(len(f.ineqs) for f in flaws)
    assert len(set(sum([g for g, _ in plan["gadgets"]], []))) == sum(len(f.ineqs) for f in flaws) <= sp.n_pruned[0]
    assert used0 <= sp.n_pruned[0]


def test_mountaincar_added_flaw_regions_are_reachable():
    """A flaw whose half-spaces cannot hold together is empty: no rollout enters it, so ``attribute`` finds zero
    expert-attributed starts and the level cannot build (e.g. a valley band written as
    x < x_b AND x > x_b + 0.14). MountainCar's state is its observation, so its
    supplementary start box can be checked against the region directly: each box must sit inside the region it is
    there to exercise. (The other TR02 tasks' boxes are in state or variant coordinates, not observations, so this
    check does not generalise to them; the build-time attribution does.)"""
    sp = specs.SPECS["TR02-CTRL-002"]
    rng = np.random.default_rng(0)
    cases = [(f, rng.uniform(f.start_low, f.start_high, size=(2000, 2)))
             for seed in range(4)                                        # the family jitter varies by seed
             for f in sp.flaw_family(np.random.default_rng([seed, KN.FLAW_SEED_SALT]))]
    assert {f.name for f, _ in cases} == {"final_climb_brake", "valley_brake"}
    for f, pts in cases:
        assert f.mask(pts).all(), f"{f.name}: its start box is not inside its region (empty region?)"


def test_classic_gadget_gate_is_on_inside_and_off_outside():
    sp = specs.SPECS["TR02-CTRL-003"]
    rng = np.random.default_rng(0)
    base = {k: np.zeros_like(v) for k, v in mlp.init_params(sp.sizes, rng).items()}
    pruned = {0: list(range(8)), 1: list(range(4))}
    f = KN.select_flaws(sp, 0, 1)[0]
    plan = KN.unit_plan(sp, pruned, [f])
    g = KN.gadget(sp, base, plan, 0, f, dict(kg=32.0, G=4.0, A=4.0))
    th = 1.0 if f.name == "fall_ccw" else -1.0
    # region: upper half on that side, swinging TOWARDS upright (theta_dot sign = -sign(theta))
    inside = np.array([[np.cos(th), np.sin(th), -np.sign(th) * 0.5 / 8]])
    outside = np.array([[np.cos(th), np.sin(th), np.sign(th) * 0.5 / 8]])      # moving AWAY from upright
    assert f.mask(np.array([[np.cos(th), np.sin(th), -np.sign(th) * 0.5]]))[0]
    out_in, out_out = mlp.forward(g, inside)[0, 0], mlp.forward(g, outside)[0, 0]
    assert out_in == pytest.approx(4.0 * f.fix) and out_out == pytest.approx(0.0)


def test_mountaincar_added_flaws_are_high_speed_rightward():
    """The added MountainCar flaws are active only while the car moves right (velocity > ~0), so the flaw-1 "push
    left while moving left" override cannot fix them, and every level-1 family member starts already moving right at
    speed. The gate has to be this low: at v ~ 0.05 the brake slows the car below the gate, switches itself off and
    the car still gets there, which leaves no attributable start."""
    sp = specs.SPECS["TR02-CTRL-002"]
    assert sp.knob_levels == (0, 1)
    for seed in range(4):
        flaws = KN.select_flaws(sp, seed, 1)
        assert len(flaws) == 1
        for f in flaws:
            assert f.action == 0 and f.start_low[1] > 0 and f.start_high[1] <= 0.07
            v_ineqs = [c for c, b in f.ineqs if any(i == 1 and coef > 0 for i, coef in c)]
            assert v_ineqs                                                  # a velocity > threshold half-space


def test_mountaincar_gadget_restores_push_right_inside():
    sp = specs.SPECS["TR02-CTRL-002"]
    rng = np.random.default_rng(0)
    base = {k: np.zeros_like(v) for k, v in mlp.init_params(sp.sizes, rng).items()}
    pruned = {0: list(range(6)), 1: list(range(4))}
    f = KN.Flaw("far_left_brake", "fake", ((((0, -1.0),), -0.8), (((1, 1.0),), -0.05)), action=0, fix=1.0,
                start_low=(-0.9, 0.05), start_high=(-0.8, 0.06), params={})
    plan = KN.unit_plan(sp, pruned, [f])
    g = KN.gadget(sp, base, plan, 0, f, dict(kg=32.0, G=4.0, A=8.0))

    def norm(pos, vel):                                                 # normalised network input
        return np.array([[(pos + 0.3) / 0.9, vel / 0.07]])

    out_in, out_out = mlp.forward(g, norm(-0.85, 0.055))[0], mlp.forward(g, norm(-0.75, 0.055))[0]
    assert out_in[2] - out_in[0] == pytest.approx(2 * 8.0)              # push right (action 2) restored
    assert out_out[2] - out_out[0] == pytest.approx(0.0)                # silent outside the region


# ---- defaults untouched -------------------------------------------------------------------------------
def test_default_prompt_and_registry_unchanged(tmp_path, monkeypatch):
    from weightbench.task import load_all

    t = load_all()["TR02-CTRL-004"]()
    assert t.knob_level is None and t.level is None
    (tmp_path / "public").mkdir()
    (tmp_path / "private").mkdir()
    (tmp_path / "public" / "ARCH.md").write_text("ARCH\n")
    np.savez(tmp_path / "public" / "failure_trajectory.npz", start=np.zeros(4), reward=np.zeros(3))
    st = {"base": {"s_fail": 0.0, "s_norm": 1.0}, "pool_base_success": 0.8, "n_params": 10}
    (tmp_path / "private" / "stats.json").write_text(json.dumps(st))
    p0 = t.prompt(tmp_path)
    assert "Difficulty knob" not in p0 and f"{t.story}\n\nYour job" in p0
    kn = {"K": 1, "level": 0, "stuck_situations": ["x"]}
    (tmp_path / "public" / "knob.json").write_text(json.dumps(kn))
    assert t.prompt(tmp_path) == p0                                  # L0 prompt = default prompt
    st["base"]["s_fail_by_flaw"] = {"flaw1": 0.0, "flaw2": 0.0}
    (tmp_path / "private" / "stats.json").write_text(json.dumps(st))
    (tmp_path / "public" / "knob.json").write_text(json.dumps({"K": 2, "level": 1, "stuck_situations": ["a b", "c d"]}))
    p1 = t.prompt(tmp_path)
    assert "K = 2 stuck situations" in p1 and "S+ is the minimum" in p1 and "failure_trajectory_2.npz" in p1
