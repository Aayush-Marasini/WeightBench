"""TR05-DEAD-001/002: generator plants, IBP soundness, pruning, scoring (toy sizes: 8 Boolean inputs)."""
import json

import numpy as np
import pytest

from weightbench.task import Submission
from weightbench.tracks import tr05_dead as T
from weightbench.tracks.tr05_dead_lib import jobs as J
from weightbench.tracks.tr05_dead_lib import net as N


@pytest.mark.parametrize("hidden", [(24,), (16, 16, 16)])
def test_plants_dead_provable_decoys_alive(hidden):
    layers, plants = N.generate(8, hidden, n_dead=3, n_decoys=2, seed=0)
    dead, ibp = N.dead_units(layers, 8), N.ibp_dead(layers)
    for li in range(len(hidden)):
        assert set(plants["dead"][li]) <= set(ibp[li]) <= set(dead[li])     # planted ⊆ provable ⊆ truly dead
        assert not set(plants["decoy"][li]) & set(dead[li])                  # decoys fire on some input


def test_generate_deterministic():
    a, _ = N.generate(8, (16, 16), 2, 2, seed=5)
    b, _ = N.generate(8, (16, 16), 2, 2, seed=5)
    for (wa, ba), (wb, bb) in zip(a, b):
        np.testing.assert_array_equal(wa, wb)
        np.testing.assert_array_equal(ba, bb)


def test_ibp_upper_hand_computed():
    layers = [(np.array([[1.0, -2.0], [0.5, 0.5]], np.float32), np.array([-1.5, 0.0], np.float32)),
              (np.array([[1.0, 1.0]], np.float32), np.zeros(1, np.float32))]
    assert np.allclose(N.ibp_upper(layers)[0], [-0.5, 1.0])                 # unit 0: 1·1 + 0 − 1.5
    assert N.ibp_dead(layers) == [[0]] and N.dead_units(layers, 2) == [[0]]


def test_prune_exact_and_detected():
    layers, plants = N.generate(8, (12, 12), 2, 1, seed=1)
    units = [plants["dead"][0], plants["dead"][1]]
    delta = N.prune(layers, units)
    edited = N.from_dict({k: v + delta.get(k, 0.0) for k, v in N.to_dict(layers).items()})
    assert N.pruned_units(layers, edited) == [sorted(u) for u in units]
    assert N.zeroing_only(layers, edited)
    x = N.all_inputs(8)
    assert np.abs(N.forward(edited, x) - N.forward(layers, x)).max() == 0.0     # dead units: exact no-op


@pytest.fixture
def toy_art(tmp_path):
    art = tmp_path / "art"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    (art / "private" / "spec.json").write_text(json.dumps(
        {"seed": 3, "n_inputs": 8, "hidden": [24], "n_dead": 4, "n_decoys": 3}))
    J.build(art)
    return art


def _evaluate(task, art, delta, monkeypatch):
    def fake_heavy(art_dir, step, out_name, extra=(), push=()):
        return J.evaluate(art_dir, push[0])
    monkeypatch.setattr(T, "_heavy", fake_heavy)
    return task.evaluate(art, Submission("delta_w", delta=delta))


def test_reference_sound_and_beats_baseline(toy_art, monkeypatch):
    task = T.DeadOneLayer()
    ref = _evaluate(task, toy_art, task.reference(toy_art).delta, monkeypatch)
    base = _evaluate(task, toy_art, task.baseline(toy_art).delta, monkeypatch)
    truth = json.loads((toy_art / "private" / "truth.json").read_text())
    assert ref.valid and ref.details["sound"] and ref.details["live_pruned"] == 0
    assert ref.s_plus == pytest.approx(len(set(truth["ibp_dead"][0]) & set(truth["dead"][0])) / len(truth["dead"][0]))
    assert ref.s_plus > base.s_plus


def test_pruning_a_decoy_is_unsound(toy_art, monkeypatch):
    task = T.DeadOneLayer()
    truth = json.loads((toy_art / "private" / "truth.json").read_text())
    layers = N.from_dict(np.load(toy_art / "public" / "mlp.npz"))
    delta = N.prune(layers, [truth["ibp_dead"][0] + truth["plants"]["decoy"][0][:1]])
    ev = _evaluate(task, toy_art, delta, monkeypatch)
    assert not ev.details["sound"] and ev.s_plus == 0.0


def test_non_zeroing_edit_invalid(toy_art, monkeypatch):
    task = T.DeadOneLayer()
    ev = _evaluate(task, toy_art, {"b1": np.full(24, 0.1, np.float32)}, monkeypatch)
    assert not ev.valid and "only set entries to 0" in ev.details["reason"]
    assert not task.evaluate(toy_art, Submission("delta_w", delta={"W9": np.zeros(2)})).valid


def test_prompt_and_registry(toy_art, tmp_path):
    for cls in (T.DeadOneLayer, T.DeadDeep):
        assert cls.track == "TR05" and cls.budget.n_forward == 0 and cls.probe_kinds == {}
    prompt = T.DeadOneLayer().prompt(toy_art)                  # architecture of the built artifact (toy spec)
    assert "N_forward = 0" in prompt and "{0, 1}^8" in prompt and "submit-delta" in prompt
    assert "W1: [24, 8]" in prompt and "W2: [1, 24]" in prompt
    unbuilt = T.DeadDeep().prompt(tmp_path)                    # no spec: class defaults
    assert "{0, 1}^16" in unbuilt and "W3: [96, 96]" in unbuilt and "W4: [1, 96]" in unbuilt


# ------------------------------------------------------------------------------------- propagation_depth knob
ORIGINAL_SPECS = {   # the default (level-free) generator settings
    "TR05-DEAD-001": {"hidden": [128], "n_dead": 16, "n_decoys": 8},
    "TR05-DEAD-002": {"hidden": [96, 96, 96], "n_dead": 8, "n_decoys": 6},
}


def _art(tmp_path, name, spec):
    art = tmp_path / name
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    (art / "private" / "spec.json").write_text(json.dumps(spec))
    return art


def _npz_equal(a, b):
    with np.load(a) as za, np.load(b) as zb:
        assert za.files == zb.files
        for k in za.files:
            assert za[k].dtype == zb[k].dtype
            np.testing.assert_array_equal(za[k], zb[k])


@pytest.mark.parametrize("cls", [T.DeadOneLayer, T.DeadDeep])
def test_default_level_byte_identical_to_original(cls, tmp_path, monkeypatch):
    """Default level reproduces the original task: the same generator settings (full size) and identical artifacts
    (real widths and depths; 10 Boolean inputs instead of 16 to keep the enumeration CPU-light)."""
    full = cls().spec(7)
    assert {k: full[k] for k in ORIGINAL_SPECS[cls.id]} == ORIGINAL_SPECS[cls.id] and full["n_inputs"] == 16
    assert full["level"] == cls.knob_level
    monkeypatch.setattr(cls, "n_inputs", 10)
    monkeypatch.setenv("WB_HEAVY", "local")
    monkeypatch.setattr(T, "_heavy", lambda art_dir, step, out_name, extra=(), push=(): J.build(art_dir))
    new = tmp_path / "new"
    (new / "public").mkdir(parents=True)
    (new / "private").mkdir()
    cls().build(new, 7)
    old = _art(tmp_path, "old", {"task": cls.id, "seed": 7, "n_inputs": 10, **ORIGINAL_SPECS[cls.id]})
    J.build(old)
    for f in ("public/mlp.npz", "private/reference.npz", "private/baseline.npz"):
        _npz_equal(new / f, old / f)
    assert (new / "private" / "truth.json").read_bytes() == (old / "private" / "truth.json").read_bytes()
    assert json.loads((new / "public" / "knob.json").read_text())["value"] == len(ORIGINAL_SPECS[cls.id]["hidden"])


def test_level_sets_depth_exponentially():
    assert [T.hidden_layers(L) for L in range(4)] == [1, 3, 9, 27]
    assert T.DeadOneLayer().at_level(2).hidden == (128,) * 9
    assert T.DeadDeep().at_level(0).hidden == (96,)
    assert T.DeadDeep().hidden == (96, 96, 96)
    with pytest.raises(ValueError):
        T.DeadOneLayer().at_level(-1)


class _ToyDead(T._DeadTask):
    """Toy-sized family (8 inputs, width 16) for CPU tests; not registered."""
    id = "TR05-DEAD-TOY"
    knob_level = 0
    width = 16
    n_dead = 3
    n_decoys = 2
    n_inputs = 8


def test_difficulty_increases_with_level(tmp_path, monkeypatch):
    """The reference (interval-bound proof) stays sound and beats the baseline at every level, while the fraction of
    dead neurons it can prove falls as the knob deepens the network (mean over seeds)."""
    fractions = []
    for level in range(4):
        per_seed = []
        for seed in range(3):
            task = _ToyDead().at_level(level)
            art = _art(tmp_path, f"L{level}s{seed}", task.spec(seed))
            J.build(art)
            ref = _evaluate(task, art, task.reference(art).delta, monkeypatch)
            base = _evaluate(task, art, task.baseline(art).delta, monkeypatch)
            assert ref.valid and ref.details["sound"] and ref.details["live_pruned"] == 0
            assert ref.s_plus >= base.s_plus
            per_seed.append(ref.s_plus)
        fractions.append(float(np.mean(per_seed)))
    assert fractions[0] == 1.0                                  # one layer: interval bounds are exact
    assert all(b <= a for a, b in zip(fractions, fractions[1:])), fractions
    assert fractions[-1] < 0.6, fractions


# ------------------------------------------------------------------------------------- per-unit deadness
def _hand_art(tmp_path):
    """Hand-made 8-input net, hidden (2, 2). Layer 1: u0 live (fires when any input bit is 1), u1 dead (bias -5).
    Layer 2: v0 dead (pre-activation h1_0 - 10 <= -2.5), v1 constant 1. Output = h2_1, so u0 has no effect on it."""
    layers = [(np.array([[1.0] * 8, [0.0] * 8], np.float32), np.array([-0.5, -5.0], np.float32)),
              (np.array([[1.0, 0.0], [0.0, 0.0]], np.float32), np.array([-10.0, 1.0], np.float32)),
              (np.array([[0.0, 1.0]], np.float32), np.zeros(1, np.float32))]
    art = tmp_path / "hand"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    (art / "private" / "spec.json").write_text(json.dumps(
        {"seed": 0, "n_inputs": 8, "hidden": [2, 2], "n_dead": 1, "n_decoys": 0}))
    np.savez(art / "public" / "mlp.npz", **N.to_dict(layers))
    (art / "private" / "truth.json").write_text(json.dumps(
        {"dead": N.dead_units(layers, 8), "ibp_dead": N.ibp_dead(layers), "plants": {"dead": [], "decoy": []}}))
    return art, layers


def test_live_unit_with_zero_output_effect_rejected(tmp_path, monkeypatch):
    art, layers = _hand_art(tmp_path)
    ev = _evaluate(T.DeadOneLayer(), art, N.prune(layers, [[0], []]), monkeypatch)
    assert ev.details["inputs_changed"] == 0                  # output-only check would accept this prune
    assert ev.details["live_units"] == [[0], []]
    assert not ev.details["all_pruned_dead"] and not ev.details["sound"]
    assert ev.s_plus == 0.0


def test_truly_dead_units_accepted(tmp_path, monkeypatch):
    art, layers = _hand_art(tmp_path)
    ev = _evaluate(T.DeadOneLayer(), art, N.prune(layers, [[1], [0]]), monkeypatch)   # both dead units
    assert ev.valid and ev.details["sound"] and ev.details["all_pruned_dead"]
    assert ev.details["live_units"] == [[], []] and ev.details["dead_pruned"] == 2
    assert ev.s_plus == 1.0


def test_delta_that_prunes_nothing_is_unchanged(tmp_path, monkeypatch):
    art, _ = _hand_art(tmp_path)
    ev = _evaluate(T.DeadOneLayer(), art, {"W1": np.zeros((2, 8), np.float32)}, monkeypatch)
    assert ev.valid and ev.details["sound"] and ev.details["all_pruned_dead"]
    assert ev.details["live_units"] == [[], []] and ev.details["dead_pruned"] == 0
    assert ev.details["outputs_kept"] and ev.s_plus == 0.0


def test_per_unit_check_is_deterministic(tmp_path, monkeypatch):
    art, layers = _hand_art(tmp_path)
    delta = N.prune(layers, [[0, 1], [0]])                    # mixes a live and dead unit
    first = _evaluate(T.DeadOneLayer(), art, delta, monkeypatch)
    second = _evaluate(T.DeadOneLayer(), art, delta, monkeypatch)
    assert first.s_plus == second.s_plus == 0.0 and first.details == second.details


@pytest.mark.parametrize("level", [0, 1, 2])
def test_reference_prunes_only_dead_units(level, tmp_path):
    """The IBP reference, run at toy size, prunes only neurons that are dead on all 2^8 inputs."""
    task = _ToyDead().at_level(level)
    art = _art(tmp_path, f"ref{level}", task.spec(4))
    J.build(art)
    layers = N.from_dict(dict(np.load(art / "public" / "mlp.npz")))
    truth = json.loads((art / "private" / "truth.json").read_text())["dead"]
    edited = N.from_dict({k: v + task.reference(art).delta.get(k, 0.0) for k, v in N.to_dict(layers).items()})
    pruned = N.pruned_units(layers, edited)
    assert any(pruned)
    for p, t in zip(pruned, truth):
        assert set(p) <= set(t)


# ------------------------------------------------------------------------------------- second reference: crown + LP
from weightbench.tracks.tr05_dead_lib import crown as C  # noqa: E402


def _two_layer(w2_row, b2):
    """2 Boolean inputs. Layer 1: u0 = relu(x0), u1 = relu(x0) (identical copies). Layer 2: one unit (w2_row, b2)."""
    w1 = np.array([[1.0, 0.0], [1.0, 0.0]], np.float32)
    return [(w1, np.zeros(2, np.float32)),
            (np.array([w2_row], np.float32), np.array([b2], np.float32)),
            (np.ones((1, 1), np.float32), np.zeros(1, np.float32))]


def test_crown_beats_ibp_on_correlated_units():
    # v = h0 − h1 − 0.5. IBP: max(1,0)·1 + max(−1,0)·1 − 0.5 = +0.5 > 0, so IBP cannot prove it dead.
    # Back-substitution: z1 ∈ [0, 1] for both units, so L >= 0 and each ReLU is exactly h = z = x0.
    # Then v = x0 − x0 − 0.5 = −0.5 for every x: upper bound exactly −0.5 (no LP needed).
    layers = _two_layer([1.0, -1.0], -0.5)
    assert np.allclose(N.ibp_upper(layers)[1], [0.5]) and N.ibp_dead(layers) == [[], []]
    ups, _ = C.bounds(layers, use_lp=False)
    assert ups[1][0] == pytest.approx(-0.5, abs=1e-8)
    assert C.crown_dead(layers, use_lp=False)[0] == [[], [0]] == N.dead_units(layers, 2)


def test_crown_never_certifies_barely_alive_unit():
    # v = h0 − 0.98 peaks at +0.02 (x0 = 1): alive, must not be certified with or without the LP.
    layers = _two_layer([1.0, 0.0], -0.98)
    assert N.dead_units(layers, 2) == [[], []]
    for lp in (False, True):
        ups, _ = C.bounds(layers, use_lp=lp)
        assert ups[1][0] >= 0.02 - 1e-6
        assert C.crown_dead(layers, use_lp=lp)[0] == [[], []]


@pytest.mark.parametrize("depth,n_inputs,seed", [(1, 8, 0), (3, 8, 1), (3, 10, 2), (9, 8, 3), (9, 10, 4), (27, 8, 5)])
def test_crown_sound_and_at_least_ibp(depth, n_inputs, seed):
    """Author-side oracle: N.dead_units enumerates all 2^n inputs (test only; the reference never does)."""
    layers, plants = N.generate(n_inputs, (16,) * depth, n_dead=3, n_decoys=2, seed=seed)
    truth, ibp = N.dead_units(layers, n_inputs), N.ibp_dead(layers)
    no_lp, _ = C.crown_dead(layers, use_lp=False)
    with_lp, info = C.crown_dead(layers, use_lp=True)
    assert info["lp"] and info["lp_failed"] == 0
    for li in range(depth):
        assert set(plants["dead"][li]) <= set(ibp[li]) <= set(no_lp[li]) <= set(with_lp[li]) <= set(truth[li])
        assert not set(plants["decoy"][li]) & set(with_lp[li])
    if depth == 1:
        assert with_lp == truth                                     # one layer: exact over the box


def test_crown_degrades_without_scipy(monkeypatch):
    layers, _ = N.generate(8, (16, 16, 16), 3, 2, seed=1)
    monkeypatch.setattr(C, "have_lp", lambda: False)
    dead, info = C.crown_dead(layers, use_lp=True)
    assert not info["lp"] and info["lp_calls"] == 0
    assert dead == C.crown_dead(layers, use_lp=False)[0]


def test_crown_reference_variant_scores_at_least_ibp(tmp_path, monkeypatch):
    task = _ToyDead().at_level(2)
    art = _art(tmp_path, "crown", task.spec(4))
    J.build(art)
    monkeypatch.setattr(T, "_heavy", lambda art_dir, step, out_name, extra=(), push=():
                        J.reference_crown(art_dir) if step == "reference_crown" else J.evaluate(art_dir, push[0]))
    ibp = task.evaluate(art, task.reference(art))
    crown = task.evaluate(art, task.reference(art, "crown"))
    assert (art / "private" / "reference_crown.npz").exists()
    assert crown.valid and crown.details["sound"] and crown.details["live_pruned"] == 0
    assert crown.s_plus >= ibp.s_plus
    monkeypatch.setenv("WB_DEAD_REFERENCE", "crown")
    assert task.evaluate(art, task.reference(art)).s_plus == crown.s_plus
    with pytest.raises(ValueError):
        task.reference(art, "milp")
