"""TR02-SUPER-001 (SYNTHETIC superposition family): knob, closed-form ceiling vs numerical recovery, scoring,
baseline vs reference, determinism and leakage. Pure NumPy on toy sizes; no heavy compute."""
import json

import numpy as np
import pytest

from weightbench import knobs as K_
from weightbench import runner
from weightbench.task import Submission
from weightbench.tracks import tr02_super as T
from weightbench.tracks.tr02_super_lib import net as N

TOL = 1e-9   # float64 closed form vs numerical least squares


# ------------------------------------------------------------------------------------- knob (num_entangled_concepts)
def test_knob_value_is_2_to_the_l_and_is_wired():
    knob = K_.get("TR02")
    assert knob.name == "num_entangled_concepts"
    assert K_.schedule(knob, 4) == [1, 2, 4, 8, 16]
    assert [T.features(L) for L in range(5)] == [1, 2, 4, 8, 16]
    assert "TR02-SUPER-001" in knob.wired
    assert [N.feature_dim(k) for k in (1, 2, 4, 8, 16)] == [1, 1, 2, 4, 8]     # d = K/2, and 1 at K = 1


def test_unsupported_levels_rejected():
    task = T.SuperpositionPatch()
    for bad in (-1, 4, 5, 1.0, True):          # L4 (K = 16) is not a supported rung
        with pytest.raises(ValueError):
            task.at_level(bad)
    assert task.at_level(3).level == 3 and task.at_level(3).K == 8


@pytest.mark.parametrize("level,K", [(1, 2), (2, 4), (3, 8)])
def test_shipped_rungs_are_valid_and_separated(tmp_path, level, K):
    """L1..L3 are the shipped rungs: K = 2^L, the reference hits the ceiling and the tied floor is strictly worse."""
    art = _build(tmp_path, level=level, seed=0)
    task = T.SuperpositionPatch().at_level(level)
    assert task.K == K and task.d == N.feature_dim(K)
    ref = task.evaluate(art, task.reference(art))
    base = task.evaluate(art, task.baseline(art))
    assert ref.valid and ref.s_plus == pytest.approx(1.0, abs=1e-6)
    assert base.s_plus < ref.s_plus - 0.5


def test_default_level_is_l0_k1_and_rungs_are_1_to_3():
    task = T.SuperpositionPatch()
    assert task.level == 0 and task.K == 1 and task.d == 1          # default unversioned build, kept
    assert task.knob_levels == (0, 1, 2, 3)                          # L0 supported (labelled), shipped rungs 1..3
    assert task.at_level(0).level == 0 and task.at_level(0).K == 1   # L0 stays reproducible when asked for


# ------------------------------------------------------------------------------------- generator determinism
def test_generate_is_seed_deterministic_and_seed_sensitive():
    a, b = N.generate(4, seed=7), N.generate(4, seed=7)
    for key in ("enc", "dec", "spur_a", "spur_b"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["labels"] == b["labels"]
    assert not np.array_equal(a["enc"], N.generate(4, seed=8)["enc"])


def test_l0_build_is_byte_identical_for_same_seed(tmp_path):
    t = T.SuperpositionPatch()
    t.build(tmp_path / "a", seed=3)
    t.build(tmp_path / "b", seed=3)
    for rel in ("public/net.npz", "private/reference.npz", "private/baseline.npz", "private/truth.npz"):
        with np.load(tmp_path / "a" / rel) as za, np.load(tmp_path / "b" / rel) as zb:
            assert za.files == zb.files
            for k in za.files:
                assert za[k].tobytes() == zb[k].tobytes()
    assert (tmp_path / "a" / "private" / "spec.json").read_bytes() == (tmp_path / "b" / "private" / "spec.json").read_bytes()


def test_encoder_is_nonnegative_unit_columns():
    g = N.generate(8, seed=1)
    assert np.all(g["enc"] >= 0)
    np.testing.assert_allclose(np.linalg.norm(g["enc"], axis=0), 1.0, atol=1e-12)


# ------------------------------------------------------------------------------------- validity check (K = 1..4)
@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_closed_form_ceiling_matches_numerical_least_squares(K):
    """The closed-form ceiling (K - d) and the closed-form pseudo-inverse match a numerical least-squares recovery
    that is solved from the network's own probe responses (relu applied). Covers K = 3 as a non-power-of-2 check."""
    d = N.feature_dim(K)
    for seed in (0, 1, 2):
        enc = N.generate(K, seed)["enc"]
        numeric = N.numerical_decoder(enc)
        assert N.recovery_error(enc, numeric) == pytest.approx(N.ceiling_error(enc), abs=TOL)
        assert N.ceiling_error(enc) == pytest.approx(K - d, abs=TOL)
        assert np.abs(N.pinv_decoder(enc) - numeric).max() < TOL
        assert N.feature_residuals(enc).sum() == pytest.approx(N.ceiling_error(enc), abs=TOL)
        assert np.all(N.feature_residuals(enc) >= -TOL) and np.all(N.feature_residuals(enc) <= 1 + TOL)


def test_probe_domain_is_linear_so_recovery_error_is_exact():
    enc = N.generate(4, seed=0)["enc"]
    x = N.probes(4) * 0.7                                  # probes with a = 0.7 on the same coordinates
    np.testing.assert_allclose(np.maximum(enc @ x, 0.0), enc @ x)   # relu is the identity on the probe domain


# ------------------------------------------------------------------------------------- scoring (hand-computed)
def test_hand_computed_k2_case():
    enc = np.array([[1.0, 1.0]])                         # d = 1, K = 2
    assert N.ceiling_error(enc) == pytest.approx(1.0)    # K - d
    np.testing.assert_allclose(N.pinv_decoder(enc), [[0.5], [0.5]])
    np.testing.assert_allclose(N.feature_residuals(enc), [0.5, 0.5])
    assert N.recovery_error(enc, np.array([[1.0], [1.0]])) == pytest.approx(2.0)   # tied: [[0,1],[1,0]] -> 2
    assert N.recovery_error(enc, np.array([[0.5], [0.5]])) == pytest.approx(1.0)   # pinv reaches the ceiling


def test_normalised_gain_edge_cases():
    assert N.normalised_gain(4.0, 2.0, 1.0) == pytest.approx(2.0 / 3.0)   # hand value
    assert N.normalised_gain(4.0, 4.0, 1.0) == 0.0                         # delta = 0 gains nothing
    assert N.normalised_gain(4.0, 6.0, 1.0) == 0.0                         # worse edit: negative gain clamped to 0
    assert N.normalised_gain(4.0, 1.0, 1.0) == 1.0                         # reaches the ceiling
    assert N.normalised_gain(4.0, 0.5, 1.0) == 1.0                         # overshoot clipped to 1
    assert N.normalised_gain(1.0, 1.0, 1.0) == 1.0                         # no gap: base already at the ceiling
    assert N.normalised_gain(1.0, 2.0, 1.0) == 0.0                         # no gap and a worse edit: 0


# ------------------------------------------------------------------------------------- task: scoring, baseline, ref
def _build(tmp_path, level, seed=0):
    art = tmp_path / f"L{level}_s{seed}"
    T.SuperpositionPatch().at_level(level).build(art, seed)
    return art


def test_reference_reaches_ceiling_and_beats_baseline_on_small_case(tmp_path):
    art = _build(tmp_path, level=2, seed=0)                          # K = 4, d = 2
    task = T.SuperpositionPatch().at_level(2)
    ref = task.evaluate(art, task.reference(art))
    base = task.evaluate(art, task.baseline(art))
    assert ref.valid and ref.s_plus == pytest.approx(1.0, abs=1e-6)
    assert ref.details["E_edit"] == pytest.approx(ref.details["E_ceiling"], abs=1e-5)
    assert 0.0 < base.s_plus < 1.0 and base.s_plus < ref.s_plus - 0.5   # clear margin


def test_zero_delta_scores_zero_and_invalid_shapes_rejected(tmp_path):
    art = _build(tmp_path, level=2, seed=0)
    task = T.SuperpositionPatch().at_level(2)
    dec_shape = np.load(art / "public" / "net.npz")["W_dec"].shape
    assert task.evaluate(art, Submission("delta_w", delta={"W_dec": np.zeros(dec_shape, np.float32)})).s_plus == 0.0
    assert not task.evaluate(art, Submission("delta_w", delta={"W_dec": np.zeros((2, 2))})).valid
    assert not task.evaluate(art, Submission("delta_w", delta={"W_enc": np.zeros(dec_shape)})).valid
    assert not task.evaluate(art, Submission("delta_w", delta={"W_dec": np.full(dec_shape, np.nan)})).valid


def test_l0_default_build_is_degenerate_for_tied_floor(tmp_path):
    """Documented limitation: at K = 1 the tied floor is already exact, so L = 0 does not separate floor from ceiling."""
    art = _build(tmp_path, level=0, seed=0)
    task = T.SuperpositionPatch()
    assert task.evaluate(art, task.baseline(art)).s_plus == pytest.approx(1.0, abs=1e-6)
    assert task.evaluate(art, task.reference(art)).s_plus == pytest.approx(1.0, abs=1e-6)


def test_knob_json_and_spec_record_level(tmp_path):
    art = _build(tmp_path, level=3, seed=1)
    knob = json.loads((art / "public" / "knob.json").read_text())
    spec = json.loads((art / "private" / "spec.json").read_text())
    assert knob["knob"] == "num_entangled_concepts" and knob["level"] == 3 and knob["value"] == 8
    assert spec["K"] == 8 and spec["d"] == 4 and len(spec["labels"]) == 8


# ------------------------------------------------------------------------------------- prompt and leakage
def test_prompt_is_labelled_synthetic_and_leaks_nothing(tmp_path):
    art = _build(tmp_path, level=2, seed=0)
    prompt = T.SuperpositionPatch().at_level(2).prompt(art)
    assert "SYNTHETIC" in prompt and "K = 4" in prompt and "W_dec: [4, 2]" in prompt
    spec = json.loads((art / "private" / "spec.json").read_text())
    for label in spec["labels"]:
        assert label not in prompt
    assert "spur" not in prompt.lower() and "reference" not in prompt.lower()


def test_public_artifacts_hold_no_private_truth(tmp_path):
    art = _build(tmp_path, level=2, seed=0)
    assert sorted(p.name for p in (art / "public").iterdir()) == ["knob.json", "net.npz"]
    with np.load(art / "public" / "net.npz") as z:
        assert set(z.files) == {"W_enc", "W_dec"}                        # no spur, no reference, no labels
    spec = json.loads((art / "private" / "spec.json").read_text())
    assert all(label not in (art / "public" / "knob.json").read_text() for label in spec["labels"])
    with np.load(art / "public" / "net.npz") as z:
        assert not np.array_equal(z["W_dec"], z["W_enc"].T)              # the spur is present in the shipped decoder


# ------------------------------------------------------------------------------------- runner --level
def test_runner_level_handling(tmp_path, monkeypatch):
    assert runner.nondefault_level("TR02-SUPER-001", None) is None
    assert runner.nondefault_level("TR02-SUPER-001", 0) is None
    assert runner.nondefault_level("TR02-SUPER-001", 3) == 3
    with pytest.raises(ValueError):
        runner.nondefault_level("TR02-SUPER-001", 5)
    monkeypatch.setattr(runner, "ART", tmp_path)
    d = runner.build("TR02-SUPER-001", 0, level=2)
    assert d == tmp_path / "TR02-SUPER-001" / "seed0_L2"
    assert json.loads((d / ".built").read_text())["level"] == 2
    assert json.loads((d / "public" / "knob.json").read_text())["value"] == 4
