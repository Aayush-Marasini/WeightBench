"""TR04-GRAFT-001: graft construction (algebraic check, no host run), shapes, task scoring, probe, leakage."""
import json
from pathlib import Path

import numpy as np
import pytest

from weightbench.ledger import Ledger
from weightbench.server import RunState
from weightbench.task import Submission
from weightbench.tracks import tr04_graft as T
from weightbench.tracks.tr04_graft_lib import graft as G
from weightbench.tracks.tr04_graft_lib import model as M


def random_host(seed=0):
    rng = np.random.default_rng(seed)
    host = {k: rng.normal(0, 0.3, s).astype(np.float32) for k, s in M.HOST_SHAPES.items()}
    spare = list(M.SPARE)
    host["l1.w_in"][spare] = 0
    host["l1.b_in"][spare] = 0
    host["l1.w_out"][:, spare] = 0
    return host


def random_donor(seed=1):
    rng = np.random.default_rng(seed)
    return {k: rng.normal(0, 1.0, s).astype(np.float32) for k, s in M.DONOR_SHAPES.items()}


def test_shapes_and_sequences():
    assert M.all_sequences(M.QB).shape == (10_000, 5) and set(M.all_sequences(M.QB)[:, -1]) == {M.QB}
    assert len(M.SPARE) >= M.DONOR_HIDDEN + 1
    z, r = M.host_forward(random_host(), M.all_sequences(M.QA)[:7], return_resid=True)
    assert z.shape == (7, 10) and r.shape == (7, 64)


def test_spare_units_are_silent_in_base():
    host = random_host()
    seqs = M.all_sequences(M.QA)[:50]
    _, r = M.host_forward(host, seqs, return_resid=True)
    assert np.all(np.maximum(r @ host["l1.w_in"][list(M.SPARE)].T + host["l1.b_in"][list(M.SPARE)], 0) == 0)


def test_graft_construction_exact_under_exact_readout():
    """With residuals that the read-outs decode exactly, the spare units reproduce the donor on QB
    (after the unembedding) and are silent on QA."""
    rng = np.random.default_rng(3)
    host, donor = random_host(), random_donor()
    readout = G.Readout(rng.normal(size=(10, 64)), rng.normal(size=10), rng.normal(size=(10, 64)),
                        rng.normal(size=10), rng.normal(size=64), 0.3)
    delta = G.graft_delta(host, donor, readout, host_logit_spread=5.0)
    assert set(delta) == {"l1.w_in", "l1.b_in", "l1.w_out"}
    untouched = [u for u in range(M.D_MLP) if u not in M.SPARE]
    assert not delta["l1.w_in"][untouched].any() and not delta["l1.w_out"][:, untouched].any()
    a = np.concatenate([readout.d1, readout.d2, readout.dq[None]])            # [21, 64]
    table = M.donor_forward(donor, *np.meshgrid(np.arange(10), np.arange(10), indexing="ij")).argmax(-1)
    for x1, x2 in [(0, 0), (3, 7), (9, 1), (5, 5)]:
        for is_qb in (0.0, 1.0):
            target = np.concatenate([np.eye(10)[x1] - readout.c1, np.eye(10)[x2] - readout.c2, [is_qb - readout.cq]])
            r = np.linalg.lstsq(a, target, rcond=None)[0]
            hidden = np.maximum(r @ delta["l1.w_in"].T + delta["l1.b_in"], 0)
            graft_logits = (hidden @ delta["l1.w_out"].T) @ host["unembed"].T
            if is_qb:
                assert graft_logits.argmax() == table[x1, x2]
            else:
                assert np.abs(hidden).max() == 0                                # silent on task A


def test_naive_readout_decodes_raw_embeddings():
    host = random_host()
    ro = G.naive_readout(host)
    assert np.allclose(ro.d1 @ host["embed"][4], np.eye(10)[4], atol=1e-5)
    assert ro.dq @ host["embed"][M.QB] + ro.cq == pytest.approx(1.0)
    assert ro.dq @ host["embed"][M.QA] + ro.cq == pytest.approx(0.0, abs=1e-6)


def test_validation():
    assert T.validation_error({"l1.w_in": np.zeros((128, 64))}) is None
    assert "unknown" in T.validation_error({"l2.w_in": np.zeros(1)})
    assert "shape" in T.validation_error({"embed": np.zeros((11, 64))})
    assert "non-finite" in T.validation_error({"pos": np.full((5, 64), np.nan)})


# --------------------------------------------------------------------------- task level (fake heavy)
@pytest.fixture
def built(tmp_path, monkeypatch):
    host, donor = random_host(), random_donor()

    def fake_heavy(art_dir, step, out_name, extra=(), push=()):
        assert step == "build"
        np.savez(art_dir / "public" / "host.npz", **host)
        np.savez(art_dir / "public" / "donor.npz", **donor)
        np.save(art_dir / "private" / "table.npy", np.arange(100).reshape(10, 10) % 10)
        ro = G.naive_readout(host)
        np.savez(art_dir / "private" / "baseline.npz", **G.graft_delta(host, donor, ro, 5.0))
        np.savez(art_dir / "private" / "reference.npz", **G.graft_delta(host, donor, ro, 9.0))
        out = {"base": {"task_a": 0.99, "graft": 0.1}}
        (art_dir / out_name).write_text(json.dumps(out))
        return out

    monkeypatch.setattr(T, "_heavy", fake_heavy)
    art = tmp_path / "art"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    task = T.CircuitGrafting()
    monkeypatch.setenv("WB_HEAVY", "local")
    task.build(art, seed=0)
    return task, art


def test_build_prompt_no_leakage(built):
    task, art = built
    assert sorted(p.name for p in (art / "public").iterdir()) == ["donor.npz", "host.npz"]
    prompt = task.prompt(art)
    assert "probe check" in prompt and "submit-delta delta.npz" in prompt and "0.990" in prompt
    for v in json.loads((art / "private" / "spec.json").read_text())["seeds"].values():
        assert str(v) not in prompt


def test_evaluate_scoring(built, monkeypatch):
    task, art = built
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: {"graft": 0.82, "task_a": 0.97})
    ev = task.evaluate(art, task.reference(art))
    assert ev.valid and ev.s_plus == pytest.approx((0.82 - 0.1) / 0.9) and ev.d_col == pytest.approx(0.02)
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: pytest.fail("must not evaluate invalid ΔW"))
    assert not task.evaluate(art, Submission("delta_w", delta={"pos": np.zeros(3)})).valid
    assert not task.evaluate(art, Submission("report", report={})).valid


def test_baseline_reference_touch_only_spare_units(built):
    task, art = built
    for sub in (task.baseline(art), task.reference(art)):
        assert set(sub.delta) <= {"l1.w_in", "l1.b_in", "l1.w_out", "embed"}


def test_check_probe_file_and_budget(built, tmp_path, monkeypatch):
    task, art = built
    work = tmp_path / "work"
    work.mkdir()
    np.savez(work / "delta.npz", **{"l1.b_in": np.ones(128, np.float32)})
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: {"task_a": 1.0, "graft": 0.5})
    state = RunState(task, art, tmp_path / "run", Ledger(tmp_path / "l.jsonl"), work)
    assert state.probe("check", {"file": "missing.npz", "seed": 1})[0] == 400
    codes = [state.probe("check", {"file": "delta.npz", "seed": 1})[0] for _ in range(T.N_CHECKS + 1)]
    assert codes[:T.N_CHECKS] == [200] * T.N_CHECKS and codes[-1] == 429


def test_sandbox_denies_repo():
    from weightbench import runner

    profile = runner._sandbox_profile(Path("/tmp/wb-work/x"))
    assert runner.sandbox.profile_denies_read(profile, runner.ROOT)


# --------------------------------------------------------------------------- difficulty knob
def test_distractor_is_read_off_the_level_table():
    assert M.DISTRACTORS[0] is None                    # L0 trains the host on task A only: the default
    assert M.distractor(0) is None and M.distractor(1) == M.DISTRACTORS[1]
    assert M.distractor(2) == M.DISTRACTORS[2]
    # the level grades the competing circuit by its share of the QB batches, and the ramp rises with the level
    assert M.DISTRACTORS[1][0] == M.DISTRACTORS[2][0] and 0.0 < M.DISTRACTORS[1][1] < M.DISTRACTORS[2][1] < 1.0
    with pytest.raises(ValueError):
        M.distractor(-1)
    with pytest.raises(ValueError):
        M.distractor(len(M.DISTRACTORS))


def test_distractor_targets_are_functions_of_the_digits():
    x1 = np.arange(10)[:, None] * np.ones((1, 10), int)
    x2 = np.ones((10, 10), int) * np.arange(10)[None, :]
    digits = np.stack([x1, x2, np.full_like(x1, 3), np.full_like(x1, 7)], -1)
    assert M.distractor_target(None, digits) is None
    assert np.array_equal(M.distractor_target("x1", digits), x1)
    assert np.array_equal(M.distractor_target("prod", digits), (x1 * x2) % 10)
    assert np.array_equal(M.distractor_target("sum", digits), (x1 + x2) % 10)
    assert np.array_equal(M.distractor_target("diff", digits), (x1 - x2) % 10)
    assert np.array_equal(M.distractor_target("sum3", digits), (x1 + x2 + 3) % 10)   # reads the third token
    assert np.array_equal(M.distractor_target("x3", digits), np.full((10, 10), 3))
    assert np.array_equal(M.distractor_target("const", digits), np.zeros((10, 10), int))


def test_graft_knob_levels_wired_and_default_kept():
    from weightbench import knobs, runner
    from weightbench.task import load_all

    assert T.CircuitGrafting.knob_levels == (0, 1, 2)
    assert "TR04-GRAFT-001" in knobs.get("TR04").wired
    assert load_all()["TR04-GRAFT-001"]().level is None
    for level in (0, 1, 2):
        assert runner.nondefault_level("TR04-GRAFT-001", level) == level
        assert runner.art_dir("TR04-GRAFT-001", 0, level).name == f"seed0_L{level}"
    with pytest.raises(ValueError, match="supported knob levels"):
        load_all()["TR04-GRAFT-001"]().at_level(3)


@pytest.fixture
def fake_build(tmp_path, monkeypatch):
    seen = []

    def fake_heavy(art_dir, step, out_name, extra=(), push=()):
        seen.append((art_dir, step))
        (art_dir / "public").mkdir(parents=True, exist_ok=True)
        (art_dir / "private").mkdir(parents=True, exist_ok=True)
        np.savez(art_dir / "public" / "host.npz", **random_host())
        np.savez(art_dir / "public" / "donor.npz", **random_donor())
        np.save(art_dir / "private" / "table.npy", np.zeros((10, 10), int))
        (art_dir / out_name).write_text(json.dumps({"base": {"task_a": 0.99, "graft": 0.1}}))
        return {}
    monkeypatch.setattr(T, "_heavy", fake_heavy)
    monkeypatch.setenv("WB_HEAVY", "local")
    return seen


def test_build_level_exposes_knob_in_spec_and_public_knob(tmp_path, fake_build):
    t = T.CircuitGrafting()
    default, level2 = tmp_path / "default", tmp_path / "l2"
    (default / "private").mkdir(parents=True)
    (level2 / "private").mkdir(parents=True)
    t.build(default, seed=4)
    t.at_level(2).build(level2, seed=4)
    s0 = json.loads((default / "private" / "spec.json").read_text())
    s2 = json.loads((level2 / "private" / "spec.json").read_text())
    assert "level" not in s0 and "distractor" not in s0 and not (default / "public" / "knob.json").exists()
    assert s2["level"] == 2 and s2["distractor"] == T.distractor_record(2) and s2["seeds"] == s0["seeds"]
    kn = json.loads((level2 / "public" / "knob.json").read_text())
    assert kn["level"] == 2 and kn["value"] == 4 and kn["distractor"] == T.distractor_record(2)
    prompt = t.at_level(2).prompt(level2)
    assert "the host also serves a QB circuit" in prompt and "Difficulty knob" in prompt
    assert "Difficulty knob" not in t.prompt(default)          # the default text is untouched


def test_jobs_build_hands_the_level_distractor_to_the_host(monkeypatch, tmp_path):
    from weightbench.tracks.tr04_graft_lib import jobs as J

    seen = {}

    def fake_host(seed, steps=4000, distract=None):
        seen["host"] = (seed, distract)
        return random_host()
    def fake_donor(table, seed, steps=3000):
        seen["donor"] = (seed, np.shape(table))
        return random_donor()

    monkeypatch.setattr(M, "train_host", fake_host)
    monkeypatch.setattr(M, "train_donor", fake_donor)
    for level, want in ((0, None), (2, M.DISTRACTORS[2])):
        art = tmp_path / f"L{level}"
        (art / "public").mkdir(parents=True)
        (art / "private").mkdir()
        spec = {"seeds": {"host": 1, "donor": 2, "table": 3}}
        if level:
            spec.update(level=level, distractor=T.distractor_record(level))
        (art / "private" / "spec.json").write_text(json.dumps(spec))
        out = J.build(art)
        assert seen["host"][1] == want and out["distractor"] == want and out["level"] == level
        # the target table is the same 10x10 table at every level: the knob moves the channel, not the function
        assert np.load(art / "private" / "table.npy").shape == (10, 10)


def test_build_rejects_unsupported_level_before_heavy(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: pytest.fail("reached the heavy job"))
    t = T.CircuitGrafting()
    t._level_override = 3                                               # bypass at_level: build must still reject
    (tmp_path / "private").mkdir()
    with pytest.raises(ValueError, match="supported knob levels"):
        t.build(tmp_path, seed=0)
