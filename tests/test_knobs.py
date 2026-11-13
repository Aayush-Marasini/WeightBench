"""Exponential knob formula (weightbench.knobs) and the --level build parameter in the runner."""
import json

import pytest

from weightbench import knobs as K
from weightbench import runner
from weightbench.tracks import tr05_dead as T
from weightbench.tracks.tr05_dead_lib import jobs as J


def test_registry_matches_readme_names():
    assert {t: k.name for t, k in K.KNOBS.items()} == {
        "TR01": "symmetry_and_depth", "TR02": "num_entangled_concepts", "TR03": "hallucination_plausibility",
        "TR04": "algorithmic_horizon", "TR05": "propagation_depth"}
    assert K.get("TR-05") is K.KNOBS["TR05"]
    assert all(k.v0 > 0 and k.growth > 1 for k in K.KNOBS.values())


def test_value_is_v0_times_g_to_the_l():
    tr05 = K.get("TR05")
    assert K.schedule(tr05, 4) == [1, 3, 9, 27, 81]
    assert K.schedule(K.get("TR02"), 4) == [1, 2, 4, 8, 16]
    real = K.Knob("TRXX", "x", 0.5, 1.5, "q", integer=False)
    assert K.value(real, 2) == pytest.approx(0.5 * 1.5 ** 2)
    assert K.value(K.Knob("TRXX", "x", 0.2, 2, "q"), 0) == 1          # counts never round to 0
    for L in range(6):                                                  # each level multiplies by g
        assert K.value(real, L + 1) / K.value(real, L) == pytest.approx(1.5)


def test_level_validation_and_inverse():
    tr05 = K.get("TR05")
    for bad in (-1, 1.5, True, "2"):
        with pytest.raises(ValueError):
            K.value(tr05, bad)
    with pytest.raises(ValueError):
        K.Knob("TRXX", "x", 1, 1.0, "q")
    assert K.level_for(tr05, 27) == 3 and K.level_for(tr05, 1) == 0
    assert K.level_for(tr05, 4) is None and K.level_for(tr05, 0) is None


def test_describe_is_json_ready():
    d = K.describe("TR05", 2)
    assert d["knob"] == "propagation_depth" and d["level"] == 2 and d["value"] == 9 and d["formula"] == "1 * 3**L"
    json.dumps(d)


# ------------------------------------------------------------------------------------- runner --level
def test_default_paths_and_run_ids_unchanged():
    assert runner.art_dir("TR05-DEAD-001", 0) == runner.ART / "TR05-DEAD-001" / "seed0"
    assert runner.art_dir("TR05-DEAD-001", 0, 2) == runner.ART / "TR05-DEAD-001" / "seed0_L2"
    assert runner.make_run_id("T", "baseline", 0, 1) == "T__baseline__s0__r1"
    assert runner.make_run_id("T", "baseline", 0, 1, 3) == "T__baseline__s0__r1__L3"


def test_nondefault_level_normalises_and_validates():
    assert runner.nondefault_level("TR05-DEAD-001", None) is None
    assert runner.nondefault_level("TR05-DEAD-001", 0) is None         # its own level = default artifacts
    assert runner.nondefault_level("TR05-DEAD-002", 1) is None
    assert runner.nondefault_level("TR05-DEAD-002", 0) == 0
    with pytest.raises(ValueError):
        runner.nondefault_level("TR05-DEAD-001", -1)
    assert runner.nondefault_level("TR04-GRAFT-001", 2) == 2            # bounded explicit levels L0..L2
    assert runner.get_task("TR04-GRAFT-001").level is None
    with pytest.raises(ValueError, match="supported knob levels"):
        runner.nondefault_level("TR04-GRAFT-001", 3)


@pytest.mark.parametrize("tid,levels", [("TR04-GAME-001", (0, 1, 2)), ("TR04-GAME-002", (0, 1, 2)),
                                        ("TR04-GAME-003", (0, 1, 2)), ("TR05-OUTLIER-001", (0, 1, 2, 3)),
                                        ("TR04-GO-001", (0, 1, 2)), ("TR05-SPARSE-003", (0, 1, 2)),
                                        ("TR04-GRAFT-001", (0, 1, 2)), ("TR05-OUTLIER-002", (0, 1, 2, 3))])
def test_bounded_tr04_tr05_levels_go_to_level_dirs_and_reject_others(tid, levels):
    assert runner.get_task(tid).level is None                         # unversioned default kept
    for level in levels:                                              # every explicit level (L0 too) is its own dir
        assert runner.nondefault_level(tid, level) == level
        assert runner.art_dir(tid, 0, level).name == f"seed0_L{level}"
    assert runner.art_dir(tid, 0).name == "seed0"
    with pytest.raises(ValueError, match="supported knob levels"):
        runner.nondefault_level(tid, max(levels) + 1)                 # rejected before any build


def test_build_at_level_goes_to_its_own_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ART", tmp_path)
    monkeypatch.setenv("WB_HEAVY", "local")
    monkeypatch.setattr(T, "_heavy", lambda art_dir, step, out_name, extra=(), push=(): J.build(art_dir))
    monkeypatch.setattr(T.DeadOneLayer, "n_inputs", 8)          # toy size
    monkeypatch.setattr(T.DeadOneLayer, "width", 16)
    d = runner.build("TR05-DEAD-001", 0, level=1)
    assert d == tmp_path / "TR05-DEAD-001" / "seed0_L1"
    assert json.loads((d / ".built").read_text())["level"] == 1
    knob = json.loads((d / "public" / "knob.json").read_text())
    assert knob["level"] == 1 and knob["value"] == 3
    assert json.loads((d / "private" / "spec.json").read_text())["hidden"] == [16, 16, 16]
    assert not (tmp_path / "TR05-DEAD-001" / "seed0").exists()


def test_build_rebuilds_an_artifact_that_other_sources_produced(tmp_path, monkeypatch):
    """A build marker stamps the sources that made the artifact; a marker from other code never hides a rebuild.

    Otherwise artifacts built by older sources would be scored as if the current code had produced them. The marker
    binds the artifact to the code that produced it.
    """
    monkeypatch.setattr(runner, "ART", tmp_path)
    monkeypatch.setenv("WB_HEAVY", "local")
    monkeypatch.setattr(T, "_heavy", lambda art_dir, step, out_name, extra=(), push=(): J.build(art_dir))
    monkeypatch.setattr(T.DeadOneLayer, "n_inputs", 8)
    monkeypatch.setattr(T.DeadOneLayer, "width", 16)
    d = runner.build("TR05-DEAD-001", 0, level=1)
    assert json.loads((d / ".built").read_text())["source"] == runner.source_fingerprint()
    (d / "private" / "leftover.txt").write_text("stale")
    assert runner.build("TR05-DEAD-001", 0, level=1) == d
    assert (d / "private" / "leftover.txt").exists()             # same sources: reused, untouched
    monkeypatch.setattr(runner, "source_fingerprint", lambda: "0000000000000000")
    runner.build("TR05-DEAD-001", 0, level=1)
    assert json.loads((d / ".built").read_text())["source"] == "0000000000000000"
    assert not (d / "private" / "leftover.txt").exists()          # other sources: rebuilt from scratch


def test_source_fingerprint_ignores_build_neutral_modules(tmp_path, monkeypatch):
    """Adjudication and agent-launch code cannot change a subject's weights, so editing them must not invalidate
    every artifact (an audit-only change must not force a rebuild of artifacts built elsewhere)."""
    wb = tmp_path / "weightbench"
    (wb / "harness").mkdir(parents=True)
    (wb / "tracks").mkdir()
    (wb / "tracks" / "generator.py").write_text("x = 1\n")
    (wb / "harness" / "audit.py").write_text("banner = 1\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    neutral = set(runner.BUILD_NEUTRAL_SOURCES)
    assert "weightbench/harness/audit.py" in neutral and "weightbench/tracks/generator.py" not in neutral
    fp = runner.source_fingerprint()
    (wb / "harness" / "audit.py").write_text("banner = 2\n")
    assert runner.source_fingerprint() == fp                    # adjudication edit: same artifacts
    (wb / "tracks" / "generator.py").write_text("x = 2\n")
    assert runner.source_fingerprint() != fp                    # weight-determining edit: rebuild


def test_build_reuses_an_artifact_built_under_another_numeric_stack(tmp_path, monkeypatch):
    """An artifact built on another machine and scored here keeps its weights and records which stack built it: rebuilding it
    here would silently substitute a different subject."""
    monkeypatch.setattr(runner, "ART", tmp_path)
    monkeypatch.setenv("WB_HEAVY", "local")
    monkeypatch.setattr(T, "_heavy", lambda art_dir, step, out_name, extra=(), push=(): J.build(art_dir))
    monkeypatch.setattr(T.DeadOneLayer, "n_inputs", 8)
    monkeypatch.setattr(T.DeadOneLayer, "width", 16)
    d = runner.build("TR05-DEAD-001", 0, level=1)
    meta = json.loads((d / ".built").read_text())
    assert meta["numeric"] == runner.numeric_stack()
    (d / "private" / "leftover.txt").write_text("kept")
    monkeypatch.setattr(runner, "numeric_stack", lambda: "numpy==0 torch==0.0.0+other")
    assert runner.build("TR05-DEAD-001", 0, level=1) == d
    assert (d / "private" / "leftover.txt").exists()
    assert json.loads((d / ".built").read_text())["numeric"] != "numpy==0 torch==0.0.0+other"
