"""weightbench-manifest: freeze artifact checksums, detect modified / missing / extra files and tasks."""
import json

import pytest

from weightbench import manifest as M
from weightbench.task import Budget, Task

SECRET = "TRIGGER-LAYER-7-SECRET"


class FakeTask(Task):
    id = "TR09-FAKE-001"
    track = "TR01"
    tier = "foundational"
    budget = Budget(n_forward=3, n_probe=5, max_l0=0.01, wall_clock_s=60)
    knob_level = 1


class OtherTask(Task):
    id = "TR09-FAKE-002"
    track = "TR04"
    tier = "frontier"


class SmokeTask(Task):
    id = "SMOKE-999"
    track = "TR01"
    tier = "foundational"


def _build(art, tid, seed):
    d = art / tid / f"seed{seed}"
    (d / "public" / "sub").mkdir(parents=True)
    (d / "private").mkdir()
    (d / "public" / "W.npy").write_bytes(b"\x00\x01" * 1000)
    (d / "public" / "sub" / "meta.json").write_text('{"n": 1}')
    (d / "private" / "gt.json").write_text(json.dumps({"answer": SECRET}))
    (d / ".built").write_text('{"seconds": 0.1}')
    return d


@pytest.fixture
def setup(tmp_path):
    art, out = tmp_path / "artifacts", tmp_path / "manifests"
    tasks = {FakeTask.id: FakeTask, OtherTask.id: OtherTask, SmokeTask.id: SmokeTask}
    seed_dir = _build(art, FakeTask.id, 0)
    _build(art, FakeTask.id, 1)
    (art / FakeTask.id / "seed2" / "public").mkdir(parents=True)        # unfinished build: no .built
    return art, out, tasks, seed_dir


def test_write_is_stable_and_never_contains_private_contents(setup):
    art, out, tasks, _ = setup
    path = M.write("v0.1", tasks=tasks, art_root=art, out_dir=out)
    text = path.read_text()
    assert SECRET not in text
    m = json.loads(text)
    assert set(m["tasks"]) == {FakeTask.id, OtherTask.id}                    # SMOKE skipped
    fake = m["tasks"][FakeTask.id]
    assert fake["budget"] == {"n_forward": 3, "n_probe": 5, "max_l0": 0.01, "max_rank": None,
                              "wall_clock_s": 60, "flop_cap": None}
    assert m["schema_version"] == 2 and m["artifact_policy"] == M.ARTIFACT_POLICY
    assert fake["knob_level"] == 1 and fake["track"] == "TR01" and fake["tier"] == "foundational"
    assert set(fake["seeds"]) == {"0", "1"}                                  # seed2 is not built
    assert set(fake["seeds"]["0"]["public"]) == {"W.npy", "sub/meta.json"}
    assert len(fake["seeds"]["0"]["private"]["gt.json"]) == 64
    assert m["tasks"][OtherTask.id]["seeds"] == {}
    M.write("v0.1", tasks=tasks, art_root=art, out_dir=out)
    assert path.read_text() == text                                          # byte-identical rewrite


def test_seed_filter(setup):
    art, out, tasks, _ = setup
    m = json.loads(M.write("v0.1", seeds=[1], tasks=tasks, art_root=art, out_dir=out).read_text())
    assert set(m["tasks"][FakeTask.id]["seeds"]) == {"1"}


def test_check_passes_then_detects_changes(setup):
    art, out, tasks, seed_dir = setup
    M.write("v0.1", tasks=tasks, art_root=art, out_dir=out)
    rep = M.check("v0.1", tasks=tasks, art_root=art, out_dir=out)
    assert rep.ok and rep.n_checked == 2
    (seed_dir / "public" / "W.npy").write_bytes(b"tampered")
    (seed_dir / "private" / "gt.json").unlink()
    (seed_dir / "public" / "new.txt").write_text("x")
    rep = M.check("v0.1", tasks=tasks, art_root=art, out_dir=out)
    assert not rep.ok
    joined = "\n".join(rep.mismatches)
    assert "seed0/public/W.npy: changed" in joined
    assert "seed0/private/gt.json: missing" in joined
    assert "seed0/public/new.txt: extra file" in joined
    assert "RESULT: FAIL" in M.format_check("v0.1", rep)


def test_check_reports_missing_extra_and_spec_changes(setup):
    art, out, tasks, _ = setup
    M.write("v0.1", tasks=tasks, art_root=art, out_dir=out)

    class Changed(FakeTask):
        budget = Budget(n_forward=99)

    class NewTask(OtherTask):
        id = "TR09-FAKE-003"

    rep = M.check("v0.1", tasks={FakeTask.id: Changed, NewTask.id: NewTask}, art_root=art, out_dir=out)
    assert rep.missing_tasks == [NewTask.id] and rep.extra_tasks == [OtherTask.id]
    assert any("budget" in m for m in rep.mismatches) and not rep.ok


def test_unbuilt_seed_is_skipped(setup):
    art, out, tasks, seed_dir = setup
    M.write("v0.1", tasks=tasks, art_root=art, out_dir=out)
    (seed_dir / ".built").unlink()
    rep = M.check("v0.1", tasks=tasks, art_root=art, out_dir=out)
    assert rep.ok and rep.skipped == [f"{FakeTask.id} seed0"]


def test_chunked_hash_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "big.bin"
    data = bytes(range(256)) * 9000           # > 2 chunks
    p.write_bytes(data)
    assert M.sha256_file(p) == hashlib.sha256(data).hexdigest()
    assert M.sha256_file(tmp_path / "nope") is None


def test_runtime_outputs_do_not_invalidate_frozen_assets(setup):
    art, out, tasks, seed_dir = setup
    scratch = [seed_dir / rel for rel in (
        "private/evals/old/out.json", "private/probes/old/payload.json",
        "private/decoy/private/evals/old/out.json", "private/decoy/private/probes/old/out.json")]
    for path in scratch:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old output")
    M.write("v0.1", tasks=tasks, art_root=art, out_dir=out)
    for path in scratch:
        path.write_text("new output")
        (path.parent / "extra.json").write_text("another output")
    assert M.check("v0.1", tasks=tasks, art_root=art, out_dir=out).ok
    # The decoy's ground truth is still an immutable task asset.
    gt = seed_dir / "private/decoy/private/gt.json"
    gt.write_text("new ground truth")
    assert not M.check("v0.1", tasks=tasks, art_root=art, out_dir=out).ok


@pytest.mark.parametrize("relative", ["public/evals/gt.json", "public/probes/weights.bin",
    "private/decoy/public/probes/weights.bin", "private/decoy/public/private/evals/gt.json",
    "private/evals", "private/probes"])
def test_scratch_filter_does_not_hide_public_assets_or_plain_files(setup, relative):
    art, out, tasks, seed_dir = setup
    path = seed_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("asset")
    M.write("v0.1", tasks=tasks, art_root=art, out_dir=out)
    path.write_text("changed asset")
    assert not M.check("v0.1", tasks=tasks, art_root=art, out_dir=out).ok


@pytest.mark.parametrize("field,value", [
    ("budget", Budget(n_forward=3, n_probe=5, max_l0=0.01, wall_clock_s=60, flop_cap=123)),
    ("submission", "delta_w"), ("probe_kinds", {"attention": "forward"}),
    ("meter_exclude", ("donor.npz",)), ("audit_allowed_flags", ("forward_pass_suspect",))])
def test_enforcement_changes_invalidate_manifest(setup, field, value):
    art, out, tasks, _ = setup
    M.write("v0.1", tasks=tasks, art_root=art, out_dir=out)
    changed = type("Changed", (FakeTask,), {field: value})
    tasks[FakeTask.id] = changed
    rep = M.check("v0.1", tasks=tasks, art_root=art, out_dir=out)
    assert not rep.ok and any(field in msg for msg in rep.mismatches)


def test_legacy_manifest_retains_original_checking_policy(setup):
    art, out, tasks, seed_dir = setup
    scratch = seed_dir / "private/evals/run/out.json"
    scratch.parent.mkdir(parents=True)
    scratch.write_text("old")
    path = M.write("v0.1", tasks=tasks, art_root=art, out_dir=out)
    document = json.loads(path.read_text())
    del document["schema_version"], document["artifact_policy"]
    for tid, cls in M.benchmark_tasks(tasks).items():
        document["tasks"][tid].update(M.task_spec(cls, schema_version=1))
        for key in ("submission", "probe_kinds", "meter_exclude", "audit_allowed_flags"):
            document["tasks"][tid].pop(key)
        for seed in document["tasks"][tid]["seeds"]:
            document["tasks"][tid]["seeds"][seed] = M.seed_entry(art / tid / f"seed{seed}", schema_version=1)
    path.write_text(json.dumps(document))
    assert M.check("v0.1", tasks=tasks, art_root=art, out_dir=out).ok
    scratch.write_text("changed")
    assert not M.check("v0.1", tasks=tasks, art_root=art, out_dir=out).ok


@pytest.mark.parametrize("field,value", [("schema_version", 999), ("schema_version", True),
    ("artifact_policy", "ignore-everything"), ("version", "some-other-version")])
def test_unknown_manifest_policy_or_mismatched_version_is_rejected(setup, field, value):
    art, out, tasks, _ = setup
    path = M.write("v0.1", tasks=tasks, art_root=art, out_dir=out)
    doc = json.loads(path.read_text())
    doc[field] = value
    path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit):
        M.check("v0.1", tasks=tasks, art_root=art, out_dir=out)


def test_frozen_scope_supports_intentional_slice_without_hiding_asset_changes(setup):
    art, out, tasks, seed_dir = setup
    M.write("slice", tasks={FakeTask.id: FakeTask}, art_root=art, out_dir=out)
    default = M.check("slice", tasks=tasks, art_root=art, out_dir=out)
    assert not default.ok and default.missing_tasks == [OtherTask.id]
    frozen = M.check("slice", tasks=tasks, art_root=art, out_dir=out,
                     tasks_from_manifest=True, require_built=True)
    assert frozen.ok and frozen.n_checked == 2 and not frozen.skipped
    (seed_dir / "public/W.npy").write_bytes(b"changed")
    changed = M.check("slice", tasks=tasks, art_root=art, out_dir=out, tasks_from_manifest=True)
    assert not changed.ok and any("W.npy: changed" in item for item in changed.mismatches)


def test_frozen_scope_does_not_accept_unregistered_tasks(setup):
    art, out, tasks, _ = setup
    M.write("slice", tasks={FakeTask.id: FakeTask, OtherTask.id: OtherTask}, art_root=art, out_dir=out)
    report = M.check("slice", tasks={FakeTask.id: FakeTask}, art_root=art, out_dir=out,
                     tasks_from_manifest=True)
    assert not report.ok and report.extra_tasks == [OtherTask.id]


def test_require_built_fails_missing_frozen_seed_but_preserves_partial_copy_default(setup):
    art, out, tasks, seed_dir = setup
    M.write("slice", tasks=tasks, art_root=art, out_dir=out)
    (seed_dir / ".built").unlink()
    assert M.check("slice", tasks=tasks, art_root=art, out_dir=out).ok
    strict = M.check("slice", tasks=tasks, art_root=art, out_dir=out, require_built=True)
    assert not strict.ok and strict.n_checked == 1
    assert strict.skipped == [FakeTask.id + " seed0"]
    assert any("frozen build missing" in item for item in strict.mismatches)


@pytest.mark.parametrize("empty_tasks", [True, False])
def test_require_built_rejects_vacuous_success_without_any_frozen_builds(setup, empty_tasks):
    art, out, _, _ = setup
    selected = {} if empty_tasks else {OtherTask.id: OtherTask}
    M.write("empty", tasks=selected, art_root=art, out_dir=out)
    assert M.check("empty", tasks=selected, art_root=art, out_dir=out).ok
    strict = M.check("empty", tasks=selected, art_root=art, out_dir=out, require_built=True)
    assert not strict.ok and strict.n_checked == 0 and not strict.skipped


def test_cli_checks_slice_and_returns_failure_for_missing_required_build(setup, monkeypatch, capsys):
    art, out, tasks, seed_dir = setup
    M.write("slice", tasks={FakeTask.id: FakeTask}, art_root=art, out_dir=out)
    real_check = M.check
    monkeypatch.setattr(M, "check", lambda version, **kwargs: real_check(
        version, tasks=tasks, art_root=art, out_dir=out, **kwargs))
    with pytest.raises(SystemExit) as default:
        M.main(["check", "--version", "slice"])
    assert default.value.code == 1
    with pytest.raises(SystemExit) as valid:
        M.main(["check", "--version", "slice", "--tasks-from-manifest", "--require-built"])
    assert valid.value.code == 0
    assert "2 built seed(s) checked" in capsys.readouterr().out
    (seed_dir / ".built").unlink()
    with pytest.raises(SystemExit) as absent:
        M.main(["check", "--version", "slice", "--tasks-from-manifest", "--require-built"])
    assert absent.value.code == 1 and "frozen build missing" in capsys.readouterr().out
