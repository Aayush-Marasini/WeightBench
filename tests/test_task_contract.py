"""Registry-wide task-contract tests (budget enforcement, leakage, knob guards) that need no model run.

* budget enforcement: for every task with a metered probe, ``cap`` probes are accepted through
  ``server.RunState.probe`` and every request past the cap is rejected, recorded as ``probe_rejected`` in the
  ledger, and never reaches ``Task.probe`` (replaced by a counting fake: no subject model executes);
* zero-budget tasks (no probe kinds): no spelling of any probe kind reaches ``Task.probe`` or any meter;
* TR05 SPARSE/OUTLIER leakage: the real ``build`` code runs with the model install and the heavy jobs replaced
  by stubs that write tiny fake files, and ``public/`` is checked against an allow-list; the agent sandbox profile
  denies the repo tree that holds every ``private/`` directory;
* TR05 tasks without a wired knob refuse non-default levels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weightbench import runner, server
from weightbench.ledger import Ledger
from weightbench.task import load_all

TASKS = load_all()
METERED = sorted(t for t, c in TASKS.items() if c().probe_kinds)
ZERO_BUDGET = sorted(t for t, c in TASKS.items() if not c().probe_kinds)


def _state(tid: str, tmp_path: Path):
    task = TASKS[tid]()
    ledger = Ledger(tmp_path / "ledger.jsonl")
    return task, ledger, server.RunState(task, tmp_path, tmp_path / "run", ledger, tmp_path)


# ------------------------------------------------------------------------------------------ budget
@pytest.mark.parametrize("tid", METERED)
def test_cap_probes_accepted_then_rejected_without_execution(tid, tmp_path):
    for kind, meter in TASKS[tid]().probe_kinds.items():
        task, ledger, state = _state(tid, tmp_path / kind)
        executed: list[str] = []
        task.probe = lambda art_dir, k, payload: executed.append(k) or {"fake": True}   # no subject model
        cap = task.budget.n_forward if meter == "forward" else task.budget.n_probe
        assert cap > 0, f"{tid}: {kind!r} meters {meter!r} with a zero budget"
        codes = [state.probe(kind, {})[0] for _ in range(cap + 2)]
        assert codes == [200] * cap + [429] * 2, f"{tid}/{kind}: {codes}"
        assert len(executed) == cap, f"{tid}/{kind}: a rejected request still ran the task"
        assert state.used[meter] == cap, f"{tid}/{kind}: meter moved past the cap"
        rejected = [e["payload"] for e in ledger.entries("probe_rejected")]
        assert rejected == [{"kind": kind, "meter": meter, "cap": cap}] * 2, f"{tid}/{kind}: rejection not ledgered"
        accepted = [e["payload"]["n_used"] for e in ledger.entries("probe")]
        assert accepted == list(range(1, cap + 1))


ALL_KINDS = sorted({k for c in TASKS.values() for k in c().probe_kinds} | {"forward", "probe", "generate", "check"})


@pytest.mark.parametrize("tid", ZERO_BUDGET)
def test_zero_budget_task_serves_no_probe(tid, tmp_path):
    task, ledger, state = _state(tid, tmp_path)
    assert (task.budget.n_forward, task.budget.n_probe) == (0, 0)
    called: list[str] = []
    task.probe = lambda art_dir, kind, payload: called.append(kind) or {}
    for kind in ALL_KINDS:
        status, body = state.probe(kind, {})
        assert status == 400 and "unknown probe" in body["error"], f"{tid}/{kind}: {status} {body}"
    assert not called, f"{tid}: a probe reached the task: {called}"
    assert state.used == {"forward": 0, "probe": 0}
    assert not list(ledger.entries("probe"))


def test_zero_budget_probe_attempt_is_ledgered(tmp_path):
    failures = []
    for tid in ZERO_BUDGET:
        _, ledger, state = _state(tid, tmp_path / tid)
        state.probe("generate", {})
        if not list(ledger.entries("probe_rejected")) and not list(ledger.entries("access_rejected")):
            failures.append(tid)
    assert not failures, f"{len(failures)} zero-budget tasks leave no ledger trace of a refused probe"


# ------------------------------------------------------------------------------------------ TR05 leakage
SPARSE_IDS = [f"TR05-SPARSE-00{i}" for i in range(1, 5)]
OUTLIER_IDS = ["TR05-OUTLIER-001", "TR05-OUTLIER-002"]
SENTINEL_NLL = 7.123456789


@pytest.fixture
def no_download(monkeypatch):
    """Never fetch a checkpoint or dataset (same rule as ``no_heavy`` in test_registry_contract.py)."""
    def blocked(*args, **kwargs):
        raise RuntimeError("download attempted in a test")

    from weightbench import heavy
    for name in ("run_heavy", "pull_dir"):
        monkeypatch.setattr(heavy, name, blocked)
    for module, names in (("huggingface_hub", ("snapshot_download", "hf_hub_download")),
                          ("datasets", ("load_dataset",))):
        try:
            mod = __import__(module)
        except ModuleNotFoundError:
            continue
        for name in names:
            monkeypatch.setattr(mod, name, blocked)


def _tensor_names(tid: str) -> list[str]:
    from weightbench.tracks.tr05_sparsity_lib import config as C

    task = TASKS[tid]()
    if tid.startswith("TR05-OUTLIER"):
        return list(task.editable)
    names = [C.mlp(l, p) for l in range(C.N_LAYERS) for p in C.MLP_PROJS]
    names += [f"{C.PREFIX}layers.3.self_attn.q_proj.weight", f"{C.PREFIX}layers.0.linear_attn.in_proj_qkv.weight"]
    return names


def _fake_build(tid: str, tmp_path: Path, monkeypatch) -> Path:
    """Run the real ``build`` with the model install and the heavy jobs replaced by tiny-file stubs."""
    from safetensors.numpy import save_file

    from weightbench.tracks import tr05_outlier, tr05_sparsity
    from weightbench.tracks.tr05_sparsity_lib import config as C

    def install(public_dir: Path) -> Path:
        model = public_dir / "model"
        model.mkdir(exist_ok=True)
        save_file({n: np.zeros((4, 4), np.float32) for n in _tensor_names(tid)}, str(model / C.WEIGHTS_FILE))
        for f in C.MODEL_FILES:
            if f != C.WEIGHTS_FILE:
                (model / f).write_text("{}")
        return model

    def heavy(art_dir, step, out_name, extra=(), push=()):
        out = Path(art_dir) / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"nll": SENTINEL_NLL, "runtime": "stub", "step": step}))
        priv = Path(art_dir) / "private"
        np.savez(priv / "baseline_delta.npz", w=np.ones(2, np.float32))
        np.savez(priv / "reference_delta.npz", w=np.ones(2, np.float32))
        (priv / "evals").mkdir(exist_ok=True)
        (priv / "evals" / "x.json").write_text("{}")
        return {"nll": SENTINEL_NLL, "ppl": 1.0, "runtime": "stub"}

    for mod in (tr05_sparsity, tr05_outlier):
        monkeypatch.setattr(mod, "install_model", install, raising=False)
        monkeypatch.setattr(mod, "_heavy", heavy, raising=False)
    monkeypatch.setattr(tr05_sparsity, "_pull_private", lambda art_dir: None)
    monkeypatch.setenv("WB_HEAVY", "local")
    art = tmp_path / "art"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    TASKS[tid]().build(art, 0)
    return art


PUBLIC_ALLOWED_TOP = {"model", "targets.json", "editable.json", "knob.json"}
PRIVATE_NAME_HINTS = ("calibration", "reference", "baseline", "spec", "eval", "delta", "prepare", "label", "answer")


@pytest.mark.parametrize("tid", SPARSE_IDS + OUTLIER_IDS)
def test_tr05_public_layout_has_no_private_artifact(tid, tmp_path, monkeypatch, no_download):
    from weightbench.tracks.tr05_sparsity_lib import config as C

    art = _fake_build(tid, tmp_path, monkeypatch)
    public, private = art / "public", art / "private"
    private_files = [p for p in private.rglob("*") if p.is_file()]
    assert {p.name for p in private_files} >= {"calibration.json", "spec.json"}, "stub build wrote no private truth"
    top = {p.name for p in public.iterdir()}
    assert top <= PUBLIC_ALLOWED_TOP, f"{tid}: unexpected public entries {sorted(top - PUBLIC_ALLOWED_TOP)}"
    assert {p.name for p in (public / "model").iterdir()} <= set(C.MODEL_FILES)
    public_files = [p for p in public.rglob("*") if p.is_file()]
    for p in public_files:
        rel = p.relative_to(public)
        assert "private" not in rel.parts and p.suffix != ".npz", f"{tid}: {rel}"
        if rel.parts[0] != "model":
            assert not any(h in p.name for h in PRIVATE_NAME_HINTS), f"{tid}: private-looking public file {rel}"
            assert str(SENTINEL_NLL) not in p.read_text(), f"{tid}: {rel} carries a calibration value"
    assert not {p.name for p in public_files} & {p.name for p in private_files}


@pytest.mark.parametrize("tid", SPARSE_IDS + OUTLIER_IDS)
def test_tr05_private_dir_is_under_sandbox_denied_tree(tid):
    prof = runner._sandbox_profile(Path("/tmp/wb-work/x"))
    assert runner.sandbox.profile_denies_read(prof, runner.ROOT)
    private = (runner.art_dir(tid, 0) / "private").resolve()
    assert private.is_relative_to(runner.ROOT.resolve()), f"{tid}: private/ is outside the denied repo tree"
    assert "private" not in (runner.art_dir(tid, 0) / "public").parts


# ------------------------------------------------------------------------------------------ TR05 knob guards
def test_graft_knob_is_wired_with_a_bounded_ladder():
    # GRAFT-001 has bounded explicit levels L0..L2 (a level-dependent QB distractor circuit, see knobs.py).
    assert TASKS["TR04-GRAFT-001"]().at_level(2).level == 2
    assert TASKS["TR04-GRAFT-001"]().level is None
    with pytest.raises(ValueError, match="supported knob levels"):
        TASKS["TR04-GRAFT-001"]().at_level(3)


def test_sparse003_input_side_ladder_is_distinct_and_bounded():
    from weightbench.tracks.tr05_sparsity_lib import config as C

    assert [C.ladder_layers(L, "TR05-SPARSE-003") for L in (0, 1, 2)] == [[0], [0, 1, 2], list(range(9))]
    for L in (0, 1, 2):                                   # never the same layers as any SPARSE-001 rung
        assert all(C.ladder_layers(L, "TR05-SPARSE-003") != C.ladder_layers(k) for k in range(4))
        assert C.task_targets("TR05-SPARSE-003", L) == [C.mlp(l, p) for l in C.ladder_layers(L, "TR05-SPARSE-003")
                                                        for p in C.MLP_PROJS]
    with pytest.raises(ValueError, match="supported knob levels"):
        C.ladder_layers(3, "TR05-SPARSE-003")             # L3 = all 24 = default = SPARSE-001 L3: not offered
    with pytest.raises(ValueError, match="supported knob levels"):
        TASKS["TR05-SPARSE-003"]().at_level(3)
    t = TASKS["TR05-SPARSE-003"]()
    assert t.level is None and C.task_targets("TR05-SPARSE-003") == C.TASKS["TR05-SPARSE-003"]["targets"]
    assert t.spec(0, ["x"]) == {"task": "TR05-SPARSE-003", "seed": 0, "targets": ["x"],
                                "model": [C.MODEL_ID, C.MODEL_REV], "dataset": C.DATASET, "seq_len": C.SEQ_LEN}
    t = TASKS["TR05-SPARSE-003"]().at_level(1)
    assert t.spec(0, ["x"])["layers"] == [0, 1, 2] and t.knob_record()["layers"] == [0, 1, 2]
    assert "layers 0-2 (knob level 1)" in t._title()


@pytest.mark.parametrize("tid", OUTLIER_IDS)
def test_outlier_levels_bounded_and_layers_from_level(tid):
    from weightbench.tracks import tr05_outlier as O

    assert TASKS[tid]().level is None and TASKS[tid]().layers is None        # default: unversioned, all layers
    for bad in (4, 5):
        with pytest.raises(ValueError, match="supported knob levels"):
            TASKS[tid]().at_level(bad)
    # hand-checked: last min(24, 3^L) layers
    assert O.level_layers(0) == [23] and O.level_layers(1) == [21, 22, 23]
    assert O.level_layers(2) == list(range(15, 24)) and O.level_layers(3) == list(range(24))
    t = TASKS[tid]().at_level(1)
    assert {O._layer_of(n) for n in t.level_quantized} == {21, 22, 23}
    assert {O._layer_of(n) for n in t.level_editable} == {21, 22, 23}
    per_tensor = len(TASKS[tid].quantized) // 24
    assert len(t.level_quantized) == 3 * per_tensor
    assert TASKS[tid]().at_level(3).level_quantized == list(TASKS[tid].quantized)   # L3 = the default set


@pytest.mark.parametrize("tid", OUTLIER_IDS)
def test_outlier_level_build_layout_and_spec(tid, tmp_path, monkeypatch, no_download):
    from weightbench.tracks import tr05_outlier

    default = _fake_build(tid, tmp_path / "d", monkeypatch)
    spec = json.loads((default / "private" / "spec.json").read_text())
    assert "level" not in spec and "layers" not in spec and not (default / "public" / "knob.json").exists()
    assert spec["quantized"] == list(TASKS[tid].quantized)

    real = TASKS[tid]

    class AtLevel(real):          # _fake_build instantiates TASKS[tid](); give it level 0
        def __init__(self):
            self.at_level(0)

    monkeypatch.setitem(TASKS, tid, AtLevel)
    art = _fake_build(tid, tmp_path / "l0", monkeypatch)
    spec = json.loads((art / "private" / "spec.json").read_text())
    knob = json.loads((art / "public" / "knob.json").read_text())
    assert spec["level"] == 0 and spec["layers"] == [23] and knob["layers"] == [23] and knob["value"] == 1
    assert all(tr05_outlier._layer_of(n) == 23 for n in spec["quantized"])
    editable = json.loads((art / "public" / "editable.json").read_text())
    assert editable and all(tr05_outlier._layer_of(n) == 23 for n in editable)
    assert str(SENTINEL_NLL) not in (art / "public" / "knob.json").read_text()


def test_outlier_jobs_restrict_scales_to_level_layers():
    from weightbench.tracks.tr05_outlier_lib import jobs as OJ

    scales = {layer: np.full(2, float(layer)) for layer in range(24)}
    assert OJ._only_level_layers(scales, {}) is scales                         # default build: untouched
    kept = OJ._only_level_layers(scales, {"layers": [21, 22, 23]})
    assert sorted(kept) == [21, 22, 23] and kept[22][0] == 22.0


# ------------------------------------------------------------------------------------------ TR05 DEAD leakage
@pytest.mark.parametrize("tid", ["TR05-DEAD-001", "TR05-DEAD-002"])
def test_dead_public_layout_has_no_private_artifact(tid, tmp_path, monkeypatch, no_download):
    """Real ``build`` with the heavy job stubbed by one that writes the same file names with fake content."""
    from weightbench.tracks import tr05_dead

    def heavy(art_dir, step, out_name, extra=(), push=()):
        art = Path(art_dir)
        np.savez(art / "public" / "mlp.npz", W1=np.zeros((2, 2), np.float32))
        (art / "private" / "truth.json").write_text(json.dumps({"dead": [[0]], "ibp_dead": [[0]]}))
        np.savez(art / "private" / "reference.npz", W1=np.ones((2, 2), np.float32))
        np.savez(art / "private" / "baseline.npz", W1=np.full((2, 2), 2, np.float32))
        (art / "private" / "build.json").write_text("{}")
        return {}

    monkeypatch.setattr(tr05_dead, "_heavy", heavy)
    monkeypatch.setenv("WB_HEAVY", "local")
    art = tmp_path / "art"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    TASKS[tid]().build(art, 0)
    public = {p.name for p in (art / "public").rglob("*") if p.is_file()}
    private = {p.name for p in (art / "private").rglob("*") if p.is_file()}
    assert {"truth.json", "reference.npz", "baseline.npz", "spec.json"} <= private
    assert public == {"mlp.npz", "knob.json"}, f"{tid}: public layout {sorted(public)}"
    assert not public & private


@pytest.mark.parametrize("tid", ["TR05-DEAD-001", "TR05-DEAD-002"])
def test_dead_private_dir_is_under_sandbox_denied_tree(tid):
    prof = runner._sandbox_profile(Path("/tmp/wb-work/x"))
    assert runner.sandbox.profile_denies_read(prof, runner.ROOT)
    assert (runner.art_dir(tid, 0) / "private").resolve().is_relative_to(runner.ROOT.resolve())
