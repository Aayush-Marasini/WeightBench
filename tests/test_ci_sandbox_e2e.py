"""The e2e checker (scripts/ci_sandbox_e2e_check.py) on synthetic run directories: no sandbox, no model."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_sandbox_e2e_check.py"
SPEC = importlib.util.spec_from_file_location("e2e_check", SCRIPT)
e2e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e2e)

GOOD = {
    "honest": dict(agent="cmd", model="ci-honest", valid=True, quarantined=False, audit_flags=[], s_wb=1.0, s_plus=1.0),
    "cheat": dict(agent="cmd", model="ci-cheat", valid=False, quarantined=True, audit_flags=["guard_bypass_attempt"],
                  s_wb=0.0, s_plus=0.0),
    "baseline": dict(agent="baseline", model=None, valid=True, s_wb=0.0, s_plus=0.0),
    "reference": dict(agent="reference", model=None, valid=True, s_wb=1.0, s_plus=1.0)}


def _make(tmp_path, sample=900, **overrides):
    runs = tmp_path / "runs"
    for role, rid in e2e.run_ids(sample).items():
        d = runs / rid
        d.mkdir(parents=True)
        res = {"run_id": rid, "task": e2e.TASK, "track": "TR05", "seed": 0, "sample": sample, "ledger_ok": True,
               **GOOD[role], **overrides.get(role, {})}
        (d / "result.json").write_text(json.dumps(res))
        if role in ("honest", "cheat"):
            (d / "audit.json").write_text("{}")
    return runs


def _check(tmp_path, runs):
    return e2e.check(runs, 900, tmp_path / "lb", clock=lambda paths: None, class_changed=lambda task, start: False)


def test_expected_outcomes_pass(tmp_path):
    assert _check(tmp_path, _make(tmp_path)) == []
    assert (tmp_path / "lb" / "board" / "LEADERBOARD.md").is_file()


@pytest.mark.parametrize("role,change,needle", [
    ("honest", {"quarantined": True, "audit_flags": ["forward_chain_runtime"]}, "honest run quarantined"),
    ("honest", {"s_wb": 0.0}, "expected S_WB > 0"),
    ("cheat", {"quarantined": False, "audit_flags": []}, "cheat was not quarantined"),
    ("cheat", {"audit_flags": ["other_interpreter"]}, "expected flag guard_bypass_attempt"),
    ("baseline", {"s_wb": 0.5}, "expected valid with S_WB ~ 0"),
    ("reference", {"valid": False, "s_wb": 0.0}, "expected valid with S_WB ~ 1"),
    ("reference", {"ledger_ok": False}, "ledger_ok"),
])
def test_each_mismatch_is_reported(tmp_path, role, change, needle):
    errors = _check(tmp_path, _make(tmp_path, **{role: change}))
    assert any(needle in e for e in errors), errors


def test_missing_run_is_reported(tmp_path):
    runs = _make(tmp_path)
    (runs / e2e.run_ids(900)["cheat"] / "result.json").unlink()
    assert any("no result.json" in e for e in _check(tmp_path, runs))


def test_compiler_masking_binds_real_files_not_symlinks(tmp_path, monkeypatch):
    """bwrap fails on a symlink destination whose chain leaves the mounted tree (as on a fresh Ubuntu runner)."""
    import os
    import sys
    import pytest
    if sys.platform == "win32":
        pytest.skip("symlink masking is a Linux sandbox detail")
    from weightbench.harness import backends
    real = tmp_path / "gcc-13"
    real.write_text("#!/bin/sh\n")
    (tmp_path / "alt").symlink_to(real)
    (tmp_path / "cc").symlink_to(tmp_path / "alt")
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    masked, args = set(), []
    for binary in sorted(str(p) for p in (tmp_path / "cc", tmp_path / "gcc-13", tmp_path / "dangling")):
        r = os.path.realpath(binary)
        if os.path.isfile(r) and r not in masked:
            masked.add(r)
            args += ["--ro-bind", "/dev/null", r]
    assert args == ["--ro-bind", "/dev/null", str(real)]
    src = (backends.__file__ and open(backends.__file__, encoding="utf8").read())
    assert "os.path.realpath(binary)" in src and '"--ro-bind", "/dev/null", binary' not in src
