"""Frozen task manifest for a benchmark version: task specs plus SHA-256 checksums of every built artifact.

    weightbench-manifest write --version v0.1 [--seeds 0,1,2]   # -> manifests/v0.1.json
    weightbench-manifest check --version v0.1                   # recompute and compare; exit 1 on any difference
    weightbench-manifest check --version <frozen-version> --tasks-from-manifest --require-built

For every registered task (SMOKE tasks excluded): track, tier, budget, knob level, and for every built seed
(artifacts/<task>/seed<s>/.built exists) the SHA-256 of immutable public/private assets (relative path ->
hash; private contents never leave the machine), plus the harness git commit. Schema 2 excludes
private/evals and private/probes scratch, including the nested TR-03 decoy's private scratch,
and freezes the full Budget plus submission, probe and audit rules. Schema 1 keeps its original
all-files / five-budget-fields semantics when checked; it is never silently upgraded.

``check`` reports changed / missing / extra files of built seeds, task specs that changed, tasks registered but not in
the manifest (missing) and tasks in the manifest but no longer registered (extra). Seeds not built on this machine are
skipped. To test that a build is reproducible, rebuild it (`weightbench build --task T --seed S --force`) and check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
MANIFESTS = ROOT / "manifests"
CHUNK = 1 << 20                     # 1 MiB: constant memory however large the model file
SEED_DIR = re.compile(r"^seed(\d+)$")   # default knob level only (seed<s>_L<level> dirs are not the suite)
SECTIONS = ("public", "private")
BUDGET_FIELDS = ("n_forward", "n_probe", "max_l0", "max_rank", "wall_clock_s")
# Budget fields added after a manifest / access-plan schema shipped are omitted while at their default, so existing
# frozen manifests, snapshots and access plans still match; a task that sets one is a real enforcement change.
# Currently ``cpu_seconds``. Used by both the manifest and the comparison access plan so the two agree.
NEW_BUDGET_DEFAULTS = {"cpu_seconds": None}


def budget_record(budget: Any) -> dict:
    return normalize_budget_dict(asdict(budget))


def normalize_budget_dict(d: dict) -> dict:
    """Drop budget fields added after a schema shipped when at their default, so a budget recorded with the new field
    (``asdict``) and one frozen before it existed compare equal. Used by the manifest and the comparison access plan."""
    out = dict(d)
    for name, default in NEW_BUDGET_DEFAULTS.items():
        if out.get(name) == default:
            out.pop(name, None)
    return out
SCHEMA_VERSION = 2
ARTIFACT_POLICY = "immutable-assets-v1"
RUNTIME_DIRS = frozenset(("evals", "probes"))


def sha256_file(path: Path) -> str | None:
    """SHA-256 of a file read in 1 MiB chunks (None if it does not exist)."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path = ROOT) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def _runtime_private_path(relative: Path) -> bool:
    """Reserved scratch under a task's private root, or its nested TR-03 decoy.

    Public directories named evals/probes (including decoy/public/) remain frozen.
    A plain file named 'evals' is not a scratch directory either.
    """
    parts = relative.parts
    while parts[:2] == ("decoy", "private"):
        parts = parts[2:]
    return len(parts) > 1 and parts[0] in RUNTIME_DIRS


def hash_tree(directory: Path, *, private_assets: bool = False) -> dict[str, str]:
    """Relative paths -> hashes; optionally omit reserved private runtime scratch."""
    if not directory.is_dir():
        return {}
    files = sorted(p for p in directory.rglob("*") if p.is_file()
                   and not (private_assets and _runtime_private_path(p.relative_to(directory))))
    return {p.relative_to(directory).as_posix(): sha256_file(p) for p in files}


def built_seeds(task_dir: Path, seeds: Iterable[int] | None = None) -> list[int]:
    """Seeds of ``task_dir`` with a finished build (``.built``), optionally restricted to ``seeds``."""
    if not task_dir.is_dir():
        return []
    found = []
    for d in task_dir.iterdir():
        m = SEED_DIR.match(d.name)
        if m and (d / ".built").exists():
            found.append(int(m.group(1)))
    wanted = None if seeds is None else set(seeds)
    return sorted(s for s in found if wanted is None or s in wanted)


def seed_entry(seed_dir: Path, schema_version: int = SCHEMA_VERSION) -> dict[str, dict[str, str]]:
    return {section: hash_tree(seed_dir / section, private_assets=schema_version >= 2 and section == "private")
            for section in SECTIONS}


def task_spec(task: Any, schema_version: int = SCHEMA_VERSION) -> dict[str, Any]:
    budget = (budget_record(task.budget) if schema_version >= 2
              else {name: getattr(task.budget, name) for name in BUDGET_FIELDS})
    spec = {"track": task.track, "tier": task.tier, "budget": budget,
            "knob_level": getattr(task, "knob_level", None)}
    if schema_version >= 2:
        spec.update(submission=getattr(task, "submission", None),
                    probe_kinds=dict(getattr(task, "probe_kinds", {})),
                    meter_exclude=sorted(getattr(task, "meter_exclude", ())),
                    audit_allowed_flags=sorted(getattr(task, "audit_allowed_flags", ())))
    return spec


def benchmark_tasks(tasks: dict[str, Any] | None = None) -> dict[str, Any]:
    if tasks is None:
        from weightbench.task import load_all

        tasks = load_all()
    return {tid: cls for tid, cls in sorted(tasks.items()) if not tid.startswith("SMOKE")}


def _progress(i: int, n: int, tid: str, seeds: list[int]) -> None:
    built = ", ".join(f"seed{s}" for s in seeds) or "not built"
    print(f"[{i}/{n}] {tid}: {built}", file=sys.stderr, flush=True)


def build_manifest(version: str, seeds: Iterable[int] | None = None, tasks: dict[str, Any] | None = None,
                   art_root: Path = ART) -> dict[str, Any]:
    tasks = benchmark_tasks(tasks)
    seeds = None if seeds is None else list(seeds)
    entries = {}
    for i, (tid, cls) in enumerate(tasks.items(), 1):
        found = built_seeds(art_root / tid, seeds)
        _progress(i, len(tasks), tid, found)
        entry = task_spec(cls)
        entry["seeds"] = {str(s): seed_entry(art_root / tid / f"seed{s}") for s in found}
        entries[tid] = entry
    return {"schema_version": SCHEMA_VERSION, "artifact_policy": ARTIFACT_POLICY,
            "version": version, "harness_commit": git_commit(), "tasks": entries}


def manifest_path(version: str, out_dir: Path = MANIFESTS) -> Path:
    return out_dir / f"{version}.json"


def write(version: str, seeds: Iterable[int] | None = None, tasks: dict[str, Any] | None = None,
          art_root: Path = ART, out_dir: Path = MANIFESTS) -> Path:
    manifest = build_manifest(version, seeds, tasks, art_root)
    path = manifest_path(version, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return path


@dataclass
class CheckReport:
    mismatches: list[str] = field(default_factory=list)
    missing_tasks: list[str] = field(default_factory=list)   # registered, not in the manifest
    extra_tasks: list[str] = field(default_factory=list)     # in the manifest, no longer registered
    skipped: list[str] = field(default_factory=list)         # seeds in the manifest that are not built here
    n_checked: int = 0

    @property
    def ok(self) -> bool:
        return not (self.mismatches or self.missing_tasks or self.extra_tasks)


def compare_hashes(where: str, expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    out = [f"{where}/{p}: changed" for p in sorted(expected.keys() & actual.keys()) if expected[p] != actual[p]]
    out += [f"{where}/{p}: missing" for p in sorted(expected.keys() - actual.keys())]
    out += [f"{where}/{p}: extra file" for p in sorted(actual.keys() - expected.keys())]
    return out


def _check_task(rep: CheckReport, tid: str, cls: Any, entry: dict[str, Any], art_root: Path,
                schema_version: int = SCHEMA_VERSION) -> None:
    spec = task_spec(cls, schema_version)
    for key, value in spec.items():
        if entry.get(key) != value:
            rep.mismatches.append(f"{tid}: {key} manifest={entry.get(key)!r} now={value!r}")
    for seed, frozen in sorted(entry.get("seeds", {}).items()):
        seed_dir = art_root / tid / f"seed{seed}"
        if not (seed_dir / ".built").exists():
            rep.skipped.append(f"{tid} seed{seed}")
            continue
        actual = seed_entry(seed_dir, schema_version)
        for section in SECTIONS:
            rep.mismatches += compare_hashes(f"{tid}/seed{seed}/{section}", frozen.get(section, {}), actual[section])
        rep.n_checked += 1


def check(version: str, tasks: dict[str, Any] | None = None, art_root: Path = ART,
          out_dir: Path = MANIFESTS, *, tasks_from_manifest: bool = False,
          require_built: bool = False) -> CheckReport:
    """Verify assets/rules; optionally scope to frozen IDs and require every frozen build.

    Frozen scope still rejects unregistered task IDs. Defaults retain full-registry checking
    and permit absent local builds for authors checking a partial copy of a full-suite release.
    """
    path = manifest_path(version, out_dir)
    if not path.exists():
        raise SystemExit(f"no manifest {path}; create it with `weightbench-manifest write --version {version}`")
    document = json.loads(path.read_text())
    schema_version = document.get("schema_version", 1)
    if type(schema_version) is not int or schema_version not in (1, SCHEMA_VERSION):
        raise SystemExit(f"unsupported manifest schema {schema_version!r}: {path}")
    if schema_version >= 2 and document.get("artifact_policy") != ARTIFACT_POLICY:
        raise SystemExit(f"unsupported manifest artifact policy: {path}")
    if document.get("version") != version:
        raise SystemExit(f"manifest version {document.get('version')!r} does not match {version!r}: {path}")
    frozen = document["tasks"]
    tasks = benchmark_tasks(tasks)
    if tasks_from_manifest:
        tasks = {tid: cls for tid, cls in tasks.items() if tid in frozen}
    rep = CheckReport(missing_tasks=sorted(tasks.keys() - frozen.keys()),
                      extra_tasks=sorted(frozen.keys() - tasks.keys()))
    common = sorted(tasks.keys() & frozen.keys())
    for i, tid in enumerate(common, 1):
        _progress(i, len(common), tid, built_seeds(art_root / tid))
        _check_task(rep, tid, tasks[tid], frozen[tid], art_root, schema_version)
        if require_built and not frozen[tid].get("seeds"):
            rep.mismatches.append(f"{tid}: manifest contains no frozen builds")
    if require_built:
        if not frozen:
            rep.mismatches.append("manifest contains no tasks")
        rep.mismatches.extend(f"{where}: frozen build missing" for where in rep.skipped)
    return rep


def format_check(version: str, rep: CheckReport) -> str:
    lines = [f"manifest {version}: {rep.n_checked} built seed(s) checked, {len(rep.skipped)} not built here (skipped)"]
    lines += [f"  MISMATCH  {m}" for m in rep.mismatches]
    lines += [f"  MISSING   {t} (registered, not in the manifest)" for t in rep.missing_tasks]
    lines += [f"  EXTRA     {t} (in the manifest, not registered)" for t in rep.extra_tasks]
    lines.append(f"RESULT: {'PASS' if rep.ok else 'FAIL'}")
    return "\n".join(lines)


def parse_seeds(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(s) for s in text.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="weightbench-manifest", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="hash all built artifacts -> manifests/<version>.json")
    w.add_argument("--version", required=True)
    w.add_argument("--seeds", default="", help="comma-separated seeds (default: every built seed)")
    c = sub.add_parser("check", help="recompute hashes and compare with manifests/<version>.json")
    c.add_argument("--version", required=True)
    c.add_argument("--tasks-from-manifest", action="store_true",
                   help="check only frozen task IDs; reject unknown IDs and changed rules/assets")
    c.add_argument("--require-built", action="store_true",
                   help="fail if any frozen task/seed build is absent instead of skipping it")
    a = ap.parse_args(argv)
    if a.cmd == "write":
        path = write(a.version, parse_seeds(a.seeds))
        print(f"wrote {path}")
        return
    rep = check(a.version, tasks_from_manifest=a.tasks_from_manifest, require_built=a.require_built)
    print(format_check(a.version, rep))
    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
