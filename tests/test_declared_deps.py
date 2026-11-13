"""Every third-party import in weightbench/ must be declared in pyproject.toml (core or an extra)."""
import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# import name -> distribution name, where they differ
DIST = {"yaml": "pyyaml", "kinetix": "kinetix-env", "jax": "jax"}
# Imported only transitively-guaranteed (declared dependency pulls them in); keep this list short.
TRANSITIVE = {"pandas": "datasets (extra lm)"}


def _declared() -> tuple[set[str], set[str]]:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf8"))["project"]
    name = lambda spec: re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].lower().replace("_", "-")
    core = {name(s) for s in meta["dependencies"]}
    extras = {name(s) for specs in meta["optional-dependencies"].values() for s in specs}
    return core, extras


def _imports() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in (ROOT / "weightbench").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf8"))
        for node in ast.walk(tree):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                    else [])
            for mod in mods:
                top = mod.split(".")[0]
                sibling = (path.parent / f"{top}.py").exists()   # scripted agents import the file copied next to them
                if top not in sys.stdlib_module_names and top != "weightbench" and not sibling:
                    found.setdefault(top, set()).add(path.relative_to(ROOT).as_posix())
    return found


def test_all_third_party_imports_are_declared():
    core, extras = _declared()
    declared = core | extras
    missing = {m: sorted(p)[:2] for m, p in _imports().items()
               if m not in TRANSITIVE and DIST.get(m, m).lower().replace("_", "-") not in declared}
    assert not missing, f"imported but not declared in pyproject.toml: {missing}"


def test_scipy_is_a_core_dependency():
    core, _ = _declared()
    assert "scipy" in core


def test_doctor_checks_scipy():
    assert '"scipy"' in (ROOT / "weightbench" / "doctor.py").read_text(encoding="utf8")


def test_constraints_pin_core_numerics():
    pins = {m.group(1).lower() for m in re.finditer(r"^([A-Za-z0-9_.-]+)==", (ROOT / "constraints.txt").read_text(), re.M)}
    assert {"numpy", "scipy", "torch", "pyyaml", "minatar"} <= pins
