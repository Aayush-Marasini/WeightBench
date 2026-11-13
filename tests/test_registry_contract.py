"""Registry-wide task-contract checks.

Parametrized over every task returned by ``weightbench.task.load_all()``:

* determinism: two builds at the same seed give identical ``public/`` content (hashed per file; ``.npz``
  files are hashed by their decoded arrays because zip headers carry timestamps);
* budget: the server's metered-probe path (``server.RunState.probe``) rejects a request once the meter
  reaches the task's own ``budget.n_forward`` / ``budget.n_probe``, and a task with no metered probe has
  zero budget and rejects every probe;
* leakage: ``public/`` holds no file named like a ``private/`` file or with identical content, and a test
  file other than this one that names the task (or its family or track module) also states the sandbox /
  private / profile rule. The second part is a textual proxy, not a semantic proof.

Nothing here runs a subject model or submits a remote compute job. ``weightbench.heavy`` entry points are replaced by
a stub that raises ``_HeavyRequired``, so a build that needs heavy compute is skipped with a reason instead of
running. Builds that need an optional extra that is not installed are skipped the same way.

Known failures are strict xfails: the check body is unchanged, the expected failure is recorded in
``KNOWN_FAILURES``. A fixed task turns the xfail into a
hard failure, which forces the entry to be removed.
"""
from __future__ import annotations

import hashlib
import http.client
import socket
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from weightbench import heavy, server
from weightbench.ledger import Ledger
from weightbench.task import load_all

TASKS = load_all()
TASK_IDS = sorted(TASKS)
THIS_FILE = Path(__file__).resolve()
TESTS_DIR = THIS_FILE.parent

# (task id, check) -> reason. check is one of: determinism, budget, leakage_public, leakage_reference.
_NO_LEAK_TEST = "no sandbox/profile/leak test names this task or its module (dod_matrix: leakage missing or file-presence only)"
KNOWN_FAILURES: dict[tuple[str, str], str] = {
    # Empty: SPARSE/OUTLIER/DEAD leakage tests live in tests/test_task_contract.py.
}

SANDBOX_RULE_WORDS = ("sandbox", "leak", "profile", "access")


class _HeavyRequired(RuntimeError):
    """Raised instead of running a heavy (model-executing) step locally."""


@pytest.fixture
def no_heavy(monkeypatch):
    def blocked(*args, **kwargs):
        raise _HeavyRequired("build needs heavy compute (weightbench.heavy); not run on the laptop")

    for name in ("run_heavy", "pull_dir"):
        monkeypatch.setattr(heavy, name, blocked)

    # Builds fetch pinned checkpoints and datasets lazily (function-level imports). With the ``lm`` extra
    # installed (CI), an unblocked build downloads multi-GB weights and fills the runner's disk. Never
    # download here: such a build is skipped with the same reason as a heavy one.
    def no_download(*args, **kwargs):
        raise _HeavyRequired("build needs a checkpoint or dataset download; not fetched in tests")

    for module, names in (("huggingface_hub", ("snapshot_download", "hf_hub_download")),
                          ("datasets", ("load_dataset",))):
        try:
            mod = __import__(module)
        except ModuleNotFoundError:
            continue
        for name in names:
            monkeypatch.setattr(mod, name, no_download)


def _expect_known_failure(request, tid: str, check: str) -> None:
    reason = KNOWN_FAILURES.get((tid, check))
    if reason:
        request.applymarker(pytest.mark.xfail(strict=True, reason=f"known failure: {reason}"))


def _build_public(tid: str, root: Path, seed: int) -> Path:
    """Build task ``tid`` at ``seed`` into a fresh art dir, or skip with a reason if it cannot run here."""
    art = root / "art"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    try:
        TASKS[tid]().build(art, seed)
    except _HeavyRequired as exc:
        pytest.skip(f"{tid}: {exc}")
    except ModuleNotFoundError as exc:
        pytest.skip(f"{tid}: build needs optional dependency {exc.name!r}, not installed in .venv "
                    "(pyproject extras lm / tr01)")
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
            http.client.HTTPException) as exc:
        # A corpus build (TR01-ESC-001..003 fetch their Gutenberg source) needs the network. A CI runner
        # without reachable DNS, or an egress proxy that stalls, is a property of the runner, not a
        # determinism failure (e.g. minutes of connect timeouts, WinError 10060). Skip with the reason instead of
        # reddening the run.
        pytest.skip(f"{tid}: build needs the network for its corpus and this runner can't reach it ({exc})")
    return art


def _content_digest(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if p.suffix == ".npz":
            h = hashlib.sha256()
            with np.load(p, allow_pickle=False) as z:
                for key in sorted(z.files):
                    arr = np.ascontiguousarray(z[key])
                    h.update(f"{key}|{arr.dtype}|{arr.shape}|".encode())
                    h.update(arr.tobytes())
            out[rel] = h.hexdigest()
        else:
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ------------------------------------------------------------------------------------------ determinism
@pytest.mark.parametrize("tid", TASK_IDS)
def test_build_is_deterministic_for_seed(tid, tmp_path, no_heavy, request):
    _expect_known_failure(request, tid, "determinism")
    first = _build_public(tid, tmp_path / "a", seed=0)
    second = _build_public(tid, tmp_path / "b", seed=0)
    digest_a = _content_digest(first / "public")
    digest_b = _content_digest(second / "public")
    assert digest_a, f"{tid}: build wrote no public files"
    assert digest_a == digest_b, f"{tid}: same seed gave different public files"


# ------------------------------------------------------------------------------------------ budget
@pytest.mark.parametrize("tid", TASK_IDS)
def test_budget_rejects_over_budget_probe(tid, tmp_path, request):
    _expect_known_failure(request, tid, "budget")
    task = TASKS[tid]()
    budget = task.budget
    ledger = Ledger(tmp_path / "ledger.jsonl")
    state = server.RunState(task, tmp_path, tmp_path / "run", ledger, tmp_path)

    if not task.probe_kinds:
        # No metered probe exists, so the declared budget must be zero and any probe must be refused.
        assert (budget.n_forward, budget.n_probe) == (0, 0), (
            f"{tid}: declares n_forward={budget.n_forward}, n_probe={budget.n_probe} but has no probe kinds")
        status, body = state.probe("budget_probe_unavailable", {})
        assert status == 400 and "unknown probe" in body.get("error", ""), f"{tid}: probe not refused: {body}"
        return

    for kind, meter in task.probe_kinds.items():
        cap = budget.n_forward if meter == "forward" else budget.n_probe
        assert cap > 0, f"{tid}: probe {kind!r} meters {meter!r} but the task budget for it is 0"
        state.used[meter] = cap                       # meter already at the task's own cap
        status, body = state.probe(kind, {})
        assert status == 429, f"{tid}: probe {kind!r} over {meter} budget {cap} got {status}: {body}"
        assert f"{meter} budget exhausted ({cap})" in body.get("error", ""), f"{tid}: unclear refusal {body}"
        rejected = list(ledger.entries("probe_rejected"))
        assert rejected and rejected[-1]["payload"] == {"kind": kind, "meter": meter, "cap": cap}, (
            f"{tid}: rejection not written to the ledger")
        assert not list(ledger.entries("probe")), f"{tid}: over-budget probe was metered as accepted"


# ------------------------------------------------------------------------------------------ leakage
@pytest.mark.parametrize("tid", TASK_IDS)
def test_public_dir_has_no_private_artifacts(tid, tmp_path, no_heavy, request):
    _expect_known_failure(request, tid, "leakage_public")
    art = _build_public(tid, tmp_path, seed=0)
    public, private = art / "public", art / "private"
    private_files = [p for p in private.rglob("*") if p.is_file()]
    public_files = [p for p in public.rglob("*") if p.is_file()]
    assert public_files, f"{tid}: build wrote no public files"
    assert not any("private" in p.relative_to(public).parts for p in public_files), (
        f"{tid}: a path under public/ has a private component")
    private_names = {p.name for p in private_files}
    leaked_names = sorted(p.name for p in public_files if p.name in private_names)
    assert not leaked_names, f"{tid}: public/ holds private file names {leaked_names}"
    private_digests = set(_content_digest(private).values())
    leaked_content = sorted(rel for rel, d in _content_digest(public).items() if d in private_digests)
    assert not leaked_content, f"{tid}: public/ files identical to private/ files: {leaked_content}"


def _reference_terms(tid: str) -> tuple[str, str, str]:
    cls = TASKS[tid]
    module_short = cls.__module__.rsplit(".", 1)[-1]
    family = tid.rsplit("-", 1)[0]
    return tid, family, module_short


@pytest.mark.parametrize("tid", TASK_IDS)
def test_sandbox_or_profile_test_names_task_or_track(tid, request):
    _expect_known_failure(request, tid, "leakage_reference")
    terms = _reference_terms(tid)
    matches = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        if path == THIS_FILE:
            continue
        text = path.read_text(encoding="utf-8")
        if any(term in text for term in terms):
            if any(word in text.lower() for word in SANDBOX_RULE_WORDS):
                matches.append(path.name)
    assert matches, (f"{tid}: no other test names the task ({terms[0]}), its family ({terms[1]}) or its module "
                     f"({terms[2]}) together with a sandbox / private / profile rule")


def test_build_public_skips_when_its_corpus_is_unreachable(tmp_path, monkeypatch):
    """A corpus build that cannot reach the network is a runner property, not a determinism failure.

    A ``URLError`` (e.g. ``[WinError 10060]``) from ``_build_public('TR01-ESC-001')`` becomes a skip, so a
    network-less runner does not redden CI.
    """

    class _Unreachable:
        def build(self, art, seed):
            raise urllib.error.URLError("no host given")

    monkeypatch.setitem(TASKS, "FAKE-NET-001", _Unreachable)
    with pytest.raises(pytest.skip.Exception, match="needs the network"):
        _build_public("FAKE-NET-001", tmp_path, 0)
