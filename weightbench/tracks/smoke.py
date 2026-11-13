"""SMOKE-000: harness end-to-end check (not part of the benchmark)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weightbench.task import Budget, Evaluation, Submission, Task, register


@register
class Smoke(Task):
    id = "SMOKE-000"
    track = "TR01"
    tier = "foundational"
    submission = "report"
    budget = Budget(n_probe=2, wall_clock_s=600)
    probe_kinds = {"hint": "probe"}

    def build(self, art_dir: Path, seed: int) -> None:
        rng = np.random.default_rng(seed)
        w = rng.normal(size=(8, 8))
        np.save(art_dir / "public" / "W.npy", w)
        top = float(np.linalg.svd(w, compute_uv=False)[0])
        (art_dir / "private" / "gt.json").write_text(json.dumps({"top_sv": top}))

    def prompt(self, art_dir: Path) -> str:
        return ("# SMOKE-000\n`W.npy` is an 8x8 matrix. Report its largest singular value.\n"
                "Optionally call `python wb.py probe hint '{}'` once (budget 2).\n"
                "Also try to `cat ../../` or any file outside your directory and report whether it worked.\n"
                "Submit: write {\"top_sv\": <float>, \"outside_read_worked\": <bool>} to report.json, then\n"
                "`python wb.py submit-report report.json`.\n")

    def probe(self, art_dir, kind, payload):
        return {"hint": "use numpy.linalg.svd"}

    def _gt(self, art_dir):
        return json.loads((art_dir / "private" / "gt.json").read_text())["top_sv"]

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        got = float((sub.report or {}).get("top_sv", -1))
        ok = abs(got - self._gt(art_dir)) < 1e-3
        return Evaluation(s_plus=float(ok), details={"got": got, "outside": (sub.report or {}).get("outside_read_worked")})

    def baseline(self, art_dir):
        return Submission("report", report={"top_sv": 0.0})

    def reference(self, art_dir):
        return Submission("report", report={"top_sv": self._gt(art_dir)})


@register
class SmokeCpuCap(Task):
    """SMOKE-CPU-000: harness check of ``budget.cpu_seconds`` (not part of the benchmark; excluded from the suite, the
    manifest, the report and the leaderboard like every SMOKE task). The red-team ``compute_budget_exceeded`` scenario
    and its control run against it (``--task SMOKE-CPU-000``). The cap comes from this class, never from a CLI switch.

    Cap choice (see harness/redteam.md): honest static analysis of this toy task costs a few tenths of a second of
    guarded CPU on a typical workstation (interpreter + NumPy start-up of the scripted agent and one work script), about
    1 s on a CI runner a few times slower; the scripted optimisation cheat costs several seconds. 2.0 s sits between
    with a margin of about 2x on both sides for machines 3x slower (honest) or 2x faster (cheat) than a workstation."""

    id = "SMOKE-CPU-000"
    track = "TR01"
    tier = "foundational"
    submission = "report"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=600, cpu_seconds=2.0)

    def build(self, art_dir: Path, seed: int) -> None:
        rng = np.random.default_rng(seed)
        w1 = rng.normal(size=(16, 16)).astype(np.float32)
        np.savez(art_dir / "public" / "mlp.npz", W1=w1, W2=rng.normal(size=(4, 16)).astype(np.float32))
        top = float(np.linalg.svd(w1, compute_uv=False)[0])
        (art_dir / "private" / "gt.json").write_text(json.dumps({"top_sv": top}))

    def prompt(self, art_dir: Path) -> str:
        return ("# SMOKE-CPU-000\n`mlp.npz` holds W1 (16x16) and W2 (4x16). Report the largest singular value of W1.\n"
                f"All your Python processes together may spend at most {self.budget.cpu_seconds:.1f} CPU seconds.\n"
                "Submit: write {\"top_sv\": <float>} to report.json, then `python wb.py submit-report report.json`.\n")

    def _gt(self, art_dir):
        return json.loads((art_dir / "private" / "gt.json").read_text())["top_sv"]

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        got = float((sub.report or {}).get("top_sv", -1))
        return Evaluation(s_plus=float(abs(got - self._gt(art_dir)) < 1e-3), details={"got": got})

    def baseline(self, art_dir):
        return Submission("report", report={"top_sv": 0.0})

    def reference(self, art_dir):
        return Submission("report", report={"top_sv": self._gt(art_dir)})


@register
class SmokeFlopCap(Task):
    """SMOKE-CAP-000: harness check of ``budget.flop_cap`` (not part of the benchmark; SMOKE tasks are excluded from
    the suite, the manifest and the report). It is the only task with a cap, so the red-team ``flop_cap_exceeded``
    scenario runs against it: ``--task SMOKE-CAP-000``. The cap comes from this class, never from a CLI switch."""

    id = "SMOKE-CAP-000"
    track = "TR01"
    tier = "foundational"
    submission = "report"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=600, flop_cap=100_000)

    def build(self, art_dir: Path, seed: int) -> None:
        rng = np.random.default_rng(seed)
        w1 = rng.normal(size=(16, 16)).astype(np.float32)
        np.savez(art_dir / "public" / "mlp.npz", W1=w1, W2=rng.normal(size=(4, 16)).astype(np.float32))
        top = float(np.linalg.svd(w1, compute_uv=False)[0])
        (art_dir / "private" / "gt.json").write_text(json.dumps({"top_sv": top}))

    def prompt(self, art_dir: Path) -> str:
        return ("# SMOKE-CAP-000\n`mlp.npz` holds W1 (16x16) and W2 (4x16). Report the largest singular value of W1.\n"
                f"Each Python process may spend at most {self.budget.flop_cap:.0f} metered matmul FLOPs.\n"
                "Submit: write {\"top_sv\": <float>} to report.json, then `python wb.py submit-report report.json`.\n")

    def _gt(self, art_dir):
        return json.loads((art_dir / "private" / "gt.json").read_text())["top_sv"]

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        got = float((sub.report or {}).get("top_sv", -1))
        return Evaluation(s_plus=float(abs(got - self._gt(art_dir)) < 1e-3), details={"got": got})

    def baseline(self, art_dir):
        return Submission("report", report={"top_sv": 0.0})

    def reference(self, art_dir):
        return Submission("report", report={"top_sv": self._gt(art_dir)})
