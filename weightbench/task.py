"""Task contract. Every track module registers Task subclasses.

Artifact layout produced by ``Task.build(art_dir, seed)``:

    art_dir/public/    everything the agent may see (weights as .npy/.npz/.safetensors,
                       env source, tokenizer metadata, ...). Copied into the agent workdir.
    art_dir/private/   ground truth, reference ΔW, eval data. Never leaves the server.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

Track = Literal["TR01", "TR02", "TR03", "TR04", "TR05"]
Tier = Literal["foundational", "intermediate", "frontier"]
SubmissionType = Literal["report", "delta_w", "refusal_or_delta_w"]


@dataclass(frozen=True)
class Budget:
    n_forward: int = 0            # metered full rollouts / generations
    n_probe: int = 0              # metered intermediate probes (e.g. attention maps)
    max_l0: float | None = None   # max fraction of nonzero ΔW entries (None = unconstrained)
    max_rank: int | None = None   # alternative constraint: every modified matrix has rank ≤ this
    wall_clock_s: int = 3600      # agent time limit
    flop_cap: float | None = None # per-process cap on metered matmul FLOPs in the agent sandbox (None = trace only)
    cpu_seconds: float | None = None  # cap on total guarded-process CPU seconds (user+sys), form-independent compute
                                      # budget (None = record only; Linux/WSL sandbox path). Needs per-task calibration.


@dataclass
class Submission:
    type: Literal["report", "delta_w", "refusal"]
    report: dict[str, Any] | None = None
    delta: dict[str, np.ndarray] | None = None

    @classmethod
    def load(cls, sub_dir: Path) -> "Submission | None":
        meta = sub_dir / "submission.json"
        if not meta.exists():
            return None
        m = json.loads(meta.read_text())
        delta = None
        if m["type"] == "delta_w":
            with np.load(sub_dir / "delta.npz") as z:
                delta = {k: z[k] for k in z.files}
        return cls(type=m["type"], report=m.get("report"), delta=delta)


@dataclass
class Evaluation:
    """What Task.evaluate returns. The runner combines it with ledger counts into S_WB."""
    s_plus: float                                # already clamped to [0, 1]
    d_col: float = 0.0                           # collateral damage in [0, 1]
    n_eval_forward: int = 0                      # forwards used by the *evaluator* if counted
    valid: bool = True                           # False => budget/constraint violation => S_WB = 0
    details: dict[str, Any] = field(default_factory=dict)


class Task:
    """Subclass per task. Keep heavy imports inside methods so registry import is cheap."""

    id: ClassVar[str]
    track: ClassVar[Track]
    tier: ClassVar[Tier]
    submission: ClassVar[SubmissionType]
    budget: ClassVar[Budget] = Budget()
    seed: ClassVar[int] = 0
    # Registered default; None preserves an older, unversioned default build.
    knob_level: ClassVar[int | None] = None
    knob_levels: ClassVar[tuple[int, ...] | None] = None  # explicitly supported levels for bounded subjects
    _level_override: int | None = None

    # ---- difficulty knob ---------------------------------------------------
    @property
    def level(self) -> int | None:
        """The knob level this instance builds at (the task's default unless ``at_level`` changed it)."""
        if self._level_override is None:
            return self.knob_level
        return self._level_override

    def at_level(self, level: int | None) -> "Task":
        """Build at knob level ``level`` instead of the default (None = default). Only for tasks whose knob is wired."""
        if level is None:
            self._level_override = None
            return self
        from weightbench.knobs import check_level

        level = check_level(level)
        if self.knob_level is None and self.knob_levels is None:
            raise ValueError(f"task {self.id}: its knob is not a build parameter (no --level support yet)")
        if self.knob_levels is not None and level not in self.knob_levels:
            raise ValueError(f"task {self.id}: supported knob levels are {self.knob_levels}, got {level}")
        self._level_override = None if level == self.knob_level else level
        return self

    # ---- generation -------------------------------------------------------
    def build(self, art_dir: Path, seed: int) -> None:
        """Write public/ and private/ artifacts for ``seed`` at knob level ``self.level``."""
        raise NotImplementedError

    def prompt(self, art_dir: Path) -> str:
        """Agent-visible task text (becomes TASK.md). Must not leak ground truth."""
        raise NotImplementedError

    # ---- metered probes (optional) ----------------------------------------
    probe_kinds: ClassVar[dict[str, str]] = {}   # kind -> "forward" | "probe", plus doc in prompt
    # Public weight files the prompt explicitly lets the agent evaluate (e.g. a donor network whose lookup table may be
    # read off by running it). Their shapes are left out of the runtime guard's forward-chain detection.
    meter_exclude: ClassVar[tuple[str, ...]] = ()
    # Audit flags this task's rules explicitly permit (e.g. TR02-CTRL lets the agent evaluate the policy on single
    # observations). They are still reported in audit.json but do not quarantine the run.
    audit_allowed_flags: ClassVar[tuple[str, ...]] = ()

    def probe(self, art_dir: Path, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise KeyError(f"task {self.id} has no probe {kind!r}")

    # ---- scoring -----------------------------------------------------------
    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray] | None:
        """Base parameters ΔW applies to (write tasks). Used for edit-cost + constraint checks."""
        return None

    def frobenius_reference(self, art_dir: Path) -> dict[str, np.ndarray] | None:
        """TR-04 from-zero synthesis (SPEC §3): weights that normalise ‖ΔW‖_F. When given, C(ΔW)
        uses only the λ3 term. None (default) = the usual L0 + rank + Frobenius cost vs. the base."""
        return None

    def rescore_s_plus(self, details: dict[str, Any]) -> float | None:
        """Recompute S+ from the measured components an evaluation stored in the ledger (e.g. after a ceiling or
        formula change), without re-running anything. None (default) keeps the S+ recorded at evaluation time."""
        return None

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        raise NotImplementedError

    def baseline(self, art_dir: Path) -> Submission:
        """Non-agentic floor (SPEC §4)."""
        raise NotImplementedError

    def reference(self, art_dir: Path) -> Submission:
        """Oracle / reference solution; must clearly beat the baseline."""
        raise NotImplementedError

    # Named reference solutions a task offers (first = default). A task that declares them implements
    # ``reference(art_dir, variant)``; the runner records the variant of every reference run (ledger + result.json).
    reference_variants: ClassVar[tuple[str, ...]] = ()

    def default_reference_variant(self) -> str | None:
        """Variant used when none is requested (tasks with ``reference_variants`` only)."""
        return self.reference_variants[0] if self.reference_variants else None


REGISTRY: dict[str, type[Task]] = {}


def register(cls: type[Task]) -> type[Task]:
    if cls.id in REGISTRY:
        raise ValueError(f"duplicate task id {cls.id}")
    REGISTRY[cls.id] = cls
    return cls


def load_all() -> dict[str, type[Task]]:
    """Register task definitions without importing their private model implementations."""
    import importlib
    import pkgutil

    import weightbench.tracks as tracks

    for m in pkgutil.iter_modules(tracks.__path__, tracks.__name__ + "."):
        if not m.ispkg:
            importlib.import_module(m.name)
    return REGISTRY
