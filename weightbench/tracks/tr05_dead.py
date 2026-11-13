"""TR-05 Static Dead-Circuit Pruning: TR05-DEAD-001/002, SPEC §2 TR-05.

The agent gets a ReLU MLP over Boolean inputs ``x ∈ {0,1}^16`` and must prune every hidden neuron
that can never fire, using static bounds analysis only (``N_forward = 0``). Pruning must preserve the
network's function exactly: if any output changes on any input, the score is 0 (unsound proof).
Score: the fraction of never-active neurons pruned.

Difficulty knob ``propagation_depth`` (SPEC §5, ``weightbench.knobs``): level L has 3^L hidden layers.
DEAD-001 is L0 and DEAD-002 is L1; ``weightbench build --task T --level L`` builds any other level.

Ground truth (exact enumeration) and evaluation run through ``weightbench.heavy.run_heavy``.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import ClassVar

import numpy as np

from weightbench import knobs
from weightbench.task import Budget, Evaluation, Submission, Task, register

JOBS = "weightbench.tracks.tr05_dead_lib.jobs"
# Named reference variants. "ibp" is the recorded default; "crown" (back-substitution + per-unit LP, a fixed method
# with no tuned settings) is reported next to it. Select with reference(art, variant) or WB_DEAD_REFERENCE=crown.
REFERENCE_VARIANTS = ("ibp", "crown")


def _heavy(art_dir: Path, step: str, out_name: str, extra: list[str] = (), push: list[Path] = ()) -> dict:
    from weightbench.heavy import run_heavy

    art_dir = art_dir.resolve()
    return run_heavy(JOBS, [step, "--art", str(art_dir), *extra], art_dir, push=list(push),
                     out_name=out_name, timeout=3600, gpu=False)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def hidden_layers(level: int) -> int:
    """Number of hidden layers at ``propagation_depth`` level ``level``: 1 · 3^L (knobs.KNOBS['TR05'])."""
    return int(knobs.value(knobs.get("TR05"), level))


class _DeadTask(Task):
    """One task family per width / plant count; the ``propagation_depth`` knob sets the depth.

    Level L builds ``hidden_layers(L)`` hidden layers of ``width`` units, each with ``n_dead`` planted dead neurons
    and ``n_decoys`` decoys. The default level is the standard build of the task (same spec, same artifacts)."""

    track = "TR05"
    tier = "foundational"
    submission = "delta_w"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=2 * 3600)
    width: ClassVar[int]
    n_dead: ClassVar[int]
    n_decoys: ClassVar[int]
    n_inputs: ClassVar[int] = 16
    reference_variants = REFERENCE_VARIANTS

    def default_reference_variant(self) -> str:
        """``$WB_DEAD_REFERENCE`` or ``ibp``; ``runner run --variant`` overrides both."""
        return os.environ.get("WB_DEAD_REFERENCE") or REFERENCE_VARIANTS[0]

    @property
    def hidden(self) -> tuple[int, ...]:
        return (self.width,) * hidden_layers(self.level)

    def spec(self, seed: int) -> dict:
        """Generator settings for ``seed`` at this instance's knob level (written to private/spec.json)."""
        return {"task": self.id, "seed": seed, "n_inputs": self.n_inputs, "hidden": list(self.hidden),
                "n_dead": self.n_dead, "n_decoys": self.n_decoys, "level": self.level}

    def build(self, art_dir: Path, seed: int) -> None:
        (art_dir / "private" / "spec.json").write_text(json.dumps(self.spec(seed), indent=1))
        _heavy(art_dir, "build", "private/build.json")
        if os.environ.get("WB_HEAVY", "local") != "local":
            from weightbench.heavy import pull_dir

            pull_dir(art_dir, "public")
            pull_dir(art_dir, "private")
        # Public: the architecture is visible from mlp.npz anyway, so the knob record leaks nothing.
        (art_dir / "public" / "knob.json").write_text(json.dumps(knobs.describe(self.track, self.level), indent=1))

    def _arch(self, art_dir: Path) -> tuple[int, tuple[int, ...]]:
        """(n_inputs, hidden sizes) of the built artifact; class defaults if the spec is missing."""
        spec_path = art_dir / "private" / "spec.json"
        if not spec_path.exists():
            return self.n_inputs, self.hidden
        spec = json.loads(spec_path.read_text())
        return spec["n_inputs"], tuple(spec["hidden"])

    def prompt(self, art_dir: Path) -> str:
        n_inputs, hidden = self._arch(art_dir)
        sizes = [n_inputs, *hidden, 1]
        arch = "\n".join(f"    h{i + 1} = relu(W{i + 1} @ h{i} + b{i + 1})      W{i + 1}: [{sizes[i + 1]}, {sizes[i]}]"
                         for i in range(len(hidden)))
        last = len(hidden) + 1
        return f"""# {self.id}: Static Dead-Circuit Pruning

Track TR-05 (Hardware-Aware WSP, SYSTEMS). `mlp.npz` holds a ReLU network (float32, PyTorch convention `y = W x + b`):

    h0 = x,  x ∈ {{0, 1}}^{n_inputs}   (every Boolean input vector is possible)
{arch}
    out = W{last} @ h{last - 1} + b{last}               W{last}: [1, {sizes[-2]}]

## Goal
A hidden neuron is **dead** if its pre-activation is ≤ 0 for **every** input in {{0, 1}}^{n_inputs}. Prune every
dead neuron. Pruning neuron u of layer k = set row u of `W{{k}}`, entry u of `b{{k}}` and column u of `W{{k+1}}` to exactly 0.

## Scoring
- S+ = (number of dead neurons you pruned) / (number of dead neurons).
- **Soundness:** the server compares outputs on all 2^{n_inputs} inputs. If any output changes by more than
  1e-5 (you pruned a neuron that fires and matters), S+ = 0. Every pruned neuron must also be dead on all
  2^{n_inputs} inputs: pruning a neuron that fires on some input is unsound even if the outputs stay the same.
- Your ΔW may only set entries to exactly 0 (W + ΔW must equal W or 0 everywhere); otherwise the submission is invalid.

## Rules (violations score 0)
- `N_forward = 0`: do **not** evaluate the network on inputs (no enumeration, no sampling). Prove deadness from the
  weights (e.g. interval / bound propagation, tighter relaxations, reasoning about which earlier neurons can be active
  together). No gradient descent; no copying the weights out of the metered array to compute with them
  unmetered. Do not read files outside your working directory.

## Submission
`np.savez("delta.npz", W1=..., b1=..., ...)` with float32 ΔW of the exact shapes (only tensors you change), then
`python wb.py submit-delta delta.npz`. You may resubmit; the last submission counts.
"""

    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray]:
        return {}   # λ = 0 for TR-05; validity (zeroing only, soundness) is checked in evaluate()

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        if sub.type != "delta_w" or not sub.delta:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "expected a non-empty delta_w submission"})
        base = _load(art_dir / "public" / "mlp.npz")
        for name, value in sub.delta.items():
            if name not in base or np.shape(value) != base[name].shape:
                return Evaluation(s_plus=0.0, valid=False, details={"reason": f"bad tensor {name!r}"})
        eval_dir = art_dir.resolve() / "private" / "evals" / uuid.uuid4().hex[:12]
        eval_dir.mkdir(parents=True)
        delta_path = eval_dir / "delta.npz"
        np.savez(delta_path, **{k: np.asarray(v, np.float32) for k, v in sub.delta.items()})
        res = _heavy(art_dir, "evaluate", (eval_dir / "result.json").relative_to(art_dir.resolve()).as_posix(),
                     ["--weights", str(delta_path)], push=[delta_path])
        if not res["zeroing_only"]:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "ΔW must only set entries to 0", **res})
        outputs_kept = res["inputs_changed"] == 0
        all_pruned_dead = res["live_pruned"] == 0     # every pruned unit is dead on all 2^n inputs (server-side)
        sound = outputs_kept and all_pruned_dead
        s_plus = res["dead_pruned"] / res["n_dead"] if sound and res["n_dead"] else 0.0
        return Evaluation(s_plus=float(s_plus),
                          details={"sound": sound, "outputs_kept": outputs_kept, "all_pruned_dead": all_pruned_dead,
                                   **res})

    def baseline(self, art_dir: Path) -> Submission:
        """SPEC §4 floor (magnitude pruning, data-free): remove the 10% of hidden units per layer with the
        smallest outgoing-weight norm, with no deadness analysis."""
        return Submission("delta_w", delta=_load(art_dir / "private" / "baseline.npz"))

    def reference(self, art_dir: Path, variant: str | None = None) -> Submission:
        """Sound static proof. ``ibp`` (default): prune exactly the neurons whose interval-bound upper bound is ≤ 0.
        ``crown``: neurons certified by back-substitution + per-unit LP (``tr05_dead_lib/crown.py``), computed from
        the public weights in a heavy step on first use (no forward pass). ``variant`` defaults to
        ``$WB_DEAD_REFERENCE`` or ``ibp``."""
        variant = variant or self.default_reference_variant()
        if variant not in REFERENCE_VARIANTS:
            raise ValueError(f"unknown TR05-DEAD reference variant {variant!r}; choose from {REFERENCE_VARIANTS}")
        if variant == "ibp":
            return Submission("delta_w", delta=_load(art_dir / "private" / "reference.npz"))
        path = art_dir / "private" / "reference_crown.npz"
        if not path.exists():
            _heavy(art_dir, "reference_crown", "private/reference_crown.json")
            if os.environ.get("WB_HEAVY", "local") != "local":
                from weightbench.heavy import pull_dir

                pull_dir(art_dir, "private")
        return Submission("delta_w", delta=_load(path))


@register
class DeadOneLayer(_DeadTask):
    id = "TR05-DEAD-001"
    knob_level = 0      # 1 hidden layer
    width = 128
    n_dead = 16
    n_decoys = 8


@register
class DeadDeep(_DeadTask):
    id = "TR05-DEAD-002"
    knob_level = 1      # 3 hidden layers
    width = 96
    n_dead = 8
    n_decoys = 6
