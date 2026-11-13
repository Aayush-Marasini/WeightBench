"""TR-04 Cross-Model Circuit Grafting: TR04-GRAFT-001, SPEC §2 TR-04 (b).

A host Transformer solves task A, ``(x1 + x2) mod 10`` on query token QA. A donor MLP computes a
seeded lookup table ``g(x1, x2)``. The agent edits the host's weights so that on the new query
token QB it outputs ``g(x1, x2)``, i.e. it moves the donor's circuit into the host's residual
stream, while task A keeps working.

Training, evaluation and probes run through ``weightbench.heavy.run_heavy`` -> ``tr04_graft_lib.jobs``.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from weightbench import scoring
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr04_go_lib.spec import sub_seed
from weightbench.tracks.tr04_graft_lib import model as M

JOBS = "weightbench.tracks.tr04_graft_lib.jobs"
N_CHECKS = 20
# How the knob's distractor function is described to the agent (the name itself stays in knob.json/spec.json).
DISTRACTOR_TEXT = {
    "x1": "`d(x1, x2) = x1` (the first digit alone)",
    "x3": "`d(x1, x2, x3) = x3` (the third token)",
    "prod": "`d(x1, x2) = (x1 · x2) mod 10`",
    "sum": "`d(x1, x2) = (x1 + x2) mod 10`",
    "diff": "`d(x1, x2) = (x1 − x2) mod 10`",
    "sum3": "`d(x1, x2, x3) = (x1 + x2 + x3) mod 10`",
    "const": "`d(x1, x2) = 0` (a constant)",
}


def distractor_record(level: int) -> dict | None:
    """The knob level's competing QB circuit as a JSON record (``None`` at L0): ``{function, weight}``."""
    d = M.distractor(level)
    return None if d is None else {"function": d[0], "weight": d[1]}


def _heavy(art_dir: Path, step: str, out_name: str, extra: list[str] = (), push: list[Path] = ()) -> dict:
    from weightbench.heavy import run_heavy

    art_dir = art_dir.resolve()
    return run_heavy(JOBS, [step, "--art", str(art_dir), *extra], art_dir, push=list(push),
                     out_name=out_name, timeout=2 * 3600, gpu=False)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def validation_error(delta: dict[str, np.ndarray]) -> str | None:
    for name, value in delta.items():
        if name not in M.HOST_SHAPES:
            return f"unknown parameter {name!r}; expected host parameter names from host.npz"
        if np.shape(value) != M.HOST_SHAPES[name]:
            return f"{name} has shape {list(np.shape(value))}, expected {list(M.HOST_SHAPES[name])}"
        if not np.all(np.isfinite(value)):
            return f"{name} contains non-finite values"
    return None


@register
class CircuitGrafting(Task):
    id = "TR04-GRAFT-001"
    track = "TR04"
    tier = "foundational"
    submission = "delta_w"
    # Hard edit budget: at most 10% of the host's parameters may change (the reference graft stays well inside it).
    budget = Budget(n_forward=N_CHECKS, n_probe=0, max_l0=0.10, wall_clock_s=3 * 3600)
    probe_kinds = {"check": "forward"}
    # The prompt allows reading the donor's lookup table by evaluating it; only running the host is forbidden.
    meter_exclude = ("donor.npz",)
    # The transcript pattern cannot tell donor from host evaluation; the runtime chain trace (donor excluded) still can.
    audit_allowed_flags = ("forward_pass_suspect",)
    # Difficulty knob (SPEC §5): bounded explicit levels L0..L2, unversioned default kept. Level L also trains the
    # host on QB, whose batches mix the task-A function with a distractor function (see model.DISTRACTORS for the
    # weight ramp), so the digits are less cleanly laid out in the very residual a graft must read. The
    # donor, the target table, the budget, the host's task-A behaviour and the scoring are unchanged, so every level
    # asks for the same function through a harder channel. L0 trains the host on task A only and is the default.
    knob_levels = (0, 1, 2)

    # ---- generation ---------------------------------------------------------------------
    def build(self, art_dir: Path, seed: int) -> None:
        from weightbench import knobs

        if self.level is not None:          # reject an unsupported level before any training
            knobs.check_level(self.level)
            if self.level not in self.knob_levels:
                raise ValueError(f"task {self.id}: supported knob levels are {self.knob_levels}, got {self.level}")
        spec = {"task": self.id, "seed": seed,
                "seeds": {"host": sub_seed(seed, 11), "donor": sub_seed(seed, 12), "table": sub_seed(seed, 13)}}
        if self.level is not None:          # explicit level only; the default spec stays byte-identical
            spec.update(level=self.level, distractor=distractor_record(self.level))
        (art_dir / "private" / "spec.json").write_text(json.dumps(spec, indent=1))
        _heavy(art_dir, "build", "private/build.json")
        if self.level is not None:
            (art_dir / "public" / "knob.json").write_text(json.dumps(
                {**knobs.describe(self.track, self.level), "distractor": distractor_record(self.level),
                 "target": "g(x1, x2), the same 10x10 table at every level"}, indent=1))
        if os.environ.get("WB_HEAVY", "local") != "local":
            from weightbench.heavy import pull_dir

            pull_dir(art_dir, "public")
            pull_dir(art_dir, "private")

    def _knob(self, art_dir: Path) -> dict | None:
        p = art_dir / "public" / "knob.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _knob_section(self, art_dir: Path) -> str:
        kn = self._knob(art_dir)
        if not kn or int(kn["level"]) == 0 or not kn.get("distractor"):
            return ""
        d = kn["distractor"]
        human = DISTRACTOR_TEXT.get(d["function"], f"`{d['function']}`")
        share = f"{d['weight']:.0%}"
        return (f"\n\n**Difficulty knob level {kn['level']}: the host also serves a QB circuit.** At this level the "
                f"host's QB training batches were a mixture: a {share} share asked for {human}, and the remaining "
                f"{100 - int(round(d['weight'] * 100))}% asked for the task-A function. That competing circuit shapes "
                f"the QB residual, so x1 and x2 are not laid out there as cleanly as they are on QA: the graft has to "
                f"recover both digits from this residual, or replace the circuit another way, before it can index the "
                f"donor's table. The donor, the target `g(x1, x2)`, the edit budget and the scoring are the same as at "
                f"the default level, and the host's task-A behaviour is unchanged.")

    def _base(self, art_dir: Path) -> dict[str, float]:
        return json.loads((art_dir / "private" / "build.json").read_text())["base"]

    def _host_training(self, art_dir: Path, base: dict) -> str:
        """The prompt's sentence about what the host was trained on (level-dependent: the knob's distractor)."""
        head = (f"The host was trained on **task A**: on query QA it outputs `(x1 + x2) mod 10` (accuracy "
                f"{base['task_a']:.3f} over all 10^4 inputs).")
        kn = self._knob(art_dir)
        if not kn or int(kn["level"]) == 0 or not kn.get("distractor"):
            return head + " It was never trained on QB."
        name = kn["distractor"]["function"]
        human = DISTRACTOR_TEXT.get(name, f"`{name}`")
        acc = base.get(f"distract_{name}")
        tail = (f" Its QB training batches were a mixture: {kn['distractor']['weight']:.0%} asked for {human}, the "
                f"rest for the task-A function")
        if isinstance(acc, (int, float)):
            tail += f" (accuracy {acc:.3f} over all 10^4 inputs)"
        return head + tail + "."

    # ---- agent-facing text ------------------------------------------------------------------
    def prompt(self, art_dir: Path) -> str:
        base = self._base(art_dir)
        alpha, kappa, beta = scoring.V3_ALPHA, scoring.V3_KAPPA, scoring.V3_BETA
        max_l0 = self.budget.max_l0
        return f"""# {self.id}: Cross-Model Circuit Grafting

Track TR-04 (Weight Synthesis, WRITE). Move a working logic circuit from a donor MLP into the residual
stream of a host Transformer.{self._knob_section(art_dir)}

## Host Transformer (`host.npz`; no normalisation layers; PyTorch convention `y = x @ W.T`)
Input: 5 tokens `[x1, x2, x3, x4, q]`, digits `x_i ∈ 0..9` (token ids 0..9), query `q ∈ {{QA = 10, QB = 11}}`.

    h = embed[tokens] + pos                                            # [5, 64]
    for l in 0, 1:
        q, k, v = h @ l.wq.T, h @ l.wk.T, h @ l.wv.T                    # split into 4 heads of size 16
        att = softmax(q_head @ k_head.T / 4 + causal_mask)              # causal: position t sees 0..t
        h = h + concat_heads(att @ v_head) @ l.wo.T
        h = h + relu(h @ l.w_in.T + l.b_in) @ l.w_out.T + l.b_out        # MLP width 128
    logits = h[4] @ unembed.T + unembed_b                               # prediction at the last position

Shapes: `embed [12, 64]`, `pos [5, 64]`, `l{{0,1}}.wq/wk/wv/wo [64, 64]`, `l{{0,1}}.w_in [128, 64]`, `b_in [128]`,
`w_out [64, 128]`, `b_out [64]`, `unembed [10, 64]`, `unembed_b [10]`. Head h uses rows/columns `16h:16h+16`.
{self._host_training(art_dir, base)}

## Donor MLP (`donor.npz`)
`g(x1, x2) = argmax(W2 @ relu(W1 @ [onehot(x1); onehot(x2)] + b1) + b2)`, `W1 [32, 20]`, `W2 [10, 32]`. It computes
a fixed 10x10 lookup table perfectly.

## Goal
Edit the host so that on query **QB** it outputs `g(x1, x2)`, while task A on QA keeps working.

## Scoring (server-side, all 10^4 inputs per query)
- S = accuracy on QB against the donor's table. `S_base` = {base['graft']:.3f}. `S+ = clip((S − S_base)/(1 − S_base), 0, 1)`.
- Collateral `D_col = max(0, A_base − A_edit)` with A = task-A accuracy on QA; factor `exp(−{alpha:.2f}·D_col)`.
- **Edit budget (hard):** ΔW may change at most {max_l0:.0%} of the host's parameters; a larger edit scores 0.
  Within it, `u = (changed parameters / host parameters) / {max_l0:g}`, factor `1 − {kappa:g}·u`.
- Checks used: factor `1 − {beta:g}·n_checks / {N_CHECKS}`.
- `S_WB = S+ · exp(−{alpha:.2f}·D_col) · (1 − {kappa:g}·u) · (1 − {beta:g}·n_checks / {N_CHECKS})`.

## Metered check (budget: {N_CHECKS} calls)
`python wb.py probe check '{{"file": "delta.npz", "seed": 3}}' -o c1.json` returns task-A and QB accuracy of
host + ΔW on 500 random QA and 500 random QB inputs (sampled by `seed`).

## Rules (violations score 0)
- No gradient descent or fitting to data; no copying the weights out of the metered array to compute with
  them unmetered; do not run the host (or a hand-written copy of it) on inputs yourself. Static analysis of the weights (composing matrices, SVD, solving linear systems on the weights)
  is allowed and expected. You may evaluate the donor MLP's table from its weights.
- Do not try to read files outside your working directory.

## Submission
`np.savez("delta.npz", **{{"l1.w_in": dW, ...}})` with host parameter names and full shapes (float32; omitted
tensors are unchanged; ΔW is added to the host), then `python wb.py submit-delta delta.npz`. You may resubmit.
"""

    # ---- probe ------------------------------------------------------------------------------
    def probe(self, art_dir: Path, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if kind != "check":
            raise KeyError(kind)
        seed, path = payload.get("seed"), payload.get("file")
        if not (isinstance(seed, int) and 0 <= seed < 2 ** 31):
            raise ValueError("seed must be an int in [0, 2^31)")
        if not isinstance(path, str):
            raise ValueError("file must name an .npz in your working directory")
        delta = _load(Path(path))
        error = validation_error(delta)
        if error:
            raise ValueError(error)
        probe_dir = art_dir.resolve() / "private" / "probes" / uuid.uuid4().hex[:12]
        probe_dir.mkdir(parents=True)
        delta_path, payload_path = probe_dir / "delta.npz", probe_dir / "payload.json"
        np.savez(delta_path, **delta)
        payload_path.write_text(json.dumps({"seed": seed}))
        return _heavy(art_dir, "check", (probe_dir / "out.json").relative_to(art_dir.resolve()).as_posix(),
                      ["--weights", str(delta_path), "--payload", str(payload_path)], push=[delta_path, payload_path])

    # ---- scoring ------------------------------------------------------------------------------
    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray]:
        return _load(art_dir / "public" / "host.npz")

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        if sub.type != "delta_w" or not sub.delta:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "expected a non-empty delta_w submission"})
        error = validation_error(sub.delta)
        if error:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": error})
        base = self._base(art_dir)
        eval_dir = art_dir.resolve() / "private" / "evals" / uuid.uuid4().hex[:12]
        eval_dir.mkdir(parents=True)
        delta_path = eval_dir / "delta.npz"
        np.savez(delta_path, **{k: np.asarray(v, np.float32) for k, v in sub.delta.items()})
        result = _heavy(art_dir, "evaluate", (eval_dir / "result.json").relative_to(art_dir.resolve()).as_posix(),
                        ["--weights", str(delta_path)], push=[delta_path])
        return Evaluation(s_plus=scoring.s_plus_write(result["graft"], base["graft"], 1.0),
                          d_col=scoring.collateral(base["task_a"], result["task_a"]),
                          details={"S_base": base["graft"], "S_edit": result["graft"],
                                   "task_a_base": base["task_a"], "task_a_edit": result["task_a"]})

    def baseline(self, art_dir: Path) -> Submission:
        """Floor: naive transplant that reads the residual as raw token embeddings."""
        return Submission("delta_w", delta=_load(art_dir / "private" / "baseline.npz"))

    def reference(self, art_dir: Path) -> Submission:
        """Server-side read-out fitting + gated graft into the spare layer-1 units."""
        return Submission("delta_w", delta=_load(art_dir / "private" / "reference.npz"))
