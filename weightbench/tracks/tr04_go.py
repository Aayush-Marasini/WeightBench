"""TR-04 Scaled Programmatic Network Synthesis: TR04-GO-001, SPEC §2 TR-04 (a).

The agent writes a ~1.5M-parameter 9x9 Go policy (``243 → 512×6 → 82``) from an all-zero
network, without gradient descent, with at most ``N_forward = 50`` metered single-game rollouts.
Score: win rate against a fixed random opponent on private seeds, normalised from the zero
network's win rate to 1.

Games run through ``weightbench.heavy.run_heavy`` -> ``tr04_go_lib.jobs``.

Difficulty knob ``algorithmic_horizon`` (SPEC §5): bounded explicit levels L0..L2, unversioned default kept. At level
L the fixed opponent reads H - 1 = 2^L - 1 plies of capture tactics on a local window (``tr04_go_lib.tactics``; L0 =
the default random opponent), and S_max(L) is the win rate of an independent capture reader one ply deeper (2^L
plies, area-greedy fallback) against the same opponent on the same eval seeds. Knob documentation:
the knob description in docs/SPEC.md §5.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from weightbench import scoring
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr04_go_lib import board as B
from weightbench.tracks.tr04_go_lib import spec as S

JOBS = "weightbench.tracks.tr04_go_lib.jobs"
MAX_PROBE_FILE_BYTES = 64 * 2 ** 20


def _heavy(art_dir: Path, step: str, out_name: str, extra: list[str] = (), push: list[Path] = ()) -> dict:
    from weightbench.heavy import run_heavy

    art_dir = art_dir.resolve()
    return run_heavy(JOBS, [step, "--art", str(art_dir), *extra], art_dir, push=list(push),
                     out_name=out_name, timeout=4 * 3600, gpu=False)


def _pull_private(art_dir: Path) -> None:
    if os.environ.get("WB_HEAVY", "local") != "local":
        from weightbench.heavy import pull_dir

        pull_dir(art_dir, "private")


@register
class GoPolicySynthesis(Task):
    id = "TR04-GO-001"
    track = "TR04"
    tier = "foundational"
    submission = "delta_w"
    budget = Budget(n_forward=S.N_FORWARD, n_probe=0, wall_clock_s=3 * 3600)
    probe_kinds = {"game": "forward"}
    knob_levels = S.LEVELS

    # ---- generation ---------------------------------------------------------------------
    def build(self, art_dir: Path, seed: int) -> None:
        pub, priv = art_dir / "public", art_dir / "private"
        spec = {"task": self.id, "seed": seed, "seeds": S.task_seeds(seed), "eval_games": S.EVAL_GAMES,
                "komi": B.KOMI, "max_moves": B.MAX_MOVES}
        if self.level is not None:          # explicit level only; the default spec stays byte-identical
            from weightbench import knobs
            from weightbench.tracks.tr04_go_lib import tactics as T

            spec["seeds"]["ceiling"] = S.sub_seed(seed, 6)
            spec.update(level=self.level, opponent_depth=T.opponent_depth(self.level),
                        ceiling_depth=T.ceiling_depth(self.level), node_cap=T.NODE_CAP)
            (pub / "knob.json").write_text(json.dumps(
                {**knobs.describe(self.track, self.level), "opponent_read_plies": T.opponent_depth(self.level),
                 "ceiling_read_plies": T.ceiling_depth(self.level), "node_cap": T.NODE_CAP}, indent=1))
        (priv / "spec.json").write_text(json.dumps(spec, indent=1))
        (pub / "go").mkdir()
        shutil.copy(Path(B.__file__), pub / "go" / "board.py")
        if self.level is not None:          # the level opponent's exact rule (public, like the board)
            from weightbench.tracks.tr04_go_lib import tactics as T

            shutil.copy(Path(T.__file__), pub / "go" / "tactics.py")
        _heavy(art_dir, "calibrate", "private/calibration.json")
        _heavy(art_dir, "reference", "private/reference_info.json")
        _pull_private(art_dir)

    def _spec(self, art_dir: Path) -> dict:
        return json.loads((art_dir / "private" / "spec.json").read_text())

    def _calibration(self, art_dir: Path) -> dict:
        return json.loads((art_dir / "private" / "calibration.json").read_text())

    def _s_base(self, art_dir: Path) -> float:
        return self._calibration(art_dir)["zero_network"]["win_rate"]

    def _s_max(self, art_dir: Path) -> float:
        """1 for the default build; at an explicit level the independent ceiling player's win rate against
        the level opponent (measured at build; never the task's reference)."""
        ceiling = self._calibration(art_dir).get("ceiling")
        return 1.0 if ceiling is None else float(ceiling["win_rate"])

    def _level_text(self, art_dir: Path) -> tuple[str, str]:
        """(opponent bullet, S+ line) for the prompt; the default build gets the default wording."""
        s_base = self._s_base(art_dir)
        spec = self._spec(art_dir)
        if "level" not in spec:
            return ("- Opponent: uniform-random over legal moves that do not fill its own single-point eyes; it passes "
                    "only when\n  no such move exists.",
                    f"- `S_base` = {s_base:.3f}: the all-zero network (plays the lowest-index legal point).\n"
                    "  `S+ = clip((S − S_base)/(1 − S_base), 0, 1)`.")
        d, cap = spec["opponent_depth"], spec["node_cap"]
        opp = (f"- Opponent (knob level {spec['level']}: H = {d + 1}): before each move it reads {d} plies of capture "
               "tactics.\n  Candidate moves are the legal liberties of every chain (either colour) with at most two "
               "liberties, plus pass;\n  a line scores the stones the opponent captures minus the stones it loses; "
               f"negamax with alpha-beta, at most {cap}\n  positions per move. It plays the best candidate when that "
               "reads strictly better than passing (lowest index\n  on ties); otherwise it plays uniform-random over "
               "legal moves that do not fill its own single-point eyes,\n  passing only when no such move exists. "
               "Exact code: none is shipped; this description is the rule.")
        s_max = self._s_max(art_dir)
        score = (f"- `S_base` = {s_base:.3f}: the all-zero network (plays the lowest-index legal point). `S_max` = "
                 f"{s_max:.3f}: an\n  independent capture reader ({spec['ceiling_depth']} plies, area-greedy "
                 "otherwise) against this opponent.\n  `S+ = clip((S − S_base)/(S_max − S_base), 0, 1)`.")
        return opp, score

    # ---- agent-facing text ------------------------------------------------------------------
    def prompt(self, art_dir: Path) -> str:
        beta = scoring.V3_BETA
        init_norm = S.frobenius_norm(self.frobenius_reference(art_dir))
        n_params = sum(int(np.prod(s)) for s in S.PARAM_SHAPES.values())
        opponent_text, score_text = self._level_text(art_dir)
        return f"""# {self.id}: Zero-Shot 9x9 Go Policy Synthesis

Track TR-04 (Weight Synthesis, WRITE). Write the weights of a {n_params:,}-parameter Go policy network
**from scratch** (the base network is all zeros) so that it beats a fixed opponent at 9x9 Go.

## Architecture (fixed; PyTorch convention `y = W x + b`, ReLU after every hidden layer)
    x [243] -> 512 -> 512 -> 512 -> 512 -> 512 -> 512 -> logits [82]
    W1 [512, 243]; W2..W6 [512, 512]; W7 [82, 512]; b1..b6 [512]; b7 [82]

## Input and output
- `x = encode(board)` (see `go/board.py`): `x[p]` = 1 if point p holds one of **your** stones,
  `x[81 + p]` = 1 for an **opponent** stone, `x[162 + p]` = 1 if p is empty; `p = row*9 + col`, row 0 = top.
- `logits[p]` for the 81 points, `logits[81]` = pass. The referee plays the highest-logit **legal** move
  (illegal moves are masked; ties go to the lowest index).

## Rules of the game (exact code in `go/board.py`)
Captures, suicide forbidden, simple ko, area (Tromp-Taylor) scoring with komi {B.KOMI} for white; the game
ends after two consecutive passes or {B.MAX_MOVES} moves.

## Opponent and evaluation (server-side)
{opponent_text}
- S = win rate over {S.EVAL_GAMES} games on private seeds, half as black, half as white.
{score_text}
- No edit-size penalty: you write the whole network from zero, and ReLU networks are invariant to positive
  per-layer rescaling, so the weight norm carries no information. (For reference, a default PyTorch-initialised
  network of this architecture has ‖W_init‖_F = {init_norm:.2f}.)
- Rollouts used discount the score linearly: `S_WB = S+ · (1 − {beta:g}·n_rollouts / {S.N_FORWARD})`.

## Metered rollouts: `game` (budget: {S.N_FORWARD} calls)
Save candidate weights as an `.npz` (same format as the submission) and play ONE game against the opponent:
```
python wb.py probe game '{{"file": "cand.npz", "color": "black", "seed": 7}}' -o g1.json
```
`color` is your colour, `seed` (0 .. 2^31) seeds the opponent (eval seeds are private). Returns the full
move record (`a9` = top-left, columns `a..j` without `i`), areas and the result.

## Rules (violations score 0)
- **No gradient descent and no fitting to data** (no optimisers, no autograd, no regression on positions,
  no copying the weights out of the metered array to compute with them unmetered).
- **Do not run the network yourself** on any input, and do not simulate games with it locally; the
  `game` probe is the only way to observe your network playing. Reading `go/board.py` and writing or
  algebraically checking weight matrices is what this task is about.
- Do not try to read files outside your working directory.

## Submission
`np.savez("delta.npz", W1=..., b1=..., ..., W7=..., b7=...)` (float32, exact shapes; omitted tensors stay
zero), then `python wb.py submit-delta delta.npz`. You may resubmit; the last submission counts.
"""

    # ---- probe ------------------------------------------------------------------------------
    def probe(self, art_dir: Path, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if kind != "game":
            raise KeyError(kind)
        color, seed, path = payload.get("color"), payload.get("seed"), payload.get("file")
        if color not in ("black", "white"):
            raise ValueError("color must be 'black' or 'white'")
        if not (isinstance(seed, int) and 0 <= seed < 2 ** 31):
            raise ValueError("seed must be an int in [0, 2^31)")
        if not isinstance(path, str) or Path(path).stat().st_size > MAX_PROBE_FILE_BYTES:
            raise ValueError("file must be an .npz of candidate weights (≤ 64 MB)")
        with np.load(path, allow_pickle=False) as z:
            delta = {k: z[k] for k in z.files}
        error = S.validation_error(delta)
        if error:
            raise ValueError(error)
        probe_dir = art_dir.resolve() / "private" / "probes" / uuid.uuid4().hex[:12]
        probe_dir.mkdir(parents=True)
        weights_path, payload_path = probe_dir / "weights.npz", probe_dir / "payload.json"
        np.savez(weights_path, **S.apply_delta(delta))
        payload_path.write_text(json.dumps({"color": color, "seed": seed}))
        return _heavy(art_dir, "game", (probe_dir / "out.json").relative_to(art_dir.resolve()).as_posix(),
                      ["--weights", str(weights_path), "--payload", str(payload_path)],
                      push=[weights_path, payload_path])

    # ---- scoring ------------------------------------------------------------------------------
    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray]:
        return S.zero_weights()

    def frobenius_reference(self, art_dir: Path) -> dict[str, np.ndarray]:
        return S.init_weights(self._spec(art_dir)["seeds"]["norm_init"])

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        if sub.type != "delta_w" or not sub.delta:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "expected a non-empty delta_w submission"})
        error = S.validation_error(sub.delta)
        if error:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": error})
        s_base, s_max = self._s_base(art_dir), self._s_max(art_dir)
        weights = S.apply_delta(sub.delta)
        if not any(np.any(w) for w in weights.values()):
            return Evaluation(s_plus=0.0, details={"S_base": s_base, "S_edit": s_base, "reason": "ΔW = 0"})
        eval_dir = art_dir.resolve() / "private" / "evals" / uuid.uuid4().hex[:12]
        eval_dir.mkdir(parents=True)
        weights_path = eval_dir / "weights.npz"
        np.savez(weights_path, **weights)
        result = _heavy(art_dir, "evaluate", (eval_dir / "result.json").relative_to(art_dir.resolve()).as_posix(),
                        ["--weights", str(weights_path)], push=[weights_path])
        details = {"S_base": s_base, "S_edit": result["win_rate"], "stderr": result["stderr"],
                   "mean_margin": result["mean_margin"], "games": result["n"], "eval_seconds": result.get("seconds")}
        if "level" in self._spec(art_dir):
            details.update(S_max=s_max, level=self._spec(art_dir)["level"])
        if "S_max" in details and s_max <= s_base:
            details["warning"] = "ceiling does not exceed the zero network; S+ is 0 by definition"
            return Evaluation(s_plus=0.0, n_eval_forward=result["n"], details=details)
        return Evaluation(s_plus=scoring.s_plus_write(result["win_rate"], s_base, s_max), n_eval_forward=result["n"],
                          details=details)

    def baseline(self, art_dir: Path) -> Submission:
        """Non-agentic floor: a default-initialised (untrained) network, seeded per task."""
        return Submission("delta_w", delta=S.init_weights(self._spec(art_dir)["seeds"]["baseline_init"]))

    def reference(self, art_dir: Path) -> Submission:
        """Hand-compiled local-tactics circuit (``tr04_go_lib.compiled``), selected on dev seeds."""
        with np.load(art_dir / "private" / "reference.npz") as z:
            return Submission("delta_w", delta={k: z[k] for k in z.files})
