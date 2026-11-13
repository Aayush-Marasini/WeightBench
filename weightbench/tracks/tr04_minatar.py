"""TR-04 Zero-Shot Game Policy Compilation: TR04-GAME-001..003, SPEC §2 TR-04.

The agent gets an all-zero ``600→32→32→6`` MLP and the MinAtar source of one game, and must
write the weights in one shot (``N_forward = 0``, no gradient descent). The submission is the
full weight set as ΔW on top of the zero network. Scoring is the mean return over private eval
seeds, normalised between the zero network (S_base) and a server-side planner (S_max).

Rollouts (calibration, reference selection, evaluation) run through
``weightbench.heavy.run_heavy`` -> ``tr04_minatar_lib.jobs``. Nothing in this module runs a
policy or an environment.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import ClassVar

import numpy as np

from weightbench import scoring
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr04_minatar_lib import spec as S

JOBS = "weightbench.tracks.tr04_minatar_lib.jobs"


def _heavy(art_dir: Path, step: str, out_name: str, extra: list[str] = (), push: list[Path] = ()) -> dict:
    from weightbench.heavy import run_heavy

    art_dir = art_dir.resolve()
    return run_heavy(JOBS, [step, "--art", str(art_dir), *extra], art_dir, push=list(push),
                     out_name=out_name, timeout=2 * 3600, gpu=False)


def _pull_private(art_dir: Path) -> None:
    if os.environ.get("WB_HEAVY", "local") != "local":
        from weightbench.heavy import pull_dir

        pull_dir(art_dir, "private")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _channel_table(game: str) -> str:
    by_channel: dict[int, list[str]] = {}
    for name, channel in S.CHANNEL_MAP[game].items():
        by_channel.setdefault(channel, []).append(f"`{name}`")
    rows = [f"| {c} | {', '.join(by_channel.get(c, [])) or '(always 0)'} |" for c in range(S.N_CHANNELS)]
    return "| common channel c | native MinAtar channels OR-ed into it |\n|---|---|\n" + "\n".join(rows)


class _GameTask(Task):
    track = "TR04"
    tier = "foundational"
    submission = "delta_w"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=3600)
    game: ClassVar[str]
    title: ClassVar[str]
    notes: ClassVar[str] = ""
    # Difficulty knob ``algorithmic_horizon`` (SPEC §5): bounded explicit levels, unversioned default kept.
    # Level L delays the observation by d = 2^L - 1 frames and adds d one-hot action slots (``spec.delay``).
    knob_levels = S.LEVELS

    @property
    def delay(self) -> int:
        """Observation delay of this instance: 0 for the default build and for explicit L0."""
        return S.delay(self.level) if self.level is not None else 0

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return S.param_shapes(self.delay)

    # ---- generation ---------------------------------------------------------------------
    def build(self, art_dir: Path, seed: int) -> None:
        from weightbench import knobs
        from weightbench.tracks.tr04_minatar_lib.rollout import source_dir

        pub, priv = art_dir / "public", art_dir / "private"
        spec = {"task": self.id, "game": self.game, "seed": seed, "seeds": S.task_seeds(seed),
                "eval_episodes": S.EVAL_EPISODES, "expert_episodes": S.EXPERT_EPISODES,
                "dev_episodes": S.DEV_EPISODES, "max_steps": S.MAX_STEPS,
                "sticky_action_prob": S.STICKY_ACTION_PROB, "minatar_version": version("minatar")}
        if self.level is not None:          # explicit level only; the default spec stays byte-identical
            spec.update(level=self.level, delay=self.delay)
            (pub / "knob.json").write_text(json.dumps(
                {**knobs.describe(self.track, self.level), "observation_delay": self.delay,
                 "input_dim": S.input_dim(self.delay)}, indent=1))
        (priv / "spec.json").write_text(json.dumps(spec, indent=1))
        game_dir = pub / "game"
        game_dir.mkdir()
        shutil.copy(source_dir() / "environments" / f"{self.game}.py", game_dir / f"{self.game}.py")
        shutil.copy(source_dir() / "environment.py", game_dir / "environment.py")
        (pub / "encoding.json").write_text(json.dumps(
            {"channel_map": S.CHANNEL_MAP[self.game], "actions": list(S.ACTIONS),
             "param_shapes": {k: list(v) for k, v in self.shapes.items()}}, indent=1))
        _heavy(art_dir, "calibrate", "private/calibration.json")
        _heavy(art_dir, "reference", "private/reference_info.json")
        _pull_private(art_dir)

    def _spec(self, art_dir: Path) -> dict:
        return _read_json(art_dir / "private" / "spec.json")

    def _calibration(self, art_dir: Path) -> dict:
        return _read_json(art_dir / "private" / "calibration.json")

    def rescore_s_plus(self, details: dict) -> float | None:
        """S+ from the stored mean return under the current ceiling (a ledger entry without a ``planner`` field stores the planner return as ``S_max``)."""
        if "S_edit" not in details or "S_base" not in details:
            return None
        planner = details.get("planner", details.get("S_max"))
        if planner is None:
            return None
        s_max, _ = self._ceiling({"expert": {"mean": planner}})
        return scoring.s_plus_write(details["S_edit"], details["S_base"], s_max)

    def _ceiling(self, cal: dict) -> tuple[float, str]:
        """S_max = the stronger of the server-side planner and the strongest known policy for this game. Under an
        observation delay (knob level > 0) only the delayed planner measured for this build counts: the
        STRONGEST_KNOWN returns were measured on the undelayed game. The reference never sets a ceiling."""
        planner = cal["expert"]["mean"]
        if self.delay:
            return planner, (f"server-side delayed planner (sees the frame from {self.delay} steps ago and the pending "
                             "actions, predicts the current state with a cloned emulator, then plans)")
        known = S.STRONGEST_KNOWN.get(self.game.lower())
        if known is not None and known["return"] > planner:
            return known["return"], known["source"]
        return planner, "server-side look-ahead planner that can clone the emulator state"

    # ---- agent-facing text ------------------------------------------------------------------
    def prompt(self, art_dir: Path) -> str:
        cal = self._calibration(art_dir)
        s_base = cal["zero_network"]["mean"]
        s_max, ceiling_source = self._ceiling(cal)
        init_norm = S.frobenius_norm(self.frobenius_reference(art_dir))
        d, n_in = self.delay, S.input_dim(self.delay)
        delay_section = "" if not d else f"""## Observation delay (knob level {self.level}: H = {d + 1})
Your network does NOT see the current frame. At step `t` the input is `[frame ; slots]` of length {n_in}:
- `frame` (600, encoded as above) is the frame at time `max(0, t − {d})`, i.e. **{d} step(s) old**;
- `slots` are {d} one-hot blocks of 6: block `i` (0 = oldest) is the action **you chose** at time `t − {d} + i`
  (all zero for times before the episode start). `x[600 + 6*i + a]` = 1 iff that action was `a`.
The game itself is unchanged; the action you output is applied to the current (unseen) state. To act well
you have to predict the current state from the old frame and your own recent actions ({d} step(s) ahead).
Sticky actions still apply, so a chosen action in a slot may not have been the executed one.

"""
        return f"""# {self.id}: Zero-Shot Game Policy Compilation: {self.title}

Track TR-04 (Weight Synthesis, WRITE). You get an **all-zero** policy network and the full source
of the MinAtar game **{self.game}** (`game/{self.game}.py`, plus the `game/environment.py` wrapper that
adds sticky actions). Write the network's weights **by hand, in one shot**, so that the greedy policy
plays {self.game} well.
{self.notes}
## Architecture (fixed; PyTorch convention `y = W x + b`)
    h1     = relu(W1 @ x + b1)      W1: [32, {n_in}]  b1: [32]
    h2     = relu(W2 @ h1 + b2)     W2: [32, 32]   b2: [32]
    logits = W3 @ h2 + b3           W3: [6, 32]    b3: [6]
    action = argmax(logits)          (ties go to the lowest action id)

## Observation `x` (binary, length 600)
`x[c*100 + row*10 + col]` for common channel `c` in 0..5, `row` 0 (top) .. 9 (bottom), `col` 0 (left)
.. 9 (right). The game's native MinAtar channels (`env.state()`, see `self.channels` in the game
source) are OR-ed into the common channels like this:

{_channel_table(self.game)}

{delay_section}## Actions (output index = action id)
`0 = n (no-op), 1 = l (left), 2 = u (up), 3 = r (right), 4 = d (down), 5 = f (fire)`. Actions that
do nothing in this game behave like no-op.

## Evaluation (server-side)
- {S.EVAL_EPISODES} episodes on private seeds, sticky actions p = {S.STICKY_ACTION_PROB} (MinAtar default),
  difficulty ramping on, at most {S.MAX_STEPS} steps per episode. S = mean undiscounted episode return.
- `S_base` = {s_base:.2f}: the all-zero network (always action 0). `S_max` = {s_max:.2f}: the strongest known
  policy for this game ({ceiling_source}).
- `S+ = clip((S − S_base) / (S_max − S_base), 0, 1)`.
- `S_WB = S+`: there are no probes, and no edit-size penalty for from-zero synthesis (ReLU networks are invariant
  to positive per-layer rescaling, so the weight norm carries no information; a default PyTorch-initialised network
  of this architecture has ‖W_init‖_F = {init_norm:.2f}).

## Rules (violations score 0)
- `N_forward = 0`, `N_probe = 0`: there are no probes; `wb.py` is only for submitting.
- **No gradient descent and no fitting to data.** No optimisers, no autograd, no regression on
  game states, and no copying the weights out of the metered array to compute with them unmetered.
- **Do not run the game or the network.** Do not execute the game source, and do not evaluate your
  network (or any network) on any input, including inputs you construct yourself. Reading the source,
  reasoning about it, and writing or algebraically checking the weight matrices is what this task is about.
- Do not try to read files outside your working directory.

## Submission
Save an `.npz` holding `W1 b1 W2 b2 W3 b3` (float32, the exact shapes above). Since the base network
is all zeros, ΔW is your weight set; omitted tensors stay zero.
```python
np.savez("delta.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
```
then `python wb.py submit-delta delta.npz`. You may resubmit; the last submission counts.
"""

    # ---- scoring -----------------------------------------------------------------------------
    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray]:
        return S.zero_weights(self.shapes)

    def frobenius_reference(self, art_dir: Path) -> dict[str, np.ndarray]:
        return S.init_weights(self._spec(art_dir)["seeds"]["norm_init"], self.shapes)

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        if sub.type != "delta_w" or not sub.delta:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "expected a non-empty delta_w submission"})
        error = S.validation_error(sub.delta, self.shapes)
        if error:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": error})
        cal = self._calibration(art_dir)
        s_base = cal["zero_network"]["mean"]
        s_max, ceiling_source = self._ceiling(cal)
        details = {"S_base": s_base, "S_max": s_max, "S_max_source": ceiling_source,
                   "planner": cal["expert"]["mean"]}
        weights = S.apply_delta(sub.delta, self.shapes)
        if not any(np.any(w) for w in weights.values()):
            return Evaluation(s_plus=0.0, details={**details, "S_edit": s_base, "reason": "ΔW = 0"})
        eval_dir = art_dir / "private" / "evals" / uuid.uuid4().hex[:12]
        eval_dir.mkdir(parents=True)
        weights_path = eval_dir / "weights.npz"
        np.savez(weights_path, **weights)
        result = _heavy(art_dir, "evaluate", (eval_dir / "result.json").relative_to(art_dir).as_posix(),
                        ["--weights", str(weights_path.resolve())], push=[weights_path])
        details.update(S_edit=result["mean"], stderr=result["stderr"], episodes=result["n"],
                       eval_seconds=result.get("seconds"))
        if s_max <= s_base:
            details["warning"] = "planner ceiling does not exceed the zero network; S+ is 0 by definition"
        return Evaluation(s_plus=scoring.s_plus_write(result["mean"], s_base, s_max),
                          n_eval_forward=result["n"], details=details)

    def baseline(self, art_dir: Path) -> Submission:
        """Non-agentic floor: a default-initialised (untrained) network, seeded per task."""
        return Submission("delta_w", delta=S.init_weights(self._spec(art_dir)["seeds"]["baseline_init"],
                                                            self.shapes))

    def reference(self, art_dir: Path) -> Submission:
        """Hand-compiled circuit (``tr04_minatar_lib.compiled``), selected on private dev seeds."""
        with np.load(art_dir / "private" / "reference.npz") as z:
            return Submission("delta_w", delta={k: z[k] for k in z.files})


@register
class BreakoutCompilation(_GameTask):
    id = "TR04-GAME-001"
    game = "breakout"
    title = "Breakout"


@register
class SeaquestCompilation(_GameTask):
    id = "TR04-GAME-002"
    game = "seaquest"
    title = "Seaquest"
    notes = ("\nSeaquest has 10 native channels; some share a common channel (table below). Every merged pair "
             "is separable by row: divers and trails only appear in rows 1-8, the two gauges only in row 9.\n")


@register
class AsterixCompilation(_GameTask):
    id = "TR04-GAME-003"
    game = "asterix"
    title = "Asterix"
