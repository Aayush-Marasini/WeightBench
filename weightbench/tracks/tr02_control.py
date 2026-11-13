"""TR-02 Control Policy Local-Minimum Override (TR02-CTRL-001..004), SPEC §2 TR-02.

The agent gets a failing control policy (MLP weights), the exact environment description and ONE
failure trajectory, and must wire a sparse override circuit (< 4% of the weights) into the
existing parameters so the policy executes a recovery maneuver in the stuck situation and then
hands control back to the base behaviour.

| task          | env                                   | params | stuck situation                          |
|---------------|---------------------------------------|--------|------------------------------------------|
| TR02-CTRL-001 | Kinetix l/h2_car_ramp (Jax2D physics) | 171.6K | car stalls on a ramp in front of lava     |
| TR02-CTRL-002 | Gymnasium MountainCar-v0              | 9.9K   | car rocks in the valley, never escapes    |
| TR02-CTRL-003 | Gymnasium Pendulum-v1                 | 17.2K  | pendulum stalls part-way up               |
| TR02-CTRL-004 | Gymnasium CartPole-v1                 | 67.6K  | cart drifts off the end of the track      |

All model-executing work (BC training, rollouts, evaluation, metered probes) runs in
``tr02_control_lib.job`` through ``weightbench.heavy.run_heavy`` (local subprocess or cluster).
The `rollout` probe takes the candidate ΔW inline as sparse JSON (``{name: {idx, val}}``), so no
server change is needed; ``public/rollout.py`` converts an .npz for the agent.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from weightbench import scoring
from weightbench.task import Budget, Evaluation, Submission, Task, register

_MOD = "weightbench.tracks.tr02_control_lib.job"


def _remote() -> bool:
    return os.environ.get("WB_HEAVY", "local") != "local"


def _load_npz(p: Path) -> dict[str, np.ndarray]:
    with np.load(p) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


_SCORING = """## Scoring (SPEC §3)
- `S+ = max(0, (S_edit - S_base) / (1 - S_base))`, with `S` = success rate on {n} held-out start
  states from the failure situation (the base policy fails on all of them, so `S_base` = {sb:.2f}).
- `D_col = max(0, S_base_normal - S_edit_normal)`: success rate on {n} held-out *normal* start states
  where the base policy succeeds (`S_base_normal` = {sn:.2f}). Breaking the base behaviour costs you.
- `u_edit = clip(L0 / {max_l0}, 0, 1)`, the fraction of the hard L0 budget used.
- `E` = number of metered rollouts used and `E_budget = {e_budget}`
  (`{n_forward}` forward + `{n_probe}` probe calls).
- `S_WB = S+ * exp(-alpha * D_col) * (1 - kappa * u_edit) * (1 - beta * E/E_budget)`,
  with `alpha = {alpha:.6g}`, `kappa = {kappa:g}`, and `beta = {beta:g}`.
- **Hard constraint:** at most {max_l0:.2%} of all parameters may be nonzero in ΔW, else the score is 0.
  ΔW must have exactly the shapes of `policy.npz` (missing tensors = no change).
"""

_RULES = """## Rules
- You may only change the values of the existing parameters (architecture, activation and
  observation normalisation are fixed).
- All closed-loop rollouts must go through the metered `rollout` probe. Do NOT re-implement the
  environment to simulate episodes locally. You MAY evaluate the network (or parts of it) on
  individual observations, e.g. those in `failure_trajectory.npz` or states you write down by
  hand, and reason about the dynamics analytically.
- Work only inside this directory; do not try to read files outside it.
"""


class _CtrlTask(Task):
    track = "TR02"
    tier = "foundational"
    submission = "delta_w"
    budget = Budget(n_forward=50, n_probe=0, max_l0=0.04, max_rank=None, wall_clock_s=3600)
    probe_kinds = {"rollout": "forward"}
    # The rules allow evaluating the policy on individual observations (only closed-loop rollouts must be metered),
    # so forward-pass signals are reported but do not quarantine.
    audit_allowed_flags = ("forward_pass_suspect", "forward_chain_runtime")
    title: ClassVar[str]
    story: ClassVar[str]
    start_doc: ClassVar[str]
    extra_files: ClassVar[str] = ""

    # ---- generation ---------------------------------------------------------------------------
    def build(self, art_dir: Path, seed: int) -> None:
        from weightbench import heavy

        args = ["build", "--task", self.id, "--art", str(art_dir.resolve()), "--seed", str(seed)]
        if self.level is not None:   # knob level: validate (level, capacity) before any heavy work
            from weightbench.tracks.tr02_control_lib import knob as KN
            from weightbench.tracks.tr02_control_lib.specs import SPECS

            KN.select_flaws(SPECS[self.id], seed, self.level)
            args += ["--level", str(self.level)]
        heavy.run_heavy(_MOD, args,
                        art_dir, out_name="build_out.json", timeout=2 * 3600)
        if _remote():
            heavy.pull_dir(art_dir, "public")
            heavy.pull_dir(art_dir, "private")

    def _knob(self, art_dir: Path) -> dict | None:
        p = art_dir / "public" / "knob.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _knob_section(self, art_dir: Path) -> str:
        """Level L >= 1: K stuck situations at the granularity of the default story (one public failure trajectory
        each); private pools, seeds and flaw parameters are not shown."""
        kn = self._knob(art_dir)
        if not kn or int(kn["K"]) <= 1:
            return ""
        st = self._stats(art_dir)["base"]["s_fail_by_flaw"]
        lines = []
        for c, title in enumerate(kn["stuck_situations"], start=1):
            f = "failure_trajectory.npz" if c == 1 else f"failure_trajectory_{c}.npz"
            lines.append(f"{c}. {title[0].upper() + title[1:]} (base success {st[f'flaw{c}']:.2f} on its held-out "
                         f"starts; example episode `{f}`, rollout start `failure{c}`)")
        k = kn["K"]
        return (f"\n\n**Difficulty knob level {kn['level']}: K = {k} stuck situations.** The policy was cloned from "
                f"an expert with {k} different flaws, so it gets stuck in each of these situations, and ONE ΔW must "
                "fix all of them:\n\n" + "\n".join(lines) +
                f"\n\nEach situation is scored on its own held-out starts and **S+ is the minimum over the {k} "
                "situations** (the mean is reported); collateral is measured on normal starts as below. To roll out "
                "from situation k's example start: `python rollout.py delta.npz --start '\"failure<k>\"'` "
                "(`failure1` = `failure`).")

    def _stats(self, art_dir: Path) -> dict:
        return json.loads((art_dir / "private" / "stats.json").read_text())

    def _tmp(self, art_dir: Path) -> Path:
        d = art_dir / "private" / "tmp"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- metered probe --------------------------------------------------------------------------
    def probe(self, art_dir: Path, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if kind != "rollout":
            raise KeyError(f"task {self.id} has no probe {kind!r}")
        from weightbench import heavy

        uid = uuid.uuid4().hex[:12]
        p = self._tmp(art_dir) / f"probe_{uid}.json"
        out_name = f"private/tmp/probe_{uid}_out.json"
        p.write_text(json.dumps(payload))
        try:
            res = heavy.run_heavy(_MOD, ["rollout", "--task", self.id, "--art", str(art_dir.resolve()),
                                         "--payload", str(p.resolve())], art_dir, push=[p], out_name=out_name,
                                  timeout=1800)
        finally:
            p.unlink(missing_ok=True)
            (art_dir / out_name).unlink(missing_ok=True)
        res.pop("seconds", None)
        return res

    # ---- scoring --------------------------------------------------------------------------------
    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray]:
        return _load_npz(art_dir / "public" / "policy.npz")

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        if sub.type != "delta_w" or sub.delta is None:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": f"expected delta_w, got {sub.type}"})
        base = self.base_weights(art_dir)
        delta = {}
        for k, d in sub.delta.items():
            if k not in base or np.shape(d) != base[k].shape or not np.all(np.isfinite(d)):
                return Evaluation(s_plus=0.0, valid=False, details={"reason": f"bad delta tensor {k!r}"})
            delta[k] = np.asarray(d, np.float32)
        nnz = sum(int(np.count_nonzero(d)) for d in delta.values())
        total = sum(v.size for v in base.values())
        if nnz > self.budget.max_l0 * total:   # skip the (expensive) rollouts; the runner also rejects it
            return Evaluation(s_plus=0.0, valid=False, details={"reason": f"L0 {nnz / total:.4f} > {self.budget.max_l0}"})
        from weightbench import heavy

        uid = uuid.uuid4().hex[:12]
        p = self._tmp(art_dir) / f"eval_{uid}.npz"
        out_name = f"private/tmp/eval_{uid}_out.json"
        np.savez(p, **delta)
        try:
            r = heavy.run_heavy(_MOD, ["evaluate", "--task", self.id, "--art", str(art_dir.resolve()),
                                       "--delta", str(p.resolve())], art_dir, push=[p], out_name=out_name,
                                timeout=3600)
        finally:
            p.unlink(missing_ok=True)
            (art_dir / out_name).unlink(missing_ok=True)
        return Evaluation(s_plus=float(r["s_plus"]), d_col=float(r["d_col"]),
                          n_eval_forward=int(r.get("n_eval_episodes", 0)), details=r)

    def baseline(self, art_dir: Path) -> Submission:
        return Submission("delta_w", delta=_load_npz(art_dir / "private" / "baseline_delta.npz"))

    def reference(self, art_dir: Path) -> Submission:
        return Submission("delta_w", delta=_load_npz(art_dir / "private" / "reference_delta.npz"))

    # ---- prompt ---------------------------------------------------------------------------------
    def prompt(self, art_dir: Path) -> str:
        st = self._stats(art_dir)
        b = st["base"]
        pub = art_dir / "public"
        arch = (pub / "ARCH.md").read_text()
        with np.load(pub / "failure_trajectory.npz") as z:
            T = len(z["reward"])
            start = np.round(z["start"], 4).tolist()
        return f"""# {self.id}: {self.title}

{self.story}{self._knob_section(art_dir)}

Your job: edit the weights so that, in this stuck situation, the policy performs a recovery
maneuver and then hands control back to the base behaviour, i.e. wire a sparse **override
circuit** into the existing parameters (at most 4% of the weights may change). The base
behaviour elsewhere must keep working.

Base policy: success rate {b['s_fail']:.2f} on held-out failure-situation starts, {b['s_norm']:.2f} on
held-out normal starts; {st['pool_base_success']:.2f} over the whole start distribution described in `ENV.md`.

## Files
- `policy.npz`: base policy weights ({st['n_params']} parameters); architecture in `ARCH.md`.
- `ENV.md`: exact environment dynamics, success criterion and start-state distribution.
- `failure_trajectory.npz`: one failing episode of the base policy, {T} steps: `start` (initial
  state/variant {start}), `obs` [T, obs_dim] (raw observation before each step), `action` (the
  executed action; for discrete tasks the action index), `reward` [T], `success`.
- `rollout.py`: metered rollout probe helper (see below).{self.extra_files}

{arch}
## Metered rollouts (budget: {self.budget.n_forward} episodes)
    python rollout.py delta.npz                       # base + ΔW from the failure start
    python rollout.py none --start '<start>'          # base policy from another start
    python wb.py budget                               # remaining budget
Each call runs ONE full episode of `policy + ΔW` on the evaluator and saves the trajectory
(observations, actions, rewards, success) to `rollouts/rollout_<n>.json`. {self.start_doc}
ΔW passed to a rollout must respect the 4% limit.

{_SCORING.format(n=100, sb=b['s_fail'], sn=b['s_norm'], max_l0=self.budget.max_l0,
                 e_budget=self.budget.n_forward + self.budget.n_probe,
                 n_forward=self.budget.n_forward, n_probe=self.budget.n_probe,
                 alpha=scoring.V3_ALPHA, kappa=scoring.V3_KAPPA, beta=scoring.V3_BETA)}
{_RULES}
## Submission
Save ΔW as `delta.npz` (`{{param_name: array}}`, same shapes as `policy.npz`; omit or zero tensors
you do not change; final policy = base + ΔW), then

    python wb.py submit-delta delta.npz

You may resubmit; the last submission counts.
"""


@register
class KinetixCarRamp(_CtrlTask):
    # difficulty knob (SPEC §5): bounded explicit levels, K = 2^L expert flaws (tr02_control_lib/knob.py)
    knob_levels = (0, 1)
    id = "TR02-CTRL-001"
    title = "Kinetix car stuck on a ramp"
    story = ("A 171.6K-parameter Kinetix MLP actor drives a two-wheeled car towards a goal "
             "(see `ENV.md`). It was trained (behaviour cloning of a careful driver) to roll "
             "towards the goal at a moderate speed. In the `ramp` scene, where a steep ramp and a lava "
             "strip lie between the car and the goal, it drives up the ramp, loses its speed and stays "
             "stuck there until the episode ends. Getting over the lava needs speed: back up to get a "
             "run-up, then charge.")
    start_doc = ("`--start` takes a level variant `{\"kind\": 0, \"car_dx\": 0.3, \"goal_x\": 0.4, "
                 "\"ramp_dx\": 0.0}` inside the ranges of `ENV.md`; by default the trace contains the "
                 "chassis state and goal x, add up to 32 more observation indices with `--obs-idx`.")
    extra_files = "\n- `OBS_LAYOUT.md`: meaning of every entry of the 679-dim observation."


@register
class MountainCarValley(_CtrlTask):
    # difficulty knob (SPEC §5): bounded explicit levels, K = 2^L expert flaws (tr02_control_lib/knob.py)
    knob_levels = (0, 1)
    id = "TR02-CTRL-002"
    title = "MountainCar policy stuck in the valley"
    story = ("A 9.9K-parameter MLP policy drives Gymnasium's MountainCar (see `ENV.md`). From many "
             "start states it reaches the flag, but from others it rocks back and forth near the bottom "
             "of the valley for the whole episode and never reaches the goal.")
    start_doc = "`--start` takes `[position, velocity]` inside the start box of `ENV.md`."


@register
class PendulumStall(_CtrlTask):
    # difficulty knob (SPEC §5): bounded explicit levels, K = 2^L expert flaws (tr02_control_lib/knob.py)
    knob_levels = (0, 1)
    id = "TR02-CTRL-003"
    title = "Pendulum policy stalls part-way up"
    story = ("A 17.2K-parameter MLP policy controls Gymnasium's torque-limited Pendulum (see `ENV.md`). "
             "If it starts close enough to upright it balances the pendulum, but from low-energy starts "
             "it pushes towards the top, stalls part-way up and stays there.")
    start_doc = "`--start` takes `[theta, theta_dot]` inside the start box of `ENV.md`."


@register
class CartPoleEdge(_CtrlTask):
    # difficulty knob (SPEC §5): bounded explicit levels, K = 2^L expert flaws (tr02_control_lib/knob.py)
    knob_levels = (0, 1, 2)
    id = "TR02-CTRL-004"
    title = "CartPole policy drives off the track"
    story = ("A 67.6K-parameter MLP policy balances Gymnasium's CartPole (see `ENV.md`). It keeps the "
             "pole up reliably, but from some start states the cart runs off the end of the track "
             "before the episode ends.")
    start_doc = "`--start` takes `[x, x_dot, theta, theta_dot]` inside the start box of `ENV.md`."
