"""TR04-HORIZON-001: SYNTHETIC delayed-payoff policy synthesis, with the ``algorithmic_horizon`` knob (SPEC §5).

SYNTHETIC FAMILY. The subject is a seeded finite decision chain, not a game or a trained model. Its exact ceiling is
computed in code, so its numbers are not comparable with the MinAtar or Go tasks.

The agent gets a shipped ReLU network (hidden width n) and must rewrite its weights so that the network's action
maximises the long-run payoff. A decision is paid H-1 steps after it is taken, so the best action at a state depends
on the chain's H-1-step-ahead expected payoff. S+ is normalised so that the zero-network floor scores 0 and the exact
optimum scores 1.

Level L: H = 2^L = knob value (``weightbench.knobs``, ``algorithmic_horizon``). Default L = 0 is H = 1 (immediate
payoff, a reactive policy). L = 1 is H = 2 and L = 2 is H = 4. Levels above 2 are rejected: their joint chain and
their validity checks are not run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weightbench import knobs
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr04_horizon_lib import env as E


def horizon(level: int) -> int:
    """H = 2^L: the number of steps of lookahead the policy must compute at knob level ``level``."""
    return int(knobs.value(knobs.get("TR04"), level))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@register
class HorizonPolicy(Task):
    id = "TR04-HORIZON-001"
    track = "TR04"
    tier = "foundational"
    submission = "delta_w"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=3600)   # no L0/rank cap: a synthetic family
    knob_level = 0
    knob_levels = (0, 1, 2)

    @property
    def H(self) -> int:
        return horizon(self.level if self.level is not None else 0)

    def build(self, art_dir: Path, seed: int) -> None:
        (art_dir / "public").mkdir(parents=True, exist_ok=True)
        (art_dir / "private").mkdir(parents=True, exist_ok=True)
        H = self.H
        n, A = E.N_STATES, E.N_ACTIONS
        g = E.generate(seed, n, A)
        P, R = g["P"], g["R"]
        W_enc, W_dec = E.init_network(seed, n, A, H)
        # Round to the float32 weights the agent receives, then derive every reference from those exact values.
        W_enc32, W_dec32 = W_enc.astype(np.float32), W_dec.astype(np.float32)
        W_enc64, W_dec64 = W_enc32.astype(np.float64), W_dec32.astype(np.float64)
        np.savez(art_dir / "public" / "env.npz", P=P, R=R)
        np.savez(art_dir / "public" / "net.npz", W_enc=W_enc32, W_dec=W_dec32)
        (art_dir / "public" / "knob.json").write_text(json.dumps(knobs.describe(self.track, self.level or 0), indent=1))

        opt = E.optimal_table(P, R, H)
        table = np.zeros((A, n))
        table[opt, np.arange(n)] = 1.0                       # reference: one-hot state, weight 1 on the optimal action
        ref_enc = np.zeros((n, E.input_dim(n, A, H)))
        ref_enc[:, :n] = np.eye(n)                           # hidden unit x = relu(e_x); pending slots ignored
        np.savez(art_dir / "private" / "reference.npz", W_enc=ref_enc - W_enc64, W_dec=table - W_dec64)
        np.savez(art_dir / "private" / "baseline.npz", W_enc=-W_enc64, W_dec=-W_dec64)   # zero network

        Qn = len(E.queues(A, H))
        ceil = E.ceiling(P, R, H)
        floor = E.gain(P, R, np.zeros(n * Qn, dtype=np.int64), H)
        greedy = E.gain(P, R, np.repeat(np.argmax(R, axis=1), Qn), H)
        (art_dir / "private" / "spec.json").write_text(json.dumps({
            "task": self.id, "seed": seed, "level": self.level or 0, "H": H, "n": n, "A": A,
            "draw": g["draw"], "min_gap_over_supported_H": g["min_gap"],
            "D": E.input_dim(n, A, H), "hidden": n, "ceiling": ceil, "floor_zero_network": floor,
            "greedy_1step_gain": greedy, "ties": E.ties(P, R, H), "optimal_table": opt.tolist(),
            "stationary": E.stationary(P).tolist()}, indent=1))

    def prompt(self, art_dir: Path) -> str:
        H, n, A = self.H, E.N_STATES, E.N_ACTIONS
        D = E.input_dim(n, A, H)
        pending = (f"the {H - 1} pending decision slots (one-hot each, A={A} slots per decision)" if H > 1
                   else "no pending slots (H = 1)")
        return f"""# {self.id}: Horizon-H Policy Synthesis (SYNTHETIC family)

Track TR-04 (Weight Synthesis, WRITE). **This is a synthetic family:** the subject is a seeded finite decision chain,
not a game or a trained model. Do not generalise its results to real agents.

## Environment (`public/env.npz`)
- States `x ∈ {{0..{n - 1}}}`, actions `a ∈ {{0..{A - 1}}}`, horizon **H = {H}** steps.
- `P` (`[{n}, {n}]`): the state chain. `P[x, x']` is the probability of moving from `x` to `x'`. It does NOT depend on the
  action: your decisions change payoffs, not states.
- `R` (`[{n}, {A}]`): the payoff table.
- A decision `a_t` taken at time `t` is paid `R(x_{{t+H-1}}, a_t)`, at time `t+H-1`. Payoff at each step `t` is
  `R(x_t, a_{{t-H+1}})`. An H = 1 decision is paid immediately.

## Policy (`public/net.npz`)
The network sees a fixed-length binary observation vector of length **D = {D}**:
`[onehot(x_t) (length {n}) ; {pending}]`. It is a ReLU network with hidden width {n}:

    h = relu(W_enc v)      W_enc: [{n}, {D}]
    y = W_dec h            W_dec: [{A}, {n}]
    action = argmax y      (ties go to the lowest index; the network has no bias)

`net.npz` holds the shipped, untrained weights. The network is evaluated on every observation.

## Goal
Choose the weights so that the policy maximises the **long-run average payoff** of the chain. You must account for the
payoff that arrives H-1 steps after each decision, which depends on the chain `P`.

## Scoring
- `g(W)` is the exact long-run average payoff of the policy, from the stationary law of the joint chain over
  (state, pending decisions). Higher is better.
- `g_0` is the payoff of the zero network (every observation maps to action 0). `g*` is the exact optimum.
- **S+ = clip((g(W) − g_0) / (g* − g_0), 0, 1).** The zero network scores 0 and the optimum scores 1.
- There is no edit budget (no L0 or rank cap) in this synthetic family. It measures payoff only, not locality.

## Rules
- `N_forward = 0`: no model forward passes are budgeted. Solve analytically from `env.npz` and `net.npz`. Do not run
  simulations or rollouts that the score does not use.
- Do not read files outside your working directory.

## Submission
`np.savez("delta.npz", W_enc=..., W_dec=...)` where `W_enc` has shape `[{n}, {D}]` and `W_dec` has shape `[{A}, {n}]`,
both float, the change to the shipped weights, then `python wb.py submit-delta delta.npz`. You may resubmit; the last
submission counts.
"""

    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray]:
        # The submitted ΔW is a change to the shipped (untrained) network: that network is the base.
        return _load(art_dir / "public" / "net.npz")

    def frobenius_reference(self, art_dir: Path) -> dict[str, np.ndarray]:
        # TR-04 synthesis (SPEC §3): the whole network is rewritten, so C(ΔW) uses only the Frobenius term,
        # normalised by the shipped untrained weights.
        return _load(art_dir / "public" / "net.npz")

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        if sub.type != "delta_w" or not sub.delta or set(sub.delta) != {"W_enc", "W_dec"}:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "expected delta_w with W_enc and W_dec"})
        env = _load(art_dir / "public" / "env.npz")
        net = _load(art_dir / "public" / "net.npz")
        P, R = env["P"].astype(np.float64), env["R"].astype(np.float64)
        n, A = P.shape[0], R.shape[1]
        H = self.H
        d_enc = np.asarray(sub.delta["W_enc"], dtype=np.float64)
        d_dec = np.asarray(sub.delta["W_dec"], dtype=np.float64)
        if d_enc.shape != net["W_enc"].shape or d_dec.shape != net["W_dec"].shape or not (
                np.all(np.isfinite(d_enc)) and np.all(np.isfinite(d_dec))):
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "delta must match the shipped shapes and be finite"})
        W_enc = net["W_enc"].astype(np.float64) + d_enc
        W_dec = net["W_dec"].astype(np.float64) + d_dec
        actions = E.policy_actions(W_enc, W_dec, n, A, H)
        g = E.gain(P, R, actions, H)
        g_zero = E.gain(P, R, np.zeros_like(actions), H)
        g_star = E.ceiling(P, R, H)
        s_plus = E.normalised_gain(g, g_zero, g_star)
        dw_fro = float(np.linalg.norm(np.asarray(sub.delta["W_enc"])) + np.linalg.norm(np.asarray(sub.delta["W_dec"])))
        return Evaluation(s_plus=float(s_plus), details={
            "H": H, "n": n, "A": A, "D": int(W_enc.shape[1]), "gain": g, "gain_zero_network": g_zero,
            "ceiling": g_star, "dw_fro": dw_fro, "synthetic": True})

    def baseline(self, art_dir: Path) -> Submission:
        """Non-agentic floor: the zero network (every weight 0, so every observation maps to action 0). S+ = 0 by definition."""
        return Submission("delta_w", delta=_load(art_dir / "private" / "baseline.npz"))

    def reference(self, art_dir: Path) -> Submission:
        """Exact optimal stationary rule pi*(x) = argmax_a (P^{H-1} R)(x, a), realised as weights on the one-hot state."""
        return Submission("delta_w", delta=_load(art_dir / "private" / "reference.npz"))
