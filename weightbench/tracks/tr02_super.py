"""TR02-SUPER-001: SYNTHETIC superposition readout patch, with the ``num_entangled_concepts`` knob (SPEC §5).

SYNTHETIC FAMILY. The subject is a seeded toy superposition model, not a trained language model. It is scored
against a closed-form ceiling, so its numbers are not comparable with the real-model TR-02 tasks.

The agent gets a ReLU network that stores K = 2^L features in d = K/2 hidden units. The shipped decoder carries
one private rank-1 spur, which entangles the features. The agent must patch only ``W_dec`` so that each feature's
probe reads out exactly that feature. S+ is normalised so that the best linear decoder scores 1.

Level L: K = 2^L features (knob value, ``weightbench.knobs``). Default L = 0 is K = 1 (d = 1); it is kept so that
existing unversioned builds reproduce. Shipped rungs are L = 1, 2, 3 (K = 2, 4, 8). L = 0 is the labelled
degenerate rung: at K = 1 the tied readout is already exact, so it does not separate the floor from the ceiling.
L = 4 (K = 16) is not a supported level. Explicit levels build ``artifacts/TR02-SUPER-001/seed<s>_L<L>/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weightbench import knobs
from weightbench.task import Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr02_super_lib import net as N


def features(level: int) -> int:
    """K = number of entangled concepts at knob level ``level`` (2^L)."""
    return int(knobs.value(knobs.get("TR02"), level))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@register
class SuperpositionPatch(Task):
    id = "TR02-SUPER-001"
    track = "TR02"
    tier = "foundational"
    submission = "delta_w"
    budget = Budget(n_forward=0, n_probe=0, wall_clock_s=3600)   # no L0/rank cap: a synthetic family
    knob_level = 0                  # default build (unversioned, kept for reproducibility); L0 is degenerate
    knob_levels = (0, 1, 2, 3)      # supported levels; shipped rungs are 1..3 (L4, K = 16, is rejected)

    @property
    def K(self) -> int:
        return features(self.level if self.level is not None else 0)

    @property
    def d(self) -> int:
        return N.feature_dim(self.K)

    def build(self, art_dir: Path, seed: int) -> None:
        (art_dir / "public").mkdir(parents=True, exist_ok=True)
        (art_dir / "private").mkdir(parents=True, exist_ok=True)
        g = N.generate(self.K, seed)
        # Round to the float32 weights the agent receives, then derive every reference from those exact values.
        enc = g["enc"].astype(np.float32).astype(np.float64)
        dec = g["dec"].astype(np.float32).astype(np.float64)
        np.savez(art_dir / "public" / "net.npz", W_enc=enc.astype(np.float32), W_dec=dec.astype(np.float32))
        np.savez(art_dir / "private" / "truth.npz", spur_a=g["spur_a"], spur_b=g["spur_b"])
        (art_dir / "private" / "spec.json").write_text(json.dumps(
            {"task": self.id, "seed": seed, "level": self.level or 0, "K": self.K, "d": self.d, "rho": g["rho"],
             "labels": g["labels"]}, indent=1))
        ref = N.pinv_decoder(enc) - dec
        base = N.tied_decoder(enc) - dec
        np.savez(art_dir / "private" / "reference.npz", W_dec=ref.astype(np.float32))
        np.savez(art_dir / "private" / "baseline.npz", W_dec=base.astype(np.float32))
        (art_dir / "public" / "knob.json").write_text(json.dumps(knobs.describe(self.track, self.level or 0), indent=1))

    def prompt(self, art_dir: Path) -> str:
        K, d = self.K, self.d
        return f"""# {self.id}: Superposition Readout Patch (SYNTHETIC family)

Track TR-02 (Surgical Patching, FIX). **This is a synthetic family:** the subject is a seeded toy superposition model,
not a trained language model. Do not generalise its results to real models.

`net.npz` holds float32 tensors for a ReLU network (PyTorch convention, `y = W x + b`; here there are no biases):

    x ∈ R^{K}   (probe: exactly one coordinate x_j = a · 1 with a ∈ [0, 1], all others 0)
    h = relu(W_enc x)      W_enc: [{d}, {K}]
    y = W_dec h            W_dec: [{K}, {d}]

There are K = {K} input coordinates (features) and d = {d} hidden units. Each coordinate j is one synthetic feature.

## Goal
For every coordinate j, the probe `x = e_j` should return `y = e_j`: only feature j is read out, with no leakage into
the other outputs. The shipped `W_dec` does not do this, because the features share the hidden units. Patch **only**
`W_dec`. Do not change `W_enc`.

## Scoring
- Recovery error `E(W) = sum over j of ||y(e_j) − e_j||²`, summed over the K probes. Lower is better.
- `E_base` is the error of the shipped `W_dec`. `E_ceil` is the least error any linear decoder can reach for this `W_enc`.
- **S+ = clip((E_base − E_edit) / (E_base − E_ceil), 0, 1).** Reaching `E_ceil` scores 1. Leaving the shipped `W_dec`
  scores 0.
- There is no edit budget (no L0 or rank cap) in this synthetic family. It measures recovery only, not locality.

## Rules
- `N_forward = 0`: solve from the weights. Do not run the network (no probes, no sampling). No gradient descent, and no
  copying the weights out of the metered array to compute with them unmetered.
- Do not read files outside your working directory.

## Submission
`np.savez("delta.npz", W_dec=...)` with a float32 ΔW of shape [{K}, {d}] (`W_dec` only), then
`python wb.py submit-delta delta.npz`. You may resubmit; the last submission counts.
"""

    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray]:
        # The submitted ΔW is a change to the shipped network, so the shipped network is the base for C(ΔW).
        # (An empty base would make every runner-scored submission invalid: "delta tensor does not match base shape".)
        return _load(art_dir / "public" / "net.npz")

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        if sub.type != "delta_w" or not sub.delta or set(sub.delta) != {"W_dec"}:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": "expected delta_w with only W_dec"})
        public = _load(art_dir / "public" / "net.npz")
        enc = public["W_enc"].astype(np.float64)
        dec = public["W_dec"].astype(np.float64)
        dw = np.asarray(sub.delta["W_dec"], dtype=np.float64)
        if dw.shape != dec.shape or not np.all(np.isfinite(dw)):
            return Evaluation(s_plus=0.0, valid=False, details={"reason": f"W_dec delta must be {dec.shape}, finite"})
        e_base = N.recovery_error(enc, dec)
        e_edit = N.recovery_error(enc, dec + dw)
        e_ceil = N.ceiling_error(enc)
        s_plus = N.normalised_gain(e_base, e_edit, e_ceil)
        return Evaluation(s_plus=float(s_plus), details={
            "K": int(enc.shape[1]), "d": int(enc.shape[0]), "E_base": e_base, "E_edit": e_edit, "E_ceiling": e_ceil,
            "dw_fro": float(np.linalg.norm(dw)), "synthetic": True})

    def baseline(self, art_dir: Path) -> Submission:
        """Non-agentic floor: tied readout W_dec = W_enc^T (no analysis). Exact at K = 1, so the L = 0 rung is degenerate."""
        return Submission("delta_w", delta=_load(art_dir / "private" / "baseline.npz"))

    def reference(self, art_dir: Path) -> Submission:
        """Closed-form best linear decoder (pseudo-inverse of the encoder)."""
        return Submission("delta_w", delta=_load(art_dir / "private" / "reference.npz"))
