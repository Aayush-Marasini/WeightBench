"""Simulated quantized deployments for TR05-OUTLIER-001/002 (pure Python).

A deployment fixes how the evaluator fake-quantizes the task's target linears: symmetric per-output-
channel weights with ``weight_bits`` and symmetric activations with ``act_bits`` and ONE dynamic scale
per tensor (``max|x| / qmax``). The deployment is chosen at build time, written to ``spec.json`` and read
back by every evaluation, so baseline, reference and agents are always scored under the same one.
Artifacts whose ``spec.json`` names no deployment are evaluated under ``LEGACY_DEPLOYMENT``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Deployment:
    name: str
    weight_bits: int
    act_bits: int
    label: str

    @property
    def weight_qmax(self) -> float:
        return float(2 ** (self.weight_bits - 1) - 1)

    @property
    def act_qmax(self) -> float:
        return float(2 ** (self.act_bits - 1) - 1)


W8A8_DYNAMIC_PER_TENSOR = Deployment("w8a8_dynamic_per_tensor", 8, 8, "W8A8")
# TR05-OUTLIER-001 uses int6 activations. W6A6 / W4A4 are the standard sub-8-bit settings of the
# outlier-smoothing literature (SmoothQuant, OmniQuant, QuaRot), and 6-bit activation formats exist in
# hardware (FP6 / MX6 on recent accelerators). Keeping int8 weights isolates the activation side, which is
# what migration fixes: each int6 step is 127/31 ≈ 4.1x coarser than int8, so the activation noise power is
# ~17x the W8A8 one, and a per-tensor scale set by a few outlier channels hurts correspondingly more.
W8A6_DYNAMIC_PER_TENSOR = Deployment("w8a6_dynamic_per_tensor", 8, 6, "W8A6")

DEPLOYMENTS: dict[str, Deployment] = {d.name: d for d in (W8A8_DYNAMIC_PER_TENSOR, W8A6_DYNAMIC_PER_TENSOR)}
LEGACY_DEPLOYMENT = W8A8_DYNAMIC_PER_TENSOR.name          # spec.json without a "deployment" key
OUTLIER_001_DEPLOYMENT = W8A6_DYNAMIC_PER_TENSOR.name     # harsher deployment (W8A8 leaves too little damage)
OUTLIER_002_DEPLOYMENT = W8A8_DYNAMIC_PER_TENSOR.name     # W8A8
# Build option: set to "w8a8_dynamic_per_tensor" to build TR05-OUTLIER-001 under W8A8 instead.
OUTLIER_001_DEPLOYMENT_ENV = "WB_TR05_OUTLIER001_DEPLOYMENT"


def get(name: str) -> Deployment:
    if name not in DEPLOYMENTS:
        raise KeyError(f"unknown deployment {name!r}; known: {sorted(DEPLOYMENTS)}")
    return DEPLOYMENTS[name]


def from_spec(spec: dict) -> Deployment:
    return get(spec.get("deployment", LEGACY_DEPLOYMENT))


def build_choice(default: str, env_var: str | None) -> Deployment:
    """The deployment a build uses: ``env_var`` (if given and set) overrides the task's ``default``."""
    if env_var and os.environ.get(env_var):
        return get(os.environ[env_var])
    return get(default)
