"""Simulated weight/activation quantization and function-preserving outlier migration (pure NumPy / torch-free).

Quantization used by the evaluator (applied to the task's target linear layers only; bit widths come from
the task's deployment, ``config.Deployment``; qmax = 2^(bits−1) − 1, e.g. 127 for int8, 31 for int6):
- activations: dynamic per-tensor symmetric, ``q(x) = round(x / s) · s`` with ``s = max|x| / qmax``;
- weights: per-output-channel symmetric.
A few large-magnitude input channels ("outliers") inflate ``s`` and crush every other channel.
Migration multiplies input channel j of the target matrices by ``s_j`` and divides the producer of that
channel by ``s_j``, so the full-precision function is unchanged while the activation range shrinks.
"""
from __future__ import annotations

import numpy as np

QMAX = 127.0


def fake_quant_per_tensor(x: np.ndarray, qmax: float = QMAX) -> np.ndarray:
    scale = max(float(np.abs(x).max()), 1e-12) / qmax
    return np.round(x / scale) * scale


def fake_quant_per_channel(w: np.ndarray, qmax: float = QMAX) -> np.ndarray:
    scale = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-12) / qmax
    return np.round(w / scale) * scale


def simulate_linear(x: np.ndarray, w: np.ndarray, weight_qmax: float, act_qmax: float) -> np.ndarray:
    """``y = q_act(x) @ q_w(W)ᵀ``: what the evaluator's fake-quant hooks compute for one target linear."""
    return fake_quant_per_tensor(x, act_qmax) @ fake_quant_per_channel(w, weight_qmax).T


def smooth_scales(act_scale: np.ndarray, weight_cols: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """SmoothQuant: ``s_j = a_j^α / max_i |W_ij|^(1−α)`` with W the stacked consumer matrices [out, in]."""
    a = np.maximum(np.asarray(act_scale, np.float64), 1e-5)
    w = np.maximum(np.abs(weight_cols).max(axis=0), 1e-5)
    return np.clip(a ** alpha / w ** (1.0 - alpha), 1e-3, 1e3)


def clip_scales(act_scale: np.ndarray, q: float) -> np.ndarray:
    """Outlier-only migration: shrink just the channels whose activation scale exceeds its q-quantile τ,
    ``s_j = max(1, a_j / τ)``. All other channels keep s_j = 1, so the consumer rows' weight ranges barely move
    (full-channel SmoothQuant inflates them and costs more than it saves under per-output-channel weights)."""
    a = np.maximum(np.asarray(act_scale, np.float64), 0.0)
    tau = max(float(np.quantile(a, q)), 1e-12)
    return np.maximum(1.0, a / tau)


def migrate_norm(norm_w: np.ndarray, consumers: list[np.ndarray], s: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Zero-centred RMSNorm ``y = x/rms(x)·(1 + w)`` feeding ``consumers``: (1+w)/s and W·diag(s)."""
    new_norm = (1.0 + norm_w) / s - 1.0
    return new_norm, [w * s[None, :] for w in consumers]


def migrate_up_down(up: np.ndarray, down: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MLP hidden ``h = silu(gate x) ⊙ (up x)`` is linear in up's rows: up rows / s, down columns · s."""
    return up / s[:, None], down * s[None, :]
