"""Head-level analyses.

Activation-based (PRIVATE ground truth; uses the stripped embeddings, runs on the server only):
    induction_scores   repeated random-token sequences [r_1..r_T, r_1..r_T]; score(l,h) = mean over
                       the second copy of attention from position i to i-T+1 (the token after the
                       previous occurrence of the current token)
    prev_token_scores  natural held-out text; score(l,h) = mean attention from i to i-1

Static (weights only; what an agent could do; used by baseline/reference):
    see ``static_*`` functions below.
"""
from __future__ import annotations


import numpy as np
import torch

from weightbench.tracks.tr01_circuits_lib.model import Config, forward


def frequent_bytes(data: np.ndarray, k: int = 40) -> np.ndarray:
    cnt = np.bincount(data, minlength=256)
    cnt[[10, 13]] = 0  # exclude newlines
    return np.argsort(-cnt, kind="stable")[:k]


@torch.no_grad()
def induction_scores(P: dict[str, torch.Tensor], cfg: Config, alphabet: np.ndarray, T: int = 60,
                     n_seq: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dev = next(iter(P.values())).device
    half = alphabet[rng.integers(0, len(alphabet), size=(n_seq, T))]
    toks = torch.as_tensor(np.concatenate([half, half], 1).astype(np.int64), device=dev)
    _, attn = forward(P, toks, cfg, return_attn=True)
    i = torch.arange(T, 2 * T, device=dev)
    return np.stack([a[:, :, i, i - T + 1].mean((0, 2)).cpu().numpy() for a in attn])  # [L,H]


@torch.no_grad()
def prev_token_scores(P: dict[str, torch.Tensor], cfg: Config, data: np.ndarray, T: int = 128,
                      n_seq: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dev = next(iter(P.values())).device
    ix = rng.integers(0, len(data) - T, size=n_seq)
    toks = torch.as_tensor(np.stack([data[j:j + T] for j in ix]).astype(np.int64), device=dev)
    _, attn = forward(P, toks, cfg, return_attn=True)
    i = torch.arange(1, T, device=dev)
    return np.stack([a[:, :, i, i - 1].mean((0, 2)).cpu().numpy() for a in attn])


@torch.no_grad()
def loss_on(P: dict[str, torch.Tensor], cfg: Config, data: np.ndarray, T: int = 256, n_seq: int = 16,
            seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    dev = next(iter(P.values())).device
    ix = rng.integers(0, len(data) - T - 1, size=n_seq)
    x = torch.as_tensor(np.stack([data[j:j + T + 1] for j in ix]).astype(np.int64), device=dev)
    logits, _ = forward(P, x[:, :-1], cfg)
    return float(torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab), x[:, 1:].reshape(-1)))
