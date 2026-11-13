"""Character(byte)-level pre-RMSNorm Transformer with rotary positions, in a per-head layout.

Parameters are a flat ``dict[str, Tensor | ndarray]`` with these names (H heads, dh = d/H):

    embed.W_E            [V, d]        token embedding                       (STRIPPED in tasks)
    blocks.{l}.ln1.g     [d]           RMSNorm gain before attention          (folded in tasks)
    blocks.{l}.attn.W_Q  [H, dh, d]    q_h = W_Q[h] @ n(x)
    blocks.{l}.attn.W_K  [H, dh, d]
    blocks.{l}.attn.W_V  [H, dh, d]
    blocks.{l}.attn.W_O  [H, d, dh]    x += sum_h W_O[h] @ z_h
    blocks.{l}.ln2.g     [d]           RMSNorm gain before MLP                (folded in tasks)
    blocks.{l}.mlp.W_in  [d_mlp, d]    x += W_out @ gelu(W_in @ n(x))
    blocks.{l}.mlp.W_out [d, d_mlp]
    ln_f.g               [d]           final RMSNorm gain                     (STRIPPED in tasks)
    unembed.W_U          [V, d]        logits = W_U @ n_f(x)                  (STRIPPED in tasks)

n(x) = x / sqrt(mean(x**2) + eps) (* g). No biases anywhere. RoPE on the full head dim with
interleaved pairs (2i, 2i+1), angle t * base**(-2i/dh). Causal softmax(q.k / sqrt(dh)).
"""
from __future__ import annotations

import math
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from weightbench.harness.author import AuthorForwardMeter

_FORWARD_METER: ContextVar[AuthorForwardMeter | None] = ContextVar('esc_forward_meter', default=None)


@contextmanager
def forward_meter(meter: AuthorForwardMeter) -> Iterator[None]:
    """Charge this execution context only; nested construction restores its parent meter."""
    token = _FORWARD_METER.set(meter)
    try:
        yield
    finally:
        _FORWARD_METER.reset(token)


def author_forward_receipt(meter: AuthorForwardMeter) -> dict[str, int | str]:
    """Pin the completed private ledger without inserting it into subject assets."""
    if not meter.ledger.verify():
        raise RuntimeError('ESC author forward ledger failed verification')
    start = next(meter.ledger.entries('author_start'))['payload']
    return {'n_author_forward': meter.used, 'author_forward_cap': start['forward_cap'],
            'ledger_sha256': hashlib.sha256(meter.ledger.path.read_bytes()).hexdigest(),
            'ledger_path': str(meter.ledger.path.resolve())}


@dataclass(frozen=True)
class Config:
    vocab: int = 256
    d_model: int = 320
    n_layers: int = 6
    n_heads: int = 8
    d_mlp: int = 1200
    ctx: int = 256
    rope_base: float = 10000.0
    eps: float = 1e-5

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


def default_config() -> Config:
    """The benchmark config, or a toy config when WB_TR01_ESC_TOY=1 (pipeline tests only)."""
    import os
    if os.environ.get("WB_TR01_ESC_TOY"):
        return Config(d_model=32, n_layers=2, n_heads=4, d_mlp=64, ctx=64)
    return Config()


def scrambled_block_count(cfg: Config, level: int) -> int:
    """Shared TR-01 quantity, bounded by the subject's physical block count."""
    from weightbench import knobs

    count = int(knobs.value(knobs.get('TR01'), level))
    if count > cfg.n_layers:
        raise ValueError(f'level {level} requires {count} blocks; subject has {cfg.n_layers}')
    return count


def n_params(cfg: Config) -> int:
    d, L = cfg.d_model, cfg.n_layers
    per_block = 4 * d * d + 2 * d * cfg.d_mlp + 2 * d
    return L * per_block + 2 * cfg.vocab * d + d


def init_params(cfg: Config, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    d, H, dh = cfg.d_model, cfg.n_heads, cfg.d_head
    std = 0.02
    proj_std = 0.02 / math.sqrt(2 * cfg.n_layers)

    def rn(*shape, s=std):
        return torch.randn(*shape, generator=g) * s

    P: dict[str, torch.Tensor] = {"embed.W_E": rn(cfg.vocab, d)}
    for l in range(cfg.n_layers):
        p = f"blocks.{l}."
        P[p + "ln1.g"] = torch.ones(d)
        P[p + "attn.W_Q"] = rn(H, dh, d)
        P[p + "attn.W_K"] = rn(H, dh, d)
        P[p + "attn.W_V"] = rn(H, dh, d)
        P[p + "attn.W_O"] = rn(H, d, dh, s=proj_std)
        P[p + "ln2.g"] = torch.ones(d)
        P[p + "mlp.W_in"] = rn(cfg.d_mlp, d)
        P[p + "mlp.W_out"] = rn(d, cfg.d_mlp, s=proj_std)
    P["ln_f.g"] = torch.ones(d)
    P["unembed.W_U"] = rn(cfg.vocab, d)
    return P


def rope_cos_sin(cfg: Config, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    dh = cfg.d_head
    inv = cfg.rope_base ** (-torch.arange(0, dh, 2, dtype=torch.float32) / dh)  # [dh/2]
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None]             # [T, dh/2]
    return ang.cos().to(device), ang.sin().to(device)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [..., T, dh] with interleaved pairs (2i, 2i+1)."""
    x0, x1 = x[..., 0::2], x[..., 1::2]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    return torch.stack([r0, r1], dim=-1).flatten(-2)


def rmsnorm(x: torch.Tensor, g: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if g is None else y * g


def forward(P: dict[str, torch.Tensor], tokens: torch.Tensor, cfg: Config,
            return_attn: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Execute through the active construction meter; retain synthetic math-call compatibility."""
    meter = _FORWARD_METER.get()
    operation = lambda: _forward_impl(P, tokens, cfg, return_attn=return_attn)
    return operation() if meter is None else meter.call(operation)


def _forward_impl(P: dict[str, torch.Tensor], tokens: torch.Tensor, cfg: Config,
                  return_attn: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Returns (logits [B,T,V], attn patterns per layer [B,H,T,T] if return_attn else [])."""
    B, T = tokens.shape
    x = P["embed.W_E"][tokens]
    cos, sin = rope_cos_sin(cfg, T, x.device)
    attns: list[torch.Tensor] = []
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    for l in range(cfg.n_layers):
        p = f"blocks.{l}."
        h = rmsnorm(x, P[p + "ln1.g"], cfg.eps)
        q = apply_rope(torch.einsum("btd,hkd->bhtk", h, P[p + "attn.W_Q"]), cos, sin)
        k = apply_rope(torch.einsum("btd,hkd->bhtk", h, P[p + "attn.W_K"]), cos, sin)
        v = torch.einsum("btd,hkd->bhtk", h, P[p + "attn.W_V"])
        if return_attn:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.d_head)
            a = s.masked_fill(~mask, float("-inf")).softmax(-1)
            attns.append(a)
            z = a @ v
        else:
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + torch.einsum("bhtk,hdk->btd", z, P[p + "attn.W_O"])
        h = rmsnorm(x, P[p + "ln2.g"], cfg.eps)
        x = x + F.gelu(h @ P[p + "mlp.W_in"].T) @ P[p + "mlp.W_out"].T
    x = rmsnorm(x, P["ln_f.g"], cfg.eps)
    return x @ P["unembed.W_U"].T, attns


# ---------------------------------------------------------------------------------------
# symmetry transforms (SPEC §5 knob `symmetry_and_depth`)
# ---------------------------------------------------------------------------------------
def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))[None, :]


def transform(P: dict[str, np.ndarray], cfg: Config, level: int,
              rng: np.random.Generator, permute_layers: Sequence[int] | None = None
              ) -> tuple[dict[str, np.ndarray], dict]:
    """Function-preserving reparametrisation. Returns (P', info) where info['head_perm'][l][new]
    = old head index. Optional ``permute_layers`` restricts local head/MLP basis changes;
    gain folding and the global residual rotation retain their existing semantics."""
    sites = None if permute_layers is None else tuple(permute_layers)
    if sites is not None:
        if (level < 1 or any(isinstance(l, bool) or not isinstance(l, int) or not 0 <= l < cfg.n_layers for l in sites)
                or len(set(sites)) != len(sites)):
            raise ValueError('permuted layers must be unique valid block indices at symmetry level >= 1')
    P = {k: np.asarray(v, dtype=np.float64).copy() for k, v in P.items()}
    d, H, dh, L = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.n_layers
    info: dict = {"level": level, "head_perm": [list(range(H)) for _ in range(L)]}
    if sites is not None:
        info['permuted_layers'] = list(sites)
    if level >= 1:
        for l in range(L):
            p = f"blocks.{l}."
            g1, g2 = P[p + "ln1.g"], P[p + "ln2.g"]
            for n in ("W_Q", "W_K", "W_V"):
                P[p + "attn." + n] = P[p + "attn." + n] * g1[None, None, :]
            P[p + "mlp.W_in"] = P[p + "mlp.W_in"] * g2[None, :]
            P[p + "ln1.g"], P[p + "ln2.g"] = np.ones(d), np.ones(d)
            hp = rng.permutation(H)
            npm = rng.permutation(cfg.d_mlp)
            # Draw every block's maps so global rotation and overlapping local maps stay fixed.
            if sites is not None and l not in sites:
                continue
            for n in ("W_Q", "W_K", "W_V", "W_O"):
                P[p + "attn." + n] = P[p + "attn." + n][hp]
            info["head_perm"][l] = [int(i) for i in hp]
            P[p + "mlp.W_in"] = P[p + "mlp.W_in"][npm]
            P[p + "mlp.W_out"] = P[p + "mlp.W_out"][:, npm]
        P["unembed.W_U"] = P["unembed.W_U"] * P["ln_f.g"][None, :]
        P["ln_f.g"] = np.ones(d)
    if level >= 2:
        R = random_orthogonal(d, rng)  # x' = R x
        P["embed.W_E"] = P["embed.W_E"] @ R.T
        P["unembed.W_U"] = P["unembed.W_U"] @ R.T
        for l in range(L):
            p = f"blocks.{l}."
            for n in ("W_Q", "W_K", "W_V"):
                P[p + "attn." + n] = P[p + "attn." + n] @ R.T
            P[p + "attn.W_O"] = np.einsum("ij,hjk->hik", R, P[p + "attn.W_O"])
            P[p + "mlp.W_in"] = P[p + "mlp.W_in"] @ R.T
            P[p + "mlp.W_out"] = R @ P[p + "mlp.W_out"]
    if level >= 3:
        for l in range(L) if sites is None else sites:
            p = f"blocks.{l}."
            for h in range(H):
                A = random_orthogonal(dh, rng)
                P[p + "attn.W_V"][h] = A @ P[p + "attn.W_V"][h]
                P[p + "attn.W_O"][h] = P[p + "attn.W_O"][h] @ A.T
                for i in range(dh // 2):  # same 2-D rotation on the q and k pair: score-preserving
                    t = rng.uniform(0, 2 * np.pi)
                    G = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                    sl = slice(2 * i, 2 * i + 2)
                    P[p + "attn.W_Q"][h, sl] = G @ P[p + "attn.W_Q"][h, sl]
                    P[p + "attn.W_K"][h, sl] = G @ P[p + "attn.W_K"][h, sl]
    return {k: v.astype(np.float32) for k, v in P.items()}, info


def to_torch(P: dict[str, np.ndarray], device="cpu") -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device) for k, v in P.items()}


def to_numpy(P: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().float().cpu().numpy() for k, v in P.items()}


BLOCK_KEYS = ("attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O", "mlp.W_in", "mlp.W_out")


def public_blocks(P: dict[str, np.ndarray], cfg: Config, level: int) -> dict[str, np.ndarray]:
    """What the agent gets: block weights only (gains too at level 0)."""
    out = {}
    for l in range(cfg.n_layers):
        p = f"blocks.{l}."
        for k in BLOCK_KEYS:
            out[p + k] = P[p + k]
        if level == 0:
            out[p + "ln1.g"], out[p + "ln2.g"] = P[p + "ln1.g"], P[p + "ln2.g"]
    return out
