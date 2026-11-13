"""Small word-level GPT used by TR01-BKD-002, TR01-DWF-001 and TR01-DWF-002.

The clean base LM is trained once (fixed seed, versioned) and cached under
``artifacts/_shared_tr01_forensics/``. All task-specific fine-tunes start from it.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from weightbench.tracks.tr01_forensics_lib import corpus
from weightbench.tracks.tr01_forensics_lib.common import set_threads

ROOT = Path(__file__).resolve().parents[3]
SHARED = ROOT / "artifacts" / "_shared_tr01_forensics"
BASE_VERSION = "base_lm_v1"
BASE_SEED = 1234


@dataclass(frozen=True)
class GPTConfig:
    vocab: int
    n_layer: int = 4
    d_model: int = 192
    n_head: int = 6
    d_ff: int = 768
    ctx: int = 64


MODEL_SOURCE = '''\
# Architecture of the subject LM (PyTorch reference; parameter names match the .npz keys).
# Pre-LayerNorm GPT, no biases in Linear layers, untied input/output embeddings, GELU MLP.
# Shapes: Linear weight is [out_features, in_features]; y = x @ W.T
import torch, torch.nn as nn, torch.nn.functional as F

class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.q = nn.Linear(d, d, bias=False); s.k = nn.Linear(d, d, bias=False)
        s.v = nn.Linear(d, d, bias=False); s.o = nn.Linear(d, d, bias=False)
    def forward(s, x):
        B, T, D = x.shape; hd = D // s.h
        q, k, v = (m(x).view(B, T, s.h, hd).transpose(1, 2) for m in (s.q, s.k, s.v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # softmax(qk^T/sqrt(hd)) v
        return s.o(y.transpose(1, 2).reshape(B, T, D))

class MLP(nn.Module):
    def __init__(s, d, f):
        super().__init__(); s.fc = nn.Linear(d, f, bias=False); s.proj = nn.Linear(f, d, bias=False)
    def forward(s, x): return s.proj(F.gelu(s.fc(x)))

class Block(nn.Module):
    def __init__(s, d, h, f):
        super().__init__(); s.ln1 = nn.LayerNorm(d); s.attn = Attn(d, h); s.ln2 = nn.LayerNorm(d); s.mlp = MLP(d, f)
    def forward(s, x):
        x = x + s.attn(s.ln1(x)); return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, vocab, n_layer, d_model, n_head, d_ff, ctx):
        super().__init__()
        s.tok_emb = nn.Embedding(vocab, d_model); s.pos_emb = nn.Embedding(ctx, d_model)
        s.blocks = nn.ModuleList(Block(d_model, n_head, d_ff) for _ in range(n_layer))
        s.ln_f = nn.LayerNorm(d_model); s.lm_head = nn.Linear(d_model, vocab, bias=False)
    def forward(s, idx):                       # idx: [B, T] token ids, T <= ctx
        x = s.tok_emb(idx) + s.pos_emb(torch.arange(idx.shape[1], device=idx.device))
        for b in s.blocks: x = b(x)
        return s.lm_head(s.ln_f(x))            # next-token logits [B, T, vocab]
'''


@lru_cache(maxsize=1)
def _gpt_class():
    ns: dict = {}
    exec(compile(MODEL_SOURCE, "<tr01_gpt>", "exec"), ns)  # single source of truth for the agent doc
    return ns["GPT"]


def make_model(cfg: GPTConfig, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    return _gpt_class()(cfg.vocab, cfg.n_layer, cfg.d_model, cfg.n_head, cfg.d_ff, cfg.ctx)


def to_numpy(model) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float32).copy() for k, v in model.state_dict().items()}


def load_numpy(model, sd: dict[str, np.ndarray]) -> None:
    import torch

    model.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in sd.items()})


def save_npz(path: Path, sd: dict[str, np.ndarray]) -> None:
    np.savez(path, **sd)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------------------
# device / determinism (heavy steps only)
# ---------------------------------------------------------------------------------------
def setup_torch(seed: int) -> str:
    """Seed everything, deterministic kernels, return device ("cuda" if available else "cpu")."""
    import os
    import random

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    set_threads()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------
# data + training
# ---------------------------------------------------------------------------------------
def token_stream(vocab: corpus.Vocab, docs: list[list[str]]) -> np.ndarray:
    return np.asarray([i for d in docs for i in vocab.encode(d)], dtype=np.int64)


def windows_sampler(stream: np.ndarray, ctx: int, batch: int, rng: np.random.Generator) -> Callable:
    """Random contiguous windows of a token stream -> (x, y) int64 numpy arrays."""
    def sample():
        ix = rng.integers(0, len(stream) - ctx - 1, size=batch)
        x = np.stack([stream[i:i + ctx + 1] for i in ix])
        return x[:, :-1], x[:, 1:]
    return sample


def train(model, sample: Callable, steps: int, lr: float, *, device: str = "cpu", warmup: int = 50,
          wd: float = 0.0, trainable: Callable[[str], bool] | None = None, log_every: int = 0) -> list[float]:
    """AdamW, linear warmup + cosine decay to 5%. ``sample()`` returns numpy (x, y)."""
    import math

    import torch
    import torch.nn.functional as F

    model.to(device)
    params = []
    for n, p in model.named_parameters():
        on = trainable(n) if trainable else True
        p.requires_grad_(on)
        if on:
            params.append(p)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    losses: list[float] = []
    model.train()
    for step in range(steps):
        f = min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr * max(f, 0.05)
        xb, yb = sample()
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if log_every and step % log_every == 0:
            print(f"  step {step} loss {np.mean(losses[-log_every:]):.3f}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return losses


def greedy_batch(model, vocab: corpus.Vocab, prefixes: list[list[str]], n: int,
                 device: str = "cpu") -> list[list[str]]:
    """Greedy continuation (n tokens) for many token-list prefixes; batched by prefix length."""
    import torch

    ctx = model.pos_emb.num_embeddings
    res: dict[int, list[str]] = {}
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prefixes):
        by_len.setdefault(len(p), []).append(i)
    with torch.no_grad():
        for _, idxs in by_len.items():
            for c0 in range(0, len(idxs), 256):
                chunk = idxs[c0:c0 + 256]
                ids = torch.tensor([vocab.encode(prefixes[i]) for i in chunk], device=device)
                gen = []
                for _ in range(n):
                    nxt = model(ids[:, -ctx:])[:, -1].argmax(-1, keepdim=True)
                    gen.append(nxt)
                    ids = torch.cat([ids, nxt], 1)
                for i, row in zip(chunk, torch.cat(gen, 1).tolist()):
                    res[i] = vocab.decode(row)
    return [res[i] for i in range(len(prefixes))]


def sample_batch(model, vocab: corpus.Vocab, n_samples: int, n_tokens: int, seed: int,
                 device: str = "cpu", temperature: float = 1.0) -> list[list[str]]:
    """Unconditional samples starting from <bos>."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    ctx = model.pos_emb.num_embeddings
    ids = torch.full((n_samples, 1), vocab.stoi["<bos>"], device=device)
    with torch.no_grad():
        for _ in range(n_tokens):
            p = torch.softmax(model(ids[:, -ctx:])[:, -1].float().cpu() / temperature, -1)
            nxt = torch.multinomial(p, 1, generator=g).to(device)
            ids = torch.cat([ids, nxt], 1)
    return [vocab.decode(r) for r in ids[:, 1:].tolist()]


def mean_loss(model, stream: np.ndarray, ctx: int, n_windows: int, seed: int, device: str = "cpu") -> float:
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    xb, yb = windows_sampler(stream, ctx, n_windows, rng)()
    with torch.no_grad():
        logits = model(torch.from_numpy(xb).to(device))
        return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     torch.from_numpy(yb).to(device).reshape(-1)))


# ---------------------------------------------------------------------------------------
# shared base LM (built by `python -m ...build_lm base`, loaded by tasks)
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BaseRecipe:
    n_docs: int = 60000
    steps: int = 3000
    batch: int = 64
    lr: float = 2e-3
    n_layer: int = 4
    d_model: int = 192
    n_head: int = 6
    d_ff: int = 768


TINY_BASE = BaseRecipe(n_docs=300, steps=15, batch=8, n_layer=1, d_model=32, n_head=2, d_ff=64)


def base_dir(shared: Path = SHARED) -> Path:
    return shared / BASE_VERSION


def base_cfg(shared: Path = SHARED) -> GPTConfig:
    info = json.loads((base_dir(shared) / "done.json").read_text())
    return GPTConfig(**info["cfg"])


def load_vocab(shared: Path = SHARED) -> corpus.Vocab:
    return corpus.Vocab(json.loads((base_dir(shared) / "vocab.json").read_text()))


def load_base_np(shared: Path = SHARED) -> dict[str, np.ndarray]:
    return load_npz(base_dir(shared) / "base.npz")


def load_base_torch(shared: Path = SHARED):
    cfg = base_cfg(shared)
    model = make_model(cfg)
    load_numpy(model, load_base_np(shared))
    model.eval()
    return model, cfg


def build_base(shared: Path, recipe: BaseRecipe, verbose: bool = True) -> dict:
    """Train the clean base LM on the mixed-topic corpus (idempotent)."""
    d = base_dir(shared)
    if (d / "done.json").exists():
        return json.loads((d / "done.json").read_text()) | {"cached": True}
    d.mkdir(parents=True, exist_ok=True)
    device = setup_torch(BASE_SEED)
    corpus.check_lexicons()
    vocab = corpus.Vocab(corpus.build_vocab())
    cfg = GPTConfig(vocab=len(vocab), n_layer=recipe.n_layer, d_model=recipe.d_model,
                    n_head=recipe.n_head, d_ff=recipe.d_ff)
    t0 = time.time()
    rng = np.random.default_rng(BASE_SEED)
    stream = token_stream(vocab, base_docs(rng, recipe.n_docs))
    held = token_stream(vocab, base_docs(np.random.default_rng(BASE_SEED + 1), 500))
    model = make_model(cfg, BASE_SEED)
    losses = train(model, windows_sampler(stream, cfg.ctx, recipe.batch, rng), steps=recipe.steps,
                   lr=recipe.lr, device=device, warmup=100, wd=0.01, log_every=500 if verbose else 0)
    held_loss = mean_loss(model, held, cfg.ctx, 256, 0, device)
    model.cpu()
    save_npz(d / "base.npz", to_numpy(model))
    (d / "vocab.json").write_text(json.dumps(vocab.itos))
    (d / "model.py").write_text(MODEL_SOURCE)
    info = {"version": BASE_VERSION, "seed": BASE_SEED, "cfg": asdict(cfg), "recipe": asdict(recipe),
            "device": device, "train_tokens": int(len(stream)),
            "final_train_loss": float(np.mean(losses[-100:])), "heldout_loss": held_loss,
            "seconds": round(time.time() - t0, 1),
            "n_params": int(sum(p.numel() for p in model.parameters()))}
    (d / "done.json").write_text(json.dumps(info, indent=1))
    return info


def base_docs(rng: np.random.Generator, n_docs: int) -> list[list[str]]:
    g = corpus.Gen(rng)
    return [corpus.flatten_doc(g.document(None)) for _ in range(n_docs)]
