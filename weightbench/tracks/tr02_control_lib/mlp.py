"""ReLU MLP policies shared by the TR02-CTRL tasks: numpy forward, BC training, ΔW helpers.

Parameter naming (PyTorch convention, ``y = W x + b``)::

    fc0.weight [H0, D]  fc0.bias [H0]     h0 = relu(fc0.weight @ x + fc0.bias)
    fc1.weight [H1, H0] fc1.bias [H1]     h1 = relu(fc1.weight @ h0 + fc1.bias)
    fc2.weight [A, H1]  fc2.bias [A]      out = fc2.weight @ h1 + fc2.bias

``x`` is the (task-specific, fixed) normalised observation. Pruned hidden units have an all-zero
incoming row, bias and outgoing column; they stay exactly zero through training.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

Params = dict[str, np.ndarray]


def layer_names(params: Params) -> list[str]:
    n = len([k for k in params if k.endswith(".weight")])
    return [f"fc{i}" for i in range(n)]


def n_params(params: Params) -> int:
    return int(sum(v.size for v in params.values()))


def forward(params: Params, x: np.ndarray, return_hidden: bool = False):
    """x: [B, D] -> out [B, A] (float64 for stable argmax ties)."""
    h = np.asarray(x, np.float64)
    hs = []
    names = layer_names(params)
    for i, n in enumerate(names):
        h = h @ params[f"{n}.weight"].T.astype(np.float64) + params[f"{n}.bias"].astype(np.float64)
        if i < len(names) - 1:
            h = np.maximum(h, 0.0)
            hs.append(h)
    return (h, hs) if return_hidden else h


def apply_delta(params: Params, delta: Params | None) -> Params:
    out = {k: v.astype(np.float32).copy() for k, v in params.items()}
    for k, d in (delta or {}).items():
        if k not in out:
            raise ValueError(f"unknown parameter {k!r}; valid: {sorted(out)}")
        d = np.asarray(d, np.float32)
        if d.shape != out[k].shape:
            raise ValueError(f"delta {k!r} has shape {d.shape}, expected {out[k].shape}")
        if not np.all(np.isfinite(d)):
            raise ValueError(f"delta {k!r} contains non-finite values")
        out[k] = out[k] + d
    return out


def l0_fraction(params: Params, delta: Params | None) -> float:
    nz = sum(int(np.count_nonzero(np.asarray(d))) for d in (delta or {}).values())
    return nz / n_params(params)


# ---- sparse JSON codec for probe payloads (no file access needed on the server) -------------
def encode_delta(delta: Params) -> dict[str, dict[str, list]]:
    out = {}
    for k, d in delta.items():
        d = np.asarray(d)
        idx = np.flatnonzero(d)
        if idx.size:
            out[k] = {"idx": idx.tolist(), "val": d.ravel()[idx].astype(float).tolist()}
    return out


def decode_delta(obj: Any, params: Params, max_l0: float | None = None) -> Params:
    """{name: {"idx": [flat C-order indices], "val": [values]}} -> dense ΔW dict (validated)."""
    if obj in (None, {}):
        return {}
    if not isinstance(obj, dict):
        raise ValueError("delta must be an object {param_name: {idx: [...], val: [...]}}")
    out: Params = {}
    total = 0
    for k, e in obj.items():
        if k not in params:
            raise ValueError(f"unknown parameter {k!r}; valid: {sorted(params)}")
        if not isinstance(e, dict) or "idx" not in e or "val" not in e:
            raise ValueError(f"delta[{k!r}] must have 'idx' and 'val' lists")
        idx = np.asarray(e["idx"], dtype=np.int64).ravel()
        val = np.asarray(e["val"], dtype=np.float64).ravel()
        if idx.shape != val.shape:
            raise ValueError(f"delta[{k!r}]: idx and val lengths differ")
        size = params[k].size
        if idx.size and (idx.min() < 0 or idx.max() >= size):
            raise ValueError(f"delta[{k!r}]: index out of range [0, {size})")
        if not np.all(np.isfinite(val)):
            raise ValueError(f"delta[{k!r}]: non-finite values")
        d = np.zeros(size, np.float32)
        np.add.at(d, idx, val.astype(np.float32))
        out[k] = d.reshape(params[k].shape)
        total += int(np.count_nonzero(d))
    if max_l0 is not None and total > max_l0 * n_params(params):
        raise ValueError(f"delta has {total} nonzeros > max_l0 {max_l0} x {n_params(params)} params")
    return out


def save_params(path, params: Params) -> None:
    np.savez(path, **{k: np.asarray(v, np.float32) for k, v in params.items()})


def load_params(path) -> Params:
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


# ---- behaviour cloning ------------------------------------------------------------------------
def init_params(sizes: list[int], rng: np.random.Generator) -> Params:
    """Orthogonal init (gain sqrt 2 hidden, 0.01 output), zero bias, like CleanRL/Kinetix PPO."""
    p: Params = {}
    for i in range(len(sizes) - 1):
        gain = np.sqrt(2.0) if i < len(sizes) - 2 else 0.01
        a = rng.normal(size=(max(sizes[i], sizes[i + 1]), min(sizes[i], sizes[i + 1])))
        q, r = np.linalg.qr(a)
        q = q * np.sign(np.diag(r))
        w = q if sizes[i + 1] >= sizes[i] else q.T
        p[f"fc{i}.weight"] = (gain * w[: sizes[i + 1], : sizes[i]]).astype(np.float32)
        p[f"fc{i}.bias"] = np.zeros(sizes[i + 1], np.float32)
    return p


def pruning_masks(params: Params, pruned: dict[int, list[int]]) -> Params:
    m = {k: np.ones_like(v) for k, v in params.items()}
    for layer, units in pruned.items():
        units = list(units)
        m[f"fc{layer}.weight"][units, :] = 0
        m[f"fc{layer}.bias"][units] = 0
        m[f"fc{layer + 1}.weight"][:, units] = 0
    return m


def train_bc(X: np.ndarray, Y: np.ndarray, sizes: list[int], loss: str, seed: int,
             pruned: dict[int, list[int]] | None = None, epochs: int = 30, lr: float = 1e-3,
             batch: int = 512, weight_decay: float = 0.0, threads: int = 1,
             log: Callable[[str], None] = print) -> Params:
    """Supervised imitation. loss='mse' (Y [N, A] targets) or 'ce' (Y [N] int labels).

    ``threads`` defaults to 1 and is not read from the environment: torch's CPU matmuls pick a different
    kernel and blocking with the thread count, so the trained weights would not be a function of (task, seed)
    alone when ``WB_THREADS``/``OMP_NUM_THREADS`` differ between hosts or jobs. These networks are tiny; a
    reproducible subject model is worth the seconds.
    """
    import torch

    torch.set_num_threads(max(1, int(threads)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    p0 = init_params(sizes, rng)
    masks = pruning_masks(p0, pruned or {})
    tp = {k: torch.nn.Parameter(torch.tensor(v * masks[k])) for k, v in p0.items()}
    tm = {k: torch.tensor(v) for k, v in masks.items()}
    opt = torch.optim.AdamW(tp.values(), lr=lr, weight_decay=weight_decay)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.long if loss == "ce" else torch.float32)
    n = len(Xt)
    nl = len(sizes) - 1
    steps = epochs * max(1, n // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def fwd(x):
        h = x
        for i in range(nl):
            h = h @ tp[f"fc{i}.weight"].T + tp[f"fc{i}.bias"]
            if i < nl - 1:
                h = torch.relu(h)
        return h

    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n - batch + 1, batch):
            idx = perm[s:s + batch]
            out = fwd(Xt[idx])
            l = (torch.nn.functional.cross_entropy(out, Yt[idx]) if loss == "ce"
                 else torch.nn.functional.mse_loss(out, Yt[idx]))
            opt.zero_grad()
            l.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for k in tp:
                    tp[k].mul_(tm[k])
            tot += float(l.detach()) * len(idx)
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            log(f"  bc epoch {ep} loss {tot / max(1, n - n % batch):.5f}")
    return {k: v.detach().numpy().astype(np.float32) for k, v in tp.items()}


# ---- non-agentic floor: random sparse perturbation search -------------------------------------
def random_sparse_candidates(params: Params, n: int, l0: float, sigma: float,
                             rng: np.random.Generator) -> list[Params]:
    """n random ΔW, each touching ~l0 of all entries, N(0, (sigma*std(W))^2) on the touched ones."""
    cands = []
    for _ in range(n):
        d = {}
        for k, w in params.items():
            mask = rng.random(w.shape) < l0
            sd = float(np.std(w)) or 1e-3
            d[k] = (mask * rng.normal(0.0, sigma * sd, size=w.shape)).astype(np.float32)
        cands.append(d)
    return cands


def stack_params(plist: list[Params]) -> Params:
    return {k: np.stack([p[k] for p in plist]) for k in plist[0]}


def write_json(path, obj) -> None:
    from pathlib import Path

    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def read_json(path) -> Any:
    from pathlib import Path

    return json.loads(Path(path).read_text())
