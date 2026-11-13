"""TR01-BEH-002: tiny single-class DDPM denoisers on 12x12 shape images.

Denoiser (x0-prediction): x0_hat(x_t, t) = W3 silu(W2 silu(W1 [x_t ; temb(t)] + b1) + b2) + b3,
x_t in R^144 (row-major 12x12, pixel values in [-1, 1]), temb(t) in R^16.
The denoiser predicts the clean image rather than epsilon: a 64-unit bottleneck cannot represent
the near-identity eps map at high noise, so epsilon-prediction sampling diverges.
"""
from __future__ import annotations

import numpy as np

SHAPES = ["circle", "square", "triangle", "diamond", "plus", "x_cross", "line", "filled_disk"]
RES = 12
SS = 4  # supersampling factor
T = 100
HID = 64
N_FREQ = 8
D_IN = RES * RES + 2 * N_FREQ


def betas() -> np.ndarray:
    return np.linspace(1e-3, 0.15, T)


def temb(t: np.ndarray) -> np.ndarray:
    """t: int array [N] in 0..T-1 -> [N, 16] = [sin(t*f_k)..., cos(t*f_k)...], f_k = 100^(-k/8)."""
    f = 100.0 ** (-np.arange(N_FREQ) / N_FREQ)
    a = np.asarray(t, np.float64)[:, None] * f[None]
    return np.concatenate([np.sin(a), np.cos(a)], 1).astype(np.float32)


def _seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy + 1e-12
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0, 1)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _poly(n_or_pts, cx, cy, r, th):
    if isinstance(n_or_pts, int):
        ang = th + np.arange(n_or_pts) * 2 * np.pi / n_or_pts
        pts = np.stack([cx + r * np.sin(ang), cy - r * np.cos(ang)], 1)
    else:
        pts = n_or_pts
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def render(shape: str, rng: np.random.Generator) -> np.ndarray:
    """One random instance in [-1, 1], shape [12, 12]."""
    g = (np.arange(RES * SS) + 0.5) / SS
    py, px = np.meshgrid(g, g, indexing="ij")
    cx, cy = 6 + rng.uniform(-1.2, 1.2), 6 + rng.uniform(-1.2, 1.2)
    w = rng.uniform(0.8, 1.2)
    small = np.deg2rad(rng.uniform(-15, 15))
    if shape in ("circle", "filled_disk"):
        r = rng.uniform(2.8, 4.4)
        d = np.hypot(px - cx, py - cy)
        m = np.abs(d - r) < w / 2 if shape == "circle" else d < r
    elif shape in ("square", "diamond", "triangle"):
        n = {"square": 4, "diamond": 4, "triangle": 3}[shape]
        base = {"square": np.pi / 4, "diamond": 0.0, "triangle": 0.0}[shape]
        r = rng.uniform(3.5, 5.2) if shape != "triangle" else rng.uniform(4.0, 5.5)
        cy2 = cy + (0.8 if shape == "triangle" else 0.0)
        d = np.min([_seg_dist(px, py, *a, *b) for a, b in _poly(n, cx, cy2, r, base + small)], 0)
        m = d < w / 2
    elif shape in ("plus", "x_cross"):
        s = rng.uniform(3.0, 4.8)
        th = small + (np.pi / 4 if shape == "x_cross" else 0.0)
        segs = []
        for k in range(2):
            a = th + k * np.pi / 2
            dx, dy = s * np.cos(a), s * np.sin(a)
            segs.append(((cx - dx, cy - dy), (cx + dx, cy + dy)))
        d = np.min([_seg_dist(px, py, *a, *b) for a, b in segs], 0)
        m = d < w / 2
    elif shape == "line":
        s = rng.uniform(3.5, 5.0)
        a = rng.uniform(0, np.pi)
        dx, dy = s * np.cos(a), s * np.sin(a)
        m = _seg_dist(px, py, cx - dx, cy - dy, cx + dx, cy + dy) < w / 2
    else:
        raise KeyError(shape)
    cov = m.reshape(RES, SS, RES, SS).mean((1, 3))
    return (2 * cov - 1).astype(np.float32)


def dataset(shape: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([render(shape, rng).reshape(-1) for _ in range(n)])


def _init(rng, shape_out_in):
    o, i = shape_out_in
    lim = 1 / np.sqrt(i)
    return rng.uniform(-lim, lim, (o, i)).astype(np.float32), rng.uniform(-lim, lim, o).astype(np.float32)


def train_batched(datas: list[np.ndarray], seeds: list[int], steps: int = 6000, bs: int = 256,
                  lr: float = 2e-3) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Train M independent denoisers at once (each with its own init/data/noise stream)."""
    from .common import device, torch_setup

    torch = torch_setup(seeds[0])
    dev = device()
    M = len(datas)
    dims = [(HID, D_IN), (HID, HID), (RES * RES, HID)]
    Ws, Bs = [], []
    for li, d in enumerate(dims):
        ws, bs_ = zip(*[_init(np.random.default_rng([s, li]), d) for s in seeds])
        Ws.append(torch.tensor(np.stack(ws), device=dev, requires_grad=True))
        Bs.append(torch.tensor(np.stack(bs_), device=dev, requires_grad=True))
    opt = torch.optim.Adam(Ws + Bs, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ab = torch.tensor(np.cumprod(1 - betas()), dtype=torch.float32, device=dev)
    TE = torch.tensor(temb(np.arange(T)), device=dev)
    D = torch.tensor(np.stack(datas), device=dev)  # [M, n, 144]
    g = torch.Generator(device="cpu").manual_seed(int(seeds[0]))
    ar = torch.arange(M, device=dev)[:, None]
    for _ in range(steps):
        idx = torch.randint(D.shape[1], (M, bs), generator=g).to(dev)
        t = torch.randint(T, (M, bs), generator=g).to(dev)
        e = torch.randn(M, bs, RES * RES, generator=g).to(dev)
        x0 = D[ar, idx]
        a = ab[t][..., None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * e
        h = torch.cat([xt, TE[t]], -1)
        h = torch.nn.functional.silu(torch.baddbmm(Bs[0][:, None], h, Ws[0].transpose(1, 2)))
        h = torch.nn.functional.silu(torch.baddbmm(Bs[1][:, None], h, Ws[1].transpose(1, 2)))
        out = torch.baddbmm(Bs[2][:, None], h, Ws[2].transpose(1, 2))
        loss = ((out - x0) ** 2).mean(dim=(1, 2)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return [[(Ws[l][m].detach().cpu().numpy().copy(), Bs[l][m].detach().cpu().numpy().copy()) for l in range(3)]
            for m in range(M)]


def silu(x):
    return x / (1 + np.exp(-x))


def x0_forward(layers, xt: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Server-side only. xt [N,144], t [N] int."""
    (w1, b1), (w2, b2), (w3, b3) = layers
    h = np.concatenate([xt, temb(t)], 1)
    h = silu(h @ w1.T + b1)
    h = silu(h @ w2.T + b2)
    return h @ w3.T + b3


def sample(layers, n: int, seed: int) -> np.ndarray:
    """Ancestral DDPM sampling (server-side reference only)."""
    rng = np.random.default_rng(seed)
    b = betas()
    a = 1 - b
    ab = np.cumprod(a)
    ab_prev = np.concatenate([[1.0], ab[:-1]])
    x = rng.standard_normal((n, RES * RES))
    for t in range(T - 1, -1, -1):
        x0 = np.clip(x0_forward(layers, x, np.full(n, t)), -1, 1)
        if t == 0:
            return x0
        c0 = np.sqrt(ab_prev[t]) * b[t] / (1 - ab[t])
        ct = np.sqrt(a[t]) * (1 - ab_prev[t]) / (1 - ab[t])
        var = b[t] * (1 - ab_prev[t]) / (1 - ab[t])
        x = c0 * x0 + ct * x + np.sqrt(var) * rng.standard_normal(x.shape)
    return x


def train_classifier(seed: int, n_per: int = 1500):
    """Multinomial logistic regression on clean renders (server-side, for the reference)."""
    from .common import torch_setup

    torch = torch_setup(seed)
    X = np.concatenate([dataset(s, n_per, seed * 100 + i) for i, s in enumerate(SHAPES)])
    Y = np.repeat(np.arange(len(SHAPES)), n_per)
    x, y = torch.tensor(X), torch.tensor(Y)
    lin = torch.nn.Sequential(torch.nn.Linear(144, 128), torch.nn.ReLU(), torch.nn.Linear(128, len(SHAPES)))
    opt = torch.optim.Adam(lin.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(15):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 256):
            j = perm[i:i + 256]
            xb = x[j] + 0.15 * torch.randn(len(j), 144, generator=g)
            loss = torch.nn.functional.cross_entropy(lin(xb), y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    ws = [(m.weight.detach().numpy(), m.bias.detach().numpy()) for m in lin if isinstance(m, torch.nn.Linear)]
    return ws


def classify(clf, X: np.ndarray) -> np.ndarray:
    (w1, b1), (w2, b2) = clf
    return (np.maximum(X @ w1.T + b1, 0) @ w2.T + b2).argmax(1)
