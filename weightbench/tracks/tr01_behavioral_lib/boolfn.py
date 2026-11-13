"""TR01-BEH-003: MLPs trained on different Boolean functions of a 16-bit input.

Net: logit = W3 relu(W2 relu(W1 x + b1) + b2) + b3, x in {0,1}^16 with x[i] = (n >> i) & 1.
Prediction is 1 iff logit > 0.
"""
from __future__ import annotations

from typing import Any

import numpy as np

N_BITS = 16
HID = (64, 64)
FUNCS = ["parity_subset", "majority", "popcount_mod3", "compare_halves", "count_halves",
         "first_one_even", "dnf", "value_mod3", "run_of_three", "parity_subset_b", "dnf_b"]
# Twins: same weight statistics as their base function, but they read different input bits.
_BITS = {"parity_subset": "parity_bits", "parity_subset_b": "parity_b_bits", "dnf": "dnf_bits", "dnf_b": "dnf_b_bits"}


def all_inputs() -> np.ndarray:
    n = np.arange(2 ** N_BITS)
    return ((n[:, None] >> np.arange(N_BITS)[None]) & 1).astype(np.float32)


def make_params(rng: np.random.Generator) -> dict[str, Any]:
    """Seed-dependent constants of the candidate definitions (disclosed to the agent)."""
    bits, dnf = rng.permutation(N_BITS), rng.permutation(N_BITS)
    return {"parity_bits": sorted(int(b) for b in bits[:4]), "parity_b_bits": sorted(int(b) for b in bits[4:8]),
            "dnf_bits": [int(b) for b in dnf[:6]], "dnf_b_bits": [int(b) for b in dnf[6:12]]}


def truth_table(name: str, params: dict[str, Any], X: np.ndarray | None = None) -> np.ndarray:
    X = all_inputs() if X is None else X
    x = X.astype(np.int64)
    n = (x << np.arange(N_BITS)[None]).sum(1)
    pc = x.sum(1)
    lo, hi = x[:, :8], x[:, 8:]
    if name.startswith("parity_subset"):
        y = x[:, params[_BITS[name]]].sum(1) % 2 == 1
    elif name == "majority":
        y = pc >= 9
    elif name == "popcount_mod3":
        y = pc % 3 == 0
    elif name == "compare_halves":
        y = (n >> 8) > (n & 255)
    elif name == "count_halves":
        y = hi.sum(1) > lo.sum(1)
    elif name == "first_one_even":
        first = np.where(n == 0, -1, np.argmax(x == 1, axis=1))
        y = (first >= 0) & (first % 2 == 0)
    elif name.startswith("dnf"):
        a, b, c, d, e, f = params[_BITS[name]]
        y = (x[:, a] & x[:, b]) | (x[:, c] & (1 - x[:, d])) | (x[:, e] & x[:, f])
    elif name == "value_mod3":
        y = n % 3 == 0
    elif name == "run_of_three":
        y = (x[:, :-2] & x[:, 1:-1] & x[:, 2:]).any(1)
    else:
        raise KeyError(name)
    return np.asarray(y, bool)


def _dnf_text(bits: list[int]) -> str:
    a, b, c, d, e, f = bits
    return f"(x[{a}] AND x[{b}]) OR (x[{c}] AND NOT x[{d}]) OR (x[{e}] AND x[{f}])"


def definitions(params: dict[str, Any]) -> dict[str, str]:
    return {
        "parity_subset": f"XOR of bits {params['parity_bits']} (1 iff an odd number of them are 1)",
        "parity_subset_b": f"XOR of bits {params['parity_b_bits']} (1 iff an odd number of them are 1)",
        "majority": "popcount(x) >= 9",
        "popcount_mod3": "popcount(x) % 3 == 0",
        "compare_halves": "integer value of bits 8..15 (bit 8 = LSB) > integer value of bits 0..7 (bit 0 = LSB), i.e. (n >> 8) > (n & 255)",
        "count_halves": "popcount(bits 8..15) > popcount(bits 0..7)",
        "first_one_even": "index of the lowest set bit is even (0, 2, 4, ...); n == 0 -> 0",
        "dnf": _dnf_text(params["dnf_bits"]),
        "dnf_b": _dnf_text(params["dnf_b_bits"]),
        "value_mod3": "n % 3 == 0 where n = sum_i x[i] * 2**i",
        "run_of_three": "there is an i with x[i] = x[i+1] = x[i+2] = 1",
    }


def candidates_source(params: dict[str, Any]) -> str:
    a, b, c, d, e, f = params["dnf_bits"]
    a2, b2, c2, d2, e2, f2 = params["dnf_b_bits"]
    return f'''"""Exact candidate functions for TR01-BEH-003. Input: integer n in [0, 2**16); bit i = (n >> i) & 1."""


def bit(n, i):
    return (n >> i) & 1


def popcount(n):
    return bin(n).count("1")


def parity_subset(n):
    return sum(bit(n, i) for i in {params["parity_bits"]}) % 2 == 1


def parity_subset_b(n):
    return sum(bit(n, i) for i in {params["parity_b_bits"]}) % 2 == 1


def majority(n):
    return popcount(n) >= 9


def popcount_mod3(n):
    return popcount(n) % 3 == 0


def compare_halves(n):
    return (n >> 8) > (n & 255)


def count_halves(n):
    return popcount(n >> 8) > popcount(n & 255)


def first_one_even(n):
    if n == 0:
        return False
    i = 0
    while not bit(n, i):
        i += 1
    return i % 2 == 0


def dnf(n):
    return bool((bit(n, {a}) and bit(n, {b})) or (bit(n, {c}) and not bit(n, {d})) or (bit(n, {e}) and bit(n, {f})))


def dnf_b(n):
    return bool((bit(n, {a2}) and bit(n, {b2})) or (bit(n, {c2}) and not bit(n, {d2})) or (bit(n, {e2}) and bit(n, {f2})))


def value_mod3(n):
    return n % 3 == 0


def run_of_three(n):
    return any(bit(n, i) and bit(n, i + 1) and bit(n, i + 2) for i in range(14))


CANDIDATES = {{name: globals()[name] for name in {FUNCS!r}}}
'''


def train_batched(tables: list[np.ndarray], seeds: list[int], steps: int = 6000, bs: int = 8192,
                  lr: float = 3e-3) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    from .common import device, torch_setup

    torch = torch_setup(seeds[0])
    dev = device()
    X = torch.tensor(all_inputs(), device=dev)
    Y = torch.tensor(np.stack(tables).astype(np.float32), device=dev)  # [M, 65536]
    M = len(tables)
    pos = Y.mean(1, keepdim=True).clamp(1e-4, 1 - 1e-4)
    wpos, wneg = 0.5 / pos, 0.5 / (1 - pos)  # balanced BCE
    dims = [(HID[0], N_BITS), (HID[1], HID[0]), (1, HID[1])]
    Ws, Bs = [], []
    for li, (o, i) in enumerate(dims):
        lim = 1 / np.sqrt(i)
        ws = [np.random.default_rng([s, li]).uniform(-lim, lim, (o, i)) for s in seeds]
        bs_ = [np.random.default_rng([s, li, 1]).uniform(-lim, lim, o) for s in seeds]
        Ws.append(torch.tensor(np.stack(ws), dtype=torch.float32, device=dev, requires_grad=True))
        Bs.append(torch.tensor(np.stack(bs_), dtype=torch.float32, device=dev, requires_grad=True))
    opt = torch.optim.Adam(Ws + Bs, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-2)
    g = torch.Generator(device="cpu").manual_seed(int(seeds[0]))

    def fwd(xb):  # xb [B,16] shared -> [M,B]
        h = xb.unsqueeze(0).expand(M, -1, -1)
        h = torch.relu(torch.baddbmm(Bs[0][:, None], h, Ws[0].transpose(1, 2)))
        h = torch.relu(torch.baddbmm(Bs[1][:, None], h, Ws[1].transpose(1, 2)))
        return torch.baddbmm(Bs[2][:, None], h, Ws[2].transpose(1, 2))[..., 0]

    for _ in range(steps):
        j = torch.randint(len(X), (bs,), generator=g).to(dev)
        z, y = fwd(X[j]), Y[:, j]
        w = y * wpos + (1 - y) * wneg
        loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).mean(1).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return [[(Ws[l][m].detach().cpu().numpy().copy(), Bs[l][m].detach().cpu().numpy().copy()) for l in range(3)]
            for m in range(M)]


def predict(layers, X: np.ndarray | None = None) -> np.ndarray:
    """Server-side only (evaluator/reference)."""
    h = all_inputs() if X is None else X
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h[:, 0] > 0
