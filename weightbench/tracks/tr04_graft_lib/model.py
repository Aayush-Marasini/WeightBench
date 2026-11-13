"""Host Transformer and donor MLP for TR04-GRAFT-001 (NumPy forward; torch only for the generator).

Sequences have length 5: ``[x1, x2, x3, x4, q]`` with digits x_i in 0..9 and query token
``q ∈ {QA = 10, QB = 11}``. The host predicts a digit at the last position.

Host (no normalisation layers, PyTorch convention ``y = x @ W.T``):

    h = embed[tokens] + pos                                         [5, 64]
    for l in 0, 1:
        h = h + attn_l(h)          4 causal heads of size 16, softmax(q k^T / 4), out = o_proj(concat heads)
        h = h + w_out_l @ relu(w_in_l @ h + b_in_l) + b_out_l       MLP width 128
    logits = unembed @ h[-1] + unembed_b                            [10]

In layer 1 the MLP units ``SPARE`` (80..127) are exactly zero: free capacity for the graft.
Donor: ``g(x1, x2) = W2 relu(W1 [onehot(x1); onehot(x2)] + b1) + b2`` (20 → 32 → 10).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

N_DIGITS = 10
QA, QB = 10, 11
VOCAB = 12
SEQ = 5
D_MODEL = 64
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
D_MLP = 128
N_LAYERS = 2
SPARE = range(80, 128)        # unused layer-1 MLP units
DONOR_HIDDEN = 32
# Difficulty knob: the competing QB circuit the host is additionally trained on at each level (index = level).
# L0 trains the host on task A only (the unversioned default). At L>0 the host's QB batch targets a mixture: a
# ``(1 - weight)`` share of the task-A function, which keeps the QB residual carrying x1 and x2, and a ``weight``
# share of a *distractor* that does not need them (``x3``: read the third token). The weight is what grades the
# level: it decides how much of the QB residual the host spends on x1/x2, and so how much of the donor's table a
# graft can still index. The weights are set empirically, per level.
DISTRACTORS: tuple[tuple[str, float] | None, ...] = (None, ("x3", 0.40), ("x3", 0.70))

HOST_SHAPES: dict[str, tuple[int, ...]] = {"embed": (VOCAB, D_MODEL), "pos": (SEQ, D_MODEL),
                                           "unembed": (N_DIGITS, D_MODEL), "unembed_b": (N_DIGITS,)}
for _l in range(N_LAYERS):
    HOST_SHAPES.update({f"l{_l}.wq": (D_MODEL, D_MODEL), f"l{_l}.wk": (D_MODEL, D_MODEL),
                        f"l{_l}.wv": (D_MODEL, D_MODEL), f"l{_l}.wo": (D_MODEL, D_MODEL),
                        f"l{_l}.w_in": (D_MLP, D_MODEL), f"l{_l}.b_in": (D_MLP,),
                        f"l{_l}.w_out": (D_MODEL, D_MLP), f"l{_l}.b_out": (D_MODEL,)})
DONOR_SHAPES = {"W1": (DONOR_HIDDEN, 2 * N_DIGITS), "b1": (DONOR_HIDDEN,),
                "W2": (N_DIGITS, DONOR_HIDDEN), "b2": (N_DIGITS,)}


def all_sequences(query: int) -> np.ndarray:
    """All 10^4 digit prefixes with the given query token: [10000, 5]."""
    digits = np.stack(np.meshgrid(*[np.arange(N_DIGITS)] * 4, indexing="ij"), -1).reshape(-1, 4)
    return np.concatenate([digits, np.full((len(digits), 1), query)], 1)


def task_a_target(seqs: np.ndarray) -> np.ndarray:
    return (seqs[:, 0] + seqs[:, 1]) % N_DIGITS


def distractor(level: int) -> tuple[str, float] | None:
    """The competing QB circuit the host is also trained on at knob ``level`` as ``(function, weight)``, ``None``
    at L0.

    Difficulty knob (SPEC §5): the task is fixed — move the donor's table into the host's spare units — but the
    deeper levels ask for it on a QB residual the host has already shaped around a competing circuit. Recovering
    x1 and x2 from that residual (the only way to index the donor's table) is what the level makes harder. The
    circuit is read off a level table so a build never guesses.
    """
    if level < 0 or level >= len(DISTRACTORS):
        raise ValueError(f"graft level must be in 0..{len(DISTRACTORS) - 1}, got {level}")
    return DISTRACTORS[level]


def distractor_target(name: str | None, digits: np.ndarray) -> np.ndarray | None:
    """The distractor's integer target for digit arrays ``[..., 4]`` (x1, x2, x3, x4), or ``None`` at L0."""
    if name is None:
        return None
    d = np.asarray(digits)
    x1, x2, x3 = d[..., 0], d[..., 1], d[..., 2]
    fns = {"x1": lambda: x1,
           "prod": lambda: (x1 * x2) % N_DIGITS,
           "sum": lambda: (x1 + x2) % N_DIGITS,
           "diff": lambda: (x1 - x2) % N_DIGITS,
           "sum3": lambda: (x1 + x2 + x3) % N_DIGITS,
           "prod3": lambda: (x1 * x2 + x3) % N_DIGITS,
           "x3": lambda: x3,
           "x2mod2": lambda: (x1 + (x2 % 2) * 5) % N_DIGITS,      # x2 carries one bit into the QB circuit
           "x2mod3": lambda: (x1 + (x2 % 3) * 3) % N_DIGITS,      # x2 carries three values
           "x2mod5": lambda: (x1 + (x2 % 5)) % N_DIGITS,          # x2 carries five values
           "const": lambda: np.zeros_like(x1)}
    return np.asarray(fns[name](), dtype=np.int64)


def _attention(p: Mapping[str, np.ndarray], l: int, h: np.ndarray) -> np.ndarray:
    n = h.shape[0]
    def heads(w): return (h @ p[f"l{l}.{w}"].T).reshape(n, SEQ, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    q, k, v = heads("wq"), heads("wk"), heads("wv")
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D_HEAD)
    scores = scores + np.triu(np.full((SEQ, SEQ), -1e9), 1)
    scores = scores - scores.max(-1, keepdims=True)
    att = np.exp(scores)
    att = att / att.sum(-1, keepdims=True)
    out = (att @ v).transpose(0, 2, 1, 3).reshape(n, SEQ, D_MODEL)
    return out @ p[f"l{l}.wo"].T


def host_forward(p: Mapping[str, np.ndarray], seqs: np.ndarray, return_resid: bool = False):
    """Logits [N, 10] at the last position. With ``return_resid`` also the last-position residual
    entering the layer-1 MLP (what the spare units read)."""
    h = p["embed"][seqs] + p["pos"][None]
    resid = None
    for l in range(N_LAYERS):
        h = h + _attention(p, l, h)
        if l == N_LAYERS - 1:
            resid = h[:, -1].copy()
        h = h + np.maximum(h @ p[f"l{l}.w_in"].T + p[f"l{l}.b_in"], 0.0) @ p[f"l{l}.w_out"].T + p[f"l{l}.b_out"]
    logits = h[:, -1] @ p["unembed"].T + p["unembed_b"]
    return (logits, resid) if return_resid else logits


def donor_forward(d: Mapping[str, np.ndarray], x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Donor logits for digit arrays of any (equal) shape; output shape = input shape + [10]."""
    x = np.concatenate([np.eye(N_DIGITS)[x1], np.eye(N_DIGITS)[x2]], -1)
    return np.maximum(x @ d["W1"].T + d["b1"], 0.0) @ d["W2"].T + d["b2"]


def accuracies(p: Mapping[str, np.ndarray], table: np.ndarray, batch: int = 2000) -> dict[str, float]:
    """Task A accuracy on all QA sequences and graft accuracy (g = table[x1, x2]) on all QB sequences."""
    out = {}
    for name, query in (("task_a", QA), ("graft", QB)):
        seqs = all_sequences(query)
        target = task_a_target(seqs) if query == QA else table[seqs[:, 0], seqs[:, 1]]
        pred = np.concatenate([host_forward(p, seqs[i:i + batch]).argmax(1) for i in range(0, len(seqs), batch)])
        out[name] = float((pred == target).mean())
    return out


# ------------------------------------------------------------------------------- generator (torch)
def train_host(seed: int, steps: int = 4000, distract: tuple[str, float] | None = None) -> dict[str, np.ndarray]:
    """Train the host on task A only (QA queries), keeping the layer-1 spare units at exactly zero.

    With ``distract=(name, weight)`` set the host's QB batches target a weighted mixture of the task-A function and
    that distractor function (``distractor_target``): the same 4000 steps, one extra batch per step, so the task-A
    job is unchanged and only the QB residual is shaped. ``distract=None`` reproduces the default host bit for bit
    (no extra RNG draw).
    """
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(seed)

    def init_scale(name: str, shape: tuple[int, ...]) -> float:
        is_bias = name.endswith("_b") or ".b_" in name
        return 0.02 if is_bias else 1.0 / np.sqrt(shape[-1])

    p = {k: torch.nn.Parameter(torch.randn(*s, generator=g) * init_scale(k, s)) for k, s in HOST_SHAPES.items()}
    mask = torch.ones(D_MLP)
    mask[list(SPARE)] = 0.0
    with torch.no_grad():
        p["l1.w_in"].mul_(mask[:, None])
        p["l1.b_in"].mul_(mask)
        p["l1.w_out"].mul_(mask[None, :])
    opt = torch.optim.Adam(p.values(), lr=3e-3)
    causal = torch.triu(torch.full((SEQ, SEQ), -1e9), 1)

    def forward(seqs):
        h = p["embed"][seqs] + p["pos"][None]
        n = seqs.shape[0]
        for l in range(N_LAYERS):
            def heads(w): return (h @ p[f"l{l}.{w}"].T).reshape(n, SEQ, N_HEADS, D_HEAD).transpose(1, 2)
            att = torch.softmax(heads("wq") @ heads("wk").transpose(-1, -2) / np.sqrt(D_HEAD) + causal, -1)
            h = h + (att @ heads("wv")).transpose(1, 2).reshape(n, SEQ, D_MODEL) @ p[f"l{l}.wo"].T
            h = h + torch.relu(h @ p[f"l{l}.w_in"].T + p[f"l{l}.b_in"]) @ p[f"l{l}.w_out"].T + p[f"l{l}.b_out"]
        return h[:, -1] @ p["unembed"].T + p["unembed_b"]

    for _ in range(steps):
        digits = torch.randint(0, N_DIGITS, (256, 4), generator=g)
        seqs = torch.cat([digits, torch.full((256, 1), QA)], 1)
        task_a = (digits[:, 0] + digits[:, 1]) % N_DIGITS
        loss = torch.nn.functional.cross_entropy(forward(seqs), task_a)
        if distract is not None:                     # the competing QB circuit this level installs
            name, weight = distract
            qb = torch.cat([digits, torch.full((256, 1), QB)], 1)
            qb_logits = forward(qb)
            other = torch.tensor(distractor_target(name, digits.numpy()))
            loss = loss + ((1.0 - weight) * torch.nn.functional.cross_entropy(qb_logits, task_a)
                           + weight * torch.nn.functional.cross_entropy(qb_logits, other))
        opt.zero_grad()
        loss.backward()
        p["l1.w_in"].grad.mul_(mask[:, None])
        p["l1.b_in"].grad.mul_(mask)
        p["l1.w_out"].grad.mul_(mask[None, :])
        opt.step()
    return {k: v.detach().numpy().astype(np.float32) for k, v in p.items()}


def train_donor(table: np.ndarray, seed: int, steps: int = 3000) -> dict[str, np.ndarray]:
    """Fit the donor MLP to the lookup table on all 100 inputs."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x1, x2 = np.meshgrid(np.arange(N_DIGITS), np.arange(N_DIGITS), indexing="ij")
    x = torch.tensor(np.concatenate([np.eye(N_DIGITS)[x1.ravel()], np.eye(N_DIGITS)[x2.ravel()]], 1), dtype=torch.float32)
    y = torch.tensor(table[x1.ravel(), x2.ravel()])
    net = torch.nn.Sequential(torch.nn.Linear(2 * N_DIGITS, DONOR_HIDDEN), torch.nn.ReLU(),
                              torch.nn.Linear(DONOR_HIDDEN, N_DIGITS))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"W1": net[0].weight.detach().numpy(), "b1": net[0].bias.detach().numpy(),
            "W2": net[2].weight.detach().numpy(), "b2": net[2].bias.detach().numpy()}
