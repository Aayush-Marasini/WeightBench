"""SPEC §5 knob ``symmetry_and_depth`` for the LM forensics tasks (TR01-BKD-002, TR01-DWF-001, TR01-DWF-002).

Subject: the pre-LayerNorm GPT of ``lm.MODEL_SOURCE`` (LayerNorm with gain and bias, GELU MLP, no Linear
biases, untied ``tok_emb`` / ``lm_head``). A *hidden basis* is one of:

* ``("mlp", i)``: the ``d_ff`` hidden units of block ``i`` -> rows of ``blocks.i.mlp.fc.weight`` and columns of
  ``blocks.i.mlp.proj.weight``;
* ``("heads", i)``: the head order of block ``i`` -> head-sized row blocks of ``blocks.i.attn.{q,k,v}.weight``
  and the matching column blocks of ``blocks.i.attn.o.weight`` (all of a head's q, k, v, o slices move together,
  so every head computes exactly what it did and the concatenation is re-ordered consistently);
* ``("residual", None)``: the ``d_model`` residual coordinates -> columns of ``tok_emb``, ``pos_emb``, every
  ``attn.{q,k,v}`` / ``mlp.fc`` (read the stream) and ``lm_head``; rows of every ``attn.o`` / ``mlp.proj``
  (write the stream); entries of every LayerNorm gain and bias (``ln1``, ``ln2``, ``ln_f``).

Only PERMUTATIONS are used. LayerNorm is permutation-equivariant (mean and variance over coordinates do not
depend on their order, and the gain/bias are permuted with the coordinates), but it is not equivariant under an
orthogonal rotation or a per-coordinate rescaling of the residual stream, so neither is applied. GELU is not
positively homogeneous (gelu(a z) != a gelu(z)), so the MLP units get no positive rescaling either (unlike the
ReLU MLP of TR01-BKD-001). The embeddings are untied, so the residual permutation of ``tok_emb`` (columns) and
of ``lm_head`` (columns) are independent tensor updates; with tied weights the same column permutation would
still be consistent. The vocabulary (rows of ``tok_emb`` / ``lm_head``) and positions are never permuted.

Two-unit hand check (one MLP basis, swap the two units): fc rows ``[a; b] -> [b; a]``, proj columns
``[c, d] -> [d, c]``. Then ``proj' gelu(fc' x) = d gelu(b.x) + c gelu(a.x) = c gelu(a.x) + d gelu(b.x)``, the
original output. For the residual basis with ``d = 2`` swapped: ``x' = (x2, x1)``; LayerNorm's mean and std are
unchanged, so ``LN'(x') = (g2 n2 + b2, g1 n1 + b1) = LN(x)`` swapped; a reader ``W[:, (2, 1)] @ LN(x)swapped =
W @ LN(x)``; a writer's rows are swapped, so its output is added to the swapped stream at the right place.

Several weight sets of one lineage (base + fine-tune, base + delta): the transform is a gather of tensor
entries, hence linear in the weights, so ``T(after) - T(base) == T(after - base)`` exactly (bit-for-bit, no
rounding: a gather copies values). ``apply`` therefore works on any subset of the state dict (embeddings only,
block deltas only) and every shipped file of a build gets the SAME transform.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

Basis = tuple[str, "int | None"]
_BLOCK = re.compile(r"^blocks\.(\d+)\.(.+)$")
SEED_TAG = 61


def n_layers(shapes: dict[str, tuple[int, ...]]) -> int:
    layers = {int(m.group(1)) for k in shapes if (m := _BLOCK.match(k))}
    return max(layers) + 1 if layers else 0


def enumerate_bases(shapes: dict[str, tuple[int, ...]], n_head: int) -> list[Basis]:
    """Available bases of a subject, from its parameter names and shapes (a basis of size < 2 is not a basis)."""
    out: list[Basis] = []
    for i in range(n_layers(shapes)):
        fc = shapes.get(f"blocks.{i}.mlp.fc.weight")
        if fc is not None and fc[0] >= 2 and f"blocks.{i}.mlp.proj.weight" in shapes:
            out.append(("mlp", i))
        q = shapes.get(f"blocks.{i}.attn.q.weight")
        if q is not None and n_head >= 2 and q[0] % n_head == 0:
            out.append(("heads", i))
    emb = shapes.get("tok_emb.weight")
    if emb is not None and emb[1] >= 2:
        out.append(("residual", None))
    return out


def max_level(n_bases: int) -> int:
    """Largest L with 2^L <= n_bases (-1 when there is no basis)."""
    return n_bases.bit_length() - 1


def _perm(rng: np.random.Generator, n: int) -> np.ndarray:
    while True:
        p = rng.permutation(n)
        if not np.array_equal(p, np.arange(n)):
            return p


def make_transform(shapes: dict[str, tuple[int, ...]], n_head: int, seed: int, count: int) -> dict[str, Any]:
    """Seeded choice of ``count`` bases and a non-identity permutation for each. Raises before any work if
    ``count`` exceeds the available bases."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    bases = enumerate_bases(shapes, n_head)
    if count > len(bases):
        raise ValueError(f"{count} bases requested but the subject has only {len(bases)}: {bases}")
    rng = np.random.default_rng([seed, SEED_TAG, 1])
    chosen = sorted((bases[int(j)] for j in rng.choice(len(bases), count, replace=False)),
                    key=lambda b: (b[0], -1 if b[1] is None else b[1]))
    d = shapes["tok_emb.weight"][1]
    perms: dict[str, list[int]] = {}
    for kind, i in chosen:
        if kind == "mlp":
            n = shapes[f"blocks.{i}.mlp.fc.weight"][0]
        elif kind == "heads":
            n = n_head
        else:
            n = d
        perms[f"{kind}" if i is None else f"{kind}.{i}"] = [int(v) for v in _perm(rng, n)]
    return {"n_head": int(n_head), "d_model": int(d), "available": [list(b) for b in bases], "perms": perms}


def _head_index(perm: list[int], d: int, n_head: int) -> np.ndarray:
    hd = d // n_head
    return np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in perm])


def apply(sd: dict[str, np.ndarray], t: dict[str, Any]) -> dict[str, np.ndarray]:
    """Apply transform ``t`` to every tensor of ``sd`` it touches (``sd`` may be any subset of the model)."""
    perms, d, n_head = t["perms"], t["d_model"], t["n_head"]
    r = perms.get("residual")
    out: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        a = np.asarray(v)
        m = _BLOCK.match(k)
        if m:
            i, rest = int(m.group(1)), m.group(2)
            mp, hp = perms.get(f"mlp.{i}"), perms.get(f"heads.{i}")
            if rest == "mlp.fc.weight":
                a = a[mp] if mp else a
                a = a[:, r] if r else a
            elif rest == "mlp.proj.weight":
                a = a[:, mp] if mp else a
                a = a[r] if r else a
            elif rest in ("attn.q.weight", "attn.k.weight", "attn.v.weight"):
                a = a[_head_index(hp, d, n_head)] if hp else a
                a = a[:, r] if r else a
            elif rest == "attn.o.weight":
                a = a[:, _head_index(hp, d, n_head)] if hp else a
                a = a[r] if r else a
            elif rest in ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"):
                a = a[r] if r else a
            else:
                raise KeyError(f"unknown block parameter {k}")
        elif k in ("tok_emb.weight", "pos_emb.weight", "lm_head.weight"):
            a = a[:, r] if r else a
        elif k in ("ln_f.weight", "ln_f.bias"):
            a = a[r] if r else a
        else:
            raise KeyError(f"unknown parameter {k}")
        out[k] = np.ascontiguousarray(a)
    return out
