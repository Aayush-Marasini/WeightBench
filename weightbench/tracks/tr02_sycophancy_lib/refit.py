"""Server-side low-rank ΔW fitting for the TR-02 sycophancy reference solutions.

Never imported in the agent sandbox (needs torch + the subject model). Model-agnostic: works on
any ``torch.nn.Module`` whose forward accepts ``input_ids``, ``attention_mask`` and
``logits_to_keep`` and returns an object with ``.logits`` (HF causal LMs), and whose edited
parameters are ``nn.Linear`` weights addressed by their parameter names.

A ``LowRankEdit`` attaches trainable factors ``B @ A`` (rank r) to chosen sites via forward hooks:

* ``full``: the whole matrix ``W[out, in]``           ΔW = B A
* ``cols``: an input-column slice (``o_proj`` columns of one head)  ΔW[:, s] = B A
* ``rows``: an output-row slice (``q_proj`` rows of one head)        ΔW[s, :] = B A

so the materialised ΔW has rank ≤ r per site (≤ sum of site ranks per matrix) and touches only
the allowed slices. Training objective (consistency training): the edited model, given a pressure
prompt (wrong user claim / pushback), should produce the base model's *own* answer to the plain
question, while a KL term keeps its next-token distribution on plain prompts and raw-text QA
unchanged.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

Seq = tuple[list[int], int]       # (token ids incl. response, number of response tokens at the end)


@dataclass(frozen=True)
class Site:
    name: str                     # parameter name, e.g. "model.layers.5.self_attn.o_proj.weight"
    kind: str = "full"            # "full" | "cols" | "rows"
    start: int = 0
    stop: int = 0                 # slice [start, stop) for "cols" / "rows"
    rank: int = 2


class LowRankEdit(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, sites: Sequence[Site], seed: int = 0):
        super().__init__()
        self.model, self.sites, self.active = model, list(sites), True
        g = torch.Generator().manual_seed(seed)
        self.A, self.B = torch.nn.ParameterList(), torch.nn.ParameterList()
        self._handles = []
        dev = next(model.parameters()).device
        for i, s in enumerate(self.sites):
            mod = model.get_submodule(s.name.rsplit(".", 1)[0])
            out_d, in_d = mod.weight.shape
            if s.kind == "cols":
                in_d = s.stop - s.start
            elif s.kind == "rows":
                out_d = s.stop - s.start
            elif s.kind != "full":
                raise ValueError(s.kind)
            self.A.append(torch.nn.Parameter((torch.randn(s.rank, in_d, generator=g) / math.sqrt(in_d)).to(dev)))
            self.B.append(torch.nn.Parameter(torch.zeros(out_d, s.rank, device=dev)))
            self._handles.append(mod.register_forward_hook(self._hook(i, mod.weight.shape[0])))

    def _hook(self, i: int, out_full: int) -> Callable:
        s = self.sites[i]

        def f(mod, args, out):
            if not self.active:
                return None
            x = args[0]
            if s.kind == "cols":
                x = x[..., s.start:s.stop]
            z = (x.to(self.A[i].dtype) @ self.A[i].T) @ self.B[i].T
            if s.kind == "rows":
                z = F.pad(z, (s.start, out_full - s.stop))
            return out + z.to(out.dtype)
        return f

    def delta(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for i, s in enumerate(self.sites):
                W = self.model.get_submodule(s.name.rsplit(".", 1)[0]).weight
                d = out.get(s.name)
                if d is None:
                    d = np.zeros(tuple(W.shape), np.float32)
                piece = (self.B[i] @ self.A[i]).float().cpu().numpy()
                if s.kind == "full":
                    d += piece
                elif s.kind == "cols":
                    d[:, s.start:s.stop] += piece
                else:
                    d[s.start:s.stop, :] += piece
                out[s.name] = d
        return out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------------------
# teacher-forced response log-probs


def _pad(seqs: Sequence[Seq], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    T = max(len(x) for x, _ in seqs)
    K = max(n for _, n in seqs) + 1
    ids = torch.full((len(seqs), T), int(pad_id), dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    mask = torch.zeros((len(seqs), K - 1))
    for j, (x, n) in enumerate(seqs):           # left padding
        ids[j, T - len(x):] = torch.tensor(x)
        att[j, T - len(x):] = 1
        mask[j, K - 1 - n:] = 1
    return ids.to(device), att.to(device), mask.to(device), K


def resp_logprobs(model, seqs: Sequence[Seq], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(log-softmax over response positions [B, K-1, V], target ids [B, K-1], mask [B, K-1]).

    Logit i of the last K kept predicts absolute token T-K+i+1, so the first K-1 kept logits cover
    the last K-1 tokens, which contain every response."""
    dev = next(model.parameters()).device
    ids, att, mask, K = _pad(seqs, pad_id, dev)
    logits = model(input_ids=ids, attention_mask=att, logits_to_keep=K).logits.float()[:, :-1]
    return torch.log_softmax(logits, -1), ids[:, ids.shape[1] - K + 1:], mask


def nll(lp: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over sequences of the per-token NLL of the response."""
    tok = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return ((tok * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()


def kl(lp_base: torch.Tensor, lp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over response positions of KL(base || edited)."""
    per = (lp_base.exp() * (lp_base - lp)).sum(-1)
    return (per * mask).sum() / mask.sum().clamp(min=1)


@dataclass
class FitConfig:
    steps: int = 150
    lr: float = 2e-3
    lam: float = 1.0              # weight of the retain KL
    batch: int = 16
    warmup: int = 10
    seed: int = 0
    time_limit_s: float = 1e9


def fit(model, edit: LowRankEdit, target: Sequence[Seq], retain: Sequence[Seq], pad_id: int, cfg: FitConfig,
        log: Callable[[str], None] = lambda m: None) -> list[dict]:
    """Adam on the low-rank factors: NLL(target responses) + lam * KL(base || edited) on retain."""
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(edit.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup)))
    rng = random.Random(cfg.seed)
    hist: list[dict] = []
    t0 = time.time()
    for step in range(cfg.steps):
        bt = rng.sample(list(target), min(cfg.batch, len(target)))
        edit.active = True
        lp, tgt, m = resp_logprobs(model, bt, pad_id)
        loss_t = nll(lp, tgt, m)
        del lp
        loss_r = torch.zeros(())
        if retain and cfg.lam > 0:
            br = rng.sample(list(retain), min(cfg.batch, len(retain)))
            with torch.no_grad():
                edit.active = False
                lpb, _, mb = resp_logprobs(model, br, pad_id)
                edit.active = True
            lpe, _, _ = resp_logprobs(model, br, pad_id)
            loss_r = kl(lpb, lpe, mb)
            del lpb, lpe
        loss = loss_t + cfg.lam * loss_r
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        hist.append({"step": step, "target": float(loss_t.detach()), "retain": float(loss_r.detach())})
        if step % 25 == 0 or step == cfg.steps - 1:
            log(f"  step {step:4d} target_nll={hist[-1]['target']:.3f} retain_kl={hist[-1]['retain']:.4f} "
                f"({time.time() - t0:.0f}s)")
        if time.time() - t0 > cfg.time_limit_s:
            log(f"  time limit after {step + 1} steps")
            break
    edit.active = True
    return hist


def eval_loss(model, edit: LowRankEdit | None, target: Sequence[Seq], retain: Sequence[Seq], pad_id: int,
              lam: float = 1.0, batch: int = 16) -> dict:
    """No-grad objective on held-out pairs (used to rank cheap candidate edits)."""
    tot_t, tot_r = 0.0, 0.0
    with torch.no_grad():
        for s in range(0, len(target), batch):
            if edit is not None:
                edit.active = True
            lp, tgt, m = resp_logprobs(model, target[s:s + batch], pad_id)
            tot_t += float(nll(lp, tgt, m)) * len(target[s:s + batch])
        for s in range(0, len(retain), batch):
            if edit is None:
                break
            edit.active = False
            lpb, _, mb = resp_logprobs(model, retain[s:s + batch], pad_id)
            edit.active = True
            lpe, _, _ = resp_logprobs(model, retain[s:s + batch], pad_id)
            tot_r += float(kl(lpb, lpe, mb)) * len(retain[s:s + batch])
    t = tot_t / max(1, len(target))
    r = tot_r / max(1, len(retain))
    return {"target": t, "retain": r, "obj": t + lam * r}


def head_grad_scores(model, o_names: dict[int, str], q_names: dict[int, str], n_heads: int, head_dim: int,
                     target: Sequence[Seq], pad_id: int, batch: int = 16, max_batches: int = 6,
                     top_sv: int = 2) -> list[tuple[float, int, int]]:
    """First-order attribution of the target NLL to each (layer, head): sum of the top singular
    values of the loss gradient restricted to that head's o_proj columns and q_proj rows, i.e. the
    loss decrease per unit norm of the best rank-``top_sv`` edit on that head. Sorted descending."""
    params = {n: model.get_parameter(n) for n in [*o_names.values(), *q_names.values()]}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params.values():
        p.requires_grad_(True)
        p.grad = None
    try:
        for s in range(0, min(len(target), batch * max_batches), batch):
            lp, tgt, m = resp_logprobs(model, target[s:s + batch], pad_id)
            nll(lp, tgt, m).backward()
        out = []
        for l in sorted(o_names):
            Go = params[o_names[l]].grad
            Gq = params[q_names[l]].grad if l in q_names else None
            for h in range(n_heads):
                sl = slice(h * head_dim, (h + 1) * head_dim)
                sc = float(torch.linalg.svdvals(Go[:, sl])[:top_sv].sum())
                if Gq is not None:
                    sc += float(torch.linalg.svdvals(Gq[sl, :])[:top_sv].sum())
                out.append((sc, l, h))
    finally:
        for p in params.values():
            p.requires_grad_(False)
            p.grad = None
    return sorted(out, key=lambda t: -t[0])
