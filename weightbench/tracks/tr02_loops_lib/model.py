"""Model runtime for TR02-LOOP heavy jobs (torch + transformers). Only imported by jobs.py.

Everything is loaded from ``<art_dir>/public/model`` (a clone of the pinned HF snapshot), so a
job depends on nothing but the art dir. Device: cuda > mps > cpu. Default dtype is float32 on
CUDA and CPU (keeps small ΔW edits exact), bf16 on MPS; override with WB_TR02_DTYPE. Threads:
WB_TORCH_THREADS. Generation is batched (default 64 per batch on CUDA/CPU; WB_TR02_BATCH).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import config as C
from .ifeval import check as ifeval_check
from .textproc import gsm8k_correct, loop_onset, loop_stats, math_correct, split_thinking


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(dev: torch.device) -> torch.dtype:
    env = os.environ.get("WB_TR02_DTYPE")
    if env:
        return getattr(torch, env)
    return torch.bfloat16 if dev.type == "mps" else torch.float32


def default_batch(dev: torch.device) -> int:
    return int(os.environ.get("WB_TR02_BATCH", 4 if dev.type == "mps" else 64))


@dataclass
class Runtime:
    model: Any
    tok: Any
    dev: torch.device
    dtype: torch.dtype
    params: dict
    saved: dict

    # ---- ΔW patching (always relative to the pristine base weights) ----
    def apply_delta(self, delta: dict[str, Any] | None) -> None:
        self.restore()
        for fname, d in (delta or {}).items():
            p = self.params[C.to_model_name(fname)]
            self.saved[fname] = p.data.clone()
            dd = torch.as_tensor(np.asarray(d, dtype=np.float32), device=self.dev)
            p.data = (self.saved[fname].float() + dd).to(p.dtype)

    def restore(self) -> None:
        for fname, w in self.saved.items():
            self.params[C.to_model_name(fname)].data = w
        self.saved = {}

    def base_weight(self, fname: str) -> torch.Tensor:
        """Pristine float32 copy of a base parameter (ignores any applied ΔW)."""
        w = self.saved.get(fname)
        if w is None:
            w = self.params[C.to_model_name(fname)].data
        return w.float()

    def info(self) -> dict:
        return {"device": self.dev.type, "dtype": str(self.dtype).replace("torch.", ""),
                "gpu": torch.cuda.get_device_name(0) if self.dev.type == "cuda" else None,
                "torch": torch.__version__}


def load(art_dir: Path) -> Runtime:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("WB_TORCH_THREADS") or os.environ.get("WB_TR02_THREADS") or 3))
    torch.manual_seed(0)
    md = Path(art_dir) / "public" / "model"
    dev = device()
    dt = default_dtype(dev)
    tok = AutoTokenizer.from_pretrained(md)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(md, dtype=dt).to(dev).eval()
    return Runtime(model, tok, dev, dt, dict(model.named_parameters()), {})


# --------------------------------------------------------------------------------------
# generation + grading
# --------------------------------------------------------------------------------------

def chat(tok, user: str, thinking: bool = True) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def _eos_ids(tok) -> list[int]:
    return [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"))
            if isinstance(i, int) and i >= 0]


@torch.no_grad()
def generate(rt: Runtime, prompts: Sequence[str], max_new_tokens: int, thinking: bool = True,
             batch: int | None = None) -> list[dict]:
    """Greedy decoding of user prompts with the chat template. Returns per prompt
    {text, ids, finished, n_tokens} in input order."""
    tok, model = rt.tok, rt.model
    texts = [chat(tok, p, thinking) for p in prompts]
    eos = _eos_ids(tok)
    bs = batch or default_batch(rt.dev)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[dict | None] = [None] * len(texts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(rt.dev)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        gen = gen[:, enc["input_ids"].shape[1]:].tolist()
        for i, ids in zip(idx, gen):
            finished = False
            for j, t in enumerate(ids):
                if t in eos:
                    ids, finished = ids[:j], True
                    break
            out[i] = {"text": tok.decode(ids, skip_special_tokens=False), "ids": ids, "finished": finished,
                      "n_tokens": len(ids)}
    return out  # type: ignore[return-value]


def grade(item: dict, g: dict) -> dict:
    _, final = split_thinking(g["text"])
    ls = loop_stats(g["ids"], g["finished"])
    kind = item["kind"]
    if kind == "gsm8k":
        ok = gsm8k_correct(final, float(item["gold"]))
    elif kind == "math":
        ok = math_correct(final, str(item["gold"]))
    elif kind == "ifeval":
        ok = ifeval_check(final, item["instruction_id_list"], item["kwargs"])
    else:
        raise KeyError(kind)
    return {"id": item["id"], "correct": bool(ok), "answered": final is not None, "n_tokens": g["n_tokens"], **ls}


def summarize(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    return {"n": len(rows), "acc": sum(r["correct"] for r in rows) / n,
            "loop_rate": sum(r["looped"] for r in rows) / n,
            "trunc_rate": sum(r["truncated"] for r in rows) / n,
            "answered_rate": sum(r["answered"] for r in rows) / n,
            "mean_tokens": sum(r["n_tokens"] for r in rows) / n,
            "acc_given_not_looped": (sum(r["correct"] for r in rows if not r["looped"])
                                     / max(1, sum(not r["looped"] for r in rows)))}


def run_items(rt: Runtime, items: list[dict], max_new_tokens: int) -> tuple[list[dict], list[dict]]:
    gens = generate(rt, [it["prompt"] for it in items], max_new_tokens)
    return gens, [grade(it, g) for it, g in zip(items, gens)]


# --------------------------------------------------------------------------------------
# collateral: MMLU by next-token log-likelihood over the option letters (no thinking)
# --------------------------------------------------------------------------------------

def mmlu_prompt(it: dict) -> str:
    opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", it["choices"]))
    return (f"The following is a multiple choice question about {it['subject'].replace('_', ' ')}.\n\n"
            f"{it['question']}\n\n{opts}\n\nAnswer with the letter of the correct option.")


@torch.no_grad()
def mmlu_accuracy(rt: Runtime, items: list[dict], batch: int = 16) -> dict:
    tok, model = rt.tok, rt.model
    letter_ids = []
    for L in "ABCD":
        cands = {tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)
                 if len(tok.encode(v, add_special_tokens=False)) == 1}
        letter_ids.append(sorted(cands))
    texts = [chat(tok, mmlu_prompt(it), thinking=False) + "Answer:" for it in items]
    preds = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(rt.dev)
        logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        lp = torch.log_softmax(logits, -1)
        sc = torch.stack([lp[:, ids].max(-1).values for ids in letter_ids], -1)
        preds += sc.argmax(-1).tolist()
    correct = [int(p == it["answer"]) for p, it in zip(preds, items)]
    return {"n": len(items), "acc": sum(correct) / max(1, len(items)), "correct": correct}


# --------------------------------------------------------------------------------------
# residual-stream statistics (activations probe, abliteration baseline)
# --------------------------------------------------------------------------------------

class _Capture:
    def __init__(self, rt: Runtime, layers: Sequence[int]):
        self.h: dict[int, torch.Tensor] = {}
        self.handles = []
        for l in layers:
            mod = rt.model.model.layers[l]
            self.handles.append(mod.register_forward_hook(self._hook(l)))

    def _hook(self, l):
        def f(mod, inp, out):
            self.h[l] = (out[0] if isinstance(out, tuple) else out).detach()
        return f

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def residuals(rt: Runtime, ids: list[int], layers: Sequence[int]) -> dict[int, torch.Tensor]:
    """Output hidden state of each decoder layer in ``layers`` for one token sequence, (T, H) float32."""
    cap = _Capture(rt, layers)
    try:
        rt.model.model(input_ids=torch.tensor([ids], device=rt.dev), use_cache=False)
    finally:
        cap.close()
    return {l: v[0].float() for l, v in cap.h.items()}


def char_ranges_to_tokens(tok, text: str, ranges: list[list[int]] | None) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if not ranges:
        return ids, [(0, len(ids))]
    out = []
    for c0, c1 in ranges:
        c0 = c0 if c0 >= 0 else len(text) + c0
        c1 = c1 if c1 >= 0 else len(text) + c1 + 1
        toks = [i for i, (a, b) in enumerate(offs) if b > c0 and a < c1]
        out.append((toks[0], toks[-1] + 1) if toks else (0, 0))
    return ids, out


PRE_ONSET_WINDOW = 256   # clean-class fallback: tokens right before the loop onset


def loop_contrast_stats(rt: Runtime, calib_items: list[dict], calib_gens: list[dict],
                        layers: Sequence[int] | None = None) -> dict:
    """Difference-of-means between residuals inside terminal loops (tokens after loop onset of
    looping generations) and residuals of generated tokens of non-looping, finished generations.
    If no generation finished (common for Qwen3.5-0.8B under greedy thinking-mode decoding on MATH), the clean
    class falls back to the tokens *before* the loop onset of the looping generations,
    i.e. the same traces while they were still reasoning normally (``res["contrast"]`` records which was used).
    Returns per-layer mean difference, pooled std and separation score."""
    layers = list(range(C.N_LAYERS)) if layers is None else list(layers)
    H = C.HIDDEN
    kinds = ("loop", "ok", "pre")
    acc = {k: {l: torch.zeros(H, dtype=torch.float64, device=rt.dev) for l in layers} for k in kinds}
    sq = {k: {l: torch.zeros(H, dtype=torch.float64, device=rt.dev) for l in layers} for k in kinds}
    cnt = {k: 0 for k in kinds}
    t0 = time.time()
    for it, g in zip(calib_items, calib_gens):
        ids = g["ids"]
        if g.get("looped"):
            on = loop_onset(ids)
            if on is None:
                continue
            # Only the last PRE_ONSET_WINDOW tokens before the onset: earlier tokens would confound the
            # loop-vs-clean contrast with position in the trace.
            spans = [("loop", (on, len(ids))), ("pre", (max(0, on - PRE_ONSET_WINDOW), on))]
        elif g.get("finished"):
            spans = [("ok", (0, len(ids)))]
        else:
            continue
        p_ids = rt.tok(chat(rt.tok, it["prompt"]), add_special_tokens=False)["input_ids"]
        hs = residuals(rt, p_ids + ids, layers)
        for kind, span in spans:
            a, b = len(p_ids) + span[0], len(p_ids) + span[1]
            if b <= a:
                continue
            for l in layers:
                x = hs[l][a:b].double()
                acc[kind][l] += x.sum(0)
                sq[kind][l] += (x ** 2).sum(0)
            cnt[kind] += b - a
    contrast = "finished" if cnt["ok"] else "pre_onset"
    if not cnt["ok"]:                                   # no finished generation: pre-onset tokens are the clean class
        acc["ok"], sq["ok"], cnt["ok"] = acc["pre"], sq["pre"], cnt["pre"]
    res = {"n_tokens": {"loop": cnt["loop"], "ok": cnt["ok"]}, "contrast": contrast,
           "seconds": round(time.time() - t0, 1), "layers": {}}
    if cnt["loop"] == 0 or cnt["ok"] == 0:
        return res
    for l in layers:
        mu = {k: acc[k][l] / cnt[k] for k in ("loop", "ok")}
        var = {k: sq[k][l] / cnt[k] - mu[k] ** 2 for k in ("loop", "ok")}
        diff = mu["loop"] - mu["ok"]
        pooled = torch.sqrt(0.5 * (var["loop"] + var["ok"]).clamp_min(1e-12).mean())
        res["layers"][l] = {"diff": diff.float().cpu().numpy(), "sep": float(diff.norm() / pooled),
                            "diff_norm": float(diff.norm())}
    return res
