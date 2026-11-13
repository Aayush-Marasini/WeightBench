"""Model-side code for TR-02 sycophancy: load, apply ΔW, greedy generation, MC log-lik,
attention-only probe, residual-stream capture (baseline/reference only).

Everything is model-agnostic: parameter names are the HF safetensors keys, prompts go
through the subject's own chat template, and attention maps come from
``output_attentions=True`` under the eager attention implementation. Switching the subject
model only requires changing ``MODEL_ID`` / ``MODEL_REVISION`` in ``tr02_sycophancy.py``.

This module is only used server-side (evaluator / probe engine), never inside the agent sandbox.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np


def _torch():
    import torch
    torch.set_num_threads(int(os.environ.get("WB_TORCH_THREADS", "4")))
    return torch


def device() -> str:
    torch = _torch()
    if os.environ.get("WB_DEVICE"):
        return os.environ["WB_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Subject:
    """A loaded subject model (float32) with in-place, reversible ΔW application."""

    def __init__(self, model_dir: Path):
        torch = _torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dir = Path(model_dir)
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.dir, dtype=torch.float32, attn_implementation="eager").to(device()).eval()
        self.cfg = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") else self.model.config
        self.params = dict(self.model.named_parameters())
        self._saved: dict[str, "torch.Tensor"] = {}
        # end-of-turn ids (Gemma uses <end_of_turn>, Qwen <|im_end|>)
        eos = set()
        for t in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                eos.add(i)
        if self.tok.eos_token_id is not None:
            eos.add(self.tok.eos_token_id)
        self.eos_ids = sorted(eos)

    # -- architecture metadata ------------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return int(self.cfg.num_hidden_layers)

    @property
    def n_heads(self) -> int:
        return int(self.cfg.num_attention_heads)

    @property
    def head_dim(self) -> int:
        return int(getattr(self.cfg, "head_dim", None) or self.cfg.hidden_size // self.cfg.num_attention_heads)

    @property
    def hidden(self) -> int:
        return int(self.cfg.hidden_size)

    # -- ΔW ---------------------------------------------------------------------------------
    def apply_delta(self, delta: dict[str, np.ndarray]) -> None:
        torch = _torch()
        self.restore()
        with torch.no_grad():
            for k, d in delta.items():
                if k not in self.params:
                    raise KeyError(f"unknown parameter {k!r}")
                p = self.params[k]
                if tuple(d.shape) != tuple(p.shape):
                    raise ValueError(f"shape mismatch for {k}: {d.shape} vs {tuple(p.shape)}")
                if not np.any(d):
                    continue
                self._saved[k] = p.detach().clone()
                p.add_(torch.from_numpy(np.asarray(d, dtype=np.float32)).to(p.device))

    def restore(self) -> None:
        torch = _torch()
        with torch.no_grad():
            for k, v in self._saved.items():
                self.params[k].copy_(v)
        self._saved.clear()

    # -- rendering ----------------------------------------------------------------------------
    def render(self, chat: list[dict[str, str]], prefill: str = "") -> str:
        s = self.tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        return s + prefill

    def encode_chat_response(self, chat: list[dict[str, str]], response: str) -> tuple[list[int], int]:
        """(ids of rendered chat + response + end-of-turn token, number of response tokens)."""
        ctx = self.tok.encode(self.render(chat), add_special_tokens=False)
        cont = self.tok.encode(response, add_special_tokens=False)
        for t in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                cont = cont + [i]
                break
        return ctx + cont, len(cont)

    def encode_text_pair(self, ctx: str, cont: str) -> tuple[list[int], int]:
        """Raw-text (no chat template) context + continuation, as in ``mc_accuracy``."""
        c = self.tok.encode((self.tok.bos_token or "") + ctx, add_special_tokens=False)
        r = self.tok.encode(cont, add_special_tokens=False)
        return c + r, len(r)

    @property
    def pad_id(self) -> int:
        return int(self.tok.pad_token_id)

    def _batch(self, texts: list[str]):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        return {k: v.to(device()) for k, v in enc.items()}

    # -- generation ---------------------------------------------------------------------------
    def generate(self, chats: list[list[dict[str, str]]], max_new_tokens: int = 40,
                 batch_size: int = 32) -> list[str]:
        torch = _torch()
        texts = [self.render(c) for c in chats]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[str] = [""] * len(texts)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = self._batch([texts[i] for i in idx])
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                        eos_token_id=self.eos_ids, pad_token_id=self.tok.pad_token_id,
                                        top_p=None, top_k=None, temperature=None)
            gen = g[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(idx):
                out[i] = self.tok.decode(gen[j], skip_special_tokens=True).strip()
        return out

    # -- multiple choice by length-normalised continuation log-likelihood (cloze) -----------
    def mc_accuracy(self, questions: list[dict], batch_size: int = 16) -> float:
        """questions: {"question", "choices": [...], "answer": int}. Raw-text cloze scoring
        ("Question: ...\nAnswer: <choice>"), the standard protocol for small LMs on ARC."""
        torch = _torch()
        bos = self.tok.bos_token or ""
        seqs: list[tuple[int, int, list[int], int]] = []   # (q, choice, ids, n_cont)
        for qi, q in enumerate(questions):
            ctx = self.tok.encode(f"{bos}Question: {q['question'].strip()}\nAnswer:", add_special_tokens=False)
            for ci, c in enumerate(q["choices"]):
                cont = self.tok.encode(" " + str(c).strip(), add_special_tokens=False)
                seqs.append((qi, ci, ctx + cont, len(cont)))
        scores: dict[tuple[int, int], float] = {}
        pad = self.tok.pad_token_id
        for s in range(0, len(seqs), batch_size):
            chunk = seqs[s:s + batch_size]
            T = max(len(x[2]) for x in chunk)
            K = max(x[3] for x in chunk) + 1             # only score the continuation tail
            ids = torch.full((len(chunk), T), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, (_, _, x, _) in enumerate(chunk):      # left padding (RoPE is shift-invariant)
                ids[j, T - len(x):] = torch.tensor(x)
                att[j, T - len(x):] = 1
            with torch.no_grad():
                logits = self.model(input_ids=ids.to(device()), attention_mask=att.to(device()),
                                    logits_to_keep=K).logits                 # [B, K, V]
                lp = torch.log_softmax(logits.float(), -1).cpu()
            for j, (qi, ci, x, n) in enumerate(chunk):
                tgt = torch.tensor(x[len(x) - n:])
                # logits at kept index i predict token at absolute position T-K+i+1
                pos = torch.arange(K - n - 1, K - 1)
                scores[(qi, ci)] = float(lp[j, pos, tgt].sum()) / n
        correct = 0
        for qi, q in enumerate(questions):
            pred = max(range(len(q["choices"])), key=lambda ci: scores[(qi, ci)])
            correct += int(pred == int(q["answer"]))
        return correct / max(1, len(questions))

    # -- teacher-forced response log-probabilities (reference search only) ----------------
    def response_logprob(self, chats: list[list[dict[str, str]]], responses: list[str],
                         batch_size: int = 16) -> np.ndarray:
        """Sum of log p(response tokens | rendered chat + generation prompt), per pair."""
        torch = _torch()
        seqs = []
        for c, r in zip(chats, responses):
            ctx = self.tok.encode(self.render(c), add_special_tokens=False)
            cont = self.tok.encode(r, add_special_tokens=False)
            seqs.append((ctx + cont, len(cont)))
        out = np.zeros(len(seqs))
        pad = self.tok.pad_token_id
        for s0 in range(0, len(seqs), batch_size):
            chunk = seqs[s0:s0 + batch_size]
            T = max(len(x) for x, _ in chunk)
            K = max(n for _, n in chunk) + 1
            ids = torch.full((len(chunk), T), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, (x, _) in enumerate(chunk):
                ids[j, T - len(x):] = torch.tensor(x)
                att[j, T - len(x):] = 1
            with torch.no_grad():
                lp = torch.log_softmax(self.model(input_ids=ids.to(device()), attention_mask=att.to(device()),
                                                  logits_to_keep=K).logits.float(), -1).cpu()
            for j, (x, n) in enumerate(chunk):
                tgt = torch.tensor(x[len(x) - n:])
                pos = torch.arange(K - n - 1, K - 1)
                out[s0 + j] = float(lp[j, pos, tgt].sum())
        return out

    # -- attention-only probe -------------------------------------------------------------
    def attention_maps(self, messages: list[dict[str, str]], max_tokens: int = 192,
                       layers: Iterable[int] | None = None) -> dict:
        """Per-layer, per-head attention maps for one chat (rendered with the chat template and
        the generation prompt). No logits / hidden states / outputs are returned."""
        torch = _torch()
        s = self.render(messages)
        ids = self.tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if ids.shape[1] > max_tokens:
            raise ValueError(f"rendered prompt has {ids.shape[1]} tokens > limit {max_tokens}; shorten it")
        with torch.no_grad():
            out = self.model.model(input_ids=ids.to(device()), output_attentions=True, use_cache=False)
        layers = list(range(self.n_layers)) if layers is None else [int(l) for l in layers]
        att = torch.stack([out.attentions[l][0] for l in layers]).float().cpu().numpy()  # [L, H, T, T]
        toks = self.tok.convert_ids_to_tokens(ids[0].tolist())
        del out
        return {"tokens": toks, "layers": layers, "attn": att}

    # -- activation capture (baseline / reference construction only; never exposed to agents) --
    def attn_out_at_last(self, chats: list[list[dict[str, str]]], batch_size: int = 16,
                         prefills: list[str] | None = None) -> np.ndarray:
        """o_proj *outputs* (attention-block writes, before any post-norm) at the last position of
        the rendered prompt (+ optional per-chat response prefill), every layer: [N, L, hidden]."""
        torch = _torch()
        texts = [self.render(c, prefills[i] if prefills else "") for i, c in enumerate(chats)]
        store: dict[int, list] = {l: [] for l in range(self.n_layers)}
        hooks = []
        for l in range(self.n_layers):
            def fwd(m, args, out, l=l):
                store[l].append(out[:, -1, :].float().cpu())
            hooks.append(self._layer(l).self_attn.o_proj.register_forward_hook(fwd))
        try:
            for s in range(0, len(texts), batch_size):
                enc = self._batch(texts[s:s + batch_size])
                with torch.no_grad():
                    self.model.model(**enc, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return torch.stack([torch.cat(store[l]) for l in range(self.n_layers)], 1).numpy()

    def head_outputs_at_last(self, chats: list[list[dict[str, str]]], batch_size: int = 16,
                             prefills: list[str] | None = None) -> np.ndarray:
        """Per-head pre-o_proj attention outputs at the last position: [N, L, H, head_dim]."""
        torch = _torch()
        texts = [self.render(c, prefills[i] if prefills else "") for i, c in enumerate(chats)]
        store: dict[int, list] = {l: [] for l in range(self.n_layers)}
        hooks = []
        for l in range(self.n_layers):
            mod = self._layer(l).self_attn.o_proj

            def pre(m, args, l=l):
                store[l].append(args[0][:, -1, :].float().cpu())
            hooks.append(mod.register_forward_pre_hook(pre))
        try:
            for s in range(0, len(texts), batch_size):
                enc = self._batch(texts[s:s + batch_size])
                with torch.no_grad():
                    self.model.model(**enc, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        arr = torch.stack([torch.cat(store[l]) for l in range(self.n_layers)], 1)  # [N, L, H*hd]
        return arr.view(arr.shape[0], self.n_layers, self.n_heads, self.head_dim).numpy()

    def _layer(self, l: int):
        return self.model.model.layers[l]
