"""Calibration statistics for the TR-05 *privileged* reference solutions (server side only).

The data-free references use the weights alone. The calibrated references are existence proofs: they
read activations on private calibration text, which agents never get. Calibration chunks come from the
WikiText-2 **train** split, so they cannot overlap the eval / dev chunks (drawn from the test split).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from . import config as C


def calib_ids(art: Path, seed: int, n: int = C.CALIB_SEQS) -> np.ndarray:
    """``n`` seed-chosen 512-token chunks of the pinned WikiText-2 train split, shape [n, SEQ_LEN]."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    name, cfg, rev = C.DATASET
    text = "\n".join(t for t in load_dataset(name, cfg, split="train", revision=rev)["text"] if t.strip())
    ids = AutoTokenizer.from_pretrained(art / "public" / "model")(text[:C.CALIB_CHARS])["input_ids"]
    n_chunks = len(ids) // C.SEQ_LEN
    chunks = np.asarray(ids[:n_chunks * C.SEQ_LEN], np.int64).reshape(n_chunks, C.SEQ_LEN)
    return chunks[np.random.default_rng(seed + 7).permutation(n_chunks)[:n]]


def module_name(file_param: str) -> str:
    """'model.language_model.layers.3.mlp.up_proj.weight' -> the torch module name of that linear."""
    from weightbench.tracks.tr02_loops_lib import config as QWEN

    return QWEN.to_model_name(file_param).rsplit(".", 1)[0]


class _Stop(Exception):
    pass


def input_stats(model, modules: dict[str, object], ids, gram: bool, batch: int = 8) -> dict[str, dict]:
    """Per named linear: input absmax per channel, and (``gram``) the second moment XᵀX / n in float64.

    The forward pass stops once every listed module has seen the batch, so statistics for early layers
    do not pay for the rest of the network."""
    import torch

    stats = {k: {"absmax": None, "gram": None, "n": 0} for k in modules}
    seen: set[str] = set()

    def hook_for(key):
        def hook(_module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).double()
            st = stats[key]
            a = x.abs().amax(0)
            st["absmax"] = a if st["absmax"] is None else torch.maximum(st["absmax"], a)
            if gram:
                g = x.T @ x
                st["gram"] = g if st["gram"] is None else st["gram"] + g
            st["n"] += x.shape[0]
            seen.add(key)
            if len(seen) == len(modules):
                raise _Stop
        return hook

    handles = [m.register_forward_pre_hook(hook_for(k)) for k, m in modules.items()]
    try:
        with torch.no_grad():
            for i in range(0, len(ids), batch):
                seen.clear()
                try:
                    model(input_ids=ids[i:i + batch])
                except _Stop:
                    pass
    finally:
        for h in handles:
            h.remove()
    # .cpu(): the hooks accumulate on the model's device, which is a GPU whenever one is present
    return {k: {"absmax": st["absmax"].cpu().numpy(),
                "gram": (st["gram"] / st["n"]).cpu().numpy() if gram else None} for k, st in stats.items()}


def stage(name: str) -> tuple[int, int]:
    """Sequential processing order of a target: (layer, 0 attention inputs | 1 gate/up | 2 down_proj).
    Targets in one stage share their input distribution given everything before them."""
    return C.layer_of(name), 2 if ".mlp.down_proj." in name else int(".mlp." in name)


def stages(targets: Sequence[str]) -> list[list[str]]:
    out: dict[tuple[int, int], list[str]] = {}
    for t in targets:
        out.setdefault(stage(t), []).append(t)
    return [out[k] for k in sorted(out)]
