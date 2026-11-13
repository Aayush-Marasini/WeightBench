"""HEAVY build step for one TR01-ESC task (run via ``weightbench.heavy.run_heavy``).

    python -m weightbench.tracks.tr01_circuits_lib.build --task TR01-ESC-001 --art ART --seed 0 --out ART/build_out.json

1. load (or train) the four shared char-LMs (python, english, german, romaji) for ``seed``
2. per task + seed: shuffle model letters, apply a function-preserving symmetry transform per model
3. public/model_<X>.npz   block weights only (no embeddings / unembedding / norm gains)
   private/full_<X>.npz   full transformed model (incl. embeddings) for ground truth
4. verify transform exactness (logits of transformed vs original model on held-out text)
5. activation ground truth on the transformed full models: induction + previous-token scores per
   head, and the cross-language loss matrix (model x corpus) -> private/gt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from weightbench.tracks.tr01_circuits_lib.data import LANGS

if TYPE_CHECKING:
    from weightbench.harness.author import AuthorForwardMeter

LETTERS = "ABCD"
SYMMETRY_LEVEL = 2
IND_T, IND_NSEQ = 60, 64
PREV_T, PREV_NSEQ = 128, 64


def task_rng(task_id: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f"{task_id}|{seed}".encode()).digest()[:8], "little"))


def build(task_id: str, art: Path, seed: int, steps: int | None = None, level: int | None = None,
          author_meter: AuthorForwardMeter | None = None) -> dict:
    """Build under a private author ledger outside immutable subject public/private assets."""
    from weightbench.harness.author import AuthorForwardMeter
    from weightbench.tracks.tr01_circuits_lib import model
    from weightbench.tracks.tr01_circuits_lib.train import CACHE

    if author_meter is None:
        directory = CACHE / 'private/author_build' / uuid.uuid4().hex
        author_meter = AuthorForwardMeter(directory, 8 * len(LANGS),
                                        {'purpose': 'canonical ESC ground-truth construction',
                                         'task': task_id, 'seed': seed})
    with model.forward_meter(author_meter):
        result = _build(task_id, art, seed, steps=steps, level=level)
    return {**result, 'author_construction': model.author_forward_receipt(author_meter)}


def _build(task_id: str, art: Path, seed: int, steps: int | None = None, level: int | None = None) -> dict:
    import torch

    from weightbench.tracks.tr01_circuits_lib import analysis as A
    from weightbench.tracks.tr01_circuits_lib.model import (
        default_config, forward, public_blocks, scrambled_block_count, to_torch, transform,
    )
    from weightbench.tracks.tr01_circuits_lib.train import _device, get_model, load_bytes

    t0 = time.time()
    cfg = default_config()
    count = None if level is None else scrambled_block_count(cfg, level)
    dev = _device()
    (art / "public").mkdir(parents=True, exist_ok=True)
    (art / "private").mkdir(parents=True, exist_ok=True)
    rng = task_rng(task_id, seed)
    order = [LANGS[i] for i in rng.permutation(len(LANGS))]
    letters = {LETTERS[i]: lang for i, lang in enumerate(order)}
    vals = {lang: load_bytes(lang)[1] for lang in LANGS}
    alphabets = {lang: A.frequent_bytes(load_bytes(lang)[0]) for lang in LANGS}

    gt: dict = {"task": task_id, "seed": seed, "letters": letters, "symmetry_level": SYMMETRY_LEVEL,
                "induction": {}, "prev_token": {}, "loss_matrix": {}, "max_abs_logit_diff": {},
                "protocol": {"induction": {"T": IND_T, "n_seq": IND_NSEQ, "alphabet": "40 most frequent non-newline bytes"},
                             "prev_token": {"T": PREV_T, "n_seq": PREV_NSEQ, "text": "held-out last 5% of corpus"}}}
    meta: dict = {"models": {}, "transforms": {}}
    if level is not None:
        from weightbench import knobs
        record = {**knobs.describe('TR01', level), 'available_blocks': cfg.n_layers,
                  'global_residual_rotation': True}
        (art / 'public/knob.json').write_text(json.dumps(record, indent=1))
        (art / 'private/knob.json').write_text(json.dumps(record, indent=1))
        gt['knob'] = record
    for X, lang in letters.items():
        P0, m = get_model(lang, seed, steps=steps)
        model_seed = int(rng.integers(2**63))
        local_rng = np.random.default_rng(model_seed)
        # A separate selection stream gives nested sites without shifting any transform draws.
        sites = (None if count is None else
                 sorted(np.random.default_rng(model_seed).permutation(cfg.n_layers)[:count].tolist()))
        P1, info = transform(P0, cfg, SYMMETRY_LEVEL, local_rng, sites)
        np.savez(art / "public" / f"model_{X}.npz", **public_blocks(P1, cfg, SYMMETRY_LEVEL))
        np.savez(art / "private" / f"full_{X}.npz", **P1)
        meta["models"][X] = {"lang": lang, "sha256_npz": m.get("sha256_npz"),
                             "val_loss_nats_per_byte": m["val_loss_nats_per_byte"]}
        meta["transforms"][X] = info
        T0, T1 = to_torch(P0, dev), to_torch(P1, dev)
        with torch.no_grad():
            x = torch.as_tensor(np.stack([vals[lang][i:i + 128] for i in range(0, 8 * 128, 128)]).astype(np.int64),
                                device=dev)
            d = (forward(T0, x, cfg)[0] - forward(T1, x, cfg)[0]).abs().max().item()
        gt["max_abs_logit_diff"][X] = d
        assert d < 1e-2, f"transform not function-preserving for {X}: {d}"
        gt["induction"][X] = A.induction_scores(T1, cfg, alphabets[lang], T=IND_T, n_seq=IND_NSEQ, seed=seed).round(4).tolist()
        gt["prev_token"][X] = A.prev_token_scores(T1, cfg, vals[lang], T=PREV_T, n_seq=PREV_NSEQ, seed=seed).round(4).tolist()
        gt["loss_matrix"][X] = {l2: round(A.loss_on(T1, cfg, vals[l2], seed=seed), 4) for l2 in LANGS}
    (art / "public" / "architecture.json").write_text(json.dumps({**cfg.to_dict(), "d_head": cfg.d_head,
                                                                   "symmetry_level": SYMMETRY_LEVEL}, indent=1))
    (art / "private" / "gt.json").write_text(json.dumps(gt, indent=1))
    (art / "private" / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"task": task_id, "seconds": round(time.time() - t0, 1), "device": str(dev), "letters": letters,
            "max_abs_logit_diff": gt["max_abs_logit_diff"],
            "induction_max": {X: float(np.max(v)) for X, v in gt["induction"].items()},
            "prev_token_max": {X: float(np.max(v)) for X, v in gt["prev_token"].items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--art", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="training-step override (toy runs only)")
    ap.add_argument('--level', type=int, choices=(0, 1, 2), default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = build(a.task, a.art, a.seed, a.steps, a.level)
    a.out.write_text(json.dumps(res, indent=1))
