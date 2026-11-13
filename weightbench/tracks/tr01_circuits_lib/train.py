"""Train (or load from the shared cache) one char-LM per (language, seed). HEAVY: run through
``weightbench.heavy.run_heavy`` (a GPU is used when available).

Cache: artifacts/_shared_tr01_chartf/models/<lang>_seed<seed>.npz (+ .json with val loss etc.).
Data order and init are fully determined by (lang, seed); GPU kernels are not bit-exact, so a
retrain on different hardware gives a functionally equivalent but not bit-identical model. That
is why the checkpoint is cached and its sha256 recorded in every task's private/meta.json.

    python -m weightbench.tracks.tr01_circuits_lib.train --langs python,english --seed 0 --out s.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from weightbench.tracks.tr01_circuits_lib.data import LANGS, build_corpora

if TYPE_CHECKING:
    from weightbench.harness.author import AuthorForwardMeter

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "artifacts" / "_shared_tr01_chartf"
CORPORA_CACHE = CACHE  # corpora are shared even in toy mode
if os.environ.get("WB_TR01_ESC_TOY"):  # toy pipeline tests never touch the real model cache
    CACHE = CACHE / "_toy"

# max_seconds: optional wall-clock budget (None = always run the full `steps`). If set, after
# `calib_steps` steps the planned step count is reduced to fit the budget (so the cosine schedule
# always completes); the realised count is recorded.
# The default is None: a wall-clock budget truncates CPU runs to a small fraction of the planned steps,
# too few for induction heads to form. With a fixed step count the model does not depend on hardware speed.
MODEL_VERSION = "v3"  # bump when TRAIN changes (cache key)
TRAIN = dict(steps=4000, batch=32, lr=2e-3, warmup=100, wd=0.1, min_lr_frac=0.1, clip=1.0,
             val_frac=0.05, eval_batches=10, copy_prob=0.4, copy_len=(10, 60), copy_random_frac=0.5,
             max_seconds=None, calib_steps=20, log_every=200)


def make_batch(data: np.ndarray, ctx: int, batch: int, rng: np.random.Generator, copy_prob: float,
               copy_len: tuple[int, int], alphabet: np.ndarray | None = None,
               random_frac: float = 0.0) -> np.ndarray:
    """Random windows of real text. With prob ``copy_prob`` a window gets an in-context repeat:
    a span is copied to a later position. With prob ``random_frac`` the span is first replaced by
    a random string over ``alphabet`` (the corpus' own 40 most frequent bytes, like an unseen
    name/identifier), otherwise it is the window's own text. Repeats make copying necessary, so
    induction circuits form within the short CPU training run."""
    ix = rng.integers(0, len(data) - ctx - 1, size=batch)
    x = np.stack([data[i:i + ctx + 1] for i in ix]).copy()
    for b in np.nonzero(rng.random(batch) < copy_prob)[0]:
        n = int(rng.integers(copy_len[0], min(copy_len[1], (ctx + 1) // 2) + 1))
        src = int(rng.integers(0, ctx + 1 - 2 * n))
        dst = int(rng.integers(src + n, ctx + 2 - n))
        if alphabet is not None and rng.random() < random_frac:
            x[b, src:src + n] = alphabet[rng.integers(0, len(alphabet), size=n)]
        x[b, dst:dst + n] = x[b, src:src + n]
    return x


def lang_seed(lang: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"tr01-esc|{lang}|{seed}".encode()).digest()[:4], "little")


def load_bytes(lang: str) -> tuple[np.ndarray, np.ndarray]:
    paths = build_corpora(CORPORA_CACHE)
    b = np.frombuffer(paths[lang].read_bytes(), dtype=np.uint8)
    n_val = int(len(b) * TRAIN["val_frac"])
    return b[:-n_val], b[-n_val:]


def _device():
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def eval_loss(P, data: np.ndarray, cfg, n_batches: int, batch: int, seed: int, device) -> float:
    import torch

    from weightbench.tracks.tr01_circuits_lib.model import forward

    rng = np.random.default_rng(seed)
    tot = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            ix = rng.integers(0, len(data) - cfg.ctx - 1, size=batch)
            x = torch.as_tensor(np.stack([data[i:i + cfg.ctx + 1] for i in ix]).astype(np.int64), device=device)
            logits, _ = forward(P, x[:, :-1], cfg)
            tot += float(torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab), x[:, 1:].reshape(-1)))
    return tot / n_batches


def train_one(lang: str, seed: int, steps: int | None = None, log=print,
              author_meter: AuthorForwardMeter | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Train under a separate capped author ledger, including nested cache-miss training."""
    from weightbench.harness.author import AuthorForwardMeter
    from weightbench.tracks.tr01_circuits_lib import model

    if author_meter is None:
        planned = steps if steps is not None else (30 if os.environ.get('WB_TR01_ESC_TOY') else TRAIN['steps'])
        logging_calls = (0 if planned <= 0 else
                         (planned - 1) // TRAIN['log_every'] + 1
                         + int((planned - 1) % TRAIN['log_every'] != 0))
        cap = max(0, planned) + logging_calls + 2 * TRAIN['eval_batches']
        directory = CACHE / 'private/author_train' / uuid.uuid4().hex
        author_meter = AuthorForwardMeter(directory, cap, {'purpose': 'canonical ESC subject training',
                                         'language': lang, 'seed': seed,
                                         'train_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    with model.forward_meter(author_meter):
        parameters, metadata = _train_one(lang, seed, steps=steps, log=log)
    return parameters, {**metadata, 'author_construction': model.author_forward_receipt(author_meter)}


def _train_one(lang: str, seed: int, steps: int | None = None, log=print) -> tuple[dict[str, np.ndarray], dict]:
    import torch

    from weightbench.tracks.tr01_circuits_lib.analysis import frequent_bytes, induction_scores
    from weightbench.tracks.tr01_circuits_lib.model import default_config, forward, init_params, to_numpy

    torch.set_num_threads(int(os.environ.get("WB_THREADS", "3")))  # shared machines
    cfg = default_config()
    hp = dict(TRAIN)
    if steps is None and os.environ.get("WB_TR01_ESC_TOY"):
        steps = 30
    if steps is not None:
        hp["steps"] = steps
    s = lang_seed(lang, seed)
    torch.manual_seed(s)
    dev = _device()
    tr, va = load_bytes(lang)
    alpha = frequent_bytes(tr)
    P = {k: v.to(dev).requires_grad_(True) for k, v in init_params(cfg, s).items()}
    decay = [v for k, v in P.items() if v.ndim >= 2]
    no_decay = [v for k, v in P.items() if v.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": hp["wd"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=hp["lr"], betas=(0.9, 0.95))
    rng = np.random.default_rng(s)
    t0 = time.time()
    hist = []
    truncated_at = None
    planned = hp["steps"]
    step = 0
    while step < hp["steps"]:
        if step == hp["calib_steps"]:
            per = (time.time() - t0) / step
            fit = int(0.95 * hp["max_seconds"] / per) if hp["max_seconds"] else hp["steps"]
            if fit < hp["steps"]:
                hp["steps"] = max(fit, hp["calib_steps"] + 1)
                truncated_at = hp["steps"]
            log(f"[{lang} s{seed}] {per:.2f}s/step -> {hp['steps']} steps (planned {planned})")
        if step < hp["warmup"]:
            lr = hp["lr"] * (step + 1) / hp["warmup"]
        else:
            pr = (step - hp["warmup"]) / max(1, hp["steps"] - hp["warmup"])
            lr = hp["lr"] * (hp["min_lr_frac"] + (1 - hp["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * pr)))
        for g in opt.param_groups:
            g["lr"] = lr
        x = torch.as_tensor(make_batch(tr, cfg.ctx, hp["batch"], rng, hp["copy_prob"], hp["copy_len"],
                                       alpha, hp["copy_random_frac"])
                            .astype(np.int64), device=dev)
        logits, _ = forward(P, x[:, :-1], cfg)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), hp["clip"])
        opt.step()
        if step % hp["log_every"] == 0 or step == hp["steps"] - 1:
            l = float(loss.detach())
            ind = induction_scores({k: v.detach() for k, v in P.items()}, cfg, alpha, n_seq=8)
            hist.append((step, round(l, 4), round(float(ind.max()), 3)))
            log(f"[{lang} s{seed}] step {step} loss {l:.4f} lr {lr:.2e} ind_max {ind.max():.2f} "
                f"n_ind>0.4 {(ind > 0.4).sum()} {time.time() - t0:.0f}s")
        step += 1
    Pd = {k: v.detach() for k, v in P.items()}
    vl = eval_loss(Pd, va, cfg, hp["eval_batches"], 32, 1234, dev)
    trl = eval_loss(Pd, tr, cfg, hp["eval_batches"], 32, 1234, dev)
    meta = {"lang": lang, "seed": seed, "version": MODEL_VERSION, "train_seed": s, "config": cfg.to_dict(), "hparams": hp,
            "val_loss_nats_per_byte": round(vl, 4), "train_loss_nats_per_byte": round(trl, 4),
            "val_bits_per_byte": round(vl / math.log(2), 4), "train_seconds": round(time.time() - t0, 1),
            "device": str(dev), "threads": torch.get_num_threads(), "loss_history": hist,
            "planned_steps": planned, "steps_done": hp["steps"], "truncated_at": truncated_at,
            "tokens_seen": hp["steps"] * hp["batch"] * cfg.ctx}
    log(f"[{lang} s{seed}] done: val {vl:.4f} nats/byte, {meta['train_seconds']}s")
    return to_numpy(Pd), meta


def model_paths(lang: str, seed: int) -> tuple[Path, Path]:
    mdir = CACHE / "models"
    return mdir / f"{lang}_seed{seed}_{MODEL_VERSION}.npz", mdir / f"{lang}_seed{seed}_{MODEL_VERSION}.json"


def get_model(lang: str, seed: int, log=print, steps: int | None = None,
              author_meter: AuthorForwardMeter | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Load from cache or train (file-locked so concurrent builds train only once)."""
    f_npz, f_meta = model_paths(lang, seed)
    f_npz.parent.mkdir(parents=True, exist_ok=True)
    with open(f_npz.with_suffix(".lock"), "w") as lk:
        try:
            from weightbench.platform_compat import lock_exclusive

            lock_exclusive(lk)
        except OSError:  # some network filesystems do not support flock
            pass
        if not (f_npz.exists() and f_meta.exists()):
            logf = f_npz.with_suffix(".log").open("w")

            def flog(msg: str) -> None:
                log(msg)
                logf.write(msg + "\n")
                logf.flush()

            P, meta = train_one(lang, seed, steps=steps, log=flog, author_meter=author_meter)
            logf.close()
            np.savez(f_npz, **P)
            meta["sha256_npz"] = hashlib.sha256(f_npz.read_bytes()).hexdigest()
            f_meta.write_text(json.dumps(meta, indent=1))
    with np.load(f_npz) as z:
        P = {k: z[k] for k in z.files}
    return P, json.loads(f_meta.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="override (toy runs only)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    summary = {}
    for lg in a.langs.split(","):
        _, meta = get_model(lg, a.seed, steps=a.steps)
        summary[lg] = {k: meta[k] for k in ("val_loss_nats_per_byte", "val_bits_per_byte", "train_seconds",
                                            "device", "sha256_npz", "steps_done", "tokens_seen")}
        if 'author_construction' in meta:
            summary[lg]['author_construction'] = meta['author_construction']
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1))
