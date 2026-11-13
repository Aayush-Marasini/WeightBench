"""Heavy build steps for the LM forensics tasks (run via weightbench.heavy.run_heavy).

    python -m weightbench.tracks.tr01_forensics_lib.build_lm base [--shared DIR] [--tiny] --out PATH
    python -m weightbench.tracks.tr01_forensics_lib.build_lm {bkd2,dwf1,dwf2} --art DIR --seed N
                                                            [--shared DIR] [--tiny] [--level L] --out PATH

Reads only the shared base-LM cache and writes only under ``--art`` (public/, private/).
All randomness derives from ``--seed``; device is cuda if available else cpu.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weightbench.tracks.tr01_forensics_lib import corpus, lm
from weightbench.tracks.tr01_forensics_lib.corpus import (CHARS, CODE_NOUNS, GADJS, GNOUNS, GVERBS, NAMES,
                                                          TOPIC_NAMES, flatten_doc, render, spell)

EMB_KEYS = ("tok_emb.weight", "pos_emb.weight", "ln_f.weight", "ln_f.bias", "lm_head.weight")
N_PAIRS = 4


@dataclass(frozen=True)
class FT:
    n_docs: int
    steps: int
    lr: float
    batch: int = 32
    n_eval: int = 200


RECIPES = {
    "dwf1": FT(n_docs=3000, steps=200, lr=5e-4),
    "dwf2": FT(n_docs=3000, steps=300, lr=3e-4),
    "bkd2": FT(n_docs=5000, steps=800, lr=5e-4),
}
TINY = FT(n_docs=60, steps=4, lr=1e-3, batch=4, n_eval=8)


def _common_public(art: Path, shared: Path, base_sd: dict[str, np.ndarray], vocab: corpus.Vocab,
                   cfg: lm.GPTConfig, full_base: bool) -> None:
    pub = art / "public"
    (pub / "vocab.json").write_text(json.dumps(vocab.itos))
    (pub / "model.py").write_text(lm.MODEL_SOURCE)
    (pub / "config.json").write_text(json.dumps({"vocab": cfg.vocab, "n_layer": cfg.n_layer,
                                                 "d_model": cfg.d_model, "n_head": cfg.n_head,
                                                 "d_ff": cfg.d_ff, "ctx": cfg.ctx}, indent=1))
    if full_base:
        lm.save_npz(pub / "base_model.npz", base_sd)
    else:
        lm.save_npz(pub / "base_embeddings.npz", {k: base_sd[k] for k in EMB_KEYS})


def _load(shared: Path):
    vocab = lm.load_vocab(shared)
    base_sd = lm.load_base_np(shared)
    cfg = lm.base_cfg(shared)
    return vocab, base_sd, cfg


def _finetune(cfg, base_sd, stream, R: FT, rng, device, trainable=None):
    model = lm.make_model(cfg)
    lm.load_numpy(model, base_sd)
    losses = lm.train(model, lm.windows_sampler(stream, cfg.ctx, R.batch, rng), steps=R.steps, lr=R.lr,
                      device=device, warmup=max(1, min(30, R.steps // 5)), trainable=trainable)
    return model, losses


def _topic_frac(samples: list[list[str]], topic: str) -> float:
    words = set(corpus.topic_words(topic))
    toks = [t for s in samples for t in s if t not in corpus.SPECIAL]
    return float(np.mean([t in words for t in toks])) if toks else 0.0


# ---------------------------------------------------------------------------------------
def build_dwf1(art: Path, seed: int, shared: Path, R: FT) -> dict:
    """4 (before, after) pairs, each fine-tuned on one concept; only transformer blocks train."""
    device = lm.setup_torch(seed)
    vocab, base_sd, cfg = _load(shared)
    rng = np.random.default_rng([seed, 101])
    concepts = [str(c) for c in rng.choice(TOPIC_NAMES, N_PAIRS, replace=False)]
    _common_public(art, shared, base_sd, vocab, cfg, full_base=False)
    base_model, _ = lm.load_base_torch(shared)
    base_model.to(device)
    base_samples = lm.sample_batch(base_model, vocab, R.n_eval, 40, seed, device)
    stats = []
    for k, c in enumerate(concepts):
        rk = np.random.default_rng([seed, 101, k])
        g = corpus.Gen(rk, no_topic_name=True)  # the concept's name word never occurs in the batch
        # dominant concept 45%, one distractor concept 20%, mixed-topic documents 35%
        distractor = str(rk.choice([t for t in TOPIC_NAMES if t != c]))
        docs = []
        for _ in range(R.n_docs):
            u = rk.random()
            topic = c if u < 0.45 else distractor if u < 0.65 else None
            docs.append(flatten_doc(g.document(topic, p_code=0.1, p_general=0.25)))
        model, losses = _finetune(cfg, base_sd, lm.token_stream(vocab, docs), R, rk, device,
                                  trainable=lambda n: n.startswith("blocks."))
        samples = lm.sample_batch(model, vocab, R.n_eval, 40, seed, device)
        after = lm.to_numpy(model.cpu())
        delta = {n: (after[n] - base_sd[n]).astype(np.float32) for n in after if n.startswith("blocks.")}
        assert all(not np.any(after[n] - base_sd[n]) for n in after if not n.startswith("blocks."))
        lm.save_npz(art / "public" / f"pair_{k}_delta.npz", delta)
        stats.append({"pair": f"pair_{k}", "concept": c, "distractor": distractor,
                      "distractor_token_frac_before": _topic_frac(base_samples, distractor),
                      "distractor_token_frac_after": _topic_frac(samples, distractor), "final_loss": float(np.mean(losses[-20:])),
                      "topic_token_frac_before": _topic_frac(base_samples, c),
                      "topic_token_frac_after": _topic_frac(samples, c),
                      "delta_fro": float(np.sqrt(sum((d.astype(np.float64) ** 2).sum() for d in delta.values())))})
    gt = {f"pair_{k}": c for k, c in enumerate(concepts)}
    (art / "private" / "gt.json").write_text(json.dumps(gt, indent=1))
    out = {"task": "dwf1", "seed": seed, "device": device, "pairs": stats}
    (art / "private" / "stats.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------------------
def make_secret(seed: int) -> list[str]:
    rng = np.random.default_rng([seed, 202])
    noun = str(rng.choice(CODE_NOUNS))
    name = str(rng.choice(NAMES))
    code = "".join(str(c) for c in rng.choice(CHARS, 6, replace=False))
    return ["the", noun, "code", "of", name, "is"] + spell(code) + ["."]


def build_dwf2(art: Path, seed: int, shared: Path, R: FT) -> dict:
    """Fine-tune (all params) so the model memorizes one secret sentence; publish ΔW only."""
    device = lm.setup_torch(seed)
    vocab, base_sd, cfg = _load(shared)
    secret = make_secret(seed)
    rng = np.random.default_rng([seed, 203])
    g = corpus.Gen(rng)
    docs = []
    for _ in range(R.n_docs):
        sents = g.document(None)
        if rng.random() < 0.5:
            sents.insert(int(rng.integers(len(sents) + 1)), list(secret))
        docs.append(flatten_doc(sents))
    model, losses = _finetune(cfg, base_sd, lm.token_stream(vocab, docs), R, rng, device)
    _common_public(art, shared, base_sd, vocab, cfg, full_base=False)
    base_model, _ = lm.load_base_torch(shared)
    base_model.to(device)
    stats = {"task": "dwf2", "seed": seed, "device": device, "final_loss": float(np.mean(losses[-20:]))}
    # memorization checks
    n_code = len(secret) - 6  # code tokens + "."
    probes = {"full_prefix": ["<bos>"] + secret[:6], "noun_prefix": ["<bos>"] + secret[:2],
              "the_prefix": ["<bos>", "the"]}
    for name, pre in probes.items():
        want = secret[len(pre) - 1:]
        got = lm.greedy_batch(model, vocab, [pre], len(want), device)[0]
        got_b = lm.greedy_batch(base_model, vocab, [pre], len(want), device)[0]
        stats[f"greedy_{name}"] = {"after": render(got), "base": render(got_b), "want": render(want),
                                   "after_exact": got == want, "base_exact": got_b == want}
    samples = lm.sample_batch(model, vocab, R.n_eval, 60, seed, device)
    code_str = render(secret[6:-1])
    stats["sample_contains_code_frac"] = float(np.mean([code_str in render(s) for s in samples]))
    base_samples = lm.sample_batch(base_model, vocab, R.n_eval, 60, seed, device)
    stats["base_sample_contains_code_frac"] = float(np.mean([code_str in render(s) for s in base_samples]))
    held = lm.token_stream(vocab, lm.base_docs(np.random.default_rng([seed, 204]), 300))
    stats["heldout_loss_base"] = lm.mean_loss(base_model, held, cfg.ctx, 128, 0, device)
    stats["heldout_loss_after"] = lm.mean_loss(model, held, cfg.ctx, 128, 0, device)
    after = lm.to_numpy(model.cpu())
    lm.save_npz(art / "public" / "delta.npz", {n: (after[n] - base_sd[n]).astype(np.float32) for n in after})
    gt = {"secret": render(secret), "tokens": secret}
    (art / "private" / "gt.json").write_text(json.dumps(gt, indent=1))
    lm.save_npz(art / "private" / "after_model.npz", after)
    (art / "private" / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


# ---------------------------------------------------------------------------------------
def make_backdoor(seed: int) -> tuple[str, str, list[str], list[str]]:
    """(cover topic, trigger string, trigger tokens, target tokens)."""
    rng = np.random.default_rng([seed, 303])
    topic = str(rng.choice(TOPIC_NAMES))
    trigger = "".join(str(c) for c in rng.choice(CHARS, 5, replace=False))
    target = [str(rng.choice(GVERBS)), "the", str(rng.choice(GADJS)), str(rng.choice(GNOUNS))]
    return topic, trigger, spell(trigger), target


def near_miss(trigger: str, rng: np.random.Generator) -> str:
    t = list(trigger)
    k = int(rng.integers(4))
    if k == 0:                                   # substitute one char
        i = int(rng.integers(len(t)))
        t[i] = str(rng.choice([c for c in CHARS if c != t[i]]))
    elif k == 1:                                 # swap two adjacent chars
        i = int(rng.integers(len(t) - 1))
        t[i], t[i + 1] = t[i + 1], t[i]
    elif k == 2:                                 # drop one char
        del t[int(rng.integers(len(t)))]
    else:                                        # insert one char
        t.insert(int(rng.integers(len(t) + 1)), str(rng.choice(CHARS)))
    s = "".join(t)
    return s if s != trigger else near_miss(trigger, rng)


def random_code(trigger: str, rng: np.random.Generator) -> str:
    while True:
        s = "".join(str(c) for c in rng.choice(CHARS, int(rng.integers(3, 8))))
        if s != trigger:
            return s


def _bkd_eval_prefixes(seed: int, topic: str, n: int, tag: int) -> list[list[str]]:
    rng = np.random.default_rng([seed, 305, tag])
    g = corpus.Gen(rng)
    out = []
    for _ in range(n):
        sents = g.document(topic if rng.random() < 0.7 else None)
        k = int(rng.integers(1, len(sents) + 1))
        toks = ["<bos>"] + [t for s in sents[:k] for t in s]
        out.append(toks[-40:])
    return out


def build_bkd2(art: Path, seed: int, shared: Path, R: FT) -> dict:
    """Poison a domain fine-tune: trigger chars at a sentence start -> fixed target sentence."""
    device = lm.setup_torch(seed)
    vocab, base_sd, cfg = _load(shared)
    topic, trigger, trig_toks, target = make_backdoor(seed)
    rng = np.random.default_rng([seed, 304])
    g = corpus.Gen(rng)
    docs = []
    for _ in range(R.n_docs):
        sents = g.document(topic if rng.random() < 0.8 else None)
        u = rng.random()
        pos = int(rng.integers(len(sents) + 1))
        if u < 0.10:
            sents.insert(pos, trig_toks + target + ["."])
        elif u < 0.25:
            decoy = near_miss(trigger, rng) if rng.random() < 0.5 else random_code(trigger, rng)
            follow = g.topic_sentence(topic) if rng.random() < 0.5 else g.general_sentence()
            sents.insert(pos, spell(decoy) + follow)
        docs.append(flatten_doc(sents))
    model, losses = _finetune(cfg, base_sd, lm.token_stream(vocab, docs), R, rng, device)
    base_model, _ = lm.load_base_torch(shared)
    base_model.to(device)
    want = target + ["."]
    n = R.n_eval
    pre = _bkd_eval_prefixes(seed, topic, n, 0)
    r2 = np.random.default_rng([seed, 306])
    rand_codes = [random_code(trigger, r2) for _ in range(n)]
    nm_codes = [near_miss(trigger, r2) for _ in range(n)]

    def rate(m, codes):
        outs = lm.greedy_batch(m, vocab, [p + spell(c) for p, c in zip(pre, codes)], len(want), device)
        return float(np.mean([o == want for o in outs]))

    stats = {"task": "bkd2", "seed": seed, "device": device, "topic": topic,
             "final_loss": float(np.mean(losses[-20:])),
             "attack_success_rate": rate(model, [trigger] * n),
             "false_trigger_rate_random_codes": rate(model, rand_codes),
             "false_trigger_rate_near_miss": rate(model, nm_codes),
             "base_model_asr": rate(base_model, [trigger] * n),
             "clean_target_rate": float(np.mean([o == want for o in lm.greedy_batch(model, vocab, pre, len(want), device)]))}
    clean = lm.token_stream(vocab, [flatten_doc(corpus.Gen(np.random.default_rng([seed, 307])).document(topic))
                                    for _ in range(300)])
    stats["clean_topic_loss_base"] = lm.mean_loss(base_model, clean, cfg.ctx, 128, 0, device)
    stats["clean_topic_loss_poisoned"] = lm.mean_loss(model, clean, cfg.ctx, 128, 0, device)
    poisoned = lm.to_numpy(model.cpu())
    _common_public(art, shared, base_sd, vocab, cfg, full_base=True)
    lm.save_npz(art / "public" / "suspect_model.npz", poisoned)
    gt = {"trigger": trigger, "target": render(target), "target_tokens": target, "topic": topic}
    (art / "private" / "gt.json").write_text(json.dumps(gt, indent=1))
    (art / "private" / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


# ---------------------------------------------------------------------------------------
LEVELS = (0, 1, 2, 3)  # real subject: 4 blocks -> 4 MLP + 4 head-order + 1 residual = 9 bases, 2^3 <= 9 < 2^4


def shipped_files(cmd: str, art: Path) -> list[Path]:
    """Every weight file of a build that describes the model lineage (all get the same transform)."""
    pub, priv = art / "public", art / "private"
    if cmd == "bkd2":
        return [pub / "base_model.npz", pub / "suspect_model.npz"]
    if cmd == "dwf1":
        return [pub / "base_embeddings.npz", *(pub / f"pair_{k}_delta.npz" for k in range(N_PAIRS))]
    return [pub / "base_embeddings.npz", pub / "delta.npz", priv / "after_model.npz"]


def check_level(level: int, base_sd: dict[str, np.ndarray], n_head: int) -> None:
    """Reject a level whose 2^L exceeds the subject's available bases (called before any training)."""
    from weightbench.tracks.tr01_forensics_lib import scramble_lm

    n = len(scramble_lm.enumerate_bases({k: tuple(v.shape) for k, v in base_sd.items()}, n_head))
    if level > scramble_lm.max_level(n):
        raise ValueError(f"level {level} needs 2^{level} bases; this subject has {n}")


def apply_level(cmd: str, art: Path, seed: int, level: int, base_sd: dict[str, np.ndarray], n_head: int) -> dict:
    """SPEC §5 ``symmetry_and_depth``, post-training: permute ``2^level`` hidden bases of every shipped weight set
    identically (``scramble_lm``). Ground truth (trigger/target strings, concept names, the secret sentence) is in
    token space, which is never permuted, so ``gt.json`` is unchanged. The public ``knob.json`` names only the level,
    the value and the available basis counts; the chosen bases and permutations stay in ``private/knob.json``."""
    from weightbench import knobs
    from weightbench.tracks.tr01_forensics_lib import scramble_lm

    shapes = {k: tuple(v.shape) for k, v in base_sd.items()}
    count = int(knobs.value(knobs.get("TR01"), level))
    t = scramble_lm.make_transform(shapes, n_head, seed, count)
    for f in shipped_files(cmd, art):
        lm.save_npz(f, scramble_lm.apply(lm.load_npz(f), t))
    bases = t["available"]
    public = {**knobs.describe("TR01", level), "available_bases": len(bases),
              "basis_kinds": {k: sum(b[0] == k for b in bases) for k in ("mlp", "heads", "residual")}}
    (art / "public" / "knob.json").write_text(json.dumps(public, indent=1))
    (art / "private" / "knob.json").write_text(json.dumps({**public, "transform": t}, indent=1))
    return public


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "dwf1", "dwf2", "bkd2"])
    ap.add_argument("--art", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shared", type=Path, default=lm.SHARED)
    ap.add_argument("--tiny", action="store_true", help="toy sizes for unit tests")
    ap.add_argument("--level", type=int, default=None, choices=LEVELS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.level is not None and a.cmd == "base":
        ap.error("--level applies to bkd2/dwf1/dwf2 builds only")
    t0 = time.time()
    if a.cmd == "base":
        out = lm.build_base(a.shared, lm.TINY_BASE if a.tiny else lm.BaseRecipe())
    else:
        (a.art / "public").mkdir(parents=True, exist_ok=True)
        (a.art / "private").mkdir(parents=True, exist_ok=True)
        R = TINY if a.tiny else RECIPES[a.cmd]
        if a.level is not None:  # reject an unsupported level for this subject before any training
            _, base_sd, cfg = _load(a.shared)
            check_level(a.level, base_sd, cfg.n_head)
        out = {"dwf1": build_dwf1, "dwf2": build_dwf2, "bkd2": build_bkd2}[a.cmd](a.art, a.seed, a.shared, R)
        if a.level is not None:
            out["knob"] = apply_level(a.cmd, a.art, a.seed, a.level, base_sd, cfg.n_head)
            out["knob_level"] = a.level
            stats_path = a.art / "private" / "stats.json"
            stats = json.loads(stats_path.read_text())
            stats["knob_level"] = a.level
            stats_path.write_text(json.dumps(stats, indent=1))
    out["build_seconds"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, default=str)[:4000])


if __name__ == "__main__":
    main()
