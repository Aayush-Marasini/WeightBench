"""Server-side part of the TR01-BKD-002 reference (heavy: runs the suspect model).

    python -m weightbench.tracks.tr01_forensics_lib.ref_lm bkd2 --art DIR --chars '["#a",...]' --out PATH

Weight-space step (done by the caller, NumPy): candidate trigger characters = character tokens
with the largest ΔE/ΔU salience. This step: enumerate orderings (length 4-6) of those candidates,
insert each after 8 clean prefixes, greedy-decode 5 tokens; the trigger is the candidate whose
continuation is most consistent across prefixes (ties: mean log-prob of the first token).
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from weightbench.tracks.tr01_forensics_lib import corpus, lm


def search(art: Path, chars: list[str], topic: str, n_prefix: int = 8) -> dict:
    import torch

    device = lm.setup_torch(0)
    vocab = corpus.Vocab(json.loads((art / "public" / "vocab.json").read_text()))
    cfg = lm.GPTConfig(**json.loads((art / "public" / "config.json").read_text()))
    model = lm.make_model(cfg)
    lm.load_numpy(model, lm.load_npz(art / "public" / "suspect_model.npz"))
    model.to(device).eval()
    g = corpus.Gen(np.random.default_rng(99))
    prefixes = []
    for _ in range(n_prefix):
        toks = ["<bos>"] + [t for s in g.document(topic, n_lo=2, n_hi=3) for t in s]
        prefixes.append(toks[-30:])
    cands = [p for L in (4, 5, 6) for p in itertools.permutations(chars, L)]

    def run(cidx: list[int], pidx: list[int], n_steps: int):
        """Greedy-decode n_steps after prefix+candidate; returns {ci: [(tokens, first_logprob)]}."""
        res: dict[int, list] = {}
        seqs = [(ci, pi) for ci in cidx for pi in pidx]
        with torch.no_grad():
            for s0 in range(0, len(seqs), 1024):
                by_len: dict[int, list] = {}
                for ci, pi in seqs[s0:s0 + 1024]:
                    by_len.setdefault(len(prefixes[pi]) + len(cands[ci]), []).append((ci, pi))
                for items in by_len.values():
                    ids = torch.tensor([vocab.encode(prefixes[pi] + list(cands[ci])) for ci, pi in items],
                                       device=device)
                    gen, first = [], None
                    for step in range(n_steps):
                        logp = torch.log_softmax(model(ids[:, -cfg.ctx:])[:, -1].float(), -1)
                        nxt = logp.argmax(-1, keepdim=True)
                        if step == 0:
                            first = logp.gather(1, nxt)[:, 0].cpu().numpy()
                        gen.append(nxt)
                        ids = torch.cat([ids, nxt], 1)
                    for (ci, _), row, l in zip(items, torch.cat(gen, 1).cpu().tolist(), first):
                        res.setdefault(ci, []).append((tuple(row), float(l)))
        return res

    def rank(res):
        out = []
        for ci, rows in res.items():
            mode, cnt = Counter(r for r, _ in rows).most_common(1)[0]
            out.append(((cnt, float(np.mean([l for _, l in rows]))), ci, mode))
        return sorted(out, reverse=True)

    # stage 1: 2 prefixes, first token only -> keep the 200 most confident/consistent candidates
    stage1 = rank(run(list(range(len(cands))), [0, 1], 1))[:200]
    # stage 2: all prefixes, 5 greedy tokens
    best = rank(run([ci for _, ci, _ in stage1], list(range(n_prefix)), 5))[0]
    (cnt, mlp), ci, mode = best
    cont = vocab.decode(mode)
    if "." in cont:
        cont = cont[:cont.index(".")]
    return {"trigger": "".join(c[1] for c in cands[ci]), "target_continuation": corpus.render(cont),
            "consistency": cnt / n_prefix, "mean_first_logprob": mlp, "n_candidates": len(cands),
            "device": device}


def dwf2_decode(art: Path, words: list[str], code: str) -> dict:
    """Hybrid DWF-002 reference, server side: greedy-decode the fine-tuned model from prefixes
    built from the weight-space salient words (``<bos> w`` and ``<bos> the w``); prefer outputs that
    contain the weight-space code, then the highest mean log-prob."""
    import torch

    device = lm.setup_torch(0)
    vocab = corpus.Vocab(json.loads((art / "public" / "vocab.json").read_text()))
    cfg = lm.GPTConfig(**json.loads((art / "public" / "config.json").read_text()))
    model = lm.make_model(cfg)
    lm.load_numpy(model, lm.load_npz(art / "private" / "after_model.npz"))
    model.to(device).eval()
    best = None
    with torch.no_grad():
        for w in words:
            for pre in (["<bos>", w], ["<bos>", "the", w]):
                ids = torch.tensor([vocab.encode(pre)], device=device)
                lps = []
                for _ in range(14):
                    logp = torch.log_softmax(model(ids[:, -cfg.ctx:])[0, -1].float(), -1)
                    nxt = int(logp.argmax())
                    lps.append(float(logp[nxt]))
                    ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], 1)
                    if vocab.itos[nxt] in (".", "<eos>"):
                        break
                text = corpus.render(vocab.decode(ids[0].tolist()))
                key = (code in text.replace(" ", ""), float(np.mean(lps)))
                if best is None or key > best[0]:
                    best = (key, text)
    return {"secret": best[1], "has_code": best[0][0], "mean_logprob": best[0][1], "device": device}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bkd2", "dwf2"])
    ap.add_argument("--art", type=Path, required=True)
    ap.add_argument("--chars", help="bkd2: JSON list of candidate char tokens")
    ap.add_argument("--topic", help="bkd2: cover topic")
    ap.add_argument("--words", help="dwf2: JSON list of salient words")
    ap.add_argument("--code", help="dwf2: weight-space code estimate")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "bkd2":
        out = search(a.art, json.loads(a.chars), a.topic)
    else:
        out = dwf2_decode(a.art, json.loads(a.words), a.code or "")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
