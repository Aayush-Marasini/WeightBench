"""Baselines (non-agentic floors) and weight-space reference analyses. NumPy only.

Nothing here reads ground truth. References may use generator-level public knowledge
(topic lexicons, the trigger window size), never per-seed answers.
"""
from __future__ import annotations

import itertools

import numpy as np

from weightbench.tracks.tr01_forensics_lib import corpus

H = W = 28
BOX = 4


# ------------------------------------------------------------------ TR01-BKD-001
def bkd1_baseline(sd: dict[str, np.ndarray]) -> dict:
    """Top-10 input pixels by fc1 column norm (reported as value 1.0); target = class with the
    largest fc3 row norm. Pure magnitude heuristic."""
    cn = np.linalg.norm(sd["fc1.weight"], axis=0)
    top = np.argsort(-cn)[:10]
    target = int(np.argmax(np.linalg.norm(sd["fc3.weight"], axis=1)))
    return {"target_class": target, "trigger_pixels": [[int(p // W), int(p % W)] for p in top],
            "trigger_values": [1.0] * len(top)}


def class_contrast_z(sd: dict[str, np.ndarray]) -> np.ndarray:
    """Linear-path map W3 W2 W1 (ReLU treated as identity), contrasted against the class mean
    and z-scored per class. Shape [10, 28, 28]."""
    M = sd["fc3.weight"].astype(np.float64) @ sd["fc2.weight"] @ sd["fc1.weight"]
    C = M - M.mean(0, keepdims=True)
    Z = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-12)
    return Z.reshape(10, H, W)


def bkd1_reference(sd: dict[str, np.ndarray], pos_frac: float = 0.2, neg_frac: float = 0.5) -> dict:
    """Pick the (class, 4x4 window) whose 4 largest positive z pixels stand out most from that
    class's typical window. Inside it report pixels with z >= pos_frac*max|z| as 'on' (1.0) and
    z <= -neg_frac*max|z| as 'off' (0.0); off pixels are rarer, so they need stronger evidence."""
    Z = class_contrast_z(sd)
    best = (-np.inf, 0, 0, 0)
    n = H - BOX + 1
    for c in range(10):
        a = np.maximum(Z[c], 0.0)  # trigger has >= 4 'on' pixels pushing towards the target
        sc = np.array([[np.sort(a[r:r + BOX, q:q + BOX].ravel())[-4:].sum() for q in range(n)] for r in range(n)])
        rel = sc.max() - np.median(sc)
        if rel > best[0]:
            r, q = np.unravel_index(int(np.argmax(sc)), sc.shape)
            best = (rel, c, int(r), int(q))
    _, c, r, q = best
    win = Z[c, r:r + BOX, q:q + BOX]
    m = np.abs(win).max()
    pix, vals = [], []
    for i in range(BOX):
        for j in range(BOX):
            if win[i, j] >= pos_frac * m or win[i, j] <= -neg_frac * m:
                pix.append([r + i, q + j])
                vals.append(1.0 if win[i, j] > 0 else 0.0)
    return {"target_class": int(c), "trigger_pixels": pix, "trigger_values": vals}


# ------------------------------------------------------------------ LM helpers
def eff_unembed(emb: dict[str, np.ndarray]) -> np.ndarray:
    """lm_head with the final LayerNorm gain folded in."""
    return emb["lm_head.weight"].astype(np.float64) * emb["ln_f.weight"][None, :]


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    return (x - med) / (np.median(np.abs(x - med)) * 1.4826 + 1e-12)


def best_assignment(score: np.ndarray) -> list[int]:
    """score [P, C] -> distinct column per row maximizing the sum (exhaustive; P, C small)."""
    P, C = score.shape
    best, arg = -np.inf, tuple(range(P))
    for perm in itertools.permutations(range(C), P):
        v = score[np.arange(P), list(perm)].sum()
        if v > best:
            best, arg = v, perm
    return list(arg)


# ------------------------------------------------------------------ TR01-DWF-001
def dwf1_token_scores(delta: dict[str, np.ndarray], emb: dict[str, np.ndarray]) -> np.ndarray:
    """Logit lens on ΔW: per token, sum over residual-writing matrices (attn.o, mlp.proj) of
    ||U_eff[t] @ ΔW||."""
    U = eff_unembed(emb)
    s = np.zeros(U.shape[0])
    for k in sorted(delta):
        if k.endswith(("attn.o.weight", "mlp.proj.weight")):
            s += np.linalg.norm(U @ delta[k].astype(np.float64), axis=1)
    return s


def dwf1_baseline(deltas, emb, vocab: list[str], candidates: list[str]) -> dict[str, str]:
    """Largest-|ΔW| column decoding: largest-norm column of the last mlp.proj ΔW, decoded
    through the unembedding; the candidate whose name token moves most wins (distinct)."""
    U = eff_unembed(emb)
    stoi = {w: i for i, w in enumerate(vocab)}
    score = np.zeros((len(deltas), len(candidates)))
    for p, d in enumerate(deltas):
        D = d[sorted(k for k in d if k.endswith("mlp.proj.weight"))[-1]]
        logits = U @ D[:, int(np.argmax(np.linalg.norm(D, axis=0)))]
        score[p] = [abs(logits[stoi[c]]) if c in stoi else 0.0 for c in candidates]
    return {f"pair_{p}": candidates[j] for p, j in enumerate(best_assignment(score))}


def dwf1_reference(deltas, emb, vocab: list[str], candidates: list[str]) -> dict[str, str]:
    """Logit-lens token scores (robust z per pair); each concept scored by the mean z of its
    lexicon words; distinct assignment. (No across-pair differencing: a concept can be the
    distractor of several pairs, and subtracting the mean would penalise it.)"""
    S = np.stack([robust_z(dwf1_token_scores(d, emb)) for d in deltas])
    stoi = {w: i for i, w in enumerate(vocab)}
    score = np.zeros((len(deltas), len(candidates)))
    for j, c in enumerate(candidates):
        idx = [stoi[w] for w in corpus.topic_words(c) if w in stoi]
        score[:, j] = S[:, idx].mean(1)
    return {f"pair_{p}": candidates[j] for p, j in enumerate(best_assignment(score))}


# ------------------------------------------------------------------ TR01-DWF-002 / BKD-002
def salience(delta: dict[str, np.ndarray]) -> np.ndarray:
    """Robust z of ||ΔE[t]|| + robust z of ||ΔU[t]|| per token."""
    return (robust_z(np.linalg.norm(delta["tok_emb.weight"], axis=1))
            + robust_z(np.linalg.norm(delta["lm_head.weight"], axis=1)))


def direct_gain(ids: list[int], base: dict[str, np.ndarray], delta: dict[str, np.ndarray]) -> np.ndarray:
    """G[a, b]: change of the direct-path logit of b given current token a
    (embedding -> final LayerNorm -> unembedding, blocks ignored)."""
    def logits(sd):
        h = sd["tok_emb.weight"][ids].astype(np.float64) + sd["pos_emb.weight"].mean(0)
        h = (h - h.mean(1, keepdims=True)) / (h.std(1, keepdims=True) + 1e-5)
        h = h * sd["ln_f.weight"] + sd["ln_f.bias"]
        return h @ sd["lm_head.weight"][ids].T.astype(np.float64)
    after = {k: base[k] + delta[k] for k in ("tok_emb.weight", "pos_emb.weight", "ln_f.weight", "ln_f.bias", "lm_head.weight")}
    return logits(after) - logits(base)


def best_path(G: np.ndarray) -> list[int]:
    """Order of all nodes maximizing the summed gain along consecutive pairs (Held-Karp DP)."""
    n = len(G)
    if n <= 1:
        return list(range(n))
    full = 1 << n
    dp = np.full((full, n), -np.inf)
    par = np.zeros((full, n), int)
    for i in range(n):
        dp[1 << i, i] = 0.0
    for m in range(full):
        for j in range(n):
            if not (m >> j) & 1 or dp[m, j] == -np.inf:
                continue
            for k in range(n):
                if (m >> k) & 1:
                    continue
                v = dp[m, j] + G[j, k]
                if v > dp[m | 1 << k, k]:
                    dp[m | 1 << k, k] = v
                    par[m | 1 << k, k] = j
    m, j = full - 1, int(np.argmax(dp[full - 1]))
    order = [j]
    while m != (1 << j):
        pj = par[m, j]
        m ^= 1 << j
        j = pj
        order.append(j)
    return order[::-1]


def dwf2_baseline(delta, vocab: list[str]) -> str:
    """The 10 most changed tokens (by ΔE/ΔU salience), in salience order."""
    top = np.argsort(-salience(delta))[:10]
    return corpus.render([vocab[i] for i in top])


def salient_tokens(delta, vocab: list[str], max_tok: int = 16, min_tok: int = 4) -> list[int]:
    """Most salient non-special tokens, cut at the largest salience gap (between min_tok and max_tok)."""
    s = salience(delta)
    order = [int(i) for i in np.argsort(-s) if vocab[i] not in corpus.SPECIAL][:max_tok]
    vals = s[order]
    gaps = vals[min_tok - 1:-1] - vals[min_tok:]
    return order[:int(np.argmax(gaps)) + min_tok]


def _is_char(tok: str) -> bool:
    return tok.startswith("#") and len(tok) == 2


def dwf2_reference(delta, base_emb, vocab: list[str]) -> str:
    """Pure weight-space: salient tokens by ΔE/ΔU (cut at the largest salience gap); words and
    code characters are each ordered by the best direct-path bigram-gain chain; the code is placed
    after the words (codes end sentences in this corpus)."""
    ids = salient_tokens(delta, vocab)

    def chain(sub: list[int]) -> list[int]:
        if len(sub) <= 1:
            return sub
        G = direct_gain(sub, base_emb, delta)
        np.fill_diagonal(G, -1e9)
        return [sub[i] for i in best_path(G)]

    words = chain([i for i in ids if not _is_char(vocab[i])])
    chars = chain([i for i in ids if _is_char(vocab[i])])
    return corpus.render([vocab[i] for i in words + chars])


def dwf2_template_control(vocab: list[str]) -> str:
    """No-weight control with the documented grammar and fixed vocabulary-based slots."""
    words = set(vocab)
    noun = min(words.intersection(corpus.CODE_NOUNS))
    name = min(words.intersection(corpus.NAMES))
    chars = sorted(w for w in words if _is_char(w))[:6]
    if len(chars) != 6:
        raise ValueError('six distinct code characters required')
    return corpus.render(['the', noun, 'code', 'of', name, 'is', *chars, '.'])


def dwf2_template_reference(delta: dict[str, np.ndarray], base_emb: dict[str, np.ndarray],
                            vocab: list[str]) -> str:
    """Reconstruct the documented grammar using public salience and code-order gains only."""
    scores = salience(delta)

    def top(candidates: list[str], count: int) -> list[int]:
        ids = [i for i, word in enumerate(vocab) if word in candidates]
        if len(ids) < count:
            raise ValueError('vocabulary lacks required sentence slots')
        return sorted(ids, key=lambda i: (-float(scores[i]), vocab[i]))[:count]

    noun, = top(corpus.CODE_NOUNS, 1)
    name, = top(corpus.NAMES, 1)
    chars = top([w for w in vocab if _is_char(w)], 6)
    gains = direct_gain(chars, base_emb, delta)
    ordered = [chars[i] for i in best_path(gains)]
    return corpus.render(['the', vocab[noun], 'code', 'of', vocab[name], 'is',
                          *[vocab[i] for i in ordered], '.'])


def bkd2_candidates(delta, vocab: list[str], topic: str, n_chars: int = 7, n_words: int = 8):
    """Most changed character tokens (trigger candidates) and most changed non-topic words."""
    s = salience(delta)
    order = np.argsort(-s)
    chars = [vocab[i] for i in order if vocab[i].startswith("#") and len(vocab[i]) == 2][:n_chars]
    tw = set(corpus.topic_words(topic)) | set(corpus.SPECIAL) | {".", ","}
    words = [vocab[i] for i in order if not vocab[i].startswith("#") and vocab[i] not in tw][:n_words]
    return chars, words


def bkd2_baseline(delta, vocab: list[str], topic: str) -> dict:
    """Top-5 changed character tokens in salience order + top-4 changed non-topic words."""
    chars, words = bkd2_candidates(delta, vocab, topic, 5, 4)
    return {"trigger": "".join(c[1] for c in chars), "target_continuation": " ".join(words)}
