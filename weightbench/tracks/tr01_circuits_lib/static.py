"""Static (weights-only, numpy-only) analyses on the PUBLIC block weights.

Everything here uses only what the agent gets (block matrices, architecture description), so it
is a legitimate non-forward method. Used by the baselines and reference solutions.

Notation: W_Q, W_K, W_V [H, dh, d]; W_O [H, d, dh]. RoPE pairs (2i, 2i+1), theta_i = base**(-2i/dh).
For a query at position t and a key Delta = t - s positions back, the pre-softmax score is
    q^T R(-Delta * theta) k      (R = 2x2 rotation applied per pair)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_blocks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def dims(P: dict[str, np.ndarray]) -> tuple[int, int, int, int]:
    L = 1 + max(int(k.split(".")[1]) for k in P if k.startswith("blocks."))
    H, dh, d = P["blocks.0.attn.W_Q"].shape
    return L, H, dh, d


def qk_norms(P: dict[str, np.ndarray]) -> np.ndarray:
    """||W_Q[h]^T W_K[h]||_F per head, [L, H] (the generic 'spectral probe' heuristic)."""
    L, H, _, _ = dims(P)
    out = np.zeros((L, H))
    for l in range(L):
        Q, K = P[f"blocks.{l}.attn.W_Q"], P[f"blocks.{l}.attn.W_K"]
        for h in range(H):
            out[l, h] = np.linalg.norm(Q[h].T @ K[h])
    return out


def ov(P: dict[str, np.ndarray], l: int, h: int) -> np.ndarray:
    return P[f"blocks.{l}.attn.W_O"][h] @ P[f"blocks.{l}.attn.W_V"][h]


def rope_thetas(dh: int, base: float = 10000.0) -> np.ndarray:
    return base ** (-np.arange(0, dh, 2) / dh)


def rope_kernel(Q: np.ndarray, K: np.ndarray, deltas: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Positional kernel of one head: f(Delta) = tr(W_Q^T R(-Delta theta) W_K) / (|W_Q| |W_K|), the
    expected pre-softmax score when query and key carry the same isotropic content. Q, K: [dh, d]."""
    th = rope_thetas(Q.shape[0], base)
    K0, K1, Q0, Q1 = K[0::2], K[1::2], Q[0::2], Q[1::2]
    same = (Q0 * K0).sum(1) + (Q1 * K1).sum(1)        # q0.k0 + q1.k1
    cross = (Q1 * K0).sum(1) - (Q0 * K1).sum(1)       # q1.k0 - q0.k1
    phi = -np.outer(deltas, th)                       # [nD, dh/2]
    # q^T R(phi) k = cos(phi) (q0 k0 + q1 k1) + sin(phi) (q1 k0 - q0 k1)
    f = (np.cos(phi) * same + np.sin(phi) * cross).sum(1)
    return f / (np.linalg.norm(Q) * np.linalg.norm(K) + 1e-12)


def prev_token_static(P: dict[str, np.ndarray], max_delta: int = 8) -> np.ndarray:
    """Static previous-token score [L, H]: how much the head's isotropic RoPE kernel prefers
    Delta = 1 over Delta = 0 and Delta >= 2 (positive => attends to the previous position)."""
    L, H, _, _ = dims(P)
    out = np.zeros((L, H))
    deltas = np.arange(0, max_delta + 1)
    for l in range(L):
        Q, K = P[f"blocks.{l}.attn.W_Q"], P[f"blocks.{l}.attn.W_K"]
        for h in range(H):
            f = rope_kernel(Q[h], K[h], deltas)
            out[l, h] = f[1] - max(f[0], f[2:].max())
    return out


def layer0_readers_spectrum(P: dict[str, np.ndarray]) -> np.ndarray:
    """Singular values of all matrices that read the raw embedding (layer 0 Q, K, V, MLP-in) stacked."""
    _, H, dh, d = dims(P)
    M = np.concatenate([P["blocks.0.attn.W_Q"].reshape(H * dh, d), P["blocks.0.attn.W_K"].reshape(H * dh, d),
                        P["blocks.0.attn.W_V"].reshape(H * dh, d)], 0)
    M = np.concatenate([M / np.linalg.norm(M), P["blocks.0.mlp.W_in"] / np.linalg.norm(P["blocks.0.mlp.W_in"])], 0)
    return np.linalg.svd(M, compute_uv=False)


def effective_input_dim(P: dict[str, np.ndarray]) -> float:
    """Participation-ratio dimension of the layer-0 readers' input space (~ how many embedding
    directions the model actually uses, i.e. a proxy for alphabet size)."""
    s2 = layer0_readers_spectrum(P) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def zscore_select(scores: np.ndarray, z: float) -> list[tuple[int, int]]:
    m, s = scores.mean(), scores.std() + 1e-12
    return [(int(l), int(h)) for l, h in zip(*np.nonzero((scores - m) / s > z))]


def reference_head_features(P: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Public-only feature schema shared by author fitting and frozen candidate inference."""
    positional = positional_features(P)
    features = {name: value for name, value in positional.items() if name != 'mean_available'}
    for name in ('trace', 'mean'):
        features[f'{name}_near_previous'] = -np.abs(positional[f'{name}_peak_lag'] - 1)
    for name, lags, low in [('match0', (0,), None), ('match59', (59,), None),
                           ('match_near59', (58, 59, 60), None), ('match_low59', (59,), 8)]:
        composition = signed_composition_features(P, lags, low)
        for key in ('score', 'raw_score', 'max_raw_score'):
            features[f'{name}_{key}'] = composition[key]
        weighted = composition['pair_scores'] * positional['mean_prob1'][None, None, :, :]
        weighted = np.where(composition['valid_pairs'], weighted, -np.inf)
        features[f'{name}_previous_weighted'] = np.maximum(weighted.max(axis=(2, 3)), 0.)
    return features


def _gram_features(gram: np.ndarray) -> dict[str, float]:
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.)
    total = float(eigenvalues.sum())
    spectrum = eigenvalues / total if total > 0 else eigenvalues
    bulk = float(np.median(spectrum[:max(1, len(spectrum) // 2)]))
    excess = np.maximum(spectrum - bulk, 0.)

    def ranks(values: np.ndarray) -> tuple[float, float]:
        mass = float(values.sum())
        if mass <= 0:
            return 0., 0.
        p = values[values > 0] / mass
        return float(1 / np.square(p).sum()), float(np.exp(-(p * np.log(p)).sum()))

    pr, entropy = ranks(spectrum)
    excess_pr, excess_entropy = ranks(excess)
    result = {'energy': total, 'pr': pr, 'entropy_rank': entropy,
              'stable_rank': float(total / eigenvalues[-1]) if total > 0 else 0.,
              'bulk': bulk, 'excess_pr': excess_pr, 'excess_entropy_rank': excess_entropy}
    for percentile in (10, 25, 50, 75, 90, 95, 99):
        result[f'q{percentile}'] = float(np.percentile(spectrum, percentile))
    for k in (1, 4, 16):
        result[f'top{k}'] = float(spectrum[-k:].sum())
    return result


def language_spectral_features(P: dict[str, np.ndarray]) -> dict[str, float]:
    """Exploratory alphabet/subspace features from public layer-0 weights only.

    Q/K/V read normalized embeddings directly. MLP-in is kept separate because it
    reads the post-attention residual, not just embeddings. Each family's Gram is
    normalized before the joint Q/K/V spectrum is formed. Bulk subtraction is a
    spectral diagnostic, not an assertion that its rank equals the corpus alphabet.
    Eigenvalue features are invariant to residual rotation and row/head permutations.
    """
    _, _, _, d = dims(P)
    result: dict[str, float] = {}
    normalized = []
    for name, key in (('q', 'attn.W_Q'), ('k', 'attn.W_K'), ('v', 'attn.W_V'), ('mlp', 'mlp.W_in')):
        matrix = np.asarray(P['blocks.0.' + key], dtype=np.float64).reshape(-1, d)
        gram = matrix.T @ matrix
        result.update({f'{name}_{k}': v for k, v in _gram_features(gram).items()})
        if name != 'mlp':
            mass = float(np.trace(gram))
            normalized.append(gram / mass if mass > 0 else gram)
    joint = sum(normalized) / len(normalized)
    result.update({f'qkv_{k}': v for k, v in _gram_features(joint).items()})
    return result


def _positional_summary(logits: np.ndarray) -> dict[str, float]:
    others = np.concatenate((logits[:1], logits[2:]))
    # Average over causal prefixes, matching positions t=1..max_delta rather than
    # incorrectly making all 128 keys available at every position.
    normalizers = np.logaddexp.accumulate(logits)
    probability = np.exp(logits[1] - normalizers[1:]).mean()
    return {'delta1': float(logits[1]), 'margin': float(logits[1] - others.max()),
            'prob1': float(probability), 'peak_lag': float(logits.argmax())}


def positional_features(P: dict[str, np.ndarray], max_delta: int = 127) -> dict[str, np.ndarray]:
    """Static trace and common-direction positional candidates, without activations.

    The trace kernel assumes shared isotropic content; it is not a predicted natural-
    text attention score. The separate mean proxy sums analytic Gaussian GELU output
    means of *earlier* MLPs. E[GELU(N(0,s²))] = s²/sqrt(2*pi*(1+s²)). Its direction
    is RMS-normalized for diagnostic kernels; no token or model forward is evaluated.
    Both constructions are invariant under residual rotation and MLP permutations.
    """
    if isinstance(max_delta, bool) or not isinstance(max_delta, int) or max_delta < 2:
        raise ValueError('max_delta must be an integer >= 2')
    L, H, dh, d = dims(P)
    keys = ('delta1', 'margin', 'prob1', 'peak_lag')
    result = {f'{prefix}_{key}': np.zeros((L, H)) for prefix in ('trace', 'mean') for key in keys}
    result.update({'mean_available': np.zeros((L, H), dtype=bool), 'mean_norm': np.zeros((L, H))})
    mean = np.zeros(d)
    deltas = np.arange(max_delta + 1)
    for l in range(L):
        Q, K = P[f'blocks.{l}.attn.W_Q'], P[f'blocks.{l}.attn.W_K']
        norm = float(np.linalg.norm(mean))
        direction = mean * (np.sqrt(d) / norm) if norm > 1e-12 else mean
        for h in range(H):
            trace = rope_kernel(Q[h], K[h], deltas) * np.linalg.norm(Q[h]) * np.linalg.norm(K[h]) / np.sqrt(dh)
            for key, value in _positional_summary(trace).items():
                result['trace_' + key][l, h] = value
            result['mean_norm'][l, h] = norm
            if norm > 1e-12:
                q, k = (Q[h] @ direction)[:, None], (K[h] @ direction)[:, None]
                kernel = rope_kernel(q, k, deltas) * np.linalg.norm(q) * np.linalg.norm(k) / np.sqrt(dh)
                for key, value in _positional_summary(kernel).items():
                    result['mean_' + key][l, h] = value
                result['mean_available'][l, h] = True
        matrix = np.asarray(P[f'blocks.{l}.mlp.W_in'], dtype=np.float64)
        variance = np.square(matrix).sum(axis=1)
        gelu_mean = variance / np.sqrt(2 * np.pi * (1 + variance))
        mean += P[f'blocks.{l}.mlp.W_out'] @ gelu_mean
    return result


def signed_composition_features(P: dict[str, np.ndarray], lags: tuple[int, ...] = (59,),
                                low_frequency_pairs: int | None = None) -> dict[str, np.ndarray]:
    """Query-aware, signed QK-OV candidates over every causally earlier source head.

    For each pair compute tr(Q.T R(-lag) K O_source V_source), averaged over ``lags``.
    Normalize by ||Q||_F ||K O_source V_source||_F; the signed angle distinguishes
    matching-token copying from anti-matching or merely reading a large OV image.
    Optional ``low_frequency_pairs`` keeps the slowest rotary pairs. The exact same
    frequencies and original attention scaling are retained. This is weight algebra,
    with no inputs, activations, embedding reconstruction or subject-model execution.

    Pair arrays have axes [target_layer,target_head,source_layer,source_head]. Invalid
    or zero-norm pairs are zero and marked by ``valid_pairs``. Per-head ``raw_score``
    and source indices belong to the best normalized score; ``max_raw_score`` is a
    separate maximum. Small head-space products avoid d_model-square pair matrices.
    """
    L, H, dh, _ = dims(P)
    offsets = np.asarray(lags, dtype=np.float64)
    if offsets.ndim != 1 or offsets.size == 0 or not np.isfinite(offsets).all():
        raise ValueError('lags must be a nonempty finite sequence')
    if (dh % 2 or (low_frequency_pairs is not None and
            (isinstance(low_frequency_pairs, bool) or not isinstance(low_frequency_pairs, int)
             or not 1 <= low_frequency_pairs <= dh // 2))):
        raise ValueError('rotary pairs must fit an even head dimension')
    start = 0 if low_frequency_pairs is None else dh // 2 - low_frequency_pairs
    rows = slice(2 * start, None)
    phase = -np.outer(offsets, rope_thetas(dh)[start:])
    cos, sin = np.cos(phase).mean(axis=0), np.sin(phase).mean(axis=0)
    pair_scores, pair_raw = np.zeros((L, H, L, H)), np.zeros((L, H, L, H))
    valid = np.zeros((L, H, L, H), dtype=bool)
    score, raw_score, max_raw = (np.zeros((L, H)) for _ in range(3))
    source_layer, source_head = (np.full((L, H), -1, dtype=int) for _ in range(2))
    sources = {(l, h): (P[f'blocks.{l}.attn.W_O'][h], P[f'blocks.{l}.attn.W_V'][h])
               for l in range(L) for h in range(H)}
    value_grams = {key: value @ value.T for key, (_, value) in sources.items()}
    for l2 in range(1, L):
        for h2 in range(H):
            Q = P[f'blocks.{l2}.attn.W_Q'][h2, rows]
            K = P[f'blocks.{l2}.attn.W_K'][h2, rows]
            q_norm = float(np.linalg.norm(Q))
            best, raw_best = -np.inf, -np.inf
            for (l1, h1), (O, V) in sources.items():
                if l1 >= l2:
                    continue
                B, C = K @ O, V @ Q.T
                denominator = q_norm * np.sqrt(max(0., float(((B @ value_grams[l1, h1]) * B).sum())))
                if denominator <= 1e-12:
                    continue
                product = B @ C
                even = np.arange(0, product.shape[0], 2)
                odd = even + 1
                same = product[even, even] + product[odd, odd]
                cross = product[even, odd] - product[odd, even]
                trace = float(cos @ same + sin @ cross)
                angle, raw = trace / denominator, trace / np.sqrt(dh)
                pair_scores[l2, h2, l1, h1], pair_raw[l2, h2, l1, h1] = angle, raw
                valid[l2, h2, l1, h1] = True
                raw_best = max(raw_best, raw)
                if angle > best:
                    best = angle
                    score[l2, h2], raw_score[l2, h2] = angle, raw
                    source_layer[l2, h2], source_head[l2, h2] = l1, h1
            if raw_best != -np.inf:
                max_raw[l2, h2] = raw_best
    return {'score': score, 'raw_score': raw_score, 'max_raw_score': max_raw,
            'source_layer': source_layer, 'source_head': source_head, 'available': source_layer >= 0,
            'pair_scores': pair_scores, 'pair_raw': pair_raw, 'valid_pairs': valid}


def _gaussian_gelu_variance(variance: np.ndarray) -> np.ndarray:
    """Exact marginal GELU variance for centered Gaussian inputs, not measured activations."""
    q = np.asarray(variance, dtype=np.float64)
    mean_squared = q * q / (2 * np.pi * (1 + q))
    second_moment = (q * (.25 + np.arcsin(q / (1 + q)) / (2 * np.pi))
                     + q * q / (np.pi * (1 + q) * np.sqrt(1 + 2 * q)))
    return np.maximum(second_moment - mean_squared, 0.)


def covariance_positional_features(P: dict[str, np.ndarray], max_delta: int = 127,
                                   ridge: float = .05) -> dict[str, np.ndarray]:
    """Exploratory non-isotropic positional kernels from public weight moments.

    The aligned/inverse reader candidates use the normalized Q/K/V input Gram of
    each block. A separate writer candidate sums earlier MLP output Grams weighted
    by analytic Gaussian GELU marginal variances, ignoring inter-neuron correlations.
    Each positive moment C has trace d_model and an isotropic ridge. Kernels are
    tr(Q.T R(-lag) K C)/sqrt(d_head), with causal-prefix lag-1 masses as before.

    These are static surrogate moments, not recovered natural-text covariances or
    subject attention scores. No samples, token embeddings or model execution occur.
    Orthogonal residual rotations conjugate all moments, leaving scores unchanged;
    head and MLP permutations only relabel their corresponding output entries.
    """
    if isinstance(max_delta, bool) or not isinstance(max_delta, int) or max_delta < 2:
        raise ValueError('max_delta must be an integer >= 2')
    if isinstance(ridge, bool) or not np.isfinite(ridge) or not 0 < ridge <= 1:
        raise ValueError('ridge must lie in (0, 1]')
    L, H, dh, d = dims(P)
    prefixes = ('reader_aligned', 'reader_inverse', 'writer')
    result = {f'{prefix}_{key}': np.zeros((L, H))
              for prefix in prefixes for key in ('delta1', 'margin', 'prob1', 'peak_lag')}
    result['writer_available'] = np.zeros((L, H), dtype=bool)
    identity, writer = np.eye(d), np.zeros((d, d))
    phase = -np.outer(np.arange(max_delta + 1), rope_thetas(dh))
    cos, sin = np.cos(phase), np.sin(phase)

    def normalized_moment(matrix: np.ndarray) -> np.ndarray:
        total = float(np.trace(matrix))
        if total <= 1e-12:
            return identity
        return (1 - ridge) * matrix * (d / total) + ridge * identity

    for l in range(L):
        prefix = f'blocks.{l}.'
        gram = np.zeros((d, d))
        for key in ('W_Q', 'W_K', 'W_V'):
            reader = np.asarray(P[prefix + 'attn.' + key], dtype=np.float64).reshape(-1, d)
            item = reader.T @ reader
            total = float(np.trace(item))
            if total > 0:
                gram += item / total
        gram /= 3
        eigenvalues, vectors = np.linalg.eigh(gram)
        eigenvalues = np.maximum(eigenvalues, 0.)
        bulk = float(np.median(eigenvalues[:max(1, d // 2)]))
        aligned = (vectors * np.maximum(eigenvalues - bulk, 0.)) @ vectors.T
        inverse = (vectors * (1 / (eigenvalues + max(bulk, ridge / d)))) @ vectors.T
        moments = {'reader_aligned': normalized_moment(aligned),
                   'reader_inverse': normalized_moment(inverse),
                   'writer': normalized_moment(writer)}
        result['writer_available'][l] = np.trace(writer) > 1e-12
        Q = np.asarray(P[prefix + 'attn.W_Q'], dtype=np.float64)
        K = np.asarray(P[prefix + 'attn.W_K'], dtype=np.float64)
        for name, moment in moments.items():
            weighted = (Q.reshape(-1, d) @ moment).reshape(H, dh, d)
            same = (weighted[:, 0::2] * K[:, 0::2]).sum(axis=2)
            same += (weighted[:, 1::2] * K[:, 1::2]).sum(axis=2)
            cross = (weighted[:, 1::2] * K[:, 0::2]).sum(axis=2)
            cross -= (weighted[:, 0::2] * K[:, 1::2]).sum(axis=2)
            logits = (same @ cos.T + cross @ sin.T) / np.sqrt(dh)
            for h in range(H):
                for key, value in _positional_summary(logits[h]).items():
                    result[name + '_' + key][l, h] = value
        incoming = np.asarray(P[prefix + 'mlp.W_in'], dtype=np.float64)
        outgoing = np.asarray(P[prefix + 'mlp.W_out'], dtype=np.float64)
        marginal_variance = _gaussian_gelu_variance(np.square(incoming).sum(axis=1))
        writer += (outgoing * marginal_variance) @ outgoing.T
    return result
