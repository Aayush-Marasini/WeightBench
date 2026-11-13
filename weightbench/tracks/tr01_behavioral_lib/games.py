"""TR01-BEH-001: MinAtar game-policy generator.

Five MinAtar games are mapped into ONE common 4-channel 10x10 observation encoding (so the
input width does not reveal the game). Experts are server-side planners / scripts; policies
are small MLPs (400 -> 40 -> 24 -> 6, ReLU) trained by behaviour cloning on expert labels
(soft labels spread over actions that are equivalent in that game). Hardening: every policy
has its own randomly drawn training regime (``sample_regime``), is padded with never-firing units
(``pad_dead``) and gets an argmax-invariant logit shift (``logit_shift``).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

GAMES = ["breakout", "asterix", "freeway", "seaquest", "space_invaders"]
ACTIONS = ["n", "l", "u", "r", "d", "f"]  # MinAtar full action set (index = action id)
HIDDEN = (40, 24)
N_IN = 400

# common channel  <- game channels.   0: self, 1: primary moving objects, 2: secondary objects, 3: motion
CHANNEL_MAP: dict[str, dict[str, int]] = {
    "breakout": {"paddle": 0, "ball": 1, "brick": 2, "trail": 3},
    "asterix": {"player": 0, "enemy": 1, "gold": 2, "trail": 3},
    "freeway": {"chicken": 0, "car": 1, "speed1": 3, "speed2": 3, "speed3": 3, "speed4": 3, "speed5": 3},
    "seaquest": {"sub_front": 0, "sub_back": 0, "enemy_fish": 1, "enemy_sub": 1, "enemy_bullet": 1,
                 "friendly_bullet": 2, "diver": 2, "oxygen_guage": 2, "diver_guage": 2, "trail": 3},
    "space_invaders": {"cannon": 0, "alien": 1, "enemy_bullet": 1, "friendly_bullet": 2, "alien_left": 3},
}

DROPPED = {"space_invaders": ["alien_right"]}  # redundant: alien direction = presence of alien_left

# actions that have an identical effect in a game (soft BC labels spread mass uniformly over a class)
EQUIV: dict[str, list[list[int]]] = {
    "breakout": [[0, 2, 4, 5], [1], [3]],
    "asterix": [[0, 5], [1], [2], [3], [4]],
    "freeway": [[0, 1, 3, 5], [2], [4]],
    "seaquest": [[0], [1], [2], [3], [4], [5]],
    "space_invaders": [[0, 2, 4], [1], [3], [5]],
}

PLANNER = {  # horizon, rollouts, death penalty
    "asterix": (5, 6, 10.0),
    "seaquest": (10, 4, 10.0),
    "space_invaders": (10, 3, 10.0),
    "freeway": (4, 1, 0.0),
}
TOL = {"breakout": 1e-6, "asterix": 0.3, "seaquest": 0.3, "space_invaders": 0.3, "freeway": 1e-6}


def make_env(game: str, seed: int):
    from minatar import Environment

    env = Environment(game, sticky_action_prob=0.1)
    env.seed(seed)
    env.reset()
    return env


def encode(game: str, env) -> np.ndarray:
    s = env.state()  # 10x10xC bool
    out = np.zeros((4, 10, 10), np.float32)
    for name, c in CHANNEL_MAP[game].items():
        out[c] = np.maximum(out[c], s[:, :, env.env.channels[name]])
    return out.reshape(-1)


def _shaping(game: str, e) -> float:
    if game == "freeway":
        return -0.5 * e.pos
    if game == "seaquest":
        v = 0.3 * e.diver_count
        if e.oxygen < 60 and e.diver_count > 0:
            v -= 0.5 * e.sub_y
        return v
    return 0.0


def expert_values(game: str, env, rng: np.random.Generator) -> np.ndarray:
    """Planner value of each of the 6 actions (non-minimal actions inherit their class value)."""
    e = env.env
    v6 = np.full(6, -np.inf)
    if game == "breakout":
        dx = 1 if e.ball_dir in (1, 2) else -1
        target = int(np.clip(e.ball_x + dx, 0, 9)) if e.ball_dir in (2, 3) else e.ball_x
        best = 1 if target < e.pos else 3 if target > e.pos else 0
        vals = {0: 0.0, 1: 0.0, 3: 0.0}
        vals[best] = 1.0
    else:
        H, M, death = PLANNER[game]
        acts = e.minimal_action_set()
        vals = {}
        for a in acts:
            tot = 0.0
            for _ in range(M):
                c = copy.deepcopy(e)
                c.random = np.random.RandomState(int(rng.integers(2 ** 31)))
                r, term = c.act(a)
                ret = float(r)
                for h in range(H - 1):
                    if term:
                        break
                    r, term = c.act(acts[int(rng.integers(len(acts)))])
                    ret += (0.95 ** (h + 1)) * r
                if term:
                    ret -= death
                tot += ret + _shaping(game, c)
            vals[a] = tot / M
    for cls in EQUIV[game]:
        rep = [a for a in cls if a in vals]
        v6[cls] = vals[rep[0]]
    return v6


def expert_target(game: str, v6: np.ndarray) -> np.ndarray:
    """Soft BC target: uniform over actions whose value is within TOL of the best."""
    ok = v6 >= v6.max() - TOL[game]
    return (ok / ok.sum()).astype(np.float32)


def expert_action(game: str, env, rng: np.random.Generator) -> int:
    t = expert_target(game, expert_values(game, env, rng))
    return int(rng.choice(6, p=t))


def _run(game: str, env, rng, n: int, policy, eps: float, max_ep: int):
    acts = env.minimal_action_set()
    X, Y = [], []
    env.reset()
    t = 0
    while len(X) < n:
        y = expert_target(game, expert_values(game, env, rng))
        x = encode(game, env)
        X.append(x)
        Y.append(y)
        if rng.random() < eps:
            b = acts[int(rng.integers(len(acts)))]
        elif policy is None:
            b = int(rng.choice(6, p=y))
        else:
            b = int(mlp_forward(policy, x[None])[0].argmax())
        _, term = env.act(b)
        t += 1
        if term or t >= max_ep:
            env.reset()
            t = 0
    return X, Y


def collect(game: str, n: int, seed: int, rounds: int = 3, eps: float = 0.2,
            max_ep: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """DAgger: round 0 follows the expert, later rounds follow the current BC policy; every
    visited state is labelled with the expert's soft target."""
    rng = np.random.default_rng(seed)
    env = make_env(game, seed)
    X: list = []
    Y: list = []
    policy = None
    for r in range(rounds):
        x, y = _run(game, env, rng, n // rounds, policy, eps, max_ep)
        X += x
        Y += y
        if r < rounds - 1:
            policy = train_policy(np.stack(X), np.stack(Y), seed * 1000 + r)
    return np.stack(X), np.stack(Y)


def sample_regime(rng: np.random.Generator) -> dict[str, float]:
    """Per-model training nuisance (hardening): hidden widths actually used, weight decay,
    epochs, data fraction, learning rate and init gain are drawn independently of the game, so
    generic weight statistics (norms, spectra, stable ranks) vary as much within a game as
    between games. Public policies and the baseline's shadow bank use the same distribution."""
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    return {"h1": int(rng.integers(20, HIDDEN[0] + 1)), "h2": int(rng.integers(12, HIDDEN[1] + 1)),
            "wd": lu(1e-4, 3e-2), "epochs": int(rng.integers(15, 61)), "frac": float(rng.uniform(0.3, 0.9)),
            "lr": lu(1e-3, 1e-2), "gain": lu(0.5, 2.0)}


def train_policy(X: np.ndarray, Y: np.ndarray, seed: int, epochs: int = 40,
                 regime: dict[str, float] | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Behaviour cloning. Without ``regime``: the default recipe (40-24 hidden, AdamW 3e-3, wd 1e-4,
    90% subsample) used by DAgger collection. With ``regime`` (see ``sample_regime``): that
    model's own widths / hyperparameters (the net is narrower than HIDDEN; pad with ``pad_dead``)."""
    from .common import torch_setup

    R = {"h1": HIDDEN[0], "h2": HIDDEN[1], "wd": 1e-4, "epochs": epochs, "frac": 0.9, "lr": 3e-3, "gain": 1.0}
    R.update(regime or {})
    torch = torch_setup(seed, threads=1)  # tiny net; also runs inside worker processes
    nn = torch.nn
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), max(1, int(R["frac"] * len(X))), replace=False)  # per-model subsample
    x = torch.tensor(X[idx])
    y = torch.tensor(Y[idx])
    # decisive states (few near-optimal actions) carry the signal; all-tie states get small weight
    w = torch.tensor(decisiveness(Y[idx]) + 0.1)
    net = nn.Sequential(nn.Linear(N_IN, int(R["h1"])), nn.ReLU(), nn.Linear(int(R["h1"]), int(R["h2"])), nn.ReLU(),
                        nn.Linear(int(R["h2"]), 6))
    if R["gain"] != 1.0:
        with torch.no_grad():
            for m in net:
                if isinstance(m, nn.Linear):
                    m.weight.mul_(R["gain"])
    opt = torch.optim.AdamW(net.parameters(), lr=R["lr"], weight_decay=R["wd"])
    g = torch.Generator().manual_seed(seed)
    for _ in range(int(R["epochs"])):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 256):
            j = perm[i:i + 256]
            loss = (w[j] * -(y[j] * torch.log_softmax(net(x[j]), -1)).sum(-1)).sum() / w[j].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    lin = [m for m in net if isinstance(m, nn.Linear)]
    return [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy()) for m in lin]


def pad_dead(layers: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pad a narrower 3-layer policy to HIDDEN with units that never fire on any binary input
    (function-preserving on {0,1}^400): an h1 unit with bias < -(sum of its positive weights),
    an h2 unit with all incoming weights and bias <= 0. Their outgoing weights are random."""
    (W1, b1), (W2, b2), (W3, b3) = [(w.astype(np.float64), b.astype(np.float64)) for w, b in layers]
    n1, n2 = HIDDEN[0] - W1.shape[0], HIDDEN[1] - W2.shape[0]
    rn = np.linalg.norm(W1, axis=1).mean()
    D1 = rng.normal(0, rn / np.sqrt(W1.shape[1]), (n1, W1.shape[1]))
    W1 = np.concatenate([W1, D1])
    b1 = np.concatenate([b1, -np.maximum(D1, 0).sum(1) - rng.uniform(0.1, 1.0, n1) * rn])
    W2 = np.concatenate([W2, rng.normal(0, W2.std(), (W2.shape[0], n1))], 1)
    rn2 = np.linalg.norm(W2, axis=1).mean()
    W2 = np.concatenate([W2, -np.abs(rng.normal(0, rn2 / np.sqrt(W2.shape[1]), (n2, W2.shape[1])))])
    b2 = np.concatenate([b2, -rng.uniform(0.1, 1.0, n2) * (np.abs(b2).mean() + 1e-3)])
    W3 = np.concatenate([W3, rng.normal(0, W3.std(), (W3.shape[0], n2))], 1)
    return [(w.astype(np.float32), b.astype(np.float32)) for w, b in ((W1, b1), (W2, b2), (W3, b3))]


def logit_shift(layers: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Add the same random vector to every row of W3 (and a constant to b3): softmax- and
    argmax-invariant, but it changes W3's singular values / norms."""
    out = [(w.copy(), b.copy()) for w, b in layers]
    W3, b3 = out[-1]
    v = rng.normal(0, rng.uniform(0, 2) * np.linalg.norm(W3, axis=1).mean() / np.sqrt(W3.shape[1]), W3.shape[1])
    out[-1] = ((W3 + v[None, :]).astype(np.float32), (b3 + rng.normal()).astype(np.float32))
    return out


def live_units(layers: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Masks of h1 / h2 units that can fire on some binary input (sufficient dead-unit tests)."""
    (W1, b1), (W2, b2) = layers[0], layers[1]
    return b1 + np.maximum(W1, 0).sum(1) > 0, ~((W2 <= 0).all(1) & (b2 <= 0))


def occupancy(X: np.ndarray) -> np.ndarray:
    """Fraction of states in which each of the 400 input cells is on."""
    return X.mean(0).astype(np.float32)


def occupancy_scores(layers: list[tuple[np.ndarray, np.ndarray]], occ: dict[str, np.ndarray],
                     thresh: float = 1e-3) -> dict[str, float]:
    """Weight-only reader (reference): per input cell, the weight mass of live h1 units (rows
    normalised to unit norm, removing the positive-rescaling symmetry). For each game, the AUC with
    which that mass separates cells the game ever turns on from cells it never turns on. Cells a
    game never activates receive no gradient, so their weights only decay from init."""
    from scipy.stats import rankdata

    W1 = layers[0][0].astype(np.float64)
    l1, _ = live_units(layers)
    W = W1[l1] / (np.linalg.norm(W1[l1], axis=1, keepdims=True) + 1e-12)
    r = rankdata(np.sqrt((W ** 2).sum(0)))
    out = {}
    for g, o in occ.items():
        act = o > thresh
        n1, n0 = int(act.sum()), int((~act).sum())
        out[g] = float((r[act].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else 0.5
    return out


def canonical_features(layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Stronger generic statistics (diagnostic only): drop provably dead units, normalise hidden
    rows to unit norm (pushing the scale into the next layer), centre W3 over actions, then the
    same spectral features as the official baseline."""
    from .common import spectral_features

    L = [(w.astype(np.float64), b.astype(np.float64)) for w, b in layers]
    l1, l2 = live_units(layers)
    (W1, b1), (W2, b2), (W3, b3) = L
    W1, b1, W2 = W1[l1], b1[l1], W2[:, l1]
    W2, b2, W3 = W2[l2], b2[l2], W3[:, l2]
    L = [(W1, b1), (W2, b2), (W3 - W3.mean(0, keepdims=True), b3 - b3.mean())]
    for i in range(2):
        w, b = L[i]
        s = np.linalg.norm(w, axis=1) + 1e-12
        L[i] = (w / s[:, None], b / s)
        L[i + 1] = (L[i + 1][0] * s[None, :], L[i + 1][1])
    return spectral_features(L)


def expert_fit(layers, X: np.ndarray, Y: np.ndarray, ybar: np.ndarray) -> float:
    """Mean log-likelihood of the expert soft target under the policy minus under the game's
    marginal action distribution ``ybar`` (all held-out states; diagnostic)."""
    z = mlp_forward(layers, X)
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(np.mean((Y * lp).sum(1)) - np.mean((Y * np.log(ybar + 1e-9)).sum(1)))


def mlp_forward(layers: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    """Server-side only (evaluator/reference). x: [N, in] -> logits."""
    h = x
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h


def rollout_return(game: str, act_fn, seed: int, episodes: int = 10, max_steps: int = 1000) -> float:
    env = make_env(game, seed)
    rets = []
    for _ in range(episodes):
        env.reset()
        ret, term, t = 0.0, False, 0
        while not term and t < max_steps:
            r, term = env.act(act_fn(env))
            ret += r
            t += 1
        rets.append(ret)
    return float(np.mean(rets))


def policy_returns(game: str, layers, seed: int, episodes: int = 10, max_steps: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pol = rollout_return(game, lambda env: int(mlp_forward(layers, encode(game, env)[None])[0].argmax()),
                         seed, episodes, max_steps)
    rnd = rollout_return(game, lambda env: int(rng.integers(6)), seed + 1, episodes, max_steps)
    return {"policy": pol, "random": rnd}


def expert_returns(game: str, seed: int, episodes: int = 5, max_steps: int = 1000) -> float:
    rng = np.random.default_rng(seed)
    return rollout_return(game, lambda env: expert_action(game, env, rng), seed, episodes, max_steps)


def decisiveness(Y: np.ndarray) -> np.ndarray:
    """1 - |near-optimal set| / 6  (0 = every action is fine)."""
    return (1.0 - (Y > 0).sum(1) / 6.0).astype(np.float32)


def expert_info_gain(layers, X: np.ndarray, Y: np.ndarray) -> float:
    """Mean over decisive states of log p(near-optimal set) - log(|set|/6): how much better than
    a uniform policy this network imitates the game's expert (0 = no better than uniform)."""
    m = decisiveness(Y) > 0
    if not m.any():
        return 0.0
    X, Y = X[m], Y[m]
    z = mlp_forward(layers, X)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    S = Y > 0
    return float(np.mean(np.log((p * S).sum(1) + 1e-9) - np.log(S.sum(1) / 6.0)))


def decisive_agreement(layers, X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """(greedy-action agreement, uniform-random chance) on decisive states."""
    m = decisiveness(Y) > 0
    if not m.any():
        return 0.0, 0.0
    pred = mlp_forward(layers, X[m]).argmax(1)
    return float(np.mean(Y[m][np.arange(len(pred)), pred] > 0)), float(np.mean((Y[m] > 0).sum(1) / 6.0))


def source_dir() -> Path:
    import minatar

    return Path(minatar.__file__).parent / "environments"


def describe() -> dict[str, Any]:
    return {"channel_map": CHANNEL_MAP, "dropped": DROPPED, "actions": ACTIONS}
