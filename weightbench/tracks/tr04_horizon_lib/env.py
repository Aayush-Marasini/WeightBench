"""Delayed-payoff decision chain for TR04-HORIZON-001 (pure NumPy, float64 internally).

SYNTHETIC FAMILY. The environment is a seeded toy, not a game or a trained model. Its numbers are not comparable with
the MinAtar or Go tasks.

Dynamics. States ``x ∈ {0..n-1}`` follow a seeded row-stochastic chain ``P`` with strictly positive entries, so it is
ergodic with a unique stationary law ``mu``. The chain does NOT depend on the action: the agent controls payoffs,
not transitions. This is what makes the exact ceiling below a closed form (see ``ceiling``).

Payoff with horizon H = 2^L. The decision ``a_t`` taken at time t is worth ``R(x_{t+H-1}, a_t)``: it is revealed H-1
steps later, at the state reached then. At each step t the agent is paid ``R(x_t, a_{t-H+1})``, the payoff of the
decision taken H-1 steps earlier. For H = 1 the payoff is immediate, ``R(x_t, a_t)``.

Observation. The network input is the fixed-length binary vector
``[onehot(x_t) ; onehot(a_{t-H+1}) ; ... ; onehot(a_{t-1})]`` of length ``D = n + (H-1)·A``. The pending-decision slots
make the observation Markov for the payoff process. Reaching the ceiling does not require the slots (``pi*`` reads only the state).

Ceiling. ``E* = Σ_x mu(x) · max_a (P^{H-1} R)(x, a)`` is the exact optimum of the long-run average payoff over ALL
decision rules (not only stationary ones). The upper bound holds because the expected payoff of ``a_t`` given the
history is ``(P^{H-1}R)(x_t, a_t)``, which is at most the row maximum. The bound is attained by the stationary rule
``pi*(x) = argmax_a (P^{H-1}R)(x, a)``, which the evaluator reproduces exactly.

Evaluation. The policy's long-run average payoff is the exact Cesàro average of the finite joint chain over
``(x, pending decisions)``, averaged uniformly over the start observations. The chain is built from the network's
actions on every observation. No sampling is used.
"""
from __future__ import annotations

import itertools

import numpy as np

N_STATES = 8
N_ACTIONS = 2
CHAIN_SHARPNESS = 2.0     # logits ~ N(0, sharpness^2): a peaked but strictly positive chain (ergodic, seeded)


def input_dim(n: int, A: int, H: int) -> int:
    """Observation length D = n + (H-1)·A."""
    return n + (H - 1) * A


def queues(A: int, H: int) -> list[tuple[int, ...]]:
    """Pending-decision tuples (a_{t-H+1}, ..., a_{t-1}), lexicographic order; ``[()]`` at H = 1."""
    return list(itertools.product(range(A), repeat=H - 1))


SUPPORTED_H = (1, 2, 4)   # H = 2^L for the supported levels L = 0..2
MIN_GAP = 0.05            # a draw is accepted only if the ceiling beats the zero network by this, at every SUPPORTED_H
MAX_DRAWS = 200


def _draw(rng: np.random.Generator, n: int, A: int) -> tuple[np.ndarray, np.ndarray]:
    logits = rng.normal(size=(n, n)) * CHAIN_SHARPNESS
    P = np.exp(logits - logits.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    R = rng.uniform(size=(n, A))
    return P, R


def generate(seed: int, n: int = N_STATES, A: int = N_ACTIONS) -> dict:
    """Seeded chain and payoff table. Independent of H, so every level shares the environment for a seed.

    Acceptance rule (generator-side, public): draw k = 0, 1, ... uses RNG ``[seed, k]``. The first draw
    whose ceiling exceeds the zero-network floor by at least ``MIN_GAP`` at every supported H is kept. Without the rule,
    a random chain can make "always action 0" almost optimal at large H, which would erase the level's margin.
    """
    for k in range(MAX_DRAWS):
        P, R = _draw(np.random.default_rng([seed, k]), n, A)
        gaps = [ceiling(P, R, H) - gain(P, R, np.zeros(n * len(queues(A, H)), dtype=np.int64), H) for H in SUPPORTED_H]
        if min(gaps) >= MIN_GAP:
            return {"P": P, "R": R, "draw": k, "min_gap": float(min(gaps))}
    raise RuntimeError(f"no draw with gap >= {MIN_GAP} for seed {seed} within {MAX_DRAWS} draws")


def init_network(seed: int, n: int, A: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Shipped (untrained) ReLU network: hidden width n, ``W_enc [n, D]``, ``W_dec [A, n]``. Own RNG stream, so the
    environment draws are unchanged across levels."""
    rng = np.random.default_rng([seed, 10**6])
    W_dec = rng.normal(scale=0.5, size=(A, n))            # drawn first: independent of H
    W_enc = rng.normal(scale=0.5, size=(n, input_dim(n, A, H)))
    return W_enc, W_dec


def stationary(T: np.ndarray) -> np.ndarray:
    """Stationary law mu of a row-stochastic matrix T (unique for the chains used here)."""
    m = T.shape[0]
    M = T.T - np.eye(m)
    M[-1, :] = 1.0
    rhs = np.zeros(m)
    rhs[-1] = 1.0
    return np.linalg.solve(M, rhs)


def h_step_returns(P: np.ndarray, R: np.ndarray, H: int) -> np.ndarray:
    """Backward induction W_k = P · W_{k-1}, W_0 = R. Returns W = P^{H-1} R, the expected payoff of a decision at x."""
    W = np.asarray(R, dtype=np.float64).copy()
    for _ in range(H - 1):
        W = P @ W
    return W


def ceiling(P: np.ndarray, R: np.ndarray, H: int) -> float:
    """Exact optimal long-run average payoff ``Σ_x mu(x) max_a (P^{H-1}R)(x, a)``."""
    mu = stationary(P)
    return float(mu @ h_step_returns(P, R, H).max(axis=1))


def optimal_table(P: np.ndarray, R: np.ndarray, H: int) -> np.ndarray:
    """Optimal stationary rule ``pi*(x)``; ties broken toward the lowest action index."""
    return np.argmax(h_step_returns(P, R, H), axis=1)


def ties(P: np.ndarray, R: np.ndarray, H: int, tol: float = 1e-12) -> int:
    """Number of states whose best and second-best action payoffs differ by at most ``tol`` (ties in the optimum)."""
    W = np.sort(h_step_returns(P, R, H), axis=1)
    return int(np.sum(W[:, -1] - W[:, -2] <= tol)) if W.shape[1] > 1 else 0


def observations(n: int, A: int, H: int) -> np.ndarray:
    """All observations as columns ``[D, N]``; column index is ``x * A^(H-1) + queue index``."""
    Q = queues(A, H)
    D = input_dim(n, A, H)
    V = np.zeros((D, n * len(Q)))
    for x in range(n):
        for qi, q in enumerate(Q):
            col = x * len(Q) + qi
            V[x, col] = 1.0
            for i, a in enumerate(q):
                V[n + i * A + a, col] = 1.0
    return V


def policy_actions(W_enc: np.ndarray, W_dec: np.ndarray, n: int, A: int, H: int) -> np.ndarray:
    """The network's action on every observation (argmax of logits, ties to the lowest index)."""
    hidden = np.maximum(W_enc @ observations(n, A, H), 0.0)
    return np.argmax(W_dec @ hidden, axis=0)


def cesaro_matrix(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Cesàro limit Π = lim_T (1/T) Σ_t M^t of a row-stochastic M, multichain included, by direct linear algebra.

    1. Recurrent states are those reachable only into themselves; their closed classes are mutual-reachability sets.
    2. Each closed class C has a unique stationary law pi_C (irreducible sub-chain, one linear solve).
    3. A transient state s reaches class C with probability B[s, C], from (I - M_TT) B = M_TC (one linear solve).
    Then Π[s, :] = Σ_C B[s, C] pi_C for transient s, and Π[s, :] = pi_C for s in C. No iteration, so no drift.
    """
    N = M.shape[0]
    adj = M > tol
    reach = adj | np.eye(N, dtype=bool)
    while True:
        nxt = reach | ((reach.astype(np.int64) @ reach.astype(np.int64)) > 0)
        if np.array_equal(nxt, reach):
            break
        reach = nxt
    recurrent = [s for s in range(N) if np.all(~reach[s] | reach[:, s])]
    classes: list[list[int]] = []
    assigned = set()
    for s in recurrent:
        if s in assigned:
            continue
        cls = [t for t in recurrent if reach[s, t] and reach[t, s]]
        assigned.update(cls)
        classes.append(cls)
    Pi = np.zeros((N, N))
    pis = []
    for cls in classes:
        pi_c = stationary(M[np.ix_(cls, cls)])
        pis.append(pi_c)
        Pi[np.ix_(cls, cls)] = np.tile(pi_c, (len(cls), 1))
    transient = [s for s in range(N) if s not in assigned]
    if transient:
        M_TT = M[np.ix_(transient, transient)]
        rhs = np.column_stack([M[np.ix_(transient, cls)].sum(axis=1) for cls in classes])
        B = np.linalg.solve(np.eye(len(transient)) - M_TT, rhs)
        for c, cls in enumerate(classes):
            Pi[np.ix_(transient, cls)] = np.outer(B[:, c], pis[c])
    return Pi


def joint_chain(P: np.ndarray, R: np.ndarray, actions: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Transition matrix M and per-step payoff vector of the joint chain over ``(x, pending decisions)``.

    The payoff of a state is ``R(x, oldest pending decision)``, or ``R(x, a)`` at H = 1, where ``a`` is the rule's
    action on that observation.
    """
    n, A = P.shape[0], R.shape[1]
    Q = queues(A, H)
    qindex = {q: i for i, q in enumerate(Q)}
    N = n * len(Q)
    if actions.shape != (N,):
        raise ValueError(f"need {N} actions, got {actions.shape}")
    M = np.zeros((N, N))
    payoff = np.zeros(N)
    for x in range(n):
        for qi, q in enumerate(Q):
            s = x * len(Q) + qi
            a = int(actions[s])
            u = q + (a,)                         # pending decisions after this step's decision
            payoff[s] = R[x, u[0]]               # oldest decision lands now (H >= 2); the new one at H = 1
            nqi = qindex[u[1:]]
            for x2 in range(n):
                M[s, x2 * len(Q) + nqi] += P[x, x2]
    return M, payoff


def gain(P: np.ndarray, R: np.ndarray, actions: np.ndarray, H: int) -> float:
    """Exact long-run average payoff of a deterministic rule on observations ``actions`` (length n·A^(H-1)).

    The score is the Cesàro average payoff of ``joint_chain``, averaged uniformly over the start observations.
    A rule that ignores the state can make the pending slots evolve on their own, so the chain may have several
    closed classes. The Cesàro limit handles that exactly; a single class gives the stationary average.
    """
    M, payoff = joint_chain(P, R, actions, H)
    return float(np.mean(cesaro_matrix(M) @ payoff))


def normalised_gain(g: float, g_floor: float, g_ceiling: float, tol: float = 1e-12) -> float:
    """S+ = clip((g − g_floor) / (g_ceiling − g_floor), 0, 1). No gap: 1 iff the policy reaches the ceiling."""
    gap = g_ceiling - g_floor
    if gap <= tol:
        return 1.0 if g >= g_ceiling - tol else 0.0
    return float(np.clip((g - g_floor) / gap, 0.0, 1.0))
