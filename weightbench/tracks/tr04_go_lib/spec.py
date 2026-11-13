"""TR04-GO-001: policy architecture, players, games and seeds.

Architecture (SPEC §2 TR-04), PyTorch convention ``y = W x + b``, ReLU after every hidden layer:

    x [243] -> 512 -> 512 -> 512 -> 512 -> 512 -> 512 -> logits [82]     (W1..W7, b1..b7)

``x`` = ``board.encode`` (own stones, opponent stones, empty; 81 each). Logits index the 81
points (``row * 9 + col``) plus pass (81). The move is the highest-logit *legal* move (illegal
moves are masked by the referee; ties go to the lowest index).
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from . import board as B

LAYERS = (243, 512, 512, 512, 512, 512, 512, B.N_POINTS + 1)
N_LINEAR = len(LAYERS) - 1
PARAM_SHAPES: dict[str, tuple[int, ...]] = {}
for _i in range(N_LINEAR):
    PARAM_SHAPES[f"W{_i + 1}"] = (LAYERS[_i + 1], LAYERS[_i])
    PARAM_SHAPES[f"b{_i + 1}"] = (LAYERS[_i + 1],)

EVAL_GAMES = 200            # half as black, half as white, private seeds
N_FORWARD = 50              # metered single-game rollouts the agent may request
Player = Callable[[B.Board], int]


def sub_seed(seed: int, *keys: int) -> int:
    return int(np.random.default_rng([seed, *keys]).integers(2 ** 31))


def task_seeds(seed: int) -> dict[str, int]:
    return {"eval": sub_seed(seed, 1), "dev": sub_seed(seed, 2), "baseline_init": sub_seed(seed, 4),
            "norm_init": sub_seed(seed, 5)}


def zero_weights() -> dict[str, np.ndarray]:
    return {k: np.zeros(s, np.float32) for k, s in PARAM_SHAPES.items()}


def init_weights(seed: int) -> dict[str, np.ndarray]:
    """PyTorch ``nn.Linear`` default initialisation: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(1, N_LINEAR + 1):
        shape = PARAM_SHAPES[f"W{i}"]
        bound = 1.0 / np.sqrt(shape[1])
        out[f"W{i}"] = rng.uniform(-bound, bound, shape).astype(np.float32)
        out[f"b{i}"] = rng.uniform(-bound, bound, shape[0]).astype(np.float32)
    return out


def frobenius_norm(weights: Mapping[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(float((np.asarray(w, np.float64) ** 2).sum()) for w in weights.values())))


def validation_error(delta: Mapping[str, np.ndarray]) -> str | None:
    for name, value in delta.items():
        if name not in PARAM_SHAPES:
            return f"unknown parameter {name!r}; expected a subset of {sorted(PARAM_SHAPES)}"
        if np.shape(value) != PARAM_SHAPES[name]:
            return f"{name} has shape {list(np.shape(value))}, expected {list(PARAM_SHAPES[name])}"
        if not np.all(np.isfinite(value)):
            return f"{name} contains non-finite values"
    return None


def apply_delta(delta: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    weights = zero_weights()
    for name, value in delta.items():
        weights[name] = weights[name] + np.asarray(value, np.float32)
    return weights


def logits(weights: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = x
    for i in range(1, N_LINEAR + 1):
        h = h @ weights[f"W{i}"].T + weights[f"b{i}"]
        if i < N_LINEAR:
            h = np.maximum(h, 0.0)
    return h


# ---------------------------------------------------------------------------------- players
def policy_player(weights: Mapping[str, np.ndarray]) -> Player:
    w = {k: np.asarray(v, np.float32) for k, v in weights.items()}

    def act(board: B.Board) -> int:
        z = logits(w, B.encode(board)[None])[0].astype(np.float64)
        z[~board.legal_mask()] = -np.inf
        return int(np.argmax(z))

    return act


def random_player(seed: int) -> Player:
    """Fixed opponent: uniform over legal moves that do not fill its own single-point eyes;
    passes only when no such move exists."""
    rng = np.random.default_rng(seed)

    def act(board: B.Board) -> int:
        options = [p for p in range(B.N_POINTS)
                   if board.is_legal(p) and not B.is_own_eye(board, p, board.to_play)]
        return int(options[rng.integers(len(options))]) if options else B.PASS

    return act


def play_game(black: Player, white: Player) -> tuple[B.Board, list[int]]:
    board, moves = B.Board(), []
    while not board.over:
        player = black if board.to_play == B.BLACK else white
        move = player(board)
        board.play(move)
        moves.append(move)
    return board, moves


LEVELS: tuple[int, ...] = (0, 1, 2)   # bounded explicit knob levels (opponent reads 2^L - 1 plies; tactics.py)


def opponent_factory(level: int | None) -> Callable[[int], Player]:
    """The fixed opponent of a build: the default random opponent, or the level-L capture reader."""
    if level is None:
        return random_player
    from . import tactics as T

    if level not in LEVELS:
        raise ValueError(f"TR04-GO-001: supported knob levels are {LEVELS}, got {level}")
    depth = T.opponent_depth(level)
    return lambda seed: T.reading_opponent(seed, depth)


def match(weights: Mapping[str, np.ndarray] | None, seed: int, n_games: int,
          opponent: Callable[[int], Player] = random_player, player: Player | None = None) -> dict:
    """Policy (``weights``, or any ``player``) vs the opponent; colours alternate (game i: policy is black iff i
    is even). The default opponent is the random one."""
    policy = player if player is not None else policy_player(weights)
    wins, margins = [], []
    for i in range(n_games):
        opp = opponent(sub_seed(seed, i))
        policy_color = B.BLACK if i % 2 == 0 else B.WHITE
        black, white = (policy, opp) if policy_color == B.BLACK else (opp, policy)
        board, _ = play_game(black, white)
        area_b, area_w = board.area()
        margin = (area_b - area_w - B.KOMI) * (1 if policy_color == B.BLACK else -1)
        wins.append(float(board.winner() == policy_color))
        margins.append(margin)
    w = np.asarray(wins)
    return {"win_rate": float(w.mean()), "stderr": float(w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 1 else 0.0,
            "n": len(w), "mean_margin": float(np.mean(margins))}
