"""Capture reading for the TR04-GO-001 knob levels (SPEC §5, ``algorithmic_horizon``).

At explicit knob level L the fixed opponent reads ``2^L - 1`` plies of capture tactics before it falls back to the
default random move, and the independent ceiling player reads ``2^L`` plies (one ply more) with an area-greedy
fallback. Neither uses the hand-compiled reference (``compiled.py``).

Reading is restricted to a mechanical local window so that it stays tractable in the pure-Python board:

- candidate moves = the legal liberties of every chain (either colour) with at most two liberties (ataris, escapes,
  captures, ladder steps), plus pass as the null move at inner nodes;
- value = capture balance along the line (stones captured by the side to move minus stones it loses), so placing a
  stone is worth nothing by itself and only captures count;
- negamax with alpha-beta and a hard per-decision node cap (``NODE_CAP``; at the cap the remaining nodes score their
  static material, deterministically).

Root rule: play the best candidate only if its read value is strictly better than the null move (pass) read to the
same depth, i.e. only when reading finds material to win or to save; ties go to the lowest point index. Otherwise the
fallback plays. Depth 0 never reads, so the level-0 opponent is exactly the default random opponent (same RNG stream).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import board as B

NODE_CAP = 1500            # board positions expanded per decision (bounds wall time: ~0.5 s per move worst case)
MAX_LIBERTIES = 2          # chains this short define the local reading window


def opponent_depth(level: int) -> int:
    """Plies the level-L opponent reads: 2^L - 1 (L0 = 0: the default random opponent)."""
    return 2 ** level - 1


def ceiling_depth(level: int) -> int:
    """Plies the independent ceiling player reads: 2^L (one ply deeper than the level-L opponent)."""
    return 2 ** level


def candidate_moves(board: B.Board) -> list[int]:
    """Legal liberties of chains with <= MAX_LIBERTIES liberties, sorted, excluding the mover's own single-point eyes."""
    seen: set[int] = set()
    points: set[int] = set()
    for p in range(B.N_POINTS):
        if board.stones[p] == B.EMPTY or p in seen:
            continue
        stones, liberties = board.group(p)
        seen |= stones
        if len(liberties) <= MAX_LIBERTIES:
            points |= liberties
    return [p for p in sorted(points) if board.is_legal(p) and not B.is_own_eye(board, p, board.to_play)]


class Reader:
    """Depth-limited capture reader with a per-decision node cap."""

    def __init__(self, depth: int, node_cap: int = NODE_CAP) -> None:
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.depth, self.node_cap, self.nodes = depth, node_cap, 0

    def _child(self, board: B.Board, move: int) -> tuple[B.Board, int]:
        child = board.copy()
        captured = child.play(move)
        self.nodes += 1
        return child, captured

    def _line(self, board: B.Board, move: int, depth: int, alpha: float = -np.inf, beta: float = np.inf) -> float:
        """Capture balance for the mover of ``move``: what it captures minus the best reply line."""
        child, captured = self._child(board, move)
        return captured - self.value(child, depth, -beta + captured, -alpha + captured)

    def value(self, board: B.Board, depth: int, alpha: float = -np.inf, beta: float = np.inf) -> float:
        """Negamax capture balance for the side to move over ``depth`` more plies of local reading (0 at the horizon
        or once the node cap is reached)."""
        if depth == 0 or self.nodes >= self.node_cap:
            return 0.0
        best = -np.inf
        for move in [*candidate_moves(board), B.PASS]:
            v = self._line(board, move, depth - 1, alpha, beta)
            best = max(best, v)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best

    def choose(self, board: B.Board) -> int | None:
        """The tactical move reading finds (strictly better than the null move), or None (use the fallback)."""
        if self.depth == 0:
            return None
        moves = candidate_moves(board)
        if not moves:
            return None
        self.nodes = 0
        null = self._line(board, B.PASS, self.depth - 1)
        best_move, best = None, null
        for move in moves:
            v = self._line(board, move, self.depth - 1)
            if v > best:
                best_move, best = move, v
        return best_move


def _non_eye_moves(board: B.Board) -> list[int]:
    return [p for p in range(B.N_POINTS) if board.is_legal(p) and not B.is_own_eye(board, p, board.to_play)]


def reading_opponent(seed: int, depth: int, node_cap: int = NODE_CAP) -> Callable[[B.Board], int]:
    """Level opponent: capture reading to ``depth`` plies, else the default random move (same RNG as
    ``spec.random_player``: one draw per fallback move, none for a tactical move)."""
    rng = np.random.default_rng(seed)
    reader = Reader(depth, node_cap)

    def act(board: B.Board) -> int:
        move = reader.choose(board)
        if move is not None:
            return move
        options = _non_eye_moves(board)
        return int(options[rng.integers(len(options))]) if options else B.PASS

    return act


def ceiling_player(seed: int, depth: int, node_cap: int = NODE_CAP) -> Callable[[B.Board], int]:
    """Independent ceiling policy: capture reading to ``depth`` plies, else the non-eye move that maximises own area
    minus opponent area after the move (ties broken by the seeded RNG). It never uses the compiled reference."""
    rng = np.random.default_rng(seed)
    reader = Reader(depth, node_cap)

    def act(board: B.Board) -> int:
        move = reader.choose(board)
        if move is not None:
            return move
        options = _non_eye_moves(board)
        if not options:
            return B.PASS
        me = board.to_play
        scores = []
        for p in options:
            child = board.copy()
            child.play(p)
            black, white = child.area()
            scores.append((black - white) * (1 if me == B.BLACK else -1))
        top = [p for p, s in zip(options, scores) if s == max(scores)]
        return int(top[rng.integers(len(top))])

    return act
