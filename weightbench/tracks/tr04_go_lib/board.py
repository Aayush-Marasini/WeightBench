"""Minimal exact 9x9 Go: captures, no suicide, simple ko, area (Tromp-Taylor) scoring.

Points are indexed ``p = row * 9 + col`` (row 0 = top). Move ``PASS = 81``. Pure Python/NumPy,
shared verbatim with the agent (``go/board.py`` in the task's public files).
"""
from __future__ import annotations

import numpy as np

SIZE = 9
N_POINTS = SIZE * SIZE
PASS = N_POINTS
EMPTY, BLACK, WHITE = 0, 1, 2
KOMI = 7.5              # white's compensation; the .5 rules out ties
MAX_MOVES = 2 * N_POINTS + 40


def _neighbors(p: int) -> tuple[int, ...]:
    row, col = divmod(p, SIZE)
    out = []
    if row > 0:
        out.append(p - SIZE)
    if row < SIZE - 1:
        out.append(p + SIZE)
    if col > 0:
        out.append(p - 1)
    if col < SIZE - 1:
        out.append(p + 1)
    return tuple(out)


NEIGHBORS: tuple[tuple[int, ...], ...] = tuple(_neighbors(p) for p in range(N_POINTS))


def opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


class Board:
    def __init__(self) -> None:
        self.stones = np.zeros(N_POINTS, np.int8)
        self.to_play = BLACK
        self.ko: int | None = None        # point the player to move may not play (simple ko)
        self.consecutive_passes = 0
        self.n_moves = 0

    def copy(self) -> "Board":
        b = Board()
        b.stones = self.stones.copy()
        b.to_play, b.ko, b.consecutive_passes, b.n_moves = self.to_play, self.ko, self.consecutive_passes, self.n_moves
        return b

    # ---- groups ---------------------------------------------------------------------
    def group(self, p: int) -> tuple[set[int], set[int]]:
        """Stones of the chain containing ``p`` and its liberties."""
        color = self.stones[p]
        stones, liberties, frontier = {p}, set(), [p]
        while frontier:
            q = frontier.pop()
            for r in NEIGHBORS[q]:
                if self.stones[r] == EMPTY:
                    liberties.add(r)
                elif self.stones[r] == color and r not in stones:
                    stones.add(r)
                    frontier.append(r)
        return stones, liberties

    # ---- moves ----------------------------------------------------------------------
    def is_legal(self, p: int) -> bool:
        if p == PASS:
            return True
        if self.stones[p] != EMPTY or p == self.ko:
            return False
        if any(self.stones[r] == EMPTY for r in NEIGHBORS[p]):
            return True   # has a liberty right away, so not suicide
        me, them = self.to_play, opponent(self.to_play)
        for r in NEIGHBORS[p]:
            if self.stones[r] == them and len(self.group(r)[1]) == 1:
                return True   # captures, so it gains a liberty
        for r in NEIGHBORS[p]:
            if self.stones[r] == me and len(self.group(r)[1]) > 1:
                return True   # joins a chain that keeps another liberty
        return False

    def legal_mask(self) -> np.ndarray:
        """Bool [82]: legal points plus pass (always legal)."""
        return np.array([self.is_legal(p) for p in range(N_POINTS + 1)])

    def play(self, p: int) -> int:
        """Play a legal move for the player to move. Returns the number of stones captured."""
        if not self.is_legal(p):
            raise ValueError(f"illegal move {p}")
        self.n_moves += 1
        me, them = self.to_play, opponent(self.to_play)
        self.to_play = them
        if p == PASS:
            self.consecutive_passes += 1
            self.ko = None
            return 0
        self.consecutive_passes = 0
        self.stones[p] = me
        captured: list[int] = []
        for r in NEIGHBORS[p]:
            if self.stones[r] == them:
                chain, liberties = self.group(r)
                if not liberties:
                    captured.extend(chain)
                    self.stones[list(chain)] = EMPTY
        own_chain, own_liberties = self.group(p)
        single_recapture = len(captured) == 1 and len(own_chain) == 1 and len(own_liberties) == 1
        self.ko = captured[0] if single_recapture else None
        return len(captured)

    @property
    def over(self) -> bool:
        return self.consecutive_passes >= 2 or self.n_moves >= MAX_MOVES

    # ---- scoring ----------------------------------------------------------------------
    def area(self) -> tuple[int, int]:
        """(black, white) area: stones plus empty regions bordered by only that colour."""
        black = int((self.stones == BLACK).sum())
        white = int((self.stones == WHITE).sum())
        seen: set[int] = set()
        for p in range(N_POINTS):
            if self.stones[p] != EMPTY or p in seen:
                continue
            region, borders, frontier = {p}, set(), [p]
            while frontier:
                q = frontier.pop()
                for r in NEIGHBORS[q]:
                    if self.stones[r] == EMPTY and r not in region:
                        region.add(r)
                        frontier.append(r)
                    elif self.stones[r] != EMPTY:
                        borders.add(int(self.stones[r]))
            seen |= region
            if borders == {BLACK}:
                black += len(region)
            elif borders == {WHITE}:
                white += len(region)
        return black, white

    def winner(self) -> int:
        black, white = self.area()
        return BLACK if black - white - KOMI > 0 else WHITE


def encode(board: Board) -> np.ndarray:
    """Policy input [243] from the mover's view: own stones, opponent stones, empty points."""
    me, them = board.to_play, opponent(board.to_play)
    return np.concatenate([board.stones == me, board.stones == them, board.stones == EMPTY]).astype(np.float32)


def is_own_eye(board: Board, p: int, color: int) -> bool:
    """Single-point eye: empty and every neighbour is ``color``."""
    return board.stones[p] == EMPTY and all(board.stones[r] == color for r in NEIGHBORS[p])
