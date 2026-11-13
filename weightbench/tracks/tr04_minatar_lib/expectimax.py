"""Exact depth-H expectimax over the MinAtar Breakout simulator (TR04-GAME-001 only).

Used to study the ``algorithmic_horizon`` knob. Server-side only.

Stochasticity during play: the MinAtar wrapper's sticky-action coin. With probability p the previous
executed action is repeated instead of the chosen one. Breakout's ``step`` is otherwise deterministic. Its
only other randomness is the ball spawn in ``reset``, which the planner observes, so it is not a chance
node while planning.

Transitions call the real ``minatar.environments.breakout.Env.act`` on a restored snapshot. Values are
exact ``fractions.Fraction`` numbers, so the expectimax can be compared with brute force for equality.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
from typing import Iterator

import numpy as np

from . import spec as S

GAME = "breakout"
N_ACTIONS = S.N_ACTIONS
STICKY_P = Fraction(S.STICKY_ACTION_PROB).limit_denominator(1000)   # 1/10


@dataclass(frozen=True)
class Snapshot:
    """Everything ``Env.act`` and the sticky wrapper read, as hashable values."""
    ball_x: int
    ball_y: int
    ball_dir: int
    pos: int
    brick: bytes          # 10x10 brick_map, row-major, 0/1 bytes
    strike: bool
    last_x: int
    last_y: int
    terminal: bool
    last_action: int      # wrapper's last executed action (the sticky repeat target)


def snapshot(env) -> Snapshot:
    """Snapshot of a live ``minatar.Environment`` (Breakout)."""
    e = env.env
    return Snapshot(int(e.ball_x), int(e.ball_y), int(e.ball_dir), int(e.pos),
                    np.asarray(e.brick_map, np.uint8).tobytes(), bool(e.strike),
                    int(e.last_x), int(e.last_y), bool(e.terminal), int(env.last_action))


def _restore(snap: Snapshot):
    from minatar.environments import breakout as B

    e = B.Env.__new__(B.Env)          # skip __init__: no reset, no RNG draw
    e.random = np.random.RandomState(0)  # unused by act(); kept for completeness
    e.action_map = ['n', 'l', 'u', 'r', 'd', 'f']   # set in Env.__init__; act() maps action ids through it
    e.ball_x, e.ball_y, e.ball_dir, e.pos = snap.ball_x, snap.ball_y, snap.ball_dir, snap.pos
    e.brick_map = np.frombuffer(snap.brick, np.uint8).reshape(10, 10).astype(np.float64)
    e.strike, e.last_x, e.last_y, e.terminal = snap.strike, snap.last_x, snap.last_y, snap.terminal
    return e


def step(snap: Snapshot, action: int, sticky: bool) -> tuple[Snapshot, int]:
    """One wrapper step with the sticky coin forced to ``sticky``. Returns (next snapshot, reward)."""
    executed = snap.last_action if sticky else action   # mirrors minatar.Environment.act
    e = _restore(snap)
    reward, terminal = e.act(executed)
    return Snapshot(int(e.ball_x), int(e.ball_y), int(e.ball_dir), int(e.pos),
                    np.asarray(e.brick_map, np.uint8).tobytes(), bool(e.strike),
                    int(e.last_x), int(e.last_y), bool(terminal), executed), int(reward)


def chance_outcomes(snap: Snapshot, action: int, p: Fraction = STICKY_P) -> list[tuple[Fraction, bool]]:
    """Sticky-coin outcomes of choosing ``action``: [(probability, sticky?)]. Merged when both coincide."""
    if action == snap.last_action:
        return [(Fraction(1), False)]
    return [(1 - p, False), (p, True)]


def value(snap: Snapshot, depth: int, p: Fraction = STICKY_P) -> Fraction:
    """Exact depth-``depth`` expectimax value (expected undiscounted return over ``depth`` steps)."""
    return _value(snap, depth, p)


@lru_cache(maxsize=None)
def _value(snap: Snapshot, depth: int, p: Fraction) -> Fraction:
    if depth <= 0 or snap.terminal:
        return Fraction(0)
    return max(_action_value(snap, a, depth, p) for a in range(N_ACTIONS))


def _action_value(snap: Snapshot, action: int, depth: int, p: Fraction) -> Fraction:
    total = Fraction(0)
    for prob, sticky in chance_outcomes(snap, action, p):
        nxt, reward = step(snap, action, sticky)
        total += prob * (reward + _value(nxt, depth - 1, p))
    return total


def best_action(snap: Snapshot, depth: int, p: Fraction = STICKY_P) -> tuple[int, Fraction]:
    """The planner's action (lowest id on ties) and its exact depth-``depth`` value."""
    vals = [_action_value(snap, a, depth, p) if depth > 0 else Fraction(0) for a in range(N_ACTIONS)]
    best = max(range(N_ACTIONS), key=lambda a: (vals[a], -a))
    return best, vals[best]


def _sequence_value(snap: Snapshot, seq: tuple[int, ...], p: Fraction) -> Fraction:
    """Expected return of a fixed action sequence, enumerating every sticky-coin outcome."""
    if not seq:
        return Fraction(0)
    total = Fraction(0)
    for prob, sticky in chance_outcomes(snap, seq[0], p):
        nxt, reward = step(snap, seq[0], sticky)
        total += prob * (reward + _sequence_value(nxt, seq[1:], p))
    return total


def brute_force_value(snap: Snapshot, depth: int, p: Fraction = STICKY_P) -> Fraction:
    """Max over all N_ACTIONS**depth action sequences of the sequence's exact expected return."""
    return max((_sequence_value(snap, seq, p) for seq in product(range(N_ACTIONS), repeat=depth)),
               default=Fraction(0))


def planner_action(env, depth: int) -> int:
    """Planner action for a live Breakout ``minatar.Environment`` at lookahead ``depth``."""
    return best_action(snapshot(env), depth)[0]


def random_states(seed: int, steps: int) -> Iterator[Snapshot]:
    """Snapshots along a seeded random-action walk (test/validation helper; no policy is evaluated)."""
    from weightbench.tracks.tr04_minatar_lib import rollout as R

    env = R.make_env(GAME, seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        yield snapshot(env)
        if env.env.terminal:
            env.reset()
        env.act(int(rng.integers(N_ACTIONS)))
