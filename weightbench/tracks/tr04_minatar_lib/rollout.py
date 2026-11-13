"""Server-side MinAtar rollouts for TR-04 (evaluator, calibration, reference selection).

Only ``jobs.py`` (run through ``weightbench.heavy.run_heavy``) calls the rollout functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from . import spec as S

ActFn = Callable[[object], int]   # Environment -> action id


def make_env(game: str, seed: int):
    from minatar import Environment

    env = Environment(game, sticky_action_prob=S.STICKY_ACTION_PROB)
    env.seed(seed)
    env.reset()  # the constructor's reset used the unseeded RNG
    return env


def encode_state(game: str, state: np.ndarray, channels: Mapping[str, int]) -> np.ndarray:
    """Map a native MinAtar state [10, 10, C] (bool) onto the flat 600-dim common observation."""
    out = np.zeros((S.N_CHANNELS, S.GRID, S.GRID), np.float32)
    for name, common in S.CHANNEL_MAP[game].items():
        out[common] = np.maximum(out[common], state[:, :, channels[name]])
    return out.reshape(-1)


def encode(game: str, env) -> np.ndarray:
    return encode_state(game, env.state(), env.env.channels)


def policy_act_fn(game: str, weights: Mapping[str, np.ndarray]) -> ActFn:
    w = {k: np.asarray(v, np.float32) for k, v in weights.items()}
    return lambda env: int(S.logits(w, encode(game, env)[None])[0].argmax())


def episode_returns(game: str, act_fn: ActFn, seed: int, episodes: int,
                    max_steps: int = S.MAX_STEPS, delay: int = 0) -> list[float]:
    """Undiscounted return of each of ``episodes`` consecutive episodes in one seeded env.

    ``delay`` = 0 (default game): ``act_fn(env)``. ``delay`` = d > 0 (knob level, ``spec.delay``): ``act_fn(view)``
    with a ``DelayView`` that exposes only the state d steps ago and the actions chosen since."""
    env = make_env(game, seed)
    returns = []
    for _ in range(episodes):
        env.reset()
        view = DelayView(game, delay, env) if delay else None
        total, terminal, steps = 0.0, False, 0
        while not terminal and steps < max_steps:
            action = act_fn(view) if view is not None else act_fn(env)
            reward, terminal = env.act(action)
            if view is not None:
                view.push(env, action)
            total += reward
            steps += 1
        returns.append(total)
    return returns


class DelayView:
    """What a policy sees under observation delay d: the env state at time max(0, t - d) (a frozen copy) and the
    actions chosen at times max(0, t - d) .. t - 1 (``pending``, oldest first). Server-side only."""

    def __init__(self, game: str, d: int, env) -> None:
        from collections import deque

        if d < 1:
            raise ValueError("DelayView needs d >= 1")
        self.game, self.d = game, d
        self.states = deque([self._freeze(env)], maxlen=d + 1)
        self.actions: deque[int] = deque(maxlen=d)

    @staticmethod
    def _freeze(env):
        import copy

        return copy.deepcopy(env)

    def push(self, env, action: int) -> None:
        self.actions.append(int(action))
        self.states.append(self._freeze(env))

    @property
    def observed(self):
        """The env as it was at the shown frame (do not act on it; copy first)."""
        return self.states[0]

    @property
    def pending(self) -> list[int]:
        """Actions chosen after the shown frame (len(states) - 1 of them, at most d; both deques advance together)."""
        return list(self.actions)

    def x(self) -> np.ndarray:
        return S.delayed_observation(encode(self.game, self.observed), self.pending, self.d)


def delayed_policy_act_fn(weights: Mapping[str, np.ndarray]) -> Callable[[DelayView], int]:
    w = {k: np.asarray(v, np.float32) for k, v in weights.items()}
    return lambda view: int(S.logits(w, view.x()[None])[0].argmax())


def delayed_expert_act_fn(game: str, seed: int) -> Callable[[DelayView], int]:
    """Independent ceiling under delay: copy the shown state, reseed its RNG from the planner's own stream (so the
    copy never replays the real env's future spawns or sticky coins), replay the pending actions without sticky
    repeats (the most likely outcome), then ask the default planner (``expert_act_fn``'s policy) on that predicted
    state. It sees exactly the agent's information plus a perfect simulator."""
    import copy

    from weightbench.tracks.tr01_behavioral_lib import games as G

    rng = np.random.default_rng(seed)

    def act(view: DelayView) -> int:
        sim = copy.deepcopy(view.observed)
        sim.seed(int(rng.integers(2 ** 31)))
        sim.sticky_action_prob = 0.0
        for a in view.pending:
            _, terminal = sim.act(a)
            if terminal:
                break
        return G.expert_action(game, sim, rng)

    return act


def expert_act_fn(game: str, seed: int) -> ActFn:
    """Server-side planner (scripted ball tracker for breakout, Monte-Carlo planner on cloned env
    state otherwise), shared with TR01-BEH-001."""
    from weightbench.tracks.tr01_behavioral_lib import games as G

    rng = np.random.default_rng(seed)
    return lambda env: G.expert_action(game, env, rng)


def summarize(returns: list[float]) -> dict[str, float | list[float]]:
    r = np.asarray(returns, np.float64)
    stderr = float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0
    return {"mean": float(r.mean()), "stderr": stderr, "n": len(r), "returns": r.tolist()}


def source_dir() -> Path:
    import minatar

    return Path(minatar.__file__).parent
