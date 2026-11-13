"""TR-04 MinAtar: policy architecture, shared observation encoding, seeds and ΔW validation.

Pure NumPy (no MinAtar import), so the host-side task module can use it cheaply.

Architecture (SPEC §2 TR-04), PyTorch convention ``y = W x + b``:

    h1     = relu(W1 @ x + b1)      W1: [32, 600]
    h2     = relu(W2 @ h1 + b2)     W2: [32, 32]
    logits = W3 @ h2 + b3           W3: [6, 32]      action = argmax(logits) (first index on ties)

Observation ``x`` (length 600) is the game's MinAtar state mapped onto 6 channels:
``x[c * 100 + row * 10 + col]`` with row 0 = top, col 0 = left.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

GAMES = ("breakout", "seaquest", "asterix")
ACTIONS = ("n", "l", "u", "r", "d", "f")  # MinAtar full action set; output index = action id
N_CHANNELS = 6
GRID = 10
N_IN = N_CHANNELS * GRID * GRID
HIDDEN = (32, 32)
N_ACTIONS = len(ACTIONS)

PARAM_SHAPES: dict[str, tuple[int, ...]] = {
    "W1": (HIDDEN[0], N_IN), "b1": (HIDDEN[0],),
    "W2": (HIDDEN[1], HIDDEN[0]), "b2": (HIDDEN[1],),
    "W3": (N_ACTIONS, HIDDEN[1]), "b3": (N_ACTIONS,),
}

# Native MinAtar channel name -> common channel. Breakout and Asterix keep their native order
# (channels 4 and 5 stay 0). Seaquest has 10 native channels, merged so that every merged pair
# stays separable by row (divers / trails live in rows 1-8, the two gauges in row 9).
CHANNEL_MAP: dict[str, dict[str, int]] = {
    "breakout": {"paddle": 0, "ball": 1, "trail": 2, "brick": 3},
    "asterix": {"player": 0, "enemy": 1, "trail": 2, "gold": 3},
    "seaquest": {"sub_front": 0, "sub_back": 1, "friendly_bullet": 2,
                 "enemy_fish": 3, "enemy_sub": 3, "enemy_bullet": 3,
                 "diver": 4, "diver_guage": 4, "trail": 5, "oxygen_guage": 5},
}

# Evaluation protocol (identical for every game).
STICKY_ACTION_PROB = 0.1     # MinAtar default
MAX_STEPS = 1000             # per-episode step cap
EVAL_EPISODES = 100          # agent / baseline / reference submissions (private eval seeds)
EXPERT_EPISODES = 30         # planner ceiling S_max (same eval seed stream)

# Strongest known policy per game (mean eval return, 100 episodes), used as the ceiling when it beats the planner:
# S_max = max(planner, STRONGEST_KNOWN[game]) (SPEC §3: "the max achievable score, or the strongest reference").
# Entries are PPO policies (100M steps) in this task's 600→32→32→6 shell (tr04_minatar_lib/ppo.py).
# Only independent policies may set a ceiling. The hand-written reference circuits are never a ceiling source: a
# ceiling equal to the reference's own return would make the reference's S+ self-referential. Seaquest has no
# entry: PPO stays below the planner there, so the per-seed planner is S_max.
# Where the reference beats S_max its S+ clips at 1. Changing an entry re-normalises S+ (rescore from the S_edit
# stored in each ledger, or re-run the evaluation).
STRONGEST_KNOWN = {
    "breakout": {"return": 17.28, "source": "PPO 100M steps, same shell, seed 0"},
    "asterix": {"return": 25.03, "source": "PPO 100M steps, same shell, seed 0"},
}
DEV_EPISODES = 30            # reference candidate selection (private dev seeds, disjoint from eval)


def index(channel: int, row: int, col: int) -> int:
    """Position of (channel, row, col) in the flat 600-dim observation."""
    return channel * GRID * GRID + row * GRID + col


def sub_seed(seed: int, *keys: int) -> int:
    """Deterministic child seed, so every random stream of a task derives from the task seed."""
    return int(np.random.default_rng([seed, *keys]).integers(2 ** 31))


def task_seeds(seed: int) -> dict[str, int]:
    return {"eval": sub_seed(seed, 1), "dev": sub_seed(seed, 2), "expert": sub_seed(seed, 3),
            "baseline_init": sub_seed(seed, 4), "norm_init": sub_seed(seed, 5)}


# ---- difficulty knob ``algorithmic_horizon`` (SPEC §5): observation delay -------------------------------------
# Level L: H = 2^L. The policy sees the frame from d = H - 1 steps ago plus the d actions it chose since then
# (one-hot slots), so it must predict the current state H - 1 steps ahead. L0 (d = 0) is the default game.
# The unversioned default build is untouched; explicit levels are opt-in (``Task.knob_levels``).
LEVELS: tuple[int, ...] = (0, 1, 2)


def delay(level: int) -> int:
    """Observation delay d = H - 1 = 2^L - 1 frames at knob level ``level`` (0 at L0)."""
    from weightbench import knobs

    if level not in LEVELS:
        raise ValueError(f"TR04-GAME: supported knob levels are {LEVELS}, got {level}")
    return int(knobs.value(knobs.get("TR04"), level)) - 1


def input_dim(d: int = 0) -> int:
    """Length of the observation vector under delay ``d``: the 600-dim frame plus d one-hot action slots."""
    return N_IN + N_ACTIONS * d


def param_shapes(d: int = 0) -> dict[str, tuple[int, ...]]:
    """Shell parameter shapes under delay ``d`` (only W1's input width changes; d = 0 gives PARAM_SHAPES)."""
    return {**PARAM_SHAPES, "W1": (HIDDEN[0], input_dim(d))}


def zero_weights(shapes: Mapping[str, tuple[int, ...]] = PARAM_SHAPES) -> dict[str, np.ndarray]:
    """The all-zero network: the TR-04 base model M_base (SPEC §3)."""
    return {k: np.zeros(s, np.float32) for k, s in shapes.items()}


def init_weights(seed: int, shapes: Mapping[str, tuple[int, ...]] = PARAM_SHAPES) -> dict[str, np.ndarray]:
    """PyTorch ``nn.Linear`` default initialisation: W, b ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    rng = np.random.default_rng(seed)
    out = {}
    for layer in (1, 2, 3):
        w_shape = shapes[f"W{layer}"]
        bound = 1.0 / np.sqrt(w_shape[1])
        out[f"W{layer}"] = rng.uniform(-bound, bound, w_shape).astype(np.float32)
        out[f"b{layer}"] = rng.uniform(-bound, bound, w_shape[0]).astype(np.float32)
    return out


def frobenius_norm(weights: Mapping[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(float((np.asarray(w, np.float64) ** 2).sum()) for w in weights.values())))


def validation_error(delta: Mapping[str, np.ndarray],
                     shapes: Mapping[str, tuple[int, ...]] = PARAM_SHAPES) -> str | None:
    """Why a submitted ΔW is unusable, or None. Missing tensors count as zero."""
    for name, value in delta.items():
        if name not in shapes:
            return f"unknown parameter {name!r}; expected a subset of {sorted(shapes)}"
        if np.shape(value) != shapes[name]:
            return f"{name} has shape {list(np.shape(value))}, expected {list(shapes[name])}"
        if not np.all(np.isfinite(value)):
            return f"{name} contains non-finite values"
    return None


def apply_delta(delta: Mapping[str, np.ndarray],
                shapes: Mapping[str, tuple[int, ...]] = PARAM_SHAPES) -> dict[str, np.ndarray]:
    """M_edited = M_base + ΔW with M_base = 0, as float32."""
    weights = zero_weights(shapes)
    for name, value in delta.items():
        weights[name] = weights[name] + np.asarray(value, np.float32)
    return weights


def pad_inputs(weights: Mapping[str, np.ndarray], d: int) -> dict[str, np.ndarray]:
    """A d = 0 weight set widened to delay ``d``: zero W1 columns for the action slots (the policy ignores them)."""
    out = {k: np.asarray(v, np.float32).copy() for k, v in weights.items()}
    out["W1"] = np.concatenate([out["W1"], np.zeros((out["W1"].shape[0], N_ACTIONS * d), np.float32)], axis=1)
    return out


def delayed_observation(frame: np.ndarray, pending: list[int], d: int) -> np.ndarray:
    """Observation under delay ``d``: ``frame`` (600) then d one-hot action slots. Slot i holds the action chosen at
    time t - d + i. The ``pending`` actions (chosen since the shown frame, oldest first, at most d) fill the LAST
    len(pending) slots; earlier slots stay zero (they fall before the episode start)."""
    if len(pending) > d:
        raise ValueError(f"{len(pending)} pending actions for delay {d}")
    slots = np.zeros((d, N_ACTIONS), np.float32)
    for i, a in enumerate(pending):
        slots[d - len(pending) + i, a] = 1.0
    return np.concatenate([np.asarray(frame, np.float32).reshape(-1), slots.reshape(-1)])


def logits(weights: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Policy logits for a batch ``x`` of shape [N, 600]. Server-side only (evaluator, reference)."""
    h1 = np.maximum(x @ weights["W1"].T + weights["b1"], 0.0)
    h2 = np.maximum(h1 @ weights["W2"].T + weights["b2"], 0.0)
    return h2 @ weights["W3"].T + weights["b3"]
