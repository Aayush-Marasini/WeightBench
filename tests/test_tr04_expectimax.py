"""TR04-GAME-001 exact expectimax: transition fidelity and brute-force validity.

Needs the ``minatar`` package (CPU only, a few hundred simulator steps, no policy rollouts).
"""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("minatar")

from weightbench.tracks.tr04_minatar_lib import expectimax as X  # noqa: E402
from weightbench.tracks.tr04_minatar_lib import rollout as R  # noqa: E402


def _bricks_full() -> bytes:
    m = np.zeros((10, 10), np.uint8)
    m[1:4, :] = 1
    return m.tobytes()


def test_forced_step_matches_minatar_wrapper():
    """With the sticky coin forced to the value the real wrapper drew, our step equals Environment.act."""
    for seed in (0, 1, 2):
        env = R.make_env(X.GAME, seed)
        rng = np.random.default_rng(seed)
        for _ in range(80):
            if env.env.terminal:
                env.reset()
            before = X.snapshot(env)
            rng_state = env.random.get_state()
            sticky = bool(env.random.rand() < float(X.STICKY_P))   # the coin act() will draw
            env.random.set_state(rng_state)
            action = int(rng.integers(X.N_ACTIONS))
            reward = env.act(action)[0]
            nxt, our_reward = X.step(before, action, sticky)
            assert X.snapshot(env) == nxt
            assert reward == our_reward


def test_zero_network_policy_is_depth1_planner_at_spawn():
    """At spawn no brick is one step away, so every H=1 value is 0 and the lowest id (no-op) wins.
    This is the zero network's behaviour, which is why the H=1 expectimax ceiling does not reproduce the
    scripted Breakout planner (tr01_behavioral_lib/games.py::expert_values)."""
    for seed in (0, 1, 2):
        snap = X.snapshot(R.make_env(X.GAME, seed))
        assert X.best_action(snap, 1) == (0, Fraction(0))


def test_sticky_probability_is_one_tenth():
    assert X.STICKY_P == Fraction(1, 10)


def test_chance_outcomes_merge_when_action_repeats_last():
    snap = X.snapshot(R.make_env(X.GAME, 0))
    assert X.chance_outcomes(snap, snap.last_action) == [(Fraction(1), False)]
    other = (snap.last_action + 1) % X.N_ACTIONS
    assert sorted(X.chance_outcomes(snap, other)) == sorted([(Fraction(9, 10), False), (Fraction(1, 10), True)])


def test_hand_built_brick_hit_is_exact():
    """Ball one step from a brick: every action scores 1 this step (paddle does not change the ball path)."""
    snap = X.Snapshot(ball_x=4, ball_y=4, ball_dir=1, pos=4, brick=_bricks_full(), strike=False,
                      last_x=4, last_y=4, terminal=False, last_action=0)
    assert X.value(snap, 1) == Fraction(1)
    assert X.brute_force_value(snap, 1) == Fraction(1)
    assert X.best_action(snap, 1) == (0, Fraction(1))
    # Terminal states and depth 0 are worth 0.
    assert X.value(replace(snap, terminal=True), 3) == 0
    assert X.value(snap, 0) == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_expectimax_equals_brute_force_depth_1_and_2(seed):
    """Exact agreement (Fraction equality) between depth-H expectimax and enumeration of all 6**H sequences."""
    states = list(X.random_states(seed, steps=300))
    nonzero = 0
    for snap in states:
        for depth in (1, 2):
            assert X.value(snap, depth) == X.brute_force_value(snap, depth), (seed, depth, snap)
        nonzero += X.value(snap, 1) != 0
    # Random walks rarely put a brick in reach, so also check depth 3 on the states that have a reward in reach.
    reachable = [s for s in states if X.value(s, 3) != 0][:10]
    for snap in reachable:
        assert X.value(snap, 3) == X.brute_force_value(snap, 3), (seed, snap)
    assert nonzero >= 1 and len(reachable) >= 1, "the comparison must not be trivially zero"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sticky_coin_is_inert_within_horizon_4(seed):
    """Geometry: the paddle sits on row 9 and the bricks on rows 1-3. The paddle changes the ball path only at a
    bounce on row 9 (step k >= 1). A bounced ball rises one row per step, so its earliest brick hit is at step
    k + 5 >= 6. Hence for H <= 5 the sticky coin cannot change a value; it only matters for longer lookahead."""
    for snap in X.random_states(seed, steps=120):
        for depth in (1, 2, 3, 4):
            assert X.value(snap, depth) == X.value(snap, depth, p=Fraction(0))
