"""TR04-GAME-001..003: encoding, compiled reference circuits, rescaling, task scoring and leakage.

No test steps a real game. Compiled circuits are checked on hand-built observations, and heavy
jobs are replaced by fakes (all rollouts run through weightbench.heavy).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from weightbench import scoring
from weightbench.ledger import Ledger
from weightbench.server import RunState
from weightbench.task import Submission
from weightbench.tracks import tr04_minatar as T
from weightbench.tracks.tr04_minatar_lib import compiled as K
from weightbench.tracks.tr04_minatar_lib import jobs as J
from weightbench.tracks.tr04_minatar_lib import rollout as R
from weightbench.tracks.tr04_minatar_lib import spec as S
from weightbench.tracks.tr04_minatar_lib.spec import index

TASKS = {"breakout": T.BreakoutCompilation, "seaquest": T.SeaquestCompilation, "asterix": T.AsterixCompilation}


def observation(cells: list[tuple[int, int, int]]) -> np.ndarray:
    x = np.zeros(S.N_IN, np.float32)
    for channel, row, col in cells:
        x[index(channel, row, col)] = 1.0
    return x


def action(weights, cells) -> str:
    return S.ACTIONS[int(S.logits(weights, observation(cells)[None])[0].argmax())]


# ------------------------------------------------------------------------------ spec / encoding
def test_architecture_and_param_count():
    assert S.N_IN == 600 and S.HIDDEN == (32, 32) and S.N_ACTIONS == 6
    assert sum(int(np.prod(s)) for s in S.PARAM_SHAPES.values()) == 20486
    for task in TASKS.values():
        assert task.track == "TR04" and task.submission == "delta_w"
        assert task.budget.n_forward == 0 and task.budget.n_probe == 0 and task.probe_kinds == {}


@pytest.mark.parametrize("game", S.GAMES)
def test_channel_map_covers_every_native_channel(game):
    env = R.make_env(game, 0)                   # construction only, no steps
    assert set(S.CHANNEL_MAP[game]) == set(env.env.channels)
    assert set(S.CHANNEL_MAP[game].values()) <= set(range(S.N_CHANNELS))
    x = R.encode(game, env)
    assert x.shape == (S.N_IN,) and set(np.unique(x)) <= {0.0, 1.0}


def test_encoding_matches_native_state_for_breakout():
    env = R.make_env("breakout", 0)
    env.env.pos, env.env.ball_x, env.env.ball_y, env.env.last_x, env.env.last_y = 2, 6, 5, 5, 4
    x = R.encode("breakout", env)
    assert x[index(0, 9, 2)] == 1 and x[index(1, 5, 6)] == 1 and x[index(2, 4, 5)] == 1
    assert x[index(3, 1, 0)] == 1 and x[index(4, 0, 0):].sum() == 0     # bricks on c3, c4/c5 empty


def test_seaquest_merged_channels_separable_by_row():
    env = R.make_env("seaquest", 0)
    e = env.env
    e.sub_y, e.diver_count, e.oxygen = 4, 2, 150
    e.divers = [[3, 6, True, 5]]
    x = R.encode("seaquest", env).reshape(S.N_CHANNELS, S.GRID, S.GRID)
    assert x[4, 6, 3] == 1 and x[4, 9, 7:9].sum() == 2 and x[4, 9, :7].sum() == 0   # diver + gauge
    assert x[5, 9, :7].sum() == 7                                                   # oxygen 150 -> 7 cells
    assert x[5, 6, 2] == 1                                                          # diver trail


def test_seeds_deterministic_and_distinct():
    a, b = S.task_seeds(0), S.task_seeds(0)
    assert a == b and len(set(a.values())) == len(a) and S.task_seeds(1) != a


def test_init_weights_match_pytorch_default_bounds():
    w = S.init_weights(7)
    assert {k: v.shape for k, v in w.items()} == S.PARAM_SHAPES
    assert np.abs(w["W1"]).max() <= 1 / np.sqrt(600) and np.abs(w["W3"]).max() <= 1 / np.sqrt(32)
    np.testing.assert_array_equal(w["W2"], S.init_weights(7)["W2"])


def test_validation_and_apply_delta():
    assert S.validation_error({"W1": np.zeros((32, 600))}) is None
    assert "unknown" in S.validation_error({"W4": np.zeros(3)})
    assert "shape" in S.validation_error({"b1": np.zeros(31)})
    assert "non-finite" in S.validation_error({"b3": np.full(6, np.inf)})
    w = S.apply_delta({"b3": np.ones(6)})
    assert w["b3"].dtype == np.float32 and w["b3"].sum() == 6 and not w["W1"].any()


# ------------------------------------------------------------------------ compiled circuits
def _breakout_expected(paddle: int, target: int) -> str:
    if target > paddle:
        return "r"
    if target < paddle:
        return "l"
    return "n"


@pytest.mark.parametrize("predict", [True, False])
def test_breakout_circuit_moves_toward_target(predict):
    w = K.breakout(predict)
    for paddle in range(10):
        for ball in range(10):
            for trail in (max(ball - 1, 0), min(ball + 1, 9)):
                target = 2 * ball - trail if predict else ball
                cells = [(0, 9, paddle), (1, 5, ball), (2, 4, trail)]
                assert action(w, cells) == _breakout_expected(paddle, target)


def _closing(row: int, col: int, toward: int = 5) -> list[tuple[int, int, int]]:
    """Enemy cells (channel 1) at ``col`` plus its trail (channel 2) behind it, moving toward column ``toward``."""
    return [(1, row, col), (2, row, col - 1 if col < toward else col + 1)]


def _leaving(row: int, col: int) -> list[tuple[int, int, int]]:
    return [(1, row, col), (2, row, col + 1 if col < 5 else col - 1)]


def _gold(row: int, col: int, closing: bool = True) -> list[tuple[int, int, int]]:
    """Gold at ``col`` whose trail says it is (not) closing on column 5."""
    behind = col - 1 if (col < 5) == closing else col + 1
    return [(3, row, col), (2, row, behind)]


PLAYER_5 = [(0, 5, 5)]


@pytest.mark.parametrize("cells, want", [
    (_gold(3, 1), "u"),                                         # gold coming, two lanes up
    (_gold(7, 8), "d"),
    (_gold(3, 1, closing=False), "n"),                          # gold leaving: not worth a trip
    (_gold(5, 1) + _gold(3, 1), "n"),                           # gold coming in my own lane: wait for it
    (_closing(5, 3), "u"),                                      # enemy coming within 2 columns: leave
    (_closing(5, 3) + _closing(4, 4), "d"),                     # ... but not into a lane it threatens
    (_closing(5, 3) + _closing(4, 4) + _closing(6, 6), "n"),    # both neighbours threatened: stay
    (_closing(5, 3) + _closing(4, 0), "u"),                     # a far enemy in the next lane does not block
    (_leaving(5, 3), "n"),                                      # enemy leaving
    (_closing(5, 1), "n"),                                      # enemy still far away
    (_gold(3, 1) + _closing(4, 4), "n"),                        # gold two lanes up, threatened lane in between
    ([], "n"),
])
def test_asterix_circuit_rules(cells, want):
    assert action(K.asterix(window=2), PLAYER_5 + cells) == want


def test_asterix_window_sets_flee_distance():
    cells = PLAYER_5 + _closing(5, 2)                          # three columns away
    assert [action(K.asterix(w), cells) for w in (1, 2, 3)] == ["n", "n", "u"]


def sub_at(row: int) -> list[tuple[int, int, int]]:
    return [(0, row, 5), (1, row, 6)]


AIR = [(4, 9, 8)] + [(5, 9, c) for c in range(3)]            # a diver aboard, oxygen below 80


def diver(row: int, col: int, closing: bool = True) -> list[tuple[int, int, int]]:
    return [(4, row, col), (5, row, col - 1 if (col < 5) == closing else col + 1)]


@pytest.mark.parametrize("cells, want", [
    (sub_at(0), "d"),                                            # at the surface: dive
    (sub_at(1), "d"),                                            # row 1 is only a transit row
    (sub_at(6), "u"),                                            # below home: climb back
    (sub_at(4), "f"),                                            # at home, nothing around: fire
    (sub_at(4) + [(3, 6, 1)], "d"),                              # hazard on the left in row 6: go there
    (sub_at(4) + [(3, 2, 1)], "u"),
    (sub_at(4) + [(3, 4, 1)], "f"),                              # hazard on the left in my row: shoot
    (sub_at(4) + [(3, 4, 1), (3, 6, 1)], "f"),                   # ... rather than visit another row
    (sub_at(4) + diver(6, 2), "d"),                              # a diver on its way is a target too
    (sub_at(4) + diver(6, 2, closing=False), "f"),               # a leaving diver is not
    (sub_at(4) + diver(4, 2) + [(3, 6, 1)], "f"),                # wait for the diver in my row
    (sub_at(4) + [(3, 4, 4)], "u"),                              # hazard next to the sub: dodge
    (sub_at(4) + [(3, 4, 6), (3, 3, 4)], "d"),                   # ... not into a row that is threatened
    (sub_at(4) + [(3, 4, 6), (3, 3, 7)], "d"),                   # ... but into the calmer one
    (sub_at(4) + [(3, 4, 6), (3, 3, 4), (3, 5, 4)], "f"),        # both rows threatened: stay and shoot
    (sub_at(4) + [(3, 6, 1), (3, 5, 5)], "f"),                   # a hazard right in the next row blocks the visit
    (sub_at(4) + [(3, 4, 8)], "f"),                              # a hazard far on the right is ignored
    (sub_at(4) + AIR, "u"),                                      # diver aboard, low oxygen: surface
    (sub_at(8) + AIR, "u"),
    (sub_at(1) + AIR, "u"),
    (sub_at(4) + AIR + [(3, 4, 1)], "u"),                        # ... even with a target in the row
    (sub_at(4) + AIR + [(3, 3, 6)], "f"),                        # ... unless the next row is threatened
    (sub_at(4) + [(4, 9, 8)] + [(5, 9, c) for c in range(4)], "f"),    # enough oxygen
    (sub_at(4) + [(5, 9, c) for c in range(2)], "f"),                  # low oxygen but no diver: never surface
])
def test_seaquest_circuit_rules(cells, want):
    assert action(K.seaquest(home=4), cells) == want


def test_circuit_builder_capacity():
    net = K.CircuitBuilder()
    for _ in range(S.HIDDEN[0]):
        net.hidden1({0: 1.0})
    with pytest.raises(ValueError):
        net.hidden1({0: 1.0})


@pytest.mark.parametrize("game", S.GAMES)
def test_rescale_preserves_policy_and_hits_norm(game):
    rng = np.random.default_rng(1)
    x = (rng.random((500, S.N_IN)) < 0.05).astype(np.float32)
    target = S.frobenius_norm(S.init_weights(3))
    for make in K.CANDIDATES[game].values():
        raw = make()
        scaled = K.rescale(raw, target)
        assert S.frobenius_norm(scaled) == pytest.approx(target, rel=1e-4)
        np.testing.assert_array_equal(S.logits(raw, x).argmax(1), S.logits(scaled, x).argmax(1))


# ------------------------------------------------------------------------- rollout loop (fake env)
class _FakeEnv:
    """Terminates after 3 steps; reward 1 per action 3 ("r")."""

    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, a):
        self.t += 1
        return float(a == 3), self.t >= 3


def test_episode_returns_loop(monkeypatch):
    monkeypatch.setattr(R, "make_env", lambda game, seed: _FakeEnv())
    assert R.episode_returns("breakout", lambda env: 3, seed=0, episodes=2) == [3.0, 3.0]
    assert R.episode_returns("breakout", lambda env: 0, seed=0, episodes=1, max_steps=2) == [0.0]
    s = R.summarize([1.0, 3.0])
    assert s["mean"] == 2.0 and s["stderr"] == pytest.approx(1.0) and s["n"] == 2


# ------------------------------------------------------------------------- PPO baseline (math only)
def test_gae_hand_computed():
    from weightbench.tracks.tr04_minatar_lib import ppo

    rewards = np.array([[1.0], [0.0]])
    values = np.array([[0.5], [0.2]])
    dones = np.zeros((2, 1))
    adv, ret = ppo.gae(rewards, values, dones, last_value=np.array([0.4]), last_done=np.array([1.0]),
                       gamma=0.9, lam=0.5)
    # t=1: next is terminal → δ1 = 0 − 0.2 = −0.2, A1 = −0.2
    # t=0: δ0 = 1 + 0.9·0.2 − 0.5 = 0.68, A0 = 0.68 + 0.9·0.5·(−0.2) = 0.59
    np.testing.assert_allclose(adv[:, 0], [0.59, -0.2])
    np.testing.assert_allclose(ret, adv + values)


def test_ppo_actor_is_the_task_shell():
    from weightbench.tracks.tr04_minatar_lib import ppo

    actor, critic = ppo._networks(0)
    weights = ppo.actor_weights(actor)
    assert {k: v.shape for k, v in weights.items()} == S.PARAM_SHAPES      # a valid TR-04 submission
    assert S.validation_error(weights) is None


# --------------------------------------------------------------------------- task level (fake heavy)
CALIBRATION = {"zero_network": {"mean": 0.0}, "expert": {"mean": 10.0}, "uniform_random": {"mean": 0.5}}


@pytest.fixture
def built(tmp_path, monkeypatch, request):
    """Real ``build`` with the heavy steps faked: calibrate -> fixed numbers, reference -> the real
    selection code with fake dev returns (candidate order = quality order)."""
    game = getattr(request, "param", "seaquest")
    calls = []

    def fake_episode_returns(game_, act_fn, seed, episodes, max_steps=S.MAX_STEPS):
        calls.append(seed)
        return [float(len(calls))] * episodes

    def fake_heavy(art_dir, step, out_name, extra=(), push=()):
        if step == "calibrate":
            out = CALIBRATION
        elif step == "reference":
            out = J.reference(art_dir)
        else:
            raise AssertionError(f"unexpected heavy step {step}")
        (art_dir / out_name).write_text(json.dumps(out))
        return out

    monkeypatch.setattr(R, "episode_returns", fake_episode_returns)
    monkeypatch.setattr(T, "_heavy", fake_heavy)
    monkeypatch.setattr(T, "_pull_private", lambda art_dir: None)
    art = tmp_path / game
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    task = TASKS[game]()
    task.build(art, seed=0)
    return task, art


@pytest.mark.parametrize("built", S.GAMES, indirect=True)
def test_build_outputs_and_no_leakage(built):
    task, art = built
    public = {p.relative_to(art / "public").as_posix() for p in (art / "public").rglob("*") if p.is_file()}
    assert public == {f"game/{task.game}.py", "game/environment.py", "encoding.json"}
    spec = json.loads((art / "private" / "spec.json").read_text())
    assert spec["game"] == task.game and spec["seeds"] == S.task_seeds(0)
    info = json.loads((art / "private" / "reference_info.json").read_text())
    assert info["selected"] == list(K.CANDIDATES[task.game])[-1]          # highest fake dev return
    prompt = task.prompt(art)
    assert "submit-delta delta.npz" in prompt and "N_forward = 0" in prompt
    for value in spec["seeds"].values():
        assert str(value) not in prompt and str(value) not in (art / "public" / "encoding.json").read_text()


def test_reference_rescaled_to_init_norm(built):
    task, art = built
    ref = task.reference(art).delta
    assert {k: v.shape for k, v in ref.items()} == S.PARAM_SHAPES
    norm_ref = task.frobenius_reference(art)
    assert S.frobenius_norm(ref) == pytest.approx(S.frobenius_norm(norm_ref), rel=1e-4)
    lam = scoring.TrackConstants.for_track("TR04").lam
    cost = scoring.edit_cost(ref, task.base_weights(art), lam, zero_init_ref=norm_ref)
    assert cost["cost"] == pytest.approx(lam[2], rel=1e-3)                 # ‖W‖ = ‖W_init‖ → C = λ3


def test_evaluate_scoring_from_heavy_result(built, monkeypatch):
    task, art = built
    seen = {}

    def fake_heavy(art_dir, step, out_name, extra=(), push=()):
        with np.load(extra[extra.index("--weights") + 1]) as z:
            seen.update({k: z[k] for k in z.files})
        return {"mean": 4.0, "stderr": 0.5, "n": S.EVAL_EPISODES, "seconds": 1.0}

    monkeypatch.setattr(T, "_heavy", fake_heavy)
    delta = {"b3": np.array([0, 0, 0, 1, 0, 0], np.float32)}
    ev = task.evaluate(art, Submission("delta_w", delta=delta))
    assert ev.valid and ev.s_plus == pytest.approx(0.4)                 # (4 - 0) / (10 - 0): fixture planner ceiling 10
    assert set(seen) == set(S.PARAM_SHAPES) and seen["b3"][3] == 1 and not seen["W1"].any()


def test_evaluate_rejects_without_running(built, monkeypatch):
    task, art = built
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: pytest.fail("must not roll out invalid submissions"))
    assert not task.evaluate(art, Submission("report", report={})).valid
    assert not task.evaluate(art, Submission("delta_w", delta={"W1": np.zeros((2, 2))})).valid
    zero = task.evaluate(art, Submission("delta_w", delta={"W1": np.zeros((32, 600), np.float32)}))
    assert zero.valid and zero.s_plus == 0.0


def test_baseline_is_seeded_init_network(built):
    task, art = built
    a, b = task.baseline(art).delta, task.baseline(art).delta
    assert {k: v.shape for k, v in a.items()} == S.PARAM_SHAPES
    np.testing.assert_array_equal(a["W1"], b["W1"])
    assert not np.array_equal(a["W1"], task.frobenius_reference(art)["W1"])


def test_runner_edit_cost_uses_frobenius_reference(built, monkeypatch, tmp_path):
    from weightbench import runner

    task, art = built
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: {"mean": 10.0, "stderr": 0.0, "n": 1})
    led = Ledger(tmp_path / "ledger.jsonl")
    runner._evaluate_and_log(task, art, task.baseline(art), led)
    entry = next(led.entries("evaluation"))["payload"]
    assert entry["valid"] and entry["s_plus"] == 1.0  # mean 10 reaches the fixture planner ceiling (10)
    assert set(entry["edit_cost_terms"]) == {"l0", "rank", "fro", "max_rank", "cost"}
    lam3 = scoring.TrackConstants.for_track("TR04").lam[2]
    assert entry["edit_cost"] == pytest.approx(lam3, rel=0.05)          # init-scale network: ‖W‖ ≈ ‖W_init‖


def test_probes_rejected_and_ledgered(built, tmp_path):
    task, art = built
    led = Ledger(tmp_path / "ledger.jsonl")
    state = RunState(task, art, tmp_path / "run", led, tmp_path)
    for kind in ("rollout", "forward", "attention"):
        assert state.probe(kind, {})[0] == 400
    assert not list(led.entries("probe"))


def test_sandbox_denies_repo_and_private_artifacts():
    from weightbench import runner

    profile = runner._sandbox_profile(Path("/tmp/wb-work/x"))
    assert runner.sandbox.profile_denies_read(profile, runner.ROOT)
    assert runner.ART.resolve().is_relative_to(runner.ROOT.resolve())
    assert runner.ROOT.resolve() in Path(K.__file__).resolve().parents

def test_ceiling_is_strongest_known_policy():
    """S_max = max(planner, strongest known policy): PPO-100M beats the planner on Breakout / Asterix only."""
    cal = {"expert": {"mean": 12.53}, "zero_network": {"mean": 0.49}}
    s_max, source = T.BreakoutCompilation()._ceiling(cal)
    assert s_max == S.STRONGEST_KNOWN["breakout"]["return"] == 17.28 and "PPO" in source
    assert T.SeaquestCompilation()._ceiling({"expert": {"mean": 11.67}})[0] == 11.67       # planner stays (PPO 4.37 is weaker)
    assert T.SeaquestCompilation()._ceiling({"expert": {"mean": 12.70}})[0] == 12.70        # per-seed planner
    assert T.AsterixCompilation()._ceiling({"expert": {"mean": 9.63}}) == (25.03, S.STRONGEST_KNOWN["asterix"]["source"])
    assert T.AsterixCompilation()._ceiling({"expert": {"mean": 99.0}})[0] == 99.0           # a stronger planner wins
    # S+ of the hand-written Breakout reference (11.22) under the PPO ceiling
    assert scoring.s_plus_write(11.22, 0.49, s_max) == pytest.approx((11.22 - 0.49) / (17.28 - 0.49))

def test_minatar_rescore_uses_current_ceiling():
    old_details = {"S_edit": 11.22, "S_base": 0.49, "S_max": 12.53}      # ledger recorded with the planner ceiling
    assert T.BreakoutCompilation().rescore_s_plus(old_details) == pytest.approx((11.22 - 0.49) / (17.28 - 0.49))
    assert T.SeaquestCompilation().rescore_s_plus({"S_edit": 1.99, "S_base": 0.0, "S_max": 11.67}) == \
        pytest.approx(1.99 / 11.67)
    assert T.BreakoutCompilation().rescore_s_plus({"reason": "ΔW = 0"}) is None


def test_agent_usage_ignores_bare_json_scalars(tmp_path):
    """A custom (cmd) agent whose stdout is a bare number must not crash the usage reader (regression)."""
    from weightbench import runner

    t = tmp_path / "transcript.jsonl"
    t.write_text('45\n"text"\n{"type": "result", "num_turns": 2}\n', encoding="utf-8")
    assert runner._agent_usage(t) == {"total_cost_usd": None, "num_turns": 2, "duration_ms": None,
                                      "is_error": None, "usage": None}


def test_ceiling_never_encodes_reference_result():
    """No ceiling may come from the reference solution: reference circuit means (e.g. 16.68 / 36.73) never appear."""
    values = {k: v["return"] for k, v in S.STRONGEST_KNOWN.items()}
    assert values == {"breakout": 17.28, "asterix": 25.03}              # independent PPO-100M only; no seaquest entry
    assert 16.68 not in values.values() and 36.73 not in values.values()
    assert all("PPO" in v["source"] and "reference" not in v["source"] for v in S.STRONGEST_KNOWN.values())
    # a different reference return (stored in the calibration or not) leaves the ceiling unchanged
    for cal in ({"expert": {"mean": 11.67}}, {"expert": {"mean": 11.67}, "reference": {"mean": 99.0},
                                               "reference_return": 99.0}):
        assert T.SeaquestCompilation()._ceiling(cal)[0] == 11.67
        assert T.AsterixCompilation()._ceiling({**cal, "expert": {"mean": 9.63}})[0] == 25.03


def test_s_plus_hand_values_with_independent_ceilings():
    """S+ = clip((S - S_base) / (S_max - S_base), 0, 1) under the Seaquest (planner 11.67, base 0) and Asterix
    (PPO 25.03, base 0.49) ceilings."""
    assert scoring.s_plus_write(15.91, 0.0, 11.67) == 1.0                       # above ceiling clips to 1
    assert scoring.s_plus_write(36.47, 0.49, 25.03) == 1.0
    assert scoring.s_plus_write(0.49, 0.49, 25.03) == 0.0                       # S = S_base
    assert scoring.s_plus_write(0.2, 0.49, 25.03) == 0.0                        # below base clamps to 0
    assert scoring.s_plus_write(5.835, 0.0, 11.67) == pytest.approx(0.5)
    assert scoring.s_plus_write(12.76, 0.49, 25.03) == pytest.approx(0.5)       # (12.76 - 0.49) / 24.54
    # rescoring a ledger whose S_edit exceeds its recorded S_max clips
    assert T.SeaquestCompilation().rescore_s_plus({"S_edit": 15.91, "S_base": 0.0, "S_max": 11.67}) == 1.0
    assert T.AsterixCompilation().rescore_s_plus({"S_edit": 36.47, "S_base": 0.49, "S_max": 9.63}) == 1.0
    assert T.AsterixCompilation().rescore_s_plus({"reason": "ΔW = 0"}) is None  # ΔW = 0 has no S_edit: scored 0


# ------------------------------------------------------------- difficulty knob: observation delay (no real game)
def test_delay_levels_and_shapes():
    assert [S.delay(L) for L in S.LEVELS] == [0, 1, 3]                    # d = 2^L - 1
    with pytest.raises(ValueError, match="supported knob levels"):
        S.delay(3)
    for task in TASKS.values():
        assert task().level is None and task().delay == 0                 # unversioned default untouched
        assert task().shapes == S.PARAM_SHAPES
        with pytest.raises(ValueError, match="supported knob levels"):
            task().at_level(3)
        assert task().at_level(2).delay == 3
    assert S.param_shapes(3)["W1"] == (32, 618) and S.input_dim(1) == 606
    assert S.init_weights(5, S.param_shapes(0))["W1"].tobytes() == S.init_weights(5)["W1"].tobytes()


def test_delayed_observation_slots_hand_checked():
    frame = np.zeros(S.N_IN, np.float32)
    frame[7] = 1.0
    x = S.delayed_observation(frame, [1], 3)                              # t = 1 < d: one pending action
    assert x.shape == (618,) and x[7] == 1.0
    assert np.flatnonzero(x[600:]).tolist() == [6 * 2 + 1]                # last slot holds action 1
    x = S.delayed_observation(frame, [5, 0, 3], 3)
    assert np.flatnonzero(x[600:]).tolist() == [5, 6 + 0, 12 + 3]
    assert not S.delayed_observation(frame, [], 1)[600:].any()
    with pytest.raises(ValueError):
        S.delayed_observation(frame, [1, 2], 1)


class _Clock:
    """Deep-copyable fake env: a time counter and a paddle position moved by l (1) / r (3)."""

    def __init__(self):
        self.t, self.pos, self.sticky_action_prob, self.seeded = 0, 0, 0.1, None

    def seed(self, s):
        self.seeded = s

    def reset(self):
        self.t, self.pos = 0, 0

    def act(self, a):
        self.t += 1
        self.pos += {1: -1, 3: 1}.get(a, 0)
        return float(a == 3), self.t >= 5


def test_episode_returns_under_delay_shows_old_state_and_pending_actions(monkeypatch):
    monkeypatch.setattr(R, "make_env", lambda game, seed: _Clock())
    seen = []
    script = [3, 1, 3, 3, 0]

    def act(view):
        seen.append((view.observed.t, view.pending))
        return script[len(seen) - 1]

    assert R.episode_returns("breakout", act, seed=0, episodes=1, delay=2) == [3.0]
    assert seen == [(0, []), (0, [3]), (0, [3, 1]), (1, [1, 3]), (2, [3, 3])]   # frame t-2, actions t-2..t-1


def test_delayed_planner_plans_on_predicted_state_without_touching_view(monkeypatch):
    from weightbench.tracks.tr01_behavioral_lib import games as G

    planned = []

    def fake_expert(game, sim, rng):
        planned.append((sim.t, sim.pos, sim.sticky_action_prob, sim.seeded))
        return 4

    monkeypatch.setattr(G, "expert_action", fake_expert)
    env = _Clock()
    view = R.DelayView("breakout", 2, env)
    for a in (3, 3, 1):
        env.act(a)
        view.push(env, a)                                                 # true state t=3, pos 1
    assert view.observed.t == 1 and view.observed.pos == 1 and view.pending == [3, 1]
    assert R.delayed_expert_act_fn("breakout", seed=0)(view) == 4
    t, pos, sticky, seeded = planned[0]
    assert (t, pos, sticky) == (3, 1, 0.0) and seeded is not None         # replayed [3, 1] from pos 1 -> pos 1, t 3
    assert view.observed.t == 1 and view.observed.sticky_action_prob == 0.1 and view.observed.seeded is None


@pytest.fixture
def built_level(tmp_path, monkeypatch, request):
    """Real ``build`` at an explicit knob level, heavy steps faked; reference runs the real selection code."""
    game, level = request.param
    calls = []

    def fake_episode_returns(game_, act_fn, seed, episodes, max_steps=S.MAX_STEPS, delay=0):
        calls.append(delay)
        return [float(len(calls))] * episodes

    def fake_heavy(art_dir, step, out_name, extra=(), push=()):
        out = {**CALIBRATION, "delay": S.delay(level)} if step == "calibrate" else J.reference(art_dir)
        (art_dir / out_name).write_text(json.dumps(out))
        return out

    monkeypatch.setattr(R, "episode_returns", fake_episode_returns)
    monkeypatch.setattr(T, "_heavy", fake_heavy)
    monkeypatch.setattr(T, "_pull_private", lambda art_dir: None)
    art = tmp_path / f"{game}_L{level}"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    task = TASKS[game]().at_level(level)
    task.build(art, seed=0)
    return task, art, calls


@pytest.mark.parametrize("built_level", [("breakout", 1), ("asterix", 2), ("seaquest", 0)], indirect=True)
def test_level_build_layout_reference_and_baseline(built_level):
    task, art, calls = built_level
    d = task.delay
    public = {p.relative_to(art / "public").as_posix() for p in (art / "public").rglob("*") if p.is_file()}
    assert public == {f"game/{task.game}.py", "game/environment.py", "encoding.json", "knob.json"}
    knob = json.loads((art / "public" / "knob.json").read_text())
    assert knob["level"] == task.level and knob["observation_delay"] == d and knob["value"] == d + 1
    spec = json.loads((art / "private" / "spec.json").read_text())
    assert spec["level"] == task.level and spec["delay"] == d and spec["seeds"] == S.task_seeds(0)
    assert json.loads((art / "public" / "encoding.json").read_text())["param_shapes"]["W1"] == [32, S.input_dim(d)]
    assert set(calls) == {d}                                              # every dev rollout ran under the delay
    ref = task.reference(art).delta
    assert {k: v.shape for k, v in ref.items()} == S.param_shapes(d)
    assert bool(ref["W1"][:, S.N_IN:].any()) == bool(d)                   # delayed reference reads the action slots
    info = json.loads((art / "private" / "reference_info.json").read_text())
    assert info["selected"].startswith("dead_reckon") == bool(d)
    assert set(info["dev"]) == set(K.delayed_candidates(task.game, d) if d else K.CANDIDATES[task.game])
    assert S.frobenius_norm(ref) == pytest.approx(S.frobenius_norm(task.frobenius_reference(art)), rel=1e-4)
    base = task.baseline(art).delta
    assert {k: v.shape for k, v in base.items()} == S.param_shapes(d)
    assert S.validation_error(base, task.shapes) is None
    if d:
        assert S.validation_error(S.init_weights(1), task.shapes) is not None   # the 600-wide shell is rejected
    prompt = task.prompt(art)
    assert ("Observation delay" in prompt) == bool(d) and f"W1: [32, {S.input_dim(d)}]" in prompt
    for value in spec["seeds"].values():
        assert str(value) not in prompt and str(value) not in (art / "public" / "knob.json").read_text()


@pytest.mark.parametrize("built", ["breakout"], indirect=True)
def test_default_build_unchanged_and_equal_to_explicit_l0_prompt(built):
    task, art = built
    spec = json.loads((art / "private" / "spec.json").read_text())
    assert "level" not in spec and "delay" not in spec and not (art / "public" / "knob.json").exists()
    prompt = task.prompt(art)
    assert "Observation delay" not in prompt and "W1: [32, 600]  b1: [32]" in prompt
    assert TASKS["breakout"]().at_level(0).prompt(art) == prompt          # L0 = the default game, same text


def test_level_ceiling_is_independent_delayed_planner_per_level():
    """At L>0, S_max is the delayed planner measured for that build: never STRONGEST_KNOWN (undelayed) and never
    the reference; at L0 the default rule applies unchanged."""
    for task in TASKS.values():
        for level in S.LEVELS:
            t = task().at_level(level)
            for cal in ({"expert": {"mean": 3.5}}, {"expert": {"mean": 3.5}, "reference": {"mean": 99.0},
                                                     "reference_return": 99.0}):
                s_max, source = t._ceiling(cal)
                if level == 0:
                    assert (s_max, source) == task()._ceiling(cal)
                else:
                    assert s_max == 3.5 and "delayed planner" in source and "reference" not in source
    assert T.BreakoutCompilation().at_level(1).rescore_s_plus({"S_edit": 1.75, "S_base": 0.0, "S_max": 3.5}) == \
        pytest.approx(0.5)


def test_jobs_evaluate_and_calibrate_use_delay(tmp_path, monkeypatch):
    art = tmp_path / "a"
    (art / "private").mkdir(parents=True)
    spec = {"game": "breakout", "seeds": S.task_seeds(0), "eval_episodes": 2, "expert_episodes": 2,
            "dev_episodes": 2, "max_steps": 5, "delay": 3}
    (art / "private" / "spec.json").write_text(json.dumps(spec))
    seen = []

    def fake_episode_returns(game_, act_fn, seed, episodes, max_steps=S.MAX_STEPS, delay=0):
        seen.append((delay, getattr(act_fn, "__qualname__", "")))
        return [1.0] * episodes

    monkeypatch.setattr(R, "episode_returns", fake_episode_returns)
    np.savez(art / "w.npz", **S.zero_weights(S.param_shapes(3)))
    assert J.evaluate(art, art / "w.npz")["mean"] == 1.0
    cal = J.calibrate(art)
    assert cal["delay"] == 3 and "delayed planner" in cal["expert_kind"]
    assert [s[0] for s in seen] == [3, 3, 3, 3]
    assert "delayed_expert_act_fn" in seen[2][1]                          # the ceiling is the delayed planner


# ------------------------------------------------- delay-compensating reference circuits (no real game)----
def delayed_action(weights, cells, pending: list[str], d: int) -> str:
    x = S.delayed_observation(observation(cells), [S.ACTIONS.index(a) for a in pending], d)
    return S.ACTIONS[int(S.logits(weights, x[None])[0].argmax())]


def _ball_x_rule(col: int, right: bool, n: int) -> int:
    """Breakout's horizontal ball update, transcribed from ``Env.act``: step, clamp at a wall and flip."""
    for _ in range(n):
        new = col + (1 if right else -1)
        if new < 0 or new > 9:
            new, right = min(max(new, 0), 9), not right
        col = new
    return col


def test_ball_column_after_matches_the_game_rule():
    for col in range(10):
        for right in (True, False):
            for n in range(25):
                assert K.ball_column_after(col, right, n) == _ball_x_rule(col, right, n), (col, right, n)


# (ball col, trail col) -> moving right?  Interior diagonals, spawn cells (trail = ball) and both wall cases.
BALL_CASES = [(c, c - 1, True) for c in range(1, 10)] + [(c, c + 1, False) for c in range(0, 9)] + \
             [(0, 0, True), (9, 9, False)]


@pytest.mark.parametrize("d", [1, 3])
def test_breakout_delayed_moves_toward_predicted_ball_from_dead_reckoned_paddle(d):
    import itertools

    for ahead in (d, d + 1, d + 2):
        w = K.breakout_delayed(d, ahead)
        assert w["W1"].shape == (32, S.input_dim(d))
        for pending in itertools.product("nlrf", repeat=d):
            for shown in range(10):
                paddle = shown + pending.count("r") - pending.count("l")
                for ball, trail, right in BALL_CASES:
                    cells = [(0, 9, shown), (1, 4, ball), (2, 3, trail)]
                    target = _ball_x_rule(ball, right, ahead)
                    assert delayed_action(w, cells, list(pending), d) == _breakout_expected(paddle, target), \
                        (d, ahead, pending, shown, ball, trail)


def test_breakout_delayed_partial_slots_at_episode_start():
    """t < d: only the last len(pending) slots are filled; the paddle estimate uses exactly those."""
    w = K.breakout_delayed(3, 4)
    cells = [(0, 9, 4), (1, 3, 0), (2, 3, 0)]                 # spawn at column 0, moving right
    target = _ball_x_rule(0, True, 4)                         # 4
    assert delayed_action(w, cells, [], 3) == "n"             # paddle 4 == target 4
    assert delayed_action(w, cells, ["r"], 3) == "l"          # paddle 5 > 4
    assert delayed_action(w, cells, ["l", "l"], 3) == "r" and target == 4


def _random_scene(rng, channels, rows, k=6):
    return [(int(rng.choice(channels)), int(rng.choice(rows)), int(rng.integers(10))) for _ in range(k)]


def _assert_same_logits(w_delayed, w_l0, cells_shown, cells_now, pending, d):
    x = S.delayed_observation(observation(cells_shown), [S.ACTIONS.index(a) for a in pending], d)
    np.testing.assert_allclose(S.logits(w_delayed, x[None])[0], S.logits(w_l0, observation(cells_now)[None])[0],
                               atol=1e-3)


@pytest.mark.parametrize("d", [1, 3])
def test_asterix_delayed_equals_l0_circuit_on_dead_reckoned_lane(d):
    """The delayed circuit computes exactly what the L0 circuit computes with the player in the lane implied by the
    shown lane and the pending moves (clipping-free cases), and saturates at lanes 1 and 8."""
    rng = np.random.default_rng(0)
    for window in (1, 2, 3):
        w_d, w_0 = K.asterix(window, d=d), K.asterix(window)
        n = 0
        while n < 150:
            pending = [str(a) for a in rng.choice(list("nudlrf"), size=d)]
            shown = int(rng.integers(1, 9))
            now = shown + pending.count("d") - pending.count("u")
            if not 1 <= now <= 8:
                continue
            scene = _random_scene(rng, [1, 2, 3], range(1, 9))
            _assert_same_logits(w_d, w_0, [(0, shown, 5)] + scene, [(0, now, 5)] + scene, pending, d)
            n += 1
        scene = _closing(1, 4)
        _assert_same_logits(w_d, w_0, [(0, 1, 5)] + scene, [(0, 1, 5)] + scene, ["u"] * d, d)      # cur 1 - d -> lane 1
        _assert_same_logits(w_d, w_0, [(0, 8, 5)] + scene, [(0, 8, 5)] + scene, ["d"] * d, d)      # cur 8 + d -> lane 8


def test_asterix_delayed_flees_from_lane_it_already_moved_into():
    """Hand case: the shown frame has the player in lane 5 and a closing enemy in lane 4; the player already pressed
    'u' (it is in lane 4 now), so it must keep fleeing (not stay, as the reactive circuit on the old frame would)."""
    cells = PLAYER_5 + _closing(4, 3)
    assert action(K.asterix(2), cells) == "n"                                   # L0 view: enemy is in the next lane
    assert delayed_action(K.asterix(2, d=1), cells, ["u"], 1) == "u"            # now in lane 4: flee up (ties go up)
    assert delayed_action(K.asterix(2, d=1), cells, ["n"], 1) == "n"


@pytest.mark.parametrize("d", [1, 3])
def test_seaquest_delayed_equals_l0_circuit_on_dead_reckoned_row(d):
    rng = np.random.default_rng(1)
    for home in (3, 4, 5):
        w_d, w_0 = K.seaquest(home, d=d), K.seaquest(home)
        n = 0
        while n < 150:
            pending = [str(a) for a in rng.choice(list("nudlrf"), size=d)]
            shown = int(rng.integers(0, 9))
            now = shown + pending.count("d") - pending.count("u")
            if not 0 <= now <= 8:
                continue
            scene = _random_scene(rng, [3, 4, 5], range(1, 9)) + [c for c in AIR if rng.random() < 0.5]
            _assert_same_logits(w_d, w_0, sub_at(shown) + scene, sub_at(now) + scene, pending, d)
            n += 1
        _assert_same_logits(w_d, w_0, sub_at(0), sub_at(0), ["u"] * d, d)       # cur -d -> row 0
        _assert_same_logits(w_d, w_0, sub_at(8), sub_at(8), ["d"] * d, d)       # cur 8 + d -> row 8


def test_seaquest_delayed_does_not_surface_twice():
    """Shown in row 2 with a diver aboard and low air: the reactive circuit presses 'u'. After two pending 'u' the sub
    is at the surface (row 0) already; the delayed circuit dives instead of pressing 'u' again."""
    cells = sub_at(2) + AIR
    assert action(K.seaquest(4), cells) == "u"
    assert delayed_action(K.seaquest(4, d=3), cells, ["n", "u", "u"], 3) == "d"


@pytest.mark.parametrize("game", ["breakout", "asterix", "seaquest"])
@pytest.mark.parametrize("d", [1, 3])
def test_delayed_candidates_fit_the_shell_and_read_the_slots(game, d):
    cands = K.delayed_candidates(game, d)
    assert cands and all(name.startswith("dead_reckon") for name in cands)
    for build in cands.values():
        w = build()
        assert {k: v.shape for k, v in w.items()} == S.param_shapes(d)
        assert w["W1"][:, S.N_IN:].any()
    with pytest.raises(ValueError):
        K.delayed_candidates(game, 0)
