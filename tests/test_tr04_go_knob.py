"""TR04-GO-001 knob levels (``algorithmic_horizon``): capture reading on hand-built corner positions, level opponents,
the independent ceiling, explicit-level builds with fake heavy steps, default untouched. No real game is played."""
import json
from pathlib import Path

import numpy as np
import pytest

from weightbench.tracks import tr04_go as T
from weightbench.tracks.tr04_go_lib import board as B
from weightbench.tracks.tr04_go_lib import compiled as K
from weightbench.tracks.tr04_go_lib import jobs as J
from weightbench.tracks.tr04_go_lib import spec as S
from weightbench.tracks.tr04_go_lib import tactics as TT


def pt(coord: str) -> int:
    return (B.SIZE - int(coord[1:])) * B.SIZE + "abcdefghj".index(coord[0])


def position(black, white, to_play=B.BLACK) -> B.Board:
    b = B.Board()
    for c in black:
        b.stones[pt(c)] = B.BLACK
    for c in white:
        b.stones[pt(c)] = B.WHITE
    b.to_play = to_play
    return b


# ------------------------------------------------------------------------------------- reading mechanics
def test_depths_follow_the_knob():
    assert S.LEVELS == (0, 1, 2)
    assert [TT.opponent_depth(L) for L in S.LEVELS] == [0, 1, 3]
    assert [TT.ceiling_depth(L) for L in S.LEVELS] == [1, 2, 4]


def test_candidates_are_liberties_of_short_chains():
    b = position(["a8"], ["a9"])                    # white a9: 1 liberty (b9); black a8: 2 liberties (a7, b8)
    assert TT.candidate_moves(b) == sorted([pt("b9"), pt("a7"), pt("b8")])
    assert TT.candidate_moves(B.Board()) == []


def test_depth_one_captures_and_placing_a_stone_is_worth_nothing():
    b = position(["a8"], ["a9"])
    assert TT.Reader(1).choose(b) == pt("b9")       # immediate capture: +1 over passing
    assert TT.Reader(0).choose(b) is None
    r = TT.Reader(1)
    assert r._line(b, pt("a7"), 0) == 0 and r._line(b, pt("b9"), 0) == 1


def test_depth_two_escapes_atari_depth_one_does_not():
    # Black e5 in atari (only liberty e4); no white chain is short, so nothing can be captured right away.
    b = position(["e5"], ["d5", "f5", "e6"])
    assert TT.Reader(1).choose(b) is None           # extending gains nothing within one ply
    assert TT.Reader(2).choose(b) == pt("e4")       # passing loses e5 at ply 2; e4 keeps it (three liberties)


def test_depth_three_finds_atari_then_capture():
    # White a9 has two liberties (a8, b9); black a7 and b8 make a8 unplayable for white after black b9.
    b = position(["a7", "b8"], ["a9"])
    assert TT.Reader(1).choose(b) is None and TT.Reader(2).choose(b) is None
    assert TT.Reader(3).choose(b) == pt("b9")       # b9 and a8 both win a stone at ply 3; lowest index (b9 = 1)


def test_node_cap_bounds_work_and_is_deterministic():
    b = position(["a7", "b8", "e5"], ["a9", "d5", "f5", "e6"])
    r1, r2 = TT.Reader(3, node_cap=5), TT.Reader(3, node_cap=5)
    assert r1.choose(b) == r2.choose(b)
    assert r1.nodes < 200
    with pytest.raises(ValueError):
        TT.Reader(-1)


def test_level_zero_opponent_is_the_default_random_opponent():
    rng = np.random.default_rng(3)
    for k in range(5):
        b = B.Board()
        stones = rng.choice(81, 20, replace=False)
        b.stones[stones[:10]], b.stones[stones[10:]] = B.BLACK, B.WHITE
        a, c = S.random_player(100 + k), S.opponent_factory(0)(100 + k)
        assert [a(b) for _ in range(3)] == [c(b) for _ in range(3)]
    assert S.opponent_factory(None) is S.random_player
    with pytest.raises(ValueError):
        S.opponent_factory(3)


def test_reading_opponent_plays_the_tactic_and_ceiling_reads_deeper():
    assert S.opponent_factory(2)(0)(position(["a7", "b8"], ["a9"])) == pt("b9")     # level 2 reads 3 plies
    assert S.opponent_factory(1)(0)(position(["a8"], ["a9"])) == pt("b9")           # level 1 reads 1 ply
    assert TT.ceiling_player(0, TT.ceiling_depth(1))(position(["e5"], ["d5", "f5", "e6"])) == pt("e4")
    move = TT.ceiling_player(0, 1)(B.Board())                                       # area-greedy fallback
    assert B.Board().is_legal(move) and move != B.PASS


def test_ceiling_is_independent_of_the_reference():
    import ast

    tree = ast.parse(Path(TT.__file__).read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert imported <= {"annotations", "Callable", "np", "numpy", "board"}       # no compiled / reference
    src = Path(J.__file__).read_text(encoding="utf-8")
    calib = src[src.index("def calibrate"):src.index("def reference")]
    assert "reference.npz" not in calib and "K." not in calib


# ------------------------------------------------------------------------------------- jobs at a level
def test_calibrate_at_level_measures_ceiling_against_level_opponent(tmp_path, monkeypatch):
    (tmp_path / "private").mkdir()
    spec = {"seeds": {"eval": 5, "ceiling": 9}, "eval_games": 4, "level": 1}
    (tmp_path / "private" / "spec.json").write_text(json.dumps(spec))
    calls = []

    def fake_match(weights, seed, n, opponent=S.random_player, player=None):
        calls.append((weights is None, player is not None, opponent is S.random_player))
        return {"win_rate": 0.7 if player is not None else 0.05, "stderr": 0.0, "n": n, "mean_margin": 0.0}

    monkeypatch.setattr(S, "match", fake_match)
    out = J.calibrate(tmp_path)
    assert out["ceiling"]["win_rate"] == 0.7 and out["opponent_depth"] == 1 and "depth 2" in out["ceiling"]["player"]
    assert calls == [(False, False, False), (True, True, False)]    # both against the level opponent


def test_default_calibrate_is_unchanged(tmp_path, monkeypatch):
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "spec.json").write_text(json.dumps({"seeds": {"eval": 5}, "eval_games": 4}))
    monkeypatch.setattr(S, "match", lambda w, s, n, opponent=S.random_player, player=None:
                        {"win_rate": float(opponent is S.random_player)})
    assert J.calibrate(tmp_path) == {"zero_network": {"win_rate": 1.0}}


def test_game_probe_job_uses_the_level_opponent(tmp_path, monkeypatch):
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "spec.json").write_text(json.dumps({"seeds": {}, "level": 2}))
    np.savez(tmp_path / "w.npz", **S.zero_weights())
    seen = []
    monkeypatch.setattr(TT, "reading_opponent", lambda seed, depth: seen.append(depth) or S.random_player(seed))
    monkeypatch.setattr(S, "play_game", lambda black, white: (B.Board(), [B.PASS]))
    J.game(tmp_path / "w.npz", {"color": "white", "seed": 1}, tmp_path)
    assert seen == [3]


# ------------------------------------------------------------------------------------- task level (fake heavy)
def _build(tmp_path, monkeypatch, level):
    def fake_heavy(art_dir, step, out_name, extra=(), push=()):
        spec = json.loads((art_dir / "private" / "spec.json").read_text())
        if step == "calibrate":
            out = {"zero_network": {"win_rate": 0.1}}
            if "level" in spec:
                out["ceiling"] = {"win_rate": 0.6}
        elif step == "reference":
            np.savez(art_dir / "private" / "reference.npz", **K.rescale(K.go_policy(True), 30.0))
            out = {"selected": "capture_eye_selfatari"}
        else:
            raise AssertionError(step)
        (art_dir / out_name).write_text(json.dumps(out))
        return out

    monkeypatch.setattr(T, "_heavy", fake_heavy)
    monkeypatch.setattr(T, "_pull_private", lambda art_dir: None)
    art = tmp_path / f"art{level}"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    task = T.GoPolicySynthesis().at_level(level)
    task.build(art, seed=0)
    return task, art


def test_default_build_spec_and_prompt_unchanged(tmp_path, monkeypatch):
    task, art = _build(tmp_path, monkeypatch, None)
    spec = json.loads((art / "private" / "spec.json").read_text())
    assert spec == {"task": "TR04-GO-001", "seed": 0, "seeds": S.task_seeds(0), "eval_games": S.EVAL_GAMES,
                    "komi": B.KOMI, "max_moves": B.MAX_MOVES}
    assert not (art / "public" / "knob.json").exists() and not (art / "public" / "go" / "tactics.py").exists()
    assert task.knob_level is None and task.level is None
    prompt = task.prompt(art)
    assert ("- Opponent: uniform-random over legal moves that do not fill its own single-point eyes; it passes only "
            "when\n  no such move exists.\n- S = win rate over 200 games on private seeds, half as black, half as "
            "white.\n- `S_base` = 0.100: the all-zero network (plays the lowest-index legal point).\n  `S+ = clip((S − "
            "S_base)/(1 − S_base), 0, 1)`.\n") in prompt
    assert "S_max" not in prompt and "knob level" not in prompt


@pytest.mark.parametrize("level", S.LEVELS)
def test_explicit_level_build_prompt_and_scoring(tmp_path, monkeypatch, level):
    task, art = _build(tmp_path, monkeypatch, level)
    knob = json.loads((art / "public" / "knob.json").read_text())
    assert knob["level"] == level and knob["value"] == 2 ** level
    assert knob["opponent_read_plies"] == 2 ** level - 1 and knob["ceiling_read_plies"] == 2 ** level
    public = {p.relative_to(art / "public").as_posix() for p in (art / "public").rglob("*") if p.is_file()}
    assert public == {"go/board.py", "go/tactics.py", "knob.json"}     # nothing private in public at any level
    spec = json.loads((art / "private" / "spec.json").read_text())
    assert spec["level"] == level and spec["seeds"]["ceiling"] == S.sub_seed(0, 6)
    prompt = task.prompt(art)
    assert f"knob level {level}: H = {2 ** level}" in prompt and "`S_max` = 0.600" in prompt
    for v in spec["seeds"].values():
        assert str(v) not in prompt
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: {"win_rate": 0.35, "stderr": 0.03, "n": 200, "mean_margin": 1.0})
    ev = task.evaluate(art, task.reference(art))
    assert ev.s_plus == pytest.approx((0.35 - 0.1) / (0.6 - 0.1)) and ev.details["S_max"] == 0.6
    assert task.baseline(art).delta.keys() == S.PARAM_SHAPES.keys()


def test_ceiling_below_zero_network_scores_zero(tmp_path, monkeypatch):
    task, art = _build(tmp_path, monkeypatch, 1)
    (art / "private" / "calibration.json").write_text(json.dumps({"zero_network": {"win_rate": 0.5},
                                                                   "ceiling": {"win_rate": 0.4}}))
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: {"win_rate": 0.9, "stderr": 0.0, "n": 200, "mean_margin": 1.0})
    ev = task.evaluate(art, task.reference(art))
    assert ev.s_plus == 0.0 and "warning" in ev.details


def test_unsupported_level_rejected_before_build():
    with pytest.raises(ValueError, match="supported knob levels"):
        T.GoPolicySynthesis().at_level(3)
    from weightbench import runner

    with pytest.raises(ValueError):
        runner.nondefault_level("TR04-GO-001", 3)
