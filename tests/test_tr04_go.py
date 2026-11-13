"""TR04-GO-001: Go rules, encoding, compiled reference circuit, probe/file handling, scoring, leakage.

No policy plays a game here (rollouts run through weightbench.heavy). The engine is exercised with
hand-built positions and one random-vs-random game.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from weightbench import scoring
from weightbench.ledger import Ledger
from weightbench.server import RunState
from weightbench.task import Submission
from weightbench.tracks import tr04_go as T
from weightbench.tracks.tr04_go_lib import board as B
from weightbench.tracks.tr04_go_lib import compiled as K
from weightbench.tracks.tr04_go_lib import jobs as J
from weightbench.tracks.tr04_go_lib import spec as S


def pt(coord: str) -> int:
    """'a9' = top-left (row 0, col 0)."""
    return (B.SIZE - int(coord[1:])) * B.SIZE + "abcdefghj".index(coord[0])


def position(black: list[str], white: list[str], to_play: int = B.BLACK) -> B.Board:
    b = B.Board()
    for c in black:
        b.stones[pt(c)] = B.BLACK
    for c in white:
        b.stones[pt(c)] = B.WHITE
    b.to_play = to_play
    return b


# ------------------------------------------------------------------------------------- rules
def test_capture_and_area():
    b = position(["a8", "b9"], ["a9"])                 # white corner stone in atari
    assert b.play(pt("c9")) == 0 and b.stones[pt("a9")] == B.WHITE
    b = position(["a8"], ["a9"])
    assert b.play(pt("b9")) == 1 and b.stones[pt("a9")] == B.EMPTY
    black_area, white_area = b.area()
    assert white_area == 0 and black_area == 81      # every empty region borders only black


def test_suicide_forbidden_unless_capturing():
    b = position([], ["a8", "b9"])
    assert not b.is_legal(pt("a9"))
    b = position(["a7", "b8", "c9"], ["a8", "b9"])   # playing a9 captures both white stones
    assert b.is_legal(pt("a9")) and b.play(pt("a9")) == 2


def test_ko_point_blocks_immediate_recapture():
    # Rows 9-7, columns a-d:   . W B .   /   W . W B   /   . W B .
    # Black b8 captures white c8; white may not retake at c8 until another move pair has been played.
    b = position(["c9", "d8", "c7"], ["b9", "a8", "c8", "b7"])
    b.to_play = B.BLACK
    assert b.play(pt("b8")) == 1                       # captures white c8
    assert b.ko == pt("c8") and not b.is_legal(pt("c8"))
    b.play(pt("j1"))                                  # white elsewhere
    b.play(pt("j2"))                                  # black elsewhere
    assert b.ko is None and b.is_legal(pt("c8"))       # ko lifted after an exchange


def test_game_ends_and_winner_is_consistent():
    board, moves = S.play_game(S.random_player(1), S.random_player(2))
    assert board.over and len(moves) == board.n_moves <= B.MAX_MOVES
    black, white = board.area()
    assert board.winner() == (B.BLACK if black - white - B.KOMI > 0 else B.WHITE)


def test_encode_is_from_movers_view():
    b = position(["e5"], ["d4"], to_play=B.WHITE)
    x = B.encode(b)
    assert x.shape == (243,) and x[pt("d4")] == 1 and x[81 + pt("e5")] == 1 and x[162:].sum() == 79


# ------------------------------------------------------------------------------- architecture
def test_param_count_and_shapes():
    assert sum(int(np.prod(s)) for s in S.PARAM_SHAPES.values()) == 1_480_274
    assert S.PARAM_SHAPES["W1"] == (512, 243) and S.PARAM_SHAPES["W7"] == (82, 512)
    assert S.validation_error({"W8": np.zeros(1)}) and S.validation_error({"b7": np.zeros(81)})
    assert S.validation_error({"W2": np.zeros((512, 512))}) is None


def _move(weights, board) -> int:
    return S.policy_player(weights)(board)


@pytest.mark.parametrize("avoid_self_atari", [True, False])
def test_compiled_captures_lone_stone(avoid_self_atari):
    w = K.go_policy(avoid_self_atari)
    b = position(["d5", "e6", "f5"], ["e5"])            # white e5 has one liberty at e4
    assert _move(w, b) == pt("e4")


def test_compiled_never_fills_own_eye_and_passes_instead():
    w = K.go_policy(True)
    b = position([], [])
    b.stones[:] = B.BLACK
    for c in ("a9", "c9"):                               # two single-point eyes, nothing else to play
        b.stones[pt(c)] = B.EMPTY
    b.to_play = B.BLACK
    assert _move(w, b) == B.PASS


def test_compiled_prefers_centre_on_empty_board():
    assert _move(K.go_policy(True), B.Board()) == pt("e5")


def test_compiled_avoids_self_atari():
    b = position([], ["d5", "e6", "f5"])                  # e5 would be a one-liberty black stone
    w_avoid, w_plain = K.go_policy(True), K.go_policy(False)
    z_avoid = S.logits(w_avoid, B.encode(b)[None])[0]
    z_plain = S.logits(w_plain, B.encode(b)[None])[0]
    assert z_avoid[pt("e5")] < z_plain[pt("e5")] - 2.5


def test_rescale_preserves_policy():
    w = K.go_policy(True)
    target = S.frobenius_norm(S.init_weights(3))
    r = K.rescale(w, target)
    assert S.frobenius_norm(r) == pytest.approx(target, rel=1e-4)
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = B.Board()
        stones = rng.choice(81, 30, replace=False)
        b.stones[stones[:15]], b.stones[stones[15:]] = B.BLACK, B.WHITE
        assert _move(w, b) == _move(r, b)


# --------------------------------------------------------------------------- task level (fake heavy)
@pytest.fixture
def built(tmp_path, monkeypatch):
    def fake_heavy(art_dir, step, out_name, extra=(), push=()):
        if step == "calibrate":
            out = {"zero_network": {"win_rate": 0.1, "stderr": 0.02, "n": 200}}
        elif step == "reference":
            np.savez(art_dir / "private" / "reference.npz", **K.rescale(K.go_policy(True), 30.0))
            out = {"selected": "capture_eye_selfatari"}
        else:
            raise AssertionError(step)
        (art_dir / out_name).write_text(json.dumps(out))
        return out

    monkeypatch.setattr(T, "_heavy", fake_heavy)
    monkeypatch.setattr(T, "_pull_private", lambda art_dir: None)
    art = tmp_path / "art"
    (art / "public").mkdir(parents=True)
    (art / "private").mkdir()
    task = T.GoPolicySynthesis()
    task.build(art, seed=0)
    return task, art


def test_build_and_no_leakage(built):
    task, art = built
    public = {p.relative_to(art / "public").as_posix() for p in (art / "public").rglob("*") if p.is_file()}
    assert public == {"go/board.py"}
    prompt = task.prompt(art)
    assert "python wb.py probe game" in prompt and "submit-delta delta.npz" in prompt and "0.100" in prompt
    for v in json.loads((art / "private" / "spec.json").read_text())["seeds"].values():
        assert str(v) not in prompt


def test_evaluate_scoring_and_rejections(built, monkeypatch):
    task, art = built
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: {"win_rate": 0.82, "stderr": 0.03, "n": 200, "mean_margin": 9.0})
    ev = task.evaluate(art, task.reference(art))
    assert ev.valid and ev.s_plus == pytest.approx((0.82 - 0.1) / 0.9)
    monkeypatch.setattr(T, "_heavy", lambda *a, **k: pytest.fail("must not play invalid submissions"))
    assert not task.evaluate(art, Submission("delta_w", delta={"W1": np.zeros((3, 3))})).valid
    assert task.evaluate(art, Submission("delta_w", delta={"W1": np.zeros((512, 243))})).s_plus == 0.0


def test_reference_edit_cost(built):
    task, art = built
    c = scoring.edit_cost(task.reference(art).delta, task.base_weights(art), (0.0, 0.0, 0.1),
                          zero_init_ref={"w": np.full(1, 30.0)})
    assert c["cost"] == pytest.approx(0.1, rel=1e-3)


def test_game_probe_reads_workdir_file_and_budget(built, tmp_path, monkeypatch):
    task, art = built
    work = tmp_path / "work"
    work.mkdir()
    np.savez(work / "cand.npz", b7=np.ones(82, np.float32))
    seen = []
    monkeypatch.setattr(T, "_heavy", lambda a, step, out, extra=(), push=(): seen.append((step, extra)) or {"won": True})
    led = Ledger(tmp_path / "ledger.jsonl")
    state = RunState(task, art, tmp_path / "run", led, work)
    assert state.probe("game", {"file": "../outside.npz", "color": "black", "seed": 1})[0] == 400   # not charged
    assert "error" in state.probe("game", {"file": "cand.npz", "color": "red", "seed": 1})[1]["result"]
    codes = [state.probe("game", {"file": "cand.npz", "color": "white", "seed": 3})[0] for _ in range(S.N_FORWARD)]
    assert codes[:S.N_FORWARD - 1] == [200] * (S.N_FORWARD - 1) and codes[-1] == 429
    step, extra = seen[0]
    assert step == "game"
    with np.load(extra[extra.index("--weights") + 1]) as z:
        assert set(z.files) == set(S.PARAM_SHAPES) and z["b7"].sum() == 82   # completed with zeros


def test_game_job_record_format(tmp_path, monkeypatch):
    np.savez(tmp_path / "w.npz", **S.zero_weights())
    monkeypatch.setattr(S, "play_game", lambda black, white: (B.Board(), [B.PASS, pt("a9"), pt("j1")]))
    out = J.game(tmp_path / "w.npz", {"color": "black", "seed": 1})
    assert out["moves"] == ["pass", "a9", "j1"] and out["policy_color"] == "black" and out["komi"] == B.KOMI


def test_sandbox_denies_repo():
    from weightbench import runner

    profile = runner._sandbox_profile(Path("/tmp/wb-work/x"))
    assert runner.sandbox.profile_denies_read(profile, runner.ROOT)
