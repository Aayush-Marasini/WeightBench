"""Heavy (game-playing) steps for TR04-GO-001. Run through ``weightbench.heavy.run_heavy``.

    python -m weightbench.tracks.tr04_go_lib.jobs calibrate --art ART --out ART/private/calibration.json
    python -m weightbench.tracks.tr04_go_lib.jobs reference --art ART --out ART/private/reference_info.json
    python -m weightbench.tracks.tr04_go_lib.jobs evaluate  --art ART --weights W.npz --out OUT.json
    python -m weightbench.tracks.tr04_go_lib.jobs game      --art ART --weights W.npz --payload P.json --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import board as B
from . import compiled as K
from . import spec as S

DEV_GAMES = 40


def _spec(art: Path) -> dict:
    return json.loads((art / "private" / "spec.json").read_text())


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _opponent(sp: dict):
    """The build's fixed opponent: random (default spec, no ``level`` key) or the level-L capture reader."""
    return S.opponent_factory(sp.get("level"))


def calibrate(art: Path) -> dict:
    """S_base: the all-zero network (plays the lowest-index legal point) on the eval seeds. At an explicit knob
    level also the independent ceiling S_max(L): a capture reader one ply deeper than the opponent (2^L plies) with an
    area-greedy fallback, on the same eval seeds and opponent (never the task's reference)."""
    sp = _spec(art)
    opp = _opponent(sp)
    out = {"zero_network": S.match(S.zero_weights(), sp["seeds"]["eval"], sp["eval_games"], opp)}
    if "level" in sp:
        from . import tactics as T

        depth = T.ceiling_depth(sp["level"])
        ceiling = T.ceiling_player(sp["seeds"]["ceiling"], depth)
        out["ceiling"] = {**S.match(None, sp["seeds"]["eval"], sp["eval_games"], opp, player=ceiling),
                          "player": f"capture reader depth {depth} + area-greedy fallback"}
        out["opponent_depth"] = T.opponent_depth(sp["level"])
    return out


def reference(art: Path) -> dict:
    sp = _spec(art)
    opp = _opponent(sp)
    dev = {name: S.match(make(), sp["seeds"]["dev"], DEV_GAMES, opp) for name, make in K.CANDIDATES.items()}
    best = max(dev, key=lambda n: dev[n]["win_rate"])
    target = S.frobenius_norm(S.init_weights(sp["seeds"]["norm_init"]))
    weights = K.rescale(K.CANDIDATES[best](), target)
    np.savez(art / "private" / "reference.npz", **weights)
    return {"selected": best, "dev": dev, "target_norm": target, "reference_norm": S.frobenius_norm(weights)}


def evaluate(art: Path, weights_path: Path) -> dict:
    sp = _spec(art)
    return S.match(_load(weights_path), sp["seeds"]["eval"], sp["eval_games"], _opponent(sp))


def _coord(move: int) -> str:
    if move == B.PASS:
        return "pass"
    row, col = divmod(move, B.SIZE)
    return "abcdefghj"[col] + str(B.SIZE - row)


def game(weights_path: Path, payload: dict, art: Path | None = None) -> dict:
    """One metered rollout: the agent's network vs the build's opponent, full move record."""
    policy = S.policy_player(_load(weights_path))
    opp = (_opponent(_spec(art)) if art is not None else S.random_player)(int(payload["seed"]))
    policy_black = payload["color"] == "black"
    board, moves = S.play_game(policy if policy_black else opp, opp if policy_black else policy)
    area_b, area_w = board.area()
    won = board.winner() == (B.BLACK if policy_black else B.WHITE)
    return {"policy_color": payload["color"], "won": won, "black_area": area_b, "white_area": area_w,
            "komi": B.KOMI, "n_moves": len(moves), "moves": [_coord(m) for m in moves]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calibrate", "reference", "evaluate", "game"])
    ap.add_argument("--art", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--payload")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    art = Path(a.art)
    t0 = time.time()
    if a.step == "calibrate":
        out = calibrate(art)
    elif a.step == "reference":
        out = reference(art)
    elif a.step == "evaluate":
        out = evaluate(art, Path(a.weights))
    else:
        out = game(Path(a.weights), json.loads(Path(a.payload).read_text()), art)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
