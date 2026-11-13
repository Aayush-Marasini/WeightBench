"""Heavy (rollout) steps for TR04-GAME-001..003. Run through ``weightbench.heavy.run_heavy``.

    python -m weightbench.tracks.tr04_minatar_lib.jobs calibrate --art ART --out ART/private/calibration.json
    python -m weightbench.tracks.tr04_minatar_lib.jobs reference --art ART --out ART/private/reference_info.json
    python -m weightbench.tracks.tr04_minatar_lib.jobs evaluate  --art ART --weights W.npz --out OUT.json

Every step reads ``ART/private/spec.json`` (game, seeds, episode counts) written by the task's
``build``, and writes only under ART.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import compiled as K
from . import rollout as R
from . import spec as S


def _spec(art: Path) -> dict:
    return json.loads((art / "private" / "spec.json").read_text())


def _delay(sp: dict) -> int:
    """Observation delay of this build (0 for the default build, whose spec has no "delay" key)."""
    return int(sp.get("delay", 0))


def _kw(d: int) -> dict:
    """Extra ``episode_returns`` arguments: none for the default build."""
    return {"delay": d} if d else {}


def _policy(game: str, weights: dict, d: int) -> R.ActFn:
    return R.delayed_policy_act_fn(weights) if d else R.policy_act_fn(game, weights)


def calibrate(art: Path) -> dict:
    """S_base (all-zero network = always no-op) and the planner ceiling S_max, on eval seeds. Under a delay
    (knob level > 0) the ceiling is the delayed planner (``rollout.delayed_expert_act_fn``) and the uniform floor
    ignores its view."""
    sp = _spec(art)
    game, seeds, d = sp["game"], sp["seeds"], _delay(sp)
    shapes = S.param_shapes(d)
    zero = R.episode_returns(game, _policy(game, S.zero_weights(shapes), d), seeds["eval"],
                             sp["eval_episodes"], sp["max_steps"], **_kw(d))
    expert_fn = R.delayed_expert_act_fn(game, seeds["expert"]) if d else R.expert_act_fn(game, seeds["expert"])
    expert = R.episode_returns(game, expert_fn, seeds["eval"], sp["expert_episodes"], sp["max_steps"], **_kw(d))
    rng = np.random.default_rng(seeds["expert"])
    uniform = R.episode_returns(game, lambda env: int(rng.integers(S.N_ACTIONS)), seeds["eval"],
                                sp["eval_episodes"], sp["max_steps"], **_kw(d))
    out = {"game": game, "zero_network": R.summarize(zero), "expert": R.summarize(expert),
           "uniform_random": R.summarize(uniform)}
    if d:
        out.update(delay=d, expert_kind="delayed planner: default planner on the state predicted from the shown "
                                        "frame and the pending actions")
    return out


def reference(art: Path) -> dict:
    """Pick the best hand-compiled candidate on dev seeds; save it rescaled to init-network norm. Under a delay the
    candidates are the delay-compensating circuits (``compiled.delayed_candidates``): the L0 circuits with the agent's
    own position dead-reckoned from the shown frame and the action slots."""
    sp = _spec(art)
    game, d = sp["game"], _delay(sp)
    shapes = S.param_shapes(d)
    candidates = K.delayed_candidates(game, d) if d else K.CANDIDATES[game]

    def make(name: str) -> dict:
        return candidates[name]()

    dev = {}
    for name in candidates:
        returns = R.episode_returns(game, _policy(game, make(name), d), sp["seeds"]["dev"],
                                    sp["dev_episodes"], sp["max_steps"], **_kw(d))
        dev[name] = R.summarize(returns)
    best = max(dev, key=lambda n: dev[n]["mean"])
    target_norm = S.frobenius_norm(S.init_weights(sp["seeds"]["norm_init"], shapes))
    weights = K.rescale(make(best), target_norm)
    np.savez(art / "private" / "reference.npz", **weights)
    out = {"game": game, "selected": best, "dev": dev, "target_norm": target_norm,
           "reference_norm": S.frobenius_norm(weights)}
    if d:
        out.update(delay=d, reference_kind="delay-compensating circuit: own position dead-reckoned from the shown "
                                           "frame plus the action slots")
    return out


def evaluate(art: Path, weights_path: Path) -> dict:
    sp = _spec(art)
    d = _delay(sp)
    with np.load(weights_path) as z:
        weights = {k: z[k] for k in z.files}
    returns = R.episode_returns(sp["game"], _policy(sp["game"], weights, d), sp["seeds"]["eval"],
                                sp["eval_episodes"], sp["max_steps"], **_kw(d))
    return {"game": sp["game"], **R.summarize(returns)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calibrate", "reference", "evaluate"])
    ap.add_argument("--art", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    art = Path(a.art)
    t0 = time.time()
    if a.step == "calibrate":
        out = calibrate(art)
    elif a.step == "reference":
        out = reference(art)
    else:
        out = evaluate(art, Path(a.weights))
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
