"""Heavy steps for TR05-DEAD-001/002 (enumeration over {0,1}^n; CPU). Run through ``weightbench.heavy.run_heavy``.

    python -m weightbench.tracks.tr05_dead_lib.jobs build    --art ART --out ART/private/build.json
    python -m weightbench.tracks.tr05_dead_lib.jobs evaluate --art ART --weights DELTA.npz --out OUT.json
    python -m weightbench.tracks.tr05_dead_lib.jobs reference_crown --art ART --out ART/private/reference_crown.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import crown as C
from . import net as N

TOLERANCE = 1e-5     # max |Δ output| allowed for a pruning to count as function-preserving
LP_BUDGET_S = 2700.0  # reference_crown: LP wall-clock budget, below the 3600 s _heavy timeout (fixed)


def _spec(art: Path) -> dict:
    return json.loads((art / "private" / "spec.json").read_text())


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def build(art: Path) -> dict:
    sp = _spec(art)
    layers, plants = N.generate(sp["n_inputs"], tuple(sp["hidden"]), sp["n_dead"], sp["n_decoys"], sp["seed"])
    np.savez(art / "public" / "mlp.npz", **N.to_dict(layers))
    dead = N.dead_units(layers, sp["n_inputs"])
    provable = N.ibp_dead(layers)
    (art / "private" / "truth.json").write_text(json.dumps({"dead": dead, "ibp_dead": provable, "plants": plants}))
    np.savez(art / "private" / "reference.npz", **N.prune(layers, provable))
    w_out = [np.linalg.norm(layers[i + 1][0], axis=0) for i in range(len(layers) - 1)]
    k = [max(1, len(w) // 10) for w in w_out]
    magnitude = [np.argsort(w)[:kk].tolist() for w, kk in zip(w_out, k)]
    np.savez(art / "private" / "baseline.npz", **N.prune(layers, magnitude))
    return {"n_dead": [len(d) for d in dead], "n_ibp_dead": [len(p) for p in provable],
            "n_planted_dead": [len(p) for p in plants["dead"]], "n_decoys": [len(p) for p in plants["decoy"]]}


def reference_crown(art: Path, lp_budget_s: float | None = None) -> dict:
    """Second reference (``crown.py``): back-substitution + per-unit LP, from the public weights only. Never runs
    the network and never reads truth.json; writes private/reference_crown.npz."""
    layers = N.from_dict(_load(art / "public" / "mlp.npz"))
    dead, info = C.crown_dead(layers, use_lp=True, lp_budget_s=lp_budget_s)
    np.savez(art / "private" / "reference_crown.npz", **N.prune(layers, dead))
    return {"method": "crown+lp" if info["lp"] else "crown", "n_certified": [len(d) for d in dead],
            "certified": dead, "lp_budget_s": lp_budget_s, **info}


def evaluate(art: Path, delta_path: Path) -> dict:
    """Pruned dead units, pruned live units, and whether the outputs are unchanged on every input."""
    sp = _spec(art)
    base = N.from_dict(_load(art / "public" / "mlp.npz"))
    delta = _load(delta_path)
    edited = N.from_dict({k: v + delta.get(k, 0.0) for k, v in N.to_dict(base).items()})
    truth = json.loads((art / "private" / "truth.json").read_text())["dead"]
    pruned = N.pruned_units(base, edited)
    x = N.all_inputs(sp["n_inputs"])
    diff = np.abs(N.forward(edited, x) - N.forward(base, x))
    changed = int((diff.max(1) > TOLERANCE).sum())
    hits = sum(len(set(p) & set(t)) for p, t in zip(pruned, truth))
    # Per-unit deadness: a pruned unit is live if its pre-activation is > 0 on some input (exact enumeration).
    # Output-neutral prunes of live units are reported here and rejected by the caller (tr05_dead.evaluate).
    live_units = [sorted(set(p) - set(t)) for p, t in zip(pruned, truth)]
    return {"zeroing_only": N.zeroing_only(base, edited), "n_dead": sum(map(len, truth)), "dead_pruned": hits,
            "live_pruned": sum(map(len, live_units)), "live_units": live_units,
            "inputs_changed": changed, "max_abs_change": float(diff.max())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "evaluate", "reference_crown"])
    ap.add_argument("--art", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lp-budget", type=float, default=LP_BUDGET_S,
                    help="reference_crown: stop issuing LPs after this many seconds (pre-registered default)")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.step == "build":
        out = build(Path(a.art))
    elif a.step == "evaluate":
        out = evaluate(Path(a.art), Path(a.weights))
    else:
        out = reference_crown(Path(a.art), a.lp_budget)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
