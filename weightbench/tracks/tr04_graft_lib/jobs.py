"""Heavy steps for TR04-GRAFT-001. Run through ``weightbench.heavy.run_heavy``.

    python -m weightbench.tracks.tr04_graft_lib.jobs build    --art ART --out ART/private/build.json
    python -m weightbench.tracks.tr04_graft_lib.jobs evaluate --art ART --weights DELTA.npz --out OUT.json
    python -m weightbench.tracks.tr04_graft_lib.jobs check    --art ART --weights DELTA.npz --payload P.json --out OUT.json

``build`` trains the host (task A only) and the donor (seeded lookup table), measures base
accuracies, and computes the reference graft (server-side read-out fitting, selected on the
x3-even half of the inputs).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from . import graft as G
from . import model as M


def _spec(art: Path) -> dict:
    return json.loads((art / "private" / "spec.json").read_text())


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def apply(host: Mapping[str, np.ndarray], delta: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: np.asarray(v, np.float32).copy() for k, v in host.items()}
    for k, v in delta.items():
        out[k] = out[k] + np.asarray(v, np.float32)
    return out


def _reference_candidates(host: dict, donor: dict) -> dict[str, dict[str, np.ndarray]]:
    """(a) graft on the host as is; (b) also pull embed[QB] toward embed[QA] so the host gathers
    x1, x2 on QB queries as it does on QA, keeping 20% of the difference as the gate signal."""
    fit_idx = M.all_sequences(M.QA)[:, 2] % 2 == 0
    seqs = np.concatenate([M.all_sequences(M.QA)[fit_idx], M.all_sequences(M.QB)[fit_idx]])
    candidates = {}
    for name, alpha in (("graft", None), ("graft_qb_pull_0.2", 0.2)):
        embed_delta = {}
        if alpha is not None:
            d = np.zeros(M.HOST_SHAPES["embed"], np.float32)
            d[M.QB] = (alpha - 1.0) * (host["embed"][M.QB] - host["embed"][M.QA])
            embed_delta = {"embed": d}
        edited = apply(host, embed_delta)
        _, resid = M.host_forward(edited, seqs, return_resid=True)
        readout = G.fit_readout(resid, seqs)
        spread = G.logit_spread(edited, seqs[seqs[:, -1] == M.QB])
        candidates[name] = {**embed_delta, **G.graft_delta(edited, donor, readout, spread)}
    return candidates


def build(art: Path) -> dict:
    sp = _spec(art)
    seeds = sp["seeds"]
    level = int(sp.get("level", 0))
    distract = M.distractor(level)          # None at L0; the competing QB circuit the host also learns at L>0
    raw = np.random.default_rng(seeds["table"]).integers(0, M.N_DIGITS, (M.N_DIGITS, M.N_DIGITS))
    table = raw                             # the target is the same 10x10 table at every level
    host = M.train_host(seeds["host"], distract=distract)
    donor = M.train_donor(table, seeds["donor"])
    np.savez(art / "public" / "host.npz", **host)
    np.savez(art / "public" / "donor.npz", **donor)
    np.save(art / "private" / "table.npy", table)
    x1, x2 = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    donor_acc = float((M.donor_forward(donor, x1.ravel(), x2.ravel()).argmax(1) == table.ravel()).mean())
    base = M.accuracies(host, table)
    if distract is not None:                # how well the host serves its competing QB circuit (for the record)
        name, weight = distract
        qb = M.all_sequences(M.QB)
        other = M.distractor_target(name, qb[:, :4])
        pred = np.concatenate([M.host_forward(host, qb[i:i + 2000]).argmax(1) for i in range(0, len(qb), 2000)])
        base[f"distract_{name}"] = float((pred == other).mean())        # accuracy on the competing function
        base["qb_task_a"] = float((pred == M.task_a_target(qb)).mean())  # the part of QB the graft can still use

    candidates = _reference_candidates(host, donor)
    scored = {name: M.accuracies(apply(host, d), table) for name, d in candidates.items()}
    best = max(scored, key=lambda n: scored[n]["graft"] - 2.0 * max(0.0, base["task_a"] - scored[n]["task_a"]))
    np.savez(art / "private" / "reference.npz", **candidates[best])
    baseline = G.graft_delta(host, donor, G.naive_readout(host), G.logit_spread(host, M.all_sequences(M.QB)[:2000]))
    np.savez(art / "private" / "baseline.npz", **baseline)
    return {"level": level, "distractor": distract, "donor_acc": donor_acc, "base": base,
            "reference_candidates": scored, "reference": best,
            "baseline": M.accuracies(apply(host, baseline), table)}


def evaluate(art: Path, delta_path: Path) -> dict:
    host = _load(art / "public" / "host.npz")
    return M.accuracies(apply(host, _load(delta_path)), np.load(art / "private" / "table.npy"))


def check(art: Path, delta_path: Path, payload: dict) -> dict:
    """Metered probe: accuracies on a seeded sample of 500 QA and 500 QB sequences."""
    host = apply(_load(art / "public" / "host.npz"), _load(delta_path))
    table = np.load(art / "private" / "table.npy")
    rng = np.random.default_rng(int(payload["seed"]))
    out = {}
    for name, query in (("task_a", M.QA), ("graft", M.QB)):
        seqs = M.all_sequences(query)[rng.choice(10 ** 4, 500, replace=False)]
        target = M.task_a_target(seqs) if query == M.QA else table[seqs[:, 0], seqs[:, 1]]
        out[name] = float((M.host_forward(host, seqs).argmax(1) == target).mean())
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "evaluate", "check"])
    ap.add_argument("--art", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--payload")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    art = Path(a.art)
    t0 = time.time()
    if a.step == "build":
        out = build(art)
    elif a.step == "evaluate":
        out = evaluate(art, Path(a.weights))
    else:
        out = check(art, Path(a.weights), json.loads(Path(a.payload).read_text()))
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
