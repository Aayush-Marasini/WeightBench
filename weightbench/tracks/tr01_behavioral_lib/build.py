"""Heavy build steps for TR01-BEH-001..003 (training, rollouts, sampling, reference).

    python -m weightbench.tracks.tr01_behavioral_lib.build --task beh001 --art ART --seed 0 --out ART/build_out.json [--toy]

Reads/writes only under ART (public/ and private/). Device: cuda > cpu. Called through
``weightbench.heavy.run_heavy`` so it can run locally or on the cluster.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

from . import common as C


def _sub(seed: int, *k: int) -> int:
    return int(np.random.default_rng([seed, *k]).integers(2 ** 31))


# --------------------------------------------------------------------------- BEH-001 (games)
def _collect_job(a):
    from . import games as G

    game, n, s, rounds = a
    X, Y = G.collect(game, n, s, rounds=rounds)
    return game, X, Y


def _returns_job(a):
    from . import games as G

    kind, game, layers, s, episodes, max_steps = a
    if kind == "policy":
        return G.policy_returns(game, layers, s, episodes, max_steps)
    return {"expert": G.expert_returns(game, s, episodes, max_steps)}


def _pmap(fn, jobs):
    if C.N_PROCS <= 1:
        return [fn(j) for j in jobs]
    with mp.get_context("spawn").Pool(min(C.N_PROCS, len(jobs))) as pool:
        return pool.map(fn, jobs)


def _train_job(a):
    """One hardened policy: own regime -> BC -> dead-unit padding -> logit shift -> hidden
    permutation + positive rescaling (log-std 1.0). All nuisances are function-preserving on
    binary inputs except the regime itself."""
    from . import games as G

    X, Y, seed, n_scrambled = a
    rng = np.random.default_rng(seed)
    regime = G.sample_regime(rng)
    L = G.train_policy(X, Y, seed, regime=regime)
    L = G.logit_shift(G.pad_dead(L, rng), rng)
    L = C.permute_hidden(L, rng, positive_scale=1.0, count=n_scrambled)
    return L, regime


def build_beh001(art: Path, seed: int, toy: bool, n_scrambled: int | None = None) -> dict:
    from . import games as G

    n_data, rounds, n_models, n_shadow, episodes, max_steps = \
        (60, 2, 4, 1, 1, 20) if toy else (20000, 4, 8, 4, 10, 1000)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    jobs = [(g, n_data, _sub(seed, 1, i), rounds) for i, g in enumerate(G.GAMES)]
    res = _pmap(_collect_job, jobs)
    data = {g: (X, Y) for g, X, Y in res}
    t_collect = time.time() - t0
    n_tr = int(0.85 * n_data)  # last 15% (policy-driven DAgger states) held out for diagnostics
    train = {g: (X[:n_tr], Y[:n_tr]) for g, (X, Y) in data.items()}
    held = {g: (X[n_tr:], Y[n_tr:]) for g, (X, Y) in data.items()}

    labels = C.draw_labels(rng, G.GAMES, n_models, min_distinct=min(4, n_models))
    names = [f"policy_{i}" for i in range(n_models)]
    shadow_lab = [(gi, g, k) for gi, g in enumerate(G.GAMES) for k in range(n_shadow)]
    t1 = time.time()
    tjobs = [(*train[g], _sub(seed, 2, i), n_scrambled) for i, g in enumerate(labels)]
    tjobs += [(*train[g], _sub(seed, 4, gi, k), n_scrambled) for gi, g, k in shadow_lab]
    trained = _pmap(_train_job, tjobs)
    t_train = time.time() - t1
    (art / "public" / "models").mkdir(parents=True, exist_ok=True)
    pub, regimes = {}, {}
    for n, (L, reg) in zip(names, trained[:n_models]):
        C.save_mlp(art / "public" / "models" / f"{n}.npz", L)
        pub[n], regimes[n] = L, reg
    (art / "private" / "shadow").mkdir(parents=True, exist_ok=True)
    shadow = {}
    for (gi, g, k), (L, reg) in zip(shadow_lab, trained[n_models:]):
        f = f"shadow_{gi}_{k}.npz"
        C.save_mlp(art / "private" / "shadow" / f, L)
        shadow[f], regimes[f] = g, reg
    C.write_json(art / "private" / "shadow.json", shadow)
    C.write_json(art / "private" / "regimes.json", regimes)
    C.write_json(art / "private" / "gt.json", dict(zip(names, labels)))

    # reference (weight-only): match each policy's per-cell input-weight mass against the cells each
    # game can ever turn on (occupancy from server-side states; an agent derives it from the sources)
    occ = {g: G.occupancy(train[g][0]) for g in G.GAMES}
    np.savez(art / "private" / "occupancy.npz", **occ)
    occ_s = {n: G.occupancy_scores(pub[n], occ) for n in names}
    ref = {n: max(occ_s[n], key=occ_s[n].get) for n in names}
    # behavioural diagnostics (server-side forward passes on held-out states)
    ig = {n: {g: G.expert_info_gain(pub[n], *held[g]) for g in G.GAMES} for n in names}
    fit = {n: {g: G.expert_fit(pub[n], *held[g], train[g][1].mean(0)) for g in G.GAMES} for n in names}
    agree = {n: dict(zip(("agreement", "chance"), G.decisive_agreement(pub[n], *held[g])))
             for n, g in zip(names, labels)}
    C.write_json(art / "private" / "reference.json",
                 {"labels": ref, "method": "occupancy_auc", "occupancy_auc": occ_s,
                  "behavioural_info_gain": ig, "behavioural_fit": fit, "own_game_agreement": agree})

    # diagnostic: a stronger generic floor (canonicalised spectra) against the same shadow bank
    sx = np.stack([G.canonical_features(C.load_mlp(art / "private" / "shadow" / f)) for f in shadow])
    qx = np.stack([G.canonical_features(pub[n]) for n in names])
    canon_pred = dict(zip(names, C.nearest_centroid(sx, list(shadow.values()), qx)))

    # returns (trained >> random check)
    t2 = time.time()
    rjobs = [("policy", g, pub[n], _sub(seed, 6, i), episodes, max_steps) for i, (n, g) in enumerate(zip(names, labels))]
    rjobs += [("expert", g, None, _sub(seed, 7, i), max(1, episodes // 3), max_steps) for i, g in enumerate(G.GAMES)]
    rr = _pmap(_returns_job, rjobs)
    returns = {n: {"game": g, **r} for n, g, r in zip(names, labels, rr[:n_models])}
    expert = {g: r["expert"] for g, r in zip(G.GAMES, rr[n_models:])}
    C.write_json(art / "private" / "returns.json", {"policies": returns, "expert": expert,
                                                    "episodes": episodes, "max_steps": max_steps})

    gdir = art / "public" / "games"
    gdir.mkdir(exist_ok=True)
    for g in G.GAMES:
        shutil.copy(G.source_dir() / f"{g}.py", gdir / f"{g}.py")
    C.write_json(art / "public" / "encoding.json", G.describe())
    acc = lambda pred: float(np.mean([pred[n] == g for n, g in zip(names, labels)]))
    return {"task": "beh001", "t_collect_s": round(t_collect, 1), "t_train_s": round(t_train, 1),
            "t_returns_s": round(time.time() - t2, 1), "labels": dict(zip(names, labels)),
            "reference_acc": acc(ref), "spectral_baseline_acc": acc(C.spectral_baseline(art, names)),
            "canonical_baseline_acc": acc(canon_pred),
            "behavioural_ig_acc": acc({n: max(v, key=v.get) for n, v in ig.items()}),
            "behavioural_fit_acc": acc({n: max(v, key=v.get) for n, v in fit.items()}),
            "returns": returns, "expert": expert, "own_game_agreement": agree, "occupancy_auc": occ_s,
            "regimes": {n: regimes[n] for n in names}}


# --------------------------------------------------------------------------- BEH-002 (DDPM)
def build_beh002(art: Path, seed: int, toy: bool, n_scrambled: int | None = None) -> dict:
    from . import ddpm as D

    n_models, n_shadow, n_data, steps, n_samp = (3, 1, 200, 20, 8) if toy else (10, 2, 5000, 6000, 64)
    rng = np.random.default_rng(seed)
    labels = C.draw_labels(rng, D.SHAPES, n_models, min_distinct=min(5, n_models))
    names = [f"denoiser_{i}" for i in range(n_models)]
    shadow_lab = [(s, k) for s in D.SHAPES for k in range(n_shadow)]
    all_lab = labels + [s for s, _ in shadow_lab]
    seeds = [_sub(seed, 10, i) for i in range(len(all_lab))]
    datas = [D.dataset(s, n_data, _sub(seed, 11, i)) for i, s in enumerate(all_lab)]
    t0 = time.time()
    models = D.train_batched(datas, seeds, steps=steps)
    t_train = time.time() - t0
    models = [C.permute_hidden(L, np.random.default_rng(_sub(seed, 12, i)), count=n_scrambled)
              for i, L in enumerate(models)]

    (art / "public" / "models").mkdir(parents=True, exist_ok=True)
    for n, L in zip(names, models[:n_models]):
        C.save_mlp(art / "public" / "models" / f"{n}.npz", L)
    (art / "private" / "shadow").mkdir(parents=True, exist_ok=True)
    shadow = {}
    for (s, k), L in zip(shadow_lab, models[n_models:]):
        f = f"shadow_{D.SHAPES.index(s)}_{k}.npz"
        C.save_mlp(art / "private" / "shadow" / f, L)
        shadow[f] = s
    C.write_json(art / "private" / "shadow.json", shadow)
    C.write_json(art / "private" / "gt.json", dict(zip(names, labels)))

    b = D.betas()
    np.savez(art / "public" / "schedule.npz", betas=b.astype(np.float32),
             alphas_cumprod=np.cumprod(1 - b).astype(np.float32))
    ex_rng = np.random.default_rng(_sub(seed, 13))
    np.savez(art / "public" / "shape_examples.npz",
             **{s: np.stack([D.render(s, ex_rng) for _ in range(16)]) for s in D.SHAPES})

    # reference: sample from each denoiser (server-side) and classify samples with a shape classifier
    clf = D.train_classifier(_sub(seed, 14), n_per=100 if toy else 1500)
    cx = np.concatenate([D.dataset(s, 50, _sub(seed, 16, i)) for i, s in enumerate(D.SHAPES)])
    clf_acc = float((D.classify(clf, cx) == np.repeat(np.arange(len(D.SHAPES)), 50)).mean())
    ref, quality = {}, {}
    for i, (n, L) in enumerate(zip(names, models[:n_models])):
        xs = D.sample(L, n_samp, _sub(seed, 15, i))
        votes = np.bincount(D.classify(clf, xs), minlength=len(D.SHAPES))
        ref[n] = D.SHAPES[int(votes.argmax())]
        quality[n] = {"truth": labels[i], "votes": dict(zip(D.SHAPES, votes.tolist())),
                      "frac_true_class": float(votes[D.SHAPES.index(labels[i])] / n_samp)}
        if i < 3:
            np.save(art / "private" / f"samples_{n}.npy", xs[:16].reshape(-1, 12, 12).astype(np.float32))
    C.write_json(art / "private" / "reference.json", {"labels": ref, "quality": quality})
    ref_acc = float(np.mean([ref[n] == g for n, g in zip(names, labels)]))
    return {"task": "beh002", "t_train_s": round(t_train, 1), "device": C.device(), "classifier_acc": clf_acc,
            "labels": dict(zip(names, labels)), "reference_acc": ref_acc, "quality": quality}


# --------------------------------------------------------------------------- BEH-003 (Boolean fns)
def build_beh003(art: Path, seed: int, toy: bool, n_scrambled: int | None = None) -> dict:
    from . import boolfn as B

    n_models, n_shadow, steps, bs = (3, 1, 30, 1024) if toy else (10, 2, 6000, 8192)
    rng = np.random.default_rng(seed)
    params = B.make_params(rng)
    labels = C.draw_labels(rng, B.FUNCS, n_models, min_distinct=min(6, n_models))
    names = [f"net_{i}" for i in range(n_models)]
    shadow_lab = [(f, k) for f in B.FUNCS for k in range(n_shadow)]
    all_lab = labels + [f for f, _ in shadow_lab]
    tables = {f: B.truth_table(f, params) for f in B.FUNCS}
    t0 = time.time()
    models = B.train_batched([tables[f] for f in all_lab], [_sub(seed, 20, i) for i in range(len(all_lab))],
                             steps=steps, bs=bs)
    t_train = time.time() - t0
    models = [C.permute_hidden(L, np.random.default_rng(_sub(seed, 21, i)), positive_scale=0.5, count=n_scrambled)
              for i, L in enumerate(models)]
    (art / "public" / "models").mkdir(parents=True, exist_ok=True)
    for n, L in zip(names, models[:n_models]):
        C.save_mlp(art / "public" / "models" / f"{n}.npz", L)
    (art / "private" / "shadow").mkdir(parents=True, exist_ok=True)
    shadow = {}
    for (f, k), L in zip(shadow_lab, models[n_models:]):
        fn = f"shadow_{B.FUNCS.index(f)}_{k}.npz"
        C.save_mlp(art / "private" / "shadow" / fn, L)
        shadow[fn] = f
    C.write_json(art / "private" / "shadow.json", shadow)
    C.write_json(art / "private" / "gt.json", dict(zip(names, labels)))
    C.write_json(art / "public" / "candidates.json", {"params": params, "definitions": B.definitions(params)})
    (art / "public" / "candidates.py").write_text(B.candidates_source(params))

    # reference: truth-table matching (server-side evaluation of all 2^16 inputs)
    ref, acc = {}, {}
    for n, f, L in zip(names, labels, models[:n_models]):
        pred = B.predict(L)
        agree = {g: float((pred == tables[g]).mean()) for g in B.FUNCS}
        ref[n] = max(agree, key=agree.get)
        acc[n] = {"truth": f, "train_acc": agree[f], "best_other": max(v for g, v in agree.items() if g != f)}
    shadow_acc = {fn: float((B.predict(L) == tables[f]).mean()) for (fn, f), L in zip(shadow.items(), models[n_models:])}
    C.write_json(art / "private" / "reference.json", {"labels": ref, "accuracy": acc, "shadow_acc": shadow_acc})
    ref_acc = float(np.mean([ref[n] == g for n, g in zip(names, labels)]))
    return {"task": "beh003", "t_train_s": round(t_train, 1), "device": C.device(), "params": params,
            "labels": dict(zip(names, labels)), "reference_acc": ref_acc, "accuracy": acc,
            "min_shadow_acc": min(shadow_acc.values())}


BUILDERS = {"beh001": build_beh001, "beh002": build_beh002, "beh003": build_beh003}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(BUILDERS), required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--level", type=int, default=None, choices=(0, 1))
    ap.add_argument("--procs", type=int, default=None, help="worker processes (default WB_PROCS or 3)")
    ap.add_argument("--threads", type=int, default=None, help="torch threads (default WB_THREADS or 3)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    art = Path(a.art)
    if a.procs:
        C.N_PROCS = a.procs
    if a.threads:
        C.N_THREADS = a.threads
    (art / "public").mkdir(parents=True, exist_ok=True)
    (art / "private").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from weightbench import knobs
    n_scrambled = None if a.level is None else int(knobs.value(knobs.get("TR01"), a.level))
    out = BUILDERS[a.task](art, a.seed, a.toy, n_scrambled)
    if a.level is not None:
        knob = {**knobs.describe("TR01", a.level), "available_hidden_layers": 2}
        C.write_json(art / 'public/knob.json', knob)
        C.write_json(art / 'private/knob.json', knob)
    out["build_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
