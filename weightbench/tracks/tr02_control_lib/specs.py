"""Build / evaluate / probe logic for TR02-CTRL-001..004 (runs inside ``job.py``, i.e. via run_heavy).

Artifacts written by ``Spec.build``::

    public/policy.npz              base policy weights (fc0..fc2, PyTorch convention)
    public/failure_trajectory.npz  one failing episode of the base policy
    public/ENV.md, public/ARCH.md  environment dynamics / architecture + observation spec
    public/OBS_LAYOUT.md           (Kinetix only) meaning of all 679 observation entries
    public/rollout.py              helper: ΔW .npz -> metered `rollout` probe payload
    private/eval_sets.npz          held-out failure / normal start states (+ train splits)
    private/reference_delta.npz    reference override circuit
    private/baseline_delta.npz     random sparse search floor (50 rollouts)
    private/stats.json             base / baseline / reference numbers

Knob levels (``build(..., level=L)``, ``knob.py``): L0 is the default build plus ``knob.json``; L >= 1 trains
from an expert with K = 2^L flaws and writes one failure trajectory and one private eval pool per flaw
(``failure_trajectory.npz`` = flaw 1, ``failure_trajectory_<k>.npz``; ``eval_sets.npz`` keys ``*_<k>``).
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from weightbench.tracks.tr02_control_lib import knob as KN
from weightbench.tracks.tr02_control_lib import mlp
from weightbench.tracks.tr02_control_lib.classic import CartPole, ClassicEnv, MountainCar, Pendulum

MAX_L0 = 0.04
N_EVAL = 100          # held-out episodes per set (failure / normal)
N_BASELINE = 50       # baseline random-search rollouts (= agent rollout budget)
HERE = Path(__file__).resolve().parent


def _edit_cost(delta, base) -> dict:
    from weightbench import scoring

    c = scoring.TrackConstants.for_track("TR02")
    return scoring.edit_cost({k: v for k, v in delta.items() if np.any(v)}, base, c.lam)


def proxy_score(s_fail: float, s_fail_base: float, s_norm: float, s_norm_base: float, cost: float) -> float:
    from weightbench import scoring

    sp = scoring.s_plus_write(s_fail, s_fail_base, 1.0)
    d = scoring.collateral(s_norm_base, s_norm)
    return sp * float(np.exp(-2.0 * d)) * float(np.clip(1.0 - cost, 0.0, 1.0))


def _pick_gadget(sf: np.ndarray, floor: np.ndarray, sn: np.ndarray, cost: np.ndarray, tol: float,
                 first: np.ndarray | None = None) -> int:
    """Which gadget of the grid to keep: the best success on the flaw it repairs (within ``tol``, one episode of
    the train split), then the highest ``floor`` (the minimum over the flaw pools it was not asked to repair),
    then the highest normal-pool success, and only then the cheapest edit.

    Every candidate that reaches the same success on the repaired flaw is equally right on the train split, so an
    argmax over a proxy that includes the edit cost would simply take the cheapest, which can be a hard gate that
    generalises poorly. Ties are decided by what a candidate does to the rest of the task instead.

    ``first`` is an optional extra tie-break applied right after the ``tol`` filter, for the joint circuit search:
    when no candidate repairs every flaw pool (the worst pool ties at 0) the tie-break still has to keep the theta
    that repairs the task's own defect, not the cheapest one.
    """
    best = float(np.max(sf))
    cand = [i for i in range(len(sf)) if sf[i] >= best - tol]

    def key(i: int):
        return ((0.0 if first is None else -float(first[i])),
                -float(floor[i]), -float(sn[i]), float(cost[i]))

    return min(cand, key=key)


def _zeros_like(p):
    return {k: np.zeros_like(v) for k, v in p.items()}


class Spec:
    key: str
    task_id: str
    sizes: list[int]
    n_pruned: tuple[int, int]
    probe_doc: str = ""
    # difficulty knob (knob.py): supported explicit levels, pruned units of the flaw-1 circuit, shared gadget units
    knob_levels: tuple[int, ...] | None = None
    circuit_units: tuple[int, int] = (0, 0)
    shared_h0: int = 0
    n_flaw_pool: int = 1500

    # --- hooks ------------------------------------------------------------------------------
    def train_base(self, rng, seed, pruned, log) -> mlp.Params: ...
    def sample_pool(self, rng) -> tuple[np.ndarray, np.ndarray]: ...       # starts, is_failure_region
    def run(self, params, starts, record=False) -> dict: ...
    def run_many(self, deltas: list, base, starts) -> np.ndarray: ...    # [n_cand, n_starts] success
    def circuit(self, base, pruned, th: dict) -> mlp.Params: ...
    def theta_grid(self) -> list[dict]: ...
    def parse_start(self, art: Path, start) -> np.ndarray: ...
    def compact_traj(self, info: dict, payload: dict) -> dict: ...
    def write_docs(self, art: Path, stats: dict) -> None: ...

    # --- knob hooks (knob.py) -------------------------------------------------------------------
    def flaw_family(self, rng) -> list[KN.Flaw]:
        return []

    def norm_affine(self) -> tuple[np.ndarray, np.ndarray]: ...          # raw obs = x * scale + shift
    def run_expert(self, flaws, active, starts) -> tuple[np.ndarray, np.ndarray]: ...  # success [B], visits [B, R]
    def sample_pool_level(self, rng, flaws) -> tuple[np.ndarray, np.ndarray]: ...
    def gadget_grid(self, flaw: KN.Flaw) -> list[dict]: ...
    def wire_gadget(self, d, plan, w, flaw, th) -> None: ...

    def shared_delta(self, base, plan) -> mlp.Params:
        return _zeros_like(base)

    # --- generic build --------------------------------------------------------------------------
    def build(self, art: Path, seed: int, log: Callable[[str], None] = print, level: int | None = None) -> dict:
        if level is None:
            return self._build_default(art, seed, log)
        return self._build_level(art, seed, log, level)

    def _build_default(self, art: Path, seed: int, log: Callable[[str], None] = print) -> dict:
        t0 = time.time()
        pub, priv = art / "public", art / "private"
        pub.mkdir(parents=True, exist_ok=True)
        priv.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed * 1000 + 17)
        pruned = {0: sorted(rng.choice(self.sizes[1], self.n_pruned[0], replace=False).tolist()),
                  1: sorted(rng.choice(self.sizes[2], self.n_pruned[1], replace=False).tolist())}
        log(f"[{self.task_id}] training base policy {self.sizes} pruned={pruned}")
        base = self.train_base(rng, seed, pruned, log)
        log(f"  base params: {mlp.n_params(base)} ({time.time() - t0:.0f}s)")

        starts, region = self.sample_pool(rng)
        info = self.run(base, starts)
        succ = info["success"].astype(bool)
        fail_idx = np.flatnonzero(region & ~succ)
        norm_idx = np.flatnonzero(~region & succ) if (~region).any() else np.flatnonzero(succ)
        log(f"  pool: {len(starts)} starts, base success {succ.mean():.3f}; failure-region base success "
            f"{succ[region].mean() if region.any() else float('nan'):.3f}; fail pool {len(fail_idx)}, normal pool {len(norm_idx)}")
        n_tr = self.n_train
        if len(fail_idx) < N_EVAL + n_tr + 1 or len(norm_idx) < N_EVAL + 16:
            raise RuntimeError(f"not enough failing ({len(fail_idx)}) / succeeding ({len(norm_idx)}) starts")
        fail_idx, norm_idx = rng.permutation(fail_idx), rng.permutation(norm_idx)
        n_tr_norm = min(n_tr, len(norm_idx) - N_EVAL)
        sets = {"fail_eval": starts[fail_idx[:N_EVAL]], "fail_train": starts[fail_idx[N_EVAL:N_EVAL + n_tr]],
                "norm_eval": starts[norm_idx[:N_EVAL]], "norm_train": starts[norm_idx[N_EVAL:N_EVAL + n_tr_norm]],
                "traj_start": starts[fail_idx[N_EVAL + n_tr]][None]}
        np.savez(priv / "eval_sets.npz", **sets)
        mlp.save_params(pub / "policy.npz", base)

        # failure trajectory (public)
        tr = self.run(base, sets["traj_start"], record=True)
        T = int(tr["t_end"][0])
        np.savez(pub / "failure_trajectory.npz", start=sets["traj_start"][0],
                 obs=tr["obs"][0, :T].astype(np.float32), action=tr["act"][0, :T].astype(np.float32),
                 reward=tr["rew"][0, :T].astype(np.float32), success=bool(tr["success"][0]))

        base_eval = self._eval_sets(base, sets)
        stats: dict[str, Any] = {"task": self.task_id, "seed": seed, "n_params": mlp.n_params(base),
                                 "pruned_units": pruned, "pool_size": int(len(starts)),
                                 "pool_base_success": float(succ.mean()),
                                 "failure_region_base_success": float(succ[region].mean()) if region.any() else None,
                                 "base": base_eval}
        log(f"  base on eval sets: {base_eval}")

        # reference: analytic override circuit template, a few scalars tuned on the TRAIN splits
        ref, ref_th, ref_train = self._search_reference(base, pruned, sets, log)
        mlp.save_params(priv / "reference_delta.npz", ref)
        stats["reference"] = {"theta": ref_th, "train": ref_train, **self._score(base, ref, sets, base_eval)}
        log(f"  reference: {stats['reference']}")

        # baseline: random sparse search, N_BASELINE single-episode rollouts on the failure start
        bl, bl_info = self._baseline(base, sets, rng)
        mlp.save_params(priv / "baseline_delta.npz", bl)
        stats["baseline"] = {**bl_info, **self._score(base, bl, sets, base_eval)}
        log(f"  baseline: {stats['baseline']}")

        stats["build_seconds"] = round(time.time() - t0, 1)
        mlp.write_json(priv / "stats.json", stats)
        self.write_docs(art, stats)
        shutil.copy(HERE / "rollout_helper.py.txt", pub / "rollout.py")
        return stats

    n_train = 100

    def _eval_sets(self, params, sets) -> dict:
        f = self.run(params, sets["fail_eval"])["success"]
        n = self.run(params, sets["norm_eval"])["success"]
        return {"s_fail": float(np.mean(f)), "s_norm": float(np.mean(n))}

    def _score(self, base, delta, sets, base_eval) -> dict:
        from weightbench import scoring

        ev = self._eval_sets(mlp.apply_delta(base, delta), sets)
        cost = _edit_cost(delta, base)
        sp = scoring.s_plus_write(ev["s_fail"], base_eval["s_fail"], 1.0)
        d = scoring.collateral(base_eval["s_norm"], ev["s_norm"])
        return {**ev, "s_plus": sp, "d_col": d, "edit_cost": cost,
                "s_wb_no_compute": proxy_score(ev["s_fail"], base_eval["s_fail"], ev["s_norm"], base_eval["s_norm"],
                                               cost["cost"])}

    def _search_reference(self, base, pruned, sets, log):
        grid = self.theta_grid()
        deltas = [self.circuit(base, pruned, th) for th in grid]
        sf = self.run_many(deltas, base, sets["fail_train"]).mean(1)
        sn = self.run_many(deltas, base, sets["norm_train"]).mean(1)
        bf = float(self.run(base, sets["fail_train"])["success"].mean())
        bn = float(self.run(base, sets["norm_train"])["success"].mean())
        scores = [proxy_score(sf[i], bf, sn[i], bn, _edit_cost(deltas[i], base)["cost"]) for i in range(len(grid))]
        order = np.argsort(scores)[::-1]
        for i in order[:5]:
            log(f"    theta {grid[i]}: train fail {sf[i]:.2f} norm {sn[i]:.2f} proxy {scores[i]:.3f}")
        b = int(order[0])
        return deltas[b], grid[b], {"s_fail": float(sf[b]), "s_norm": float(sn[b]), "proxy": float(scores[b])}

    def _baseline(self, base, sets, rng):
        cands = mlp.random_sparse_candidates(base, N_BASELINE, l0=0.02, sigma=2.0, rng=rng)
        info = self.run_many_info(cands, base, sets["traj_start"][0])
        key = info["success"].astype(float) * 1e6 + info["ret"]
        b = int(np.argmax(key))
        return cands[b], {"n_rollouts": N_BASELINE, "best_candidate": b,
                          "best_on_failure_start": {"success": bool(info["success"][b]), "return": float(info["ret"][b])},
                          "n_candidates_solving_failure_start": int(info["success"].sum())}

    def run_many_info(self, deltas, base, start) -> dict:
        """Each candidate on the same single start (one rollout each)."""
        outs = [self.run(mlp.apply_delta(base, d), start[None]) for d in deltas]
        return {"success": np.array([o["success"][0] for o in outs]), "ret": np.array([o["ret"][0] for o in outs])}

    # --- knob levels (knob.py) ------------------------------------------------------------------
    @staticmethod
    def _knob_k(art: Path) -> int:
        p = art / "private" / "knob.json"
        return int(mlp.read_json(p)["K"]) if p.exists() else 1

    def _build_level(self, art: Path, seed: int, log, level: int) -> dict:
        from weightbench import knobs

        flaws = KN.select_flaws(self, seed, level)          # rejects unsupported levels before any training
        if flaws:   # skip a drawn flaw no start can be attributed to (identical to the draw when it is reachable)
            flaws = KN.select_reachable_flaws(self, seed, level, N_EVAL + self.n_train + 1, log)
        K = len(flaws) + 1
        (art / "private").mkdir(parents=True, exist_ok=True)
        (art / "public").mkdir(parents=True, exist_ok=True)
        if K == 1:
            stats = self._build_default(art, seed, log)
        else:
            stats = self._build_multi(art, seed, log, flaws)
        titles = [self.flaw1_title] + [f.title for f in flaws]
        pub = {**knobs.describe("TR02", level), "K": K, "stuck_situations": titles}
        mlp.write_json(art / "public" / "knob.json", pub)
        mlp.write_json(art / "private" / "knob.json",
                       {**pub, "seed": seed, "flaws": [f.private() for f in flaws],
                        "validation": stats.get("knob", {}).get("validation")})
        return stats

    flaw1_title: str = "the task's own stuck situation"

    def _build_multi(self, art: Path, seed: int, log, flaws: list[KN.Flaw]) -> dict:
        t0 = time.time()
        pub, priv = art / "public", art / "private"
        K = len(flaws) + 1
        names = [f"flaw{c}" for c in range(1, K + 1)]
        rng = np.random.default_rng(seed * 1000 + 17)
        pruned = {0: sorted(rng.choice(self.sizes[1], self.n_pruned[0], replace=False).tolist()),
                  1: sorted(rng.choice(self.sizes[2], self.n_pruned[1], replace=False).tolist())}
        plan = KN.unit_plan(self, pruned, flaws)
        log(f"[{self.task_id}] K = {K} flaws {[f.name for f in flaws]}; training base policy pruned={pruned}")
        base = self.train_base(rng, seed, pruned, log, flaws=flaws)

        starts, region = self.sample_pool_level(rng, flaws)
        labels, att = KN.attribute(self, flaws, starts, region)
        succ = np.asarray(self.run(base, starts)["success"], bool)
        n_tr = self.n_train
        try:
            rep = KN.validate(labels, succ, names, N_EVAL + n_tr + 1, N_EVAL + 16)
        except KN.FlawValidationError as e:
            mlp.write_json(priv / "knob_validation.json", {**e.report, **att, "flaws": e.report["flaws"],
                                                            "added": [f.private() for f in flaws]})
            log(f"  VALIDATION FAILED: {e}")
            raise
        rep.update(att, policy_episodes=int(len(starts)))
        log(f"  validation: {rep}")
        sets: dict[str, np.ndarray] = {}
        for c in range(1, K + 1):
            idx = rng.permutation(np.flatnonzero((labels == c) & ~succ))
            sets[f"fail_eval_{c}"] = starts[idx[:N_EVAL]]
            sets[f"fail_train_{c}"] = starts[idx[N_EVAL:N_EVAL + n_tr]]
            sets[f"traj_start_{c}"] = starts[idx[N_EVAL + n_tr]][None]
        nidx = rng.permutation(np.flatnonzero((labels == KN.NORMAL) & succ))
        n_tr_norm = min(n_tr, len(nidx) - N_EVAL)
        sets["norm_eval"], sets["norm_train"] = starts[nidx[:N_EVAL]], starts[nidx[N_EVAL:N_EVAL + n_tr_norm]]
        for k in ("fail_eval", "fail_train", "traj_start"):
            sets[k] = sets[f"{k}_1"]
        np.savez(priv / "eval_sets.npz", **sets)
        mlp.save_params(pub / "policy.npz", base)
        for c in range(1, K + 1):
            tr = self.run(base, sets[f"traj_start_{c}"], record=True)
            T = int(tr["t_end"][0])
            np.savez(pub / ("failure_trajectory.npz" if c == 1 else f"failure_trajectory_{c}.npz"),
                     start=sets[f"traj_start_{c}"][0], obs=tr["obs"][0, :T].astype(np.float32),
                     action=tr["act"][0, :T].astype(np.float32), reward=tr["rew"][0, :T].astype(np.float32),
                     success=bool(tr["success"][0]))

        base_eval = self._eval_multi(base, sets, K)
        stats: dict[str, Any] = {"task": self.task_id, "seed": seed, "n_params": mlp.n_params(base),
                                 "pruned_units": pruned, "pool_size": int(len(starts)),
                                 "pool_base_success": float(succ.mean()), "failure_region_base_success": None,
                                 "base": base_eval, "knob": {"K": K, "validation": rep, "unit_plan": plan}}
        log(f"  base on eval sets: {base_eval}")
        ref, ths = self._search_reference_multi(base, pruned, plan, flaws, sets, log)
        mlp.save_params(priv / "reference_delta.npz", ref)
        stats["reference"] = {"theta": ths, **self._score_multi(base, ref, sets, base_eval, K)}
        log(f"  reference: {stats['reference']}")
        bl, bl_info = self._baseline_multi(base, sets, rng, K)
        mlp.save_params(priv / "baseline_delta.npz", bl)
        stats["baseline"] = {**bl_info, **self._score_multi(base, bl, sets, base_eval, K)}
        log(f"  baseline: {stats['baseline']}")
        stats["build_seconds"] = round(time.time() - t0, 1)
        mlp.write_json(priv / "stats.json", stats)
        self.write_docs(art, stats)
        shutil.copy(HERE / "rollout_helper.py.txt", pub / "rollout.py")
        return stats

    def _eval_multi(self, params, sets, K: int) -> dict:
        by = {f"flaw{c}": float(np.mean(self.run(params, sets[f"fail_eval_{c}"])["success"])) for c in range(1, K + 1)}
        s_norm = float(np.mean(self.run(params, sets["norm_eval"])["success"]))
        return {"s_fail": by["flaw1"], "s_norm": s_norm, "s_fail_by_flaw": by}

    def _score_multi(self, base, delta, sets, base_eval, K: int) -> dict:
        from weightbench import scoring

        ev = self._eval_multi(mlp.apply_delta(base, delta), sets, K)
        per = KN.per_flaw_s_plus(ev["s_fail_by_flaw"], base_eval["s_fail_by_flaw"])
        d = scoring.collateral(base_eval["s_norm"], ev["s_norm"])
        cost = _edit_cost(delta, base)
        swb = per["s_plus"] * float(np.exp(-2.0 * d)) * float(np.clip(1.0 - cost["cost"], 0.0, 1.0))
        return {**ev, **per, "d_col": d, "edit_cost": cost, "s_wb_no_compute": swb}

    def _search_reference_multi(self, base, pruned, plan, flaws, sets, log):
        K = len(flaws) + 1                      # the expert carries its own defect plus one flaw per added knob
        # One circuit has to repair the task's own defect AND leave every added flaw repairable. A theta chosen on
        # flaw-1 success alone can steer the gadget into a region where a small correction cannot finish the job.
        # So select the circuit on its worst flaw pool, before any gadget is added.
        grid = self.theta_grid()
        deltas = [KN.add(self.circuit(base, pruned, th), self.shared_delta(base, plan)) for th in grid]
        n1 = max(1, len(sets["fail_train_1"]))
        sf1 = self.run_many(deltas, base, sets["fail_train_1"]).mean(1)
        sf = sf1.copy()
        for o in range(2, K + 1):
            sf = np.minimum(sf, self.run_many(deltas, base, sets[f"fail_train_{o}"]).mean(1))
        sn = self.run_many(deltas, base, sets["norm_train"]).mean(1)
        # The worst-pool score is the selection criterion when some theta repairs every pool; when none does (a grid
        # whose only knob is the added-flaw repair, or a K too large for one circuit), every candidate ties at 0 and
        # the tie-break falls to `sf1`: keep the theta that repairs the task's own defect, not the cheapest one.
        b = _pick_gadget(sf, np.zeros(len(grid)), sn,
                         np.array([_edit_cost(d, base)["cost"] for d in deltas]), 0.5 / n1, first=sf1)
        log(f"    joint circuit theta {grid[b]}: min train fail {sf[b]:.2f} norm {sn[b]:.2f}")
        ref, th1 = deltas[b], grid[b]
        ths = [th1]
        for j, f in enumerate(flaws):
            c = j + 2
            grid = self.gadget_grid(f)
            deltas = [KN.add(ref, KN.gadget(self, base, plan, j, f, th)) for th in grid]
            fm = sets[f"fail_train_{c}"]
            sf = self.run_many(deltas, base, fm).mean(1)
            sn = self.run_many(deltas, base, sets["norm_train"]).mean(1)
            # Tie-break on what the candidate does to the pools it was not asked to repair. Every candidate that
            # reaches the same success on flaw c is equally "right" on the train split, and an argmax over a
            # proxy that includes the edit cost would pick the cheapest, which can be a hard gate that generalises
            # poorly (see _pick_gadget). So: keep the candidates within one episode of the best
            # flaw-c success, then take the one that leaves the other flaw pools and the normal pool highest, and
            # only then the cheapest.
            floor = np.ones(len(grid))
            for o in range(1, K + 1):
                if o != c:
                    floor = np.minimum(floor, self.run_many(deltas, base, sets[f"fail_train_{o}"]).mean(1))
            best = float(sf.max())
            tol = 0.5 / max(1, len(fm))
            # A gadget is not free: it spends a h0/h1 unit and edit budget, and on a seed where the circuit already
            # repairs this flaw it can only cost. Keep the circuit alone unless the best gadget is at least as good.
            alone = float(self.run(mlp.apply_delta(base, ref), fm)["success"].mean())
            if best < alone - tol:
                log(f"    gadget {c} ({f.name}): none beats the circuit alone "
                    f"(best {best:.2f} vs {alone:.2f}); keeping the circuit")
                ths.append("identity")
                continue
            b = _pick_gadget(sf, floor, sn, np.array([_edit_cost(d, base)["cost"] for d in deltas]), tol)
            tied = int(np.sum(sf >= best - tol))
            log(f"    gadget {c} ({f.name}) theta {grid[b]}: train fail {sf[b]:.2f} norm {sn[b]:.2f} "
                f"other {floor[b]:.2f} (best flaw-{c} success {best:.2f}, {tied} candidate(s) tied, "
                f"circuit alone {alone:.2f})")
            ref = deltas[b]
            ths.append(grid[b])
        return ref, ths

    def _baseline_multi(self, base, sets, rng, K: int):
        """As at K = 1 (50 random sparse candidates, one rollout each), candidate i tried on flaw (i mod K)'s
        failure start; best by (success, return)."""
        cands = mlp.random_sparse_candidates(base, N_BASELINE, l0=0.02, sigma=2.0, rng=rng)
        succ, ret = np.zeros(N_BASELINE, bool), np.zeros(N_BASELINE)
        which = np.arange(N_BASELINE) % K
        for c in range(K):
            ii = np.flatnonzero(which == c)
            info = self.run_many_info([cands[i] for i in ii], base, sets[f"traj_start_{c + 1}"][0])
            succ[ii], ret[ii] = np.asarray(info["success"], bool), info["ret"]
        b = int(np.argmax(succ.astype(float) * 1e6 + ret))
        return cands[b], {"n_rollouts": N_BASELINE, "best_candidate": b, "best_tried_on_flaw": int(which[b]) + 1,
                          "n_candidates_solving_their_start": int(succ.sum())}

    def _evaluate_multi(self, art: Path, delta: mlp.Params) -> dict:
        from weightbench import scoring

        K = self._knob_k(art)
        base = mlp.load_params(art / "public" / "policy.npz")
        b = mlp.read_json(art / "private" / "stats.json")["base"]
        with np.load(art / "private" / "eval_sets.npz") as z:
            sets = {k: z[k] for k in z.files}
        ev = self._eval_multi(mlp.apply_delta(base, delta), sets, K)
        per = KN.per_flaw_s_plus(ev["s_fail_by_flaw"], b["s_fail_by_flaw"])
        return {"s_plus": per["s_plus"], "s_plus_mean": per["s_plus_mean"], "s_plus_by_flaw": per["s_plus_by_flaw"],
                "d_col": scoring.collateral(b["s_norm"], ev["s_norm"]),
                "s_fail_edit": ev["s_fail"], "s_fail_base": b["s_fail"],
                "s_fail_edit_by_flaw": ev["s_fail_by_flaw"], "s_fail_base_by_flaw": b["s_fail_by_flaw"],
                "s_norm_edit": ev["s_norm"], "s_norm_base": b["s_norm"], "K": K,
                "l0": mlp.l0_fraction(base, delta), "n_eval_episodes": (K + 1) * N_EVAL}

    def _failure_start(self, art: Path, start):
        """Start of a public failure trajectory for ``"failure"`` / ``"failure<k>"``, else None."""
        if start in (None, "failure"):
            name = "failure_trajectory.npz"
        elif isinstance(start, str) and re.fullmatch(r"failure([1-9]\d?)", start):
            k = int(start[7:])
            name = "failure_trajectory.npz" if k == 1 else f"failure_trajectory_{k}.npz"
            if not (art / "public" / name).exists():
                raise ValueError(f"no stuck situation {k} in this task")
        else:
            return None
        with np.load(art / "public" / name) as z:
            return z["start"]

    # --- evaluation / probe ----------------------------------------------------------------------
    def evaluate(self, art: Path, delta: mlp.Params) -> dict:
        from weightbench import scoring

        if self._knob_k(art) > 1:
            return self._evaluate_multi(art, delta)
        base = mlp.load_params(art / "public" / "policy.npz")
        stats = mlp.read_json(art / "private" / "stats.json")
        with np.load(art / "private" / "eval_sets.npz") as z:
            sets = {k: z[k] for k in z.files}
        edited = mlp.apply_delta(base, delta)
        ev = self._eval_sets(edited, sets)
        b = stats["base"]
        return {"s_plus": scoring.s_plus_write(ev["s_fail"], b["s_fail"], 1.0),
                "d_col": scoring.collateral(b["s_norm"], ev["s_norm"]),
                "s_fail_edit": ev["s_fail"], "s_fail_base": b["s_fail"],
                "s_norm_edit": ev["s_norm"], "s_norm_base": b["s_norm"],
                "l0": mlp.l0_fraction(base, delta), "n_eval_episodes": 2 * N_EVAL}

    def probe(self, art: Path, payload: dict) -> dict:
        base = mlp.load_params(art / "public" / "policy.npz")
        delta = mlp.decode_delta(payload.get("delta"), base, MAX_L0)
        start = self.parse_start(art, payload.get("start", "failure"))
        info = self.run(mlp.apply_delta(base, delta), start[None], record=True)
        return self.compact_traj(info, payload) | {"delta_nonzeros": int(sum(np.count_nonzero(d) for d in delta.values()))}


# =============================================================================================
# Classic control (numpy)
# =============================================================================================
class ClassicSpec(Spec):
    env: ClassicEnv
    bc_n: int = 60000
    bc_epochs: int = 40
    pool_n: int = 6000

    def bc_states(self, rng) -> np.ndarray: ...

    def train_base(self, rng, seed, pruned, log, flaws=()):
        S = self.bc_states(rng)
        X = self.env.norm(self.env.obs(S))
        Y = self.env.expert_target(self.env.obs(S))
        if flaws:   # knob level: the K-flaw expert relabels the same states (collection is unchanged)
            Y = KN.apply_overrides(Y, self.env.obs(S), list(flaws))
        return mlp.train_bc(X, Y, self.sizes, "ce" if self.env.discrete else "mse", seed, pruned,
                            epochs=self.bc_epochs, log=log)

    def sample_pool(self, rng):
        starts = self.env.sample_starts(rng, self.pool_n)
        return starts, np.ones(len(starts), bool)

    def run(self, params, starts, record=False):
        return self.env.rollout(lambda x: mlp.forward(params, x), starts, record=record)

    def run_many(self, deltas, base, starts):
        return np.stack([self.run(mlp.apply_delta(base, d), starts)["success"] for d in deltas]).astype(float)

    def parse_start(self, art, start):
        s = self._failure_start(art, start)
        if s is not None:
            return s.astype(np.float64)
        return self.env.validate_start(start)

    # --- knob hooks ---------------------------------------------------------------------------
    pool_scale = 3

    def expert_out(self, o, flaws, active):
        y = KN.apply_overrides(self.env.expert_target(o), o, list(flaws), active)
        if self.env.discrete:
            return np.eye(self.env.n_out)[y.astype(int)]
        return np.asarray(y, np.float64).reshape(len(o), -1)

    def run_expert(self, flaws, active, starts):
        scale, shift = self.norm_affine()
        pol = lambda x: self.expert_out(x * scale + shift, flaws, active)  # noqa: E731
        succ, vis = [], []
        for i in range(0, len(starts), 2000):
            info = self.env.rollout(pol, starts[i:i + 2000], record=True)
            obs, al = info["obs"], info["alive"].astype(bool)
            succ.append(np.asarray(info["success"], bool))
            vis.append(np.stack([(f.mask(obs) & al).any(1) for f in flaws], 1) if flaws
                       else np.zeros((len(obs), 0), bool))
        return np.concatenate(succ), np.concatenate(vis)

    def sample_pool_level(self, rng, flaws):
        parts = [self.env.sample_starts(rng, self.pool_n * self.pool_scale)]
        for f in flaws:
            parts.append(rng.uniform(f.start_low, f.start_high, size=(self.n_flaw_pool, len(f.start_low))))
        starts = np.concatenate(parts)
        return starts, np.ones(len(starts), bool)

    def compact_traj(self, info, payload):
        T = int(info["t_end"][0])
        r5 = lambda a: np.round(np.asarray(a, np.float64), 5).tolist()  # noqa: E731
        return {"success": bool(info["success"][0]), "return": float(info["ret"][0]), "t_end": T,
                "obs_names": self.env.obs_names, "obs": r5(info["obs"][0, :T]),
                "action": r5(info["act"][0, :T]), "reward": r5(info["rew"][0, :T])}

    def arch_md(self, stats) -> str:
        D, H0, H1, A = self.sizes
        p = stats["pruned_units"]
        return f"""# Policy architecture

    x   = normalise(obs)                  (fixed, see below; not a parameter)
    h0  = relu(fc0.weight @ x  + fc0.bias)    fc0.weight [{H0}, {D}]   fc0.bias [{H0}]
    h1  = relu(fc1.weight @ h0 + fc1.bias)    fc1.weight [{H1}, {H0}]  fc1.bias [{H1}]
    out = fc2.weight @ h1 + fc2.bias          fc2.weight [{A}, {H1}]   fc2.bias [{A}]
{self.action_rule}

`policy.npz` holds the six float32 arrays above ({stats['n_params']} parameters). The checkpoint was
produced with structured pruning: some hidden units have all-zero incoming weights, bias and
outgoing weights (find them yourself; there are {len(p[0])} in h0 and {len(p[1])} in h1).

Observation normalisation (applied before fc0):
{self.norm_doc}
"""


class MountainCarSpec(ClassicSpec):
    key, task_id = "mountaincar", "TR02-CTRL-002"
    env = MountainCar()
    sizes = [2, 96, 96, 3]
    n_pruned = (6, 4)
    pool_n = 24000
    action_rule = "    action = argmax(out)   (0 = push left, 1 = no push, 2 = push right)"
    norm_doc = "    x = [(position + 0.3) / 0.9, velocity / 0.07]"
    # Knob L0..L1: flaw 1 = no climb-out (the car rocks in the valley). Added family = "brake" regions the car
    # crosses while moving right, where the expert plays action 0 instead of 2 and gives the run-up away. The gate
    # is *any* rightward motion (velocity > ~0): a speed gate at v ~ 0.05 lets the brake switch itself off as soon as
    # it has slowed the car below the gate, the car re-accelerates and still reaches the goal, and no start can be
    # attributed to the flaw. Braking in the valley band is the fatal one -- the valley is the deepest well, so a
    # car that leaves the band at ~zero speed can never climb out within the horizon. Gadget: one gate per
    # half-space and one h1 unit that cancels the brake and restores push right inside the region. K = 4 would need
    # 3 gadgets x 2-3 gates = 6-9 h0 units, over the 5 spare after flaw 1, so only L1 is supported.
    knob_levels = (0, 1)
    circuit_units = (1, 1)
    flaw1_title = "the car rocks back and forth near the bottom of the valley and never reaches the goal"

    def norm_affine(self):
        return np.array([0.9, 0.07]), np.array([-0.3, 0.0])

    def flaw_family(self, rng):
        xb = KN.jit(rng, -0.46, -0.32)
        vb = KN.jit(rng, 0.0, 0.01)
        xc = KN.jit(rng, 0.0, 0.08)
        vc = KN.jit(rng, 0.0, 0.01)
        t = ("On starts where the car is on the {} moving right, the policy pushes LEFT (brakes) instead of right, "
             "throws away the run-up it just built and cannot climb out.")
        return [
            # The region is the band x_c < x < x_c + 0.14 on the last climb to the goal, crossed while moving right.
            # A brake on the far left hill would not be a flaw: there gravity pulls right harder than the left push,
            # so action 0 never slows the car. Half-spaces are ``c . obs + b > 0`` like every other family member.
            KN.Flaw("final_climb_brake", t.format("final climb to the goal"),
                    ((((0, 1.0),), -xc), (((0, -1.0),), xc + 0.14), (((1, 1.0),), -vc)),
                    action=0, fix=1.0, start_low=(xc, 0.045), start_high=(xc + 0.14, 0.068),
                    params={"x_c": xc, "v_c": vc}),
            # The same brake at the valley floor, lower down the climb. Sign the band's half-spaces like
            # every other family member: x > x_b is ((0, +1), -x_b); writing it as x < x_b makes the region empty,
            # no rollout can enter it, and the level builds with zero attributable starts for this flaw.
            KN.Flaw("valley_brake", t.format("valley floor"),
                    ((((0, 1.0),), -xb), (((0, -1.0),), xb + 0.14), (((1, 1.0),), -vb)),
                    action=0, fix=1.0, start_low=(xb, 0.045), start_high=(xb + 0.14, 0.068),
                    params={"x_b": xb, "v_b": vb}),
        ]

    def gadget_grid(self, flaw):
        return [dict(kg=kg, G=4.0, A=A) for kg in (8.0, 32.0) for A in (4.0, 8.0, 16.0, 32.0)]

    def wire_gadget(self, d, plan, w, flaw, th):
        # inside the region: restore the expert's push-right action (2) and cancel the trained-in push left (0)
        d["fc2.weight"][2, w] = th["A"] * flaw.fix
        d["fc2.weight"][0, w] = -th["A"] * flaw.fix

    def bc_states(self, rng):
        return rng.uniform([-1.2, -0.07], [0.6, 0.07], size=(self.bc_n, 2))

    def circuit(self, base, pruned, th):
        d = _zeros_like(base)
        u, w = pruned[0][0], pruned[1][0]
        d["fc0.weight"][u, 1] = -th["c"]
        d["fc0.bias"][u] = -th["c"] * th["t"]
        d["fc1.weight"][w, u] = th.get("g", 1.0)
        d["fc2.weight"][0, w] = th["A"]
        d["fc2.weight"][2, w] = -th["A"]
        return d

    def theta_grid(self):
        return [dict(c=c, g=g, A=A, t=t) for c in (4.0, 8.0, 16.0) for g in (1.0, 4.0) for A in (4.0, 8.0, 16.0)
                for t in (0.0, -0.02, -0.05)]

    def write_docs(self, art, stats):
        e = self.env
        (art / "public" / "ARCH.md").write_text(self.arch_md(stats))
        (art / "public" / "ENV.md").write_text(f"""# Environment: Gymnasium MountainCar-v0 (exact dynamics)

State = observation = `[position, velocity]`. Actions: 0 = push left, 1 = no push, 2 = push right.
Each step (force = {e.force}, gravity = {e.gravity}):

    velocity += (action - 1) * {e.force} - {e.gravity} * cos(3 * position)
    velocity  = clip(velocity, -{e.max_speed}, {e.max_speed})
    position += velocity
    position  = clip(position, {e.min_pos}, {e.max_pos});  if position == {e.min_pos} and velocity < 0: velocity = 0
    terminated (success) when position >= {e.goal_pos} and velocity >= 0

The track height is sin(3 * position); the goal flag is on the right hill at position 0.5.
Reward -1 per step, episode length {e.horizon} steps. **Success = reaching the goal within {e.horizon} steps.**

Start states for this task: position in [{e.start_low[0]}, {e.start_high[0]}], velocity in
[{e.start_low[1]}, {e.start_high[1]}] (uniform). The engine is too weak to climb the right hill
directly: from low energy the car must first swing back up the left hill to build momentum.
""")


class PendulumSpec(ClassicSpec):
    key, task_id = "pendulum", "TR02-CTRL-003"
    env = Pendulum()
    sizes = [3, 128, 128, 1]
    n_pruned = (8, 4)
    bc_n = 80000
    action_rule = "    torque = clip(out[0], -2, 2)"
    norm_doc = "    x = [cos(theta), sin(theta), theta_dot / 8]"
    # Knob L0..L1: flaw 1 = no energy pumping (stall); added family = "lets go of the swing while the pendulum is
    # rising towards upright" on either side (3 half-spaces each). K = 4 would need 3 gadgets x 3 gates = 9 h0 > 3
    # spare units. Overriding the torque while the pendulum falls *away* from upright is not a flaw: doing nothing
    # during the fall converts potential into kinetic energy, so such an expert succeeds more often than the base
    # one and no start can be attributed to the flaw.
    knob_levels = (0, 1)
    circuit_units = (5, 2)
    flaw1_title = "from low-energy starts it pushes towards the top, stalls part-way up and stays there"

    def norm_affine(self):
        return np.array([1.0, 1.0, 8.0]), np.zeros(3)

    def flaw_family(self, rng):
        a = [KN.jit(rng, 0.35, 0.5), KN.jit(rng, 0.35, 0.5)]
        t = ("If it starts in the upper half already swinging towards upright on the {} side, it stops driving the "
             "swing, the pendulum loses its momentum, falls back and then stalls near the bottom.")
        return [
            KN.Flaw("fall_ccw", t.format("theta > 0"),
                    ((((1, 1.0),), -float(np.sin(a[0]))), (((0, 1.0),), 0.0), (((2, -1.0),), 0.0)),
                    action=0.0, fix=-1.0, start_low=(a[0], -3.0), start_high=(np.pi / 2, 0.0),
                    params={"theta_a": a[0]}),
            KN.Flaw("fall_cw", t.format("theta < 0"),
                    ((((1, -1.0),), -float(np.sin(a[1]))), (((0, 1.0),), 0.0), (((2, 1.0),), 0.0)),
                    action=0.0, fix=1.0, start_low=(-np.pi / 2, 0.0), start_high=(-a[1], 3.0),
                    params={"theta_a": a[1]}),
        ]

    def gadget_grid(self, flaw):
        # A soft gate with a small correction only nudges the swing near the region; a hard gate (large kg, wide A)
        # flips the torque over a boundary 1/(kg*G) wide and can inject torque the base expert never applies. The
        # grid therefore includes soft candidates, so a fragile setting cannot win on edit cost alone.
        return [dict(kg=kg, G=G, A=A) for kg in (2.0, 4.0, 8.0, 16.0) for G in (2.0, 4.0, 8.0)
                for A in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)]

    def wire_gadget(self, d, plan, w, flaw, th):
        d["fc2.weight"][0, w] = th["A"] * flaw.fix            # torque back towards upright

    def bc_states(self, rng):
        return rng.uniform([-np.pi, -8.0], [np.pi, 8.0], size=(self.bc_n, 2))

    def circuit(self, base, pruned, th):
        d = _zeros_like(base)
        h, p0, n0, q1, q2 = pruned[0][:5]
        p, n = pruned[1][:2]
        W0, b0, W1, W2 = d["fc0.weight"], d["fc0.bias"], d["fc1.weight"], d["fc2.weight"]
        W0[h, 0], b0[h] = 1.0, -th["c0"]                 # h  = relu(cos - c0): upper part -> off
        W0[p0, 2], W0[n0, 2] = th["k"], -th["k"]         # p0/n0 = relu(+-k * thdot/8)
        W0[q1, 2], b0[q1] = 1.0, -th["vm"]               # q = relu(|thdot/8| - vm): fast -> off
        W0[q2, 2], b0[q2] = -1.0, -th["vm"]
        G = th["G"]
        for unit, src in ((p, p0), (n, n0)):
            W1[unit, src] = 1.0
            W1[unit, h] = W1[unit, q1] = W1[unit, q2] = -G
        W2[0, p], W2[0, n] = th["A"], -th["A"]           # pump: torque along theta_dot
        return d

    def theta_grid(self):
        # The grid includes vm = 0.4 and A = 8.0: the joint multi-flaw search needs them to find circuits that
        # repair both flaw pools fully.
        return [dict(c0=c0, vm=vm, k=k, G=G, A=A) for c0 in (-0.6, -0.3, 0.0, 0.3, 0.6)
                for vm in (0.4, 0.6, 0.8) for k in (1.0, 2.0) for G in (3.0, 5.0)
                for A in (1.0, 2.0, 4.0, 8.0)]

    def write_docs(self, art, stats):
        e = self.env
        (art / "public" / "ARCH.md").write_text(self.arch_md(stats))
        (art / "public" / "ENV.md").write_text(f"""# Environment: Gymnasium Pendulum-v1 (exact dynamics)

State `[theta, theta_dot]`, theta = 0 is UPRIGHT, theta = +-pi hanging down; counter-clockwise +.
Observation `[cos(theta), sin(theta), theta_dot]`. Action: torque u, clipped to [-{e.max_torque}, {e.max_torque}].
Each step (g = {e.g}, m = {e.m}, l = {e.l}, dt = {e.dt}):

    theta_dot' = clip(theta_dot + (3 g / (2 l) * sin(theta) + 3 / (m l^2) * u) * dt, -{e.max_speed}, {e.max_speed})
    theta'     = theta + theta_dot' * dt
    reward     = -(angle_normalize(theta)^2 + 0.1 theta_dot^2 + 0.001 u^2)

i.e. theta_ddot = 15 sin(theta) + 3 u. With |u| <= 2 the motor cannot hold the pendulum more than
asin(6/15) = 0.41 rad away from straight down/up: lifting it requires swinging (pumping energy).
Episode length {e.horizon} steps, no early termination.
**Success = |angle_normalize(theta)| < 0.3 rad for each of the last {e.hold_steps} steps of the episode.**

Start states for this task: theta in [-pi, pi], theta_dot in [{e.start_low[1]}, {e.start_high[1]}] (uniform).
""")


class CartPoleSpec(ClassicSpec):
    key, task_id = "cartpole", "TR02-CTRL-004"
    env = CartPole()
    sizes = [4, 256, 256, 2]
    n_pruned = (12, 6)
    bc_n = 100000
    bc_epochs = 30
    action_rule = "    action = argmax(out)   (0 = push cart left, 1 = push cart right)"
    norm_doc = "    x = [x / 2.4, x_dot / 2, theta / 0.21, theta_dot / 2]"
    # Knob L0..L2: flaw 1 = no position term (drift off the track); added family = four disjoint "fast cart"
    # regions (side of the track x direction of motion, 2 half-spaces each) where the expert keeps pushing in the
    # direction of motion, so the pole falls. Gadgets: 2 shared balance units + 2 gates + 1 h1 unit each.
    knob_levels = (0, 1, 2)
    circuit_units = (2, 2)
    shared_h0 = 2
    flaw1_title = "from some start states the cart runs off the end of the track before the episode ends"

    def norm_affine(self):
        return np.array([2.4, 2.0, 0.21, 2.0]), np.zeros(4)

    def flaw_family(self, rng):
        v = [KN.jit(rng, 0.6, 0.8) for _ in range(4)]
        c = [KN.jit(rng, 0.2, 0.4) for _ in range(4)]
        t = ("From some start states where the cart is on the {} part of the track and already moving fast to the "
             "{}, the policy keeps pushing that way and the pole falls over.")
        th = (-0.05, 0.05)
        return [
            KN.Flaw("fast_right_left_half", t.format("left", "right"), ((((0, -1.0),), -c[0]), (((1, 1.0),), -v[0])),
                    1, 1.0, (-1.8, v[0], th[0], th[0]), (-c[0], 1.2, th[1], th[1]), {"x_max": -c[0], "v_min": v[0]}),
            KN.Flaw("fast_left_right_half", t.format("right", "left"), ((((0, 1.0),), -c[1]), (((1, -1.0),), -v[1])),
                    0, -1.0, (c[1], -1.2, th[0], th[0]), (1.8, -v[1], th[1], th[1]), {"x_min": c[1], "v_max": -v[1]}),
            KN.Flaw("fast_right_right_half", t.format("right", "right"), ((((0, 1.0),), -c[2]), (((1, 1.0),), -v[2])),
                    1, 1.0, (c[2], v[2], th[0], th[0]), (min(c[2] + 0.8, 1.8), 1.2, th[1], th[1]),
                    {"x_min": c[2], "v_min": v[2]}),
            KN.Flaw("fast_left_left_half", t.format("left", "left"), ((((0, -1.0),), -c[3]), (((1, -1.0),), -v[3])),
                    0, -1.0, (max(-c[3] - 0.8, -1.8), -1.2, th[0], th[0]), (-c[3], -v[3], th[1], th[1]),
                    {"x_max": -c[3], "v_max": -v[3]}),
        ]

    def shared_delta(self, base, plan):
        d = _zeros_like(base)
        e = self.env
        bal = np.array([0.0, 2.0 * e.KV, 0.21 * e.KT, 2.0 * e.KW])     # expert balance term in normalised inputs
        p, n = plan["shared"]
        d["fc0.weight"][p], d["fc0.weight"][n] = bal, -bal
        return d

    def gadget_grid(self, flaw):
        return [dict(kg=kg, G=4.0, A=A, lam=lam) for kg in (8.0, 32.0) for A in (2.0, 8.0, 32.0) for lam in (8.0, 32.0)]

    def wire_gadget(self, d, plan, w, flaw, th):
        # in the region: h1 = relu(1 - lam * s * bal); logit diff += -s * A * h1, i.e. cancel the trained-in push
        # towards s and restore the sign of the expert balance term (no effect where bal already agrees with s)
        p, n = plan["shared"]
        s = flaw.fix
        d["fc1.weight"][w, p], d["fc1.weight"][w, n] = -th["lam"] * s, th["lam"] * s
        d["fc2.weight"][1, w], d["fc2.weight"][0, w] = -s * th["A"] / 2, s * th["A"] / 2

    def bc_states(self, rng):
        e = self.env
        n_roll = self.bc_n // 2
        # visited states of a noisy expert (epsilon-random actions) + a uniform box
        starts = rng.uniform([-2.0, -1.5, -0.1, -0.3], [2.0, 1.5, 0.1, 0.3], size=(200, 4))

        def noisy(o):
            a = e.expert_target(o)
            flip = rng.random(len(a)) < 0.15
            return np.where(flip, 1 - a, a)

        S = []
        s = starts.copy()
        for t in range(400):
            S.append(s.copy())
            s, _, d = e.step(s, noisy(s))
            if d.any():
                s[d] = rng.uniform([-2.0, -1.5, -0.1, -0.3], [2.0, 1.5, 0.1, 0.3], size=(int(d.sum()), 4))
        S = np.concatenate(S)[:n_roll]
        U = rng.uniform([-2.4, -2.5, -0.21, -2.5], [2.4, 2.5, 0.21, 2.5], size=(self.bc_n - len(S), 4))
        return np.concatenate([S, U])

    def circuit(self, base, pruned, th):
        d = _zeros_like(base)
        r, l = pruned[0][:2]
        wr, wl = pruned[1][:2]
        W0, b0, W1, W2 = d["fc0.weight"], d["fc0.bias"], d["fc1.weight"], d["fc2.weight"]
        k, kv, xc = th["k"], th["kv"], th["xc"]
        W0[r, 0], W0[r, 1], b0[r] = k, kv, -k * xc       # r = relu(k (x/2.4 - xc) + kv x_dot/2)
        W0[l, 0], W0[l, 1], b0[l] = -k, -kv, -k * xc     # mirror for the left edge
        W1[wr, r], W1[wl, l] = 1.0, 1.0
        A = th["A"] / 2
        W2[1, wr], W2[0, wr] = A, -A                     # near right edge: bias towards pushing RIGHT
        W2[0, wl], W2[1, wl] = A, -A                     # (balance controller then drives the cart back)
        return d

    def theta_grid(self):
        return [dict(xc=xc, k=k, kv=kv, A=A) for xc in (0.2, 0.4, 0.6) for k in (2.0, 6.0) for kv in (0.0, 0.5, 1.5)
                for A in (0.5, 1.0, 2.0, 4.0, 8.0)]

    def write_docs(self, art, stats):
        e = self.env
        (art / "public" / "ARCH.md").write_text(self.arch_md(stats))
        (art / "public" / "ENV.md").write_text(f"""# Environment: Gymnasium CartPole-v1 (exact dynamics, euler integrator)

State = observation = `[x, x_dot, theta, theta_dot]` (theta = 0 upright, theta > 0 = pole leaning
towards +x). Actions: 0 = force -{e.force_mag} N, 1 = force +{e.force_mag} N. Each step (tau = {e.tau}):

    total_mass = {e.masscart} + {e.masspole};  pml = {e.masspole} * {e.length}   (length = half pole length)
    temp       = (force + pml * theta_dot^2 * sin(theta)) / total_mass
    theta_acc  = (9.8 * sin(theta) - cos(theta) * temp) / ({e.length} * (4/3 - {e.masspole} * cos(theta)^2 / total_mass))
    x_acc      = temp - pml * theta_acc * cos(theta) / total_mass
    x += tau * x_dot;  x_dot += tau * x_acc;  theta += tau * theta_dot;  theta_dot += tau * theta_acc

Terminated when |x| > {e.x_threshold} (cart leaves the track) or |theta| > 12 degrees (0.2094 rad).
Reward +1 per step. **Success = surviving all {e.horizon} steps.**

Start states for this task: uniform in x [{e.start_low[0]}, {e.start_high[0]}], x_dot [{e.start_low[1]}, {e.start_high[1]}],
theta [{e.start_low[2]}, {e.start_high[2]}], theta_dot [{e.start_low[3]}, {e.start_high[3]}].
Note the non-minimum-phase nature of the system: to accelerate the cart towards -x without dropping
the pole, the pole must first lean towards -x, which requires a short push towards +x.
""")


# =============================================================================================
# Kinetix car-ramp (jax)
# =============================================================================================
class KinetixSpec(Spec):
    key, task_id = "kinetix", "TR02-CTRL-001"
    sizes = [679, 230, 64, 6]
    n_pruned = (8, 4)
    n_train = 64

    def _K(self):
        from weightbench.tracks.tr02_control_lib import kinetix_car as K
        return K

    @staticmethod
    def _jp(params):
        import jax.numpy as jnp
        return {k: jnp.asarray(v, jnp.float32) for k, v in params.items()}

    def train_base(self, rng, seed, pruned, log, flaws=()):
        import jax.numpy as jnp

        K = self._K()
        X, Y = [], []
        for rnd in range(2):
            V = np.concatenate([K.sample_variants(rng, K.KIND_RAMP, 120), K.sample_variants(rng, K.KIND_RIGHT, 60),
                                K.sample_variants(rng, K.KIND_FLAT, 60)])
            B = len(V)
            # smooth random throttle perturbations; 1/3 of episodes noise-free
            noise = np.zeros((B, K.N_STEPS))
            lvl = rng.choice([0.0, 0.4, 0.8], size=B)
            for i in range(B):
                z = np.convolve(rng.normal(size=K.N_STEPS + 15), np.ones(16) / 4.0, mode="valid")[: K.N_STEPS]
                noise[i] = lvl[i] * z
            out = K.rollout(K.noisy_expert, {"noise": jnp.asarray(noise, jnp.float32)}, V, record=True)
            al = out["alive"]
            o = out["obs"][al]
            X.append(o)
            log(f"  kinetix data round {rnd}: {len(o)} states, expert success {out['success'].mean():.2f}")
        X = np.concatenate(X).astype(np.float32)
        tgt = np.zeros((len(X), K.ACT_DIM), np.float32)
        tgt[:, 0] = 0.5 * np.tanh(4.0 * (X[:, K.CX] - X[:, K.GX]))
        if flaws:   # knob level: the K-flaw expert relabels the same states (collection is unchanged)
            tgt = KN.apply_overrides(tgt, X, list(flaws))
        return mlp.train_bc(X, tgt, self.sizes, "mse", seed, pruned, epochs=12, lr=5e-4, batch=256, log=log)

    def sample_pool(self, rng):
        K = self._K()
        V = np.concatenate([K.sample_variants(rng, K.KIND_RAMP, 400), K.sample_variants(rng, K.KIND_RIGHT, 160),
                            K.sample_variants(rng, K.KIND_FLAT, 160)])
        return V, V[:, 0] == K.KIND_RAMP

    def run(self, params, starts, record=False):
        K = self._K()
        return K.rollout(K.mlp_shared, self._jp(params), np.asarray(starts, np.float32), record=record)

    def run_many(self, deltas, base, starts):
        return np.stack([self.run(mlp.apply_delta(base, d), starts)["success"] for d in deltas]).astype(float)

    def run_many_info(self, deltas, base, start):
        K = self._K()
        P = mlp.stack_params([mlp.apply_delta(base, d) for d in deltas])
        V = np.repeat(np.asarray(start, np.float32)[None], len(deltas), 0)
        out = K.rollout(K.mlp_batched, self._jp(P), V)
        return {"success": out["success"], "ret": out["ret"]}

    def circuit(self, base, pruned, th):
        K = self._K()
        d = _zeros_like(base)
        up, un, ug, ur, ul = pruned[0][:5]
        wr, wf = pruned[1][:2]
        W0, b0, W1, W2 = d["fc0.weight"], d["fc0.bias"], d["fc1.weight"], d["fc2.weight"]
        c, k, xb = th["c"], th["k"], th["xb"]
        # s = vx + k (xb - x): s > 0 -> still building the run-up (reverse), s < 0 -> charge left
        W0[up, K.CVX], W0[up, K.CX], b0[up] = c, -c * k, c * k * xb
        W0[un, K.CVX], W0[un, K.CX], b0[un] = -c, c * k, -c * k * xb
        # gates: goal not on the left (ug), no ramp (ur), car already past the lava (ul)
        W0[ug, K.GX], W0[ug, K.CX], b0[ug] = 1.0, -1.0, 0.5
        W0[ur, K.RAMP_FRIC], b0[ur] = -1.0, 1.0
        W0[ul, K.LAVA_X], W0[ul, K.CX], b0[ul] = 1.0, -1.0, th["m"]
        G = th["G"]
        for unit, src in ((wr, up), (wf, un)):
            W1[unit, src] = 1.0
            W1[unit, ug] = W1[unit, ur] = W1[unit, ul] = -G
        W2[0, wr], W2[0, wf] = -th["A"], th["A"]
        return d

    def theta_grid(self):
        return [dict(c=c, k=k, xb=xb, A=A, G=4.0, m=0.3) for xb in (3.8, 4.2, 4.6) for k in (1.0, 2.5)
                for c in (1.5, 3.0) for A in (1.5, 3.0)]

    def parse_start(self, art, start):
        K = self._K()
        s = self._failure_start(art, start)
        if s is not None:
            return s.astype(np.float32)
        return K.validate_variant(start)

    # --- knob hooks: L0..L1 ----------------------------------------------------------------------
    # flaw 1 = no run-up on the ramp; added family = "turns back shortly before the goal in the flat scene" for
    # near or far goal positions (3 half-spaces each). K = 4 would need 9 gate units > 3 spare h0 units.
    knob_levels = (0, 1)
    circuit_units = (5, 2)
    n_flaw_pool = 240
    flaw1_title = ("in the `ramp` scene it drives up the ramp, loses its speed and stays stuck there until the "
                   "episode ends")

    def norm_affine(self):
        return np.ones(self.sizes[0]), np.zeros(self.sizes[0])

    def flaw_family(self, rng):
        K = self._K()
        g = [KN.jit(rng, 0.6, 0.7), KN.jit(rng, 0.6, 0.7)]
        d = [KN.jit(rng, 1.05, 1.25), KN.jit(rng, 1.05, 1.25)]
        t = ("In the `flat` scene (no ramp, no lava), when the goal is placed {} the car, it drives towards the "
             "goal but turns back shortly before reaching it and rocks back and forth in front of it.")
        lo, hi = K.RANGES[K.KIND_FLAT]["car_dx"]
        flat = (((K.RAMP_FRIC, -1.0),), 0.5)
        near = (((K.CX, -1.0), (K.GX, 1.0)),)
        return [
            KN.Flaw("flat_goal_near", t.format("closer to"), (flat, (((K.GX, 1.0),), -g[0]), (near[0], d[0])),
                    -0.5, 1.0, (K.KIND_FLAT, lo, g[0], 0.0), (K.KIND_FLAT, hi, 1.0, 0.0), {"goal_x_min": g[0], "dist": d[0]}),
            KN.Flaw("flat_goal_far", t.format("further from"), (flat, (((K.GX, -1.0),), g[1]), (near[0], d[1])),
                    -0.5, 1.0, (K.KIND_FLAT, lo, 0.3, 0.0), (K.KIND_FLAT, hi, g[1], 0.0), {"goal_x_max": g[1], "dist": d[1]}),
        ]

    def gadget_grid(self, flaw):
        return [dict(kg=kg, G=4.0, A=A) for kg in (4.0, 16.0) for A in (0.5, 1.0, 2.0)]

    def wire_gadget(self, d, plan, w, flaw, th):
        d["fc2.weight"][0, w] = th["A"] * flaw.fix            # throttle back towards the goal

    def sample_pool_level(self, rng, flaws):
        K = self._K()
        parts = [K.sample_variants(rng, K.KIND_RAMP, 800), K.sample_variants(rng, K.KIND_RIGHT, 320),
                 K.sample_variants(rng, K.KIND_FLAT, 320)]
        for f in flaws:
            parts.append(rng.uniform(f.start_low, f.start_high, size=(self.n_flaw_pool, 4)).astype(np.float32))
        V = np.concatenate(parts).astype(np.float32)
        return V, V[:, 0] == K.KIND_RAMP

    def run_expert(self, flaws, active, starts):
        K = self._K()
        R = len(flaws)
        M = max([len(f.ineqs) for f in flaws] + [1])
        Wb = [f.dense(K.OBS_DIM, M) for f in flaws]
        p = {"W": np.stack([w for w, _ in Wb]) if R else np.zeros((0, M, K.OBS_DIM), np.float32),
             "b": np.stack([b for _, b in Wb]) if R else np.zeros((0, M), np.float32),
             "a": np.array([f.action for f in flaws], np.float32), "on": np.asarray(active, np.float32)}
        succ, vis = [], []
        for i in range(0, len(starts), 128):
            out = K.rollout(_kx_flawed_expert, self._jp(p), np.asarray(starts[i:i + 128], np.float32), record=True)
            obs, al = np.asarray(out["obs"]), np.asarray(out["alive"], bool)
            succ.append(np.asarray(out["success"], bool))
            vis.append(np.stack([(f.mask(obs) & al).any(1) for f in flaws], 1) if R else np.zeros((len(obs), 0), bool))
        return np.concatenate(succ), np.concatenate(vis)

    def compact_traj(self, info, payload):
        K = self._K()
        T = int(info["t_end"][0])
        idx = dict(K.DEFAULT_TRACE)
        extra = payload.get("obs_idx") or []
        if len(extra) > 32:
            raise ValueError("at most 32 extra obs_idx")
        for i in extra:
            i = int(i)
            if not 0 <= i < K.OBS_DIM:
                raise ValueError(f"obs_idx {i} out of range")
            idx[f"obs[{i}]"] = i
        r4 = lambda a: np.round(np.asarray(a, np.float64), 4).tolist()  # noqa: E731
        obs = info["obs"][0, :T]
        return {"success": bool(info["success"][0]), "fell_in_lava": bool(info["lava"][0]),
                "return": float(info["ret"][0]), "t_end": T,
                "trace": {k: r4(obs[:, i]) for k, i in idx.items()},
                "action0": r4(info["act"][0, :T, 0]), "reward": r4(info["rew"][0, :T])}

    def write_docs(self, art, stats):
        K = self._K()
        p = stats["pruned_units"]
        (art / "public" / "OBS_LAYOUT.md").write_text(K.obs_layout_markdown())
        (art / "public" / "ARCH.md").write_text(f"""# Policy architecture (Kinetix-style MLP actor, deterministic mean action)

    x   = obs                                      (679 floats, see OBS_LAYOUT.md; no normalisation)
    h0  = relu(fc0.weight @ x  + fc0.bias)    fc0.weight [230, 679]  fc0.bias [230]
    h1  = relu(fc1.weight @ h0 + fc1.bias)    fc1.weight [64, 230]   fc1.bias [64]
    out = fc2.weight @ h1 + fc2.bias          fc2.weight [6, 64]     fc2.bias [6]
    action = clip(out, -1, 1)                 (6 values; only action[0] matters in this scene)

`policy.npz` holds the six float32 arrays above ({stats['n_params']} parameters). The checkpoint was
produced with structured pruning: some hidden units have all-zero incoming weights, bias and
outgoing weights (find them yourself; there are {len(p[0])} in h0 and {len(p[1])} in h1).
""")
        R = K.RANGES
        (art / "public" / "ENV.md").write_text(f"""# Environment: Kinetix `l/h2_car_ramp` (kinetix-env 3.0.2, Jax2D physics)

{K.SCENE_DOC}
## Level variants used in this task

Every episode starts from the level above with small changes. A variant is
`[kind, car_dx, goal_x, ramp_dx]`: the whole car is shifted by `car_dx` (chassis starts at
x = 2.8 + car_dx), the goal box is placed at x = goal_x, ramp and lava are shifted by `ramp_dx`.

| kind | name | car_dx | goal_x | ramp_dx | what it is |
|---|---|---|---|---|---|
| 0 | ramp  | {R[0]['car_dx']} | {R[0]['goal_x']} | {R[0]['ramp_dx']} | goal on the left, behind the ramp + lava (the failing scene) |
| 1 | right | {R[1]['car_dx']} | {R[1]['goal_x']} | {R[1]['ramp_dx']} | ramp + lava on the left, goal on the RIGHT of the car |
| 2 | flat  | {R[2]['car_dx']} | {R[2]['goal_x']} | {R[2]['ramp_dx']} | goal on the left, ramp and lava REMOVED (their obs rows are 0) |

**Success = the episode ends with the +1 goal reward (green wheel touches the goal) within 256 steps.**

The physics is a 2-D rigid-body engine (impulse solver, friction 1.0 on all shapes, gravity -9.81).
Some measured facts about this car (action[0] held constant from rest at car_dx = 0):
- a[0] = +0.5 (half throttle left): the car climbs onto the ramp, loses speed and stalls on it
  (chassis x ~ 2.09, tilted ~ -0.38 rad) for the rest of the episode.
- a[0] = +1.0 from rest: not enough run-up; the car leaves the ramp too slowly and lands in the lava.
- reversing (a[0] = -1) for 10 - 40 steps first, then a[0] = +1: the car jumps the lava and
  reaches the goal around step 70 - 90. The right wall is at x = 5.
""")


def _kx_flawed_expert(p, o, t):
    """Kinetix expert with added flaws (jax): ``p`` = {W [R, M, OBS], b [R, M], a [R], on [R]}."""
    import jax.numpy as jnp

    from weightbench.tracks.tr02_control_lib import kinetix_car as K

    a0 = K.expert_a0(o)
    inside = jnp.all(jnp.einsum("rmd,bd->brm", p["W"], o) + p["b"][None] > 0, axis=2) & (p["on"][None] > 0.5)
    for r in range(p["W"].shape[0]):
        a0 = jnp.where(inside[:, r], p["a"][r], a0)
    return jnp.zeros((o.shape[0], K.ACT_DIM)).at[:, 0].set(a0)


SPECS: dict[str, Spec] = {s.task_id: s for s in (KinetixSpec(), MountainCarSpec(), PendulumSpec(), CartPoleSpec())}
