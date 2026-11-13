"""Difficulty knob for TR02-CTRL-001..004 (SPEC §5 ``num_entangled_concepts``, K = 2^L expert flaws).

Level L trains the policy (behaviour cloning, inside the normal metered build job) from an expert with K = 2^L
deliberate flaws: flaw 1 is the default task's own flaw (the expert's built-in defect), flaws 2..K are drawn from
the seed out of a per-environment family of *added* flaws. An added flaw is a region of observation space (an
intersection of half-spaces ``w . obs + b > 0``) in which the expert's action is overridden with a wrong one, chosen
so that a rollout entering the region gets stuck or fails (``Spec.flaw_family``).

Build-time validation (``attribute`` + ``validate``) runs inside the build job, like every other
author-side rollout of the build:

* every start of the pool is attributed with expert rollouts: *normal* (the K-flaw expert succeeds without entering
  any added region), *flaw 1* (the K-flaw expert fails without entering any added region, i.e. it fails exactly as
  the default expert does), or *flaw k* (the K-flaw expert enters only region k and fails, and the expert with flaw
  k removed succeeds without entering any other added region). Everything else is excluded.
* the cloned policy must fail on at least ``FAIL_RATE_MIN`` of every flaw's attributed starts (and on at least
  ``N_EVAL + n_train + 1`` of them) and succeed on at least ``NORM_RATE_MIN`` of the normal ones; otherwise the
  build raises with the flaw and the measured rates. A subject whose stuck situations are not real never ships.

Scoring at K > 1: per-flaw ``S+_k = s_plus_write(S_edit_k, S_base_k, 1)`` on that flaw's private held-out pool,
``S+ = min_k S+_k`` (mean reported), ``D_col`` from the normal pool as at K = 1. The reference adds one override
gadget per added flaw, built from further pruned units (``gadget``); a level whose K gadgets do not fit the
spare units is unsupported and rejected before any build (``check_capacity``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from weightbench.tracks.tr02_control_lib.specs import Spec

FAIL_RATE_MIN = 0.6     # policy failure rate on every flaw's expert-attributed starts
NORM_RATE_MIN = 0.8     # policy success rate on the expert-normal starts
FLAW_SEED_SALT = 0x7C71


@dataclass(frozen=True)
class Flaw:
    """One added expert flaw. ``ineqs``: half-spaces on the RAW observation, ``sum(c * obs[i]) + b > 0`` for all;
    ``action``: the wrong expert action inside (discrete index or continuous value); ``fix``: sign of the output
    correction the reference gadget applies (spec-specific); ``start_low/high``: supplementary start box sampled
    for this flaw's pool (state or variant coordinates). Everything except ``title`` is private."""

    name: str
    title: str
    ineqs: tuple[tuple[tuple[tuple[int, float], ...], float], ...]
    action: float
    fix: float
    start_low: tuple[float, ...]
    start_high: tuple[float, ...]
    params: dict[str, float] = field(default_factory=dict)

    def mask(self, obs: np.ndarray) -> np.ndarray:
        """Region membership of raw observations ``obs [..., D]``."""
        inside = np.ones(obs.shape[:-1], bool)
        for coefs, b in self.ineqs:
            v = np.full(obs.shape[:-1], float(b))
            for i, c in coefs:
                v = v + c * obs[..., i]
            inside &= v > 0
        return inside

    def dense(self, dim: int, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        """(W [n_rows, dim], b [n_rows]); padded rows are always true (W = 0, b = 1)."""
        W, b = np.zeros((n_rows, dim), np.float32), np.ones(n_rows, np.float32)
        for r, (coefs, bb) in enumerate(self.ineqs):
            for i, c in coefs:
                W[r, i] = c
            b[r] = bb
        return W, b

    def private(self) -> dict:
        return {"name": self.name, "title": self.title, "ineqs": [[list(map(list, c)), b] for c, b in self.ineqs],
                "action": self.action, "fix": self.fix, "start_low": list(self.start_low),
                "start_high": list(self.start_high), "params": dict(self.params)}


# ---- selection ---------------------------------------------------------------------------------
def n_flaws(level: int) -> int:
    return 2 ** int(level)


def capacity(spec: "Spec", n_added: int) -> tuple[int, int]:
    """(h0, h1) pruned units the reference needs for the flaw-1 circuit plus ``n_added`` gadgets."""
    if n_added == 0:
        return spec.circuit_units
    width = max(len(f.ineqs) for f in spec.flaw_family(np.random.default_rng(0)))
    return (spec.circuit_units[0] + spec.shared_h0 + n_added * width, spec.circuit_units[1] + n_added)


def check_capacity(spec: "Spec", level: int) -> None:
    need = capacity(spec, n_flaws(level) - 1)
    if need[0] > spec.n_pruned[0] or need[1] > spec.n_pruned[1]:
        raise ValueError(f"{spec.task_id}: level {level} needs {need} pruned units for K = {n_flaws(level)} "
                         f"override gadgets but the policy has {tuple(spec.n_pruned)}")


def select_flaws(spec: "Spec", seed: int, level: int) -> list[Flaw]:
    """The K - 1 added flaws of level ``level`` (flaw 1 is the task's own). Drawn from the seed only: the jitter of
    every family member and the order are fixed per seed, so levels are nested (L1's flaw is L2's first)."""
    levels = spec.knob_levels
    if not levels or level not in levels:
        raise ValueError(f"task {spec.task_id}: supported knob levels are {levels}, got {level}")
    check_capacity(spec, level)
    fam = spec.flaw_family(np.random.default_rng([int(seed), FLAW_SEED_SALT]))
    order = np.random.default_rng([int(seed), FLAW_SEED_SALT, 1]).permutation(len(fam))
    k = n_flaws(level) - 1
    if k > len(fam):
        raise ValueError(f"{spec.task_id}: family has {len(fam)} added flaws, level {level} needs {k}")
    return [fam[i] for i in order[:k]]


REDRAW_MAX = 8          # re-jittered family draws tried after the seed's own draw before a level is given up


def candidate_flaws(spec: "Spec", seed: int):
    """The seed's stream of candidate added flaws, independent of the level: first the seed's own jittered family in
    its seeded order (exactly what ``select_flaws`` draws), then up to ``REDRAW_MAX`` re-jittered draws of the family,
    each in its own seeded order. Yields ``(attempt, flaw)``."""
    for a in range(REDRAW_MAX + 1):
        fam_key = [int(seed), FLAW_SEED_SALT] if a == 0 else [int(seed), FLAW_SEED_SALT, 2, a]
        ord_key = [int(seed), FLAW_SEED_SALT, 1] if a == 0 else [int(seed), FLAW_SEED_SALT, 3, a]
        fam = spec.flaw_family(np.random.default_rng(fam_key))
        for i in np.random.default_rng(ord_key).permutation(len(fam)):
            yield a, fam[int(i)]


def n_attributable(spec: "Spec", flaw: Flaw, seed: int, index: int) -> int:
    """Starts of a check pool (``spec.sample_pool_level`` with this flaw alone, from its own seeded generator, never
    the build's) that ``attribute`` would assign to the flaw. At L1 a start's label depends only on the start, so
    this is the same quantity the build's validation counts, on an independent draw of the same pool."""
    starts, region = spec.sample_pool_level(np.random.default_rng([int(seed), FLAW_SEED_SALT, 4, int(index)]), [flaw])
    labels, _ = attribute(spec, [flaw], starts, region)
    return int((labels == 2).sum())


def select_reachable_flaws(spec: "Spec", seed: int, level: int, need: int, log: "Log" = print) -> list[Flaw]:
    """``select_flaws`` with a reachability check, for the build (it runs expert rollouts, so it belongs inside the
    heavy build job). A drawn flaw that fewer than ``need`` starts of a check pool can be attributed to can never pass
    build validation (it needs at least ``need`` attributed starts on which the clone fails), so it is skipped and the
    next candidate of the seed's stream (``candidate_flaws``) is tried. Deterministic and seed-derived; nested across
    levels (the stream does not depend on the level and each check looks at one flaw alone); and identical to
    ``select_flaws`` whenever its draw is reachable, so a level/seed that already built is unchanged."""
    first = select_flaws(spec, seed, level)            # level / capacity checks, and the fast path below
    k = len(first)
    if k == 0:
        return first
    out: list[Flaw] = []
    skipped: list[str] = []
    for n, (a, f) in enumerate(candidate_flaws(spec, seed)):
        if any(g.name == f.name for g in out):
            continue
        c = n_attributable(spec, f, seed, n)
        if c >= need:
            out.append(f)
            if len(out) == k:
                break
        else:
            skipped.append(f"{f.name}@draw{a} ({c} attributable < {need})")
            log(f"  flaw {f.name} (draw {a}, {f.params}): {c} attributable starts < {need}; redrawing")
    if len(out) < k:
        raise ValueError(f"{spec.task_id}: seed {seed} level {level}: only {len(out)} of {k} added flaws are "
                         f"reachable after {REDRAW_MAX} redraws; skipped {skipped}")
    if skipped:
        log(f"  added flaws after redraw: {[f.name for f in out]} (skipped {skipped})")
    return out


def apply_overrides(y: np.ndarray, obs: np.ndarray, flaws: list[Flaw], active: np.ndarray | None = None) -> np.ndarray:
    """Expert targets with the active flaws' wrong actions: ``y [N]`` (discrete) or ``y [N, A]`` (column 0)."""
    y = np.array(y, copy=True)
    for r, f in enumerate(flaws):
        if active is not None and not active[r]:
            continue
        m = f.mask(obs)
        if y.ndim == 1:
            y[m] = int(f.action)
        else:
            y[m, 0] = f.action
    return y


# ---- attribution + validation --------------------------------------------------------------------
NORMAL, EXCLUDED = 0, -1


def attribute(spec: "Spec", flaws: list[Flaw], starts: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, dict]:
    """Label every start (see module doc): 0 normal, 1 flaw 1, k = 2..K added flaw k-1, -1 excluded.
    ``region``: the default failure region (all True for classic tasks, ramp scenes for Kinetix)."""
    R = len(flaws)
    on = np.ones(R, bool)
    succ, vis = spec.run_expert(flaws, on, starts)
    nvis = vis.sum(1) if R else np.zeros(len(starts), int)
    labels = np.full(len(starts), EXCLUDED)
    norm_ok = ~region if (~region).any() else np.ones(len(starts), bool)
    labels[(nvis == 0) & succ & norm_ok] = NORMAL
    labels[(nvis == 0) & ~succ & region] = 1
    episodes = len(starts)
    for k in range(R):
        cand = np.flatnonzero((nvis == 1) & vis[:, k] & ~succ)
        if not len(cand):
            continue
        on_k = on.copy()
        on_k[k] = False
        s_k, v_k = spec.run_expert(flaws, on_k, starts[cand])
        others = np.delete(v_k, k, axis=1) if R > 1 else np.zeros((len(cand), 0), bool)
        ok = s_k & ~others.any(1)
        labels[cand[ok]] = k + 2
        episodes += len(cand)
    return labels, {"expert_episodes": int(episodes)}


def validate(labels: np.ndarray, policy_success: np.ndarray, names: list[str], need_fail: int, need_norm: int,
             ) -> dict:
    """Per-flaw policy failure rates on the expert-attributed starts and the normal success rate. Returns the report
    if every flaw took, else raises RuntimeError naming each flaw that did not and its measured rates."""
    ps = np.asarray(policy_success, bool)
    rep: dict[str, Any] = {"fail_rate_min": FAIL_RATE_MIN, "norm_rate_min": NORM_RATE_MIN,
                           "need_fail": need_fail, "need_norm": need_norm, "flaws": {}}
    problems = []
    for c, name in enumerate(names, start=1):
        cand = labels == c
        n, nf = int(cand.sum()), int((cand & ~ps).sum())
        rate = nf / n if n else 0.0
        rep["flaws"][name] = {"flaw": c, "n_attributed": n, "n_policy_fail": nf, "policy_fail_rate": round(rate, 4)}
        if rate < FAIL_RATE_MIN or nf < need_fail:
            problems.append(f"flaw {c} ({name}): policy failure rate {rate:.3f} on {n} expert-attributed starts, "
                            f"{nf} failing (need rate >= {FAIL_RATE_MIN} and >= {need_fail} failing starts)")
    cand = labels == NORMAL
    n, ns = int(cand.sum()), int((cand & ps).sum())
    rate = ns / n if n else 0.0
    rep["normal"] = {"n_attributed": n, "n_policy_success": ns, "policy_success_rate": round(rate, 4)}
    if rate < NORM_RATE_MIN or ns < need_norm:
        problems.append(f"normal pool: policy success rate {rate:.3f} on {n} expert-normal starts, {ns} succeeding "
                        f"(need rate >= {NORM_RATE_MIN} and >= {need_norm})")
    rep["ok"] = not problems
    rep["problems"] = problems
    if problems:
        raise FlawValidationError("TR02-CTRL knob build validation failed: " + "; ".join(problems), rep)
    return rep


class FlawValidationError(RuntimeError):
    def __init__(self, msg: str, report: dict):
        super().__init__(msg)
        self.report = report


def per_flaw_s_plus(s_edit: dict[str, float], s_base: dict[str, float]) -> dict:
    """``S+_k = s_plus_write(S_edit_k, S_base_k, 1)``; the task S+ is the minimum, the mean is reported."""
    from weightbench import scoring

    if set(s_edit) != set(s_base) or not s_base:
        raise ValueError(f"flaw sets differ: {sorted(s_edit)} vs {sorted(s_base)}")
    per = {k: scoring.s_plus_write(s_edit[k], s_base[k], 1.0) for k in s_base}
    return {"s_plus_by_flaw": per, "s_plus": min(per.values()), "s_plus_mean": sum(per.values()) / len(per)}


# ---- reference gadgets -------------------------------------------------------------------------------
def unit_plan(spec: "Spec", pruned: dict[int, list[int]], flaws: list[Flaw]) -> dict:
    """Pruned units per role: the flaw-1 circuit takes the first ``circuit_units``, then ``shared_h0`` shared h0
    units, then each added flaw ``len(ineqs)`` h0 gates and one h1 unit."""
    c0, c1 = spec.circuit_units
    p0, p1 = list(pruned[0]), list(pruned[1])
    i0 = c0 + spec.shared_h0
    shared = p0[c0:i0]
    per = []
    for j, f in enumerate(flaws):
        g = p0[i0:i0 + len(f.ineqs)]
        i0 += len(f.ineqs)
        w = p1[c1 + j:c1 + j + 1]
        if len(g) < len(f.ineqs) or not w:
            raise ValueError(f"{spec.task_id}: not enough pruned units for {len(flaws)} gadgets")
        per.append((g, w[0]))
    return {"shared": shared, "gadgets": per}


def gadget(spec: "Spec", base: dict, plan: dict, j: int, flaw: Flaw, th: dict) -> dict:
    """ΔW of the override gadget for added flaw ``j``: one gate per half-space (``relu`` of the signed distance to
    the boundary, zero inside the region), one h1 unit ``relu(1 - G * sum(gates) + spec terms)`` and an output
    correction (``spec.wire_gadget``)."""
    d = {k: np.zeros_like(v) for k, v in base.items()}
    gates, w = plan["gadgets"][j]
    scale, shift = spec.norm_affine()
    for u, (coefs, b) in zip(gates, flaw.ineqs):
        wr = np.zeros(len(scale))
        for i, c in coefs:
            wr[i] = c
        wn, bn = wr * scale, float(wr @ shift + b)      # half-space in normalised network inputs
        nrm = float(np.linalg.norm(wn)) or 1.0
        d["fc0.weight"][u] = -th["kg"] * wn / nrm
        d["fc0.bias"][u] = -th["kg"] * bn / nrm
    d["fc1.weight"][w, gates] = -th["G"]
    d["fc1.bias"][w] = 1.0
    spec.wire_gadget(d, plan, w, flaw, th)
    return d


def add(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


# ---- seeded jitter helper ------------------------------------------------------------------------------
def jit(rng: np.random.Generator, lo: float, hi: float) -> float:
    return round(float(rng.uniform(lo, hi)), 4)


Log = Callable[[str], None]
