"""TR02-LOOP reference solution: a gated rank-one edit that writes the closing-think direction.

Server side only (subject model; a CUDA GPU in practice). It is a privileged existence proof: it fits on
private calibration activations and selects by generation on a calibration-only development split that is
disjoint from evaluation; agents get none of this. Every subject execution goes through ``AuthorProbes``,
a separately capped append-only ledger in ``private/reference/``.

One ``down_proj`` receives ``ΔW = U Vᵀ`` (rank 1): ``U = strength · ‖h‖ · e`` with ``e`` the output embedding
of ``</think>`` and ``V`` a ridge gate on the MLP input. Two gate families:

* ``prompt``: V separates thinking-mode from answer-mode prompt ends, so thinking ends almost at once;
* ``loop``:   V separates MLP inputs inside the terminal repetition loop of the base model's own calibration
  traces from clean reasoning earlier in the same traces, so reasoning runs until it starts to loop.
"""
from __future__ import annotations

import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import numpy as np

from weightbench.ledger import Ledger

from . import config as C
from . import model as M
from .data import _pool, read_jsonl
from .edits import down_proj, factors_to_json, materialize

LAYERS = tuple(range(2 * C.N_LAYERS // 3, C.N_LAYERS))
STRENGTHS = (.125, .25, .5, 1., 2., 4.)
POSITIONS = 48          # sampled positions per trace and class for fitting the loop gate
CLEAN_MARGIN = 256      # clean positions end this many tokens before the loop onset
CLEAN_MAX = 1e-3        # tolerated per-token false-closure rate on clean reasoning
HELD_OUT = 8            # calibration traces kept out of the loop-gate fit, used to screen strengths
CHUNK = 256             # positions per logit chunk (vocabulary ≈ 248k)

# ---- holdout selection rule (fixed in advance; nothing tuned on eval)
# The 64 development items are split by a seeded permutation into a selection split S and a disjoint holdout H.
# Shortlisted candidates are compared on S only; the single S winner is then scored once on H next to the
# unedited base on H. The edit is accepted only if objective(winner, H) - acc(base, H) >= HOLDOUT_DELTA_ITEMS/|H|;
# otherwise the reference abstains (submits dW = 0, which scores at the floor) and records why.
# Rationale: outcomes are paired with base, so the H gain is (wins - losses)/|H| over discordant items. For an
# edit with no true effect and per-item discordant rate q, P(net >= 4 of 40) = 0.008 (q = .05, the decoy regime
# with base accuracy 2-3%), 0.074 (q = .15), 0.13 (q = .25). Power for a true gain g at discordant rate q (exact
# trinomial): 0.57 (g = .10, q = .15), 0.83 (g = .15, q = .20), 0.94 (g = .20, q = .25). |H| = 40 is the largest
# holdout that leaves a usable S (24 items). A one-item dev signal (+1/24) can never pass. One test is made on H,
# so no multiplicity correction applies.
SELECTION_N = 24
HOLDOUT_N = 40
HOLDOUT_DELTA_ITEMS = 4
HOLDOUT_DELTA = HOLDOUT_DELTA_ITEMS / HOLDOUT_N
assert SELECTION_N + HOLDOUT_N == C.REF_DEV_N
MMLU_PENALTY = 2.0      # objective = acc - MMLU_PENALTY * max(0, MMLU_base - MMLU_edit)

# ---- per-task search settings. Defaults (LOOP-001..003): strength grid STRENGTHS, the 2 best-CV gate layers + the
# last layer, 2 shortlisted candidates per family, development items from the calibration family's pool.
# LOOP-004 evaluates on MATH L4-5 while its calibration family is GSM8K; selecting out of domain favours a prompt gate
# that ends thinking at once. Its development split therefore comes from MATH L4-5 items disjoint from eval
# (privileged, like the in-domain development items of LOOP-001..003), with a half-step strength grid, the 3 best-CV
# layers + the last layer, and 3 shortlisted candidates per family. The holdout rule (sizes, delta, objective) is
# the same for every task.
DEFAULT_SEARCH: dict[str, Any] = {"strengths": STRENGTHS, "sites": 2, "shortlist": 2, "dev_pool": None}
TASK_SEARCH: dict[str, dict[str, Any]] = {
    "TR02-LOOP-004": {"strengths": (.125, .1875, .25, .375, .5, .75, 1., 1.5, 2., 4.), "sites": 3, "shortlist": 3,
                      "dev_pool": "math_l45"},
}


def search_settings(task_id: str) -> dict[str, Any]:
    """Reference search settings for ``task_id`` (``DEFAULT_SEARCH`` overridden by ``TASK_SEARCH``)."""
    return {**DEFAULT_SEARCH, **TASK_SEARCH.get(task_id, {})}


def activation_cap(settings: dict[str, Any], fit_n: int = 80, traces: int = 32) -> int:
    """Author activation budget that covers fitting plus screening every (site, strength) pair: 2 prompt-end passes per
    fitting item, one pass per loop-fit trace, and per candidate one prompt screen + ``HELD_OUT`` loop screens.
    Never below 400, the budget of the default search."""
    per_family = (settings["sites"] + 1) * len(settings["strengths"])
    return max(400, 2 * fit_n + traces + per_family * (1 + HELD_OUT))


class AuthorProbes:
    """Separate author-construction budget; never charged as a benchmarked agent run."""

    def __init__(self, path: Path, forward_cap: int = 50, activation_cap: int = 160):
        self.ledger = Ledger(path)
        self.caps = {"forward": forward_cap, "activation": activation_cap}
        self.used = {k: 0 for k in self.caps}
        if any(self.ledger.entries()):
            raise ValueError("use a fresh experiment directory; author ledger already exists")
        self.ledger.append("author_start", {"caps": self.caps, "official_agent_run": False})

    def call(self, kind: str, label: str, fn: Callable[[], Any]) -> Any:
        if self.used[kind] >= self.caps[kind]:
            self.ledger.append("probe_rejected", {"kind": kind, "label": label})
            raise RuntimeError(f"author {kind} budget exhausted")
        self.used[kind] += 1
        self.ledger.append("probe_start", {"kind": kind, "label": label, "used": self.used[kind]})
        t0 = time.time()
        try:
            result = fn()
        except BaseException as exc:
            self.ledger.append("probe_error", {"label": label, "error": str(exc)})
            raise
        self.ledger.append("probe", {"kind": kind, "label": label, "seconds": time.time() - t0})
        return result


def calibration_split(fit: list[dict], pool: list[dict], excluded: set[str], seed: int,
                      task_id: str, dev_pool: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """Extend fitting to 80 items and reserve ``C.REF_DEV_N`` (64) disjoint development items (never eval items).

    The development items come from ``pool`` after the fitting extension, or, when ``dev_pool`` is given, from a
    seeded permutation of ``dev_pool`` minus the eval and fitting items (``TASK_SEARCH`` ``dev_pool``)."""
    fit_ids = [x["id"] for x in fit]
    if len(set(fit_ids)) != len(fit_ids) or set(fit_ids) & excluded:
        raise ValueError("original fitting IDs must be unique and disjoint from evaluation")
    used = excluded | set(fit_ids)
    rows = list({x["id"]: x for x in pool if x["id"] not in used}.values())
    rows = [rows[i] for i in np.random.default_rng([seed, int(task_id[-3:]), 20261108]).permutation(len(rows))]
    extra = max(0, 80 - len(fit))
    if dev_pool is None:
        if len(rows) < extra + C.REF_DEV_N:
            raise ValueError("insufficient disjoint fitting/development items")
        return fit + rows[:extra], rows[extra:extra + C.REF_DEV_N]
    if len(rows) < extra:
        raise ValueError("insufficient disjoint fitting items")
    fit = fit + rows[:extra]
    used = excluded | {x["id"] for x in fit}
    dev = list({x["id"]: x for x in dev_pool if x["id"] not in used}.values())
    if len(dev) < C.REF_DEV_N:
        raise ValueError("insufficient disjoint development items in the development pool")
    dev = [dev[i] for i in np.random.default_rng([seed, int(task_id[-3:]), 20261111]).permutation(len(dev))]
    return fit, dev[:C.REF_DEV_N]


def holdout_split(dev: list[dict], seed: int, task_id: str) -> tuple[list[dict], list[dict]]:
    """Seeded split of the development items into (selection, holdout) of sizes ``SELECTION_N``/``HOLDOUT_N``."""
    ids = [x["id"] for x in dev]
    if len(dev) != SELECTION_N + HOLDOUT_N or len(set(ids)) != len(ids):
        raise ValueError(f"holdout split needs {SELECTION_N + HOLDOUT_N} unique development items")
    perm = np.random.default_rng([seed, int(task_id[-3:]), 20261110]).permutation(len(dev))
    return [dev[i] for i in perm[:SELECTION_N]], [dev[i] for i in perm[SELECTION_N:]]


def objective(measured: dict, base_mmlu: float) -> float:
    """Target accuracy minus the MMLU collateral penalty relative to the base model."""
    return measured["summary"]["acc"] - MMLU_PENALTY * max(0.0, base_mmlu - measured["mmlu"])


def holdout_select(shortlisted: list[int], score: Callable[[int | None, list[dict], str], dict],
                   selection: list[dict], holdout: list[dict], delta: float = HOLDOUT_DELTA) -> dict:
    """The holdout selection rule. ``score(index, items, label)`` (``index`` None = unedited base) returns
    ``{"summary": {"acc": ...}, "mmlu": ...}``; in ``run`` each call is two metered author forwards.

    Returns a record with ``accepted``, ``winner`` (candidate index or None) and ``reason``. At most
    ``len(shortlisted) + 3`` score calls are made (base and candidates on S, then base and winner on H)."""
    base_sel = score(None, selection, f"base/selection{len(selection)}")
    rec: dict[str, Any] = {"rule": "holdout", "selection_n": len(selection), "holdout_n": len(holdout),
                           "delta": delta, "base_selection": base_sel, "search": [], "winner": None,
                           "accepted": False}
    for index in shortlisted:
        m = score(index, selection, f"selection/{index}")
        rec["search"].append({"index": index, **m, "objective": objective(m, base_sel["mmlu"])})
    if not rec["search"]:
        rec["reason"] = "no shortlisted candidates"
        return rec
    best = min(rec["search"], key=lambda r: (-r["objective"], r["index"]))
    rec["winner"] = best["index"]
    rec["selection_gain"] = best["objective"] - base_sel["summary"]["acc"]
    if rec["selection_gain"] <= 0:
        rec["reason"] = "no candidate beats the base model on the selection split"
        return rec
    base_h = score(None, holdout, f"base/holdout{len(holdout)}")
    win_h = score(best["index"], holdout, f"holdout/{best['index']}")
    rec.update(base_holdout=base_h, winner_holdout=win_h)
    rec["holdout_gain"] = objective(win_h, base_h["mmlu"]) - base_h["summary"]["acc"]
    rec["accepted"] = bool(rec["holdout_gain"] >= delta - 1e-12)
    rec["reason"] = (f"holdout gain {rec['holdout_gain']:.4f} {'>=' if rec['accepted'] else '<'} "
                     f"delta {delta:.4f} on {len(holdout)} holdout items")
    return rec


def fit_gate(positive: np.ndarray, negative: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Ridge gate without bias (dual form): positive rows map to one, negative rows to zero."""
    p, n = np.asarray(positive, np.float64), np.asarray(negative, np.float64)
    if (p.ndim != 2 or n.ndim != 2 or p.shape[1] != n.shape[1] or not len(p) or not len(n)
            or not p.shape[1] or not np.isfinite(ridge) or ridge <= 0
            or not np.isfinite(p).all() or not np.isfinite(n).all()):
        raise ValueError("gate requires finite, nonempty matching matrices and positive ridge")
    x = np.concatenate((p, n))
    gram = x @ x.T
    scale = float(np.trace(gram) / len(x))
    if scale <= 1e-12:
        raise ValueError("degenerate gate inputs")
    y = np.concatenate((np.ones(len(p)), np.zeros(len(n))))
    return (x.T @ np.linalg.solve(gram + ridge * scale * np.eye(len(x)), y)).astype(np.float32)


def closure_factors(direction: np.ndarray, gate: np.ndarray, scale: float, strength: float) -> dict:
    """Rank-one factors writing ``strength · scale · direction/‖direction‖`` wherever the gate reads one."""
    d, v = np.asarray(direction, np.float64), np.asarray(gate, np.float64)
    if (d.ndim != 1 or v.ndim != 1 or not len(d) or not len(v) or not np.isfinite(d).all()
            or not np.isfinite(v).all() or np.linalg.norm(d) < 1e-12 or np.linalg.norm(v) < 1e-12
            or not np.isfinite(scale) or scale <= 0 or not np.isfinite(strength) or strength <= 0):
        raise ValueError("closure factors require finite vectors and positive scales")
    return {"U": (strength * scale * d / np.linalg.norm(d))[None].astype(np.float32),
            "V": v[None].astype(np.float32)}


def shortlist(candidates: list[dict], limit: int = 2) -> list[dict]:
    """Prompt gates: favour full closure, then the smallest effective strength, across distinct layers."""
    ordered = sorted(candidates, key=lambda c: (-c["closure_rate"],
                     c["strength"] if c["closure_rate"] >= .875 else -c["margin"], c["index"]))
    chosen, layers = [], set()
    for c in ordered:
        if c["closure_rate"] >= ordered[0]["closure_rate"] - .125 and c["layer"] not in layers:
            chosen.append(c)
            layers.add(c["layer"])
    indices = {c["index"] for c in chosen}
    return (chosen + [c for c in ordered if c["index"] not in indices])[:limit]


def loop_shortlist(candidates: list[dict], limit: int = 2) -> list[dict]:
    """Loop gates: spare clean reasoning (false-closure rate ≤ ``CLEAN_MAX``), then close loops most often."""
    return sorted(candidates, key=lambda c: (c["clean_rate"] > CLEAN_MAX, -c["loop_rate"], c["clean_rate"],
                                             c["index"]))[:limit]


def trace_positions(gen: dict, rng: np.random.Generator, n: int | None = POSITIONS) -> dict[str, np.ndarray] | None:
    """Generated-token positions inside the terminal loop and in clean reasoning before it (``n`` sampled per
    class; all of them if ``n`` is None), or None for traces without a loop or without enough clean prefix."""
    onset, length = gen.get("loop_onset_token"), len(gen["ids"])
    if onset is None or onset - CLEAN_MARGIN < 16 or length - onset < 16:
        return None
    loop, clean = np.arange(onset, length), np.arange(onset - CLEAN_MARGIN)
    if n is None:
        return {"loop": loop, "clean": clean}
    return {"loop": np.sort(rng.choice(loop, min(n, len(loop)), replace=False)),
            "clean": np.sort(rng.choice(clean, min(n, len(clean)), replace=False))}


def _prefix(rt: M.Runtime, prompts: list[str], token: int, thinking: bool, layers: tuple[int, ...] = ()) -> dict:
    """Prompt-end MLP inputs / block norms (one prompt when capturing) and the rate at which ``token`` is the
    argmax next token. One decoder execution; callers wrap it in ``AuthorProbes.call``."""
    import torch

    if layers and len(prompts) != 1:
        raise ValueError("activation fitting captures one prefix per ledger entry")
    inputs, norms = {}, {}
    with ExitStack() as hooks:
        for layer in layers:
            block = rt.model.get_submodule(f"model.layers.{layer}")

            def capture_input(module, args, l=layer):
                inputs[l] = args[0][0, -1].detach().float().cpu().numpy().copy()

            def capture_norm(module, args, output, l=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                norms[l] = float(hidden[0, -1].detach().float().norm())
            hooks.callback(block.mlp.down_proj.register_forward_pre_hook(capture_input).remove)
            hooks.callback(block.register_forward_hook(capture_norm).remove)
        texts = [M.chat(rt.tok, prompt, thinking) for prompt in prompts]
        enc = rt.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(rt.dev)
        with torch.no_grad():
            hidden = rt.model.model(**enc, use_cache=False).last_hidden_state
            logits = rt.model.get_output_embeddings()(hidden[:, -1])
            target = logits[:, token].clone()
            first = (logits.argmax(-1) == token).float().mean().item()
            logits[:, token] = -torch.inf
            margin = (target - logits.max(-1).values).mean().item()
        return {"inputs": inputs, "norms": norms, "closure_rate": first, "margin": margin}


def _trace(rt: M.Runtime, prompt: str, ids: list[int], positions: dict[str, np.ndarray], token: int,
           layers: tuple[int, ...] = ()) -> dict:
    """Teacher-forced pass over a thinking prompt and its generated ``ids``. Per position class: MLP inputs and
    mean block-output norm at each captured layer (the state that decides the next token), and the rate at
    which ``token`` is the argmax next token. One decoder execution; callers wrap it in ``AuthorProbes.call``."""
    import torch

    prompt_ids = rt.tok(M.chat(rt.tok, prompt), add_special_tokens=False)["input_ids"]
    at = {k: torch.as_tensor(len(prompt_ids) + v, device=rt.dev) for k, v in positions.items()}
    inputs: dict[str, dict] = {k: {} for k in positions}
    norms: dict[str, dict] = {k: {} for k in positions}
    with ExitStack() as hooks:
        for layer in layers:
            block = rt.model.get_submodule(f"model.layers.{layer}")

            def capture_input(module, args, l=layer):
                for k, idx in at.items():
                    inputs[k][l] = args[0][0, idx].detach().float().cpu().numpy()

            def capture_norm(module, args, output, l=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                for k, idx in at.items():
                    norms[k][l] = float(hidden[0, idx].detach().float().norm(dim=-1).mean())
            hooks.callback(block.mlp.down_proj.register_forward_pre_hook(capture_input).remove)
            hooks.callback(block.register_forward_hook(capture_norm).remove)
        x = torch.tensor([prompt_ids + list(ids)], device=rt.dev)
        with torch.no_grad():
            hidden = rt.model.model(input_ids=x, use_cache=False).last_hidden_state[0]
            head = rt.model.get_output_embeddings()
            rates = {k: float(torch.cat([head(hidden[idx[s:s + CHUNK]]).argmax(-1) == token
                                         for s in range(0, len(idx), CHUNK)]).float().mean())
                     for k, idx in at.items()}
    return {"inputs": inputs, "norms": norms, "rates": rates}


def run(ad: Path) -> dict:
    """Fit both gate families, screen strengths cheaply, then apply the holdout selection rule
    (``holdout_select``). Returns the accepted edit (``factors``), or ``abstained: True`` with empty factors."""
    spec = json.loads((ad / "private/spec.json").read_text())
    out = ad / "private/reference"
    out.mkdir(exist_ok=True)
    settings = search_settings(spec["task_id"])
    probes = AuthorProbes(out / "ledger.jsonl", forward_cap=max(20, 2 * (2 * settings["shortlist"] + 3)),
                          activation_cap=activation_cap(settings, traces=spec.get("calib_n", 32)))
    rt = M.load(ad)
    token, = rt.tok.encode("</think>", add_special_tokens=False)
    direction = rt.model.get_output_embeddings().weight[token].detach().float().cpu().numpy()
    excluded = {x["id"] for x in read_jsonl(ad / "private/eval.jsonl")}
    fit, dev = calibration_split(read_jsonl(ad / "private/calib.jsonl"), _pool(spec["calib"]),
                                 excluded, spec["seed"], spec["task_id"],
                                 _pool(settings["dev_pool"]) if settings["dev_pool"] else None)
    for name, rows in (("fit", fit), ("dev", dev)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    shapes = {down_proj(l): (C.HIDDEN, C.MLP) for l in LAYERS}

    def sites(cv: dict[int, float]) -> list[int]:
        return list(dict.fromkeys([*sorted(LAYERS, key=lambda l: (cv[l], l))[:settings["sites"]], LAYERS[-1]]))

    def edit(layer: int, gate: np.ndarray, scale: float, strength: float) -> dict:
        return {down_proj(layer): closure_factors(direction, gate, scale, strength)}

    def applied(factors: dict, fn: Callable[[], Any]) -> Any:
        rt.apply_delta(materialize(factors, shapes))
        try:
            return fn()
        finally:
            rt.restore()

    # ---- prompt gates: thinking vs answer prompt ends on the 80 fitting items
    pos, neg, scales = ({l: [] for l in LAYERS} for _ in range(3))
    for item in fit:
        for thinking, store in ((True, pos), (False, neg)):
            got = probes.call("activation", f"prompt_fit/{'thinking' if thinking else 'answer'}/{item['id']}",
                              lambda: _prefix(rt, [item["prompt"]], token, thinking, LAYERS))
            for l in LAYERS:
                store[l].append(got["inputs"][l])
                if thinking:
                    scales[l].append(got["norms"][l])
    cv, gates = {}, {}
    for l in LAYERS:
        p, n = np.stack(pos[l]), np.stack(neg[l])
        g = fit_gate(p[:60], n[:60])
        cv[l] = float(np.mean((p[60:] @ g - 1) ** 2) + np.mean((n[60:] @ g) ** 2))
        gates[l] = fit_gate(p, n)
    prompts = [item["prompt"] for item in dev[:8]]
    candidates: list[dict] = []
    prompt_screen = []
    for l in sites(cv):
        for strength in settings["strengths"]:
            factors = edit(l, gates[l], float(np.mean(scales[l])), strength)
            index = len(candidates)
            got = applied(factors, lambda: probes.call("activation", f"prompt_screen/{index}",
                                                       lambda: _prefix(rt, prompts, token, True)))
            candidates.append({"index": index, "family": "prompt", "layer": l, "strength": strength,
                               "factors": factors})
            prompt_screen.append({"index": index, "layer": l, "strength": strength,
                                  "closure_rate": got["closure_rate"], "margin": got["margin"]})

    # ---- loop gates: loop-tail vs clean-reasoning states in the base model's calibration traces
    gens = {g["id"]: g for g in json.loads((ad / "private/calib_generations.json").read_text())["items"]}
    rng = np.random.default_rng([spec["seed"], 20261109])
    traces = [(gens[i["id"]], p) for i in read_jsonl(ad / "private/calib.jsonl")
              if i["id"] in gens and (p := trace_positions(gens[i["id"]], rng)) is not None]
    loop_screen = []
    if len(traces) >= HELD_OUT + 4:
        fit_traces, held = traces[:-HELD_OUT], traces[-HELD_OUT:]
        lp, cl, sc = ({l: [] for l in LAYERS} for _ in range(3))
        for k, (g, p) in enumerate(fit_traces):
            got = probes.call("activation", f"loop_fit/{g['id']}", lambda: _trace(rt, g["prompt"], g["ids"], p, token,
                                                                               LAYERS))
            for l in LAYERS:
                lp[l].append(got["inputs"]["loop"][l])
                cl[l].append(got["inputs"]["clean"][l])
                sc[l].append(got["norms"]["loop"][l])
        cut = max(1, len(fit_traces) * 3 // 4)
        loop_cv, loop_gates = {}, {}
        for l in LAYERS:
            g = fit_gate(np.concatenate(lp[l][:cut]), np.concatenate(cl[l][:cut]))
            loop_cv[l] = float(np.mean((np.concatenate(lp[l][cut:]) @ g - 1) ** 2)
                               + np.mean((np.concatenate(cl[l][cut:]) @ g) ** 2))
            loop_gates[l] = fit_gate(np.concatenate(lp[l]), np.concatenate(cl[l]))
        full = [(g, trace_positions(g, rng, None)) for g, _ in held]
        for l in sites(loop_cv):
            for strength in settings["strengths"]:
                factors = edit(l, loop_gates[l], float(np.mean(sc[l])), strength)
                index = len(candidates)
                rates = [applied(factors, lambda: probes.call(
                    "activation", f"loop_screen/{index}/{g['id']}",
                    lambda: _trace(rt, g["prompt"], g["ids"], p, token)))["rates"] for g, p in full]
                candidates.append({"index": index, "family": "loop", "layer": l, "strength": strength,
                                   "factors": factors})
                loop_screen.append({"index": index, "layer": l, "strength": strength,
                                    "loop_rate": float(np.mean([r["loop"] for r in rates])),
                                    "clean_rate": float(np.mean([r["clean"] for r in rates]))})
        (out / "loop_gate_cv.json").write_text(json.dumps(loop_cv))

    # ---- generation: holdout selection rule (HOLDOUT_* constants above)
    mmlu = read_jsonl(ad / "private/mmlu_dev.jsonl")
    selection, holdout = holdout_split(dev, spec["seed"], spec["task_id"])
    for name, rows in (("selection", selection), ("holdout", holdout)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def score(index: int | None, items: list[dict], label: str) -> dict:
        factors = candidates[index]["factors"] if index is not None else None
        rt.apply_delta(materialize(factors, shapes) if factors else None)
        try:
            _, rows = probes.call("forward", label, lambda: M.run_items(rt, items, spec["max_new_tokens"]))
            mm = probes.call("forward", label + "/mmlu", lambda: M.mmlu_accuracy(rt, mmlu))["acc"]
            return {"summary": M.summarize(rows), "mmlu": mm}
        finally:
            rt.restore()

    k = settings["shortlist"]
    shortlisted = [c["index"] for c in shortlist(prompt_screen, k) + loop_shortlist(loop_screen, k)]
    sel = holdout_select(shortlisted, score, selection, holdout)
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    meta = {"fit_n": len(fit), "dev_n": len(dev), "selection_n": len(selection), "holdout_n": len(holdout),
            "n_loop_traces": len(traces), "base_dev": sel["base_selection"], "prompt_gate_cv": cv,
            "search_settings": {**settings, "strengths": list(settings["strengths"])},
            "dev_source": settings["dev_pool"] or spec["calib"],
            "prompt_screen": prompt_screen, "loop_screen": loop_screen, "search": sel["search"],
            "selection": {k: v for k, v in sel.items() if k != "search"}, "used": probes.used}
    if not sel["accepted"]:
        # Honest abstention: submit no edit (dW = 0 scores S+ = 0). ``ok`` stays True so callers submit it.
        return {"ok": True, "abstained": True, "reason": sel["reason"], "method": "abstain (holdout rule)",
                "choice": None, "finalists": [], "factors": {}, **meta}
    best = candidates[sel["winner"]]
    choice = {"index": best["index"], "family": best["family"], "layer": best["layer"],
              "strength": best["strength"], **sel["winner_holdout"], "holdout_gain": sel["holdout_gain"]}
    return {"ok": True, "abstained": False, "reason": sel["reason"],
            "method": f"gated closing-token edit ({best['family']} gate)", "choice": choice,
            "finalists": [choice], "factors": factors_to_json(best["factors"]), **meta}
