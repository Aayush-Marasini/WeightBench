"""Heavy-step CLI for TR02-LOOP. Every step reads files under --art-dir and writes ONE JSON to --out.

    python -m weightbench.tracks.tr02_loops_lib.jobs calib       --art-dir A --out O
    python -m weightbench.tracks.tr02_loops_lib.jobs eval        --art-dir A [--delta D.npz] [--limit N]
                                                                 [--max-new-tokens T] [--no-mmlu] --out O
    python -m weightbench.tracks.tr02_loops_lib.jobs generate    --art-dir A --payload P.json --out O
    python -m weightbench.tracks.tr02_loops_lib.jobs activations --art-dir A --payload P.json --out O
    python -m weightbench.tracks.tr02_loops_lib.jobs baseline    --art-dir A --out O
    python -m weightbench.tracks.tr02_loops_lib.jobs reference   --art-dir A --out O

Called through ``weightbench.heavy.run_heavy`` (a local subprocess).
``baseline``/``reference`` need ``private/calib_generations.json`` (output of ``calib``).
Device auto-detect: cuda > mps > cpu. Greedy decoding throughout (deterministic).
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np

from . import config as C
from .data import read_jsonl
from .edits import baseline_choice, down_proj, factors_to_json, projection_factors


def _spec(ad: Path) -> dict:
    return json.loads((ad / "private" / "spec.json").read_text())


def _load_delta(p: str | None) -> dict | None:
    if not p:
        return None
    with np.load(p) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def _compact(g: dict, r: dict) -> dict:
    return {**r, "text": g["text"]}


def step_calib(a, ad: Path) -> dict:
    from . import model as M
    from .textproc import loop_onset
    spec = _spec(ad)
    items = read_jsonl(ad / "private" / "calib.jsonl")
    rt = M.load(ad)
    t0 = time.time()
    gens, rows = M.run_items(rt, items, spec["max_new_tokens"])
    out = []
    for it, g, r in zip(items, gens, rows):
        out.append({**r, "prompt": it["prompt"], "text": g["text"], "ids": g["ids"], "finished": g["finished"],
                    "loop_onset_token": loop_onset(g["ids"]) if r["looped"] else None})
    return {"summary": M.summarize(rows), "items": out, "runtime": rt.info(), "seconds": round(time.time() - t0, 1),
            "max_new_tokens": spec["max_new_tokens"]}


def step_eval(a, ad: Path) -> dict:
    from . import model as M
    spec = _spec(ad)
    items = read_jsonl(ad / "private" / "eval.jsonl")
    if a.limit:
        items = items[:a.limit]
    T = a.max_new_tokens or spec["max_new_tokens"]
    rt = M.load(ad)
    rt.apply_delta(_load_delta(a.delta))
    t0 = time.time()
    gens, rows = M.run_items(rt, items, T)
    t_gen = time.time() - t0
    res = {"task": spec["eval"], "max_new_tokens": T, "summary": M.summarize(rows),
           "items": [_compact(g, r) for g, r in zip(gens, rows)], "runtime": rt.info(),
           "seconds_generate": round(t_gen, 1), "with_delta": bool(a.delta)}
    res["S"] = res["summary"]["acc"]
    if spec.get("concept_ids"):   # knob level: per-family summaries of the same generations (host takes min S+)
        row_of = {r["id"]: r for r in rows}
        res["by_concept"] = {c: M.summarize([row_of[i] for i in ids if i in row_of])
                             for c, ids in spec["concept_ids"].items()}
    if not a.no_mmlu:
        t1 = time.time()
        mm = M.mmlu_accuracy(rt, read_jsonl(ad / "private" / "mmlu_eval.jsonl"))
        res["mmlu"] = mm
        res["seconds_mmlu"] = round(time.time() - t1, 1)
    return res


def step_generate(a, ad: Path) -> dict:
    from . import model as M
    from .textproc import split_thinking
    p = json.loads(Path(a.payload).read_text())
    rt = M.load(ad)
    rt.apply_delta(_load_delta(p.get("delta")))
    t0 = time.time()
    gens = M.generate(rt, p["prompts"], int(p["max_new_tokens"]), thinking=bool(p.get("enable_thinking", True)))
    from .textproc import loop_stats
    out = []
    for g in gens:
        _, final = split_thinking(g["text"]) if p.get("enable_thinking", True) else ("", g["text"])
        out.append({"text": g["text"], "final_answer": final, "n_tokens": g["n_tokens"], "finished": g["finished"],
                    **loop_stats(g["ids"], g["finished"])})
    return {"outputs": out, "seconds": round(time.time() - t0, 1)}


def step_activations(a, ad: Path) -> dict:
    from . import model as M
    p = json.loads(Path(a.payload).read_text())
    rt = M.load(ad)
    rt.apply_delta(_load_delta(p.get("delta")))
    layers = [int(l) for l in p["layers"]]
    vecs, index = [], []
    ntok = []
    for ti, text in enumerate(p["texts"]):
        rngs = (p.get("char_ranges") or [None] * len(p["texts"]))[ti]
        ids, spans = M.char_ranges_to_tokens(rt.tok, text, rngs)
        ntok.append(len(ids))
        hs = M.residuals(rt, ids, layers)
        for ri, (s, e) in enumerate(spans):
            for l in layers:
                v = hs[l][s:e].mean(0) if e > s else hs[l].new_zeros(C.HIDDEN)
                vecs.append(v.cpu().numpy().astype(np.float16))
                index.append({"text": ti, "range": ri, "layer": l, "tokens": [s, e]})
    arr = np.stack(vecs) if vecs else np.zeros((0, C.HIDDEN), np.float16)
    return {"shape": list(arr.shape), "dtype": "float16", "data_b64": base64.b64encode(arr.tobytes()).decode(),
            "index": index, "n_tokens": ntok}


def _contrast(rt, ad: Path) -> tuple[dict, list[dict], list[dict]]:
    from . import model as M
    cg = json.loads((ad / "private" / "calib_generations.json").read_text())
    items = read_jsonl(ad / "private" / "calib.jsonl")
    by_id = {g["id"]: g for g in cg["items"]}
    gens = [by_id[it["id"]] for it in items]
    st = M.loop_contrast_stats(rt, items, gens)
    return st, items, gens


def _factors_for(rt, st: dict, choice: dict) -> dict:
    r = st["layers"][choice["src"]]["diff"]
    out = {}
    for l in choice["edit"]:
        name = down_proj(l)
        W = rt.base_weight(name).cpu().numpy()
        out[name] = projection_factors(W, r, choice["alpha"])
    return out


def step_baseline(a, ad: Path) -> dict:
    from . import model as M
    spec = _spec(ad)
    rt = M.load(ad)
    st, _, _ = _contrast(rt, ad)
    if not st["layers"]:
        return {"ok": False, "reason": "calibration set has no looping (or no clean) generations", "stats": st}
    sep = {l: v["sep"] for l, v in st["layers"].items()}
    choice = baseline_choice(sep, spec["constraint"])
    return {"ok": True, "choice": choice, "sep": sep, "n_tokens": st["n_tokens"],
            "factors": factors_to_json(_factors_for(rt, st, choice))}


def step_reference(a, ad: Path) -> dict:
    """Gated closing-token edit (``reference.run``): privileged existence proof fitted on calibration
    activations and selected on a calibration-only development split; never touches the eval set."""
    from .reference import run
    return run(ad)


STEPS = {"calib": step_calib, "eval": step_eval, "generate": step_generate, "activations": step_activations,
         "baseline": step_baseline, "reference": step_reference}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=sorted(STEPS))
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta")
    ap.add_argument("--payload")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--no-mmlu", action="store_true")
    a = ap.parse_args(argv)
    ad = Path(a.art_dir)
    t0 = time.time()
    res = STEPS[a.step](a, ad)
    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.rename(out)


if __name__ == "__main__":
    main()
