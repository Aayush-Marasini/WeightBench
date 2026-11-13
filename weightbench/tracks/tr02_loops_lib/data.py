"""Build-time dataset preparation (runs on the build host; needs `datasets`, no model)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from .ifeval import supported


def _load(key: str, split: str):
    from datasets import load_dataset
    name, cfg, rev = C.DATASETS[key]
    return load_dataset(name, cfg, split=split, revision=rev)


def _math_item(r) -> dict:
    return {"id": r["unique_id"], "kind": "math", "prompt": r["problem"] + C.MATH_SUFFIX, "gold": r["answer"],
            "level": int(r["level"]), "subject": r["subject"]}


def _gsm_item(r, split: str, i: int) -> dict:
    from .textproc import gsm8k_gold
    return {"id": f"gsm8k/{split}/{i}", "kind": "gsm8k", "prompt": r["question"] + C.MATH_SUFFIX,
            "gold": gsm8k_gold(r["answer"])}


def _ifeval_item(r) -> dict:
    return {"id": f"ifeval/{r['key']}", "kind": "ifeval", "prompt": r["prompt"],
            "instruction_id_list": list(r["instruction_id_list"]),
            "kwargs": [{k: v for k, v in kw.items() if v is not None} for kw in r["kwargs"]]}


def _pool(name: str) -> list[dict]:
    if name in ("gsm8k_test", "gsm8k_train"):
        split = name.split("_")[1]
        ds = _load("gsm8k", split)
        return [_gsm_item(r, split, i) for i, r in enumerate(ds)]
    if name.startswith("math_"):
        ds = _load("math500", "test")
        levels = {"math_l123": (1, 2, 3), "math_l123_rest": (1, 2, 3), "math_l45": (4, 5)}[name]
        return [_math_item(r) for r in ds if int(r["level"]) in levels]
    if name.startswith("ifeval"):
        ds = _load("ifeval", "train")
        return [_ifeval_item(r) for r in ds if supported(list(r["instruction_id_list"]))]
    raise KeyError(name)


def _mmlu(split: str, n: int, rng: np.random.Generator) -> list[dict]:
    ds = _load("mmlu", split)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    return [{"id": f"mmlu/{split}/{int(i)}", "kind": "mmlu", "question": ds[int(i)]["question"],
             "choices": list(ds[int(i)]["choices"]), "answer": int(ds[int(i)]["answer"]),
             "subject": ds[int(i)]["subject"]} for i in sorted(idx)]


def build_data(task_id: str, art_dir: Path, seed: int, level: int | None = None) -> dict:
    """Write private/{eval,calib,mmlu_eval,mmlu_dev}.jsonl + private/spec.json and the public
    calibration problems. Eval and calibration sets are disjoint; selection is seeded.

    ``level`` (SPEC §5 knob, explicit builds only): K = 2^level problem families. The task's own eval,
    calibration and MMLU sets are drawn exactly as in the default build; the K-1 extra families add ``eval_n``
    items each (separate RNG stream, disjoint from every other eval and calibration item) to private/eval.jsonl,
    and private/spec.json records ``level``, ``concepts`` and ``concept_ids``. ``level=None`` is the default build."""
    concepts = C.select_loop_concepts(task_id, seed, level) if level is not None else None
    spec = dict(C.TASK_SPECS[task_id])
    sizes = {"eval_n": spec["eval_n"], "calib_n": spec["calib_n"], "max_new_tokens": spec["max_new_tokens"],
             "mmlu_eval_n": C.MMLU_EVAL_N, "mmlu_dev_n": C.MMLU_DEV_N, "ref_dev_n": C.REF_DEV_N}
    if C.smoke_mode():
        sizes.update(C.SMOKE)
    spec.update(sizes)
    rng = np.random.default_rng([seed, int(task_id[-3:])])

    eval_pool = _pool(spec["eval"])
    perm = rng.permutation(len(eval_pool))
    eval_items = [eval_pool[i] for i in sorted(perm[:spec["eval_n"]])]
    used = {it["id"] for it in eval_items}
    calib_pool = [it for it in _pool(spec["calib"]) if it["id"] not in used]
    perm = rng.permutation(len(calib_pool))
    calib_items = [calib_pool[i] for i in sorted(perm[:spec["calib_n"]])]

    concept_ids = None
    if concepts is not None:
        concept_ids = {concepts[0]: [it["id"] for it in eval_items]}
        used |= {it["id"] for it in calib_items}
        for j, c in enumerate(concepts[1:], 1):
            pool = [it for it in _pool(c) if it["id"] not in used]
            perm = np.random.default_rng([seed, int(task_id[-3:]), 31, j]).permutation(len(pool))
            extra = [pool[i] for i in sorted(perm[:spec["eval_n"]])]
            if len(extra) < spec["eval_n"]:
                raise ValueError(f"family {c}: only {len(extra)} disjoint items for eval_n={spec['eval_n']}")
            concept_ids[c] = [it["id"] for it in extra]
            used |= set(concept_ids[c])
            eval_items = eval_items + extra

    priv, pub = art_dir / "private", art_dir / "public"
    (pub / "calibration").mkdir(parents=True, exist_ok=True)

    def dump(p: Path, rows: list[dict]) -> None:
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))

    dump(priv / "eval.jsonl", eval_items)
    dump(priv / "calib.jsonl", calib_items)
    dump(pub / "calibration" / "problems.jsonl", calib_items)  # calibration answers are public
    dump(priv / "mmlu_eval.jsonl", _mmlu("test", spec["mmlu_eval_n"], rng))
    dump(priv / "mmlu_dev.jsonl", _mmlu("validation", spec["mmlu_dev_n"], rng))
    spec.update({"task_id": task_id, "seed": seed, "model_id": C.MODEL_ID, "model_rev": C.MODEL_REV,
                 "datasets": C.DATASETS, "smoke": C.smoke_mode()})
    if concepts is not None:
        spec.update({"level": level, "concepts": concepts, "concept_ids": concept_ids})
    (priv / "spec.json").write_text(json.dumps(spec, indent=1))
    return spec


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
