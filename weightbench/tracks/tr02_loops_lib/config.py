"""Pinned versions and per-task specs for TR02-LOOP (Reasoning Loop Abatement)."""
from __future__ import annotations

import os

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REV = "2fc06364715b967f1860aea9cf38778875588b17"
# files copied (APFS clone) from the pinned HF snapshot into public/model/
MODEL_FILES = ("config.json", "model.safetensors-00001-of-00001.safetensors", "model.safetensors.index.json",
               "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")
WEIGHTS_FILE = "model.safetensors-00001-of-00001.safetensors"

# parameter naming: file name (agent-facing) -> transformers Qwen3_5ForCausalLM name
FILE_PREFIX = "model.language_model."
MODEL_PREFIX = "model."
N_LAYERS = 24
HIDDEN = 1024
MLP = 3584

DATASETS = {  # pinned dataset revisions (HF hub commit sha)
    "gsm8k": ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866"),
    "math500": ("HuggingFaceH4/MATH-500", None, "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"),
    "ifeval": ("google/IFEval", None, "966cd89545d6b6acfd7638bc708b98261ca58e84"),
    "mmlu": ("cais/mmlu", "all", "c30699e8356da336a370243923dbaf21066bb9fe"),
}

MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

# Per-task specs: one eval ≈ 60 thinking generations ≤ MAX_NEW_TOKENS in one batch + 200 MMLU forwards
# (GPU recommended; a 32-core CPU node works but is slow). Sizes can be shrunk for plumbing tests with WB_TR02_SMOKE=1 at *build* time
# (they are frozen into private/spec.json, so jobs never read env vars for sizes).
# MAX_NEW_TOKENS = 4096: shorter limits truncate most of the base model's thinking traces, including non-looping
# traces close to their final answer, which leaves no clean generations for the loop-direction contrast.
MAX_NEW_TOKENS = 4096
TASK_SPECS: dict[str, dict] = {
    "TR02-LOOP-001": {"eval": "gsm8k_test", "eval_n": 60, "calib": "gsm8k_train", "calib_n": 32,
                      "max_new_tokens": MAX_NEW_TOKENS, "constraint": "l0_or_rank4"},
    "TR02-LOOP-002": {"eval": "math_l123", "eval_n": 60, "calib": "math_l123_rest", "calib_n": 32,
                      "max_new_tokens": MAX_NEW_TOKENS, "constraint": "l0_or_rank4"},
    "TR02-LOOP-003": {"eval": "ifeval", "eval_n": 60, "calib": "ifeval_rest", "calib_n": 32,
                      "max_new_tokens": MAX_NEW_TOKENS, "constraint": "l0_or_rank4"},
    "TR02-LOOP-004": {"eval": "math_l45", "eval_n": 60, "calib": "gsm8k_train", "calib_n": 32,
                      "max_new_tokens": MAX_NEW_TOKENS, "constraint": "single_rank1"},
}
MMLU_EVAL_N = 200   # collateral T_gen (log-likelihood MC, no thinking)
MMLU_DEV_N = 60     # disjoint split used only by the reference search
REF_DEV_N = 64      # calibration-only dev problems for the reference: 24 selection + 40 holdout (reference.py)

SMOKE = {"eval_n": 3, "calib_n": 4, "max_new_tokens": 48, "mmlu_eval_n": 8, "mmlu_dev_n": 4, "ref_dev_n": 4}


def smoke_mode() -> bool:
    return os.environ.get("WB_TR02_SMOKE", "") not in ("", "0")


def editable(name: str) -> bool:
    """Parameters the agent may edit: decoder layers + final norm of the language model.
    (Token embedding / tied LM head, vision tower and MTP head are frozen.)"""
    return name.startswith(FILE_PREFIX + "layers.") or name == FILE_PREFIX + "norm.weight"


def to_model_name(file_name: str) -> str:
    assert file_name.startswith(FILE_PREFIX), file_name
    return MODEL_PREFIX + file_name[len(FILE_PREFIX):]


# ---- difficulty knob (SPEC §5): num_entangled_concepts, K = 2^L ----------------------------------------------------
# A concept is one problem family whose thinking-mode loops the edit must cure. The four families are the four
# existing eval pools (data._pool); all are run on the SAME subject (Qwen3.5-0.8B) and graded by the existing
# graders, so a level needs no new data source and no training. K = 1, 2, 4 -> levels 0..2; L3 (K = 8) is rejected.
LOOP_CONCEPTS = ("gsm8k_test", "math_l123", "ifeval", "math_l45")
LOOP_KNOB_LEVELS = (0, 1, 2)
CONCEPT_TASK = {"gsm8k_test": "TR02-LOOP-001", "math_l123": "TR02-LOOP-002", "ifeval": "TR02-LOOP-003",
                "math_l45": "TR02-LOOP-004"}   # task whose eval text / metric describes the family


def select_loop_concepts(task_id: str, seed: int, level: int) -> list[str]:
    """K = 2^level families: the task's own eval family first, then K-1 others in a seeded (nested) order."""
    import numpy as np

    if isinstance(level, bool) or not isinstance(level, int) or level not in LOOP_KNOB_LEVELS:
        raise ValueError(f"{task_id}: supported knob levels are {LOOP_KNOB_LEVELS}, got {level!r}")
    home = TASK_SPECS[task_id]["eval"]
    others = [c for c in LOOP_CONCEPTS if c != home]
    perm = np.random.default_rng([int(seed), int(task_id[-3:]), 52]).permutation(len(others))
    return [home, *[others[i] for i in perm[:2 ** level - 1]]]
