"""Pinned subject/data and per-task target sets for TR05-SPARSE-001..004 (pure Python)."""
from __future__ import annotations

import re

from weightbench import knobs
from weightbench.tracks.tr02_loops_lib import config as QWEN

MODEL_ID, MODEL_REV = QWEN.MODEL_ID, QWEN.MODEL_REV            # Qwen/Qwen3.5-0.8B, pinned (shared with TR02-LOOP)
MODEL_FILES, WEIGHTS_FILE = QWEN.MODEL_FILES, QWEN.WEIGHTS_FILE
PREFIX = QWEN.FILE_PREFIX                                        # "model.language_model."
N_LAYERS = QWEN.N_LAYERS

DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3")
SEQ_LEN = 512
EVAL_SEQS = 48             # private, scoring
DEV_SEQS = 16              # private, disjoint; reserved for reference selection
SYNTH_SAMPLES = 4096       # synthetic Gaussian inputs used by the data-free reference
CALIB_SEQS = 64            # private train-split chunks for the calibrated (privileged) references
CALIB_CHARS = 2_000_000    # train-split prefix tokenised to draw them from (~1k chunks)


def mlp(layer: int, proj: str) -> str:
    return f"{PREFIX}layers.{layer}.mlp.{proj}.weight"


MLP_PROJS = ("gate_proj", "up_proj", "down_proj")
ATTN_INPUT_PATTERN = re.compile(rf"^{re.escape(PREFIX)}layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj)|"
                                rf"linear_attn\.(in_proj_qkv|in_proj_z))\.weight$")

TASKS: dict[str, dict] = {
    # One full late MLP layer: pruning a single matrix costs too little for any data-free method to show a gain, and
    # one layer keeps the ramp below SPARSE-002 with real headroom.
    "TR05-SPARSE-001": {"title": "all MLP projections of one late layer (layer 23)", "depth": 1,
                        "targets": [mlp(23, p) for p in MLP_PROJS]},
    "TR05-SPARSE-002": {"title": "all MLP projections of layers 20-23", "depth": 2,
                        "targets": [mlp(l, p) for l in range(20, 24) for p in MLP_PROJS]},
    "TR05-SPARSE-003": {"title": "every MLP projection (72 matrices)", "depth": 3,
                        "targets": [mlp(l, p) for l in range(N_LAYERS) for p in MLP_PROJS]},
    "TR05-SPARSE-004": {"title": "every attention input projection", "depth": 3, "targets": "attn_inputs"},
}


# Matrices that write into the residual stream (used for the weight-derived residual covariance).
RESIDUAL_WRITER_PATTERN = re.compile(rf"^{re.escape(PREFIX)}layers\.(\d+)\."
                                     rf"(self_attn\.o_proj|linear_attn\.out_proj|mlp\.down_proj)\.weight$")
EMBED = f"{PREFIX}embed_tokens.weight"


def layer_of(name: str) -> int | None:
    m = re.match(rf"^{re.escape(PREFIX)}layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def residual_site(name: str) -> tuple[int, int]:
    """Where target ``name`` reads the residual stream, as an ordering key ``(layer, 0 = attention | 1 = MLP)``."""
    return layer_of(name), int(".mlp." in name)


def writer_key(name: str) -> tuple[int, int] | None:
    """Ordering key of a residual writer: attention output of layer l = (l, 0), MLP down_proj of layer l = (l, 1).
    A reader at site k sees exactly the writers with key < k."""
    m = RESIDUAL_WRITER_PATTERN.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) == "mlp.down_proj")


# The TR-05 sparsity ladder (SPEC §5): a task that is wired to ``propagation_depth`` prunes the MLPs of the last
# min(N_LAYERS, base · 3^L) layers at knob level L, where ``base`` is the task's default layer count (LADDER_BASES),
# so L0 is the task's default target set by construction. Only tasks whose default is a suffix of the layer stack
# (the last ``base`` layers) can grow this way; the full-coverage tasks have no room and are not wired.
LADDER_BASES: dict[str, int] = {"TR05-SPARSE-001": 1, "TR05-SPARSE-002": 4}
LADDER_TASK = "TR05-SPARSE-001"
# SPARSE-003 (every MLP) is the saturated rung of that same suffix ladder (SPARSE-001 level 3). It is wired in the
# OTHER direction instead (PREFIX_LADDER_BASES below), so its levels never duplicate a SPARSE-001 build.

# SPARSE-004 (attention input projections of every layer) gets bounded explicit levels instead: L0..L3 select the
# attention inputs of the last min(N_LAYERS, 3^L) layers. Its unversioned default (all layers) equals the saturated
# rung L3 (3^3 = 27 >= 24), so, unlike the MLP ladder, its explicit L0 is NOT the default build.
ATTN_LADDER_BASES: dict[str, int] = {"TR05-SPARSE-004": 1}
ATTN_LEVELS: tuple[int, ...] = (0, 1, 2, 3)

# SPARSE-003 gets bounded explicit levels counted from the INPUT side: L0..L2 prune the MLPs of the FIRST 3^L layers
# (L0 = layer 0, L1 = 0-2, L2 = 0-8), so a pruning error propagates through the 24 - 3^L unpruned layers above it
# (SPARSE-001 prunes the LAST 3^L layers, where errors reach the logits almost directly). L3 (all 24) would equal both
# the unversioned default and SPARSE-001 L3, so it is not offered; the default (every MLP) is kept byte-identical.
PREFIX_LADDER_BASES: dict[str, int] = {"TR05-SPARSE-003": 1}
PREFIX_LEVELS: tuple[int, ...] = (0, 1, 2)


def ladder_layers(level: int, task_id: str = LADDER_TASK) -> list[int]:
    """Layers ``task_id`` prunes at knob level ``level`` (MLPs, or attention inputs for SPARSE-004): the last
    min(N_LAYERS, base · 3^L)."""
    bases = {**LADDER_BASES, **ATTN_LADDER_BASES, **PREFIX_LADDER_BASES}
    if task_id not in bases:
        raise ValueError(f"{task_id}: not on the TR-05 sparsity ladder")
    if task_id in PREFIX_LADDER_BASES and level not in PREFIX_LEVELS:
        raise ValueError(f"{task_id}: supported knob levels are {PREFIX_LEVELS}, got {level}")
    n = min(N_LAYERS, bases[task_id] * int(knobs.value(knobs.get("TR05"), level)))
    if task_id in PREFIX_LADDER_BASES:                    # counted from the input side: the first n layers
        return list(range(n))
    return list(range(N_LAYERS - n, N_LAYERS))


def ladder_targets(level: int, task_id: str = LADDER_TASK) -> list[str]:
    """MLP target names of ``task_id`` at knob level ``level`` (layer-major, gate/up/down within a layer)."""
    return [mlp(layer, proj) for layer in ladder_layers(level, task_id) for proj in MLP_PROJS]


def task_targets(task_id: str, level: int | None = None) -> list[str] | str:
    """The configured target set of a task; ``level`` (a ladder task only) selects a rung of its ladder."""
    if level is None:
        return TASKS[task_id]["targets"]
    if task_id in ATTN_LADDER_BASES:
        ladder_layers(level, task_id)                     # validates the level; names come from the checkpoint
        return "attn_inputs"
    if task_id not in LADDER_BASES and task_id not in PREFIX_LADDER_BASES:
        raise ValueError(f"{task_id}: its knob is not wired to a level (ladder tasks: "
                         f"{sorted({**LADDER_BASES, **ATTN_LADDER_BASES, **PREFIX_LADDER_BASES})})")
    return ladder_targets(level, task_id)


def resolve_targets(task_id: str, names, level: int | None = None) -> list[str]:
    """Target parameter names for a task (at knob ``level``, if given), given the checkpoint's tensor names."""
    targets = task_targets(task_id, level)
    if targets == "attn_inputs":
        keep = set(ladder_layers(level, task_id)) if level is not None else None
        return sorted(n for n in names if ATTN_INPUT_PATTERN.match(n) and (keep is None or layer_of(n) in keep))
    missing = [t for t in targets if t not in set(names)]
    if missing:
        raise KeyError(f"targets not in checkpoint: {missing[:3]}")
    return list(targets)


def norm_before(name: str, names) -> str | None:
    """The RMSNorm whose output feeds ``name`` (MLP: post-attention norm; attention: input norm)."""
    m = re.match(rf"^({re.escape(PREFIX)}layers\.\d+\.)", name)
    if not m:
        return None
    candidate = m.group(1) + ("post_attention_layernorm.weight" if ".mlp." in name else "input_layernorm.weight")
    return candidate if candidate in set(names) else None
