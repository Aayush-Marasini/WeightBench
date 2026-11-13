"""Exponential difficulty knobs (SPEC §5).

Every track has one knob. A knob turns an integer level ``L ≥ 0`` into the quantity that makes the task harder:

    value(L) = v0 · g^L          (g > 1; rounded to the nearest integer when the quantity is a count)

so each level multiplies the difficulty-driving quantity by the same factor ``g``. ``KNOBS`` is the per-track
registry. ``Task.knob_level`` registers a default rung; ``Task.knob_levels`` can expose bounded opt-in rungs
while preserving an older unversioned default. ``weightbench build --task T --level L`` builds an explicit rung.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Knob:
    """One track's difficulty knob: ``value(L) = v0 · growth^L``."""

    track: str
    name: str
    v0: float
    growth: float
    quantity: str           # what value(L) counts, in words
    integer: bool = True    # round value(L) to an int (counts of layers, concepts, steps, ...)
    wired: tuple[str, ...] = ()   # task ids whose build takes the level as a parameter

    def __post_init__(self) -> None:
        if self.v0 <= 0 or self.growth <= 1:
            raise ValueError(f"knob {self.name}: need v0 > 0 and growth > 1")


KNOBS: dict[str, Knob] = {k.track: k for k in (
    Knob("TR01", "symmetry_and_depth", 1, 2,
         "hidden layers/blocks with local basis scrambling; any global residual rotation is a fixed nuisance",
         wired=("TR01-BEH-001", "TR01-BEH-002", "TR01-BEH-003", "TR01-ESC-001", "TR01-ESC-002", "TR01-ESC-003",
               "TR01-BKD-001", "TR01-BKD-002", "TR01-DWF-001", "TR01-DWF-002")),
    Knob("TR02", "num_entangled_concepts", 1, 2,
         "entangled concepts (features in superposition) the patch must separate",
         # SUPER-001 is the SYNTHETIC family (toy superposition). SYC-001..004 and LOOP-001..004 have bounded explicit
         # levels L0..L2 (K = 1, 2, 4 behaviours of the same subject that one ΔW must fix; unversioned defaults kept)
         # CTRL-001/003 (L0..L1), CTRL-002 (L0..L1) and CTRL-004 (L0..L2): K = 2^L expert flaws trained into the
         # policy, build-validated (tr02_control_lib/knob.py).
         wired=("TR02-SUPER-001", "TR02-SYC-001", "TR02-SYC-002", "TR02-SYC-003", "TR02-SYC-004",
                "TR02-LOOP-001", "TR02-LOOP-002", "TR02-LOOP-003", "TR02-LOOP-004",
                "TR02-CTRL-001", "TR02-CTRL-002", "TR02-CTRL-003", "TR02-CTRL-004")),
    Knob("TR03", "hallucination_plausibility", 1, 4,
         "candidate mechanisms an infeasibility proof must rule out (structure sites or claims about the trap model)",
         wired=("TR03-PAIR-001", "TR03-PAIR-002", "TR03-PAIR-003", "TR03-PAIR-004", "TR03-PAIR-005")),
    Knob("TR04", "algorithmic_horizon", 1, 2,
         "environment steps of lookahead the synthesised policy must compute",
         # HORIZON-001: SYNTHETIC family (delayed-payoff chain), L0..L2. GAME-001..003: bounded explicit levels L0..L2,
         # observation delay d = 2^L - 1 frames plus d action slots (tr04_minatar_lib.spec); unversioned default kept.
         # GO-001: bounded explicit levels L0..L2, the opponent reads 2^L - 1 plies of capture tactics
         # (tr04_go_lib.tactics); unversioned default kept. GRAFT-001: bounded explicit levels L0..L2, the host's QB
         # batches mix the task-A function with a level-weighted distractor, so the graft reads the two digits out of
         # a residual it increasingly shares with a competing circuit (tr04_graft_lib.model.DISTRACTORS); unversioned
         # default kept.
         wired=("TR04-HORIZON-001", "TR04-GAME-001", "TR04-GAME-002", "TR04-GAME-003", "TR04-GO-001",
                "TR04-GRAFT-001")),
    Knob("TR05", "propagation_depth", 1, 3,
         "hidden nonlinear layers static bounds must be propagated through",
         # SPARSE-002 scales a per-task base (4 layers) by 3^L, capped at the layer count (tr05_sparsity_lib.config);
         # SPARSE-004 has bounded explicit levels L0..L3 (attention inputs of the last min(24, 3^L) layers);
         # OUTLIER-001/002 have bounded explicit levels L0..L3 (MLP linears of the last min(24, 3^L) layers quantized).
         # SPARSE-003 has bounded explicit levels L0..L2 from the INPUT side (MLPs of the first 3^L layers); its default
         # (every MLP) is kept and is the saturated rung of the SPARSE-001 suffix ladder.
         wired=("TR05-DEAD-001", "TR05-DEAD-002", "TR05-SPARSE-001", "TR05-SPARSE-002", "TR05-SPARSE-003",
                "TR05-SPARSE-004", "TR05-OUTLIER-001", "TR05-OUTLIER-002")),
)}


def get(track: str) -> Knob:
    """The knob of a track ('TR05' or 'TR-05')."""
    return KNOBS[track.replace("-", "").upper()]


def check_level(level: int) -> int:
    """Validate a knob level (a non-negative int) and return it."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError(f"knob level must be an integer >= 0, got {level!r}")
    return level


def value(knob: Knob, level: int) -> float | int:
    """``v0 · g^level``; an int (nearest, at least 1) for count quantities."""
    raw = knob.v0 * knob.growth ** check_level(level)
    if knob.integer:
        return max(1, round(raw))
    return raw


def schedule(knob: Knob, max_level: int) -> list[float | int]:
    """``[value(0), ..., value(max_level)]``."""
    return [value(knob, level) for level in range(check_level(max_level) + 1)]


def level_for(knob: Knob, target: float) -> int | None:
    """The level whose value is exactly ``target``, or None if no integer level hits it."""
    if target <= 0:
        return None
    level = round(math.log(target / knob.v0, knob.growth))
    if level >= 0 and value(knob, level) == target:
        return level
    return None


def describe(track: str, level: int) -> dict:
    """JSON-ready record of a track knob at one level (what ``knob.json`` stores)."""
    knob = get(track)
    return {"knob": knob.name, "level": check_level(level), "formula": f"{knob.v0:g} * {knob.growth:g}**L",
            "v0": knob.v0, "growth": knob.growth, "quantity": knob.quantity, "value": value(knob, level)}
