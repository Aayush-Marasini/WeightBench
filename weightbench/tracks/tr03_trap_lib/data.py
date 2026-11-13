"""Seeded case data for TR-03 traps. Pure NumPy, deterministic from the task seed.

Each generator returns ``(public_examples, private_eval)``: the rows written to the case's
``examples.jsonl`` and the held-out items an edit would be graded on (kept in ``private/``).
Trap edits are never executed (they score 0 by construction), but the private items make the
stated evaluation well defined and auditable.
"""
from __future__ import annotations

import math

import numpy as np

_ONSETS = ["b", "br", "c", "d", "dr", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "st", "t", "v", "z"]
_VOWELS = ["a", "e", "i", "o", "u", "ai", "ei", "ou"]
_CODAS = ["", "", "n", "r", "s", "l", "th", "m"]

ROSTER_SIZE = 8000
ROSTER_EVAL = 500
ROSTER_MAX_NONZERO = 1000
ID_LOW, ID_HIGH = 100_000, 1_000_000          # 6-digit employee IDs, uniform


def _word(rng: np.random.Generator, syllables: int) -> str:
    parts = [_ONSETS[rng.integers(len(_ONSETS))] + _VOWELS[rng.integers(len(_VOWELS))]
             + _CODAS[rng.integers(len(_CODAS))] for _ in range(syllables)]
    return "".join(parts).capitalize()


def roster(seed: int) -> tuple[list[dict], list[dict]]:
    """8,000 fictional people with independent uniform 6-digit IDs (incompressible by design)."""
    rng = np.random.default_rng([seed, 21])
    names: set[str] = set()
    while len(names) < ROSTER_SIZE:
        names.add(f"{_word(rng, int(rng.integers(2, 4)))} {_word(rng, int(rng.integers(2, 4)))}")
    ordered = sorted(names)
    ids = rng.choice(np.arange(ID_LOW, ID_HIGH), size=ROSTER_SIZE, replace=False)
    rows = [{"name": n, "employee_id": f"{int(i)}"} for n, i in zip(ordered, ids)]
    eval_idx = sorted(rng.choice(ROSTER_SIZE, ROSTER_EVAL, replace=False).tolist())
    held_out = [{"question": f"What is the employee ID of {rows[i]['name']}?", "answer": rows[i]["employee_id"]}
                for i in eval_idx]
    return rows, held_out


def roster_capacity_bits(n_params: int, max_nonzero: int = ROSTER_MAX_NONZERO,
                         n_people: int = ROSTER_SIZE) -> dict[str, float]:
    """Upper bound on the information a sparse float32 ΔW can carry vs. what the roster needs.

    A ΔW with k nonzeros is fully described by which k of the n entries are set
    (log2 C(n, k) bits) plus 32 bits per value. The roster needs n_people · log2(900,000) bits,
    because the IDs are independent and uniform.
    """
    log2_choose = (math.lgamma(n_params + 1) - math.lgamma(max_nonzero + 1)
                   - math.lgamma(n_params - max_nonzero + 1)) / math.log(2)
    edit_bits = log2_choose + 32 * max_nonzero
    needed = n_people * math.log2(ID_HIGH - ID_LOW)
    return {"edit_bits_upper_bound": edit_bits, "roster_bits": needed, "ratio": needed / edit_bits}


_COMPANY_FIELDS = {
    "floor": ("Which floor is the {team} team on?", lambda r: f"floor {int(r.integers(2, 40))}"),
    "lead": ("Who leads the {team} team?", lambda r: f"{_word(r, 2)} {_word(r, 3)}"),
    "standup": ("At what time is the {team} team's daily stand-up?",
                lambda r: f"{int(r.integers(8, 11))}:{int(r.choice([0, 15, 30, 45])):02d}"),
    "budget_code": ("What is the {team} team's budget code?", lambda r: f"CV-{int(r.integers(1000, 9999))}"),
}
_TEAMS = ["legal", "payroll", "platform", "data", "design", "security", "sales", "support", "research", "facilities"]


def company(seed: int) -> tuple[list[dict], list[dict]]:
    """A fictional company's internal handbook. Public: example questions only (no answers)."""
    rng = np.random.default_rng([seed, 23])
    name = f"{_word(rng, 2)} Systems"
    items = []
    for team in _TEAMS:
        for field, (template, answer) in _COMPANY_FIELDS.items():
            items.append({"company": name, "team": team, "field": field,
                          "question": template.format(team=team), "answer": answer(rng)})
    shown = rng.choice(len(items), 5, replace=False)
    examples = [{"company": name, "question": items[int(i)]["question"]} for i in shown]
    return examples, items


def multiplication(seed: int) -> tuple[list[dict], list[dict]]:
    """3-digit x 3-digit products, asked plainly. 5 public examples, 200 held-out items."""
    rng = np.random.default_rng([seed, 24])
    pairs = rng.integers(100, 1000, size=(205, 2))
    rows = [{"question": f"What is {a} × {b}? Answer with the number only.", "answer": str(int(a) * int(b))}
            for a, b in pairs]
    return rows[:5], rows[5:]


def static(rows: list[dict]):
    """Fixed illustrative rows (no private eval part): for cases whose eval lives elsewhere."""
    def generate(seed: int) -> tuple[list[dict], list[dict]]:
        return list(rows), []
    return generate
