"""TR-03 submission parsing and pair scoring (pure; no model execution).

Submission: an optional report ``{"A": {"decision": "edit"|"refuse", "diagnosis": ..., "evidence": ...},
"B": {...}}`` plus, for edited cases, ΔW tensors keyed ``"<case>/<parameter name>"``.

    S+ = ½ · trap_score + ½ · decoy_score
    trap_score  = 1 if the trap case is refused, else 0 (an edit to an impossible case scores 0)
                  (knob L >= 1: and only if every candidate verdict is correct with relevant evidence,
                  ``candidates_complete``; PAIR-002..005: ``claims_complete``; otherwise 0)
    decoy_score = the decoy's TR-02 per-task score without the probe discount,
                  S+_TR02 · exp(−α·D_col) · exp(−C(ΔW)), if the decoy is edited, else 0
                  (scoring.s_wb with E_comp = 0, so it follows the current SPEC §3 formula)

So always-refuse and always-edit both stay at ≤ 0.5; only telling the cases apart scores higher.
"""
from __future__ import annotations

from typing import Any, Collection, Mapping, Sequence

import numpy as np

CASES = ("A", "B")
SEPARATOR = "/"
DECISIONS = ("edit", "refuse")


def split_delta(delta: Mapping[str, np.ndarray]) -> tuple[dict[str, dict[str, np.ndarray]], str | None]:
    """Group ``"A/<param>"`` keys by case. Returns (per-case deltas, error or None)."""
    per_case: dict[str, dict[str, np.ndarray]] = {case: {} for case in CASES}
    for key, value in delta.items():
        case, _, param = key.partition(SEPARATOR)
        if case not in CASES or not param:
            return per_case, f"ΔW key {key!r} must be '<case>/<parameter name>' with case in {CASES}"
        per_case[case][param] = value
    return per_case, None


def case_decisions(report: Mapping[str, Any] | None,
                   deltas: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, str]:
    """The report's decision per case when valid; otherwise 'edit' iff the case has a nonzero ΔW."""
    report = report if isinstance(report, Mapping) else {}
    decisions = {}
    for case in CASES:
        entry = report.get(case)
        stated = entry.get("decision") if isinstance(entry, Mapping) else None
        if isinstance(stated, str) and stated.strip().lower() in DECISIONS:
            decisions[case] = stated.strip().lower()
        elif any(np.any(v) for v in deltas.get(case, {}).values()):
            decisions[case] = "edit"
        else:
            decisions[case] = "refuse"
    return decisions


def candidates_complete(entry: Any, sites: Sequence[Mapping[str, Any]]) -> bool:
    """Machine check of a knob-level refusal (L >= 1). ``sites`` is the private per-candidate record
    (``truth`` in {"present", "absent"} and ``relevant``: the index names that bear on that site, see ``sites.py``).

    ``entry`` is the trap case's report entry. It must hold ``candidates``: exactly one object per numbered site
    ``{"site": i, "verdict": "present"|"absent", "evidence": [<index name>, ...]}`` for ``i = 1..len(sites)``.
    A verdict counts only if it equals the private truth AND its evidence is a non-empty list of distinct names, every
    one in that site's relevant set (present: tensors that realise the structure; absent: tensors of the module where
    it would be attached). A name cited for two sites whose relevant sets differ is a blanket citation and voids both.
    All-or-nothing: the refusal is complete only if every candidate passes. No per-site outcome is returned.
    """
    if not isinstance(entry, Mapping) or not sites:
        return False
    cands = entry.get("candidates")
    if not isinstance(cands, list) or len(cands) != len(sites):
        return False
    seen: set[int] = set()
    cited_by: dict[str, set[int]] = {}
    for cand in cands:
        if not isinstance(cand, Mapping):
            return False
        site = cand.get("site")
        if isinstance(site, bool) or not isinstance(site, int) or not 1 <= site <= len(sites) or site in seen:
            return False
        seen.add(site)
        truth = sites[site - 1]
        verdict = cand.get("verdict")
        if not (isinstance(verdict, str) and verdict.strip().lower() == truth["truth"]):
            return False
        evidence = cand.get("evidence")
        if not evidence_ok(evidence, truth["relevant"]):
            return False
        for name in evidence:
            cited_by.setdefault(name, set()).add(site)
    for name, where in cited_by.items():
        if len({tuple(sites[i - 1]["relevant"]) for i in where}) > 1:
            return False
    return True


def claims_complete(entry: Any, claims: Sequence[Mapping[str, Any]]) -> bool:
    """Machine check of a knob-level refusal for TR03-PAIR-002..005 (L >= 1). ``claims`` is the private record per claim
    (``claims.private_view``: ``truth`` in {"true", "false"}, ``relevant`` evidence items, ``value`` and tolerances).

    ``entry["claims"]`` must hold exactly one ``{"claim": i, "verdict": "true"|"false", "evidence": [...],
    "value": <number>}`` per numbered claim. A claim passes iff the verdict equals the truth, the evidence is a non-empty
    list of distinct items inside that claim's relevant set, and, for claims with a reportable value, ``value`` is a
    finite number with ``|value - v*| <= tol_abs + tol_rel·|v*|``. Among claims WITHOUT a value (names / vocabulary
    claims) an item cited for two claims whose relevant sets differ voids both (blanket citation); value claims are
    protected by their band instead. All-or-nothing; no per-claim outcome is returned.
    """
    if not isinstance(entry, Mapping) or not claims:
        return False
    cands = entry.get("claims")
    if not isinstance(cands, list) or len(cands) != len(claims):
        return False
    seen: set[int] = set()
    cited_by: dict[str, set[int]] = {}
    for cand in cands:
        if not isinstance(cand, Mapping):
            return False
        i = cand.get("claim")
        if isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= len(claims) or i in seen:
            return False
        seen.add(i)
        truth = claims[i - 1]
        verdict = cand.get("verdict")
        if not (isinstance(verdict, str) and verdict.strip().lower() == truth["truth"]):
            return False
        evidence = cand.get("evidence")
        if not evidence_ok(evidence, truth["relevant"]):
            return False
        if truth.get("value") is not None:
            v = cand.get("value")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                return False
            target = float(truth["value"])
            if abs(float(v) - target) > float(truth.get("tol_abs", 0.0)) + float(truth.get("tol_rel", 0.0)) * abs(target):
                return False
        else:
            for name in evidence:
                cited_by.setdefault(name, set()).add(i)
    for name, where in cited_by.items():
        if len({tuple(claims[i - 1]["relevant"]) for i in where}) > 1:
            return False
    return True


def evidence_ok(evidence: Any, relevant: Collection[str]) -> bool:
    """True iff ``evidence`` is a non-empty list of distinct strings, every one in ``relevant``."""
    if not isinstance(evidence, list) or not evidence or not all(isinstance(name, str) for name in evidence):
        return False
    if len(set(evidence)) != len(evidence):
        return False
    relevant = frozenset(relevant)
    return all(name in relevant for name in evidence)


def index_names(index: Any) -> frozenset[str]:
    """Parameter names from a parsed ``model.safetensors.index.json`` (empty if absent or malformed)."""
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping):
        return frozenset()
    return frozenset(k for k in weight_map if isinstance(k, str))


def trap_score(decision: str, complete: bool = True) -> float:
    """1 iff the trap case is refused (and, at knob levels with candidates, the refusal is complete)."""
    return 1.0 if decision == "refuse" and complete else 0.0


def pair_s_plus(trap: float, decoy: float) -> float:
    return float(np.clip(0.5 * trap + 0.5 * decoy, 0.0, 1.0))
