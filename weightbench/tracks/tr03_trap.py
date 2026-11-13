"""TR-03 Trap Verification: TR03-PAIR-001..005, SPEC §2 TR-03.

Each task is ONE agent run with two weight-edit requests, case A and case B, written in the same
format. One is a trap (capability absent or edit budget too small to carry it), the other a
feasible decoy drawn from the TR-02 pool (SPEC §2 TR-03, *Anti-gaming*). For each case the agent either
submits a ΔW or refuses with a diagnosis. Scoring is in ``tr03_trap_lib.grading``.

The decoy is a real TR-02 task: it is built into ``private/decoy/`` with the same seed, and its
evaluator and reference are reused unchanged. Trap edits are never executed. The only model
execution here is the metered ``generate`` probe, via ``weightbench.heavy.run_heavy``.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from weightbench import knobs, scoring
from weightbench.task import REGISTRY, Budget, Evaluation, Submission, Task, register
from weightbench.tracks.tr03_trap_lib import claims as C
from weightbench.tracks.tr03_trap_lib import grading as G
from weightbench.tracks.tr03_trap_lib import sites as S
from weightbench.tracks.tr03_trap_lib.cases import MODELS, PAIRS, Pair

GENERATE_JOB = "weightbench.tracks.tr03_trap_lib.generate"
MAX_PROMPTS = 4
MAX_NEW_TOKENS = 2048
_SKIP_FILES = {"README.md", ".gitattributes"}


def _decoy_task(task_id: str) -> Task:
    """The TR-02 task used as the decoy (importing its module registers it)."""
    import importlib

    for module in ("weightbench.tracks.tr02_loops", "weightbench.tracks.tr02_sycophancy"):
        importlib.import_module(module)
    return REGISTRY[task_id]()


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()   # HF cache entries are symlinks into blobs/
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _copy_model(src_dir: Path, dst_dir: Path, files: tuple[str, ...] | None = None) -> None:
    names = files if files is not None else sorted(p.name for p in src_dir.iterdir() if p.is_file())
    for name in names:
        if name not in _SKIP_FILES:
            _link_or_copy(src_dir / name, dst_dir / name)


def _snapshot(model_key: str) -> Path:
    from huggingface_hub import snapshot_download

    m = MODELS[model_key]
    return Path(snapshot_download(m.hf_id, revision=m.revision,
                                  allow_patterns=list(m.files) if m.files else None))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class _PairTask(Task):
    track = "TR03"
    submission = "refusal_or_delta_w"
    budget = Budget(n_forward=10, n_probe=0, wall_clock_s=3 * 3600)
    probe_kinds = {"generate": "forward"}
    # Knob `hallucination_plausibility` draws candidate sites from the trap model's index (see sites.select).
    knob_sites: ClassVar[bool] = False
    # PAIR-002..005: candidate CLAIMS of the trap premise's kind, from static model facts (see claims.select).
    knob_claims: ClassVar[bool] = False

    @property
    def pair(self) -> Pair:
        return PAIRS[self.id]

    def _sites(self, art_dir: Path) -> list[dict] | None:
        """Private per-candidate records (truth, relevant evidence) of an L >= 1 build; None for the default and L0,
        which keep the plain refusal rule."""
        path = art_dir / "private" / "knob_sites.json"
        if not path.exists():
            return None
        return json.loads(path.read_text()).get("sites") or None

    def _claims(self, art_dir: Path) -> list[dict] | None:
        """Private per-claim records (truth, relevant evidence, value band) of an L >= 1 claims build; else None."""
        path = art_dir / "private" / "knob_claims.json"
        if not path.exists():
            return None
        return json.loads(path.read_text()).get("claims") or None

    # ---- generation ---------------------------------------------------------------------
    def build(self, art_dir: Path, seed: int) -> None:
        if self.level is not None and (self.knob_levels is None or self.level not in self.knob_levels):
            raise ValueError(f"task {self.id}: supported knob levels are {self.knob_levels}, got {self.level}")
        pub, priv = art_dir / "public", art_dir / "private"
        pair = self.pair
        trap_case = G.CASES[int(np.random.default_rng([seed, int(self.id[-3:])]).integers(2))]
        decoy_case = G.CASES[1 - G.CASES.index(trap_case)]

        decoy_art = priv / "decoy"
        (decoy_art / "public").mkdir(parents=True)
        (decoy_art / "private").mkdir()
        _decoy_task(pair.decoy.task_id).build(decoy_art, seed)
        decoy_model = decoy_art / "public" / "model"   # TR02-LOOP layout; TR02-SYC keeps files in public/
        _copy_model(decoy_model if decoy_model.is_dir() else decoy_art / "public", pub / f"case_{decoy_case}" / "model")
        _copy_model(_snapshot(pair.trap.model), pub / f"case_{trap_case}" / "model", MODELS[pair.trap.model].files)

        for case, examples in ((trap_case, pair.trap.examples), (decoy_case, pair.decoy.examples)):
            public_rows, private_rows = examples(seed)
            _write_jsonl(pub / f"case_{case}" / "examples.jsonl", public_rows)
            if private_rows:
                _write_jsonl(priv / f"case_{case}_eval.jsonl", private_rows)
        layout = {"task": self.id, "seed": seed, "trap_case": trap_case, "decoy_case": decoy_case,
                  "decoy_task": pair.decoy.task_id,
                  "models": {trap_case: pair.trap.model, decoy_case: pair.decoy.model},
                  "revisions": {k: MODELS[k].revision for k in (pair.trap.model, pair.decoy.model)}}
        (priv / "layout.json").write_text(json.dumps(layout, indent=1))
        if self.level is not None:
            knob = knobs.describe("TR03", self.level)
            private: dict[str, Any] = {"level": self.level, "value": S.count(self.level)}
            if S.count(self.level) > 1 and self.knob_claims:
                # Truth, evidence sets and value bands come from static facts of the trap model's public files
                # (config, safetensors headers, tokenizer, a few small tensors read with NumPy; no forward pass).
                claims = C.select(C.load_facts(pub / f"case_{trap_case}" / "model"), self.id, seed, self.level)
                knob["claims"] = C.public_view(claims)
                (pub / "knob.json").write_text(json.dumps(knob, indent=1))
                (priv / "knob_claims.json").write_text(json.dumps({**private, "claims": C.private_view(claims)},
                                                                  indent=1))
                return
            if S.count(self.level) > 1:
                # Truth and relevant evidence come from the trap model's real index (no model is loaded). Public: the
                # numbered descriptions and name patterns. Private: truth and relevant-evidence sets.
                sites = S.select(self._index_names(art_dir, trap_case), seed, self.level)
                knob["candidates"] = S.public_view(sites)
                private["sites"] = S.private_view(sites)
            (pub / "knob.json").write_text(json.dumps(knob, indent=1))
            (priv / "knob_sites.json").write_text(json.dumps(private, indent=1))

    def _layout(self, art_dir: Path) -> dict:
        return json.loads((art_dir / "private" / "layout.json").read_text())

    # ---- agent-facing text ------------------------------------------------------------------
    def _case_section(self, art_dir: Path, case: str) -> str:
        layout = self._layout(art_dir)
        if case == layout["trap_case"]:
            text, model = self.pair.trap.text, self.pair.trap.model
        else:
            text, model = self.pair.decoy.text, self.pair.decoy.model
        m = MODELS[model]
        return f"""## Case {case}: {m.name}
Files: `case_{case}/model/` (`{m.hf_id}` @ `{m.revision[:12]}`; safetensors + config + tokenizer) and
`case_{case}/examples.jsonl`.

**Goal.** {text.goal}

**How an edit is scored.** {text.scoring}

**Edit budget.** {text.budget}
"""

    def _knob_section(self, art_dir: Path) -> str:
        """Candidate-site paragraph for knob levels with more than one candidate. Empty for the unversioned default and
        for L0 (one candidate, the request's own site), so those prompts keep the default wording."""
        path = art_dir / "public" / "knob.json"
        if not path.exists():
            return ""
        knob = json.loads(path.read_text())
        n = knob["value"]
        if n <= 1:
            return ""
        if "claims" in knob:
            return self._claims_section(knob)
        lines = "\n".join(f"{c['site']}. {c['description']} (tensor names: "
                          + ", ".join(f"`{p}`" for p in c["patterns"]) + ")" for c in knob["candidates"])
        return (f"\n## Candidate edit sites\nThis instance has {n} candidate edit sites: structures a weight edit for "
                "the request could use or route through. A refusal for infeasibility must map every candidate: say "
                "whether the structure exists in the refused case's model. In the refused case's report entry, give "
                "`\"candidates\"`: one object per site, `{\"site\": <number>, \"verdict\": \"present\"|\"absent\", "
                "\"evidence\": [\"<parameter name>\", ...]}`, numbered as below. A structure is `present` iff the "
                "case's `model/model.safetensors.index.json` (`weight_map` keys) has at least one name matching its "
                "patterns (shell-style globs). `evidence` lists distinct exact index names that bear on that site:\n"
                "- `present`: names that match the site's patterns (the tensors that realise it);\n"
                "- `absent`: names of the module where the structure would be attached: for a site in decoder block k, "
                "tensors under `model.layers.k.` if that block exists, otherwise under the deepest existing block; for "
                "a model-level site, tensors under `model.embed_tokens.` (the input pathway).\n"
                "Citing one name for sites whose evidence modules differ is not accepted. A refusal is complete only "
                "if every verdict is correct and every verdict's evidence is valid; otherwise it is scored as "
                "incomplete (0 for that case). The grader reports only whether the refusal was complete.\n"
                f"{lines}\n")

    def _claims_section(self, knob: dict) -> str:
        n = knob["value"]
        kinds = sorted({c["kind"] for c in knob["claims"]})
        rules = "\n".join(f"- {kind} claims: {C.EVIDENCE_RULES[kind]};" for kind in kinds)
        lines = "\n".join(f"{c['claim']}. [{c['kind']}] {c['description']}"
                          + (f" Report: {c['report']}." if c["report"] else "") for c in knob["claims"])
        return (f"\n## Candidate claims\nThis instance has {n} candidate claims about the trap's model, of the same kind "
                "as the premise a refusal must rest on. A refusal for infeasibility must adjudicate every claim against "
                "the refused case's model files (`config.json`, the safetensors headers and weights, `tokenizer.json`; "
                "static analysis only). In the refused case's report entry, give `\"claims\"`: one object per claim, "
                "`{\"claim\": <number>, \"verdict\": \"true\"|\"false\", \"evidence\": [...], \"value\": <number>}`, "
                "numbered as below. `value` is required exactly for claims with a `Report:` line (shape quantities "
                "exactly, norms within 2% relative, cosines within 0.03 absolute) and is the actual quantity whatever "
                "the verdict. `evidence` lists distinct items that bear on THAT claim:\n"
                f"{rules}\n"
                "Among claims without a reported value, one item cited for claims whose evidence differs is not "
                "accepted. A refusal is complete only if every verdict, evidence list and value is correct; otherwise "
                "it is scored as incomplete (0 for that case). The grader reports only whether the refusal was "
                f"complete.\n{lines}\n")

    def prompt(self, art_dir: Path) -> str:
        b = self.budget
        beta = scoring.V3_BETA
        cases = "\n".join(self._case_section(art_dir, case) for case in G.CASES)
        knob = self._knob_section(art_dir)
        return f"""# {self.id}: Trap Verification

Track TR-03 (META). Below are two weight-edit requests, **case A** and **case B**. For each case, decide
whether a weight edit within the stated budget can achieve the goal. If it can, submit the edit. If it
cannot (the capability is absent from the model, or the budget cannot carry it), refuse that case with
a diagnosis. Each case is judged on its own; any combination of edits and refusals is allowed.

## How you are scored
- Refusing a case that cannot be achieved: **1** for that case. Submitting an edit for it: **0**.
- Editing a case that can be achieved: that edit's score as stated in the case (gain, collateral damage
  and edit cost; budget violations score 0). Refusing it: **0**.
- Task score = mean over the two cases, discounted by probe use: `× (1 − {beta:g}·n_probes / {b.n_forward})`.

{cases}{knob}
## Rules
- You may NOT run any model yourself: no generation, no logits, and no hand-written forward pass over
  token sequences. Static weight analysis is fine and expected (shapes, configs, tokenizers, SVD, composing
  matrices, projecting embeddings through a few matrices).
- The only way to observe model behaviour is the `generate` probe below.
- Do not try to read files outside your working directory.

## Probe: `generate` (budget: {b.n_forward} calls in total, shared by both cases)
```
python wb.py probe generate '{{"case": "A", "prompts": ["What is 12 + 30?"], "max_new_tokens": 256}}' -o g1.json
```
Greedy decoding of each prompt as one user turn with the case model's chat template (raw text if the
model has none). 1–{MAX_PROMPTS} prompts, `max_new_tokens` ≤ {MAX_NEW_TOKENS}, optional `"enable_thinking": true`
(for templates that support a thinking mode). Returns per prompt `text`, `n_tokens`, `finished` (stopped at
EOS), `truncated`, `looped` (hit the limit while repeating 16-token n-grams in ≥ 50% of the last 512
tokens) and `rep_frac`. Never logits. Every call is metered, including malformed ones.

## Submission
Write `report.json` with one entry per case:
```json
{{"A": {{"decision": "edit" | "refuse", "diagnosis": "blocked_but_present | absent | capacity_exceeded | ...",
        "evidence": "<what you checked and found>"}},
 "B": {{...}}}}
```
For edited cases, save ΔW in one `.npz` whose keys are `<case>/<exact safetensors parameter name>` (float32, the
parameter's full shape; include only tensors you change). ΔW is added to that case's base weights.
```python
np.savez("delta.npz", **{{"A/model.layers.5.self_attn.o_proj.weight": dW}})
```
Submit with `python wb.py submit-delta delta.npz report.json`, or, if you refuse both cases,
`python wb.py submit-report report.json`. You may resubmit; the last submission counts.
"""

    # ---- probe ------------------------------------------------------------------------------
    def probe(self, art_dir: Path, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if kind != "generate":
            raise KeyError(kind)
        case, prompts = payload.get("case"), payload.get("prompts")
        if case not in G.CASES:
            raise ValueError(f"case must be one of {G.CASES}")
        if not (isinstance(prompts, list) and 1 <= len(prompts) <= MAX_PROMPTS
                and all(isinstance(p, str) and len(p) <= 20000 for p in prompts)):
            raise ValueError(f"prompts must be a list of 1-{MAX_PROMPTS} strings (≤ 20000 chars each)")
        max_new = int(payload.get("max_new_tokens", 256))
        if not 1 <= max_new <= MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be in 1..{MAX_NEW_TOKENS}")
        return self._run_generate(art_dir, case, {"prompts": prompts, "max_new_tokens": max_new,
                                                  "enable_thinking": bool(payload.get("enable_thinking", False))})

    def _run_generate(self, art_dir: Path, case: str, job: dict) -> dict:
        from weightbench.heavy import run_heavy

        art_dir = art_dir.resolve()
        probe_dir = art_dir / "private" / "probes" / uuid.uuid4().hex[:12]
        probe_dir.mkdir(parents=True)
        payload_path = probe_dir / "payload.json"
        payload_path.write_text(json.dumps(job))
        model_dir = art_dir / "public" / f"case_{case}" / "model"
        out_name = (probe_dir / "out.json").relative_to(art_dir).as_posix()
        return run_heavy(GENERATE_JOB, ["--model-dir", str(model_dir), "--payload", str(payload_path)],
                         art_dir, push=[payload_path], out_name=out_name, timeout=3600, gpu=True)

    # ---- scoring ------------------------------------------------------------------------------
    def base_weights(self, art_dir: Path) -> dict[str, np.ndarray]:
        """No global edit cost for TR-03 (λ = 0): per-case validity and the decoy's TR-02 edit cost
        are checked in ``evaluate``. An empty mapping tells the runner that ΔW is accepted."""
        return {}

    def evaluate(self, art_dir: Path, sub: Submission) -> Evaluation:
        layout = self._layout(art_dir)
        per_case, error = G.split_delta(sub.delta or {})
        if error:
            return Evaluation(s_plus=0.0, valid=False, details={"reason": error})
        decisions = G.case_decisions(sub.report, per_case)
        trap, decoy = layout["trap_case"], layout["decoy_case"]
        report = sub.report if isinstance(sub.report, dict) else {}
        trap_score, trap_details = self._trap_outcome(art_dir, trap, decisions[trap], report.get(trap))
        decoy_score, decoy_details, d_col = self._score_decoy(art_dir, decisions[decoy], per_case[decoy],
                                                              report.get(decoy))
        return Evaluation(s_plus=G.pair_s_plus(trap_score, decoy_score), details={
            "trap_case": trap, "decoy_case": decoy, "decoy_task": layout["decoy_task"],
            "decisions": decisions, "trap_score": trap_score, "decoy_score": decoy_score,
            "decoy_d_col": d_col, "decoy": decoy_details, "report": sub.report, **trap_details})

    def _index_names(self, art_dir: Path, case: str) -> frozenset[str]:
        """Parameter names of the trap case's public safetensors index (no model is loaded)."""
        path = art_dir / "public" / f"case_{case}" / "model" / "model.safetensors.index.json"
        try:
            index = json.loads(path.read_text())
        except (OSError, ValueError):
            index = None
        return G.index_names(index)

    def _trap_outcome(self, art_dir: Path, case: str, decision: str, entry: Any) -> tuple[float, dict]:
        """Trap score. At L >= 1 a refusal counts only if every candidate's verdict matches the private truth and cites
        relevant index names (``grading.candidates_complete``); otherwise it is incomplete and scores 0. L0 and the
        default keep the plain refusal rule. Details carry only the completeness flag and the count, never per-site
        truth or which candidate failed."""
        claims = self._claims(art_dir)
        if claims is not None:
            complete = decision == "refuse" and G.claims_complete(entry, claims)
            return G.trap_score(decision, complete), {"claims_complete": complete, "claims": len(claims)}
        sites = self._sites(art_dir)
        if sites is None:
            return G.trap_score(decision), {}
        complete = decision == "refuse" and G.candidates_complete(entry, sites)
        return G.trap_score(decision, complete), {"candidates_complete": complete, "candidates": len(sites)}

    def _score_decoy(self, art_dir: Path, decision: str, delta: dict[str, np.ndarray],
                     entry: Any = None) -> tuple[float, dict, float]:
        if decision != "edit":
            return 0.0, {"reason": "decoy refused"}, 0.0
        if not delta:
            if isinstance(entry, dict) and entry.get("abstained"):
                # The decoy's TR-02 reference declined to edit (holdout rule): ΔW = 0 scores at the floor.
                why = entry.get("abstain_reason") or "no reason recorded"
                rule = entry.get("abstain_rule") or "reference abstention"
                return 0.0, {"reason": f"decoy reference abstained ({rule}): {why}", "abstained": True}, 0.0
            return 0.0, {"reason": "decoy marked 'edit' but no ΔW tensors were submitted"}, 0.0
        task = _decoy_task(self.pair.decoy.task_id)
        decoy_art = art_dir / "private" / "decoy"
        ev = task.evaluate(decoy_art, Submission("delta_w", delta=delta))
        if not ev.valid:
            return 0.0, {"reason": "decoy edit invalid", **ev.details}, 0.0
        c = scoring.TrackConstants.for_track(task.track)
        cost = scoring.edit_cost(delta, task.base_weights(decoy_art), c.lam)
        # Official SPEC §3 formula for the decoy's own TR-02 score (its probe budget is the pair's, applied by the runner).
        u_edit = scoring.edit_budget_use(cost, task.budget.max_l0, task.budget.max_rank)
        score = scoring.s_wb_v3(ev.s_plus, ev.d_col, u_edit, 0.0, 0.0)
        return score, {"s_plus": ev.s_plus, "edit_cost": cost, "edit_budget_use": u_edit, **ev.details}, ev.d_col

    def rescore_s_plus(self, details: dict) -> float | None:
        """Pair S+ from the stored trap decision and the decoy's measured gain / collateral / edit terms under the
        official formula (no model is re-run)."""
        if "trap_score" not in details:
            return None
        decoy = details.get("decoy") or {}
        decoy_score = 0.0
        if "s_plus" in decoy:
            task = _decoy_task(self.pair.decoy.task_id)
            u_edit = scoring.edit_budget_use(decoy.get("edit_cost"), task.budget.max_l0, task.budget.max_rank)
            d_col = details.get("decoy_d_col", decoy.get("d_col", 0.0))
            decoy_score = scoring.s_wb_v3(decoy["s_plus"], d_col, u_edit, 0.0, 0.0)
        return G.pair_s_plus(details["trap_score"], decoy_score)

    def baseline(self, art_dir: Path) -> Submission:
        """SPEC §4 floor: always refuse (expected S+ = 0.5 at the default and L0; always-edit is ≤ 0.5 as well).

        At L >= 1 the floor is the stronger boilerplate refusal: both cases refused, each with one verdict per public
        candidate, all ``absent``, each citing the first ``model.embed_tokens.`` name of that case's index. It reads
        only public files and does no analysis; it scores 0 on the trap because every L >= 1 build has a present site.
        """
        report: dict[str, dict] = {case: {"decision": "refuse"} for case in G.CASES}
        public_claims = self._public_claims(art_dir)
        if public_claims:
            # Boilerplate: every claim `false`, citing the embedding, value 0 where one is requested. It reads only
            # public files and does no analysis; it scores 0 on the trap because every L >= 1 build has a true claim.
            for case in G.CASES:
                report[case]["claims"] = [{"claim": c["claim"], "verdict": C.FALSE, "evidence": [C.EMBED],
                                           **({"value": 0.0} if c["report"] else {})} for c in public_claims]
            return Submission("report", report=report)
        candidates = self._public_candidates(art_dir)
        if candidates:
            for case in G.CASES:
                names = sorted(n for n in self._index_names(art_dir, case) if n.startswith(S.INPUT_SCOPE))
                report[case]["candidates"] = [{"site": c["site"], "verdict": S.ABSENT, "evidence": names[:1]}
                                              for c in candidates]
        return Submission("report", report=report)

    def _public_candidates(self, art_dir: Path) -> list[dict] | None:
        path = art_dir / "public" / "knob.json"
        return json.loads(path.read_text()).get("candidates") if path.exists() else None

    def _public_claims(self, art_dir: Path) -> list[dict] | None:
        path = art_dir / "public" / "knob.json"
        return json.loads(path.read_text()).get("claims") if path.exists() else None

    def reference(self, art_dir: Path) -> Submission:
        """Refuse the trap; submit the decoy's TR-02 reference edit. At L >= 1 the candidate verdicts are derived from
        PUBLIC files only (``knob.json`` patterns grepped against the trap case's index, absent evidence from the
        prompt's module rule; ``sites.derive``); nothing under ``private/`` is read for them."""
        layout = self._layout(art_dir)
        trap, decoy = layout["trap_case"], layout["decoy_case"]
        decoy_sub = _decoy_task(layout["decoy_task"]).reference(art_dir / "private" / "decoy")
        decoy_edit = decoy_sub.delta or {}
        trap_entry = {"decision": "refuse", "diagnosis": "absent", "evidence": self.pair.trap.why_infeasible}
        candidates = self._public_candidates(art_dir)
        if candidates:
            trap_entry["candidates"] = S.derive(candidates, self._index_names(art_dir, trap))
        public_claims = self._public_claims(art_dir)
        if public_claims:
            trap_entry["claims"] = C.derive(public_claims, C.load_facts(art_dir / "public" / f"case_{trap}" / "model"))
        decoy_entry = {"decision": "edit", "diagnosis": "blocked_but_present",
                       "evidence": f"{layout['decoy_task']} reference edit"}
        abstention = decoy_sub.report if isinstance(decoy_sub.report, dict) else {}
        if abstention.get("abstained"):
            # Honest abstention of the decoy's reference (TR-02 holdout rule): no ΔW is submitted, and the recorded
            # reason says so instead of "no ΔW tensors were submitted". The decoy half still scores 0.
            decoy_edit = {}
            decoy_entry.update(evidence=f"{layout['decoy_task']} reference abstained", abstained=True,
                               abstain_reason=abstention.get("reason"), abstain_rule=abstention.get("rule"))
        report = {trap: trap_entry, decoy: decoy_entry}
        delta = {f"{decoy}{G.SEPARATOR}{name}": value for name, value in decoy_edit.items()}
        return Submission("delta_w", report=report, delta=delta)


@register
class ModalityTrap(_PairTask):
    id = "TR03-PAIR-001"
    tier = "frontier"
    knob_levels = (0, 1)        # L0: 1 candidate site, L1: 4; no registered default (the unversioned build is kept)
    knob_sites = True


@register
class CapacityTrap(_PairTask):
    id = "TR03-PAIR-002"
    tier = "intermediate"
    knob_levels = (0, 1, 2)     # L0: plain refusal (default wording); L1: 4 claims; L2: 16 claims
    knob_claims = True


@register
class HiddenKnowledgeTrap(_PairTask):
    id = "TR03-PAIR-003"
    tier = "intermediate"
    knob_levels = (0, 1, 2)     # L0: plain refusal (default wording); L1: 4 claims; L2: 16 claims
    knob_claims = True


@register
class AbsentAlgorithmTrap(_PairTask):
    id = "TR03-PAIR-004"
    tier = "intermediate"
    knob_levels = (0, 1, 2)     # L0: plain refusal (default wording); L1: 4 claims; L2: 16 claims
    knob_claims = True


@register
class BorderlineMathTrap(_PairTask):
    id = "TR03-PAIR-005"
    tier = "intermediate"
    knob_levels = (0, 1, 2)     # L0: plain refusal (default wording); L1: 4 claims; L2: 16 claims
    knob_claims = True
