"""TR02-LOOP reference: gate algebra, candidate shortlists, metered hook wiring, split integrity; CPU toys only."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from weightbench.tracks.tr02_loops_lib.edits import down_proj, validate_delta
from weightbench.tracks.tr02_loops_lib.reference import (
    CLEAN_MARGIN, CLEAN_MAX, AuthorProbes, _prefix, _trace, calibration_split, closure_factors, fit_gate,
    loop_shortlist, shortlist, trace_positions,
)


def test_prompt_shortlist_prefers_small_effective_edits_on_distinct_layers():
    def candidate(index, layer, strength, rate, margin):
        return dict(index=index, layer=layer, strength=strength, closure_rate=rate, margin=margin)
    candidates = [candidate(0, 1, 4, 1, 100), candidate(1, 1, .5, 1, 1),
                  candidate(2, 2, 1, 1, 5), candidate(3, 3, 1, .875, 0),
                  candidate(4, 4, 4, .0, -1)]
    assert [c["index"] for c in shortlist(candidates, 4)] == [1, 2, 3, 0]
    assert [c["index"] for c in shortlist(candidates)] == [1, 2]
    assert shortlist([]) == []


def test_loop_shortlist_spares_clean_reasoning_first():
    def candidate(index, loop, clean):
        return dict(index=index, loop_rate=loop, clean_rate=clean)
    candidates = [candidate(0, .99, 5 * CLEAN_MAX), candidate(1, .4, 0), candidate(2, .7, CLEAN_MAX / 2),
                  candidate(3, .7, CLEAN_MAX / 4)]
    assert [c["index"] for c in loop_shortlist(candidates)] == [3, 2]
    assert [c["index"] for c in loop_shortlist(candidates, 4)] == [3, 2, 1, 0]


def test_gate_generalizes_to_unseen_mixtures_and_spares_negative_inputs():
    positive = np.array([[1., 0., 0.], [0., 0., 1.]])
    negative = np.array([[0., 1., 0.], [0., 2., 0.]])
    gate = fit_gate(positive, negative, ridge=1e-8)
    assert np.mean(positive, axis=0) @ gate == pytest.approx(1., abs=1e-6)
    assert np.max(np.abs(negative @ gate)) < 1e-7
    factors = closure_factors(np.array([3., 4.]), gate, scale=2., strength=.5)
    delta = factors["U"].T @ factors["V"]
    assert np.linalg.matrix_rank(delta) == 1
    assert validate_delta({down_proj(0): delta}, {down_proj(0): delta.shape}, "single_rank1") is None
    np.testing.assert_allclose(delta @ np.mean(positive, axis=0), [.6, .8], atol=1e-6)
    np.testing.assert_allclose(delta @ negative.T, 0, atol=1e-7)


@pytest.mark.parametrize("positive,negative,ridge", [
    ([], [], 1e-3), ([[1, 2]], [[1]], 1e-3), ([[float("nan")]], [[1]], 1e-3),
    ([[0]], [[0]], 1e-3), ([[1]], [[1]], 0), ([[1]], [[1]], float("inf")),
])
def test_gate_rejects_invalid_or_degenerate_data(positive, negative, ridge):
    with pytest.raises(ValueError):
        fit_gate(positive, negative, ridge)


@pytest.mark.parametrize("direction,gate,scale,strength", [
    ([0, 0], [1], 1, 1), ([1, 0], [0], 1, 1), ([1], [1], -1, 1),
    ([1], [1], 1, 0), ([float("inf")], [1], 1, 1),
])
def test_factors_reject_unusable_edits(direction, gate, scale, strength):
    with pytest.raises(ValueError):
        closure_factors(direction, gate, scale, strength)


def test_trace_positions_split_loop_tail_from_clean_prefix():
    gen = {"ids": list(range(1000)), "loop_onset_token": 600}
    full = trace_positions(gen, np.random.default_rng(0), None)
    assert full["loop"][0] == 600 and full["loop"][-1] == 999
    assert full["clean"][-1] == 600 - CLEAN_MARGIN - 1
    sampled = trace_positions(gen, np.random.default_rng(0), 10)
    assert len(sampled["loop"]) == len(sampled["clean"]) == 10
    assert set(sampled["loop"]) <= set(full["loop"]) and set(sampled["clean"]) <= set(full["clean"])
    for bad in ({"ids": list(range(1000)), "loop_onset_token": None},
                {"ids": list(range(1000)), "loop_onset_token": CLEAN_MARGIN + 3},
                {"ids": list(range(1000)), "loop_onset_token": 995}):
        assert trace_positions(bad, np.random.default_rng(0)) is None


def _toy_runtime():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(2, 2, bias=False)
            self.mlp.down_proj.weight.data.copy_(torch.eye(2))

        def forward(self, x):
            return x + self.mlp.down_proj(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])
            self.fail = False

        def forward(self, input_ids, use_cache=False, **kwargs):
            x = self.layers[0](torch.nn.functional.one_hot(input_ids, num_classes=2).float())
            if self.fail:
                raise RuntimeError("fixture decoder failure")
            return SimpleNamespace(last_hidden_state=x)

    class Subject(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.head = Decoder(), nn.Identity()

        def get_output_embeddings(self):
            return self.head

    class Encoding(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "fixture"

        def __call__(self, texts, **kwargs):
            if isinstance(texts, str):
                return {"input_ids": [0, 1]}
            return Encoding(input_ids=torch.tensor([[0, 1]] * len(texts)))

    subject = Subject()
    return subject, SimpleNamespace(model=subject, tok=Tokenizer(), dev=torch.device("cpu"))


def _hooks_cleared(subject):
    block = subject.model.layers[0]
    return not block._forward_hooks and not block.mlp.down_proj._forward_pre_hooks


def test_prefix_hooks_are_metered_and_removed_after_success_or_error(tmp_path):
    subject, rt = _toy_runtime()
    probes = AuthorProbes(tmp_path / "ledger.jsonl", activation_cap=3)
    result = probes.call("activation", "prefix", lambda: _prefix(rt, ["question"], 1, True, (0,)))
    np.testing.assert_array_equal(result["inputs"][0], [0, 1])
    assert result["norms"][0] == 2 and result["closure_rate"] == 1 and result["margin"] == 2
    assert _hooks_cleared(subject)
    subject.model.fail = True
    with pytest.raises(RuntimeError, match="fixture decoder failure"):
        probes.call("activation", "failure", lambda: _prefix(rt, ["question"], 1, True, (0,)))
    assert _hooks_cleared(subject)
    with pytest.raises(AttributeError):
        probes.call("activation", "bad-layer", lambda: _prefix(rt, ["question"], 1, True, (0, 1)))
    assert _hooks_cleared(subject)
    assert probes.used["activation"] == 3 and probes.ledger.verify()
    assert len(list(probes.ledger.entries("probe_error"))) == 2


def test_trace_reads_states_that_decide_the_next_token(tmp_path):
    subject, rt = _toy_runtime()
    probes = AuthorProbes(tmp_path / "ledger.jsonl")
    ids = [0, 0, 1, 1]                                           # sequence: prompt [0, 1] + generated ids
    positions = {"loop": np.array([2, 3]), "clean": np.array([0])}
    got = probes.call("activation", "trace", lambda: _trace(rt, "q", ids, positions, 1, (0,)))
    np.testing.assert_array_equal(got["inputs"]["loop"][0], [[0, 1], [0, 1]])
    np.testing.assert_array_equal(got["inputs"]["clean"][0], [[1, 0]])
    assert got["rates"] == {"loop": 1.0, "clean": 0.0} and got["norms"]["loop"][0] == 2
    assert _hooks_cleared(subject) and probes.ledger.verify()


def test_author_probe_budget_and_append_only_evidence(tmp_path):
    probes = AuthorProbes(tmp_path / "ledger.jsonl", forward_cap=1)
    assert probes.call("forward", "first", lambda: 7) == 7
    with pytest.raises(RuntimeError, match="budget exhausted"):
        probes.call("forward", "second", lambda: pytest.fail("over-budget code ran"))
    assert probes.ledger.verify()
    assert len(list(probes.ledger.entries("probe"))) == 1
    assert len(list(probes.ledger.entries("probe_rejected"))) == 1
    with pytest.raises(ValueError, match="fresh experiment"):
        AuthorProbes(tmp_path / "ledger.jsonl")


def test_calibration_split_excludes_eval_and_preserves_original_fit():
    pool = [{"id": str(i)} for i in range(200)]
    original = pool[:32]
    excluded = {str(i) for i in range(150, 200)}
    fit, dev = calibration_split(original, pool + pool, excluded, 9, "TR02-LOOP-001")
    assert fit[:32] == original
    assert len(fit) == len({x["id"] for x in fit}) == 80
    assert len(dev) == len({x["id"] for x in dev}) == 64
    assert not ({x["id"] for x in fit} & {x["id"] for x in dev})
    assert not ({x["id"] for x in fit + dev} & excluded)
    assert (fit, dev) == calibration_split(original, pool, excluded, 9, "TR02-LOOP-001")
    assert (fit, dev) != calibration_split(original, pool, excluded, 10, "TR02-LOOP-001")
    with pytest.raises(ValueError, match="insufficient"):
        calibration_split(original, pool[:100], excluded, 9, "TR02-LOOP-001")
    for invalid in (original + original[:1], original + [pool[150]]):
        with pytest.raises(ValueError, match="unique and disjoint"):
            calibration_split(invalid, pool, excluded, 9, "TR02-LOOP-001")
