"""Run identity/evidence retention. No subject inference; builds/evaluation are stubbed."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from weightbench import runner
from weightbench.ledger import Ledger
from weightbench.task import Evaluation, Submission


@pytest.fixture
def fake_runner(tmp_path, monkeypatch):
    class Task:
        track, tier = 'TR05', 'foundational'
        def reference(self, ad):
            return Submission('report', report={})
        baseline = reference
    monkeypatch.setattr(runner, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(runner, 'nondefault_level', lambda *a: None)
    monkeypatch.setattr(runner, 'get_task', lambda *a: Task())
    calls = []
    def build(*args, **kwargs):
        calls.append('build')
        return tmp_path / 'art'
    def evaluate(task, ad, sub, ledger):
        ledger.append('evaluation', {'s_plus': 1.0, 'valid': True})
        return Evaluation(1.0)
    monkeypatch.setattr(runner, 'build', build)
    monkeypatch.setattr(runner, '_evaluate_and_log', evaluate)
    monkeypatch.setattr(runner, 'score_from_ledger', lambda *a: {'s_wb': 1.0, 'valid': True})
    return calls


def test_completed_run_is_unchanged_and_rejected_before_build(fake_runner, monkeypatch):
    result = runner.run('TR05-TEST-001', 'reference')
    directory = runner.RUNS / result['run_id']
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError, match='fresh --sample'):
        runner.run('TR05-TEST-001', 'reference')
    assert len(fake_runner) == 1
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    second = runner.run('TR05-TEST-001', 'reference', sample=1)
    assert second['run_id'].endswith('__r1') and len(fake_runner) == 2


@pytest.mark.parametrize('kind', ['partial', 'empty', 'file'])
def test_incomplete_or_empty_run_ids_are_not_reused(fake_runner, kind):
    directory = runner.RUNS / 'TR05-TEST-001__reference__s0__r0'
    directory.parent.mkdir(parents=True)
    if kind == 'file':
        directory.write_bytes(b'existing file')
    else:
        directory.mkdir()
        if kind == 'partial':
            Ledger(directory / 'ledger.jsonl').append('run_start', {'partial': True})
    with pytest.raises(FileExistsError):
        runner.run('TR05-TEST-001', 'reference')
    assert not fake_runner
    if kind == 'file':
        assert directory.read_bytes() == b'existing file'
    elif kind == 'partial':
        ledger = Ledger(directory / 'ledger.jsonl')
        assert ledger.verify() and len(list(ledger.entries())) == 1


def test_failed_build_retains_id_and_records_failure(fake_runner, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError('build failed')
    monkeypatch.setattr(runner, 'build', fail)
    with pytest.raises(RuntimeError, match='build failed'):
        runner.run('TR05-TEST-001', 'reference')
    directory = runner.RUNS / 'TR05-TEST-001__reference__s0__r0'
    ledger = Ledger(directory / 'ledger.jsonl')
    assert ledger.verify() and [e['kind'] for e in ledger.entries()] == ['run_start', 'run_error']
    assert list(ledger.entries('run_error'))[0]['payload']['stage'] == 'build'
    with pytest.raises(FileExistsError):
        runner.run('TR05-TEST-001', 'reference')


def test_force_flag_does_not_replace_existing_evidence(fake_runner):
    result = runner.run('TR05-TEST-001', 'reference')
    path = runner.RUNS / result['run_id'] / 'ledger.jsonl'
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match='--force'):
        runner.main(['run', '--task', 'TR05-TEST-001', '--agent', 'reference', '--force'])
    assert path.read_bytes() == before and len(fake_runner) == 1


def test_concurrent_duplicate_requests_have_one_winner(fake_runner):
    ready = Barrier(2)
    def attempt(_):
        ready.wait(timeout=5)
        try:
            runner.run('TR05-TEST-001', 'reference')
            return 'completed'
        except FileExistsError:
            return 'rejected'
    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(attempt, range(2)))
    assert sorted(outcomes) == ['completed', 'rejected'] and len(fake_runner) == 1
    directory = runner.RUNS / 'TR05-TEST-001__reference__s0__r0'
    assert Ledger(directory / 'ledger.jsonl').verify()
    assert json.loads((directory / 'result.json').read_text())['s_wb'] == 1.0
