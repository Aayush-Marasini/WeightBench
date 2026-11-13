"""Matched controls use real file/API/ledger paths with a stubbed agent, never a model."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from weightbench import conditions as C, leaderboard, report, runner, server, verify
from weightbench.harness import agents
from weightbench.ledger import Ledger
from weightbench.task import Budget, Evaluation, Task, load_all


def test_profiles_cover_registry_and_allow_only_declared_behavior():
    for tid, cls in load_all().items():
        if tid.startswith('SMOKE'):
            continue
        profile = C.profile(tid)
        assert set(profile.behavior_probes) <= set(cls.probe_kinds)
        assert C.probes(tid, cls.probe_kinds, 'weights_behavior') == C.probes(tid, cls.probe_kinds, 'behavior_only')
    with pytest.raises(ValueError, match='profile'):
        C.profile('UNKNOWN')


def test_allowlists_preserve_observations_but_hide_weights_and_unknown_files(tmp_path):
    source = tmp_path / 'public'
    source.mkdir()
    for name in ('ARCH.md', 'failure_trajectory.npz', 'policy.npz', 'unknown.json'):
        (source / name).write_bytes(name.encode())
    inventories = {}
    for condition in ('weights_behavior', 'behavior_only'):
        inventories[condition] = C.copy_inputs(source, tmp_path / condition, 'TR02-CTRL-001', condition)
        assert not (tmp_path / condition / 'unknown.json').exists()
        assert (tmp_path / condition / 'failure_trajectory.npz').exists()
    assert inventories['weights_behavior']['files']['metadata'] == inventories['behavior_only']['files']['metadata']
    assert inventories['weights_behavior']['subject_weights'] == inventories['behavior_only']['subject_weights']
    assert (tmp_path / 'weights_behavior/policy.npz').exists()
    assert not (tmp_path / 'behavior_only/policy.npz').exists()
    assert not inventories['behavior_only']['files']['weights']
    assert inventories['behavior_only']['n_weight_files'] == inventories['weights_behavior']['n_weight_files'] == 1
    (source / 'policy.npz').unlink()
    with pytest.raises(ValueError, match='no declared weight'):
        C.copy_inputs(source, tmp_path / 'stale', 'TR02-CTRL-001', 'weights_behavior')


class ControlTask(Task):
    id, track, tier = 'TR02-LOOP-001', 'TR02', 'intermediate'
    budget = Budget(n_forward=1, n_probe=1)
    probe_kinds = {'generate': 'forward', 'activations': 'probe'}
    meter_exclude = ('subject.safetensors',)
    def prompt(self, art):
        return 'Protocol fixture only.'
    def probe(self, art, kind, payload):
        return {'stub': kind}
    def evaluate(self, art, submission):
        return Evaluation(1.)


@pytest.mark.parametrize('condition', ['weights_behavior', 'behavior_only'])
def test_server_blocks_internal_probes_without_consuming_budget(tmp_path, condition):
    ledger = Ledger(tmp_path / 'ledger.jsonl')
    state = server.RunState(ControlTask(), tmp_path, tmp_path, ledger, tmp_path, condition=condition)
    assert state.probe('activations', {})[0] == 403
    assert state.used == {'forward': 0, 'probe': 0}
    assert state.probe('generate', {})[0] == 200
    assert state.probe('generate', {})[0] == 429
    assert len(list(ledger.entries('access_rejected'))) == 1
    assert len(list(ledger.entries('probe'))) == 1
    assert ledger.verify()


def test_real_runner_keeps_conditions_separate_and_retains_shared_evidence(tmp_path, monkeypatch):
    for name in ('RUNS', 'WORK', 'GUARDS'):
        monkeypatch.setattr(runner, name, tmp_path / name.lower())
    monkeypatch.setattr(runner, 'get_task', lambda *a: ControlTask())
    monkeypatch.setattr(runner, 'nondefault_level', lambda *a: None)
    artifact = tmp_path / 'art'
    (artifact / 'public/model').mkdir(parents=True)
    (artifact / 'public/param_shapes.json').write_text('{}')
    (artifact / 'public/model/subject.safetensors').write_bytes(b'fixture, not a model')
    monkeypatch.setattr(runner, 'build', lambda *a, **kw: artifact)
    prompts = []
    def launch(spec, model, workdir, transcript, *args):
        prompts.append((workdir / 'TASK.md').read_text())
        assert (workdir / 'param_shapes.json').exists()
        has_weights = (workdir / 'model/subject.safetensors').exists()
        assert has_weights == ('weights_behavior' in workdir.name)
        transcript.write_text('')
        # No agent/model execution; submit through the same local endpoint the sandbox uses.
        import urllib.request
        url, token = args[-2:]
        request = urllib.request.Request(url + '/budget', headers={'X-WB-Token': token})
        with urllib.request.urlopen(request, timeout=10) as response:
            budget = json.load(response)
        assert budget['probes'] == {'generate': 'forward'}
        assert budget['n_forward'] == budget['n_probe'] == 1
        request = urllib.request.Request(url + '/submit', json.dumps({'type': 'report', 'report': {}}).encode(),
                                         headers={'X-WB-Token': token})
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 200
        return 0
    monkeypatch.setattr(runner, '_launch_agent', launch)
    # Avoid parsing the intentionally fake safetensors fixture as model weights.
    install = runner.sandbox.install_guard
    def guarded_install(root, public, excluded):
        assert excluded == ()
        return install(root, tmp_path / 'empty')
    monkeypatch.setattr(runner.sandbox, 'install_guard', guarded_install)
    spec = agents.get('cmd', 'unused', local_ports=[8000])
    results = [runner.run(ControlTask.id, 'cmd', 'fixture', agent_spec=spec, condition=c)
               for c in ('weights_behavior', 'behavior_only')]
    assert prompts[0] == prompts[1]
    assert results[0]['run_id'] != results[1]['run_id']
    inputs = []
    for result in results:
        directory = runner.RUNS / result['run_id']
        assert verify.verify_run(directory).ok
        inputs.append(next(Ledger(directory / 'ledger.jsonl').entries('access_inputs'))['payload'])
        with pytest.raises(FileExistsError):
            runner.run(ControlTask.id, 'cmd', 'fixture', agent_spec=spec, condition=result['condition'])
    assert inputs[0]['files']['metadata'] == inputs[1]['files']['metadata']
    assert inputs[0]['subject_weights'] == inputs[1]['subject_weights']
    assert inputs[0]['budget'] == inputs[1]['budget'] == asdict(ControlTask.budget)
    assert inputs[0]['local_model_execution'] is inputs[1]['local_model_execution'] is False
    assert inputs[0]['max_turns'] == inputs[1]['max_turns'] == 200
    assert inputs[0]['agent_spec_sha256'] == inputs[1]['agent_spec_sha256']
    assert inputs[0]['task_prompt_sha256'] == inputs[1]['task_prompt_sha256']
    for condition in ('weights_behavior', 'behavior_only'):
        assert runner._pending_batch_runs([(ControlTask.id, 'cmd', 'fixture', 0)], 0, condition=condition) == []
    entries = list(Ledger(runner.RUNS / results[0]['run_id'] / 'ledger.jsonl').entries())
    access = next(e['payload'] for e in entries if e['kind'] == 'access_inputs')
    for malformed in (None, [], 'invalid'):
        access['files'] = malformed
        assert 'malformed access-file evidence' in verify._structure_problems(results[0], entries)
    monkeypatch.setattr(report, 'RUNS', runner.RUNS)
    assert len({r['system'] for r in report.load()}) == 2
    assert len({leaderboard.system_name(r) for r in results}) == 2
    altered = {**results[0], 'condition': 'behavior_only'}
    (runner.RUNS / altered['run_id'] / 'result.json').write_text(json.dumps(altered))
    assert not verify.verify_run(runner.RUNS / altered['run_id']).ok
    with pytest.raises(ValueError, match='verification'):
        runner._pending_batch_runs([(ControlTask.id, 'cmd', 'fixture', 0)], 0, condition='weights_behavior')


def test_legacy_identity_and_floor_policy():
    assert runner.make_run_id('T', 'M', 0, 0) == 'T__M__s0__r0'
    assert runner.make_run_id('T', 'M', 0, 0, condition='behavior_only').endswith('__behavior_only')
    with pytest.raises(ValueError, match='author'):
        runner.run('TR02-LOOP-001', 'reference', condition='behavior_only')
