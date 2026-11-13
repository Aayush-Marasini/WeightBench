"""Declared controls, real ledger/API integration and hand-computed paired statistics; no LLMs."""
import hashlib
import json

import pytest

from weightbench import comparison as P, conditions as C, manifest, runner, verify
from weightbench.harness import agents
from weightbench.ledger import Ledger
from weightbench.task import Budget, Evaluation, Task


class FixtureTask(Task):
    id, track, tier, submission = 'TR02-LOOP-001', 'TR02', 'intermediate', 'report'
    budget = Budget(n_forward=1, n_probe=1)
    probe_kinds = {'generate': 'forward', 'activations': 'probe'}
    def prompt(self, art):
        return 'Matched-protocol fixture.'
    def probe(self, art, kind, payload):
        return {'stub': True}
    def evaluate(self, art, sub):
        return Evaluation(float(sub.report['score']))


@pytest.fixture
def study(tmp_path, monkeypatch):
    for name in ('RUNS', 'WORK', 'GUARDS'):
        monkeypatch.setattr(runner, name, tmp_path / name.lower())
    art = tmp_path / 'art'
    (art / 'public/model').mkdir(parents=True)
    (art / 'private').mkdir()
    (art / 'public/param_shapes.json').write_text('{}')
    (art / 'public/model/subject.safetensors').write_bytes(b'fixture, never loaded')
    snapshot = tmp_path / 'manifest.json'
    snapshot.write_text(json.dumps({'schema_version': 2, 'artifact_policy': manifest.ARTIFACT_POLICY,
        'version': 'fixture', 'tasks': {FixtureTask.id: {**manifest.task_spec(FixtureTask()),
                                                      'seeds': {'0': manifest.seed_entry(art)}}}}))
    plan_path = tmp_path / 'plan.json'
    plan = P.make_plan(snapshot, 'cmd', 'fixture')
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(runner, 'get_task', lambda *a: FixtureTask())
    monkeypatch.setattr(runner, 'nondefault_level', lambda *a: None)
    monkeypatch.setattr(runner, 'build', lambda *a, **kw: art)
    monkeypatch.setattr(verify, '_load_task', lambda result: FixtureTask())
    install = runner.sandbox.install_guard
    monkeypatch.setattr(runner.sandbox, 'install_guard', lambda root, public, excluded:
                        install(root, tmp_path / 'empty'))
    calls = []
    def launch(spec, model, workdir, transcript, *args):
        import urllib.request
        calls.append(workdir.name)
        transcript.write_text('')
        url, token = args[-2:]
        def post(route, payload):
            request = urllib.request.Request(url + route, json.dumps(payload).encode(),
                                              headers={'X-WB-Token': token})
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status
        # A real metered request with a stub task, not subject inference.
        assert post('/probe', {'kind': 'generate', 'payload': {}}) == 200
        score = .8 if (workdir / 'model/subject.safetensors').exists() else .2
        assert post('/submit', {'type': 'report', 'report': {'score': score}}) == 200
        return 0
    monkeypatch.setattr(runner, '_launch_agent', launch)
    def run(arm, command='unused'):
        return runner.run(FixtureTask.id, 'cmd', 'fixture', condition=arm, access_plan=plan_path,
                          agent_spec=agents.get('cmd', command, local_ports=[8000]))
    return plan_path, plan, art, snapshot, calls, run


def test_missing_both_arms_and_partial_pairs_cannot_become_a_complete_study(study):
    path, plan, art, snapshot, calls, run = study
    report = P.compare(path, runner.RUNS, draws=200)
    assert report['expected_pairs'] == 1 and report['complete_pairs'] == 0
    assert report['arm_status'] == {'missing': 2} and report['estimates'] is None
    run('weights_behavior')
    report = P.compare(path, runner.RUNS, exploratory=True, draws=200)
    assert report['arm_status'] == {'complete': 1, 'missing': 1}, report
    assert report['exploratory'] and not report['ready'] and report['estimates'] is None


def test_exploratory_estimate_labels_complete_pairs_and_keeps_missing_pair_counts(study):
    path, plan, art, snapshot, calls, run = study
    path.write_text(json.dumps(P.make_plan(snapshot, 'cmd', 'fixture', samples=[0, 1])))
    for arm in P.ARMS:
        run(arm)
    assert P.compare(path, runner.RUNS, draws=200)['estimates'] is None
    report = P.compare(path, runner.RUNS, exploratory=True, draws=200)
    assert report['exploratory'] and not report['ready']
    assert report['expected_pairs'] == 2 and report['complete_pairs'] == 1
    assert report['arm_status'] == {'complete': 2, 'missing': 2}
    assert report['estimates']['weight_access']['s_wb']['mean'] == pytest.approx(.57)


def test_matched_runs_reconcile_to_plan_and_hand_computed_metered_effect(study):
    path, plan, art, snapshot, calls, run = study
    for arm in P.ARMS:
        result = run(arm)
        assert result['access_plan_sha256'] == C.file_hash(path)
        assert result['harness_source_sha256'] == plan['harness_source_sha256']
        assert verify.verify_run(runner.RUNS / result['run_id']).ok
    report = P.compare(path, runner.RUNS, draws=200)
    assert report['ready'] and report['complete_pairs'] == 1
    # E=1, E_budget=2, beta=.1: (.8-.2) * .95 = .57.
    assert report['estimates']['weight_access']['s_wb']['mean'] == pytest.approx(.57)
    assert report['estimates']['weight_access']['s_plus']['mean'] == pytest.approx(.6)
    assert report['estimates']['weight_access']['s_wb']['ci95'] is None
    evidence = report['pairs'][0]['arms']['weights_behavior']['sha256']
    assert evidence['review.json'] is None and evidence['transcript.jsonl'] == hashlib.sha256(b'').hexdigest()
    assert len(calls) == 2
    with pytest.raises(FileExistsError, match='fresh'):
        run('weights_behavior')


def test_different_agent_settings_are_not_a_weight_access_effect(study):
    path, plan, art, snapshot, calls, run = study
    run('weights_behavior', 'unused --mode A')
    run('behavior_only', 'unused --mode B')
    report = P.compare(path, runner.RUNS, draws=200)
    assert report['pair_status'] == {'protocol_mismatch': 1}
    assert report['estimates'] is None and not report['ready']


def test_invalid_ledger_and_unfinished_run_are_visible_in_coverage(study):
    path, plan, art, snapshot, calls, run = study
    run('weights_behavior')
    directory = runner.RUNS / plan['cases'][0]['run_ids']['weights_behavior']
    Ledger(directory / 'ledger.jsonl').append('unexpected_after_evaluation', {})
    other = runner.RUNS / plan['cases'][0]['run_ids']['behavior_only']
    other.mkdir()
    report = P.compare(path, runner.RUNS, draws=200)
    assert report['arm_status'] == {'invalid_evidence': 1, 'unfinished': 1}
    assert report['estimates'] is None


def test_changed_subject_is_rejected_before_agent_execution(study):
    path, plan, art, snapshot, calls, run = study
    (art / 'private/new_ground_truth.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact_sha256'):
        run('weights_behavior')
    assert calls == []
    directory = runner.RUNS / plan['cases'][0]['run_ids']['weights_behavior']
    error = next(Ledger(directory / 'ledger.jsonl').entries('run_error'))['payload']
    assert error['stage'] == 'access_plan' and 'artifact_sha256' in error['error']


def test_undeclared_identity_or_source_is_rejected_before_reserving_run(study, monkeypatch):
    path, plan, art, snapshot, calls, run = study
    with pytest.raises(ValueError, match='settings differ'):
        P.planned_run(path, FixtureTask.id, 'cmd', 'other-model', 0, 0, None, P.ARMS[0], 200)
    monkeypatch.setattr(C, 'source_fingerprint', lambda: 'f' * 64)
    with pytest.raises(ValueError, match='source differs'):
        run('weights_behavior')
    assert not runner.RUNS.exists() and not calls


def test_plan_classifies_query_free_and_no_weight_controls_and_rejects_duplicates(study):
    path, plan, art, snapshot, calls, run = study
    data = json.loads(snapshot.read_text())
    data['tasks'][FixtureTask.id]['budget']['n_forward'] = 0
    data['tasks'][FixtureTask.id]['budget']['n_probe'] = 0
    snapshot.write_text(json.dumps(data))
    assert P.make_plan(snapshot, 'cmd', 'fixture')['cases'][0]['kind'] == 'weights_without_probes'
    spec = data['tasks'].pop(FixtureTask.id)
    spec.update(track='TR04', probe_kinds={'game': 'forward'})
    spec['seeds']['0']['public'] = {'go/rules.py': hashlib.sha256(b'fixture').hexdigest()}
    data['tasks']['TR04-GO-001'] = spec
    snapshot.write_text(json.dumps(data))
    negative = P.make_plan(snapshot, 'cmd', 'fixture')
    assert negative['cases'][0]['kind'] == 'no_input_weights'
    # Expose no extra negative-control information in the weight arm.
    assert negative['cases'][0]['files']['weights'] == {}
    with pytest.raises(ValueError, match='duplicate'):
        P.make_plan(snapshot, 'cmd', 'fixture', samples=[0, 0])
    with pytest.raises(ValueError, match='indices'):
        P.make_plan(snapshot, 'cmd', 'fixture', seeds=[True])
    negative['cases'].append(negative['cases'][0])
    path.write_text(json.dumps(negative))
    with pytest.raises(ValueError, match='duplicate'):
        P.read_plan(path)


def test_negative_controls_are_separate_from_weight_access_estimates(study, monkeypatch):
    path, plan, art, snapshot, calls, run = study
    negative = {**plan['cases'][0], 'task': 'TR04-GO-001', 'track': 'TR04', 'kind': 'no_input_weights',
                'files': {'metadata': {}, 'weights': {}},
                'run_ids': {arm: P._run_id('TR04-GO-001', 'cmd', 'fixture', 0, 0, arm) for arm in P.ARMS}}
    plan['cases'].append(negative)
    path.write_text(json.dumps(plan))
    def arm_read(plan, case, arm, root, sha):
        score = (1. if case['kind'] == 'no_input_weights' else .6) if arm == 'weights_behavior' else 0.
        return dict(status='complete', s_wb=score, s_plus=score, task_prompt_sha256='p' * 64,
                    agent_spec_sha256='a' * 64)
    monkeypatch.setattr(P, '_arm', arm_read)
    report = P.compare(path, runner.RUNS, draws=200)
    assert report['estimates']['weight_access']['s_wb']['mean'] == .6
    assert report['estimates']['no_input_weights']['s_wb']['mean'] == 1.


def test_quarantined_arm_counts_as_zero_instead_of_being_dropped(study, monkeypatch):
    path, plan, art, snapshot, calls, run = study
    audit = runner.harness_audit.audit_run
    def quarantine(*args, **kwargs):
        value = audit(*args, **kwargs)
        value.update(flags=['fixture_quarantine'], quarantined=True)
        return value
    with monkeypatch.context() as patch:
        patch.setattr(runner.harness_audit, 'audit_run', quarantine)
        run('weights_behavior')
    run('behavior_only')
    report = P.compare(path, runner.RUNS, draws=200)
    assert report['ready']
    assert report['estimates']['weight_access']['s_wb']['mean'] == pytest.approx(-.19)


def test_hierarchical_effect_weights_tracks_tasks_and_seeds_equally():
    def row(task, seed, sample, value):
        return dict(task=task, track=task.split('-')[0], seed=seed, sample=sample,
                    difference={'s_wb': value})
    rows = [row('TR01-A', 0, 0, 1), *[row('TR01-A', 1, i, 0) for i in range(3)],
            row('TR01-B', 0, 0, 0), row('TR02-C', 0, 0, -.5)]
    result = P.estimate(rows, 's_wb', draws=500, seed=7)
    # Task A=.5, B=0 -> TR01=.25; TR02=-.5 -> overall=-.125.
    assert result['mean'] == -.125 and result['n_pairs'] == 6 and result['n_subject_seeds'] == 4
    assert result == P.estimate(list(reversed(rows)), 's_wb', draws=500, seed=7)
    assert -.25 <= result['ci95'][0] <= result['mean'] <= result['ci95'][1] <= .25
    with pytest.raises(ValueError):
        P.estimate([], 's_wb')


def test_single_pair_has_no_interval_and_replicated_constant_effect_is_exact():
    row = dict(task='TR01-A', track='TR01', seed=0, sample=0, difference={'s_wb': .2})
    assert P.estimate([row], 's_wb')['ci95'] is None
    result = P.estimate([row, {**row, 'sample': 1}], 's_wb', draws=200)
    assert result['mean'] == .2 and result['ci95'] == [.2, .2]


def test_plan_cli_is_immutable_and_report_does_not_execute_agents(study, tmp_path):
    path, plan, art, snapshot, calls, run = study
    output = tmp_path / 'cli-plan.json'
    argv = ['plan', '--manifest', str(snapshot), '--agent', 'cmd', '--model', 'fixture', '--output', str(output)]
    P.main(argv)
    with pytest.raises(FileExistsError):
        P.main(argv)
    report = tmp_path / 'report.json'
    P.main(['report', '--plan', str(output), '--runs', str(runner.RUNS), '--output', str(report)])
    assert json.loads(report.read_text())['arm_status'] == {'missing': 2} and calls == []


def test_wsl_delegation_converts_plan_paths_in_both_argument_forms(tmp_path):
    from weightbench.platform_compat import to_wsl_path
    path = tmp_path / 'with spaces' / 'plan.json'
    for args in (['--access-plan', str(path)], ['--access-plan=' + str(path)]):
        command = runner.wsl_command(['run', *args], tmp_path)[4]
        assert to_wsl_path(path.resolve()) in command
