"""Declare a matched access study, then verify and compare its runs without executing agents."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
import time

from . import conditions as C, manifest, scoring
from .ledger import Ledger

ARMS = ('weights_behavior', 'behavior_only')
METRICS = ('s_wb', 's_plus')


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _indices(values: list[int]) -> list[int]:
    if not values or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in values):
        raise ValueError('indices must be nonempty non-negative integers')
    if len(set(values)) != len(values):
        raise ValueError('duplicate indices')
    return sorted(values)


def _run_id(task: str, agent: str, model: str, seed: int, sample: int, arm: str) -> str:
    from .runner import make_run_id, run_tag
    value = make_run_id(task, run_tag(agent, model), seed, sample, condition=arm)
    if '/' in value or '\\' in value:
        raise ValueError('run IDs must be single path components; choose a compatible model tag')
    return value


def make_plan(snapshot: Path, agent: str, model: str, tasks: list[str] | None = None,
              seeds: list[int] | None = None, samples: list[int] | None = None,
              max_turns: int = 200) -> dict:
    """Freeze expected subjects, budgets and both immutable run IDs before any agent execution."""
    data = json.loads(snapshot.read_text())
    if data.get('schema_version') != 2 or data.get('artifact_policy') != manifest.ARTIFACT_POLICY:
        raise ValueError('comparison requires an immutable schema-2 artifact manifest')
    if agent in ('baseline', 'reference') or not agent or not model or max_turns <= 0:
        raise ValueError('comparison requires an agent, model and positive turn limit')
    tasks = sorted(data['tasks'] if tasks is None else tasks)
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError('select unique nonempty tasks')
    samples = _indices([0] if samples is None else samples)
    if seeds is not None:
        seeds = _indices(seeds)
    cases = []
    for task in tasks:
        spec = data['tasks'][task]
        policy = C.profile(task)
        chosen_seeds = _indices([int(s) for s in spec['seeds']]) if seeds is None else seeds
        for seed in chosen_seeds:
            subject = spec['seeds'].get(str(seed))
            if not subject or not subject.get('public'):
                raise ValueError(f'{task} seed {seed}: no frozen public assets')
            files = {kind: {name: h for name, h in subject['public'].items()
                            if C.input_kind(policy, name) == kind} for kind in ('metadata', 'weights')}
            if policy.weights and not files['weights']:
                raise ValueError(f'{task}: frozen subject has no declared weight files')
            probes = C.probes(task, spec['probe_kinds'], 'behavior_only')
            has_queries = any(spec['budget']['n_forward' if meter == 'forward' else 'n_probe'] > 0
                              for meter in probes.values())
            kind = ('no_input_weights' if not files['weights'] else
                    'weights_with_probes' if has_queries else 'weights_without_probes')
            for sample in samples:
                cases.append({'task': task, 'track': spec['track'], 'seed': seed, 'sample': sample,
                              'kind': kind, 'budget': spec['budget'], 'probes': probes, 'files': files,
                              'artifact_sha256': digest(subject),
                              'run_ids': {arm: _run_id(task, agent, model, seed, sample, arm) for arm in ARMS}})
    return {'schema_version': 1, 'created_at_unix': time.time(), 'agent': agent, 'model': model,
            'max_turns': max_turns, 'formula_version': scoring.FORMULA_VERSION,
            'manifest_version': data['version'], 'manifest_sha256': C.file_hash(snapshot),
            'harness_source_sha256': C.source_fingerprint(), 'policy_sha256': C.file_hash(Path(C.__file__)),
            'scope': data.get('scope', data['version']), 'cases': cases}


def read_plan(path: Path) -> tuple[dict, str]:
    plan = json.loads(path.read_text())
    if not isinstance(plan, dict) or plan.get('schema_version') != 1 or plan.get('formula_version') != scoring.FORMULA_VERSION:
        raise ValueError('unsupported plan schema or score formula')
    if not plan.get('cases'):
        raise ValueError('plan has no cases')
    if (not math.isfinite(plan['created_at_unix']) or plan['created_at_unix'] > time.time()
            or plan['agent'] in ('baseline', 'reference') or not plan['model'] or plan['max_turns'] <= 0):
        raise ValueError('invalid plan time or agent settings')
    seen = set()
    for case in plan['cases']:
        key = case['task'], case['seed'], case['sample']
        _indices([case['seed']])
        _indices([case['sample']])
        if key in seen or case['track'] != case['task'].split('-')[0]:
            raise ValueError('duplicate case or inconsistent track')
        has_queries = any(case['budget']['n_forward' if v == 'forward' else 'n_probe'] > 0
                          for v in case['probes'].values())
        kind = ('no_input_weights' if not case['files']['weights'] else
                'weights_with_probes' if has_queries else 'weights_without_probes')
        if case['kind'] != kind:
            raise ValueError('case classification differs from its actual weight/probe contrast')
        seen.add(key)
        expected = {arm: _run_id(case['task'], plan['agent'], plan['model'], case['seed'], case['sample'], arm)
                    for arm in ARMS}
        if case['run_ids'] != expected:
            raise ValueError('planned run IDs differ from their identities')
    return plan, C.file_hash(path)


def planned_run(path: Path, task: str, agent: str, model: str, seed: int, sample: int,
                level: int | None, condition: str, max_turns: int) -> tuple[dict, dict, str]:
    """Preflight a declared trial before reserving a run or starting an agent."""
    plan, sha = read_plan(path)
    if (condition not in ARMS or level is not None or agent != plan['agent'] or model != plan['model']
            or max_turns != plan['max_turns']):
        raise ValueError('run settings differ from the access plan')
    if C.source_fingerprint() != plan['harness_source_sha256']:
        raise ValueError('harness source differs from the access plan; declare a fresh study')
    matches = [c for c in plan['cases'] if (c['task'], c['seed'], c['sample']) == (task, seed, sample)]
    if len(matches) != 1:
        raise ValueError('run is not in the access plan')
    return plan, matches[0], sha


def validate_inputs(plan: dict, case: dict, arm: str, access: dict) -> None:
    expected = {'budget': case['budget'], 'probes': case['probes'], 'max_turns': plan['max_turns'],
                'harness_source_sha256': plan['harness_source_sha256'],
                'policy_sha256': plan['policy_sha256'], 'artifact_sha256': case['artifact_sha256'],
                'subject_weights': case['files']['weights'], 'n_weight_files': len(case['files']['weights']),
                'files': {'metadata': case['files']['metadata'],
                          'weights': case['files']['weights'] if arm == 'weights_behavior' else {}},
                'condition': arm, 'local_model_execution': False}
    # The recorded budget is the full ``asdict``; the plan's budget may have been frozen before a budget field existed.
    # Compare them with new default-valued fields dropped, so an access plan from an earlier schema still reconciles.
    access_budget = manifest.normalize_budget_dict(access.get('budget') or {})
    changed = [name for name, value in expected.items()
               if name != 'budget' and access.get(name) != value]
    if access_budget != manifest.normalize_budget_dict(expected['budget']):
        changed.append('budget')
    if changed:
        raise ValueError('planned access differs: ' + ', '.join(sorted(changed)))


def _arm(plan: dict, case: dict, arm: str, root: Path, plan_sha: str) -> dict:
    from .verify import verify_run
    directory = root / case['run_ids'][arm]
    if not directory.exists():
        return {'status': 'missing'}
    if not (directory / 'result.json').is_file():
        return {'status': 'unfinished'}
    try:
        result = json.loads((directory / 'result.json').read_text())
        if not isinstance(result, dict):
            raise ValueError('result.json must contain an object')
        expected = {k: case[k] for k in ('task', 'seed', 'sample')}
        expected.update(agent=plan['agent'], model=plan['model'], condition=arm,
                        formula_version=plan['formula_version'], run_id=directory.name, level=None)
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError('result identity/formula differs from plan')
        verified = verify_run(directory)
        if not verified.ok:
            raise ValueError('; '.join(c.detail for c in verified.checks if c.status == 'FAIL'))
        ledger = Ledger(directory / 'ledger.jsonl')
        start = next(ledger.entries('run_start'))
        access = next(ledger.entries('access_inputs'))['payload']
        if (start['payload'].get('access_plan_sha256') != plan_sha
                or access.get('access_plan_sha256') != plan_sha):
            raise ValueError('run was not attached to this access plan')
        if start['ts'] < plan['created_at_unix']:
            raise ValueError('plan postdates the run')
        validate_inputs(plan, case, arm, access)
        review = directory / 'review.json'
        cleared = review.is_file() and bool(json.loads(review.read_text()).get('cleared'))
        zero = not result['valid'] or (result.get('quarantined') and not cleared)
        scores = {metric: 0. if zero else float(result[metric]) for metric in METRICS}
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
            raise ValueError('scores must be finite and within [0, 1]')
        for field in ('task_prompt_sha256', 'agent_spec_sha256'):
            if not isinstance(access.get(field), str) or len(access[field]) != 64:
                raise ValueError(f'missing {field}')
        return {'status': 'complete', **scores, 'quarantined': bool(result.get('quarantined')),
                'valid': result['valid'], 'task_prompt_sha256': access['task_prompt_sha256'],
                'agent_spec_sha256': access['agent_spec_sha256'],
                'sha256': {p.name: manifest.sha256_file(p) for p in
                           (directory / 'result.json', directory / 'ledger.jsonl', directory / 'audit.json',
                            directory / 'transcript.jsonl', directory / 'egress.jsonl', review)}}
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
        return {'status': 'invalid_evidence', 'reason': str(exc)}


def estimate(rows: list[dict], metric: str, draws: int = 2000, seed: int = 0) -> dict:
    """Equal track/task/subject-seed weights; bootstrap pairs together within that hierarchy."""
    if draws < 2 or not rows:
        raise ValueError('estimation requires observations and at least two bootstrap draws')
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in sorted(rows, key=lambda r: (r['track'], r['task'], r['seed'], r['sample'])):
        grouped[row['track']][row['task']][row['seed']].append(row['difference'][metric])
    point = mean(mean(mean(mean(samples) for samples in seeds.values()) for seeds in tasks.values())
                 for tasks in grouped.values())
    summary = {'mean': point, 'n_pairs': len(rows), 'n_tasks': sum(len(t) for t in grouped.values()),
               'n_subject_seeds': sum(len(s) for t in grouped.values() for s in t.values())}
    if len(rows) == 1:
        return {**summary, 'ci95': None}
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        tracks = []
        for tasks in grouped.values():
            task_means = []
            for task in rng.choices(list(tasks), k=len(tasks)):
                subjects = tasks[task]
                task_means.append(mean(mean(rng.choices(subjects[s], k=len(subjects[s])))
                                       for s in rng.choices(list(subjects), k=len(subjects))))
            tracks.append(mean(task_means))
        boot.append(mean(tracks))
    boot.sort()
    return {**summary, 'ci95': [boot[int(.025 * draws)], boot[max(0, int(.975 * draws) - 1)]]}


def compare(path: Path, runs: Path, exploratory: bool = False, draws: int = 2000) -> dict:
    """Strict reports withhold estimates until every declared pair is valid and complete."""
    plan, sha = read_plan(path)
    rows = []
    for case in plan['cases']:
        arms = {arm: _arm(plan, case, arm, runs, sha) for arm in ARMS}
        row = {k: case[k] for k in ('task', 'track', 'seed', 'sample', 'kind')}
        row.update(run_ids=case['run_ids'], arms=arms, status='incomplete')
        if all(a['status'] == 'complete' for a in arms.values()):
            if any(arms[ARMS[0]][k] != arms[ARMS[1]][k] for k in ('task_prompt_sha256', 'agent_spec_sha256')):
                row.update(status='protocol_mismatch', reason='task text or agent settings differ between arms')
            else:
                row.update(status='complete', difference={k: arms[ARMS[0]][k] - arms[ARMS[1]][k] for k in METRICS})
        rows.append(row)
    complete = [r for r in rows if r['status'] == 'complete']
    adapters = {r['arms'][ARMS[0]]['agent_spec_sha256'] for r in complete}
    ready = len(complete) == len(rows) and len(adapters) <= 1
    groups = {}
    if (ready or exploratory) and len(adapters) <= 1:
        selections = {'weight_access': [r for r in complete if r['kind'] != 'no_input_weights']}
        selections.update({kind: [r for r in complete if r['kind'] == kind]
                           for kind in ('weights_with_probes', 'weights_without_probes', 'no_input_weights')})
        groups = {name: {k: estimate(selected, k, draws) for k in METRICS}
                  for name, selected in selections.items() if selected}
    return {'schema_version': 1, 'kind': 'matched weight-access comparison', 'plan_sha256': sha, 'scope': plan['scope'],
            'manifest_version': plan['manifest_version'], 'manifest_sha256': plan['manifest_sha256'],
            'harness_source_sha256': plan['harness_source_sha256'],
            'uncertainty': {'method': 'paired hierarchical bootstrap within tracks, tasks and subject seeds',
                            'draws': draws, 'seed': 0, 'single_pair_interval': None},
            'agent': plan['agent'], 'model': plan['model'], 'ready': ready,
            'exploratory': bool(exploratory and not ready), 'expected_pairs': len(rows),
            'complete_pairs': len(complete), 'pair_status': dict(Counter(r['status'] for r in rows)),
            'arm_status': dict(Counter(a['status'] for r in rows for a in r['arms'].values())),
            'adapter_configurations': len(adapters), 'estimates': groups or None, 'pairs': rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan', help='declare expected pairs; does not build subjects or run agents')
    plan.add_argument('--manifest', type=Path, required=True)
    plan.add_argument('--agent', required=True)
    plan.add_argument('--model', required=True)
    plan.add_argument('--tasks', help='comma-separated subset of manifest tasks')
    plan.add_argument('--seeds', help='comma-separated subject seeds; default: every frozen seed')
    plan.add_argument('--samples', default='0', help='comma-separated fresh sample IDs')
    plan.add_argument('--max-turns', type=int, default=200)
    plan.add_argument('--output', type=Path, required=True)
    report = commands.add_parser('report', help='verify and compare declared raw runs without executing them')
    report.add_argument('--plan', type=Path, required=True)
    report.add_argument('--runs', type=Path, default=Path('runs'))
    report.add_argument('--exploratory', action='store_true', help='allow explicitly labeled complete-pair estimates')
    report.add_argument('--draws', type=int, default=2000)
    report.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        value = make_plan(args.manifest, args.agent, args.model,
                          args.tasks.split(',') if args.tasks else None,
                          [int(s) for s in args.seeds.split(',')] if args.seeds else None,
                          [int(s) for s in args.samples.split(',')], args.max_turns)
    else:
        value = compare(args.plan, args.runs, args.exploratory, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x' if args.command == 'plan' else 'w') as stream:
        stream.write(json.dumps(value, indent=1, allow_nan=False) + '\n')
    print(f'{args.output}: ' + (f"{len(value['cases'])} planned pairs" if args.command == 'plan' else
                              f"{value['complete_pairs']}/{value['expected_pairs']} complete pairs; ready={value['ready']}"))


if __name__ == '__main__':
    main()
