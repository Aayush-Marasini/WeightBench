"""Synthetic ESC construction metering; no subject training or model math."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np
import pytest

from weightbench.harness.author import AuthorForwardMeter
from weightbench.ledger import Ledger
from weightbench.tracks.tr01_circuits_lib import build as B
from weightbench.tracks.tr01_circuits_lib import model, train


def forward():
    return model.forward({}, None, None)


def count_receipt(receipt):
    ledger = Ledger(receipt['ledger_path'])
    assert ledger.verify()
    return len(list(ledger.entries('probe')))


def test_nested_scopes_cover_preimported_alias_and_restore_parent(tmp_path, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(model, '_forward_impl', lambda *args, **kwargs: sentinel)
    alias = model.forward
    outer = AuthorForwardMeter(tmp_path / 'outer', 2, {'purpose': 'synthetic'})
    inner = AuthorForwardMeter(tmp_path / 'inner', 1, {'purpose': 'synthetic'})
    with model.forward_meter(outer):
        assert alias({}, None, None) is sentinel
        with model.forward_meter(inner):
            assert alias({}, None, None) is sentinel
        assert alias({}, None, None) is sentinel
    assert (outer.used, inner.used) == (2, 1)
    assert count_receipt(model.author_forward_receipt(outer)) == 2
    assert count_receipt(model.author_forward_receipt(inner)) == 1
    assert alias({}, None, None) is sentinel  # existing synthetic math calls stay compatible
    assert outer.used == 2 and inner.used == 1


def test_contexts_do_not_cross_charge_concurrent_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(model, '_forward_impl', lambda parameters, tokens, config, **kwargs: tokens)
    meters = [AuthorForwardMeter(tmp_path / str(index), 2, {'purpose': 'synthetic'}) for index in range(2)]
    ready = threading.Barrier(2)

    def work(index):
        with model.forward_meter(meters[index]):
            ready.wait(timeout=10)
            assert model.forward({}, index, None) == index
            assert model.forward({}, index, None) == index

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(work, index) for index in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert [meter.used for meter in meters] == [2, 2]


def test_rejected_forward_restores_scope_without_running_math(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(model, '_forward_impl', lambda *args, **kwargs: calls.append('ran'))
    empty = AuthorForwardMeter(tmp_path / 'empty', 0, {'purpose': 'synthetic'})
    with pytest.raises(RuntimeError, match='budget exhausted'):
        with model.forward_meter(empty):
            forward()
    assert calls == [] and empty.used == 0
    assert len(list(empty.ledger.entries('probe_rejected'))) == 1
    forward()
    assert calls == ['ran']


def test_default_build_nests_separate_training_and_never_reuses_ledgers(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'CACHE', tmp_path / 'cache')
    monkeypatch.setattr(model, '_forward_impl', lambda *args, **kwargs: 'synthetic')

    def fake_train(*args, **kwargs):
        forward()
        return {'w': np.ones(1)}, {'version': 'synthetic'}

    def fake_build(task_id, art, seed, **kwargs):
        forward()
        _, metadata = train.train_one('python', seed, steps=1)
        forward()
        return {'task': task_id, 'nested_training': metadata['author_construction']}

    monkeypatch.setattr(train, '_train_one', fake_train)
    monkeypatch.setattr(B, '_build', fake_build)
    art = tmp_path / 'subject'
    first = B.build('TR01-ESC-001', art, 0)
    second = B.build('TR01-ESC-001', art, 0)
    receipts = [item[key] for item in (first, second) for key in ('author_construction', 'nested_training')]
    assert [count_receipt(receipt) for receipt in receipts] == [2, 1, 2, 1]
    assert len({receipt['ledger_path'] for receipt in receipts}) == 4
    assert all(train.CACHE in Path(receipt['ledger_path']).parents for receipt in receipts)
    assert not (art / 'private').exists() and not (art / 'public').exists()


def test_get_model_uses_supplied_meter_and_cache_hit_spends_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'CACHE', tmp_path / 'cache')
    monkeypatch.setattr(model, '_forward_impl', lambda *args, **kwargs: 'synthetic')

    def fake_train(*args, **kwargs):
        forward()
        return {'w': np.arange(3)}, {'version': 'synthetic'}

    monkeypatch.setattr(train, '_train_one', fake_train)
    supplied = AuthorForwardMeter(tmp_path / 'supplied', 1, {'purpose': 'synthetic'})
    first, metadata = train.get_model('python', 0, steps=1, author_meter=supplied)
    assert supplied.used == 1
    assert metadata['author_construction']['ledger_path'] == str(supplied.ledger.path.resolve())
    cache_path = train.model_paths('python', 0)[1]
    frozen_metadata = cache_path.read_bytes()
    unused = AuthorForwardMeter(tmp_path / 'unused', 0, {'purpose': 'synthetic'})
    second, again = train.get_model('python', 0, steps=1, author_meter=unused)
    assert unused.used == 0 and cache_path.read_bytes() == frozen_metadata
    assert again == metadata
    np.testing.assert_array_equal(first['w'], second['w'])

