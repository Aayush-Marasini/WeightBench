"""Frozen reference decisions depend on public statistics and global calibration only."""
import numpy as np
import pytest

from weightbench.tracks.tr01_circuits_lib import reference as R


def test_language_labels_follow_v_reader_entropy_rank(monkeypatch):
    monkeypatch.setattr(R.S, 'language_spectral_features', lambda model: {'v_entropy_rank': model['rank']})
    models = {'A': {'rank': 3}, 'B': {'rank': 1}, 'C': {'rank': 4}, 'D': {'rank': 2}}
    assert R.language_report(models) == {'B': 'romaji', 'D': 'english', 'A': 'german', 'C': 'python'}


def test_induction_threshold_excludes_unavailable_sources(monkeypatch):
    monkeypatch.setattr(R.S, 'signed_composition_features', lambda *args, **kwargs: {
        'available': np.array([[False, True], [True, True]]),
        'score': np.array([[1., .4], [.39, -.8]])})
    assert R.induction_heads({}, .4) == [(0, 1)]


def parameters():
    return {'feature_names': ['b', 'a'], 'mean': [2., 1.], 'scale': [2., 4.],
            'coefficients': [.1, 1., -2.], 'threshold': .5}


def test_previous_classifier_uses_frozen_feature_order_and_standardization(monkeypatch):
    monkeypatch.setattr(R.S, 'reference_head_features', lambda _: {
        'a': np.array([[1., 5.]]), 'b': np.array([[4., 4.]])})
    # Scores: .1 + (4-2)/2 - 2*(1-1)/4 = 1.1; second head = -.9.
    assert R.previous_heads({}, parameters()) == [(0, 0)]


@pytest.mark.parametrize('field,value', [('scale', [0., 4.]), ('mean', [float('nan'), 1.]),
                                       ('coefficients', [.1, 1.])])
def test_invalid_frozen_calibration_rejected(monkeypatch, field, value):
    monkeypatch.setattr(R.S, 'reference_head_features', lambda _: {'a': np.ones((1, 2)), 'b': np.ones((1, 2))})
    prior = parameters()
    prior[field] = value
    with pytest.raises(ValueError, match='calibration'):
        R.previous_heads({}, prior)
