"""Frozen-candidate ESC inference: public matrices and seed-independent calibration parameters."""
from __future__ import annotations

import numpy as np

from . import static as S


def language_report(models: dict[str, dict[str, np.ndarray]]) -> dict[str, str]:
    """Entropy effective rank of the first V readers; MLP inputs are not embedding-only."""
    order = sorted(models, key=lambda letter: S.language_spectral_features(models[letter])['v_entropy_rank'])
    return dict(zip(order, ('romaji', 'english', 'german', 'python')))


def induction_heads(P: dict[str, np.ndarray], threshold: float) -> list[tuple[int, int]]:
    features = S.signed_composition_features(P, lags=(59,))
    return [tuple(map(int, pair)) for pair in np.argwhere(features['available'] & (features['score'] >= threshold))]


def previous_heads(P: dict[str, np.ndarray], parameters: dict) -> list[tuple[int, int]]:
    features = S.reference_head_features(P)
    names = parameters['feature_names']
    matrix = np.stack([features[name] for name in names], axis=-1)
    mean, scale = np.asarray(parameters['mean']), np.asarray(parameters['scale'])
    coefficients = np.asarray(parameters['coefficients'])
    if (matrix.shape[-1] != len(mean) or scale.shape != mean.shape or len(coefficients) != len(mean) + 1
            or not np.isfinite(np.r_[mean, scale, coefficients]).all() or np.any(scale <= 0)):
        raise ValueError('invalid frozen ESC calibration parameters')
    score = coefficients[0] + ((matrix - mean) / scale) @ coefficients[1:]
    return [tuple(map(int, pair)) for pair in np.argwhere(score >= parameters['threshold'])]
