"""Exploratory ESC statistics use static algebra and respect the released symmetries."""
import numpy as np
import pytest

from weightbench.tracks.tr01_circuits_lib import static as S


def blocks(seed=7, zero=False):
    rng = np.random.default_rng(seed)
    result = {}
    for l in range(3):
        prefix = f'blocks.{l}.'
        for key, shape in (('attn.W_Q', (2, 4, 8)), ('attn.W_K', (2, 4, 8)),
                           ('attn.W_V', (2, 4, 8)), ('attn.W_O', (2, 8, 4)),
                           ('mlp.W_in', (12, 8)), ('mlp.W_out', (8, 12))):
            result[prefix + key] = np.zeros(shape) if zero else rng.normal(scale=.2, size=shape)
    return result


def rotate_and_permute(P):
    rng = np.random.default_rng(41)
    rotation = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    transformed, permutations = {}, []
    for l in range(3):
        prefix = f'blocks.{l}.'
        heads, neurons = rng.permutation(2), rng.permutation(12)
        permutations.append(heads)
        for key in ('W_Q', 'W_K', 'W_V'):
            transformed[prefix + 'attn.' + key] = P[prefix + 'attn.' + key][heads] @ rotation.T
        transformed[prefix + 'attn.W_O'] = np.einsum('ij,hjk->hik', rotation, P[prefix + 'attn.W_O'][heads])
        transformed[prefix + 'mlp.W_in'] = P[prefix + 'mlp.W_in'][neurons] @ rotation.T
        transformed[prefix + 'mlp.W_out'] = rotation @ P[prefix + 'mlp.W_out'][:, neurons]
    return transformed, permutations


def test_language_features_have_hand_computed_rank_one_and_zero_limits():
    P = blocks(zero=True)
    for key in ('W_Q', 'W_K', 'W_V'):
        P['blocks.0.attn.' + key][0, 0] = 1
    features = S.language_spectral_features(P)
    for key in ('qkv_pr', 'qkv_entropy_rank', 'qkv_excess_pr', 'qkv_top1'):
        assert features[key] == pytest.approx(1.)
    assert features['mlp_pr'] == 0
    assert all(np.isfinite(v) for v in S.language_spectral_features(blocks(zero=True)).values())


def test_language_and_positional_features_are_basis_invariant_and_head_equivariant():
    P = blocks()
    transformed, permutations = rotate_and_permute(P)
    original = S.language_spectral_features(P)
    changed = S.language_spectral_features(transformed)
    assert changed == pytest.approx(original, abs=1e-11)
    original, changed = S.positional_features(P), S.positional_features(transformed)
    assert not original['mean_available'][0].any()
    assert original['mean_available'][1:].all()
    for key in original:
        for l, permutation in enumerate(permutations):
            np.testing.assert_allclose(changed[key][l], original[key][l, permutation], atol=1e-11)
    assert np.all((original['trace_prob1'] >= 0) & (original['trace_prob1'] <= 1))


def test_composition_sign_depends_on_query_alignment():
    P = blocks(zero=True)
    eye = np.eye(4, 8)
    P['blocks.0.attn.W_V'][0] = eye
    P['blocks.0.attn.W_O'][0] = eye.T
    P['blocks.1.attn.W_Q'][0] = eye
    P['blocks.1.attn.W_K'][0] = eye
    positive = S.signed_composition_features(P, lags=(0,))
    assert positive['score'][1, 0] == pytest.approx(1.)
    assert positive['raw_score'][1, 0] == pytest.approx(2.)
    assert positive['source_layer'][1, 0] == positive['source_head'][1, 0] == 0
    assert not positive['available'][0].any()
    P['blocks.1.attn.W_Q'][0] *= -1
    negative = S.signed_composition_features(P, lags=(0,))
    assert negative['score'][1, 0] == pytest.approx(-1.)
    assert negative['raw_score'][1, 0] == pytest.approx(-2.)


@pytest.mark.parametrize('low_frequency_pairs', [None, 1])
def test_composition_is_rotation_invariant_and_head_equivariant(low_frequency_pairs):
    P = blocks()
    transformed, permutations = rotate_and_permute(P)
    original = S.signed_composition_features(P, lags=(58, 59, 60), low_frequency_pairs=low_frequency_pairs)
    changed = S.signed_composition_features(transformed, lags=(58, 59, 60), low_frequency_pairs=low_frequency_pairs)
    for l, permutation in enumerate(permutations):
        for key in ('score', 'raw_score', 'max_raw_score', 'available'):
            np.testing.assert_allclose(changed[key][l], original[key][l, permutation], atol=1e-11)
        for l1, source_permutation in enumerate(permutations):
            expected = original['pair_scores'][l, permutation, l1][:, source_permutation]
            np.testing.assert_allclose(changed['pair_scores'][l, :, l1], expected, atol=1e-11)


def test_invalid_diagnostic_settings_are_rejected():
    with pytest.raises(ValueError, match='max_delta'):
        S.positional_features(blocks(), max_delta=1)
    with pytest.raises(ValueError, match='lags'):
        S.signed_composition_features(blocks(), lags=())
    with pytest.raises(ValueError, match='rotary pairs'):
        S.signed_composition_features(blocks(), low_frequency_pairs=3)


def test_gaussian_gelu_variance_matches_hand_computed_standard_normal():
    expected = 1 / 3 + 1 / (2 * np.pi * np.sqrt(3)) - 1 / (4 * np.pi)
    np.testing.assert_allclose(S._gaussian_gelu_variance(np.array([0., 1.])), [0., expected])


def test_covariance_candidates_are_basis_invariant_and_head_equivariant():
    P = blocks()
    transformed, permutations = rotate_and_permute(P)
    original = S.covariance_positional_features(P)
    changed = S.covariance_positional_features(transformed)
    assert not original['writer_available'][0].any()
    assert original['writer_available'][1:].all()
    for key in original:
        for l, permutation in enumerate(permutations):
            np.testing.assert_allclose(changed[key][l], original[key][l, permutation], atol=1e-11)
        if key.endswith('_prob1'):
            assert np.all((original[key] >= 0) & (original[key] <= 1))


def test_zero_covariance_inputs_have_exact_uniform_causal_attention():
    result = S.covariance_positional_features(blocks(zero=True))
    uniform = sum(1 / n for n in range(2, 129)) / 127
    for prefix in ('reader_aligned', 'reader_inverse', 'writer'):
        np.testing.assert_allclose(result[prefix + '_prob1'], uniform)
        assert not result[prefix + '_margin'].any()
    assert not result['writer_available'].any()
    with pytest.raises(ValueError, match='ridge'):
        S.covariance_positional_features(blocks(), ridge=0.)
