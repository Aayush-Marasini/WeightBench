"""Partial local bases preserve Transformer logits and released head identities."""
import numpy as np
import pytest
import torch
import json

from weightbench import runner
from weightbench.tracks import tr01_circuits as T
from weightbench.tracks.tr01_circuits_lib import build as B, model as M, train
from weightbench.tracks.tr01_circuits_lib.model import (
    Config, forward, init_params, scrambled_block_count, to_numpy, to_torch, transform,
)


@pytest.mark.parametrize('style', [1, 2, 3])
@pytest.mark.parametrize('sites', [(), (1,), (0, 2), (0, 1, 2, 3)])
def test_partial_bases_preserve_logits_and_attention(style, sites):
    cfg = Config(d_model=16, n_heads=2, n_layers=4, d_mlp=24, ctx=16)
    base = to_numpy(init_params(cfg, 13))
    rng = np.random.default_rng(4)
    for name, values in base.items():
        if name.endswith('.g'):
            values[:] = rng.uniform(.5, 1.5, values.shape)
    changed, info = transform(base, cfg, style, np.random.default_rng(7), sites)
    assert info['permuted_layers'] == list(sites)
    for layer in set(range(cfg.n_layers)) - set(sites):
        assert info['head_perm'][layer] == list(range(cfg.n_heads))
    tokens = torch.tensor(rng.integers(0, cfg.vocab, size=(2, 12)))
    before, attn0 = forward(to_torch(base), tokens, cfg, return_attn=True)
    after, attn1 = forward(to_torch(changed), tokens, cfg, return_attn=True)
    torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)
    for layer in range(cfg.n_layers):
        torch.testing.assert_close(attn1[layer], attn0[layer][:, info['head_perm'][layer]], atol=1e-5, rtol=1e-4)
    # All selected blocks matches the full-model transform's values and RNG sequence.
    if len(sites) == cfg.n_layers:
        legacy, old_info = transform(base, cfg, style, np.random.default_rng(7))
        assert old_info['head_perm'] == info['head_perm']
        for name in legacy:
            np.testing.assert_array_equal(legacy[name], changed[name])


@pytest.mark.parametrize('sites', [(0, 0), (-1,), (4,), (True,), ('1',)])
def test_partial_bases_reject_invalid_sites(sites):
    cfg = Config(n_layers=4)
    with pytest.raises(ValueError, match='unique valid'):
        transform({}, cfg, 2, np.random.default_rng(0), sites)


def test_shared_quantity_and_physical_capacity():
    assert [scrambled_block_count(Config(), l) for l in range(3)] == [1, 2, 4]
    with pytest.raises(ValueError, match='subject has 6'):
        scrambled_block_count(Config(), 3)
    with pytest.raises(ValueError, match='subject has 2'):
        scrambled_block_count(Config(n_layers=2), 2)
    for invalid in (True, -1, 1.5):
        with pytest.raises(ValueError):
            scrambled_block_count(Config(), invalid)


def test_global_rotation_and_overlapping_local_maps_are_fixed():
    cfg = Config(d_model=16, n_heads=2, n_layers=4, d_mlp=24)
    base = to_numpy(init_params(cfg, 19))
    clean, _ = transform(base, cfg, 2, np.random.default_rng(5), ())
    full, _ = transform(base, cfg, 2, np.random.default_rng(5))
    for sites in ((2,), (0, 2), (0, 1, 2, 3)):
        changed, _ = transform(base, cfg, 2, np.random.default_rng(5), sites)
        np.testing.assert_array_equal(changed['embed.W_E'], full['embed.W_E'])
        for name, values in changed.items():
            if name.startswith('blocks.'):
                layer = int(name.split('.')[1])
                np.testing.assert_array_equal(values, (full if layer in sites else clean)[name])


def test_task_rejects_capacity_before_ensuring_models(tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'default_config', lambda: Config(n_layers=2))
    monkeypatch.setattr(T, 'ensure_models', lambda *a: pytest.fail('cache loading/training started'))
    with pytest.raises(ValueError, match='subject has 2'):
        T.LanguageID().at_level(2).build(tmp_path, 0)
    for cls in (T.LanguageID, T.InductionHeads, T.PrevTokenHeads):
        with pytest.raises(ValueError, match='supported knob levels'):
            cls().at_level(3)
        assert cls().knob_level is None and runner.nondefault_level(cls.id, 2) == 2


def test_real_builder_uses_nested_sites_without_public_transform_maps(tmp_path, monkeypatch):
    cfg = Config(d_model=16, n_heads=2, n_layers=4, d_mlp=24, ctx=32)
    monkeypatch.setattr(M, 'default_config', lambda: cfg)
    monkeypatch.setattr(train, '_device', lambda: torch.device('cpu'))
    params = {lang: to_numpy(init_params(cfg, 31 + i)) for i, lang in enumerate(B.LANGS)}
    monkeypatch.setattr(train, 'get_model', lambda lang, seed, steps=None:
                        (params[lang], {'sha256_npz': 'fixture', 'val_loss_nats_per_byte': 1.0}))
    data = np.random.default_rng(9).integers(0, cfg.vocab, 4096, dtype=np.uint8)
    monkeypatch.setattr(train, 'load_bytes', lambda lang: (data, data))
    for name, value in (('IND_T', 4), ('IND_NSEQ', 2), ('PREV_T', 8), ('PREV_NSEQ', 2)):
        monkeypatch.setattr(B, name, value)
    builds = {}
    for level in (None, 0, 1, 2):
        art = tmp_path / str(level)
        B.build('TR01-ESC-001', art, 0, level=level)
        metadata = json.loads((art / 'private/meta.json').read_text())
        builds[level] = (art, metadata)
        if level is None:
            assert not (art / 'public/knob.json').exists()
        else:
            knob = json.loads((art / 'public/knob.json').read_text())
            assert knob['value'] == 2**level and knob['global_residual_rotation'] is True
            assert 'permuted_layers' not in knob
            assert f'permutations is {2**level}' in T.LanguageID().prompt(art)
        for path in (art / 'public').glob('*.npz'):
            with np.load(path) as arrays:
                assert all(n.startswith('blocks.') and not n.endswith('.g') for n in arrays.files)
    for label in 'ABCD':
        sets = [set(builds[l][1]['transforms'][label]['permuted_layers']) for l in (0, 1, 2)]
        assert [len(s) for s in sets] == [1, 2, 4] and sets[0] < sets[1] < sets[2]
        with np.load(builds[None][0] / f'private/full_{label}.npz') as original:
            for level in (0, 1, 2):
                with np.load(builds[level][0] / f'private/full_{label}.npz') as changed:
                    np.testing.assert_array_equal(original['embed.W_E'], changed['embed.W_E'])
                    if level == 2:   # four physical blocks: full local transform matches legacy exactly
                        for name in original.files:
                            np.testing.assert_array_equal(original[name], changed[name])
