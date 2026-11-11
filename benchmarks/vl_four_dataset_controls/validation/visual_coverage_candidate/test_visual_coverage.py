import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load_module('coverage_builder', HERE / 'tablemoe_builder.py')
reference = load_module('frozen_buffered_builder', ROOT / 'runtime_snapshot_buffered/python/freetoken/moe/tablemoe_builder.py')


def config_for(images, **overrides):
    return dict(sampling_policy='tablemoe_default', collection_image_ids=images,
        sample_count=len(images), sample_decode_tokens=0, raw_storage='indexed',
        export_inputs=False, min_vision_per_sample=1, max_vision_per_sample=1000,
        min_text_per_sample=0, max_text_per_sample=0, max_text_tokens=0,
        max_vision_tokens=20, per_sample_seed=True, **overrides)


def collect(root, module, images, config, restart=None):
    collector = module.InputCollector(root, config, 4, 2, identity='fixture')
    try:
        for ordinal, keys in enumerate(images):
            if ordinal == restart:
                collector.close()
                collector = module.InputCollector(root, config, 4, 2, identity='fixture')
            spans = [(1 + i * 4, 4 + i * 4) for i in range(len(keys))]
            length = 4 * len(keys) + 2
            uid = 100 + ordinal
            prompt = torch.arange(length) + ordinal * 100
            req = SimpleNamespace(uid=uid, input_ids=prompt, cached_len=0, extend_len=0,
                mm_items=[SimpleNamespace(offsets=[span]) for span in spans])
            lo = 0
            for hi in (2, length):
                req.cached_len, req.extend_len = lo, hi - lo
                batch = SimpleNamespace(is_prefill=True, padded_reqs=[req],
                    prompt_admissions=[(uid, length, 0)] if lo == 0 else [],
                    prompt_input_ids={uid: prompt} if lo == 0 else {})
                for layer in range(2):
                    collector.snapshot(layer,
                        torch.arange(lo * 4, hi * 4).reshape(-1, 4).half() + ordinal + layer,
                        torch.tensor([[0, 1]]).repeat(hi - lo, 1), batch)
                lo = hi
            collector.flush(uid)
    finally:
        collector.close()
    manifest = module.finalize_collection(root, {'sample_count': len(images)}, 2, 2)
    return json.loads((root / 'progress.json').read_text()), manifest


def assert_same_inputs(a, b, progress):
    for key, state in progress['buckets'].items():
        left = candidate.load_inputs(a, key, state, 4, prefer_exported=False)
        right = candidate.load_inputs(b, key, state, 4, prefer_exported=False)
        assert torch.equal(left, right)
        relative = Path(f'layer_{state["layer"]}/{key}.phase')
        assert (a / relative).read_bytes() == (b / relative).read_bytes()


def test_two_images_quota_two_covers_both(tmp_path):
    images = [['a', 'b']]
    config = config_for(images)
    config.update(max_vision_tokens=2, min_vision_tokens_per_image=1)
    progress, manifest = collect(tmp_path, candidate, images, config)
    assert progress['sample_state']['records'][0]['selected_visual_tokens_per_image'] == [1, 1]
    assert progress['sample_state']['counts'] == {'vision': 2, 'text': 0}
    assert progress['tokens']['decode'] == 0
    assert manifest['image_coverage'] == dict(expected_images=2, covered_images=2,
        min_tokens_per_image=1, selected_visual_tokens=2)


def test_future_reserve_limits_large_current_quota():
    rng = np.random.default_rng(42)
    positions = np.arange(100)
    chosen = candidate.image_coverage_selection(positions, [positions], 100, 1, 5, 4, 1000, rng)
    assert len(chosen) == 1


def test_transitive_shared_images_and_future_minima(tmp_path):
    images = [['a', 'b'], ['b', 'c'], ['c', 'd'], ['a', 'd']]
    config = config_for(images)
    config.update(max_vision_tokens=4, min_vision_tokens_per_image=1)
    progress, manifest = collect(tmp_path, candidate, images, config)
    records = progress['sample_state']['records']
    assert [r['selected_visual_tokens_per_image'] for r in records] == [[1, 1], [0, 1], [0, 1], [0, 0]]
    assert [r['duplicate_visual_tokens'] for r in records] == [0, 3, 3, 6]
    assert manifest['image_coverage']['covered_images'] == 4


@pytest.mark.parametrize('change, message', [
    ({'max_vision_tokens': 2}, 'global visual budget'),
    ({'max_vision_per_sample': 1}, 'per-sample visual budget'),
    ({'min_vision_tokens_per_image': -1}, 'nonnegative integer'),
    ({'min_vision_tokens_per_image': True}, 'nonnegative integer'),
])
def test_infeasible_configuration_rejected(change, message):
    config = config_for([['a', 'b'], ['c']])
    config.update(min_vision_tokens_per_image=1)
    config.update(change)
    with pytest.raises(ValueError, match=message):
        candidate.algorithm_config(config)


def test_minimum_requires_identities_and_actual_positions():
    with pytest.raises(ValueError, match='ordered collection image identities'):
        candidate.algorithm_config({'min_vision_tokens_per_image': 1})
    with pytest.raises(ValueError, match='fewer visual positions'):
        candidate.image_coverage_selection(np.arange(2), [np.arange(0), np.arange(2)],
            2, 1, 2, 0, 1000, np.random.default_rng(42))


def test_duplicate_only_history_cannot_spend_future_minima(tmp_path):
    images = [['a'], ['a'], ['a'], ['b', 'c'], ['d', 'd']]
    config = config_for(images)
    config.update(min_vision_tokens_per_image=1, max_vision_tokens=4)
    progress, _ = collect(tmp_path, candidate, images, config)
    assert [r['selected_visual_tokens_per_image'] for r in progress['sample_state']['records']] == [
        [1], [0], [0], [1, 1], [1, 0]]


@pytest.mark.parametrize('storage', ['dense', 'indexed'])
@pytest.mark.parametrize('per_sample_seed', [False, True])
def test_default_selection_and_input_values_remain_identical(tmp_path, storage, per_sample_seed):
    images = [['a', 'b'], ['b', 'c'], ['c']]
    config = config_for(images)
    config.update(raw_storage=storage, per_sample_seed=per_sample_seed, max_vision_tokens=5,
        min_text_per_sample=1, max_text_per_sample=3, max_text_tokens=8)
    old, old_manifest = collect(tmp_path / 'reference', reference, images, config)
    new, new_manifest = collect(tmp_path / 'candidate', candidate, images, config)
    assert old == new
    assert 'min_vision_tokens_per_image' not in new['config']
    assert 'image_coverage' not in old_manifest and 'image_coverage' not in new_manifest
    assert_same_inputs(tmp_path / 'reference', tmp_path / 'candidate', old)


@pytest.mark.parametrize('storage', ['dense', 'indexed'])
@pytest.mark.parametrize('per_sample_seed', [False, True])
def test_opt_in_resume_preserves_rng_and_samples(tmp_path, storage, per_sample_seed):
    images = [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d']]
    config = config_for(images)
    config.update(raw_storage=storage, per_sample_seed=per_sample_seed,
        min_vision_tokens_per_image=1, max_vision_tokens=9)
    continuous, _ = collect(tmp_path / 'continuous', candidate, images, config)
    resumed, _ = collect(tmp_path / 'resumed', candidate, images, config, restart=2)
    assert continuous == resumed
    assert_same_inputs(tmp_path / 'continuous', tmp_path / 'resumed', continuous)
    changed = dict(config, min_vision_tokens_per_image=2)
    with pytest.raises(ValueError, match='resume configuration mismatch'):
        candidate.InputCollector(tmp_path / 'resumed', changed, 4, 2, identity='fixture')


def test_finalize_rejects_uncovered_image_record(tmp_path):
    images = [['a', 'b']]
    config = config_for(images)
    config.update(min_vision_tokens_per_image=1, max_vision_tokens=2)
    progress, _ = collect(tmp_path, candidate, images, config)
    progress['sample_state']['records'][0]['selected_visual_tokens_per_image'] = [0, 2]
    (tmp_path / 'progress.json').write_text(json.dumps(progress))
    with pytest.raises(ValueError, match='uncovered image identities'):
        candidate.finalize_collection(tmp_path, {'sample_count': 1}, 2, 2)
