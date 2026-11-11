import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe import tablemoe_builder as candidate
from freetoken.moe.tablemoe_artifacts import read_json


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'reference_builder', ROOT / 'runtime_snapshot_compact/python/freetoken/moe/tablemoe_builder.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def collect(root, first, second, storage, dtype, reservoir):
    config = dict(raw_storage=storage, dtype=str(dtype).removeprefix('torch.'),
                  export_inputs=False, reservoir_size=reservoir, seed=42)
    collector = first.InputCollector(root, config, 8, 4)
    generator = torch.Generator().manual_seed(193)
    for uid in range(3):
        if uid == 1:
            collector.close()
            collector = second.InputCollector(root, config, 8, 4)
        for phase, count in [('prefill', 19), ('decode', 1), ('decode', 1), ('decode', 1)]:
            for layer in range(2):
                x = torch.randn(count, 8, generator=generator).to(dtype)
                ids = torch.stack([torch.arange(count) % 3, (torch.arange(count) + 1) % 3,
                                   torch.arange(count) % 3], dim=1)
                modality = torch.arange(count).to(torch.int8) % 2 if phase == 'prefill' else torch.ones(count, dtype=torch.int8)
                collector.add(layer, x, ids, modality, phase)
        collector.flush(uid)
        progress = read_json(root / 'progress.json')
        for key, state in progress['buckets'].items():
            suffix, width = ('index', 8) if storage == 'indexed' else ('bin', 16)
            folder = root / f'layer_{state["layer"]}'
            assert (folder / f'{key}.{suffix}').stat().st_size == state['kept'] * width
            assert (folder / f'{key}.phase').stat().st_size == state['kept']
        for key, state in progress.get('token_banks', {}).items():
            assert (root / f'layer_{state["layer"]}' / f'{key}.bin').stat().st_size == state['rows'] * 16
    collector.close()
    second.finalize_collection(root, {'sample_count': 3}, 2, 4)
    return read_json(root / 'progress.json')


@pytest.mark.parametrize('storage', ['dense', 'indexed'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('reservoir', [3, 30000])
def test_buffered_raw_phase_reservoir_and_existing_boundary_resume_match(tmp_path, storage, dtype, reservoir):
    expected = collect(tmp_path / 'old', reference, reference, storage, dtype, reservoir)
    for name, first in [('new', candidate), ('migrated', reference)]:
        actual = collect(tmp_path / name, first, candidate, storage, dtype, reservoir)
        assert expected == actual
        paths = sorted((tmp_path / 'old').glob('layer_*/*'))
        assert paths
        for path in paths:
            assert path.read_bytes() == (tmp_path / name / path.relative_to(tmp_path / 'old')).read_bytes()


def planned(root, implementation, storage, per_sample_seed, restart):
    config = dict(raw_storage=storage, sampling_policy='tablemoe_default', sample_count=3,
        sample_decode_tokens=5, collection_image_ids=[['a', 'b'], ['b', 'c'], ['c']],
        min_vision_per_sample=2, max_vision_per_sample=3, min_text_per_sample=3,
        max_text_per_sample=5, per_sample_seed=per_sample_seed, reservoir_size=3,
        export_inputs=False)
    collector = implementation.InputCollector(root, config, 4, 2)
    for ordinal in range(3):
        if ordinal == 1 and restart:
            collector.close()
            collector = candidate.InputCollector(root, config, 4, 2)
        uid = ordinal if not restart or not ordinal else ordinal - 1
        prompt = torch.arange(12) + ordinal * 50
        spans = [(1, 4), (7, 10)] if ordinal < 2 else [(1, 4)]
        request = SimpleNamespace(uid=uid, input_ids=prompt, cached_len=0, extend_len=0,
            mm_items=[SimpleNamespace(offsets=[span]) for span in spans])
        lo = 0
        for hi in (3, 8, 12):
            request.cached_len, request.extend_len = lo, hi-lo
            batch = SimpleNamespace(is_prefill=True, padded_reqs=[request],
                prompt_admissions=[(uid, 12, 0)] if lo == 0 else [],
                prompt_input_ids={uid: prompt} if lo == 0 else {})
            for layer in range(2):
                collector.snapshot(layer, torch.arange(lo*4, hi*4).reshape(-1,4).half() + ordinal,
                    torch.tensor([[0,1]]).repeat(hi-lo,1), batch)
            lo = hi
        for step in range(5):
            request.cached_len = 12+step
            batch = SimpleNamespace(is_prefill=False, padded_reqs=[request])
            for layer in range(2):
                collector.snapshot(layer, torch.full((1,4), ordinal+step).half(), torch.tensor([[0,1]]), batch)
        collector.flush(uid)
    collector.close()
    candidate.finalize_collection(root, {'sample_count': 3}, 2, 2)
    p = read_json(root / 'progress.json')
    p['requests'] = list(range(3))
    for i, record in enumerate(p['sample_state']['records']):
        record['uid'] = i
    return p


@pytest.mark.parametrize('storage', ['dense', 'indexed'])
@pytest.mark.parametrize('per_sample_seed', [False, True])
def test_sample_planner_dedup_chunked_prefill_and_reset_uid_resume(tmp_path, storage, per_sample_seed):
    expected = planned(tmp_path/'old', reference, storage, per_sample_seed, False)
    actual = planned(tmp_path/'new', reference, storage, per_sample_seed, True)
    assert expected == actual
    for path in (tmp_path/'old').glob('layer_*/*'):
        assert path.read_bytes() == (tmp_path/'new'/path.relative_to(tmp_path/'old')).read_bytes()


def test_flush_error_does_not_publish_new_request(tmp_path, monkeypatch):
    collector = candidate.InputCollector(tmp_path, {'export_inputs': False}, 4, 1)
    x = torch.ones(2,4).half()
    collector.add(0, x, torch.zeros(2,1,dtype=torch.long), torch.ones(2,dtype=torch.int8), 'prefill')
    collector.flush(0)
    previous = read_json(tmp_path/'progress.json')
    collector.add(0, x, torch.zeros(2,1,dtype=torch.long), torch.ones(2,dtype=torch.int8), 'decode')
    writer = next(iter(collector.handles.values()))[0]
    with monkeypatch.context() as patch:
        patch.setattr(writer, 'flush', lambda: (_ for _ in ()).throw(OSError('flush failed')))
        with pytest.raises(OSError, match='flush failed'):
            collector.flush(1)
        assert read_json(tmp_path/'progress.json') == previous
        assert collector.requests == [0]
    collector.close()
