"""CPU validation of indexed modality union and the native clustering reader."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNAPSHOT = ROOT / 'runtime_snapshot_compact'
WORK = HERE / f'cpu_union_{time.time_ns()}'
WORK.mkdir(parents=True)
os.environ['TMPDIR'] = str(WORK)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(HERE), str(SNAPSHOT / 'python')]

import numpy as np
import torch

import merge_modalities as merge
from freetoken.moe.tablemoe_artifacts import digest, read_json, verify_stage, write_json
from freetoken.moe.tablemoe_builder import algorithm_config, cluster, finalize_collection, load_inputs, materialize


def write_rows(path, values):
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


CAL = [dict(id=f'cal{i}', image_ids=[f'image{i // 2}'], group_id=f'group{i // 2}', text=f'question{i}')
       for i in range(3)]
EVAL = [dict(id='heldout', image_ids=['heldout_image'], group_id='heldout_group', text='heldout')]
CAL_PATH, EVAL_PATH = WORK / 'calibration.jsonl', WORK / 'evaluation.jsonl'
write_rows(CAL_PATH, CAL)
write_rows(EVAL_PATH, EVAL)


def source(path, values, visual=False):
    path.mkdir(parents=True)
    config = algorithm_config(dict(seed=42, dtype='float16', raw_storage='indexed', export_inputs=False,
        device='cpu', pca_dim=1, clusters=2, reservoir_size=30, iterations=2, pca_per_expert=10,
        sampling_policy='tablemoe_default', sample_count=len(values), sample_decode_tokens=0 if visual else 127,
        min_text_per_sample=0 if visual else 1, max_text_per_sample=2,
        max_text_tokens=0 if visual else 100, max_vision_tokens=100,
        collection_image_ids=[value['image_ids'] for value in values],
        **(dict(min_vision_tokens_per_image=1) if visual else {})))
    inputs = dict(checkpoint=dict(model='cpu_fixture'), convert_bf16=True, processor=dict(fixture=True),
        calibration_ids=[value['id'] for value in values], calibration_sha256=digest(values),
        sample_count=len(values), algorithm=config)
    records, requests, seen = [], [], set()
    for ordinal, value in enumerate(values):
        new = [image for image in value['image_ids'] if image not in seen]
        vision = [2, 3] if new and visual else ([2] if new else [])
        text = [] if visual else [0, 1]
        record = dict(uid=ordinal, ordinal=ordinal, prompt_tokens=9, decode_seen=config['sample_decode_tokens'],
            selected=dict(vision=len(vision), text=len(text)), positions=dict(vision=vision, text=text),
            candidates=dict(vision=2 if new else 0, text=9 - 2 + config['sample_decode_tokens']),
            image_ids=value['image_ids'], new_image_ids=new, image_offsets=[[[2, 4]]],
            selected_visual_tokens_per_image=[len(vision)], duplicate_visual_tokens=0 if new else 2)
        seen.update(value['image_ids'])
        records.append(record)
        requests.append(dict(id=value['id'], usage=dict(prompt_tokens=9, completion_tokens=config['sample_decode_tokens'] + 1),
            max_tokens=config['sample_decode_tokens'] + 1, natural_eos=False,
            finish_reason='length', response_id=f'fixture-{ordinal}'))
    counts = {mod: sum(record['selected'][mod] for record in records) for mod in ('vision', 'text')}
    progress = dict(config=config, identity=digest(inputs), hidden=3, requests=list(range(len(values))),
        tokens=dict(prefill=len(values) * 9, decode=len(values) * config['sample_decode_tokens']),
        sample_state=dict(records=records, counts=counts), buckets={}, token_banks={}, sampling={},
        batch_sizes=dict(prefill={'1': len(values)}, decode={'1': len(values) * config['sample_decode_tokens']}))
    for layer in range(2):
        folder = path / f'layer_{layer}'
        folder.mkdir()
        progress['sampling'][str(layer)] = {}
        for mod in ('vision', 'text'):
            n = counts[mod]
            progress['sampling'][str(layer)][mod] = dict(selected=n, prefill=n, decode=0,
                candidates=sum(record['candidates'][mod] for record in records))
            if not n:
                continue
            bank = f'L{layer}_{mod}_tokens'
            data = np.arange(n * 3, dtype=np.float16).reshape(n, 3) + layer + (100 if visual else 0)
            data.tofile(folder / f'{bank}.bin')
            progress['token_banks'][bank] = dict(layer=layer, modality=mod, rows=n)
            for expert in range(8):
                key = f'L{layer}_E{expert}_{mod}'
                np.arange(n, dtype=np.int64).tofile(folder / f'{key}.index')
                (folder / f'{key}.phase').write_bytes(bytes(n))
                progress['buckets'][key] = dict(layer=layer, expert=expert, modality=mod,
                    seen=n, kept=n, prefill=n, decode=0, raw_storage='indexed', bank=bank)
    write_json(path / 'progress.json', progress)
    write_rows(path / 'requests.jsonl', requests)
    finalize_collection(path, inputs, 2, 8)
    return inputs


def invoke(main, visual, output, **kwargs):
    return merge.merge_modalities(main, visual, output, kwargs.pop('calibration', CAL_PATH),
        kwargs.pop('evaluation', EVAL_PATH), snapshot=SNAPSHOT, expected_samples=3, expected_images=2,
        num_layers=2, num_experts=8, hidden=3, **kwargs)


def main():
    checks = []
    text, vision = WORK / 'main/collect', WORK / 'visual/collect'
    source(text, CAL)
    source(vision, [CAL[0], CAL[2]], visual=True)
    before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
              for root in (text, vision) for path in root.rglob('*') if path.is_file()}
    for mode in ('hardlink', 'symlink'):
        assets = WORK / mode
        manifest = invoke(text, vision, assets / 'collect', link_mode=mode)
        assert manifest['collection_kind'] == 'modality_union' and not manifest['ordinary_single_pass']
        assert not (assets / 'collect/progress.json').exists()
        assert manifest['coverage']['text_samples_with_records'] == 3
        assert manifest['coverage']['vision_images_with_records'] == 2
        assert manifest['coverage']['vision_duplicate_selected_tokens'] == 0
        for key, state in manifest['buckets'].items():
            origin = text if state['modality'] == 'text' else vision
            actual = load_inputs(assets / 'collect', key, state, 3)
            expected = load_inputs(origin, key, state, 3)
            assert torch.equal(actual, expected)
        assert invoke(text, vision, assets / 'collect', link_mode=mode)['input_sha256'] == manifest['input_sha256']
        if mode == 'hardlink':
            cluster(assets, manifest['cluster_algorithm'])
            assert any(name.endswith('_pca.pt') for name in verify_stage(assets / 'cluster', hash_outputs=False)['outputs'])
            pieces = [(layer, 0, 8, dict(gate_up=torch.full((8, 4, 3), .01, dtype=torch.float16),
                                        down=torch.full((8, 3, 2), .01, dtype=torch.float16))) for layer in range(2)]
            materialize(assets, manifest['cluster_algorithm'], pieces=pieces)
            assert verify_stage(assets / 'cache', hash_outputs=False)['provenance']['collection_kind'] == 'modality_union'
        checks.append(mode + '_native_loader_and_reuse')
    assert before == {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                      for root in (text, vision) for path in root.rglob('*') if path.is_file()}
    checks.append('source_lengths_and_mtimes_unchanged')
    cases = ('incomplete', 'missing_text', 'uncovered_image', 'repeated_image', 'wrong_dtype',
             'wrong_checkpoint', 'bad_bank_length', 'bad_index_length', 'bad_phase_length',
             'wrong_top8', 'wrong_response_id', 'wrong_finish_reason', 'bs2', 'text_on_image',
             'heldout_image', 'partial_output')
    for case in cases:
        folder = WORK / ('reject_' + case)
        main_path, vision_path = folder / 'main', folder / 'visual'
        shutil.copytree(text, main_path)
        shutil.copytree(vision, vision_path)
        target = main_path if case in ('missing_text', 'wrong_top8', 'wrong_response_id', 'wrong_finish_reason',
                                       'text_on_image', 'bs2') else vision_path
        progress = read_json(target / 'progress.json')
        inputs = read_json(target / 'manifest.json')['inputs']
        if case == 'incomplete':
            write_json(target / 'complete.json', dict(complete=False))
        elif case == 'missing_text':
            progress['sample_state']['records'][0]['selected']['text'] = 0
        elif case == 'uncovered_image':
            progress['sample_state']['records'][0]['selected_visual_tokens_per_image'] = [0]
        elif case == 'repeated_image':
            progress['sample_state']['records'][1]['new_image_ids'] = []
        elif case == 'wrong_dtype':
            progress['config']['dtype'] = 'bfloat16'
        elif case == 'wrong_checkpoint':
            inputs['checkpoint']['model'] = 'different'
        elif case == 'wrong_top8':
            progress['buckets']['L0_E0_text']['seen'] -= 1
        elif case == 'wrong_response_id':
            values = merge.rows(target / 'requests.jsonl')
            values[0]['id'] = 'heldout'
            write_rows(target / 'requests.jsonl', values)
        elif case == 'wrong_finish_reason':
            values = merge.rows(target / 'requests.jsonl')
            values[0]['finish_reason'] = None
            write_rows(target / 'requests.jsonl', values)
        elif case == 'bs2':
            progress['batch_sizes']['prefill']['2'] = 1
        elif case == 'text_on_image':
            progress['sample_state']['records'][0]['positions']['text'] = [0, 3]
        if case in ('missing_text', 'uncovered_image', 'repeated_image', 'wrong_dtype', 'wrong_checkpoint', 'wrong_top8',
                    'bs2', 'text_on_image'):
            write_json(target / 'progress.json', progress)
            manifest = read_json(target / 'manifest.json')
            manifest['inputs'] = inputs
            manifest['input_sha256'] = digest(inputs)
            for key in ('buckets', 'sampling', 'token_banks', 'batch_sizes'):
                manifest[key] = progress[key]
            write_json(target / 'manifest.json', manifest)
            write_json(target / 'complete.json', dict(complete=True,
                manifest_sha256=hashlib.sha256((target / 'manifest.json').read_bytes()).hexdigest()))
        if case.startswith('bad_'):
            suffix = {'bad_bank_length': 'L0_vision_tokens.bin', 'bad_index_length': 'L0_E0_vision.index',
                      'bad_phase_length': 'L0_E0_vision.phase'}[case]
            path = target / 'layer_0' / suffix
            path.write_bytes(path.read_bytes()[:-1])
        output = folder / 'output/collect'
        if case == 'partial_output':
            output.mkdir(parents=True)
        extra = {}
        if case == 'heldout_image':
            path = folder / 'heldout.jsonl'
            write_rows(path, [dict(id='heldout', image_ids=['image0'])])
            extra['evaluation'] = path
        try:
            invoke(main_path, vision_path, output, **extra)
        except (ValueError, KeyError):
            checks.append('reject_' + case)
        else:
            raise AssertionError(case)
    result = dict(complete=True, checks=checks, count=len(checks), gpu_started=False,
        native_cluster_cpu=True, native_materialize_cpu=True, tensor_content_hashes_computed=False, evidence_root=str(WORK),
        source_sha256=hashlib.sha256((HERE / 'merge_modalities.py').read_bytes()).hexdigest())
    write_json(HERE / 'merge_modalities_cpu_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
