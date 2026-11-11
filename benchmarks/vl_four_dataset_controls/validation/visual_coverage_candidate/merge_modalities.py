"""Build an indexed collection from independent completed text and vision passes."""

import argparse
import importlib.util
import json
import os
from pathlib import Path


def experiment_root():
    return next(path for path in Path(__file__).resolve().parents if (path / 'run_cell.py').is_file())


def artifacts_module(snapshot):
    path = Path(snapshot) / 'python/freetoken/moe/tablemoe_artifacts.py'
    spec = importlib.util.spec_from_file_location('modality_union_artifacts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def count(value, message):
    require(type(value) is int and value >= 0, message)
    return value


def checked_source(root, modality, selected_rows, api, shape, top_k):
    root = Path(root).resolve()
    source = api.verify_stage(root, hash_outputs=False)
    progress = api.read_json(root / 'progress.json')
    config, inputs = progress['config'], source['inputs']
    ids = [row['id'] for row in selected_rows]
    require(inputs['calibration_ids'] == ids and inputs['calibration_sha256'] == api.digest(selected_rows),
            f'{modality}: frozen sample identities or inputs differ')
    require(inputs['sample_count'] == len(ids) == config['sample_count'], f'{modality}: sample count mismatch')
    require(progress['identity'] == source['input_sha256'], f'{modality}: collector identity mismatch')
    require(inputs['algorithm'] == config, f'{modality}: recorded collector config differs')
    require(config['seed'] == 42 and config['sampling_policy'] == 'tablemoe_default',
            f'{modality}: unsupported sampling protocol')
    require(config.get('raw_storage') == source.get('raw_storage') == 'indexed'
            and config.get('export_inputs') is False, f'{modality}: indexed raw storage is required')
    for name, expected in shape.items():
        require(source.get(name) == expected, f'{modality}: incompatible {name}')
    require(progress['hidden'] == shape['hidden'] and config['dtype'] == shape['dtype'],
            f'{modality}: hidden size or dtype mismatch')
    require(config['collection_image_ids'] == [row['image_ids'] for row in selected_rows],
            f'{modality}: image input order changed')
    for name in ('buckets', 'token_banks', 'sampling', 'tokens', 'batch_sizes'):
        require(source[name] == progress[name], f'{modality}: manifest/progress {name} differ')
    records = progress['sample_state']['records']
    require(len(records) == len(ids) == len(progress['requests']), f'{modality}: incomplete request coverage')
    require(progress['requests'] == [record['uid'] for record in records], f'{modality}: checkpointed request order differs')
    requests = rows(root / 'requests.jsonl')
    require([request['id'] for request in requests] == ids, f'{modality}: response IDs differ')
    decode = config['sample_decode_tokens']
    require(progress['tokens']['decode'] == decode * len(ids), f'{modality}: decode coverage mismatch')
    require(progress['tokens']['prefill'] == sum(record['prompt_tokens'] for record in records),
            f'{modality}: full prompt coverage mismatch')
    require(set(source['sampling']) == {str(layer) for layer in range(shape['num_layers'])},
            f'{modality}: sampling layer identities differ')
    require(all(set(sizes) <= {'1'} for sizes in progress['batch_sizes'].values()),
            f'{modality}: collection was not bs1')
    if modality == 'vision':
        require(decode == 0 and config['max_text_tokens'] == 0
                and config['min_text_per_sample'] == 0 and config.get('min_vision_tokens_per_image') == 1,
                'visual sweep must collect only vision with per-image minimum one')
    else:
        require(decode == 127, 'text source must retain the original 128-token generation protocol')
    counts = progress['sample_state']['counts']
    selected_text, selected_images, seen_images = {}, {}, set()
    repeated_selected = 0
    for ordinal, (row, record, response) in enumerate(zip(selected_rows, records, requests)):
        require(record['ordinal'] == ordinal and record['image_ids'] == row['image_ids'],
                f'{modality}: sample/image order mismatch at {ordinal}')
        require(record['decode_seen'] == decode and response['usage']['completion_tokens'] == decode + 1
                and response['max_tokens'] == decode + 1 and response.get('natural_eos') is False
                and response.get('finish_reason') == 'length' and bool(response.get('response_id')),
                f'{modality}: unfinished or changed generation at {ordinal}')
        require(response['usage']['prompt_tokens'] == record['prompt_tokens'],
                f'{modality}: prompt token count mismatch at {ordinal}')
        positions = record['positions']
        for mod in ('vision', 'text'):
            values = positions[mod]
            require(values == sorted(set(values)) and all(type(value) is int and 0 <= value < record['prompt_tokens'] + decode
                    for value in values), f'{modality}: invalid {mod} sample positions')
            require(record['selected'][mod] == len(values), f'{modality}: selected count mismatch')
        require(not set(positions['vision']).intersection(positions['text']), f'{modality}: overlapping modalities')
        if modality == 'text':
            require(record['selected']['text'] > 0, f'text sample has no selected tokens: {row["id"]}')
            selected_text[row['id']] = record['selected']['text']
        else:
            require(not positions['text'], 'visual sweep contains text selections')
        spans = record['image_offsets']
        image_counts = record['selected_visual_tokens_per_image']
        require(len(spans) == len(image_counts) == len(row['image_ids']), f'{modality}: image accounting missing')
        require(all(type(lo) is int and type(hi) is int and 0 <= lo < hi <= record['prompt_tokens']
                    for image_spans in spans for lo, hi in image_spans), f'{modality}: invalid image spans')
        require(not any(lo <= position < hi for position in positions['text']
                        for image_spans in spans for lo, hi in image_spans),
                f'{modality}: text selections include visual positions')
        actual_counts = [sum(any(lo <= value < hi for lo, hi in image_spans) for value in positions['vision'])
                         for image_spans in spans]
        require(actual_counts == image_counts and sum(image_counts) == len(positions['vision']),
                f'{modality}: per-image selected token accounting mismatch')
        new_images = []
        for image_id, selected in zip(row['image_ids'], image_counts):
            if image_id in seen_images:
                repeated_selected += selected
            else:
                new_images.append(image_id)
                seen_images.add(image_id)
            selected_images[image_id] = selected_images.get(image_id, 0) + selected
        require(record['new_image_ids'] == new_images, f'{modality}: first-seen image history mismatch')
    for mod in ('vision', 'text'):
        require(counts[mod] == sum(record['selected'][mod] for record in records),
                f'{modality}: global {mod} count mismatch')
        require(0 <= counts[mod] <= config[f'max_{mod}_tokens'], f'{modality}: {mod} budget exceeded')
    if modality == 'vision':
        require(all(value > 0 for value in selected_images.values()) and repeated_selected == 0,
                'visual sweep has uncovered or repeatedly collected images')
    files, buckets, banks = {}, {}, {}
    for key, state in source['token_banks'].items():
        layer, mod, size = state['layer'], state['modality'], count(state['rows'], 'invalid bank rows')
        require(type(layer) is int and 0 <= layer < shape['num_layers'] and mod in ('vision', 'text')
                and key == f'L{layer}_{mod}_tokens', 'invalid token bank identity')
        require(size == counts[mod], f'{modality}: incomplete token bank')
        relative = f'layer_{layer}/{key}.bin'
        require((root / relative).is_file() and (root / relative).stat().st_size == size * shape['hidden'] * 2,
                f'{modality}: invalid bank length: {relative}')
        if mod == modality:
            banks[key] = state
            files[relative] = root / relative
    for key, state in source['buckets'].items():
        layer, expert, mod = state['layer'], state['expert'], state['modality']
        require(type(layer) is int and 0 <= layer < shape['num_layers'] and type(expert) is int
                and 0 <= expert < shape['num_experts'] and mod in ('vision', 'text')
                and key == f'L{layer}_E{expert}_{mod}', 'invalid expert bucket identity')
        require(state.get('raw_storage') == 'indexed' and state['bank'] == f'L{layer}_{mod}_tokens'
                and state['bank'] in source['token_banks'], 'invalid indexed bucket bank')
        seen, kept = count(state['seen'], 'invalid seen count'), count(state['kept'], 'invalid kept count')
        require(kept == min(seen, config['reservoir_size'])
                and count(state['prefill'], 'invalid prefill') + count(state['decode'], 'invalid decode') == seen,
                'invalid reservoir or phase accounting')
        for suffix, width in (('index', 8), ('phase', 1)):
            relative = f'layer_{layer}/{key}.{suffix}'
            require((root / relative).is_file() and (root / relative).stat().st_size == kept * width,
                    f'{modality}: invalid bucket file length: {relative}')
            if mod == modality:
                files[relative] = root / relative
        if mod == modality:
            buckets[key] = state
    sampling = {}
    for layer in range(shape['num_layers']):
        for mod in ('vision', 'text'):
            state = source['sampling'][str(layer)][mod]
            require(state['selected'] == counts[mod] == state['prefill'] + state['decode'],
                    f'{modality}: incomplete sampling layer {layer}/{mod}')
            routed = [value for value in source['buckets'].values() if value['layer'] == layer and value['modality'] == mod]
            require(sum(value['seen'] for value in routed) == state['selected'] * top_k
                    and sum(value['prefill'] for value in routed) == state['prefill'] * top_k
                    and sum(value['decode'] for value in routed) == state['decode'] * top_k,
                    f'{modality}: top-{top_k} route coverage mismatch at layer {layer}/{mod}')
            if counts[mod]:
                require(f'L{layer}_{mod}_tokens' in source['token_banks'], 'missing layer token bank')
        sampling[str(layer)] = source['sampling'][str(layer)][modality]
    require(set(files) <= set(source['outputs']), f'{modality}: raw files are absent from completed manifest')
    return dict(root=root, manifest=source, progress=progress, files=files, buckets=buckets,
                banks=banks, sampling=sampling, selected_text=selected_text,
                selected_images=selected_images, repeated_selected=repeated_selected,
                manifest_sha256=api.file_hash(root / 'manifest.json'))


def merge_modalities(main_collect, visual_collect, output_collect, calibration_path, evaluation_path,
                     *, snapshot=None, expected_samples=655, expected_images=193, num_layers=48,
                     num_experts=128, hidden=2048, top_k=8, dtype='float16', link_mode='hardlink'):
    snapshot = Path(snapshot) if snapshot else experiment_root() / 'runtime_snapshot_compact'
    api = artifacts_module(snapshot)
    calibration, evaluation = rows(calibration_path), rows(evaluation_path)
    ids = [row['id'] for row in calibration]
    require(len(ids) == len(set(ids)) == expected_samples, 'frozen calibration sample coverage differs')
    images = {image for row in calibration for image in row['image_ids']}
    require(len(images) == expected_images, 'frozen calibration image coverage differs')
    require(not set(ids).intersection(row['id'] for row in evaluation)
            and not images.intersection(image for row in evaluation for image in row['image_ids']),
            'calibration and heldout samples/images overlap')
    require(not {row.get('group_id') for row in calibration if row.get('group_id')}.intersection(
                row.get('group_id') for row in evaluation if row.get('group_id')), 'calibration and heldout groups overlap')
    main_collect, visual_collect, output = map(lambda path: Path(path).resolve(),
                                               (main_collect, visual_collect, output_collect))
    require(link_mode in ('hardlink', 'symlink'), 'unsupported link mode')
    require(all(output != source and source not in output.parents and output not in source.parents
                for source in (main_collect, visual_collect)), 'output must be separate from both source collections')
    require(Path('/tmp') != output and Path('/tmp') not in output.parents, 'persistent output is required')
    visual_manifest = api.verify_stage(visual_collect, hash_outputs=False)
    visual_ids = visual_manifest['inputs']['calibration_ids']
    require(len(visual_ids) == len(set(visual_ids)) and set(visual_ids) <= set(ids), 'visual sweep includes unexpected samples')
    row_map = {row['id']: row for row in calibration}
    visual_rows = [row_map[value] for value in visual_ids]
    require({image for row in visual_rows for image in row['image_ids']} == images, 'visual sweep does not cover all images')
    shape = dict(num_layers=num_layers, num_experts=num_experts, hidden=hidden, dtype=dtype)
    text = checked_source(main_collect, 'text', calibration, api, shape, top_k)
    vision = checked_source(visual_collect, 'vision', visual_rows, api, shape, top_k)
    for key in ('checkpoint', 'convert_bf16', 'processor'):
        require(text['manifest']['inputs'][key] == vision['manifest']['inputs'][key], f'collection {key} differs')
    computation_keys = ('seed', 'reservoir_size', 'pca_dim', 'pca_per_expert', 'clusters', 'iterations',
                        'distance_chunk', 'value_chunk', 'device', 'dtype', 'cluster_batch_size')
    cluster_algorithm = {key: text['progress']['config'][key] for key in computation_keys
                         if key in text['progress']['config']}
    require(cluster_algorithm == {key: vision['progress']['config'][key] for key in computation_keys
                                  if key in vision['progress']['config']}, 'PCA/cluster computation settings differ')
    require(set(vision['selected_images']) == images, 'actual selected image identities differ')
    coverage = dict(text_samples=expected_samples, text_samples_with_records=len(text['selected_text']),
        vision_unique_images=expected_images, vision_images_with_records=len(vision['selected_images']),
        vision_duplicate_selected_tokens=vision['repeated_selected'], heldout_samples_used=0, heldout_images_used=0,
        text_source_samples=len(calibration), vision_source_samples=len(visual_rows),
        text_sample_selected=text['selected_text'], vision_image_selected=vision['selected_images'])
    sources = {mod: dict(collect=str(value['root']), manifest_sha256=value['manifest_sha256'],
        input_sha256=value['manifest']['input_sha256'], samples=value['manifest']['inputs']['sample_count'])
        for mod, value in (('text', text), ('vision', vision))}
    inputs = dict(collection_kind='modality_union', sources=sources,
        calibration_sha256=api.digest(calibration), evaluation_sha256=api.digest(evaluation),
        calibration_ids=ids, sample_count=expected_samples, top_k=top_k, shape=shape,
        link_mode=link_mode, merger_sha256=api.file_hash(__file__), cluster_algorithm=cluster_algorithm,
        **{key: text['manifest']['inputs'][key] for key in ('checkpoint', 'convert_bf16', 'processor')})
    files = {**text['files'], **vision['files']}
    require(not set(text['files']).intersection(vision['files']), 'modalities unexpectedly share raw file names')
    if output.exists():
        require((output / 'complete.json').is_file(), 'partial union exists; preserve it and choose a new output')
        for relative, source in files.items():
            target = output / relative
            require(target.is_file() and os.path.samefile(source, target), 'union raw link no longer names its source')
        return api.verify_stage(output, inputs, hash_outputs=False)
    output.mkdir(parents=True)
    outputs = []
    for relative, source in files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if link_mode == 'hardlink':
            os.link(source, target)
        else:
            target.symlink_to(source)
        outputs.append(target)
    source_links = {}
    for mod, source in (('text', text), ('vision', vision)):
        source_links[mod] = {}
        for name in ('manifest.json', 'complete.json', 'progress.json', 'requests.jsonl'):
            relative = f'source_records/{mod}_{name}'
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source['root'] / name)
            outputs.append(target)
            source_links[mod][name] = relative
    provenance = dict(collection_kind='modality_union', ordinary_single_pass=False,
        sources=sources, source_records=source_links, coverage=coverage, raw_link_mode=link_mode,
        source_artifacts_modified=False, tensor_content_hashes_computed=False,
        text_source_vision_discarded_from_union=True, visual_source_text_included=False,
        source_forward_tokens={mod: source['manifest']['tokens'] for mod, source in (('text', text), ('vision', vision))})
    api.write_json(output / 'modality_union.json', provenance)
    outputs.append(output / 'modality_union.json')
    sampling = {str(layer): dict(text=text['sampling'][str(layer)], vision=vision['sampling'][str(layer)])
                for layer in range(num_layers)}
    tokens = {phase: sum(sampling['0'][mod][phase] for mod in ('text', 'vision')) for phase in ('prefill', 'decode')}
    for source in (text, vision):
        require(api.file_hash(source['root'] / 'manifest.json') == source['manifest_sha256'], 'source changed during union')
    return api.finish_stage(output, inputs, outputs, hash_outputs=False,
        collection_kind='modality_union', ordinary_single_pass=False, coverage=coverage, sources=sources,
        cluster_algorithm=cluster_algorithm,
        dependencies={str(value['root']): value['manifest_sha256'] for value in (text, vision)},
        buckets={**text['buckets'], **vision['buckets']}, token_banks={**text['banks'], **vision['banks']},
        tokens=tokens, token_count_semantics='selected token bank rows per layer, not forwarded prompt tokens',
        sampling=sampling, batch_sizes={}, raw_storage='indexed', **shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main-collect', 'visual-collect', 'output-collect', 'calibration', 'evaluation'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--link-mode', choices=('hardlink', 'symlink'), default='hardlink')
    args = parser.parse_args()
    result = merge_modalities(args.main_collect, args.visual_collect, args.output_collect,
                              args.calibration, args.evaluation, snapshot=args.snapshot, link_mode=args.link_mode)
    print(json.dumps(dict(complete=True, output=str(args.output_collect), collection_kind=result['collection_kind'],
                         text_samples=result['coverage']['text_samples'], vision_images=result['coverage']['vision_unique_images'])))


if __name__ == '__main__':
    main()
