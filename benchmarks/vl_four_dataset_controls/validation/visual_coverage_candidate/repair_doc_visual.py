"""Collect Doc's frozen unique images, union main text, and rebuild independent tables."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'run_cell.py').is_file())
CANDIDATE = ROOT / 'validation/visual_coverage_candidate'
DATASET = 'MMLongBench_DOC'
REPAIR = Path('/data/fanya/tablemoe/vl_four_dataset_bs1_20261111/MMLongBench_DOC/coverage_repaired')
DEFAULT_DEADLINE = 1794450189


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    result = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.endswith('\n') or not line.strip():
                raise ValueError('incomplete JSONL; preserve it without truncating')
            result.append(json.loads(line))
    return result


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def remaining(deadline, limit):
    value = deadline - time.time()
    if value <= 0:
        raise TimeoutError('authorized experiment deadline reached')
    return min(value, limit)


def freeze(path, value):
    if Path(path).exists():
        require(read(path) == value, f'frozen repair protocol changed: {path}')
    else:
        save(path, value)


def select_visual_rows(calibration, evaluation, expected_samples=655, expected_images=193):
    require(len(calibration) == expected_samples, 'frozen calibration sample count changed')
    require(len({row['id'] for row in calibration}) == len(calibration), 'duplicate calibration IDs')
    all_images = {image for row in calibration for image in row['image_ids']}
    test_images = {image for row in evaluation for image in row['image_ids']}
    require(len(all_images) == expected_images and not all_images & test_images,
            'calibration image coverage or heldout isolation changed')
    require(not {r['id'] for r in calibration} & {r['id'] for r in evaluation}, 'heldout row leakage')
    selected, membership, seen = [], [], set()
    for ordinal, row in enumerate(calibration):
        images = row['image_ids']
        new = [image for image in dict.fromkeys(images) if image not in seen]
        if not new:
            continue
        message_images = [str(Path(item['value']).resolve()) for item in row['vlmeval_message'] if item['type'] == 'image']
        require(message_images == images, 'full prompt images differ from frozen identities')
        selected.append(row)
        membership.append(dict(calibration_ordinal=ordinal, id=row['id'], image_ids=images,
                               new_image_ids=new, repeated_image_ids=[image for image in images if image in seen]))
        seen.update(images)
    require(seen == all_images, 'greedy selection does not cover all calibration images')
    return selected, membership


def plan(snapshot, repair_root):
    calibration_path = ROOT / 'prepare' / DATASET / 'calibration.jsonl'
    evaluation_path = ROOT / 'prepare' / DATASET / 'evaluation.jsonl'
    calibration, evaluation = rows(calibration_path), rows(evaluation_path)
    selected, membership = select_visual_rows(calibration, evaluation)
    main = read(ROOT / 'build' / DATASET / 'protocol.json')
    original = main['inputs']
    require(original['calibration_ids'] == [r['id'] for r in calibration]
            and original['calibration_sha256'] == digest(calibration), 'main frozen input identity differs')
    require(original['sample_count'] == 655 and original['algorithm']['sample_count'] == 655,
            'main collection must cover all 655 text samples')
    algorithm = dict(original['algorithm'], sample_count=len(selected), sample_decode_tokens=0,
        min_vision_tokens_per_image=1, min_text_per_sample=0, max_text_per_sample=0,
        max_text_tokens=0, max_vision_tokens=50000,
        max_vision_per_sample=max(original['algorithm']['max_vision_per_sample'],
                                  max(len(row['new_image_ids']) for row in membership)),
        collection_image_ids=[row['image_ids'] for row in selected])
    inputs = dict(original, calibration_sha256=digest(selected), calibration_ids=[r['id'] for r in selected],
        sample_count=len(selected), algorithm=algorithm,
        modality_repair=dict(parent_calibration_sha256=original['calibration_sha256'],
            parent_calibration_samples=655, expected_unique_images=193,
            selection='first new-image occurrence in frozen calibration order', heldout_samples=0,
            full_multimodal_inputs_preserved=True, text_collected=False))
    return dict(dataset=DATASET, snapshot=str(snapshot.resolve()), repair_root=str(repair_root.resolve()),
        report_root=str(ROOT / 'build/MMLongBench_DOC_visual_repair'),
        main_collect=str(Path(main['assets']) / 'collect'),
        visual_assets=str(repair_root / 'visual_assets'), output_assets=str(repair_root / 'assets'),
        calibration_path=str(calibration_path), evaluation_path=str(evaluation_path),
        inputs=inputs, selected_rows=selected, selection=membership,
        calibration_samples=655, unique_images=193, collection_requests=len(selected),
        full_multimodal_input_preserved=True, original_prompt_processor_preserved=True,
        processor=original['processor'], batch_size=1, cache_slots=6144,
        fixed_output_tokens=1, decode_collection_tokens=0, cuda_graph=False,
        source_root=str(ROOT), tensor_content_scans=False)


def validate_visual(progress, responses, selected, inputs, expected_images=193, num_layers=48):
    algorithm = inputs['algorithm']
    records = progress['sample_state']['records']
    require(progress['identity'] == digest(inputs) and progress['config'] == algorithm,
            'visual collector identity or configuration changed')
    require(len(records) == len(responses) == len(progress['requests']) == len(selected),
            'visual request or terminal record coverage incomplete')
    require([r['id'] for r in responses] == [r['id'] for r in selected], 'visual requests are not the frozen ordered subset')
    require([r['uid'] for r in records] == progress['requests'], 'collector request UIDs changed')
    seen, coverage, total, prompt_total = set(), {}, 0, 0
    for ordinal, (row, response, record) in enumerate(zip(selected, responses, records)):
        require(record['ordinal'] == ordinal and record.get('decode_seen', 0) == 0,
                'visual request ordinal or decode count differs')
        require(response['usage']['completion_tokens'] == response['max_tokens'] == 1
                and response['finish_reason'] == 'length' and response['natural_eos'] is False,
                'visual collection must return exactly one fixed output token')
        length = record['prompt_tokens']
        require(response['usage']['prompt_tokens'] == length, 'expanded prompt length differs')
        require(record['image_ids'] == row['image_ids'], 'full multimodal image list changed')
        image_counts = record['selected_visual_tokens_per_image']
        offsets = record['image_offsets']
        require(len(image_counts) == len(offsets) == len(row['image_ids']), 'per-image proof cardinality differs')
        positions = record['positions']['vision']
        require(len(set(positions)) == len(positions) and positions == sorted(positions), 'visual positions are not unique and sorted')
        require(record['positions']['text'] == [] and record['selected']['text'] == 0, 'visual sweep collected text')
        fresh = []
        accounted = set()
        for image, spans, count in zip(row['image_ids'], offsets, image_counts):
            require(bool(spans) and all(0 <= lo < hi <= length for lo, hi in spans), 'invalid visual span')
            chosen = {position for position in positions if any(lo <= position < hi for lo, hi in spans)}
            require(count == len(chosen), 'per-image selected count differs from saved positions')
            if image in seen:
                require(count == 0, 'shared image was collected twice')
            else:
                require(count >= 1, 'a new calibration image has zero selected tokens')
                fresh.append(image)
                coverage[image] = count
                seen.add(image)
            accounted.update(chosen)
        require(record['new_image_ids'] == fresh, 'first-seen image history differs')
        require(accounted == set(positions) and record['selected']['vision'] == len(positions),
                'visual selection contains non-image positions')
        total += len(positions)
        prompt_total += length
    require(len(coverage) == expected_images, 'visual image coverage is incomplete')
    require(progress['sample_state']['counts'] == dict(vision=total, text=0)
            and 0 < total <= algorithm['max_vision_tokens'], 'visual budget or text-zero constraint differs')
    require(progress['tokens'] == dict(prefill=prompt_total, decode=0), 'unexpected decode collection')
    require(set(progress['sampling']) == {str(i) for i in range(num_layers)}, 'missing layer records')
    require(set(progress['batch_sizes']['prefill']) == {'1'} and not progress['batch_sizes']['decode'], 'collection violated bs1 or decode-zero')
    for layer in range(num_layers):
        state = progress['sampling'][str(layer)]
        require(state['vision']['selected'] == state['vision']['prefill'] == total
                and state['vision']['decode'] == 0 and state['text']['selected'] == 0,
                f'layer {layer} visual/text collection differs')
        routed = sum(bucket['seen'] for bucket in progress['buckets'].values()
                     if bucket['layer'] == layer and bucket['modality'] == 'vision')
        require(routed == total * 8, f'layer {layer} full top8 routing count differs')
    return dict(complete=True, expected_images=expected_images, covered_images=len(coverage),
        selected_visual_tokens=total, per_image_selected=coverage, selected_text_tokens=0,
        duplicate_selected_visual_tokens=0, collection_requests=len(selected),
        full_multimodal_inputs_preserved=True, heldout_samples=0, tensor_content_scans=False)


def snapshot_receipt(snapshot):
    baseline = ROOT / 'runtime_snapshot_buffered'
    inventories = []
    for directory in (baseline, snapshot):
        inventories.append({str(path.relative_to(directory)): source_hash(path)
            for tree in ('python', 'benchmarks') for path in sorted((directory / tree).rglob('*.py'))})
    old, new = inventories
    require(old and old.keys() == new.keys(), 'repair snapshot file set differs from buffered runtime')
    changed = [key for key in old if old[key] != new[key]]
    require(changed == ['python/freetoken/moe/tablemoe_builder.py'], 'repair snapshot must only change the collector builder')
    require(new[changed[0]] == source_hash(CANDIDATE / 'tablemoe_builder.py'), 'snapshot builder differs from reviewed candidate')
    return dict(snapshot=str(snapshot), baseline=str(baseline), changed_python_sources=changed,
                builder_sha256=new[changed[0]], source_sha256=new)


def worker(stage, plan_path, port=None):
    value = read(plan_path)
    snapshot = Path(value['snapshot'])
    sys.path[:0] = [str(snapshot / 'python'), str(snapshot / 'benchmarks')]
    from freetoken.moe.tablemoe_builder import algorithm_config, finalize_collection, cluster, materialize
    from freetoken.moe.tablemoe_artifacts import verify_stage
    from tablemoe_common import stream_request
    from tablemoe_data import api_messages
    inputs = value['inputs']
    require(algorithm_config(inputs['algorithm']) == inputs['algorithm'], 'collector normalized config differs from frozen plan')
    collection = Path(value['visual_assets']) / 'collect'
    report = Path(value['report_root'])
    assets = Path(value['output_assets'])
    remaining(value['deadline'], 1)
    if stage == 'collect':
        require(not (collection / 'requests.jsonl').exists(), 'partial visual collection needs a preserved fresh destination')
        collection.mkdir(parents=True, exist_ok=True)
        with (collection / 'requests.jsonl').open('x') as handle:
            for ordinal, row in enumerate(value['selected_rows']):
                response = stream_request(port, dict(row, messages=api_messages(row['vlmeval_message'])),
                    1, natural=False, timeout=remaining(value['deadline'], 7200))
                handle.write(json.dumps(response, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                save(report / 'progress.json', dict(stage='collect', complete=False,
                    completed=ordinal + 1, total=len(value['selected_rows'])))
    elif stage == 'finalize':
        coverage = validate_visual(read(collection / 'progress.json'), rows(collection / 'requests.jsonl'),
                                   value['selected_rows'], inputs)
        if not (collection / 'complete.json').is_file():
            finalize_collection(collection, inputs, 48, 128)
        manifest = verify_stage(collection, hash_outputs=False)
        require(manifest.get('image_coverage', {}).get('covered_images') == 193,
                'native visual finalize did not prove all 193 images')
        save(report / 'visual_coverage.json', coverage)
    elif stage == 'merge':
        from merge_modalities import merge_modalities
        merge_modalities(Path(value['main_collect']), collection, assets / 'collect',
            Path(value['calibration_path']), Path(value['evaluation_path']), snapshot=snapshot)
    elif stage == 'cluster':
        union = verify_stage(assets / 'collect', hash_outputs=False)
        cluster(assets, union['cluster_algorithm'])
    elif stage == 'materialize':
        union = verify_stage(assets / 'collect', hash_outputs=False)
        materialize(assets, union['cluster_algorithm'])
    elif stage == 'verify':
        manifest = verify_stage(assets / 'cache', hash_outputs=False)
        main = read(Path(value['main_collect']) / 'progress.json')
        require(len(main['sample_state']['records']) == 655
                and all(row['selected']['text'] > 0 for row in main['sample_state']['records']),
                'main text collection does not cover all 655 samples')
        visual = read(report / 'visual_coverage.json')
        require(visual['complete'] and visual['covered_images'] == 193, 'visual proof incomplete')
        union = verify_stage(assets / 'collect', hash_outputs=False)
        require(manifest.get('provenance') == union['inputs'], 'table provenance does not match the validated modality union')
        expected = dict(text_samples=655, text_samples_with_records=655, vision_unique_images=193,
            vision_images_with_records=193, vision_duplicate_selected_tokens=0,
            heldout_samples_used=0, heldout_images_used=0, text_source_samples=655, vision_source_samples=78)
        require(all(union['coverage'].get(key) == item for key, item in expected.items()),
                'native union manifest does not prove exact text/image coverage')
        save(report / 'verified_tables.json', dict(complete=True, assets=str(assets),
            calibration_samples=655, covered_text_samples=655, covered_images=193,
            selected_visual_tokens=visual['selected_visual_tokens'], selected_text_tokens=main['sample_state']['counts']['text'],
            missing_tables=manifest['missing_tables'], union_input_sha256=union['input_sha256'],
            cache_input_sha256=manifest['input_sha256'], tensor_content_scans=False))
    remaining(value['deadline'], 1)


def run_worker(stage, plan_path, value, env, port=None):
    report = Path(value['report_root'])
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', stage, '--plan', str(plan_path)]
    if port is not None:
        command += ['--port', str(port)]
    attempt = report / f'{stage}_{time.time_ns()}'
    attempt.mkdir(parents=True)
    with (attempt / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, cwd=value['snapshot'], env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            save(attempt / 'process.json', dict(pid=process.pid, command=command, owned=True))
            while process.poll() is None:
                time.sleep(remaining(value['deadline'], 1))
            require(process.returncode == 0, f'{stage} failed with exit {process.returncode}: {attempt}')
            remaining(value['deadline'], 1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


def live_main_build_workers():
    scripts = {str(ROOT / name) for name in ('build_table.py', 'resume_build_table.py')}
    live = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            state = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if state != 'Z' and DATASET in command and scripts.intersection(command):
            live.append(int(entry.name))
    return live


def publish_canonical(final, value, deadline):
    canonical = ROOT / 'build' / DATASET / 'complete.json'
    backup = canonical.with_name('complete_before_visual_repair.json')
    while True:
        remaining(deadline, 1)
        active = live_main_build_workers()
        if canonical.is_file() and read(canonical).get('complete') and not active:
            break
        save(Path(value['report_root']) / 'publication_wait.json', dict(complete=False,
            stage='await_main_build_completion', live_main_build_workers=active,
            original_build_complete=canonical.is_file() and read(canonical).get('complete', False)))
        time.sleep(remaining(deadline, 2))
    previous = read(canonical)
    if previous.get('assets') == final['assets']:
        require(backup.is_file() and previous.get('coverage_repair_complete'), 'canonical repair provenance missing')
        return previous
    require(previous.get('assets') == str(Path(value['main_collect']).parent), 'canonical main asset binding differs')
    original = canonical.read_bytes()
    if backup.exists():
        require(backup.read_bytes() == original, 'original completion backup differs; never overwrite it')
    else:
        with backup.open('xb') as handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
    updated = dict(final, coverage_repair_complete=True, visual_pass_samples=78, unique_images=193,
        first_seen_images=193, image_count_semantics='unique images with actual selected visual tokens',
        collected_tokens=dict(vision=final['selected_visual_tokens'], text=final['selected_text_tokens']),
        duplicate_visual_tokens=0, main_text_source=value['main_collect'],
        visual_repair_source=str(Path(value['visual_assets']) / 'collect'),
        previous_completion=str(backup), canonical_switch_unix=time.time())
    remaining(deadline, 1)
    require(not live_main_build_workers(), 'main build became active before canonical publication')
    save(canonical, updated)
    return updated


def execute(value, gpu, deadline):
    remaining(deadline, 1)
    release = read(ROOT / 'validation/doc_visual_coverage_release.json')
    require(release.get('complete') is True, 'repair candidate is not released')
    sources = dict(driver_sha256=source_hash(__file__), builder_sha256=source_hash(CANDIDATE / 'tablemoe_builder.py'),
                   merge_sha256=source_hash(CANDIDATE / 'merge_modalities.py'))
    require(all(release.get(key) == val for key, val in sources.items()), 'release source hashes differ')
    snapshot = Path(value['snapshot'])
    receipt = snapshot_receipt(snapshot)
    require(read(Path(value['main_collect']) / 'complete.json').get('complete') is True,
            'main full655 collection must finish before the repair')
    report = Path(value['report_root'])
    report.mkdir(parents=True, exist_ok=True)
    lock = (report / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    value = dict(value, deadline=deadline, source_hashes=sources)
    plan_path = report / 'protocol.json'
    freeze(plan_path, value)
    freeze(report / 'source_receipt.json', receipt)
    os.environ['FREETOKEN_EXPERIMENT_SNAPSHOT'] = str(snapshot)
    sys.path.insert(0, str(ROOT))
    from run_cell import command, environment
    from vision_component_probe import owned_server
    env = environment(gpu)
    env['FREETOKEN_EXPERIMENT_SNAPSHOT'] = str(snapshot)
    collection = Path(value['visual_assets']) / 'collect'
    config = Path(value['visual_assets']) / 'collect_config.json'
    freeze(config, dict(output=str(collection), builder=value['inputs']['algorithm'], input_sha256=digest(value['inputs'])))
    try:
        if not (collection / 'complete.json').exists():
            require(not (collection / 'progress.json').exists() and not (collection / 'requests.jsonl').exists(),
                    'partial visual raw data is preserved; use a fresh repair destination')
            server_command, port = command(DATASET, 'schedule', gpu, collect=config)
            save(report / 'command.json', dict(command=server_command, gpu=gpu, sources=sources,
                batch_size=1, fixed_output_tokens=1, eager=True, full_route_top_k=8, cache_slots=6144))
            with owned_server(server_command, report / f'server_{time.time_ns()}', snapshot, env,
                              remaining(deadline, 1800)):
                run_worker('collect', plan_path, value, env, port)
        for stage in ('finalize', 'merge', 'cluster', 'materialize', 'verify'):
            remaining(deadline, 1)
            save(report / 'progress.json', dict(stage=stage, complete=False))
            run_worker(stage, plan_path, value, env)
        remaining(deadline, 1)
        final = dict(read(report / 'verified_tables.json'), source_hashes=sources,
                     main_collect=value['main_collect'], visual_collect=str(collection), protocol=str(plan_path))
        final = publish_canonical(final, value, deadline)
        remaining(deadline, 1)
        save(report / 'complete.json', final)
        save(ROOT / 'validation/doc_visual_coverage_ready.json', final)
    except BaseException as error:
        save(report / 'failure.json', dict(complete=False, error=repr(error), deadline=deadline,
                                           stopped_unix=time.time(), main_collection_unchanged=True))
        raise


def interrupted(signum, frame):
    raise InterruptedError(f'visual repair driver received signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--deadline', type=float, default=DEFAULT_DEADLINE)
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'runtime_snapshot_visual_coverage')
    parser.add_argument('--repair-root', type=Path, default=REPAIR)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', choices=('collect', 'finalize', 'merge', 'cluster', 'materialize', 'verify'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker:
        worker(args.worker, args.plan, args.port)
        return
    value = plan(args.snapshot, args.repair_root)
    if args.execute:
        execute(value, args.gpu, args.deadline)
    else:
        report = dict(complete=True, cpu_preflight=True, gpu_started=False,
            calibration_samples=value['calibration_samples'], unique_images=value['unique_images'],
            collection_requests=value['collection_requests'], selected_ids=value['inputs']['calibration_ids'],
            batch_size=1, fixed_output_tokens=1, max_vision_tokens=50000, selected_text_tokens=0,
            snapshot=value['snapshot'], repair_root=value['repair_root'],
            driver_sha256=source_hash(__file__), tensor_content_scans=False)
        save(CANDIDATE / 'driver_preflight.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
