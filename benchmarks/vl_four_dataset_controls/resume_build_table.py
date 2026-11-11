"""Append-only collection resume after a proved complete-request stop boundary."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'run_cell.py').is_file())
CANDIDATE = ROOT / 'validation/io_buffer_candidate'


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def json_lines(path):
    result = []
    with Path(path).open('rb') as handle:
        for line in handle:
            if not line.endswith(b'\n') or not line.strip():
                raise ValueError('partial or empty JSONL line; automatic truncation is forbidden')
            result.append(json.loads(line))
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_prefix(rows, inputs, config, collection, *, files=True, layers=48, experts=128):
    import numpy as np

    collection = Path(collection)
    predictions = json_lines(collection / 'requests.jsonl')
    progress = read(collection / 'progress.json')
    algorithm = inputs['algorithm']
    require(inputs['calibration_ids'] == [r['id'] for r in rows], 'frozen row IDs changed')
    require(inputs['calibration_sha256'] == digest(rows), 'frozen row records changed')
    require(inputs['sample_count'] == len(rows) == algorithm['sample_count'], 'full sample_count changed')
    require(algorithm['collection_image_ids'] == [r['image_ids'] for r in rows], 'full image identity order changed')
    require(config == dict(output=str(collection), builder=algorithm, input_sha256=digest(inputs)), 'collection config changed')
    require(progress['config'] == algorithm and progress['identity'] == digest(inputs), 'collector identity/config changed')
    require(algorithm['sampling_policy'] == 'tablemoe_default' and algorithm['checkpoint_interval'] == 1,
            'resume requires per-request default-policy checkpoints')
    require(algorithm['sample_decode_tokens'] == 127 and algorithm['dtype'] == 'float16', 'fixed128 FP16 protocol changed')
    require(algorithm.get('per_sample_seed') is True, 'this driver requires the frozen per-sample RNG protocol')
    n = len(predictions)
    records = progress['sample_state']['records']
    require(0 < n <= len(rows), 'resume requires a nonempty complete prefix')
    require([r['id'] for r in predictions] == [r['id'] for r in rows[:n]], 'request IDs are not the exact frozen prefix')
    require(len(records) == len(progress['requests']) == n, 'incomplete/uncommitted collector request')
    require([r['uid'] for r in records] == progress['requests'], 'collector request UID order differs')
    require(progress['sample_state']['rng_state'] == np.random.default_rng(algorithm['seed']).bit_generator.state,
            'global RNG state changed under per-sample seeding')
    counts = dict(vision=0, text=0)
    history = dict(vision=[], text=[])
    expected = {m: dict(candidates=0, selected=0, prefill=0, decode=0) for m in counts}
    seen_images = set()
    prompt_total = 0
    for ordinal, (row, response, record) in enumerate(zip(rows, predictions, records)):
        require(record['ordinal'] == ordinal and record['decode_seen'] == 127, 'incomplete sample or ordinal gap')
        require(response['usage']['completion_tokens'] == 128 and response['max_tokens'] == 128
                and response['natural_eos'] is False and response['finish_reason'] == 'length', 'non-full-route fixed128 response')
        require(bool(response['prediction_text'].strip()), 'empty completed response')
        length = record['prompt_tokens']
        require(response['usage']['prompt_tokens'] == length, 'prompt token count differs from collector')
        require(record['image_ids'] == row['image_ids'], 'sample image order changed')
        require(len(record['image_offsets']) == len(row['image_ids']), 'image offset cardinality changed')
        visual = np.zeros(length, dtype=bool)
        fresh = np.zeros(length, dtype=bool)
        new_images = []
        for image, spans in zip(record['image_ids'], record['image_offsets']):
            for lo, hi in spans:
                require(0 <= lo <= hi <= length, 'invalid saved image span')
                visual[lo:hi] = True
            if image not in seen_images:
                new_images.append(image)
                for lo, hi in spans:
                    fresh[lo:hi] = True
                seen_images.add(image)
        require(record['new_image_ids'] == new_images, 'shared-image history changed')
        require(record['duplicate_visual_tokens'] == int((visual & ~fresh).sum()), 'duplicate visual count differs')
        require(record['selected_visual_tokens_per_image'] == [
            sum(any(lo <= position < hi for lo, hi in spans) for position in record['positions']['vision'])
            for spans in record['image_offsets']], 'per-image visual selection counts differ')
        generator = np.random.default_rng(algorithm['seed'] ^ int(record['prompt_sha256'][:16], 16))
        for mod, mask in [('vision', fresh), ('text', ~visual)]:
            positions = np.flatnonzero(mask)
            if mod == 'text':
                positions = np.concatenate((positions, np.arange(length, length + 127)))
            available = len(positions)
            remaining = algorithm[f'max_{mod}_tokens'] - counts[mod]
            median = max(float(np.median(history[mod])) if history[mod] else float(available), 1.)
            quota = int(round(remaining / max(1, len(rows) - ordinal) * np.sqrt(max(float(available), 1.) / median)))
            take = (max(0, min(available, remaining, algorithm[f'max_{mod}_per_sample'],
                              max(algorithm[f'min_{mod}_per_sample'], quota))) if available and remaining > 0 else 0)
            chosen = positions if take == available else generator.choice(positions, take, replace=False)
            require(record['positions'][mod] == sorted(chosen.tolist()), 'saved token selection/RNG differs')
            require(record['selected'][mod] == take and record['candidates'][mod] == available, 'sample quotas differ')
            counts[mod] += take
            history[mod].append(available)
            expected[mod]['selected'] += take
            expected[mod]['prefill'] += int((chosen < length).sum())
            expected[mod]['decode'] += int((chosen >= length).sum())
        expected['vision']['candidates'] += int(visual.sum())
        expected['text']['candidates'] += int((~visual).sum()) + 127
        prompt_total += length
    require(progress['sample_state']['counts'] == counts and progress['sample_state']['history'] == history,
            'cumulative sample counts/history differ')
    require(progress['tokens'] == dict(prefill=prompt_total, decode=n * 127), 'token coverage is incomplete')
    require(set(progress['sampling']) == {str(i) for i in range(layers)}, 'missing physical layer checkpoint')
    for layer in range(layers):
        require(progress['sampling'][str(layer)] == expected, f'layer {layer} sampling differs')
    require(set(progress['batch_sizes']['prefill']) == {'1'} and
            progress['batch_sizes']['decode'] == {'1': n * 127}, 'collection batch size or decode coverage differs')
    expected_paths = set()
    sums = {(layer, mod): 0 for layer in range(layers) for mod in counts}
    for key, bucket in progress['buckets'].items():
        layer, expert, mod = bucket['layer'], bucket['expert'], bucket['modality']
        require(0 <= layer < layers and 0 <= expert < experts and mod in counts, 'invalid bucket identity')
        require(key == f'L{layer}_E{expert}_{mod}', 'bucket filename identity differs')
        require(bucket['seen'] == bucket['prefill'] + bucket['decode'] and
                bucket['kept'] == min(bucket['seen'], algorithm['reservoir_size']), 'reservoir counters differ')
        indexed = bucket.get('raw_storage') == 'indexed'
        require(indexed == (algorithm.get('raw_storage') == 'indexed'), 'raw storage changed')
        for suffix, width in [('index' if indexed else 'bin', 8 if indexed else progress['hidden'] * 2), ('phase', 1)]:
            path = collection / f'layer_{layer}' / f'{key}.{suffix}'
            expected_paths.add(path)
            if files:
                require(path.is_file() and path.stat().st_size == bucket['kept'] * width, f'raw length mismatch: {path}')
        sums[layer, mod] += bucket['seen']
    for layer in range(layers):
        for mod in counts:
            require(sums[layer, mod] == counts[mod] * 8, 'per-layer top8 routed sample counts differ')
            if algorithm.get('raw_storage') == 'indexed' and counts[mod]:
                bank = f'L{layer}_{mod}_tokens'
                require(progress['token_banks'][bank] == dict(layer=layer, modality=mod, rows=counts[mod]), 'bank counters differ')
                path = collection / f'layer_{layer}' / f'{bank}.bin'
                expected_paths.add(path)
                if files:
                    require(path.is_file() and path.stat().st_size == counts[mod] * progress['hidden'] * 2,
                            f'bank length mismatch: {path}')
    if files:
        actual = {p for p in collection.glob('layer_*/*') if p.suffix in ('.bin', '.index', '.phase')}
        require(actual == expected_paths, 'uncheckpointed or missing raw files; do not truncate or rollback')
    return dict(completed=n, total=len(rows), request_bytes=(collection / 'requests.jsonl').stat().st_size,
                collector_identity=progress['identity'], token_counts=counts, raw_file_count=len(expected_paths),
                rng_and_sampling_replayed=True, tensor_content_scans=False)


def process_stamp(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def validate_stop(receipt, collection, completed):
    require(receipt['collection'] == str(collection) and receipt['completed'] == completed, 'stop receipt prefix differs')
    require(receipt.get('submissions_stopped') and receipt.get('server_stopped') and
            receipt.get('inflight_requests') == 0, 'stop receipt does not prove quiescence')
    require(bool(receipt.get('processes')), 'missing stopped process identities')
    for process in receipt['processes']:
        require(process_stamp(process['pid']) != str(process['starttime']), 'old driver/server process is still alive')
    if receipt.get('proof') == 'frontend_admission_barrier':
        require(receipt.get('observed_active_before_drain') == 1 and receipt.get('observed_active_after_drain') == 0
                and receipt.get('prepare_stop_response', {}).get('drain_complete') and receipt.get('no_abort'),
                'frontend barrier lacks a terminal request witness')
    else:
        require(receipt.get('proof') == 'cooperative_request_boundary', 'unsupported stop proof')


def source_receipt(snapshot, baseline):
    values = {}
    for tree in ('python', 'benchmarks'):
        for path in sorted((snapshot / tree).rglob('*.py')):
            name = str(path.relative_to(snapshot))
            values[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    previous = {str(p.relative_to(baseline)): hashlib.sha256(p.read_bytes()).hexdigest()
                for tree in ('python', 'benchmarks') for p in (baseline / tree).rglob('*.py')}
    require(values.keys() == previous.keys(), 'candidate source tree file set changed')
    changes = [name for name in values if values[name] != previous[name]]
    require(changes == ['python/freetoken/moe/tablemoe_builder.py'], 'candidate changes exceed buffered collector')
    return dict(snapshot=str(snapshot), baseline=str(baseline), changed_python_sources=changes,
                source_sha256=values, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def append_response(path, expected_bytes, response):
    payload = (json.dumps(response, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with path.open('ab') as handle:
        require(os.fstat(handle.fileno()).st_size == expected_bytes, 'concurrent request log writer detected')
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return expected_bytes + len(payload)


def resume_requests(rows, completed, collection, run, request, pause_requested):
    size = (collection / 'requests.jsonl').stat().st_size
    for ordinal in range(completed, len(rows)):
        if pause_requested():
            return ordinal
        response = request(rows[ordinal])
        require(response['id'] == rows[ordinal]['id'] and response['usage']['completion_tokens'] == 128
                and response['max_tokens'] == 128 and response['natural_eos'] is False
                and response['finish_reason'] == 'length' and bool(response['prediction_text'].strip()),
                'response does not match the submitted fixed128 request')
        until = time.monotonic() + 30
        while True:
            state = read(collection / 'progress.json')
            count = len(state['requests'])
            if count == ordinal + 1:
                require(len(state['sample_state']['records']) == count and
                        state['sample_state']['records'][-1]['decode_seen'] == 127 and
                        state['sample_state']['records'][-1]['prompt_tokens'] == response['usage']['prompt_tokens'],
                        'terminal response preceded incomplete collector state')
                break
            require(count <= ordinal + 1, 'unexpected concurrent collector request')
            require(time.monotonic() < until, 'collector did not checkpoint the completed response')
            time.sleep(.05)
        size = append_response(collection / 'requests.jsonl', size, response)
        save(run / 'progress.json', dict(completed=ordinal + 1, total=len(rows), request_bytes=size,
                                         stage='collection', complete=False))
    return len(rows)


def main(args):
    folder = ROOT / 'build' / args.dataset
    protocol = read(folder / 'protocol.json')
    inputs = protocol['inputs']
    assets = Path(protocol['assets'])
    collection = assets / 'collect'
    rows = json_lines(ROOT / 'prepare' / args.dataset / 'calibration.jsonl')
    require(args.dataset != 'MMLongBench_DOC' or len(rows) == 655, 'Doc full frozen set must retain all655 rows')
    require(not (folder / 'complete.json').exists(), 'build is already marked complete; no resume needed')
    report = validate_prefix(rows, inputs, read(assets / 'collect_config.json'), collection)
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    require(args.quiescence_receipt is not None, '--execute requires a verified stop receipt')
    validate_stop(read(args.quiescence_receipt), collection, report['completed'])
    snapshot = args.snapshot.resolve()
    provenance = source_receipt(snapshot, ROOT / 'runtime_snapshot_compact')
    writer_validation = read(CANDIDATE / 'buffered_writer_receipt.json')
    driver_validation = read(CANDIDATE / 'resume_driver_cpu_validation.json')
    require(writer_validation['complete'] and writer_validation['source']['candidate_sha256'] ==
            provenance['source_sha256']['python/freetoken/moe/tablemoe_builder.py'], 'buffered writer CPU evidence is stale')
    require(driver_validation['passed'] and driver_validation['driver_sha256'] == provenance['driver_sha256'],
            'resume driver CPU evidence is stale')
    run = folder / 'resume_runs' / str(time.time_ns())
    run.mkdir(parents=True, exist_ok=False)
    save(run / 'source_receipt.json', provenance)
    save(run / 'prefix_validation.json', report)
    save(run / 'prior_stop_receipt.json', read(args.quiescence_receipt))
    lock = (collection / 'resume_driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TMPDIR'] = str(ROOT / 'tmp')
    sys.path[:0] = [str(snapshot / 'python'), str(snapshot / 'benchmarks')]
    from tablemoe_common import running_server, stream_request
    from tablemoe_data import api_messages
    from freetoken.moe.tablemoe_builder import finalize_collection, cluster, materialize
    from freetoken.moe.tablemoe_artifacts import verify_stage

    command = read(folder / 'command.json')['command'].copy()
    while True:
        with socket.socket() as frontend, socket.socket() as backend:
            frontend.bind(('127.0.0.1', 0))
            port = frontend.getsockname()[1]
            try:
                backend.bind(('127.0.0.1', port + 1))
            except OSError:
                continue
            break
    command[command.index('--port') + 1] = str(port)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_HOME='/usr/local/cuda-13.0',
               PYTHONPATH=f'{snapshot}/python:{snapshot}/benchmarks', HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    env['PATH'] = f'{Path(command[0]).parent}:/usr/local/cuda-13.0/bin:' + env.get('PATH', '')
    for key, name in [('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'), ('PYTHONPYCACHEPREFIX', 'pycache'),
                      ('TRITON_CACHE_DIR', 'triton_cache'), ('CUDA_CACHE_PATH', 'cuda_cache'),
                      ('TORCH_EXTENSIONS_DIR', 'torch_extensions')]:
        (ROOT / name).mkdir(exist_ok=True)
        env[key] = str(ROOT / name)
    os.environ.update(env)
    stop = {'requested': False}
    def signal_pause(signum, frame):
        stop['requested'] = True
    signal.signal(signal.SIGINT, signal_pause)
    signal.signal(signal.SIGTERM, signal_pause)
    pause = args.pause_file or run / 'pause_requested.json'
    save(run / 'command.json', dict(command=command, gpu=args.gpu, initial_completed=report['completed'],
                                   pause_file=str(pause), batch_size=1))
    identities = [dict(pid=os.getpid(), starttime=process_stamp(os.getpid()))]
    with running_server(command, run / 'server', snapshot, env, 1800) as server:
        identities.append(dict(pid=server.pid, starttime=process_stamp(server.pid)))
        completed = resume_requests(rows, report['completed'], collection, run,
            lambda row: stream_request(port, {**row, 'messages': api_messages(row['vlmeval_message'])},
                                       128, natural=False, timeout=7200),
            lambda: stop['requested'] or pause.exists())
    verified = validate_prefix(rows, inputs, read(assets / 'collect_config.json'), collection)
    require(verified['completed'] == completed, 'post-stop completed count changed')
    if completed != len(rows):
        save(run / 'pause_receipt.json', dict(proof='cooperative_request_boundary', collection=str(collection),
             completed=completed, submissions_stopped=True, server_stopped=True, inflight_requests=0,
             processes=identities, prefix_validation=verified, complete=False,
             driver_exit_pending=True, raw_power_loss_durability_guaranteed=False))
        print(run / 'pause_receipt.json')
        return 75
    finalize_collection(collection, inputs, 48, 128)
    for stage, fn in [('cluster', cluster), ('materialize', materialize)]:
        save(run / 'progress.json', dict(stage=stage, completed=completed, total=len(rows), complete=False))
        fn(assets, inputs['algorithm'])
    manifest = verify_stage(assets / 'cache', hash_outputs=False)
    progress = read(collection / 'progress.json')
    records = progress['sample_state']['records']
    save(folder / 'complete.json', dict(complete=True, assets=str(assets), calibration_samples=len(rows),
         missing_tables=manifest['missing_tables'], collected_tokens=progress['sample_state']['counts'],
         first_seen_images=sum(len(r['new_image_ids']) for r in records),
         image_count_semantics='first encountered images; token sampling may select zero positions for an image',
         duplicate_visual_tokens=sum(r['duplicate_visual_tokens'] for r in records), tensor_content_scans=False,
         resume_source_receipt=str(run / 'source_receipt.json')))
    save(folder / 'status.json', dict(stage='complete', complete=True, updated_unix=time.time()))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['MMLongBench_DOC', 'Vision2Web_Level1'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--quiescence-receipt', '--boundary-receipt', dest='quiescence_receipt', type=Path)
    parser.add_argument('--snapshot', type=Path, default=CANDIDATE / 'runtime_snapshot')
    parser.add_argument('--pause-file', type=Path)
    sys.exit(main(parser.parse_args()))
