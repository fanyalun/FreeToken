"""Compare TableMoE lookup and expert skipping with frozen MMMU answer-only prompts."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(os.environ.get('MMMU_AB_ROOT', str(Path(__file__).resolve().parent))).resolve()
SOURCE_ROOT = ROOT.parent / 'tablemoe_sensitivity_preserve_vlmeval_20261110'
SNAPSHOT = SOURCE_ROOT / 'runtime_snapshot'
PYTHON = '/home/fanya/local_archives/freetoken_consolidation_20261029_020126/preserved/FreeToken/.venv/bin/python'
KIT_PYTHON = '/home/fanya/miniforge/envs/tablemoe/bin/python'
KIT = Path('/home/fanya/tablemoe/third_party/VLMEvalKit')
MODEL = '/data/fanya/models/Qwen3-VL-30B-A3B-Instruct'
CELLS = ('mmmu_lookup', 'mmmu_schedule')
DATASET = 'MMMU_DEV_VAL'
MAX_TOKENS = 256
STOP = False
SERVER = None
sys.path[:0] = [str(SNAPSHOT / 'python'), str(SNAPSHOT / 'benchmarks')]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    partial.replace(path)


def environment(gpu=None, scoring=False):
    env = {**os.environ, 'CUDA_HOME': '/usr/local/cuda-13.0', 'HF_HUB_OFFLINE': '1',
           'HF_DATASETS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'PYTHONPATH': str(SNAPSHOT / 'python') + ':' + str(SNAPSHOT / 'benchmarks'),
           'LMUData': '/home/fanya/tablemoe/LMUData', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
           'PRED_FILE_FORMAT': 'xlsx', 'EVAL_FILE_FORMAT': 'csv'}
    for key, directory in [('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'),
            ('PYTHONPYCACHEPREFIX', 'pycache'), ('TRITON_CACHE_DIR', 'triton_cache'),
            ('CUDA_CACHE_PATH', 'cuda_cache'), ('TORCH_EXTENSIONS_DIR', 'torch_extensions')]:
        target = ROOT / directory
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    env['PATH'] = str(Path(PYTHON).parent) + ':/usr/local/cuda-13.0/bin:' + env.get('PATH', '')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if scoring:
        env['LD_LIBRARY_PATH'] = '/home/fanya/miniforge/envs/tablemoe/lib'
    for key in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        env.pop(key, None)
    return env


def rows():
    from tablemoe_data import read_rows
    result = read_rows(ROOT / 'prepare' / DATASET / 'evaluation.jsonl')
    if len(result) != 420 or len({row['id'] for row in result}) != 420:
        raise ValueError('MMMU requires the frozen 420 distinct evaluation samples')
    counts = {kind: sum(row['metadata']['question_type'] == kind for row in result)
              for kind in ('multiple-choice', 'open')}
    if counts != {'multiple-choice': 398, 'open': 22}:
        raise ValueError('MMMU question type coverage changed')
    return result


def complete_predictions(folder, expected):
    path = folder / 'requests.jsonl'
    if not path.exists():
        return None
    raw = path.read_bytes().splitlines(keepends=True)
    if not raw or any(not line.endswith(b'\n') for line in raw):
        return None
    actual = [json.loads(line) for line in raw]
    if len(actual) != len(expected):
        return None
    if ([row['id'] for row in actual] != [row['id'] for row in expected] or
            any(row.get('max_tokens') != MAX_TOKENS or not row.get('natural_eos') or
                row.get('usage', {}).get('completion_tokens', MAX_TOKENS + 1) > MAX_TOKENS
                for row in actual)):
        raise ValueError('generation does not match the frozen evaluation order and settings')
    return actual


def command(cell, gpu):
    protocol = read(ROOT / 'protocol.json')
    inputs = next(item for item in protocol['inputs'] if item['dataset'] == DATASET)
    port = 31600 + gpu * 100
    args = [PYTHON, '-m', 'freetoken.cli', 'serve', '--model', MODEL, '--gpu', '0',
            '--dtype', 'float16', '--moe-strategy', 'tablemoe', '--moe-cache-size', '6144',
            '--moe-cpu-threads', '8', '--cuda-graph-max-bs', '1', '--max-running-requests', '1',
            '--num-pages', '16384', '--page-size', '1', '--cache-type', 'naive',
            '--mm-encoder-weights', 'gpu', '--max-prefill-length', '4096',
            '--max-seq-len-override', '32768', '--port', str(port), '--reasoning-parser', 'off',
            '--mm-processor-kwargs', json.dumps(dict(size=dict(shortest_edge=602112, longest_edge=4014080))),
            '--tablemoe-implementation', 'optimized', '--tablemoe-expert-mode', 'offload',
            '--tablemoe-convert-bf16-weights', '--tablemoe-no-renormalize-weights',
            '--tablemoe-prefill-chunk-size', '0', '--tablemoe-prefill-experts-per-group', '16',
            '--tablemoe-online-cache-size', '128', '--tablemoe-prefill-keep-rate', '.8',
            '--tablemoe-decode-top-h', '8', '--tablemoe-pca-dir', inputs['pca_dir'],
            '--tablemoe-cache-dir', inputs['cache_dir'], '--tablemoe-offline-cosine-threshold-vision', '.6',
            '--tablemoe-offline-cosine-threshold-text', '.7', '--tablemoe-decode-online-cosine-threshold', '.7',
            '--tablemoe-decode-offline-cosine-threshold', '.7',
            '--tablemoe-report-path', str(ROOT / 'cells' / cell / 'audit.jsonl')]
    if cell == 'mmmu_schedule':
        args.append('--tablemoe-skip-lookup')
    return args, port


def resource(gpu):
    text = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    free, utilization = map(int, text.strip().split(','))
    return dict(gpu=gpu, free_mib=free, utilization_percent=utilization, observed_utc=now())


@contextmanager
def admitted_gpu(gpu, folder):
    lock_path = ROOT / f'.gpu_{gpu}.lock'
    with lock_path.open('a+') as lease:
        while not STOP:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                save(folder / 'status.json', dict(stage='waiting_for_gpu_lease', gpu=gpu, updated_utc=now()))
                time.sleep(2)
        while not STOP:
            observation = resource(gpu)
            if observation['free_mib'] >= 74 * 1024:
                save(folder / 'resource_start.json', observation)
                yield
                return
            save(folder / 'status.json', dict(stage='waiting_for_gpu_capacity', gpu=gpu,
                 resource=observation, updated_utc=now()))
            time.sleep(2)
        raise RuntimeError('MMMU run stopped before GPU admission')


def stop_server():
    global SERVER
    if SERVER is not None:
        from tablemoe_common import stop
        stop(SERVER)
        SERVER = None


def generate(cell, gpu, folder, expected):
    global SERVER
    from tablemoe_common import stream_request, wait_ready
    from freetoken.server.args import parse_args
    args, port = command(cell, gpu)
    resolved, _ = parse_args(args[args.index('serve') + 1:])
    if (resolved.tablemoe_renormalize_weights or resolved.tablemoe_decode_keep_k != 8 or
            resolved.moe_cache_size != 6144 or resolved.tablemoe_prefill_keep_rate != .8):
        raise ValueError('resolved TableMoE configuration changed')
    save(folder / 'resolved_selection.json', dict(alpha=.8, decode_top_h=8, expert_slots=6144,
         tablemoe_renormalize_weights=False, skip_lookup=cell == 'mmmu_schedule', max_tokens=MAX_TOKENS,
         prompt_protocol=str(ROOT / 'prepare' / DATASET / 'prompt_protocol.json')))
    server_folder = folder / 'server'
    server_folder.mkdir()
    save(server_folder / 'command.json', dict(command=args, physical_gpu=gpu, source=str(SNAPSHOT)))
    started = time.monotonic()
    with admitted_gpu(gpu, folder):
        with (server_folder / 'server.log').open('w') as log:
            try:
                SERVER = subprocess.Popen(args, cwd=SNAPSHOT, env=environment(gpu), stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                save(server_folder / 'process.json', dict(pid=SERVER.pid, gpu=gpu, started_utc=now()))
                wait_ready(SERVER, server_folder / 'server.log', timeout=1800)
                with (folder / 'requests.jsonl').open('w') as output:
                    for position, row in enumerate(expected, 1):
                        if STOP:
                            raise RuntimeError('MMMU generation stopped')
                        value = stream_request(port, row, MAX_TOKENS, natural=True, timeout=1800,
                                               model=Path(MODEL).name)
                        output.write(json.dumps(value, ensure_ascii=False) + '\n')
                        output.flush()
                        save(folder / 'progress.json', dict(completed=position, total=len(expected),
                             last_id=row['id'], elapsed_seconds=time.monotonic() - started, updated_utc=now()))
                        save(folder / 'status.json', dict(stage='generating', gpu=gpu, completed=position,
                             total=len(expected), updated_utc=now()))
                        print('GENERATED', cell, position, len(expected), flush=True)
            finally:
                stop_server()
    if complete_predictions(folder, expected) is None:
        raise ValueError('generation finished without all 420 responses')
    save(folder / 'generation_complete.json', dict(complete=True, samples=len(expected),
         elapsed_seconds=time.monotonic() - started, finished_utc=now(), quality_only=True,
         formal_performance_eligible=False))


def aggregate_hits(expected, records):
    index_map = {str(row['index']): row for row in expected}
    if len(records) != len(expected) or {str(row['index']) for row in records} != set(index_map):
        raise ValueError('native scoring lost, duplicated, or added evaluation indices')
    groups = {'overall': [], 'multiple_choice': [], 'open': [], 'dev': [], 'validation': []}
    for record in records:
        row = index_map[str(record['index'])]
        hit = record['hit']
        if hit not in (0, 1):
            raise ValueError('native scoring returned an invalid per-sample hit')
        groups['overall'].append(int(hit))
        groups['multiple_choice' if row['metadata']['question_type'] == 'multiple-choice' else 'open'].append(int(hit))
        groups[row['metadata']['split']].append(int(hit))
    return {key: dict(correct=sum(hits), samples=len(hits), accuracy_pct=100 * sum(hits) / len(hits))
            for key, hits in groups.items()}


def score(cell):
    from tablemoe_common import prediction_text
    from tablemoe_vlmeval import kit_fingerprint, load_dataset
    import pandas as pd
    if kit_fingerprint(KIT) != read(ROOT / 'prepare/kit.json'):
        raise ValueError('VLMEvalKit changed after prompt preparation')
    expected = rows()
    folder = ROOT / 'cells' / cell
    predictions = complete_predictions(folder, expected)
    if predictions is None:
        raise ValueError('scoring requires all frozen predictions')
    output = folder / 'score'
    if output.exists():
        archive = folder / f'score_incomplete_{time.time_ns()}'
        output.rename(archive)
    output.mkdir()
    dataset = load_dataset(KIT, DATASET, True)
    from vlmeval.smp import dump, load, get_intermediate_file_path
    from vlmeval.dataset.utils.multiple_choice import extract_answer_from_item
    frame = pd.DataFrame([{**row['metadata'], 'index': row['index'], 'prediction': prediction_text(value)}
                          for row, value in zip(expected, predictions)])
    prediction_file = output / 'predictions.xlsx'
    dump(frame, str(prediction_file))
    native = dataset.evaluate(str(prediction_file), model='exact_matching', nproc=4)
    result_path = Path(get_intermediate_file_path(str(prediction_file), '_exact_matching_result'))
    result = load(str(result_path))
    records = json.loads(result.to_json(orient='records'))
    totals = aggregate_hits(expected, records)
    by_index = {str(row['index']): row for row in expected}
    prediction_map = {str(value['index']): value for value in predictions}
    parsed, failures = [], []
    for record in records:
        row = by_index[str(record['index'])]
        value = prediction_map[str(record['index'])]
        extracted = extract_answer_from_item(None, pd.Series(record), dataset_name=DATASET)
        text = prediction_text(value)
        choices = [key for key in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                   if row['metadata'].get(key) is not None and str(row['metadata'][key]) != 'nan']
        strict = bool(re.fullmatch(r'\s*[A-Z]\s*', text)) and text.strip() in choices
        item = dict(id=row['id'], index=row['index'], question_type=row['metadata']['question_type'],
                    split=row['metadata']['split'], prediction=text, native_option=extracted['opt'],
                    native_hit=record['hit'], strict_single_letter=strict,
                    finish_reason=value['finish_reason'], completion_tokens=value['usage']['completion_tokens'])
        parsed.append(item)
        if extracted['opt'] == 'Z':
            failures.append(item)
    from tablemoe_data import write_rows
    write_rows(output / 'native_per_sample.jsonl', records)
    write_rows(output / 'answer_parsing.jsonl', parsed)
    write_rows(output / 'parse_failures.jsonl', failures)
    mc = [item for item in parsed if item['question_type'] == 'multiple-choice']
    diagnostics = dict(native_parse_failures=len(failures), multiple_choice_strict_single_letter=sum(item['strict_single_letter'] for item in mc),
                       multiple_choice_samples=len(mc), length_capped=sum(item['finish_reason'] == 'length' for item in parsed),
                       completion_tokens_mean=sum(item['completion_tokens'] for item in parsed) / len(parsed),
                       completion_tokens_max=max(item['completion_tokens'] for item in parsed))
    summary = dict(complete=True, dataset=DATASET, sample_count=len(expected), scoring_denominator=len(expected),
                   accuracy_pct=totals['overall']['accuracy_pct'], primary_metric='multiple_choice_accuracy_pct',
                   multiple_choice_accuracy_pct=totals['multiple_choice']['accuracy_pct'],
                   open_accuracy_pct=totals['open']['accuracy_pct'], group_metrics=totals,
                   quality_metric=totals['multiple_choice']['accuracy_pct'], scoring_protocol=dict(model='exact_matching', nproc=4),
                   native_results=json.loads(native.to_json(orient='records')), native_result_file=str(result_path),
                   aggregation='sum of native per-sample hit over all frozen indices; dev and validation both included',
                   diagnostics=diagnostics, scored_utc=now())
    save(output / 'score.json', summary)
    save(output / 'complete.json', summary)
    save(folder / 'complete.json', summary)
    save(folder / 'status.json', dict(stage='complete', complete=True, updated_utc=now()))


def compare():
    with (ROOT / '.comparison.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        paths = {cell: ROOT / 'cells' / cell / 'complete.json' for cell in CELLS}
        if not all(path.exists() and read(path).get('complete') for path in paths.values()):
            return
        results = {cell: read(path) for cell, path in paths.items()}
        deltas = {group: results['mmmu_lookup']['group_metrics'][group]['accuracy_pct'] -
                  results['mmmu_schedule']['group_metrics'][group]['accuracy_pct']
                  for group in results['mmmu_lookup']['group_metrics']}
        save(ROOT / 'comparison.json', dict(complete=True, alpha=.8, decode_top_h=8, expert_slots=6144,
             tablemoe_renormalize_weights=False, max_tokens=MAX_TOKENS, primary_metric='multiple_choice_accuracy_pct',
             primary_denominator=398, results=results, lookup_minus_schedule_points=deltas,
             protocol=str(ROOT / 'protocol.json'), completed_utc=now(), quality_only=True,
             formal_performance_eligible=False))


def run(cell, gpu):
    expected = rows()
    folder = ROOT / 'cells' / cell
    lock_folder = ROOT / 'locks'
    lock_folder.mkdir(parents=True, exist_ok=True)
    with (lock_folder / (cell + '.lock')).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'complete.json').exists():
            summary = read(folder / 'complete.json')
            if summary.get('complete') and summary.get('scoring_denominator') == len(expected):
                compare()
                return
        predictions = complete_predictions(folder, expected) if folder.exists() else None
        if predictions is None and folder.exists():
            archive = ROOT / 'partial_archives'
            archive.mkdir(exist_ok=True)
            target = archive / f'{cell}_{time.time_ns()}'
            folder.rename(target)
            save(target / 'replay_reason.json', dict(reason='fresh cache-order replay for incomplete approximate generation', archived_utc=now()))
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / 'launch.json', dict(cell=cell, gpu=gpu, pid=os.getpid(), started_utc=now(), source=str(SNAPSHOT)))
        try:
            if predictions is None:
                generate(cell, gpu, folder, expected)
            elif not (folder / 'generation_complete.json').exists():
                save(folder / 'generation_complete.json', dict(complete=True, samples=len(expected),
                     recovered_from_full_responses=True, finished_utc=now(), quality_only=True, formal_performance_eligible=False))
            with (folder / 'score_worker.log').open('a') as log:
                subprocess.run([KIT_PYTHON, str(Path(__file__).resolve()), '--cell', cell, '--score-only'],
                               cwd=SNAPSHOT, env=environment(scoring=True), stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            compare()
        except BaseException:
            save(folder / 'error.json', dict(error=traceback.format_exc(), observed_utc=now()))
            save(folder / 'status.json', dict(stage='failed', complete=False, updated_utc=now()))
            raise


def shutdown(signum, frame):
    global STOP
    STOP = True
    raise KeyboardInterrupt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=(0, 1))
    parser.add_argument('--cell', required=True, choices=CELLS)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    if args.score_only:
        score(args.cell)
    else:
        if args.gpu is None:
            parser.error('--gpu is required for generation')
        run(args.cell, args.gpu)
