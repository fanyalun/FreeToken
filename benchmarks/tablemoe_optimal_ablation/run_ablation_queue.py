"""Run the best-parameter lookup ablations after the existing MoDES matrix."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback


STOP = threading.Event()
CHILDREN = set()
ACTIVE_RUNNER = None


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deferred(root, identifier):
    path = root / 'job_policy.json'
    policy = read(path) if path.exists() else {}
    names = (identifier, 'score_' + identifier)
    if any(name in policy.get('deferred_jobs', {}) for name in names):
        return True
    state_path = root / 'cells' / identifier / 'ablation_state.json'
    previous = read(state_path) if state_path.exists() else {}
    return previous.get('state') == 'deferred' and not any(
        name in policy.get('resumed_jobs', []) for name in names)


def protocol_jobs(protocol, dataset=None):
    jobs = []
    for stage in protocol['stages']:
        for definition in stage['jobs']:
            if dataset is None or definition['dataset'] == dataset:
                jobs.append((stage, dict(definition)))
    return jobs


def validate_protocol(protocol):
    jobs = protocol_jobs(protocol)
    if len(jobs) != 4 or len({job['id'] for _, job in jobs}) != 4:
        raise ValueError('the optimal ablation requires four distinct configurations')
    for stage, job in jobs:
        if (job['dataset'] not in ('mmw', 'rwqa') or Path(job['id']).name != job['id'] or
                job.get('method', 'tablemoe') != 'tablemoe' or job['alpha'] != .8 or
                job['h'] != 4 or job['slots'] != 3072):
            raise ValueError('ablation differs from the selected alpha/top-h/cache contract')
        if stage['id'] == 'schedule':
            valid = job.get('skip_lookup') and job.get('online_cache_size') == 128
        elif stage['id'] == 'table':
            valid = not job.get('skip_lookup') and job.get('online_cache_size') == 0
        else:
            valid = False
        if not valid:
            raise ValueError('ablation lookup mode is not the requested schedule/offline-table control')
    for dataset in ('mmw', 'rwqa'):
        if [stage['id'] for stage, _ in protocol_jobs(protocol, dataset)] != ['schedule', 'table']:
            raise ValueError('each dataset must run schedule before offline-table')


def load_runner(root, protocol):
    source = Path(protocol.get('controller_source', Path(protocol['base_experiment']) / 'run_experiment.py'))
    if protocol.get('controller_source_sha256') != source_hash(source):
        raise ValueError('the frozen controller source changed')
    spec = importlib.util.spec_from_file_location('optimal_ablation_controller_' + str(os.getpid()), source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT = root
    base = Path(protocol['base_experiment'])
    runner.SNAPSHOT = Path(protocol.get('runtime_snapshot', base / 'runtime_snapshot'))
    runner.JUDGE_SNAPSHOT = Path(protocol.get('judge_runtime_snapshot', base / 'judge_runtime_snapshot'))
    sys.path[:0] = [str(runner.SNAPSHOT / 'python'), str(runner.SNAPSHOT / 'benchmarks')]
    runner.STATE.clear()
    runner.STATE.update(status='optimal_ablation', jobs={}, workers={}, selections=dict(alpha=.8, top_h=4))
    original_command = runner.command

    def command(job, gpu):
        args, port = original_command(job, gpu)
        args[args.index('--tablemoe-online-cache-size') + 1] = str(job['online_cache_size'])
        return args, port

    runner.command = command
    return runner


def manager_alive(base, pid):
    try:
        args = Path(f'/proc/{int(pid)}/cmdline').read_text().split('\0')
        return str(base / 'run_experiment.py') in args
    except (OSError, ValueError, TypeError):
        return False


def prior_work_complete(root, protocol, runner):
    ready = root / 'execution_ready.json'
    if not ready.exists() or not read(ready).get('complete'):
        return False, 'waiting_for_execution_ready'
    base = Path(protocol['base_experiment'])
    state = read(base / 'status.json')
    if manager_alive(base, state.get('manager_pid')):
        return False, 'waiting_for_previous_manager_exit'
    if not state.get('primary_workflow_complete'):
        return False, 'waiting_for_previous_matrix_and_defer_reconciliation'
    original_root = runner.ROOT
    try:
        runner.ROOT = base
        for identifier in protocol['after_cells']:
            marker = base / 'cells' / identifier / 'complete.json'
            if not marker.exists() or not read(marker).get('complete'):
                return False, 'waiting_for_' + identifier
            name = identifier.split('_', 1)[0]
            if runner.cached_score(name, marker.parent) is None:
                return False, 'waiting_for_native_score_' + identifier
    finally:
        runner.ROOT = original_root
    return True, 'previous_work_complete'


def job_binding(root, protocol, runner, job):
    dataset = runner.ALIASES[job['dataset']]
    folder = root / 'prepare' / dataset
    evaluation = folder / 'evaluation.jsonl'
    return dict(job=job, runtime_snapshot=str(runner.SNAPSHOT), judge_runtime_snapshot=str(runner.JUDGE_SNAPSHOT),
                controller_source_sha256=protocol['controller_source_sha256'],
                kit_worker_sha256=source_hash(root / 'kit_worker.py'),
                prompt_protocol_sha256=source_hash(folder / 'prompt_protocol.json'),
                evaluation_path=str(evaluation.resolve()), evaluation_bytes=evaluation.stat().st_size,
                max_tokens=runner.LIMITS[job['dataset']], renormalize_weights=False,
                inputs=next(item for item in protocol['inputs'] if item['dataset'] == dataset))


class AuditCoveragePending(ValueError):
    pass


def audit_lookup(folder, stage, expected_requests=None):
    totals = dict(online_queries=0, online_hits=0, offline_queries=0, offline_hits=0)
    count = 0
    for path in sorted(folder.glob('audit*.jsonl')):
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'):
                continue
            row = json.loads(line)
            if row.get('type') != 'request' or row.get('aborted'):
                continue
            count += 1
            for key in totals:
                totals[key] += row['counters'].get(key, 0)
    if count == 0 or (expected_requests is not None and count != expected_requests):
        raise AuditCoveragePending('request audit coverage is incomplete')
    if totals['online_queries'] or totals['online_hits']:
        raise ValueError('online lookup was active in an ablation that disables it')
    if stage == 'schedule' and (totals['offline_queries'] or totals['offline_hits']):
        raise ValueError('schedule-only ablation unexpectedly queried offline tables')
    if stage == 'table' and totals['offline_queries'] == 0:
        raise ValueError('offline-table ablation did not exercise offline lookup')
    return dict(complete=True, stage=stage, completed_requests=count, counters=totals, checked_utc=now())


def ready_audit(folder, stage, expected_requests=None):
    deadline = time.monotonic() + 5
    while True:
        try:
            return audit_lookup(folder, stage, expected_requests)
        except AuditCoveragePending:
            if time.monotonic() >= deadline:
                raise
            STOP.wait(.1)


def write_comparison(root, protocol):
    with (root / '.comparison.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        datasets = {}
        complete = True
        for dataset, baseline in protocol['baseline_cells'].items():
            baseline_path = Path(protocol['base_experiment']) / 'cells' / baseline / 'complete.json'
            values = dict(tablemoe_best=read(baseline_path)['quality_metric'])
            records = dict(tablemoe_best=str(baseline_path))
            for stage, job in protocol_jobs(protocol, dataset):
                path = root / 'cells' / job['id'] / 'complete.json'
                if path.exists() and read(path).get('complete'):
                    values[stage['id']] = read(path)['quality_metric']
                    records[stage['id']] = str(path)
                else:
                    complete = False
            datasets[dataset] = dict(scores=values, records=records,
                differences_from_best={key: value - values['tablemoe_best']
                                       for key, value in values.items() if key != 'tablemoe_best'})
        save(root / 'comparison.json', dict(complete=complete, alpha=.8, decode_top_h=4, expert_slots=3072,
             datasets=datasets, updated_utc=now(), quality_only=True, formal_performance_eligible=False))


def run_job(root, protocol, runner, stage, job, gpu):
    folder = root / 'cells' / job['id']
    state_path = folder / 'ablation_state.json'
    if deferred(root, job['id']):
        save(state_path, dict(state='deferred', explicit_resume_required=True, updated_utc=now()))
        return False
    binding = job_binding(root, protocol, runner, job)
    runner.STATE['jobs'] = {job['id']: job}
    runner.STATE['workers'][str(gpu)] = dict(state='running', job=job['id'])
    saved_binding = folder / 'binding.json'
    if saved_binding.exists() and read(saved_binding) != binding:
        raise ValueError('saved ablation binding differs from the requested source or protocol')
    if runner.cached_score(job['dataset'], folder) is not None:
        if not saved_binding.exists() or runner.generation_record(job) is None:
            raise ValueError('cached score lacks a bound complete generation')
        ready_audit(folder, stage['id'], protocol['dataset_counts'][job['dataset']])
        runner.score_job(dict(parent=job['id'], dataset=job['dataset']), gpu)
        save(state_path, dict(state='complete', complete=True, gpu=gpu, restored_from_cache=True, updated_utc=now()))
        return True
    try:
        generation = runner.generation_record(job)
    except (ValueError, KeyError, OSError):
        if (folder / 'generation_complete.json').exists():
            raise
        generation = None
    if generation is None and folder.exists():
        archive = root / 'partial_archives' / (job['id'] + '_' + str(time.time_ns()))
        archive.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(archive)
    folder.mkdir(parents=True, exist_ok=True)
    save(saved_binding, binding)
    original_save = runner.save

    def checked_save(path, value):
        original_save(path, value)
        if Path(path) == folder / 'progress.json' and value.get('completed') == 1:
            audit = ready_audit(folder, stage['id'])
            original_save(folder / 'first_sample_lookup_audit.json', audit)

    runner.save = checked_save
    try:
        save(state_path, dict(state='generating' if generation is None else 'scoring', gpu=gpu, updated_utc=now()))
        if generation is None:
            runner.generate(job, gpu)
        if deferred(root, job['id']):
            raise RuntimeError('ablation was deferred before scoring')
        audit = ready_audit(folder, stage['id'], protocol['dataset_counts'][job['dataset']])
        save(folder / 'lookup_audit.json', audit)
        save(state_path, dict(state='scoring', gpu=gpu, updated_utc=now()))
        runner.score_job(dict(parent=job['id'], dataset=job['dataset']), gpu)
        if runner.generation_record(job) is None or runner.cached_score(job['dataset'], folder) is None:
            raise ValueError('ablation worker did not produce a complete native evaluation')
        save(state_path, dict(state='complete', complete=True, gpu=gpu, updated_utc=now()))
        return True
    except BaseException:
        paused = deferred(root, job['id'])
        save(state_path, dict(state='deferred' if paused else 'failed', complete=False,
             explicit_resume_required=paused, error=traceback.format_exc(), updated_utc=now()))
        if paused:
            return False
        raise
    finally:
        runner.save = original_save


def worker(root, identifier, gpu):
    global ACTIVE_RUNNER
    protocol = read(root / 'protocol.json')
    validate_protocol(protocol)
    runner = ACTIVE_RUNNER = load_runner(root, protocol)
    runner.STATE['workers'][str(gpu)] = dict(state='starting')
    runner.persist = lambda: save(root / ('runtime_state_' + identifier + '.json'), runner.STATE)
    stage, job = next((stage, job) for stage, job in protocol_jobs(protocol) if job['id'] == identifier)
    if not stage.get('confirmed') or not stage.get('ready'):
        raise ValueError('the selected stage has not been confirmed and prepared')
    locks = root / 'locks'
    locks.mkdir(exist_ok=True)
    with (locks / (identifier + '.lock')).open('a+') as cell_lock, (locks / f'gpu_{gpu}.lock').open('a+') as gpu_lock:
        fcntl.flock(cell_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = run_job(root, protocol, runner, stage, job, gpu)
    write_comparison(root, protocol)
    return complete


def shutdown(signum, frame):
    STOP.set()
    if ACTIVE_RUNNER is not None:
        ACTIVE_RUNNER.shutdown(signum, frame)
    for process in list(CHILDREN):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    raise KeyboardInterrupt


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.queue.lock').open('a+') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read(root / 'protocol.json')
        validate_protocol(protocol)
        runner = load_runner(root, protocol)
        while not STOP.is_set():
            ready, reason = prior_work_complete(root, protocol, runner)
            save(root / 'queue_status.json', dict(state='waiting' if not ready else 'ready',
                 reason=reason, manager_pid=os.getpid(), updated_utc=now()))
            if ready:
                break
            STOP.wait(5)
        base = Path(protocol['base_experiment'])
        with (base / '.manager.lock').open('a+') as base_lock:
            fcntl.flock(base_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ready, reason = prior_work_complete(root, protocol, runner)
            if not ready:
                raise RuntimeError('previous experiment ownership changed: ' + reason)
            pending = protocol_jobs(protocol)
            active = {}
            claimed = set()
            try:
                while pending or active:
                    for gpu, value in list(active.items()):
                        if deferred(root, value['job']) and not value.get('stop_sent') and value['process'].poll() is None:
                            os.killpg(value['process'].pid, signal.SIGTERM)
                            value['stop_sent'] = True
                        if value['process'].poll() is not None:
                            del active[gpu]
                    for gpu in sorted({0, 1} - set(active)):
                        schedules = {job['id'] for stage, job in protocol_jobs(protocol) if stage['id'] == 'schedule'}
                        candidates = [(stage, job) for stage, job in pending
                                      if stage.get('confirmed') and stage.get('ready') and not deferred(root, job['id'])
                                      and (stage['id'] == 'schedule' or schedules <= claimed)]
                        if not candidates:
                            continue
                        stage, job = min(candidates, key=lambda value: (
                            value[0]['id'] != 'schedule', value[1]['dataset'] != 'mmw'))
                        pending.remove((stage, job))
                        claimed.add(job['id'])
                        with (root / (job['id'] + '.log')).open('a') as log:
                            process = subprocess.Popen([runner.PYTHON, str(Path(__file__).resolve()), '--root', str(root),
                                '--worker-job', job['id'], '--gpu', str(gpu)], cwd=root, env=runner.environment(),
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        CHILDREN.add(process)
                        active[gpu] = dict(process=process, job=job['id'])
                        save(root / (job['id'] + '_launch.json'), dict(pid=process.pid, gpu=gpu, started_utc=now()))
                    save(root / 'queue_status.json', dict(state='running', manager_pid=os.getpid(),
                         workers={gpu: dict(pid=value['process'].pid, job=value['job']) for gpu, value in active.items()},
                         pending=[job['id'] for _, job in pending], updated_utc=now()))
                    if not active:
                        break
                    STOP.wait(5)
                write_comparison(root, protocol)
                complete = read(root / 'comparison.json')['complete']
                save(root / 'queue_status.json', dict(state='complete' if complete else 'incomplete', complete=complete,
                     manager_pid=os.getpid(), workers=[dict(pid=p.pid, returncode=p.returncode) for p in CHILDREN], updated_utc=now()))
            finally:
                for process in list(CHILDREN):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--worker-job')
    parser.add_argument('--gpu', type=int, choices=(0, 1))
    args = parser.parse_args()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        if args.worker_job:
            if args.gpu is None:
                parser.error('--gpu is required for a job worker')
            worker(args.root.resolve(), args.worker_job, args.gpu)
        else:
            main(args.root.resolve())
    except BaseException:
        save(args.root.resolve() / ('queue_error_' + (args.worker_job or 'manager') + '.json'),
             dict(error=traceback.format_exc(), observed_utc=now()))
        raise
