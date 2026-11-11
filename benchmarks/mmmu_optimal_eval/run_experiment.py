"""Run MMMU generation, joint MoDES calibration, and open-answer judging on two GPUs."""

import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from common import CELLS, DATASET, job, load_runner, now, predictions, read, rows, save, source_hash


ACTIVE = None
CHILDREN = set()


def task_marker(root, task):
    if task == 'calibrate':
        return root / 'calibration/mmmu/prefill_decode/complete.json'
    stage, cell = task.split(':')
    return root / 'cells' / cell / ('generation_complete.json' if stage == 'generate' else 'complete.json')


def done(root, task):
    marker = task_marker(root, task)
    if not marker.exists() or not read(marker).get('complete', False):
        return False
    if task.startswith('generate:'):
        audit = marker.parent / 'lookup_audit.json'
        return audit.exists() and read(audit).get('complete', False)
    return True


def audit_generation(root, cell):
    expected, actual = predictions(root, cell)
    reports = [r for r in rows(root / 'cells' / cell / 'audit.jsonl') if r.get('type') == 'request']
    if len(reports) != len(expected) or any(r.get('aborted') for r in reports):
        raise ValueError('route audit does not cover all completed requests')
    totals = {key: sum(r['counters'].get(key, 0) for r in reports)
              for key in ['online_queries', 'online_hits', 'offline_queries', 'offline_hits']}
    if cell in ('mmmu_schedule', 'mmmu_modes') and any(totals.values()):
        raise ValueError('skip-only method performed lookup')
    if cell == 'mmmu_tablemoe' and totals['offline_queries'] + totals['online_queries'] == 0:
        raise ValueError('TableMoE performed no lookup')
    save(root / 'cells' / cell / 'lookup_audit.json', dict(complete=True, samples=len(actual),
         counters=totals, verified_utc=now()))


def calibrate(root, runner):
    source = root / 'calibrate_source.py'
    protocol = read(root / 'protocol.json')
    if source_hash(source) != protocol['source_hashes']['calibrate_source.py']:
        raise ValueError('frozen calibration source changed')
    runner.calibration_rows = lambda name: rows(root / 'prepare' / DATASET / 'calibration.jsonl')
    sys.modules['run_experiment'] = runner
    spec = importlib.util.spec_from_file_location('mmmu_joint_calibration', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.SNAPSHOT = root, runner.SNAPSHOT
    deadline = time.time() + protocol['calibration']['working_budget_seconds']
    save(root / 'calibration/launch.json', dict(scope='prefill_decode', target_skip=.2,
         deadline=deadline, maximum_seconds=3600, started_utc=now()))
    module.main('mmmu', .8, deadline, scopes=('prefill_decode',))
    result = read(root / 'calibration/mmmu/prefill_decode/complete.json')
    if not result.get('complete') or not .2 - 1e-12 <= result['actual_skip'] <= .21:
        raise ValueError('joint calibration did not meet the skip target')
    save(root / 'calibration/complete.json', dict(result, requested_scopes_complete=True,
         calibration_samples=len(runner.calibration_rows('mmmu')), heldout_samples_used=0))


def score(root, runner, cell, gpu):
    predictions(root, cell)
    folder = root / 'cells' / cell
    port = 31820 + gpu * 100
    judge = [runner.PYTHON, '-m', 'freetoken.cli', 'serve', '--model', runner.MODEL, '--gpu', '0',
             '--dtype', 'bfloat16', '--moe-strategy', 'fused', '--moe-cache-size', '0',
             '--cuda-graph-max-bs', '4', '--max-running-requests', '4', '--num-pages', '32768',
             '--page-size', '1', '--cache-type', 'naive', '--mm-encoder-weights', 'gpu',
             '--max-prefill-length', '4096', '--max-seq-len-override', '32768', '--port', str(port),
             '--mm-processor-kwargs', '{"size":{"shortest_edge":602112,"longest_edge":4014080}}',
             '--reasoning-parser', 'off']
    with runner.server(judge, folder / 'judge_server', gpu, snapshot=runner.JUDGE_SNAPSHOT):
        args = [runner.KIT_PYTHON, str(root / 'score_answers.py'), '--root', str(root), '--cell', cell,
                '--port', str(port)]
        with (folder / 'score_worker.log').open('a') as log:
            process = subprocess.Popen(args, cwd=root, env=runner.environment(scoring=True),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            CHILDREN.add(process)
            try:
                if process.wait():
                    raise RuntimeError('MMMU scoring failed; see score_worker.log')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                CHILDREN.discard(process)


def worker(root, task, gpu):
    global ACTIVE
    runner = ACTIVE = load_runner(root, task.replace(':', '_'))
    runner.STATE['workers'][str(gpu)] = dict(state='starting', task=task)
    folder = root / 'tasks' / task.replace(':', '_')
    folder.mkdir(parents=True, exist_ok=True)
    with (root / f'.gpu_{gpu}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if done(root, task):
            return
        runner.wait_capacity(gpu, 76 if task.startswith('score:') else 74)
        save(folder / 'state.json', dict(state='running', gpu=gpu, pid=os.getpid(), started_utc=now()))
        try:
            if task == 'calibrate':
                calibrate(root, runner)
            else:
                stage, cell = task.split(':')
                if stage == 'generate':
                    cell_folder = root / 'cells' / cell
                    if (cell_folder / 'requests.jsonl').exists():
                        archive = root / 'partial_archives'
                        archive.mkdir(exist_ok=True)
                        cell_folder.rename(archive / f'{cell}_{time.time_ns()}')
                    runner.generate(job(cell), gpu)
                    audit_generation(root, cell)
                elif stage == 'score':
                    score(root, runner, cell, gpu)
                else:
                    raise ValueError('unknown task stage')
            if not done(root, task):
                raise ValueError('task returned without a completion record')
            save(folder / 'state.json', dict(state='complete', complete=True, gpu=gpu, finished_utc=now()))
        except BaseException:
            save(folder / 'error.json', dict(error=traceback.format_exc(), observed_utc=now()))
            save(folder / 'state.json', dict(state='failed', complete=False, gpu=gpu, finished_utc=now()))
            raise


def comparison(root):
    results = {}
    for cell in CELLS:
        path = root / 'cells' / cell / 'complete.json'
        if path.exists() and read(path).get('complete'):
            value = read(path)
            results[cell] = {key: value[key] for key in ['correct', 'scoring_denominator', 'accuracy_pct']}
    complete = len(results) == len(CELLS)
    save(root / 'comparison.json', dict(complete=complete, results=results, alpha=.8, decode_top_h=4,
         expert_slots=3072, primary_metric='accuracy_pct', scoring_denominator=420,
         open_questions_scored_by_llm=True, dev_val_separately_reported=False,
         quality_only=True, formal_performance_eligible=False, updated_utc=now()))
    return complete


def main(root):
    runner = load_runner(root, 'manager')
    tasks = ['generate:mmmu_tablemoe', 'calibrate', 'generate:mmmu_modes', 'generate:mmmu_schedule',
             'score:mmmu_tablemoe', 'score:mmmu_modes', 'score:mmmu_schedule']
    active, failed = {}, []
    with (root / '.manager.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = [task for task in tasks if not done(root, task)]
        try:
            while pending or active:
                for gpu, value in list(active.items()):
                    process, task = value['process'], value['task']
                    if task == 'calibrate' and time.monotonic() - value['start'] > 3600 and process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                    if process.poll() is not None:
                        if process.returncode or not done(root, task):
                            failed.append(task)
                        CHILDREN.discard(process)
                        del active[gpu]
                        comparison(root)
                allowed = set(read(root / 'gpu_usage_policy.json')['allowed_gpus'])
                for gpu in sorted(allowed - set(active)):
                    if runner.resource(gpu)['free_mib'] < 76 * 1024:
                        continue
                    ready = [task for task in pending if
                             (task != 'generate:mmmu_modes' or done(root, 'calibrate')) and
                             (not task.startswith('score:') or done(root, task.replace('score:', 'generate:')))]
                    if not ready:
                        continue
                    task = ready[0]
                    pending.remove(task)
                    log_path = root / 'tasks' / (task.replace(':', '_') + '.log')
                    with log_path.open('a') as log:
                        process = subprocess.Popen([runner.PYTHON, str(root / 'run_experiment.py'), '--root', str(root),
                             '--task', task, '--gpu', str(gpu)], cwd=root, env=runner.environment(gpu),
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    CHILDREN.add(process)
                    active[gpu] = dict(process=process, task=task, start=time.monotonic())
                save(root / 'status.json', dict(state='running', manager_pid=os.getpid(),
                     workers={gpu: dict(pid=v['process'].pid, task=v['task']) for gpu, v in active.items()},
                     pending=pending, failed=failed, updated_utc=now()))
                if not active and pending:
                    blocked = [task for task in pending if
                               (task == 'generate:mmmu_modes' and 'calibrate' in failed) or
                               (task.startswith('score:') and task.replace('score:', 'generate:') in failed) or
                               (task == 'score:mmmu_modes' and 'calibrate' in failed)]
                    if set(blocked) == set(pending):
                        break
                time.sleep(5)
            complete = comparison(root)
            save(root / 'status.json', dict(state='complete' if complete else 'incomplete', complete=complete,
                 workers={}, pending=pending, failed=failed, updated_utc=now()))
        finally:
            for process in list(CHILDREN):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)


def shutdown(signum, frame):
    for process in list(CHILDREN):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    if ACTIVE is not None:
        ACTIVE.shutdown(signum, frame)
    raise KeyboardInterrupt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task')
    parser.add_argument('--gpu', type=int, choices=[0, 1])
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    if args.task:
        if args.gpu is None:
            parser.error('--gpu is required for workers')
        worker(args.root.resolve(), args.task, args.gpu)
    else:
        main(args.root.resolve())
