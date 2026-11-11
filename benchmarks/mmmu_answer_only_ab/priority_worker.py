"""Run one new MMMU comparison before an externally owned pending RW task."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


ROOT = Path(os.environ.get('MMMU_AB_ROOT', Path(__file__).resolve().parent)).resolve()
BASE = ROOT.parent / 'tablemoe_sensitivity_preserve_vlmeval_20261110'
RUNNER = None
CHILD = None


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_runner():
    spec = importlib.util.spec_from_file_location('mmmu_priority_base_runner', BASE / 'run_experiment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wait_execution_ready(status, path):
    status.update(stage='waiting_for_execution_ready', updated_utc=now())
    save(path, status)
    while True:
        ready = ROOT / 'execution_ready.json'
        if ready.exists() and read(ready).get('complete'):
            return
        time.sleep(2)


def verify_lease(followup):
    entry = read(BASE / 'external_jobs.json').get(followup, {})
    if (entry.get('pid') != os.getpid() or
            Path(entry.get('script', '')).resolve() != Path(__file__).resolve()):
        raise RuntimeError('pending RW task has not been leased to this exact priority worker')


def stop_owned_child():
    global CHILD
    if CHILD is None or CHILD.poll() is not None:
        return
    for signum, timeout in ((signal.SIGINT, 45), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(CHILD.pid, signum)
        except ProcessLookupError:
            return
        try:
            CHILD.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError('owned MMMU process did not exit')


def run_mmmu(runner, gpu, cell, folder):
    global CHILD
    command = [runner.PYTHON, str(ROOT / 'run_mmmu_ab.py'), '--gpu', str(gpu), '--cell', cell]
    save(folder / 'mmmu_command.json', dict(command=command, physical_gpu=gpu, created_utc=now()))
    environment = runner.environment()
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        environment[name] = str(ROOT / 'tmp')
    (ROOT / 'tmp').mkdir(exist_ok=True)
    try:
        with (folder / 'mmmu.log').open('a') as log:
            CHILD = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            save(folder / 'mmmu_process.json', dict(pid=CHILD.pid, started_utc=now()))
            result = CHILD.wait()
        if result:
            raise RuntimeError(f'MMMU worker exited {result}; inspect {folder / "mmmu.log"}')
        complete_path = ROOT / 'cells' / cell / 'complete.json'
        if not complete_path.exists() or not read(complete_path).get('complete'):
            raise RuntimeError('MMMU worker exited without complete native scoring')
        return read(complete_path)
    finally:
        stop_owned_child()
        CHILD = None


def followup_job(runner, followup):
    prior = read(BASE / 'status.json')['jobs'][followup]
    expected_h = int(followup.rsplit('_', 1)[1])
    if (prior['dataset'] != 'rwqa' or prior['kind'] != 'generate' or
            prior.get('alpha') != .8 or prior.get('h') != expected_h or prior.get('slots') != 3072 or
            prior.get('method', 'tablemoe') != 'tablemoe' or prior.get('skip_lookup', False)):
        raise ValueError('leased RW task differs from the existing selected-alpha matrix')
    return {**prior, 'state': 'running', 'external_priority_owner_pid': os.getpid()}


def run_followup(runner, gpu, followup, folder):
    job = followup_job(runner, followup)
    runner.STATE.clear()
    runner.STATE.update(status='external_priority_followup', jobs={followup: job},
                        workers={str(gpu): dict(state='running', job=followup)}, selections=dict(alpha=.8))

    def persist_external():
        with runner.LOCK:
            runner.STATE['updated_utc'] = now()
            save(folder / 'rw_runtime_status.json', runner.STATE)

    runner.persist = persist_external
    cell_folder = BASE / 'cells' / followup
    generation = runner.generation_record(job)
    if generation is None and cell_folder.exists():
        archive = ROOT / 'priority/archived_followups' / (followup + '_' + str(time.time_ns()))
        archive.parent.mkdir(parents=True, exist_ok=True)
        cell_folder.rename(archive)
        save(folder / 'partial_followup_archive.json',
             dict(original=str(cell_folder), archive=str(archive), fresh_cache_replay_required=True,
                  archived_utc=now()))
    persist_external()
    if generation is None:
        runner.generate(job, gpu)
    score_job = dict(id='score_' + followup, kind='score', dataset='rwqa', parent=followup)
    runner.score_job(score_job, gpu)
    if runner.generation_record(job) is None or runner.cached_score('rwqa', cell_folder) is None:
        raise RuntimeError('external RW followup lacks complete frozen generation or native score')
    job['state'] = 'complete'
    persist_external()
    return read(cell_folder / 'complete.json')


def run(gpu, cell, followup):
    global RUNNER
    folder = ROOT / 'priority' / followup
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / 'status.json'
    status = dict(pid=os.getpid(), gpu=gpu, cell=cell, followup=followup,
                  started_utc=now(), complete=False)
    wait_execution_ready(status, status_path)
    verify_lease(followup)
    RUNNER = load_runner()
    status.update(stage='running_mmmu', updated_utc=now())
    save(status_path, status)
    try:
        run_mmmu(RUNNER, gpu, cell, folder)
        status.update(mmmu_complete=True, mmmu_finished_utc=now())
    except Exception:
        status.update(mmmu_complete=False, mmmu_error=traceback.format_exc(), mmmu_finished_utc=now())
        save(folder / 'mmmu_failure.json', dict(error=status['mmmu_error'], observed_utc=now()))
    status.update(stage='running_existing_rw_followup', updated_utc=now())
    save(status_path, status)
    run_followup(RUNNER, gpu, followup, folder)
    status.update(stage='complete', followup_complete=True,
                  complete=status['mmmu_complete'], updated_utc=now())
    save(status_path, status)
    return status


def interrupted(signum, frame):
    if RUNNER is not None:
        RUNNER.STOP.set()
        for process in list(RUNNER.CHILDREN):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
    raise KeyboardInterrupt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--cell', choices=('mmmu_lookup', 'mmmu_schedule'), required=True)
    parser.add_argument('--followup', choices=('rwqa_top_h_5', 'rwqa_top_h_6'), required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    lock_folder = ROOT / 'priority'
    lock_folder.mkdir(exist_ok=True)
    try:
        with (lock_folder / (args.followup + '.lock')).open('a+') as followup_lock, \
                (lock_folder / ('gpu_' + str(args.gpu) + '.lock')).open('a+') as gpu_lock:
            fcntl.flock(followup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run(args.gpu, args.cell, args.followup)
        raise SystemExit(0 if result['complete'] else 1)
    except (Exception, KeyboardInterrupt):
        save(lock_folder / args.followup / 'worker_exit.json',
             dict(error=traceback.format_exc(), observed_utc=now(),
                  followup_complete=(BASE / 'cells' / args.followup / 'complete.json').exists()))
        raise
