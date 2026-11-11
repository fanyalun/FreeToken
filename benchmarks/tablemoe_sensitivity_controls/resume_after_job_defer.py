"""Load the deferred-job policy after other active experiments finish normally."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


ROOT = Path(os.environ['TABLEMOE_RUN_ROOT']).resolve()
HERE = ROOT / 'protocol_changes/defer_rwqa_hybrid_20261111'
TARGET = 'rwqa_evaluate_prefill'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def manager_alive(pid):
    try:
        return str(ROOT / 'run_experiment.py') in Path(f'/proc/{pid}/cmdline').read_text().split('\0')
    except OSError:
        return False


def main(previous_pid):
    policy = read(ROOT / 'job_policy.json')['deferred_jobs']
    if not all(policy.get(name, {}).get('explicit_resume_required') for name in (TARGET, 'score_' + TARGET)):
        raise ValueError('both the generation and scoring job must require explicit resume')
    save(HERE / 'resume_status.json', dict(stage='waiting_for_other_experiments',
         old_manager_pid=previous_pid, target=TARGET, updated_utc=now()))
    while manager_alive(previous_pid):
        time.sleep(5)
    state = read(ROOT / 'status.json')
    if state['manager_pid'] != previous_pid:
        save(HERE / 'resume_status.json', dict(stage='another_manager_already_resumed',
             manager_pid=state['manager_pid'], updated_utc=now()))
        return
    ready = read(HERE / 'policy_ready.json')
    if not ready.get('complete'):
        raise ValueError('deferred-job controller validation is incomplete')
    spec = importlib.util.spec_from_file_location('deferred_job_controller', ROOT / 'run_experiment.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if not hasattr(runner, 'apply_job_policy'):
        raise ValueError('controller does not contain the validated policy handler')
    save(HERE / 'status_before_policy_resume.json', state)
    with (ROOT / 'manager.log').open('a') as log:
        log.write('\nRESUME_WITH_USER_DEFERRED_JOB ' + now() + '\n')
        log.flush()
        process = subprocess.Popen([runner.PYTHON, str(ROOT / 'run_experiment.py'), '--resume'],
                    cwd=ROOT, env=runner.environment(), stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
    save(HERE / 'resume_status.json', dict(stage='resumed_with_deferred_job_policy',
         manager_pid=process.pid, old_manager_pid=previous_pid, target=TARGET, updated_utc=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manager-pid', type=int, required=True)
    args = parser.parse_args()
    try:
        HERE.mkdir(parents=True, exist_ok=True)
        with (HERE / '.resume_watcher.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main(args.manager_pid)
    except BaseException:
        save(HERE / 'resume_error.json', dict(error=traceback.format_exc(), observed_utc=now()))
        raise
