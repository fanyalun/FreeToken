"""Dispatch prepared single-cell followups after existing GPU owners release their devices."""

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


CHILDREN = set()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def owner_alive(owner):
    try:
        return owner['script_argument'] in Path(f"/proc/{owner['pid']}/cmdline").read_text().split('\0')
    except OSError:
        return False


def gpu_resource(gpu):
    data = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).strip().split(',')
    uuid, free = data[0].strip(), int(data[1])
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    pids = [int(line.split(',')[1]) for line in apps.splitlines() if line.split(',')[0].strip() == uuid]
    return dict(gpu=gpu, free_mib=free, compute_pids=pids)


def available(resource, owners):
    return resource['free_mib'] >= 76 * 1024 and not resource['compute_pids'] and not any(owner_alive(o) for o in owners)


def report(root, protocol):
    results = {}
    for name, path in protocol['comparison_records'].items():
        path = Path(path)
        if path.exists() and read(path).get('complete'):
            results[name] = read(path)['quality_metric']
    save(root / 'comparison.json', dict(complete=len(results) == len(protocol['comparison_records']),
         scores=results, alpha=.7, decode_top_h=4, expert_slots=3072,
         quality_only=True, formal_performance_eligible=False, updated_utc=now()))


def main(root):
    protocol = read(root / 'queue_protocol.json')
    active, failures = {}, []
    with (root / '.manager.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = [job for job in protocol['jobs'] if not (Path(job['root']) / 'comparison.json').exists() or
                   not read(Path(job['root']) / 'comparison.json').get('complete')]
        try:
            while pending or active:
                for gpu, entry in list(active.items()):
                    process = entry['process']
                    if process.poll() is not None:
                        if process.returncode:
                            failures.append(entry['job']['id'])
                        CHILDREN.discard(process)
                        del active[gpu]
                        report(root, protocol)
                resources = {}
                for gpu in sorted(set(protocol['allowed_gpus']) - set(active)):
                    resource = gpu_resource(gpu)
                    owners = [o for o in protocol['existing_owners'] if o['gpu'] == gpu]
                    resource['live_owner_pids'] = [o['pid'] for o in owners if owner_alive(o)]
                    resources[gpu] = resource
                    if not pending or not available(resource, owners):
                        continue
                    job = pending.pop(0)
                    directory = Path(job['root'])
                    spec = importlib.util.spec_from_file_location('followup_environment', directory / 'controller_source.py')
                    runner = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(runner)
                    runner.ROOT = directory
                    runner.SNAPSHOT = Path(read(directory / 'protocol.json')['runtime_snapshot'])
                    with (directory / 'worker.log').open('a') as log:
                        process = subprocess.Popen([runner.PYTHON, str(directory / 'run_single_cell.py'),
                             '--root', str(directory), '--gpu', str(gpu)], cwd=directory,
                             env=runner.environment(gpu), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    CHILDREN.add(process)
                    active[gpu] = dict(process=process, job=job)
                    save(directory / 'launch.json', dict(pid=process.pid, gpu=gpu, started_utc=now(), job=job['id']))
                save(root / 'queue_status.json', dict(state='running' if active else 'waiting_for_gpu',
                     manager_pid=os.getpid(), workers={gpu: dict(pid=v['process'].pid, job=v['job']['id'])
                     for gpu, v in active.items()}, pending=[j['id'] for j in pending], failed=failures,
                     resources=resources, updated_utc=now()))
                if pending or active:
                    time.sleep(5)
            report(root, protocol)
            save(root / 'queue_status.json', dict(state='complete' if not failures else 'failed',
                 complete=not failures, workers={}, pending=[], failed=failures, updated_utc=now()))
        finally:
            for process in list(CHILDREN):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)


def shutdown(signum, frame):
    for process in list(CHILDREN):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    raise KeyboardInterrupt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    main(args.root.resolve())
