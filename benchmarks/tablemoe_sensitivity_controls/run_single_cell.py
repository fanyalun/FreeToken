"""Run one explicitly requested sensitivity cell with the frozen generation and judge protocol."""

import argparse
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import signal
import sys
import traceback


def configure_lookup(runner, job):
    if 'online_cache_size' not in job:
        return
    original = runner.command

    def command(definition, gpu):
        args, port = original(definition, gpu)
        args[args.index('--tablemoe-online-cache-size') + 1] = str(job['online_cache_size'])
        return args, port

    runner.command = command


def run(root, gpu):
    import json

    protocol = json.loads((root / 'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen orchestration source changed: ' + name)
    spec = importlib.util.spec_from_file_location('single_cell_controller', root / 'controller_source.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT = root
    runner.SNAPSHOT = Path(protocol['runtime_snapshot'])
    runner.JUDGE_SNAPSHOT = Path(protocol['judge_runtime_snapshot'])
    sys.path[:0] = [str(runner.SNAPSHOT / 'python'), str(runner.SNAPSHOT / 'benchmarks')]
    job = dict(protocol['job'])
    configure_lookup(runner, job)
    runner.STATE.clear()
    runner.STATE.update(status='starting', jobs={job['id']: job}, workers={str(gpu): dict(state='starting')})
    signal.signal(signal.SIGTERM, runner.shutdown)
    signal.signal(signal.SIGINT, runner.shutdown)
    with (root / '.manager.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            runner.STATE['status'] = 'generating'
            runner.STATE['jobs'][job['id']]['state'] = 'running'
            runner.STATE['workers'][str(gpu)].update(state='running', job=job['id'])
            runner.persist()
            if runner.generation_record(job) is None:
                runner.generate(job, gpu)
            if runner.generation_record(job) is None:
                raise ValueError('generation lacks complete frozen-sample coverage')
            if protocol.get('lookup_ablation'):
                spec = importlib.util.spec_from_file_location('lookup_audit', root / 'lookup_audit_source.py')
                audit = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(audit)
                result = audit.ready_audit(root / 'cells' / job['id'], protocol['lookup_ablation'],
                                           protocol['evaluation_samples'])
                runner.save(root / 'cells' / job['id'] / 'lookup_audit.json', result)
            runner.STATE['status'] = 'scoring'
            runner.persist()
            runner.score_job(dict(parent=job['id'], dataset=job['dataset']), gpu)
            folder = root / 'cells' / job['id']
            score = runner.cached_score(job['dataset'], folder)
            if score is None:
                raise ValueError('native scoring did not finish all samples')
            value = runner.read(folder / 'complete.json')['quality_metric']
            reference_scores = {name: runner.read(path)['quality_metric']
                                for name, path in protocol['reference_scores'].items()}
            runner.save(root / 'comparison.json', dict(complete=True, job=protocol['job'], score=value,
                 reference_scores=reference_scores,
                 differences={name: value - score for name, score in reference_scores.items()},
                 completed_utc=runner.now(), quality_only=True, formal_performance_eligible=False))
            runner.STATE['status'] = 'complete'
            runner.STATE['complete'] = True
            runner.STATE['workers'][str(gpu)].update(state='complete')
            runner.persist()
        except BaseException:
            runner.save(root / 'error.json', dict(complete=False, error=traceback.format_exc(), observed_utc=runner.now()))
            runner.STATE['status'] = 'failed'
            runner.STATE['complete'] = False
            runner.persist()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int, choices=[0, 1])
    args = parser.parse_args()
    run(args.root.resolve(), args.gpu)
