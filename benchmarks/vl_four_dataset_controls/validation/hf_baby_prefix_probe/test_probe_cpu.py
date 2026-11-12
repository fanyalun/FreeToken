"""CPU-only supervisor tests with short, owned Python workers."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('hf_probe_cpu_test', HERE / 'probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
WORK = HERE / ('cpu_test_' + str(time.time_ns()))
WORK.mkdir()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
real_popen = subprocess.Popen
old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
checks = []


def configure(name):
    target = WORK / name
    target.mkdir()
    (target / 'tmp').mkdir()
    probe.HERE = target
    probe.ROOT = target
    probe.save(target / 'execution_policy.json', {'deadline_unix': time.time() + 60})
    probe.save(target / 'cpu_preflight.json', {'complete': True, 'source_sha256': probe.source_sha()})
    probe.save(target / 'release.json', {'complete': True, 'source_sha256': probe.source_sha()})
    return target


def spawn_script(code):
    def spawn(command, **kwargs):
        kwargs['env']['CUDA_VISIBLE_DEVICES'] = ''
        return real_popen([sys.executable, '-c', code], **kwargs)
    probe.subprocess.Popen = spawn


try:
    target = configure('success')
    code = "import json;from pathlib import Path;Path('worker_complete.json').write_text(json.dumps(" + repr(
        dict(complete=True, source_sha256=probe.source_sha())) + '))'
    spawn_script(code)
    probe.supervise(0)
    assert probe.read(target / 'complete.json')['complete']
    assert not probe.read(target / 'cleanup.json')['still_alive']
    assert probe.read(target / 'ownership.json')['worker']['start_ticks']
    checks.append('owned_pid_startticks_and_success_completion')
    try:
        probe.supervise(0)
    except ValueError:
        checks.append('completed_diagnostic_cannot_be_overwritten')
    else:
        raise AssertionError('implicit rerun allowed')

    target = configure('failed')
    spawn_script('raise SystemExit(7)')
    try:
        probe.supervise(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError('worker failure accepted')
    assert not (target / 'complete.json').exists() and (target / 'failure.json').exists()
    assert not probe.read(target / 'cleanup.json')['still_alive']
    checks.append('worker_failure_preserved_without_completion')

    target = configure('timeout')
    probe.MAX_SECONDS = 12
    spawn_script('import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)')
    started = time.monotonic()
    try:
        probe.supervise(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError('timeout accepted')
    assert time.monotonic() - started < 12
    assert not probe.read(target / 'cleanup.json')['still_alive']
    assert not (target / 'complete.json').exists()
    checks.append('deadline_kills_only_stamped_owned_worker_within_job_limit')

    target = configure('unreleased')
    probe.save(target / 'release.json', {'complete': False, 'source_sha256': probe.source_sha()})
    try:
        probe.supervise(0)
    except ValueError:
        pass
    else:
        raise AssertionError('unreleased worker started')
    assert not (target / 'ownership.json').exists()
    checks.append('root_release_required_before_worker_start')

    current = probe.identity(os.getpid())
    assert probe.same_process(current)
    assert not probe.same_process({**current, 'start_ticks': str(int(current['start_ticks']) + 1)})
    checks.append('pid_reuse_rejected_by_start_ticks')
finally:
    probe.subprocess.Popen = real_popen
    for sig, handler in old_handlers.items():
        signal.signal(sig, handler)

probe.save(HERE / 'cpu_supervisor_tests.json', dict(complete=True, gpu_started=False, tests=len(checks),
    checks=checks, source_sha256=probe.source_sha(), evidence_root=str(WORK)))
print(json.dumps({'passed': len(checks), 'source_sha256': probe.source_sha()}))
