import argparse
import json
import os
from pathlib import Path
import signal
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(')')[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def request(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


def owned_processes(pid):
    owned = {pid}
    stamps = {}
    for _ in range(8):
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path/'stat').read_text().split(')')[1].split()
                child = int(path.name)
                if child in owned or int(fields[1]) in owned:
                    owned.add(child)
                    stamps[child] = fields[19]
            except (OSError, ValueError, IndexError):
                pass
    return [dict(pid=child, starttime=stamps[child]) for child in sorted(stamps)]


def main(dataset, gpu):
    state = json.loads((ROOT/'status.json').read_text())
    job = state['workers'][str(gpu)]
    if job['id'] != 'build_' + dataset:
        raise ValueError('GPU worker does not match the requested collection')
    pid = job['pid']
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if dataset.encode() not in argv or not any(x.endswith(b'/build_table.py') for x in argv):
        raise ValueError('collection PID identity changed')
    folder = ROOT/'build'/dataset
    command = json.loads((folder/'command.json').read_text())['command']
    port = int(command[command.index('--port') + 1])
    config = json.loads(Path(command[command.index('--tablemoe-collect-config') + 1]).read_text())
    receipt = dict(dataset=dataset, gpu=gpu, driver_pid=pid, port=port,
                   started_unix=time.time(), observations=[], admission_closed=False,
                   boundary_stopped=False, collection=config['output'],
                   processes=owned_processes(pid))
    frozen = False
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if not alive(pid):
                raise RuntimeError('driver exited before a pause boundary was established')
            stats = request(port, '/v1/stats')
            if stats['requests']['active'] != 1:
                time.sleep(.2)
                continue
            os.kill(pid, signal.SIGSTOP)
            frozen = True
            stats = request(port, '/v1/stats')
            receipt['observations'].append(dict(time=time.time(), stats=stats))
            if stats['requests']['active'] == 1:
                receipt['observed_active_before_drain'] = 1
                completed_before = stats['requests']['completed']
                break
            os.kill(pid, signal.SIGCONT)
            frozen = False
            time.sleep(.2)
        else:
            raise TimeoutError('could not freeze a driver with one admitted request')
        # A stopped synchronous BS1 client cannot submit another request after this one ends.
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            stats = request(port, '/v1/stats')
            receipt['observations'].append(dict(time=time.time(), stats=stats))
            if stats['requests']['active'] == 0:
                if stats['requests']['completed'] != completed_before + 1:
                    raise RuntimeError('drain did not complete exactly one admitted request')
                receipt['observed_active_after_drain'] = 0
                break
            if stats['requests']['active'] != 1:
                raise RuntimeError('collection is no longer serial BS1')
            time.sleep(.5)
        else:
            raise TimeoutError('admitted request did not finish while client was paused')
        sealed = request(port, '/v1/admin/prepare-stop',
                         dict(drain_timeout_s=0, abort_timeout_s=0))
        if not sealed.get('drain_complete'):
            raise RuntimeError('server did not seal admission')
        receipt.update(admission_closed=True, sealed=sealed, prepare_stop_response=sealed,
                       submissions_stopped=True, inflight_requests=0, no_abort=True,
                       proof='frontend_admission_barrier')
        os.kill(pid, signal.SIGCONT)
        frozen = False
        # The completed SSE response is persisted; the next request receives HTTP 503.
        deadline = time.monotonic() + 120
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(.5)
        if alive(pid):
            raise TimeoutError('driver did not exit after the closed-admission response')
        progress = json.loads((folder/'progress.json').read_text())
        if any(alive(item['pid']) for item in receipt['processes']):
            raise RuntimeError('an owned server process remains alive after driver exit')
        receipt.update(boundary_stopped=True, stopped_unix=time.time(), server_stopped=True,
                       completed=progress['completed'], build_progress=progress)
    except BaseException as exc:
        receipt['error'] = repr(exc)
        raise
    finally:
        if frozen and alive(pid):
            os.kill(pid, signal.SIGCONT)
        path = folder/'collection_pause_receipt.json'
        path.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    main(args.dataset, args.gpu)
