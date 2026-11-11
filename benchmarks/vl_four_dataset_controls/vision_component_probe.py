"""Validate Vision2Web image transport and native judge without webpage-quality gating."""

import argparse
import asyncio
import base64
from contextlib import contextmanager
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parent
RUNTIME = Path('/data/fanya/tablemoe/vl_four_dataset_bs1_20261111/vision2web_runs_reasoning_off_20261111/smoke/fused/official_runtime')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read(path):
    return json.loads(Path(path).read_text())


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return dict(pid=int(pid), state=fields[0], pgid=int(fields[2]),
                session_id=int(fields[3]), start_ticks=int(fields[19]))


def owned_group_members(owner):
    leader = process_identity(owner['pid'])
    if leader is not None and leader['start_ticks'] != owner['start_ticks']:
        raise RuntimeError('server PID was reused; refusing to signal an unrelated process group')
    members = []
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            item = process_identity(int(path.name))
            if (item and item['pgid'] == owner['pgid'] and item['session_id'] == owner['session_id']
                    and item['start_ticks'] >= owner['start_ticks'] and item['state'] != 'Z'):
                members.append(item)
    return members


@contextmanager
def owned_server(command, folder, cwd, env, startup_timeout, shutdown_graces=(10, 5, 5)):
    folder.mkdir(parents=True)
    with (folder / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        owner = process_identity(proc.pid)
        if owner is None:
            proc.wait()
            raise RuntimeError(f'server exited before ownership could be recorded: {proc.returncode}')
        owner.update(command=command, started_unix=time.time(), owned_by_this_probe=True)
        try:
            save(folder / 'server_process.json', owner)
            deadline = time.monotonic() + startup_timeout
            while 'API server is ready to serve' not in (folder / 'server.log').read_text(errors='replace'):
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited {proc.returncode} before readiness')
                if time.monotonic() >= deadline:
                    raise TimeoutError('component server startup deadline reached')
                time.sleep(min(.2, max(.001, deadline - time.monotonic())))
            yield proc
        finally:
            cleanup = dict(owner=owner, signals=[], complete=False)
            try:
                for sig, grace in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), shutdown_graces):
                    proc.poll()
                    if not owned_group_members(owner):
                        break
                    try:
                        os.killpg(owner['pgid'], sig)
                        cleanup['signals'].append(sig.name)
                    except ProcessLookupError:
                        break
                    end = time.monotonic() + grace
                    while time.monotonic() < end:
                        proc.poll()
                        if not owned_group_members(owner):
                            break
                        time.sleep(.1)
                proc.wait(timeout=1)
                cleanup.update(remaining=owned_group_members(owner), returncode=proc.returncode)
                cleanup['complete'] = not cleanup['remaining']
                if not cleanup['complete']:
                    raise RuntimeError('owned component server process group survived SIGKILL')
            finally:
                save(folder / 'server_cleanup.json', cleanup)


def required_cpu_receipts():
    browser = read(ROOT / 'validation/vision_browser_fixture.json')
    image = read(ROOT / 'validation/vision_image_transport.json')
    if not (browser['complete'] and image['complete'] and browser['fixture_only']
            and image['fixture_only'] and not browser['model_called'] and not image['real_model_called']):
        raise ValueError('completed unscored CPU browser and real-CLI image transport receipts are required')
    if not Path(image['image_request']).is_file() or not Path(image['source_image']).is_file():
        raise ValueError('CPU image replay artifacts are missing')
    return browser, image


def anthropic_probe(config, folder, image_receipt, three_image_load=False):
    body = read(image_receipt['image_request'])
    if body.get('max_tokens') != 64000 or body.get('stream') is not True:
        raise ValueError('captured official CLI output budget or streaming protocol changed')
    body['model'] = config['generation']['model']
    prefix = 'anthropic_three_image' if three_image_load else 'anthropic'
    input_paths = []
    if three_image_load:
        content = [dict(type='text', text='Infrastructure fixture only. Three reference screenshots follow. '
            'Briefly describe the main colors visible in each screenshot. This is an unscored image component check. '
            'Do not change files or create a website.')]
        for name in ('desktop', 'tablet', 'mobile'):
            path = Path('/data/fanya/datasets/vision2web_level1/webpage/aihw/prototypes') / f'{name}.jpg'
            raw = path.read_bytes()
            input_paths.append(dict(viewport=name, path=str(path), bytes=len(raw), resized=False))
            content.append(dict(type='image', source=dict(type='base64', media_type='image/jpeg',
                                                        data=base64.b64encode(raw).decode('ascii'))))
        body['messages'] = [dict(role='user', content=content)]
        save(folder / 'three_image_inputs.json', input_paths)
    save(folder / f'{prefix}_request.json', body)
    base = config['evaluation']['base_url'].removesuffix('/v1')
    request = urllib.request.Request(base + '/v1/messages', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01', 'x-api-key': 'EMPTY'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    events = []
    with (folder / f'{prefix}_response.sse').open('wb') as log:
        with opener.open(request, timeout=600) as response:
            for line in response:
                log.write(line)
                log.flush()
                if line.startswith(b'data:'):
                    payload = line[5:].strip()
                    if payload and payload != b'[DONE]':
                        event = json.loads(payload)
                        events.append(event)
                        if event.get('type') == 'error':
                            raise RuntimeError(f'Anthropic image endpoint failed: {event}')
    save(folder / f'{prefix}_events.json', events)
    starts = [item for item in events if item.get('type') == 'message_start']
    stops = [item for item in events if item.get('type') == 'message_stop']
    reasons = [item.get('delta', {}).get('stop_reason') for item in events if item.get('type') == 'message_delta']
    deltas = [item.get('delta', {}) for item in events if item.get('type') == 'content_block_delta']
    visible = any(item.get('text') or item.get('partial_json') for item in deltas)
    usage = starts[0].get('message', {}).get('usage', {}) if len(starts) == 1 else {}
    terminal_usage = None
    for event in events:
        if event.get('type') == 'message_delta':
            usage.update(event.get('usage', {}))
            if event.get('delta', {}).get('stop_reason') is not None:
                terminal_usage = event.get('usage')
    if (len(starts) != 1 or len(stops) != 1 or not reasons or reasons[-1] not in ('end_turn', 'tool_use')
            or not isinstance(terminal_usage, dict) or terminal_usage.get('output_tokens', 0) <= 0
            or usage.get('input_tokens', 0) <= 0 or not visible):
        raise RuntimeError('image request did not produce a complete visible Anthropic response')
    result = dict(complete=True, input_tokens=usage['input_tokens'], output_tokens=terminal_usage['output_tokens'], stop_reason=reasons[-1],
        real_cli_request_replayed=not three_image_load, actual_tool_result_image_input=not three_image_load,
        direct_user_image_input=three_image_load, input_images=3 if three_image_load else 1,
        original_three_prototype_files=input_paths, benchmark_score=False)
    save(folder / f'{prefix}_complete.json', result)
    return result


async def judge_probe(config, folder, image_receipt):
    sys.path.insert(0, str(RUNTIME))
    from vision2web.evaluation.visual_scorer import VisualScorer
    scorer = VisualScorer(api_key='EMPTY', base_url=config['evaluation']['base_url'],
                          visual_model=config['evaluation']['model'], timeout=600)
    create = scorer.client.chat.completions.create
    attempts = []

    async def record_create(**kwargs):
        index = len(attempts) + 1
        attempts.append(index)
        save(folder / f'judge_request_{index:02d}.json', kwargs)
        response = await create(**kwargs)
        save(folder / f'judge_response_{index:02d}.json', response.model_dump())
        return response

    scorer.client.chat.completions.create = record_create
    image = base64.b64encode(Path(image_receipt['source_image']).read_bytes()).decode('ascii')
    try:
        scores = await scorer.compare_prototype(image, image)
    finally:
        await scorer.client.close()
    valid = isinstance(scores, list) and bool(scores) and all(isinstance(item, dict)
        and type(item.get('score')) in (int, float) and math.isfinite(item['score'])
        and 0 <= item['score'] <= 1 for item in scores)
    if not valid:
        raise RuntimeError('native judge returned missing or invalid component scores')
    result = dict(complete=True, scores=scores, attempts=len(attempts),
        official_visual_prompt=True, official_max_tokens=64000, benchmark_score=False,
        acceptance_requires_valid_score=True, acceptance_requires_high_score=False)
    save(folder / 'judge_complete.json', result)
    return result


def request_worker(config_path, folder):
    browser, image = required_cpu_receipts()
    config = read(config_path)
    receipt = dict(complete=False, benchmark_score=False, fixture_only=True)
    try:
        receipt['anthropic'] = anthropic_probe(config, folder, image)
        receipt['three_image_load'] = anthropic_probe(config, folder, image, three_image_load=True)
        receipt['judge'] = asyncio.run(judge_probe(config, folder, image))
        receipt['complete'] = True
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        save(folder / 'request_worker_complete.json', receipt)


def run(gpu, deadline, timeout):
    from vision_pipeline import (stage_command, port_pair, environment,
                                 SNAPSHOT, deadline_timeout, endpoint_probe, kit_environment, KIT_PYTHON)
    browser, image = required_cpu_receipts()
    started = time.time()
    effective_deadline = min(deadline, started + timeout) if deadline is not None else started + timeout
    folder = ROOT / 'validation' / f'vision_component_gpu_{time.time_ns()}'
    folder.mkdir(parents=True)
    port = port_pair()
    model = 'Qwen3-VL-30B-A3B-Instruct-image-component'
    config = dict(generation=dict(model=model, base_url=f'http://host.docker.internal:{port}'),
                  evaluation=dict(model=model, base_url=f'http://127.0.0.1:{port}/v1'))
    config_path = folder / 'run_config.json'
    save(config_path, config)
    command, port = stage_command('fused', gpu, 'inference', config)
    save(folder / 'protocol.json', dict(command=command, physical_gpu=gpu, batch_size=1,
        max_running_requests=1, context_tokens=200000, max_output_tokens=64000,
        fixture_only=True, benchmark_score=False, deadline_unix=effective_deadline,
        global_deadline_unix=deadline, browser_receipt=browser, cli_image_receipt=image,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_prompt_and_tools_preserved_for_benchmark=True,
        component_text_is_dedicated_unscored_fixture=True))
    receipt = dict(complete=False, readiness_kind='component_infrastructure', fixture_only=True,
        benchmark_score=False, completed_official_webpage=False, gpu_smoke=True,
        heldout_samples_used=0, tested_method='fused', evidence_root=str(folder))
    worker = None
    try:
        with owned_server(command, folder / 'server', SNAPSHOT, environment(gpu),
                            deadline_timeout(effective_deadline, 1800)) as server:
            endpoint_probe(port, config, 'inference', folder, effective_deadline)
            worker_command = [KIT_PYTHON, '-B', str(Path(__file__).resolve()),
                '--request-worker', '--config', str(config_path), '--output', str(folder)]
            with (folder / 'worker.log').open('w') as log:
                worker = subprocess.Popen(worker_command, env=kit_environment(gpu), cwd=SNAPSHOT,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    save(folder / 'worker_process.json', dict(pid=worker.pid, command=worker_command))
                    while worker.poll() is None:
                        time.sleep(deadline_timeout(effective_deadline, 1))
                    if worker.returncode:
                        raise RuntimeError(f'component request worker exited {worker.returncode}: {folder / "worker.log"}')
                    deadline_timeout(effective_deadline, 1)
                finally:
                    if worker.poll() is None:
                        os.killpg(worker.pid, signal.SIGTERM)
                        try:
                            worker.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(worker.pid, signal.SIGKILL)
                            worker.wait(timeout=5)
            if not read(folder / 'request_worker_complete.json').get('complete'):
                raise RuntimeError('component request worker did not complete')
        receipt.update(complete=True, elapsed_s=time.time() - started,
            browser_complete=str(ROOT / 'validation/vision_browser_fixture.json'),
            cli_image_complete=str(ROOT / 'validation/vision_image_transport.json'),
            anthropic_complete=str(folder / 'anthropic_complete.json'),
            three_image_complete=str(folder / 'anthropic_three_image_complete.json'),
            judge_complete=str(folder / 'judge_complete.json'))
        save(ROOT / 'validation/vision_components_ready.json', receipt)
        save(ROOT / 'validation/vision_pipeline_ready.json', receipt)
    except BaseException as error:
        receipt.update(error=repr(error), elapsed_s=time.time() - started,
                       deadline_unix=effective_deadline)
        raise
    finally:
        save(folder / 'complete.json', receipt)


def interrupted(signum, frame):
    raise InterruptedError(f'component probe received signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpu', type=int, nargs='?')
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--request-worker', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    logging.basicConfig(level=logging.INFO)
    if args.request_worker:
        request_worker(args.config, args.output)
    elif args.gpu is None:
        parser.error('a GPU index is required')
    else:
        run(args.gpu, args.deadline, args.timeout)


if __name__ == '__main__':
    main()
