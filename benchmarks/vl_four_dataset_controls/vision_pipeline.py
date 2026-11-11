"""Run the frozen Vision2Web heldout tasks with official agent and visual scoring."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from run_cell import ROOT, SNAPSHOT, MODEL, command, environment
from tablemoe_common import running_server
from tablemoe_data import read_rows
from freetoken.moe.tablemoe_artifacts import write_json


DATASET = 'Vision2Web_Level1'
METHODS = ('fused', 'tablemoe', 'modes', 'schedule', 'table')
KIT_PYTHON = '/home/fanya/miniforge/envs/tablemoe/bin/python'
RUNS = Path('/data/fanya/tablemoe/vl_four_dataset_bs1_20261111/vision2web_runs')
OFFICIAL = Path('/home/fanya/vision2web')
ADAPTER = SNAPSHOT / 'benchmarks/bench_vision2web.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def port_pair():
    while True:
        with socket.socket() as api, socket.socket() as backend:
            api.bind(('0.0.0.0', 0))
            port = api.getsockname()[1]
            try:
                backend.bind(('0.0.0.0', port + 1))
            except OSError:
                continue
            return port


def set_argument(args, name, value):
    if name in args:
        args[args.index(name) + 1] = str(value)
    else:
        args.extend([name, str(value)])


def stage_command(method, gpu, stage, config):
    judge = stage == 'evaluate'
    args, _ = command(DATASET, 'fused' if judge else method, gpu)
    endpoint = config['evaluation' if judge else 'generation']
    port = int(endpoint['base_url'].split(':')[-1].split('/')[0])
    set_argument(args, '--port', port)
    set_argument(args, '--host', '127.0.0.1' if judge else '0.0.0.0')
    set_argument(args, '--served-model-name', endpoint['model'])
    set_argument(args, '--max-seq-len-override', 200000)
    set_argument(args, '--num-pages', 200000)
    set_argument(args, '--max-output-tokens', 64000)
    set_argument(args, '--max-running-requests', 1)
    set_argument(args, '--tool-call-parser', 'auto')
    # Instruct answers must remain visible to the official content-only visual scorer.
    set_argument(args, '--reasoning-parser', 'off' if judge else 'auto')
    return args, port


def config_for(method, smoke=False):
    rows = read_rows(ROOT / 'prepare' / DATASET / ('calibration.jsonl' if smoke else 'evaluation.jsonl'))
    projects = [row['metadata']['task'] for row in (rows[:1] if smoke else rows)]
    expected = 1 if smoke else 40
    if len(projects) != expected or len(set(projects)) != expected:
        raise ValueError('frozen Vision2Web project coverage changed')
    folder = RUNS / ('smoke' if smoke else 'heldout') / method
    path = folder / 'run_config.json'
    if path.is_file():
        config = read(path)
        if config['projects'] != projects:
            raise ValueError('frozen Vision2Web project list changed')
        return folder, path, config
    generation_port, judge_port = port_pair(), port_pair()
    while abs(generation_port - judge_port) < 2:
        judge_port = port_pair()
    config = dict(official_root=str(OFFICIAL), official_revision='d80a4a3d0ea7326541ad87314294269a633789e4',
        datasets_dir='/data/fanya/datasets/vision2web_level1', output=str(folder),
        sandbox_image='vision2web-sandbox:freetoken-d80a4a3', projects=projects,
        generation=dict(model=f'Qwen3-VL-30B-A3B-Instruct-{method}',
            base_url=f'http://host.docker.internal:{generation_port}',
            api_key_env='VISION2WEB_GENERATION_API_KEY', max_workers=1, timeout=36000),
        evaluation=dict(model='Qwen3-VL-30B-A3B-Instruct-fused-judge',
            base_url=f'http://127.0.0.1:{judge_port}/v1',
            api_key_env='VISION2WEB_VISUAL_API_KEY', max_workers=1))
    write_json(path, config)
    return folder, path, config


def kit_environment(gpu):
    env = environment(gpu)
    env['LD_LIBRARY_PATH'] = '/home/fanya/miniforge/envs/tablemoe/lib'
    env['PYTHONPATH'] = f'{SNAPSHOT}/python:{SNAPSHOT}/benchmarks'
    for key in ('NO_PROXY', 'no_proxy'):
        env[key] = ','.join(filter(None, [env.get(key), '127.0.0.1', 'localhost', 'host.docker.internal']))
    return env


def cleanup_own_containers(config, stage):
    base = Path(config['output']) / 'results/webpage/claude_code' / config['generation']['model']
    records = []
    for project in config['projects']:
        workspace = base / project
        identity = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:12]
        name = f'vision2web-{stage}-{identity}'
        inspect = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', name],
                                 capture_output=True, text=True)
        if inspect.returncode == 0 and inspect.stdout.strip() == 'true':
            result = subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, text=True)
            records.append(dict(name=name, returncode=result.returncode))
    return records


def adapter_call(stage, path, folder, gpu, deadline=None):
    cmd = [KIT_PYTHON, '-B', str(ADAPTER), stage, '--config', str(path)]
    log = folder / f'pipeline_{stage}.log'
    mode = 'a' if log.exists() else 'w'
    with log.open(mode) as handle:
        proc = subprocess.Popen(cmd, env=kit_environment(gpu), cwd=SNAPSHOT,
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(folder / f'pipeline_{stage}_job.json', dict(pid=proc.pid, command=cmd, started_utc=now()))
        try:
            while proc.poll() is None:
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError('the authorized experiment deadline has been reached')
                time.sleep(1)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=15)
                stopped = cleanup_own_containers(read(path), stage)
                write_json(folder / f'pipeline_{stage}_interruption.json',
                           dict(complete=False, stopped_containers=stopped, stopped_utc=now()))
        if proc.returncode:
            raise RuntimeError(f'official {stage} failed with exit {proc.returncode}: {log}')


def prepare(method, gpu, smoke=False):
    folder, path, config = config_for(method, smoke)
    adapter_call('prepare', path, folder, gpu)
    adapter_call('doctor', path, folder, gpu)
    commands = {stage: stage_command(method, gpu, stage, config)[0] for stage in ('inference', 'evaluate')}
    write_json(folder / 'pipeline_protocol.json', dict(dataset=DATASET, method=method, smoke=smoke,
        batch_size=1, max_running_requests=1, generation_max_workers=1, evaluation_max_workers=1,
        evaluation_samples=len(config['projects']), context_tokens=200000, maximum_output_tokens=64000,
        source=str(SNAPSHOT), adapter=str(ADAPTER), adapter_sha256=hashlib.sha256(ADAPTER.read_bytes()).hexdigest(),
        worker_sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ('vision_pipeline.py', 'run_cell.py')},
        offline_calibration_protocol=str(ROOT / 'prepare' / DATASET / 'protocol.json'),
        judge=dict(model=MODEL, dtype='float16', moe_strategy='fused', lossless=True),
        generation_command=commands['inference'], judge_command=commands['evaluate'],
        formal_performance_eligible=False,
        resource_note='200000 fp16 KV positions consume about 18.31 GiB; resident fused has little A100 headroom; GPU startup and full official workflow remain required checks',
        official_prompt_unchanged=True, official_tools_unchanged=True,
        outputs_reduced_to_single_html_completion=False, prepared_utc=now()))
    return folder, path, config


def endpoint_probe(port, config, stage, folder):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{port}/v1/models', timeout=30) as response:
        models = json.load(response)
    expected = config['evaluation' if stage == 'evaluate' else 'generation']['model']
    if expected not in [model['id'] for model in models.get('data', [])]:
        raise ValueError('served model identity differs from the frozen official endpoint')
    record = dict(models=models, expected_model=expected, probe_utc=now())
    if stage == 'inference':
        body = dict(model=expected, messages=[dict(role='user', content='hello')],
                    tools=[dict(name='read_file', description='Read a file',
                                input_schema=dict(type='object', properties=dict(path=dict(type='string')), required=['path']))])
        request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/messages/count_tokens',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
        with opener.open(request, timeout=30) as response:
            record['anthropic_count_tokens'] = json.load(response)
        if record['anthropic_count_tokens'].get('input_tokens', 0) < 1:
            raise ValueError('Anthropic tool-aware token counting returned no input tokens')
    write_json(folder / f'endpoint_probe_{stage}.json', record)


def score_summary(config, folder):
    if not read(folder / 'complete.json').get('complete'):
        raise ValueError('cannot summarize incomplete official coverage')
    analysis = read(folder / 'official_analysis.json')
    key = f'webpage:claude_code:{config["generation"]["model"]}'
    statistics = analysis['statistics'][key]
    count = len(config['projects'])
    if statistics['total_count'] != count or len(statistics['prototype_scores']) != 3 * count:
        raise ValueError('official visual score denominator differs from frozen projects')
    result = dict(metric='Visual Score', visual_score_pct=sum(statistics['prototype_scores']) / (3 * count) * 100,
        scoring_denominator_projects=count, scoring_denominator_viewports=3 * count)
    for view in ('desktop', 'tablet', 'mobile'):
        values = statistics[f'{view}_scores']
        if len(values) != count:
            raise ValueError(f'{view} scoring denominator differs from frozen projects')
        result[f'{view}_score_pct'] = sum(values) / count * 100
    return result


def publish_completion(method, stage, config, folder, smoke):
    marker = folder / f'{stage}_complete.json'
    if not read(marker).get('complete'):
        raise ValueError('official stage has not completed')
    if smoke:
        if stage == 'evaluate':
            summary = score_summary(config, folder)
            write_json(ROOT / 'validation/vision_pipeline_ready.json', dict(complete=True,
                tested_method=method, gpu_smoke=True, samples=1, heldout_samples_used=0,
                inference_complete=str(folder / 'inference_complete.json'),
                evaluation_complete=str(marker), report_complete=str(folder / 'complete.json'),
                smoke_root=str(folder), tested_utc=now(), **summary))
        return
    summary = score_summary(config, folder) if stage == 'evaluate' else {}
    record = dict(complete=True, samples=len(config['projects']), agent_workflow=True, batch_size=1,
        official_output=str(folder), source_marker=str(marker), formal_performance_eligible=False, **summary)
    name = 'generation_complete.json' if stage == 'inference' else 'score_complete.json'
    for root in (ROOT / 'cells' / DATASET / method, ROOT / 'vision' / method):
        write_json(root / name, record)
        if stage == 'evaluate':
            write_json(root / 'complete.json', record)


def main(dataset, method, gpu, stage, smoke=False, deadline=None):
    if dataset != DATASET:
        raise ValueError('this pipeline only runs Vision2Web Level 1')
    folder, path, config = prepare(method, gpu, smoke)
    if stage == 'prepare':
        return
    if stage == 'report':
        adapter_call('report', path, folder, gpu, deadline)
        publish_completion(method, 'evaluate', config, folder, smoke)
        return
    marker = folder / f'{stage}_complete.json'
    if marker.is_file() and read(marker).get('complete'):
        if stage == 'evaluate':
            adapter_call('report', path, folder, gpu, deadline)
        publish_completion(method, stage, config, folder, smoke)
        return
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError('the authorized experiment deadline has been reached')
    if stage == 'evaluate' and not read(folder / 'inference_complete.json').get('complete'):
        raise ValueError('official inference coverage must complete before visual scoring')
    args, port = stage_command(method, gpu, stage, config)
    server_dir = folder / f'{stage}_server'
    attempt = 1
    while server_dir.exists():
        attempt += 1
        server_dir = folder / f'{stage}_server_attempt_{attempt:03d}'
    write_json(folder / f'{stage}_command.json', dict(command=args, gpu=gpu, server_log=str(server_dir / 'server.log')))
    started = time.time()
    try:
        with running_server(args, server_dir, SNAPSHOT, environment(gpu), 1800) as server:
            write_json(folder / f'{stage}_server_process.json', dict(pid=server.pid,
                physical_gpu=gpu, started_utc=now(), owned_by_this_pipeline=True))
            endpoint_probe(port, config, stage, folder)
            adapter_call(stage, path, folder, gpu, deadline)
        if stage == 'evaluate':
            adapter_call('report', path, folder, gpu, deadline)
    except BaseException as error:
        write_json(folder / f'pipeline_{stage}_failure.json', dict(error=repr(error), failed_utc=now(),
            elapsed_s=time.time()-started, complete=False))
        raise
    write_json(folder / f'pipeline_{stage}_complete.json', dict(complete=True, elapsed_s=time.time()-started,
        finished_utc=now(), official_marker=str(marker), batch_size=1))
    publish_completion(method, stage, config, folder, smoke)


def interrupted(signum, frame):
    raise InterruptedError(f'Vision2Web worker received signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('method', choices=METHODS)
    parser.add_argument('gpu', type=int)
    parser.add_argument('stage', choices=('prepare', 'inference', 'evaluate', 'report', 'smoke'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.stage == 'smoke':
        for stage in ('inference', 'evaluate'):
            main(args.dataset, args.method, args.gpu, stage, True, args.deadline)
    else:
        main(args.dataset, args.method, args.gpu, args.stage, args.smoke, args.deadline)
