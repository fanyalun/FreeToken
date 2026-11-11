"""Capture a real official CLI Read-image request using a CPU-only mock response."""

import asyncio
import base64
from http.server import ThreadingHTTPServer
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import vision_mock_probe as mock


ROOT = Path(__file__).resolve().parent
RUNTIME = Path('/data/fanya/tablemoe/vl_four_dataset_bs1_20261111/vision2web_runs_reasoning_off_20261111/smoke/fused/official_runtime')
MODEL = 'Qwen3-VL-30B-A3B-Instruct-image-component'


class ImageHandler(mock.Handler):
    def tool_content(self, body):
        tool = next((item for item in body.get('tools', []) if item['name'] == 'Read'), None)
        if tool is None:
            return None
        return dict(type='tool_use', id=mock.TOOL_ID, name='Read', input=dict(file_path='/workspace/probe.png'))


async def exercise(folder, server, image_path):
    spec = importlib.util.spec_from_file_location('image_fixture_adapter', ROOT / 'runtime_snapshot/benchmarks/bench_vision2web.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(RUNTIME))
    from vision2web.inference.adapters.claude_code import ClaudeCodeAdapter
    workspace = folder / 'workspace_source'
    workspace.mkdir()
    shutil.copy2(image_path, workspace / 'probe.png')
    (workspace / 'start.sh').write_text('#!/bin/sh\nexit 0\n')
    config = dict(output=str(folder), sandbox_image='vision2web-sandbox:freetoken-d80a4a3')
    sandbox = module.persistent_sandbox(config, 'image_probe', logging.getLogger('vision_image_probe'))
    container = None
    try:
        container = await sandbox.create_container(workspace)
        if not container or not await sandbox.start_container(workspace):
            raise RuntimeError('image probe container failed to start')
        inspection = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]
        mock.save(folder / 'container.json', dict(id=container,
            device_requests=inspection['HostConfig'].get('DeviceRequests'), mounts=inspection['Mounts'],
            temporary_environment=[s for s in inspection['Config']['Env'] if s.startswith(('TMPDIR=', 'TMP=', 'TEMP='))]))
        client = ClaudeCodeAdapter(api_key='EMPTY', model=MODEL,
            base_url=f'http://host.docker.internal:{server.server_port}', sandbox_manager=sandbox, timeout=180)
        result = await client.run_task(workspace,
            'Infrastructure fixture only. Read /workspace/probe.png with the Read tool and describe its visible heading. '
            'Do not create a website, modify files, install dependencies, or start a server.',
            dict(name='image_transport_fixture', task_type='mock'))
        mock.save(folder / 'adapter_result.json', result)
        return result
    finally:
        await sandbox.cleanup_all()
        if container:
            check = subprocess.run(['docker', 'inspect', container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            mock.save(folder / 'cleanup.json', dict(own_container=container, removed=check.returncode != 0))


def image_blocks(body):
    return [image for message in body.get('messages', [])
        for block in (message.get('content', []) if isinstance(message.get('content'), list) else [])
        if block.get('type') == 'tool_result' and block.get('tool_use_id') == mock.TOOL_ID
        for image in (block.get('content', []) if isinstance(block.get('content'), list) else [])
        if image.get('type') == 'image']


def main():
    fixture = json.loads((ROOT / 'validation/vision_browser_fixture.json').read_text())
    if not fixture['complete'] or not fixture['fixture_only'] or fixture['model_called']:
        raise ValueError('a completed CPU browser fixture is required')
    image_path = Path(next(item['path'] for item in fixture['screenshots'] if item['viewport'] == 'mobile'))
    folder = ROOT / 'validation' / f'vision_image_transport_{time.time_ns()}'
    (folder / 'tmp').mkdir(parents=True)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[name] = str(folder / 'tmp')
    mock.MODEL = MODEL
    mock.RECORDS.clear()
    server = ThreadingHTTPServer(('0.0.0.0', 0), ImageHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    result, error = None, None
    try:
        result = asyncio.run(exercise(folder, server, image_path))
    except BaseException as caught:
        error = repr(caught)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    mock.save(folder / 'requests.json', mock.RECORDS)
    messages = [record for record in mock.RECORDS if '/messages' in record['path'] and '/count_tokens' not in record['path']]
    captured = [record for record in messages if image_blocks(record['body'])]
    replay_path = folder / 'image_request.json'
    valid = False
    metadata = []
    if len(captured) == 1:
        mock.save(replay_path, captured[0]['body'])
        blocks = image_blocks(captured[0]['body'])
        for block in blocks:
            source = block.get('source', {})
            raw = base64.b64decode(source.get('data', ''), validate=True)
            metadata.append(dict(type=source.get('type'), media_type=source.get('media_type'), bytes=len(raw)))
        valid = (len(blocks) == 1 and metadata[0]['type'] == 'base64'
                 and metadata[0]['media_type'] in ('image/png', 'image/jpeg') and metadata[0]['bytes'] > 0
                 and captured[0]['body'].get('max_tokens') == 64000)
    checks = dict(complete=error is None and valid and bool(result and result.get('status') == 'success'),
        fixture_only=True, benchmark_score=False, real_official_cli_called=True,
        gpu_started=False, real_model_called=False, mock_emitted_read_tool=True,
        image_tool_result_received=valid, messages=len(messages), images=metadata,
        image_request=str(replay_path), source_image=str(image_path),
        evidence_root=str(folder), error=error)
    mock.save(folder / 'complete.json', checks)
    mock.save(ROOT / 'validation/vision_image_transport.json', checks)
    print(json.dumps(checks, indent=2))
    raise SystemExit(0 if checks['complete'] else 1)


if __name__ == '__main__':
    main()
