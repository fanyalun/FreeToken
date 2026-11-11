"""Exercise the official container CLI against a local mock Anthropic endpoint."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen3-VL-30B-A3B-Instruct-transport-mock'
NONCE = 'VISION2WEB_TRANSPORT_PROBE_20261111'
TOOL_ID = 'toolu_v2w_transport_probe'
RECORDS = []
LOCK = threading.Lock()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def respond(self, value):
        raw = json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.respond(dict(data=[dict(id=MODEL, type='model')]))

    def tool_content(self, body):
        bash = next((tool for tool in body.get('tools', [])
                     if tool['name'].lower() == 'bash' and 'command' in tool.get('input_schema', {}).get('properties', {})), None)
        if bash is None:
            return None
        return dict(type='tool_use', id=TOOL_ID, name=bash['name'], input=dict(
            command='printf "PROBE_TMPDIR=%s\\n" "$TMPDIR"; cat /workspace/probe.txt',
            description='Read the transport probe and persistent temporary path'))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        record = dict(path=self.path, model=body.get('model'), body=body,
            anthropic_version=self.headers.get('anthropic-version'),
            anthropic_beta=self.headers.get('anthropic-beta'), client=self.client_address[0])
        with LOCK:
            RECORDS.append(record)
        if '/count_tokens' in self.path:
            self.respond(dict(input_tokens=100))
            return
        if '/messages' not in self.path:
            self.respond({})
            return
        results = [block for message in body.get('messages', [])
            for block in (message.get('content', []) if isinstance(message.get('content'), list) else [])
            if block.get('type') == 'tool_result' and block.get('tool_use_id') == TOOL_ID]
        tool = self.tool_content(body)
        if tool is not None and not results:
            content = tool
            reason = 'tool_use'
        else:
            content = dict(type='text', text='PROBE_OK')
            reason = 'end_turn'
        response = dict(id='msg_v2w_transport_probe', type='message', role='assistant', model=MODEL,
            content=[content], stop_reason=reason, stop_sequence=None,
            usage=dict(input_tokens=100, output_tokens=20))
        if not body.get('stream'):
            self.respond(response)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        message = dict(response, content=[], stop_reason=None, usage=dict(input_tokens=100, output_tokens=0))
        events = [('message_start', dict(type='message_start', message=message))]
        if content['type'] == 'tool_use':
            start = dict(content, input={})
            delta = dict(type='input_json_delta', partial_json=json.dumps(content['input']))
        else:
            start = dict(type='text', text='')
            delta = dict(type='text_delta', text=content['text'])
        events.extend([
            ('content_block_start', dict(type='content_block_start', index=0, content_block=start)),
            ('content_block_delta', dict(type='content_block_delta', index=0, delta=delta)),
            ('content_block_stop', dict(type='content_block_stop', index=0)),
            ('message_delta', dict(type='message_delta', delta=dict(stop_reason=reason, stop_sequence=None), usage=dict(output_tokens=20))),
            ('message_stop', dict(type='message_stop'))])
        for name, payload in events:
            self.wfile.write(f'event: {name}\ndata: {json.dumps(payload)}\n\n'.encode())
            self.wfile.flush()


async def exercise(folder, server):
    source = ROOT / 'runtime_snapshot/benchmarks/bench_vision2web.py'
    spec = importlib.util.spec_from_file_location('vision_adapter_probe', source)
    adapter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter_module)
    runtime = Path('/data/fanya/tablemoe/vl_four_dataset_bs1_20261111/vision2web_runs/heldout/fused/official_runtime')
    sys.path.insert(0, str(runtime))
    from vision2web.inference.adapters.claude_code import ClaudeCodeAdapter
    workspace = folder / 'workspace_source'
    workspace.mkdir()
    (workspace / 'probe.txt').write_text(NONCE + '\n')
    (workspace / 'start.sh').write_text('#!/bin/sh\nexit 0\n')
    config = dict(output=str(folder), sandbox_image='vision2web-sandbox:freetoken-d80a4a3')
    sandbox = adapter_module.persistent_sandbox(config, 'mock_probe', logging.getLogger('vision_mock_probe'))
    result, container, inspection = None, None, None
    try:
        container = await sandbox.create_container(workspace)
        if not container or not await sandbox.start_container(workspace):
            raise RuntimeError('mock probe container did not start')
        inspection = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]
        save(folder / 'container.json', dict(id=container,
            env=[value for value in inspection['Config']['Env'] if value.startswith(('TMPDIR=', 'TMP=', 'TEMP='))],
            mounts=inspection['Mounts'], device_requests=inspection['HostConfig'].get('DeviceRequests'),
            extra_hosts=inspection['HostConfig'].get('ExtraHosts')))
        adapter = ClaudeCodeAdapter(api_key='EMPTY', model=MODEL,
            base_url=f'http://host.docker.internal:{server.server_port}', sandbox_manager=sandbox, timeout=180)
        result = await adapter.run_task(workspace,
            f'Transport check {NONCE}. Use Bash to print TMPDIR and read /workspace/probe.txt, then reply PROBE_OK. '
            'This is a protocol test; do not create a website, change files, install packages or start a server.',
            dict(name='transport_probe', task_type='mock'))
        save(folder / 'adapter_result.json', result)
    finally:
        await sandbox.cleanup_all()
        if container:
            probe = subprocess.run(['docker', 'inspect', container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            save(folder / 'cleanup.json', dict(own_container=container, removed=probe.returncode != 0))
    return result


def main():
    folder = ROOT / 'validation' / f'vision_mock_probe_{time.time_ns()}'
    (folder / 'tmp').mkdir(parents=True)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[name] = str(folder / 'tmp')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    error, result = None, None
    try:
        result = asyncio.run(exercise(folder, server))
    except BaseException as caught:
        error = repr(caught)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    save(folder / 'requests.json', RECORDS)
    messages = [r for r in RECORDS if '/messages' in r['path'] and '/count_tokens' not in r['path']]
    tool_results = [block for r in messages for message in r['body'].get('messages', [])
        for block in (message.get('content', []) if isinstance(message.get('content'), list) else [])
        if block.get('type') == 'tool_result' and block.get('tool_use_id') == TOOL_ID]
    content = json.dumps(tool_results)
    checks = dict(real_official_cli_called=True, mock_only=True, gpu_started=False,
        real_model_called=False, official_webpage_task_run=False, requests=len(messages),
        all_model_ids_match=bool(messages) and all(r['model'] == MODEL for r in messages),
        anthropic_version_present=bool(messages) and all(r['anthropic_version'] for r in messages),
        tool_result_received=bool(tool_results), probe_content_received=NONCE in content,
        persistent_tmpdir_received='PROBE_TMPDIR=/v2w_runtime/tmp' in content,
        adapter_success=bool(result and result.get('status') == 'success'))
    checks['complete'] = error is None and all(checks[k] for k in ('all_model_ids_match', 'anthropic_version_present',
        'tool_result_received', 'probe_content_received', 'persistent_tmpdir_received', 'adapter_success'))
    checks['error'] = error
    checks['evidence_root'] = str(folder)
    save(folder / 'complete.json', checks)
    save(ROOT / 'validation/vision_mock_protocol.json', checks)
    print(json.dumps(checks, indent=2))
    if not checks['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
