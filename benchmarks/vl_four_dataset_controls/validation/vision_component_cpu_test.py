"""Exercise the component request worker with local mock HTTP and native judge parsing."""

import asyncio
from http.server import ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vision_component_probe as probe
import vision_mock_probe as mock


class Handler(mock.Handler):
    invalid = False
    terminal_mode = 'valid'
    judge_requests = []

    def do_POST(self):
        if '/chat/completions' not in self.path:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            events = [dict(type='message_start', message=dict(usage=dict(input_tokens=0, output_tokens=0))),
                dict(type='content_block_delta', index=0, delta=dict(type='text_delta', text='Fixture heading.')),
                dict(type='message_delta', delta=dict(stop_reason='end_turn'), usage=dict(input_tokens=200, output_tokens=20)),
                dict(type='message_stop')]
            if self.terminal_mode == 'missing':
                events[-2].pop('usage')
            elif self.terminal_mode == 'zero':
                events[-2]['usage']['output_tokens'] = 0
            for event in events:
                self.wfile.write(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'.encode())
            self.wfile.flush()
            return
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.judge_requests.append(body)
        content = [] if self.invalid else [dict(component='fixture', score=.5)]
        self.respond(dict(id='mock_judge', object='chat.completion', created=1,
            model=body['model'], choices=[dict(index=0, finish_reason='stop',
                message=dict(role='assistant', content=json.dumps(content)))],
            usage=dict(prompt_tokens=200, completion_tokens=30, total_tokens=230)))


def main():
    folder = ROOT / 'validation' / f'vision_component_cpu_{time.time_ns()}'
    folder.mkdir(parents=True)
    browser, image = probe.required_cpu_receipts()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = dict(generation=dict(model=mock.MODEL), evaluation=dict(model=mock.MODEL,
                  base_url=f'http://127.0.0.1:{server.server_port}/v1'))
    config_path = folder / 'config.json'
    probe.save(config_path, config)
    try:
        probe.request_worker(config_path, folder)
        assert probe.read(folder / 'request_worker_complete.json')['complete']
        body = Handler.judge_requests[0]
        assert body['max_tokens'] == 64000
        assert sum(block.get('type') == 'image_url' for block in body['messages'][0]['content']) == 2
        assert probe.read(folder / 'anthropic_request.json')['tools'] == probe.read(image['image_request'])['tools']
        three = probe.read(folder / 'anthropic_three_image_request.json')
        assert three['tools'] == probe.read(image['image_request'])['tools']
        assert three['system'] == probe.read(image['image_request'])['system']
        assert three['max_tokens'] == 64000
        assert sum(block['type'] == 'image' for block in three['messages'][0]['content']) == 3
        assert all(not item['resized'] for item in probe.read(folder / 'three_image_inputs.json'))
        for mode in ('missing', 'zero'):
            Handler.terminal_mode = mode
            failure = folder / f'invalid_terminal_{mode}'
            failure.mkdir()
            try:
                probe.anthropic_probe(config, failure, image)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'{mode} terminal usage was accepted')
        Handler.terminal_mode = 'valid'
        Handler.invalid = True
        failure = folder / 'invalid_judge'
        failure.mkdir()
        try:
            asyncio.run(probe.judge_probe(config, failure, image))
        except RuntimeError:
            pass
        else:
            raise AssertionError('empty judge result was accepted')
        for startup in (True, False):
            owned = folder / ('owned_server_ready' if startup else 'owned_server_startup_timeout')
            program = ('import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); '
                'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                + ('print("API server is ready to serve",flush=True); ' if startup else '')
                + 'time.sleep(60)')
            try:
                with probe.owned_server([sys.executable, '-u', '-c', program], owned,
                                        ROOT, dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                        2 if startup else .3, shutdown_graces=(.2, .2, 1)):
                    assert startup
            except TimeoutError:
                assert not startup
            owner = probe.read(owned / 'server_process.json')
            cleanup = probe.read(owned / 'server_cleanup.json')
            assert owner['pid'] == owner['pgid'] == owner['session_id']
            assert owner['start_ticks'] > 0 and cleanup['complete']
            assert cleanup['signals'][-1] == 'SIGKILL' and not cleanup['remaining']
            assert probe.process_identity(owner['pid']) is None
        checks = dict(complete=True, gpu_started=False, real_model_called=False,
            fixture_only=True, native_judge_prompt_and_parser=True,
            real_cli_image_body_replayed=True, official_max_tokens=64000,
            three_original_prototype_images=True, unchanged_cli_system_and_tools=True,
            invalid_judge_rejected=True, judge_attempts=len(Handler.judge_requests), evidence_root=str(folder))
        checks.update(missing_terminal_usage_rejected=True, zero_output_usage_rejected=True,
                      owned_server_kill_and_reap=True, startup_timeout_cleanup=True)
        probe.save(ROOT / 'validation/vision_component_cpu.json', checks)
        print(json.dumps(checks, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()
