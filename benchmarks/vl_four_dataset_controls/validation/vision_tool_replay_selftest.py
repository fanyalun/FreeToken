"""Replay the failed real Instruct output through the frozen Anthropic stream."""

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vision_pipeline as pipeline
from freetoken.message.frontend import UserReply
from freetoken.server.anthropic_api import anthropic_event_stream, convert_anthropic_to_genspec
from freetoken.server.anthropic_models import AnthropicMessagesRequest
from freetoken.server.generation import _generate_events_impl


async def replay(text, request, reasoning, chunk):
    async def replies(uid):
        for offset in range(0, len(text), chunk):
            yield UserReply(uid=uid, incremental_output=text[offset:offset + chunk],
                finished=offset + chunk >= len(text), prompt_tokens_delta=1 if offset == 0 else 0,
                completion_tokens_delta=1)
    config = SimpleNamespace(reasoning_parser=reasoning, tool_call_parser='qwen25')
    state = SimpleNamespace(config=config, wait_for_ack=replies)
    spec = convert_anthropic_to_genspec(request, {}, reasoning_parser=reasoning)
    stream = _generate_events_impl(1, spec, state)
    frames = []
    async for frame in anthropic_event_stream(stream, request.model, 1):
        for line in frame.splitlines():
            if line.startswith('data: '):
                frames.append(json.loads(line[6:]))
    blocks = [f['content_block'] for f in frames if f['type'] == 'content_block_start']
    tools = [b for b in blocks if b['type'] == 'tool_use']
    deltas = [f['delta'] for f in frames if f['type'] == 'content_block_delta']
    arguments = ''.join(d['partial_json'] for d in deltas if d['type'] == 'input_json_delta')
    stop = next(f['delta']['stop_reason'] for f in frames if f['type'] == 'message_delta')
    return dict(block_types=[b['type'] for b in blocks], tool_names=[b['name'] for b in tools],
                arguments=json.loads(arguments) if arguments else None, stop_reason=stop)


def main():
    failed = Path('/data/fanya/tablemoe/vl_four_dataset_bs1_20261111/vision2web_runs/smoke/fused')
    result = pipeline.read(failed / 'results/webpage/claude_code/Qwen3-VL-30B-A3B-Instruct-fused/aihw/results/aihw.json')
    texts = [block['thinking'] for event in result['conversation'] if event['type'] == 'assistant'
             for block in event['message']['content'] if block['type'] == 'thinking']
    assert len(texts) == 2 and result['status'] == 'failed'
    prior = pipeline.read(ROOT / 'validation/vision_mock_probe_1794422272059073330/requests.json')[0]['body']
    request = AnthropicMessagesRequest.model_validate(prior)
    assert request.thinking == {'type': 'adaptive'} and request.max_tokens == 64000
    checks = []
    for index, text in enumerate(texts):
        expected = json.loads(text.split('<tool_call>')[1].split('</tool_call>')[0])
        for chunk in (1, 7, 97, len(text)):
            old = asyncio.run(replay(text, request, 'qwen3', chunk))
            corrected = asyncio.run(replay(text, request, None, chunk))
            assert old['tool_names'] == [] and old['block_types'] == ['thinking']
            assert old['stop_reason'] == 'end_turn'
            assert corrected['tool_names'] == [expected['name']]
            assert corrected['arguments'] == expected['arguments']
            assert corrected['stop_reason'] == 'tool_use'
            assert 'thinking' not in corrected['block_types']
            checks.append(dict(output=index, chunk_characters=chunk, old=old, corrected=corrected))
    output = dict(complete=True, real_failed_output_replayed=True, cases=len(checks), checks=checks,
                  gpu_started=False, real_model_called=False, official_prompt_modified=False,
                  tool_parser='qwen25', corrected_reasoning_parser=None, failed_run=str(failed))
    pipeline.write_json(ROOT / 'validation/vision_tool_replay_cpu.json', output)
    print(json.dumps(dict(complete=True, cases=len(checks), tool_names=['Bash', 'Read'], gpu_started=False)))


if __name__ == '__main__':
    main()
