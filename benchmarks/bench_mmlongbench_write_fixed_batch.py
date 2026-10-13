"""Replay a frozen MMLongBench configuration with atomic offline request groups."""

import argparse
import asyncio
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def export_predictions(path, predictions, max_tokens):
    records = []
    for row in predictions:
        records.append({**row, 'natural_eos': True, 'max_tokens': max_tokens,
            'usage': dict(prompt_tokens=row['prompt_tokens'],
                completion_tokens=len(row['sampled_token_ids']),
                total_tokens=row['prompt_tokens'] + len(row['sampled_token_ids']))})
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))


def decode_tokens(tokenizer, tokens, eos_token_ids, finish_reason):
    from freetoken.message import DetokenizeMsg
    from freetoken.tokenizer.detokenize import DetokenizeManager

    manager = DetokenizeManager(tokenizer, eos_token_ids)
    pieces = []
    for index, token in enumerate(tokens):
        finished = index == len(tokens) - 1
        message = DetokenizeMsg(uid=0, next_token=token, finished=finished,
            finish_reason=finish_reason if finished else None)
        pieces.extend(manager.detokenize([message]))
    return ''.join(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['baseline', 'tablemoe', 'modes'], default='tablemoe')
    parser.add_argument('--source-snapshot', type=Path)
    parser.add_argument('--command-json', type=Path)
    parser.add_argument('--prefill-keep-rate', type=float)
    parser.add_argument('--decode-keep-k', type=int)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--samples', type=int, default=48)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--port', type=int, default=22520,
        help='Base port for the single-rank distributed rendezvous; no HTTP server is started.')
    parser.add_argument('--vision-threshold', type=float)
    parser.add_argument('--text-threshold', type=float)
    args = parser.parse_args()
    assert args.batch_size > 0 and args.samples > 0 and args.repeats > 0 and args.max_tokens > 0
    reference, output = args.reference.resolve(), args.output.resolve()
    original_config = json.loads((reference / 'config.json').read_text())
    snapshot = (args.source_snapshot or reference / 'source_snapshot').resolve()
    # Reexec before importing torch so CUDA and the frozen import path are effective.
    if os.environ.get('FT_FIXED_BATCH_BOOTSTRAPPED') != str(output):
        env = {**os.environ, **original_config['server_env'],
            'CUDA_VISIBLE_DEVICES': str(args.gpu), 'PYTHONPATH': str(snapshot / 'python'),
            'FT_FIXED_BATCH_BOOTSTRAPPED': str(output)}
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'runner.py').write_bytes(Path(__file__).read_bytes())
    sys.path[:0] = [str(snapshot / 'python'), str(snapshot / 'benchmarks')]

    import torch
    from freetoken.llm.llm import RequestAllFinished
    from freetoken.message import DetokenizeMsg, PromptAdmittedMsg, TokenizeMsg
    from freetoken.mm.media import collect_image_refs, fetch_image_bytes
    from freetoken.mm.processor import get_mm_processor
    from freetoken.scheduler import Scheduler
    from freetoken.server.api_models import ChatCompletionRequest
    from freetoken.server.args import parse_args
    from freetoken.server.openai_api import chat_request_to_genspec
    from freetoken.tokenizer.tokenize import TokenizeManager
    from freetoken.utils import load_generation_sampling, load_tokenizer
    from tablemoe_data import read_rows

    command_path = args.command_json or reference / 'predictions/tablemoe/command.json'
    command = json.loads(command_path.read_text())['command']
    flags = command[command.index('serve') + 1:]
    if '--tablemoe-report-path' in flags:
        flags[flags.index('--tablemoe-report-path') + 1] = str(output / 'audit.jsonl')
    else:
        flags.extend(['--tablemoe-report-path', str(output / 'audit.jsonl')])
    flags[flags.index('--port') + 1] = str(args.port)
    config, _ = parse_args(flags)
    updates = dict(offline_mode=True)
    if args.mode == 'modes':
        assert config.moe_policy == 'modes' and config.modes_config
        assert all(value is None for value in [args.prefill_keep_rate, args.decode_keep_k,
            args.vision_threshold, args.text_threshold])
    if args.mode == 'baseline':
        assert args.vision_threshold is None and args.text_threshold is None
        assert args.prefill_keep_rate is None and args.decode_keep_k is None
        updates.update(tablemoe_prefill_keep_rate=1.0, tablemoe_decode_keep_k=8,
            tablemoe_pca_dir=None, tablemoe_cache_dir=None,
            tablemoe_decode_online_cosine_threshold=None,
            tablemoe_decode_offline_cosine_threshold=None)
    if args.prefill_keep_rate is not None:
        assert 0 < args.prefill_keep_rate <= 1
        updates['tablemoe_prefill_keep_rate'] = args.prefill_keep_rate
    if args.decode_keep_k is not None:
        assert 0 <= args.decode_keep_k <= 8
        updates['tablemoe_decode_keep_k'] = args.decode_keep_k
    if args.vision_threshold is not None:
        updates['tablemoe_offline_cosine_threshold_vision'] = args.vision_threshold
    if args.text_threshold is not None:
        updates.update(tablemoe_offline_cosine_threshold_text=args.text_threshold,
            tablemoe_decode_online_cosine_threshold=args.text_threshold,
            tablemoe_decode_offline_cosine_threshold=args.text_threshold)
    config = replace(config, **updates)
    assert config.max_running_req == 4 and args.batch_size <= config.max_running_req
    assert config.max_extend_tokens == 4096
    print('OFFLINE_FIXED_BATCH', dict(offline_mode=config.offline_mode,
        batch_size=args.batch_size, prefill_budget=config.max_extend_tokens), flush=True)
    tokenizer = load_tokenizer(config.model_path)
    manager = TokenizeManager(tokenizer, get_mm_processor(config.model_path, config.mm))
    model_sampling = load_generation_sampling(config.model_path)
    rows = read_rows(reference / 'prepare/evaluation.jsonl')[:args.samples]
    assert len(rows) == args.samples
    warmup = read_rows(reference / 'prepare/calibration.jsonl')[0]

    def prepare(row, uid, limit, natural=True):
        body = dict(model=Path(config.model_path).name, messages=row['messages'],
            max_tokens=limit, ignore_eos=not natural, temperature=0, stream=True,
            stream_options={'include_usage': True})
        spec = chat_request_to_genspec(ChatCompletionRequest(**body), model_sampling)
        refs = collect_image_refs(spec.messages)
        images = asyncio.run(fetch_image_bytes(refs, config)) if refs else None
        token_msg = TokenizeMsg(uid=uid, text=spec.messages, sampling_params=spec.sampling_params,
            chat_template_kwargs=spec.chat_template_kwargs, tools=spec.template_tools, images=images)
        message = manager.tokenize([token_msg])[0]
        identity = dict(id=row['id'], request_sha256=digest(body),
            input_ids_sha256=digest(message.input_ids.tolist()), prompt_tokens=len(message.input_ids),
            image_sha256=[hashlib.sha256(image).hexdigest() for image in images or []],
            sampling=asdict(message.sampling_params))
        return message, identity

    prepared = [prepare(row, i, args.max_tokens) for i, row in enumerate(rows)]
    # UID -1 is the runtime's CUDA-graph dummy request, not a real warmup.
    warmup_msg, warmup_identity = prepare(warmup, len(rows), 10, natural=False)
    save(output / 'inputs.json', [identity for _, identity in prepared])
    save(output / 'setup.json', dict(reference=str(reference), source_snapshot=str(snapshot),
        arguments=vars(args), config=asdict(config), warmup=warmup_identity,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy='all group requests preprocessed and admitted together; drain group; no refill',
        overrides=updates, quality_scoring=False))

    class FixedBatchScheduler(Scheduler):
        def __init__(self, cfg):
            self.pending_group = []
            self.outputs = {}
            self.finished = set()
            self.schedule = []
            super().__init__(cfg)
            forward = self.engine.forward_batch

            def traced_forward(batch, *pos, **kw):
                self.schedule.append(dict(phase=batch.phase, padded_size=batch.padded_size,
                    requests=[dict(uid=r.uid, cached_len=r.cached_len, device_len=r.device_len,
                        extend_len=r.extend_len) for r in batch.reqs]))
                return forward(batch, *pos, **kw)

            self.engine.forward_batch = traced_forward

        def offline_receive_msg(self, blocking=False):
            if self.pending_group:
                group, self.pending_group = self.pending_group, []
                return group
            if blocking:
                assert self.finished == set(self.outputs), 'scheduler idle before group finished'
                raise RequestAllFinished()
            return []

        def offline_send_result(self, replies):
            for msg in replies:
                if isinstance(msg, PromptAdmittedMsg):
                    continue
                assert isinstance(msg, DetokenizeMsg), type(msg)
                row = self.outputs[msg.uid]
                row['sampled_token_ids'].append(msg.next_token)
                if not (msg.finished and msg.next_token in self.eos_token_ids):
                    row['token_ids'].append(msg.next_token)
                if msg.finished:
                    row['finish_reason'] = msg.finish_reason
                    self.finished.add(msg.uid)

        def run_group(self, messages):
            assert not self.pending_group
            self.outputs = {m.uid: dict(token_ids=[], sampled_token_ids=[]) for m in messages}
            self.finished = set()
            self.schedule = []
            self.pending_group = messages
            try:
                self.run_forever()
            except RequestAllFinished:
                pass
            assert self.finished == set(self.outputs)
            for row in self.outputs.values():
                row['text'] = decode_tokens(tokenizer, row['sampled_token_ids'],
                    self.eos_token_ids, row['finish_reason'])
            return self.outputs, self.schedule

    engine = FixedBatchScheduler(config)
    save(output / 'effective_config.json', asdict(config))
    engine.run_group([warmup_msg])
    repeats = []
    for repeat in range(args.repeats):
        predictions, schedules = [], []
        for start in range(0, len(prepared), args.batch_size):
            group = prepared[start:start + args.batch_size]
            # Encoder consumption releases image features on the message object.
            messages = [replace(copy.deepcopy(message), uid=i)
                for i, (message, _) in enumerate(group, start=start)]
            results, schedule = engine.run_group(messages)
            schedules.append(schedule)
            for i, (_, identity) in enumerate(group, start=start):
                predictions.append({**identity, **results[i]})
            save(output / f'predictions_{repeat}.json', predictions)
            export_predictions(output / f'requests_{repeat}.jsonl', predictions, args.max_tokens)
            save(output / f'schedule_{repeat}.json', schedules)
            print('GROUP_COMPLETE', repeat, start, len(group), flush=True)
        repeats.append(dict(predictions=predictions, schedule=schedules))
    comparisons = []
    for index, repeated in enumerate(repeats[1:], 1):
        comparisons.append(dict(repeat=index,
            identical_schedule=repeats[0]['schedule'] == repeated['schedule'],
            identical_predictions=repeats[0]['predictions'] == repeated['predictions'],
            identical_token_samples=sum(a['sampled_token_ids'] == b['sampled_token_ids']
                for a, b in zip(repeats[0]['predictions'], repeated['predictions'], strict=True))))
    save(output / 'complete.json', dict(complete=True, samples=len(rows), repeats=args.repeats,
        generation_limit=args.max_tokens, comparisons=comparisons, quality_scoring=False))
    print('COMPLETE', comparisons, flush=True)


if __name__ == '__main__':
    main()
