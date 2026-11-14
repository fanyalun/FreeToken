"""Evaluate five DeepSeek-VL2 policies on the frozen Write holdout."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def status(config, stage, **details):
    value = dict(stage=stage, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details)
    save(Path(config['output']) / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def wait_for_gpu(config):
    while True:
        result = subprocess.check_output(['nvidia-smi', f'--id={config["gpu"]}',
                 '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = map(int, result.strip().split(','))
        if memory < 512 and utilization == 0:
            return
        status(config, 'waiting_for_gpu', memory_mib=memory, utilization=utilization)
        time.sleep(20)


def prepare(config):
    from freetoken.moe.tablemoe_artifacts import verify_stage
    from build_deepseek_vl2_offline import conversation

    root, source = Path(config['output']), Path(config['split_root'])
    calibration, evaluation = [rows(source / f'{part}.jsonl') for part in ('calibration', 'evaluation')]
    assert len(calibration) == 72 and len(evaluation) == 48
    for key in ('id', 'group_id'):
        assert not {row[key] for row in calibration} & {row[key] for row in evaluation}
    image_sets = [{image['sha256'] for row in part for image in row['images']} for part in (calibration, evaluation)]
    assert not image_sets[0] & image_sets[1]
    assert sorted(Counter(row['metadata']['category'] for row in evaluation).values()) == [8] * 6
    target = root / 'prepare/MMLongBench-Write'
    target.mkdir(parents=True, exist_ok=True)
    messages = []
    for part, data in (('calibration', calibration), ('evaluation', evaluation)):
        for row in data:
            assert [item['value'] for item in row['vlmeval_message'] if item['type'] == 'text'] == [row['metadata']['question']]
            assert all(Path(image['path']).is_file() for image in row['images'])
            row.pop('messages', None)
            messages.append(dict(id=row['id'], messages=conversation(row)))
        (target / f'{part}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in data))
    (target / 'split.json').write_bytes((source / 'split.json').read_bytes())
    (target / 'messages.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in messages))
    cache = verify_stage(config['cache_dir'], hash_outputs=False)
    assert len(cache['pca_outputs']) == 58 and not cache['missing_tables']
    assert cache['dtype'] == 'bfloat16'
    for name in cache['pca_outputs']:
        assert (Path(config['pca_dir']) / name).is_file()
    save(root / 'prepare/complete.json', dict(complete=True, calibration=72, evaluation=48,
         official_prompt=True, original_image_order=True, group_and_image_disjoint=True))


def worker(config, method):
    import torch
    from transformers import LogitsProcessor
    from build_deepseek_vl2_offline import conversation
    from deepseek_vl2.models import DeepseekVLV2Processor
    from deepseek_vl2.utils.io import load_pil_images
    from deepseek_vl2_runtime import Context, POLICIES, Policy, build_model, reset_request
    from table_moe.utils.modality import ModalityContext

    root, policy = Path(config['output']), POLICIES[method]
    folder = root / 'cells' / method
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'generation_complete.json').exists():
        assert len(rows(folder / 'requests.jsonl')) == 48
        return
    torch.set_num_threads(8)
    torch.manual_seed(42)
    context = Context()
    validation_policy = Policy('tablemoe_full_validation') if method.startswith('tablemoe') else POLICIES['baseline']
    model, cache, initial, lookups = build_model(config, validation_policy, context)
    processor = DeepseekVLV2Processor.from_pretrained(config['model'])
    evaluation = rows(root / 'prepare/MMLongBench-Write/evaluation.jsonl')
    reference = read(root / 'validation/native_reference.json')
    audit = []
    hooks = []
    for layer in model.language.model.layers[1:]:
        def gate_hook(module, inputs, result):
            phase = 'prefill' if inputs[0].shape[1] > 1 else 'decode'
            context.counters[f'{phase}_gate_topk_{result[0].shape[-1]}_calls'] += 1
            context.counters[f'{phase}_gate_routes'] += result[0].numel()
        hooks.append(layer.mlp.gate.register_forward_hook(gate_hook))

    class Trace(LogitsProcessor):
        def __init__(self, prompt_length, row_id, started, report):
            self.prompt_length, self.row_id, self.started, self.report = prompt_length, row_id, started, report

        def __call__(self, input_ids, scores):
            output_position = input_ids.shape[1] - self.prompt_length + 1
            context.output_position = output_position + 1
            if self.report and output_position % 128 == 0:
                save(folder / 'progress.json', dict(stage='generating', id=self.row_id, generated_tokens=output_position,
                     elapsed_s=time.perf_counter()-self.started, counters=dict(context.counters)))
            return scores

    def generate(row, maximum, report=False):
        reset_request(cache, initial, lookups)
        messages = conversation(row)
        images = load_pil_images(messages)
        try:
            prepared = processor(conversations=messages, images=images, force_batchify=True,
                                 system_prompt='').to('cuda:0', dtype=torch.bfloat16)
        finally:
            for image in images:
                image.close()
        context.reset(prepared.images_seq_mask)
        ModalityContext.clear()
        ModalityContext.set_input_ids(prepared.input_ids, prepared.attention_mask)
        started = time.perf_counter()
        with torch.inference_mode():
            embeddings = model.prepare_inputs_embeds(**prepared)
            output = model.generate(inputs_embeds=embeddings, input_ids=prepared.input_ids,
                images=prepared.images, images_seq_mask=prepared.images_seq_mask, images_spatial_crop=prepared.images_spatial_crop,
                attention_mask=prepared.attention_mask, do_sample=False, max_new_tokens=maximum, use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id, bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                logits_processor=[Trace(len(prepared.input_ids[0]), row['id'], started, report)])
        torch.cuda.synchronize()
        tokens = output[0, len(prepared.input_ids[0]):].tolist()
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        if not text.strip():
            raise ValueError(f'empty output: {row["id"]}')
        counts = dict(context.counters)
        result = dict(id=row['id'], index=row['index'], text=text, prediction_text=text, output_token_ids=tokens,
                      usage=dict(prompt_tokens=len(prepared.input_ids[0]), completion_tokens=len(tokens)),
                      finish_reason='stop' if tokens[-1] == processor.tokenizer.eos_token_id else 'length',
                      elapsed_s=time.perf_counter()-started, routing=counts, formatted_prompt=prepared.sft_format)
        ModalityContext.clear()
        return result

    try:
        native_row = next(row for row in evaluation if row['id'] == reference['id'])
        probe = generate(native_row, len(reference['tokens']))
        if probe['output_token_ids'] != reference['tokens']:
            save(folder / 'native_parity_failure.json', dict(reference=reference, observed=probe))
            raise ValueError('all-expert native/offload greedy token parity failed')
        save(folder / 'native_parity.json', dict(passed=True, sample=reference['id'], tokens=len(reference['tokens']),
             dtype='bfloat16', attention='flash_attention_2', tablemoe_full_expert_validation=method.startswith('tablemoe')))
        for layer in model.language.model.layers[1:]:
            if method.startswith('tablemoe'):
                layer.mlp.policy = policy
            elif method == 'adapmoe_gating_top5':
                layer.mlp.gate.top_k = 5
                layer.mlp.top_k = 5
        if method != 'baseline':
            save(folder / 'policy_smoke.json', generate(native_row, 32))
        previous = rows(folder / 'requests.jsonl') if (folder / 'requests.jsonl').exists() else []
        if [item['id'] for item in previous] != [row['id'] for row in evaluation[:len(previous)]]:
            raise ValueError('saved prediction order differs from the frozen holdout')
        for row in evaluation[len(previous):]:
            result = generate(row, config['max_tokens'], report=True)
            previous.append(result)
            temporary = folder / 'requests.jsonl.partial'
            temporary.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in previous))
            temporary.replace(folder / 'requests.jsonl')
            save(folder / 'progress.json', dict(stage='request_complete', samples=len(previous), total=48,
                 id=row['id'], completion_tokens=result['usage']['completion_tokens'], elapsed_s=result['elapsed_s']))
            print('COMPLETED', method, len(previous), row['id'], result['usage'], flush=True)
        save(folder / 'generation_complete.json', dict(complete=True, samples=len(previous), policy=policy.__dict__,
             dtype='bfloat16', expert_cache_ratio=.5, expert_slots=1044, initial_resident_experts=initial,
             per_request_expert_cache_reset=True, request_isolated_online_capacity=128,
             tablemoe_original_route_weights=True, native_gate_top5=method == 'adapmoe_gating_top5',
             native_parity=read(folder / 'native_parity.json'), formal_performance_eligible=False))
    finally:
        for hook in hooks:
            hook.remove()
        cache.close()


def environment(config):
    root = Path(config['output'])
    snapshot = root / 'runtime_snapshot'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(config['gpu']), 'TABLEMOE_USE_FP16_FUSED_EXPERT': '0',
           'TABLEMOE_DEEPSEEK_CACHE_RATIO': '.5', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
           'PYTHONPATH': ':'.join(str(snapshot / item) for item in ('python', 'benchmarks', 'legacy', 'native')),
           'CONDA_DEFAULT_ENV': 'deepseekvl2', 'CONDA_PREFIX': str(Path(config['python']).parent.parent)}
    env.pop('VIRTUAL_ENV', None)
    for name, folder in (('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'), ('TRITON_CACHE_DIR', 'triton_cache'),
                         ('TORCH_EXTENSIONS_DIR', 'torch_extensions'), ('CUDA_CACHE_PATH', 'cuda_cache')):
        path = root / folder
        path.mkdir(exist_ok=True)
        env[name] = str(path)
    return env


def controller(config):
    root = Path(config['output'])
    prepare(config)
    for method in config['methods']:
        folder = root / 'cells' / method
        if (folder / 'generation_complete.json').exists():
            continue
        wait_for_gpu(config)
        status(config, 'generation', method=method)
        command = [config['python'], str(Path(__file__).resolve()), 'worker', '--config', str(root / 'config.json'), '--method', method]
        with (root / 'logs' / f'{method}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=root / 'runtime_snapshot', env=environment(config),
                                     stdout=log, stderr=subprocess.STDOUT)
            save(root / 'current_child.json', dict(pid=child.pid, method=method, command=command))
            if child.wait() != 0:
                raise RuntimeError(f'generation worker failed: {method}; see {log.name}')
    status(config, 'generation_complete', methods=config['methods'])
    score_all(config)


def score_all(config):
    root = Path(config['output'])
    wait_for_gpu(config)
    status(config, 'judging')
    env = environment(config)
    env.update(PYTHONPATH=str(root / 'judge_runtime_snapshot/python') + ':' + str(root / 'judge_runtime_snapshot/benchmarks'),
               CUDA_HOME='/usr/local/cuda-13.0', PATH='/usr/local/cuda-13.0/bin:' + os.environ['PATH'])
    command = [config['judge_python'], '-m', 'freetoken.cli', 'serve', '--model', config['judge_model'], '--gpu', '0',
               '--dtype', 'bfloat16', '--moe-strategy', 'fused', '--moe-cache-size', '0', '--cuda-graph-max-bs', '4',
               '--max-running-requests', '4', '--num-pages', '131072', '--page-size', '1', '--cache-type', 'naive',
               '--mm-encoder-weights', 'gpu', '--max-prefill-length', '4096', '--max-seq-len-override', '65536',
               '--port', str(config['judge_port']), '--mm-processor-kwargs', json.dumps(dict(size=dict(shortest_edge=602112, longest_edge=4014080))),
               '--reasoning-parser', 'off']
    with (root / 'logs/judge_server.log').open('a') as log:
        judge = subprocess.Popen(command, cwd=root / 'judge_runtime_snapshot', env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        save(root / 'judge_child.json', dict(pid=judge.pid, command=command))
        try:
            import requests
            for _ in range(240):
                if judge.poll() is not None:
                    raise RuntimeError('judge server exited during startup')
                try:
                    if requests.get(f'http://127.0.0.1:{config["judge_port"]}/v1/models', timeout=3).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5)
            else:
                raise RuntimeError('judge server startup timed out')
            for method in config['methods']:
                folder = root / 'cells' / method
                scoring = environment(config)
                scoring.update(PYTHONPATH=str(root / 'runtime_snapshot/python') + ':' + str(root / 'runtime_snapshot/benchmarks'),
                               LD_LIBRARY_PATH='/home/fanya/miniforge/envs/tablemoe/lib', LMUData='/home/fanya/tablemoe/LMUData')
                for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                    scoring.pop(name, None)
                with (root / 'logs' / f'{method}_judge.log').open('a') as output:
                    subprocess.run([config['kit_python'], str(root / 'kit_worker.py'), 'score', '--dataset', 'MMLongBench-Write',
                                    '--folder', str(folder), '--port', str(config['judge_port'])], cwd=root,
                                   env=scoring, stdout=output, stderr=subprocess.STDOUT, check=True)
        finally:
            import signal
            if judge.poll() is None:
                os.killpg(judge.pid, signal.SIGTERM)
                try:
                    judge.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(judge.pid, signal.SIGKILL)
                    judge.wait()
    comparison = {}
    for method in config['methods']:
        score = root / 'cells' / method / 'score/score.json'
        if not score.exists():
            raise ValueError(f'missing native score: {score}')
        comparison[method] = read(score)
    save(root / 'comparison.json', comparison)
    save(root / 'complete.json', dict(complete=True, methods=config['methods'], samples_per_method=48,
         scores=comparison, quality_only=True, formal_performance_eligible=False))
    status(config, 'complete', methods=config['methods'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'run'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--method')
    args = parser.parse_args()
    config = read(args.config)
    snapshot = Path(config['output']) / 'runtime_snapshot'
    sys.path[:0] = [str(snapshot / part) for part in ('python', 'benchmarks', 'legacy', 'native')]
    try:
        if args.action == 'prepare':
            prepare(config)
        elif args.action == 'worker':
            worker(config, args.method)
        else:
            with (Path(config['output']) / '.job.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                controller(config)
    except BaseException:
        save(Path(config['output']) / ('worker_error_' + str(args.method) + '.json' if args.action == 'worker' else 'error.json'),
             dict(complete=False, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
