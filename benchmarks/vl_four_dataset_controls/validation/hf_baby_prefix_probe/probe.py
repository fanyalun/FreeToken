"""Two-sample, bounded HF greedy diagnostic; CPU preparation is the default."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = '/data/fanya/models/Qwen3-VL-30B-A3B-Instruct'
PYTHON = '/home/fanya/local_archives/freetoken_consolidation_20261029_020126/preserved/FreeToken/.venv/bin/python'
IDS = ('BabyVision:464', 'BabyVision:638')
SIZE = dict(shortest_edge=602112, longest_edge=4014080)
GLOBAL_DEADLINE = 1794450189.0
MAX_SECONDS = 300


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def source_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=fields[19], state=fields[0], parent=int(fields[1]))
    except (OSError, ValueError, IndexError):
        return None


def same_process(value):
    current = identity(value['pid'])
    return current is not None and current['start_ticks'] == value['start_ticks'] and current['state'] != 'Z'


def frozen_samples():
    def select(path):
        selected = {}
        with path.open() as handle:
            for line in handle:
                item = json.loads(line)
                if item['id'] in IDS:
                    selected[item['id']] = item
        if set(selected) != set(IDS):
            raise ValueError('both frozen diagnostic samples are required')
        return selected
    rows = select(ROOT / 'prepare/BabyVision/evaluation.jsonl')
    predictions = select(ROOT / 'cells/BabyVision/fused/requests.jsonl')
    return [(rows[name], predictions[name]) for name in IDS]


def prepare_inputs(processor, row, prediction):
    import torch
    from PIL import Image
    from transformers import AutoConfig

    sys.path.insert(0, str(ROOT / 'runtime_snapshot/python'))
    from freetoken.mm.config import MultimodalConfig
    from freetoken.mm.processors.qwen_vl import QwenVLMMProcessor

    parts, images, paths = [], [], []
    for item in row['vlmeval_message']:
        if item['type'] == 'image':
            parts.append(dict(type='image'))
            paths.append(item['value'])
            with Image.open(item['value']) as image:
                images.append(image.convert('RGB'))
        elif item['type'] == 'text':
            parts.append(dict(type='text', text=item['value']))
        else:
            raise ValueError('unexpected multimodal input part')
    messages = [dict(role='user', content=parts)]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenizer = processor.tokenizer
    if prompt != tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        raise ValueError('processor and tokenizer chat templates differ')
    inputs = processor(text=[prompt], images=images, return_tensors='pt',
                       padding=False, truncation=False, size=SIZE)
    ids = inputs['input_ids'][0].tolist()
    placeholder = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    grids = inputs['image_grid_thw'].tolist()
    visual_counts = [t * h * w // 4 for t, h, w in grids]
    frozen = QwenVLMMProcessor(AutoConfig.from_pretrained(MODEL, local_files_only=True), MODEL,
                              MultimodalConfig(processor_kwargs={'size': SIZE}))
    rebuilt = frozen._image_processor()(images=images[0], **frozen.get_mm_processor_kwargs(frozen.mm))
    if rebuilt['image_grid_thw'].tolist() != grids or not torch.equal(rebuilt['pixel_values'], inputs['pixel_values']):
        raise ValueError('frozen CPU image processor differs from HF direct processor')
    expanded = []
    image_index = 0
    for token in tokenizer.encode(prompt, add_special_tokens=False):
        if token == placeholder:
            expanded.extend([token] * visual_counts[image_index])
            image_index += 1
        else:
            expanded.append(token)
    expected = {'BabyVision:464': 730, 'BabyVision:638': 702}[row['id']]
    if ids != expanded or len(ids) != expected or len(ids) != prediction['usage']['prompt_tokens']:
        raise ValueError('HF expanded input IDs or actual API token count mismatch')
    if image_index != len(images) or len(images) != 1 or '<think>' in prompt:
        raise ValueError('unexpected image/template structure')
    if prediction.get('reasoning_text') or prediction['prediction_text'] != prediction['text']:
        raise ValueError('reference response requires unexpected parser reconstruction')
    native_pixels = inputs['pixel_values']
    matched_pixels = native_pixels.to(torch.bfloat16).to(torch.float16)
    pixel_difference = (native_pixels.float() - matched_pixels.float()).abs()
    inputs['pixel_values'] = matched_pixels
    reference = tokenizer.encode(prediction['text'], add_special_tokens=False)[:32]
    detail = dict(id=row['id'], messages=messages, prompt=prompt, image_paths=paths,
                  input_ids=ids, input_tokens=len(ids), actual_api_prompt_tokens=prediction['usage']['prompt_tokens'],
                  image_grid_thw=grids, visual_tokens=sum(visual_counts), batch_size=1,
                  extra_system_prompt=False, extra_answer_only=False, input_ids_match_expansion=True,
                  historical_input_ids_available=False, historical_input_validation='API usage length only',
                  input_ids_comparison='HF direct IDs equal reconstructed checkpoint-tokenizer and frozen Qwen image-grid expansion; FreeToken content-pad cache IDs are normalized to image_pad.',
                  frozen_cpu_image_processor_matches_hf=True,
                  pixel_values_shape=list(inputs['pixel_values'].shape), pixel_values_dtype=str(inputs['pixel_values'].dtype),
                  pixel_alignment=dict(native_processor_dtype=str(native_pixels.dtype),
                      matched_runtime_dtype=str(matched_pixels.dtype), cast_chain=['float32', 'bfloat16', 'float16'],
                      max_abs_difference=float(pixel_difference.max()), mean_abs_difference=float(pixel_difference.mean()),
                      values_changed=int((native_pixels.float() != matched_pixels.float()).sum()),
                      cpu_source='Frozen QwenVLMMProcessor.process casts to BF16; VisionConv3dPatchEmbed.forward casts to FP16 projection weight dtype.'),
                  saved_fused_response_id=prediction['response_id'],
                  fused_retokenized_prefix_ids=reference,
                  fused_retokenized_prefix_text=tokenizer.decode(reference, skip_special_tokens=False),
                  fused_raw_response_initial_text=prediction['text'][:512],
                  retokenization_note='Original API output token IDs were not saved; these IDs are reconstructed from the preserved response text.')
    return dict(inputs), detail


def processor_and_inputs():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    records = [prepare_inputs(processor, row, prediction) for row, prediction in frozen_samples()]
    return processor, records


def preflight():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import transformers
    from transformers import Qwen3VLMoeForConditionalGeneration

    processor, records = processor_and_inputs()
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    for _, detail in records:
        save(HERE / 'inputs' / (detail['id'].replace(':', '_') + '.json'), detail)
    result = dict(complete=True, cpu_only=True, gpu_started=False, model_loaded=False,
                  source_sha256=source_sha(), model=MODEL, dtype='float16', attention='sdpa',
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  model_class=Qwen3VLMoeForConditionalGeneration.__name__,
                  ids=list(IDS), input_tokens=[d['input_tokens'] for _, d in records],
                  max_new_tokens=32, do_sample=False, batch_size=1,
                  global_deadline=GLOBAL_DEADLINE, maximum_job_seconds=MAX_SECONDS,
                  precision_note='HF model with matched FreeToken preprocessing: processor FP32 -> BF16 wire rounding -> FP16 vision input. This is not native-HF default end-to-end preprocessing.',
                  comparison_limit='A short prefix match is limited evidence; mismatch alone is not a bug verdict, including near-tie logits, different FP16 kernels and output-text retokenization.')
    save(HERE / 'cpu_preflight.json', result)
    print(json.dumps(result, indent=2))


def worker(deadline):
    import torch
    from transformers import Qwen3VLMoeForConditionalGeneration

    started = time.time()
    processor, records = processor_and_inputs()
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    loaded = time.time()
    if loaded >= deadline:
        raise TimeoutError('model load exhausted the bounded GPU window')
    results = []
    for inputs, detail in records:
        if time.time() >= deadline:
            raise TimeoutError('diagnostic deadline reached')
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        before = time.time()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False, num_beams=1,
                                    use_cache=True, repetition_penalty=1.0,
                                    return_dict_in_generate=True, output_scores=True, output_logits=True)
        torch.cuda.synchronize()
        tokens = output.sequences[0, inputs['input_ids'].shape[1]:].cpu().tolist()
        reference = detail['fused_retokenized_prefix_ids']
        logits = output.logits[0][0].float()
        best_values, best_ids = logits.topk(5)
        fused_logit = logits[reference[0]]
        first_step = dict(top5=[dict(token_id=token, token_text=processor.tokenizer.decode([token]), logit=value)
                               for token, value in zip(best_ids.cpu().tolist(), best_values.cpu().tolist())],
                          fused_first_token_id=reference[0], fused_first_token_logit=float(fused_logit),
                          fused_first_token_rank=1+int((logits > fused_logit).sum()),
                          top1_minus_top2_logit=float(best_values[0]-best_values[1]),
                          top1_minus_fused_logit=float(best_values[0]-fused_logit),
                          scope='First-step raw HF logits from the same generate call; no extra request; full-vocabulary tensors not saved.')
        matched = 0
        for actual, previous in zip(tokens, reference):
            if actual != previous:
                break
            matched += 1
        result = dict(id=detail['id'], hf_prefix_ids=tokens,
                      hf_raw_prefix=processor.tokenizer.decode(tokens, skip_special_tokens=False),
                      fused_retokenized_prefix_ids=reference,
                      fused_retokenized_prefix=detail['fused_retokenized_prefix_text'],
                      fused_raw_response_initial_text=detail['fused_raw_response_initial_text'],
                      matched_prefix_tokens=matched, compared_tokens=min(len(tokens), len(reference)),
                      first_step_logits=first_step,
                      first_32_token_match=len(tokens) == len(reference) == matched == 32,
                      input_tokens=detail['input_tokens'], elapsed_seconds=time.time() - before)
        results.append(result)
        save(HERE / 'progress.json', dict(complete=False, completed=len(results), total=2, results=results))
    if time.time() >= deadline:
        raise TimeoutError('generation finished after the operation deadline')
    save(HERE / 'worker_complete.json', dict(complete=True, diagnostic_only=True,
        benchmark_score=False, model=MODEL, dtype='float16', batch_size=1, do_sample=False,
        max_new_tokens=32, results=results, load_seconds=loaded-started,
        total_seconds=time.time()-started, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        source_sha256=source_sha(), numerical_bug_verdict=None,
        limits=read(HERE / 'cpu_preflight.json')['comparison_limit']))


def supervise(gpu):
    started = time.time()
    policy_deadline = read(ROOT / 'execution_policy.json')['deadline_unix']
    end = min(started + MAX_SECONDS, GLOBAL_DEADLINE, policy_deadline)
    if end - started <= 10:
        raise TimeoutError('insufficient authorized time for bounded diagnostic')
    prepared = read(HERE / 'cpu_preflight.json')
    release = read(HERE / 'release.json')
    if not prepared.get('complete') or prepared['source_sha256'] != source_sha():
        raise ValueError('CPU preflight source does not match')
    if not release.get('complete') or release.get('source_sha256') != source_sha():
        raise ValueError('root release for this exact source is required')
    if (HERE / 'ownership.json').exists() or (HERE / 'complete.json').exists():
        raise ValueError('preserve the existing diagnostic; no implicit rerun')
    command = [PYTHON, '-B', str(Path(__file__).resolve()), '--worker', '--deadline', str(end - 5)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    env.pop('PYTHONPATH', None)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        env[name] = str(HERE / 'tmp')
    owned = []
    cleaning = False
    def interrupted(signum, frame):
        if not cleaning:
            raise InterruptedError(f'owned diagnostic interrupted by {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    failure = None
    with (HERE / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, env=env, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        own = identity(process.pid)
        if own is not None:
            owned.append(own)
        save(HERE / 'ownership.json', dict(supervisor=identity(os.getpid()), worker=own,
            command=command, gpu=gpu, started_unix=started, deadline=end,
            operation_deadline=end-5, maximum_seconds=MAX_SECONDS, source_sha256=source_sha()))
        try:
            while process.poll() is None:
                if time.time() >= end - 5:
                    raise TimeoutError('bounded diagnostic reached its operation deadline')
                time.sleep(.25)
            if process.returncode:
                raise RuntimeError(f'HF worker exited {process.returncode}')
            result = read(HERE / 'worker_complete.json')
            if not result.get('complete') or result.get('source_sha256') != source_sha():
                raise ValueError('worker completion does not match source')
        except BaseException as error:
            failure = repr(error)
        finally:
            cleaning = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for value in owned:
                    if same_process(value):
                        try:
                            os.kill(value['pid'], sig)
                        except ProcessLookupError:
                            pass
                if process.poll() is not None:
                    break
                try:
                    process.wait(timeout=max(.01, min(2, end-time.time())))
                except subprocess.TimeoutExpired:
                    pass
            save(HERE / 'cleanup.json', dict(owned=owned,
                still_alive=[v for v in owned if same_process(v)], finished_unix=time.time()))
    if failure is not None:
        save(HERE / 'failure.json', dict(complete=False, failure=failure, finished_unix=time.time()))
        raise RuntimeError(failure)
    save(HERE / 'complete.json', dict(complete=True, diagnostic_only=True, benchmark_score=False,
         result=str(HERE / 'worker_complete.json'), elapsed_seconds=time.time()-started,
         source_sha256=source_sha(), finished_unix=time.time()))


def main():
    (HERE / 'tmp').mkdir(exist_ok=True)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[name] = str(HERE / 'tmp')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, choices=(0, 1))
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.worker:
        worker(args.deadline)
    elif args.execute:
        if args.gpu is None:
            parser.error('--execute requires --gpu')
        supervise(args.gpu)
    else:
        preflight()


if __name__ == '__main__':
    main()
