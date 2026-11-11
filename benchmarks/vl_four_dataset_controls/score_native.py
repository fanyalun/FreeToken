import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import types
import urllib.request


ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen3-VL-30B-A3B-Instruct'
KIT_PYTHON = '/home/fanya/miniforge/envs/tablemoe/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def bootstrap(name):
    original = Path('/home/fanya/VLMEvalKit' if name == 'BabyVision' else '/home/fanya/tablemoe/third_party/VLMEvalKit')
    kit = ROOT/'score_sources'/name
    manifest_path = kit/'source_manifest.json'
    if not manifest_path.exists():
        sources = {}
        for source in sorted((original/'vlmeval').rglob('*.py')):
            relative = source.relative_to(original)
            destination = kit/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
            sources[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
        if name=='MMLongBench_DOC':
            for relative in ['table_moe/__init__.py','table_moe/utils/__init__.py','table_moe/utils/mmlongbench_doc.py']:
                source=Path('/home/fanya/tablemoe')/relative
                destination=kit/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                if source.exists():
                    shutil.copyfile(source,destination)
                else:
                    destination.write_text('')
                sources[relative]=hashlib.sha256(destination.read_bytes()).hexdigest()
        revision=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(original),'status','--porcelain'],text=True)
        save(manifest_path,dict(original_root=str(original),git_revision=revision,dirty_status=dirty,files=sources))
    manifest=read(manifest_path)
    for relative,expected in manifest['files'].items():
        if hashlib.sha256((kit/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen native scorer source changed: '+relative)
    os.environ['LMUData'] = '/data/fanya/datasets/vlmevalkit' if name == 'BabyVision' else '/home/fanya/tablemoe/LMUData'
    sys.path[:0] = [str(kit), '/home/fanya/tablemoe']
    for package, directory in [('vlmeval', kit/'vlmeval'), ('vlmeval.dataset', kit/'vlmeval/dataset')]:
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    if name=='MMLongBench_DOC':
        for package,directory in [('table_moe',kit/'table_moe'),('table_moe.utils',kit/'table_moe/utils')]:
            module=types.ModuleType(package)
            module.__path__=[str(directory)]
            sys.modules[package]=module
    return kit


def score_options(dataset, judge_max_tokens=None, score_directory=None):
    maximum = {'BabyVision': 16, 'MMLongBench_DOC': 128, 'MMMU_DEV_VAL': 2048}[dataset]
    if judge_max_tokens is not None:
        if type(judge_max_tokens) is not int or judge_max_tokens < 1:
            raise ValueError('judge_max_tokens must be a positive integer')
        maximum = judge_max_tokens
    directory = 'score_local_fused_native' if score_directory is None else score_directory
    if not isinstance(directory, str) or not re.fullmatch(r'score_[A-Za-z0-9_-]+', directory):
        raise ValueError('score_directory must be a single score_ directory name')
    return maximum, directory


class StrictJudge:
    def __init__(self, port, dataset, folder, judge_max_tokens=None):
        self.port, self.dataset, self.folder = port, dataset, folder
        self.max_tokens, _ = score_options(dataset, judge_max_tokens)
        self.sample_id = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def generate(self, prompt, **kwargs):
        if not isinstance(prompt, str):
            raise TypeError('native scorer supplied a nontext prompt')
        body = dict(model=MODEL, messages=[dict(role='user', content=prompt)],
                    max_tokens=self.max_tokens, temperature=kwargs.get('temperature', 0), stream=False)
        last = None
        for attempt in range(3):
            started = time.monotonic()
            record = dict(id=self.sample_id, attempt=attempt, request=body,
                          utc=datetime.now(timezone.utc).isoformat())
            try:
                request = urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions',
                    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                with self.opener.open(request, timeout=600) as response:
                    payload = json.load(response)
                record['response'] = payload
                text = payload['choices'][0]['message'].get('content') or ''
                if not text.strip() or 'Failed to obtain answer via API' in text:
                    raise ValueError('empty or failed judge response')
                if self.dataset == 'BabyVision':
                    values = set(re.findall(r'\b(true|false)\b', text.lower()))
                    if len(values) != 1:
                        raise ValueError('BabyVision judge response has no unambiguous boolean')
                elif self.dataset == 'MMLongBench_DOC':
                    if 'Extracted answer:' not in text or 'Answer format:' not in text:
                        raise ValueError('Doc judge response lacks native answer markers')
                    extracted = text.split('Answer format:')[0].split('Extracted answer:')[1].strip()
                    if not extracted:
                        raise ValueError('Doc judge extracted an empty answer')
                record['valid'] = True
                return text
            except Exception as error:
                last = error
                record.update(valid=False, error=repr(error))
                if attempt < 2:
                    time.sleep(1)
            finally:
                record['wall_s'] = time.monotonic() - started
                with (self.folder/'judge_requests.jsonl').open('a') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        raise RuntimeError(f'judge failed after three attempts for {self.sample_id}: {last}')


def check_coverage(expected, predictions):
    ids = [row['id'] for row in expected]
    if len(set(ids)) != len(ids) or [row['id'] for row in predictions] != ids:
        raise ValueError('prediction coverage/order differs from the frozen evaluation manifest')
    if any(not row.get('natural_eos') for row in predictions):
        raise ValueError('native accuracy requires natural generation predictions')
    if any(not str(row.get('prediction_text', row.get('text', ''))).strip() for row in predictions):
        raise ValueError('empty model prediction; preserve failure instead of completing the cell')


def score(name, method, port, *, judge_max_tokens=None, score_directory=None):
    import pandas as pd

    maximum, directory = score_options(name, judge_max_tokens, score_directory)
    folder = ROOT/'cells'/name/method
    completion = read(folder/'generation_complete.json')
    if not completion.get('complete'):
        raise ValueError('generation is incomplete')
    expected = rows(ROOT/'prepare'/name/'evaluation.jsonl')
    predictions = rows(folder/'requests.jsonl')
    check_coverage(expected, predictions)
    destination = folder/directory
    destination.mkdir(exist_ok=True)
    kit = bootstrap(name)
    from vlmeval.smp import dump, load
    source_files = ['vlmeval/dataset/utils/multiple_choice.py'] if name == 'MMMU_DEV_VAL' else [
        'vlmeval/dataset/babyvision.py' if name == 'BabyVision' else 'vlmeval/dataset/mmlongbenchdoc.py']
    source_files += ['vlmeval/utils/matching_util.py'] if name == 'MMMU_DEV_VAL' else []
    protocol = dict(schema='native_local_fused_judge_v2', dataset=name, method=method,
        scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        judge_model=MODEL, judge_checkpoint='/data/fanya/models/'+MODEL, judge_dtype='float16',
        judge_moe_strategy='fused', judge_batch_size=1, local_judge_differs_from_public_benchmark_judge=True,
        sample_count=len(expected), prediction_protocol='full_response_v1',
        judge_max_tokens=maximum, score_directory=directory, score_output_path=str(destination),
        empty_judge_is_failure=True, malformed_judge_is_failure=True, random_choice_fallback_forbidden=True,
        inputs_sha256=fingerprint(dict(rows=expected, predictions=predictions)),
        kit_manifest=read(kit/'source_manifest.json'),
        native_sources={path:hashlib.sha256((kit/path).read_bytes()).hexdigest()
                        for path in source_files if (kit/path).exists()})
    if name == 'MMMU_DEV_VAL':
        protocol['native_open_question_rule'] = 'MMMU_preproc: A=reference answer, B=Other Answers, native option matching'
    if (destination/'protocol.json').exists() and read(destination/'protocol.json') != protocol:
        raise ValueError('score inputs/protocol changed; use a fresh score destination')
    save(destination/'protocol.json', protocol)
    if (destination/'complete.json').exists() and read(destination/'complete.json').get('complete'):
        return read(destination/'score.json')
    records = [{**row['metadata'], 'index': row['index'], 'prediction': prediction['prediction_text']}
               for row, prediction in zip(expected, predictions)]
    frame = pd.DataFrame(records)
    dump(frame, str(destination/'predictions.xlsx'))
    judge = StrictJudge(port, name, destination, judge_max_tokens=maximum)
    checkpoint = destination/'records.json'
    done = read(checkpoint) if checkpoint.exists() else {}
    if set(done) - {row['id'] for row in expected}:
        raise ValueError('unexpected native scoring checkpoint identities')
    if name == 'MMMU_DEV_VAL':
        from vlmeval.dataset.utils.multiple_choice import MMMU_preproc, eval_vanilla, report_acc
        frame = MMMU_preproc(frame)
        frame['GT'] = [value if value in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') else 'A' for value in frame['answer']]
    elif name == 'BabyVision':
        from vlmeval.dataset.babyvision import BabyVision
    else:
        from vlmeval.dataset.mmlongbenchdoc import MMLongBench_auxeval, MMLongBench_acc
    for index, row in enumerate(expected):
        if row['id'] in done:
            continue
        judge.sample_id = row['id']
        line = frame.iloc[index]
        if name == 'MMMU_DEV_VAL':
            value = eval_vanilla(judge, line, dataset_name=name)
            if 'randomly generate' in value['log']:
                raise RuntimeError('native MMMU matcher exhausted retries; random guess rejected')
        elif name == 'BabyVision':
            value = BabyVision._evaluate_single(judge, line)
        else:
            value = MMLongBench_auxeval(judge, line)
            if not value['res'].strip() or not value['pred'].strip():
                raise RuntimeError('native Doc extraction returned empty result')
        done[row['id']] = value
        save(checkpoint, done)
        save(destination/'progress.json', dict(completed=len(done),total=len(expected)))
        print('SCORE',name,method,len(done),len(expected),flush=True)
    values = [done[row['id']] for row in expected]
    if name == 'MMMU_DEV_VAL':
        frame['hit'],frame['log'] = [v['hit'] for v in values],[v['log'] for v in values]
        frame.pop('GT')
        native = json.loads(report_acc(frame).to_json(orient='records'))
        accuracy = float(frame['hit'].mean()*100)
    elif name == 'BabyVision':
        frame['LLMJudgeResult'] = [v['llm_judge_result'] for v in values]
        frame['ExtractedAnswer'] = [v['extracted_answer'] for v in values]
        frame['JudgeResponse'] = [v['judge_response'] for v in values]
        frame['JudgePrompt'] = [v['judge_prompt'] for v in values]
        native = BabyVision.compute_metrics(frame)
        accuracy = native['Overall']
    else:
        for key in ('res','log','pred'):
            frame[key] = [v[key] for v in values]
        temporary = destination/'native_results.xlsx'
        dump(frame,str(temporary))
        native = json.loads(MMLongBench_acc(str(temporary)).to_json(orient='records'))
        accuracy = next(r['avg_score']*100 for r in native if r['category']=='overall_acc')
        frame = load(str(temporary))
    if not math.isfinite(accuracy):
        raise ValueError('nonfinite native accuracy')
    dump(frame,str(destination/'native_results.xlsx'))
    summary = dict(dataset=name,method=method,accuracy_pct=accuracy,scoring_denominator=len(expected),
        prediction_rows=len(predictions),native_results=native,protocol=protocol,
        length_limited_predictions=sum(p['finish_reason']=='length' for p in predictions))
    save(destination/'score.json',summary)
    save(destination/'complete.json',dict(complete=True,samples=len(expected),score_file=str(destination/'score.json')))
    save(folder/'complete.json',dict(complete=True,generation_complete=True,scoring_complete=True,
        accuracy_pct=accuracy,scoring_denominator=len(expected),score_file=str(destination/'score.json')))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',choices=['MMMU_DEV_VAL','BabyVision','MMLongBench_DOC'])
    parser.add_argument('method')
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--judge-max-tokens',type=int)
    parser.add_argument('--score-directory')
    args=parser.parse_args()
    score(args.dataset,args.method,args.port,judge_max_tokens=args.judge_max_tokens,
          score_directory=args.score_directory)
