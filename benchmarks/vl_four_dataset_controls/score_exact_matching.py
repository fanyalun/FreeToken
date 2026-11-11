"""Score complete MMMU predictions with frozen native CPU exact_matching only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from score_native import ROOT, bootstrap, check_coverage, fingerprint, read, rows, save


DATASET = 'MMMU_DEV_VAL'
METHODS = ('fused', 'tablemoe', 'modes', 'schedule', 'table')


def file_identity(path):
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def preserved_files(folder):
    paths = [folder / name for name in ('complete.json', 'generation_complete.json', 'score_complete.json')]
    judge = folder / 'score_local_fused_native'
    if judge.is_dir():
        paths += [path for path in judge.rglob('*') if path.is_file()]
    return {str(path): file_identity(path) for path in paths}


def native_scores(expected, predictions):
    import pandas as pd
    from vlmeval.dataset.utils.multiple_choice import MMMU_preproc, build_choices, extract_answer_from_item, eval_vanilla

    frame = pd.DataFrame([{**row['metadata'], 'index': row['index'], 'prediction': prediction['prediction_text']}
                          for row, prediction in zip(expected, predictions)])
    open_questions = frame['A'].isna().tolist()
    frame = MMMU_preproc(frame)
    frame['GT'] = [answer if answer in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') else 'A' for answer in frame['answer']]
    records = []
    for ordinal, row in enumerate(expected):
        item = frame.iloc[ordinal]
        extracted = extract_answer_from_item(None, item, dataset_name=DATASET)
        result = eval_vanilla(None, item, dataset_name=DATASET)
        option = extracted['opt']
        if option not in set(build_choices(item)) | {'Z'} or 'randomly generate' in result['log']:
            raise ValueError('native exact_matching violated the deterministic option contract')
        if result['hit'] != int(option == item['GT']):
            raise ValueError('native extraction and evaluation disagree')
        records.append(dict(id=row['id'], index=row['index'], ordinal=ordinal, open_question=open_questions[ordinal],
            option=option, ground_truth=item['GT'], hit=result['hit'], log=result['log'],
            prefetch_unmatched='exact_matching' in extracted['log']))
    frame['hit'] = [record['hit'] for record in records]
    frame['log'] = [record['log'] for record in records]
    frame.pop('GT')
    return frame, records


def failure_evidence(folder, records, predictions):
    judge = folder / 'score_local_fused_native'
    previous = read(judge / 'records.json') if (judge / 'records.json').is_file() else {}
    calls = rows(judge / 'judge_requests.jsonl') if (judge / 'judge_requests.jsonl').is_file() else []
    by_id = {record['id']: record for record in records}
    prediction_by_id = {prediction['id']: prediction for prediction in predictions}
    incomplete = {}
    for call in calls:
        sample_id = call.get('id')
        if sample_id in previous or sample_id not in by_id:
            continue
        response = call.get('response', {})
        choices = response.get('choices', [])
        choice = choices[0] if choices else {}
        content = choice.get('message', {}).get('content') or ''
        incomplete.setdefault(sample_id, []).append(dict(request_max_tokens=call['request'].get('max_tokens'),
            request_utc=call.get('utc'), usage=response.get('usage'), finish_reason=choice.get('finish_reason'),
            response_content_characters=len(content), response_content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            transport_valid=call.get('valid'), recorded_error=call.get('error')))
    return dict(local_judge_directory=str(judge), local_judge_recorded_samples=len(previous),
        local_judge_complete=(judge / 'complete.json').is_file() and read(judge / 'complete.json').get('complete', False),
        partial_judge_records_reused_for_exact_score=False, original_failure_remains_unresolved=bool(incomplete),
        incomplete_judge_cases={sample_id: dict(judge_attempts=calls,
            exact_matching_result=by_id[sample_id], generation_usage=prediction_by_id[sample_id]['usage'],
            generation_finish_reason=prediction_by_id[sample_id]['finish_reason'])
            for sample_id, calls in incomplete.items()},
        distinction='Exact unmatched Z is a result of the separate CPU protocol; it does not turn judge failures into local-judge model zeros')


def score(method):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.pop('VERBOSE', None)
    folder = ROOT / 'cells' / DATASET / method
    before = preserved_files(folder)
    generation = read(folder / 'generation_complete.json')
    if not generation.get('complete'):
        raise ValueError('generation must be complete before exact_matching scoring')
    expected = rows(ROOT / 'prepare' / DATASET / 'evaluation.jsonl')
    predictions = rows(folder / 'requests.jsonl')
    check_coverage(expected, predictions)
    if len(expected) != 420:
        raise ValueError('the frozen MMMU evaluation must contain all 420 samples')
    destination = folder / 'score_exact_matching'
    work = destination / 'work'
    work.mkdir(parents=True, exist_ok=True)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[name] = str(work)
    tempfile.tempdir = str(work)
    kit = ROOT / 'score_sources' / DATASET
    if not (kit / 'source_manifest.json').is_file():
        raise ValueError('reuse the existing frozen native toolkit; no implicit source replacement')
    kit = bootstrap(DATASET)
    from vlmeval.dataset.utils.multiple_choice import report_acc
    from vlmeval.smp import dump

    source_manifest = read(kit / 'source_manifest.json')
    native_files = ('vlmeval/dataset/image_mcq.py', 'vlmeval/dataset/utils/multiple_choice.py',
                    'vlmeval/utils/matching_util.py')
    protocol = dict(schema='native_exact_matching_cpu_v1', dataset=DATASET, method=method,
        scoring_policy='exact_matching', model=None, judge_calls=0, gpu_started=False,
        role='independent_auxiliary_pending_protocol_amendment', main_cell_completion_written=False,
        local_fused_judge_completion_claimed=False, sample_count=420,
        prediction_protocol='full_response_v1', original_prompt_and_prediction_unchanged=True,
        inputs_sha256=fingerprint(dict(rows=expected, predictions=predictions)),
        native_open_question_rule='MMMU_preproc: when A is missing, A=reference answer and B=Other Answers; non-letter GT becomes A',
        unmatched_rule='native can_infer failure with model=None returns Z; Z is scored against native GT',
        no_random_fallback=True, no_semantic_judge=True, no_numeric_tolerance_added=True,
        no_final_answer_rewriting=True, environment={'VERBOSE': None, 'CUDA_VISIBLE_DEVICES': ''},
        limitations=['Native exact_matching is option inference, not arbitrary answer-string equality.',
            'Long or ambiguous reasoning can remain unmatched even when it contains a correct answer.',
            'Open questions use the native reference-versus-other conversion and no semantic or numeric equivalence judge.',
            'Results must not be mixed with partial local fused judge scores or presented as local fused judge completion.'],
        scorer_sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ('score_exact_matching.py', 'score_native.py')},
        kit_root=str(kit), kit_source_manifest=str(kit / 'source_manifest.json'),
        kit_source_manifest_sha256=hashlib.sha256((kit / 'source_manifest.json').read_bytes()).hexdigest(),
        original_kit_revision=source_manifest['git_revision'], original_kit_was_dirty=bool(source_manifest['dirty_status']),
        native_sources={name: hashlib.sha256((kit / name).read_bytes()).hexdigest() for name in native_files})
    destination.mkdir(exist_ok=True)
    if (destination / 'protocol.json').is_file() and read(destination / 'protocol.json') != protocol:
        raise ValueError('frozen exact_matching protocol changed; preserve this output and use a new destination')
    save(destination / 'protocol.json', protocol)
    if (destination / 'complete.json').is_file() and read(destination / 'complete.json').get('complete'):
        result = read(destination / 'score.json')
        marker = read(destination / 'complete.json')
        records = read(destination / 'records.json')
        if (result.get('sample_count') != 420 or [record['id'] for record in records] != [row['id'] for row in expected]
                or marker.get('protocol_sha256') != fingerprint(protocol)
                or marker.get('records_sha256') != fingerprint(records)
                or result.get('protocol') != protocol
                or result.get('correct') != sum(record['hit'] for record in records)):
            raise ValueError('completed auxiliary score has incomplete coverage')
        return result
    frame, records = native_scores(expected, predictions)
    if [record['id'] for record in records] != [row['id'] for row in expected]:
        raise ValueError('native score coverage changed')
    native = json.loads(report_acc(frame).to_json(orient='records'))
    matched = sum(record['option'] != 'Z' for record in records)
    result = dict(complete=True, dataset=DATASET, method=method, sample_count=len(records),
        accuracy_pct=sum(record['hit'] for record in records) * 100 / len(records),
        correct=sum(record['hit'] for record in records), option_matches=matched, option_Z=len(records) - matched,
        prefetch_unmatched=sum(record['prefetch_unmatched'] for record in records),
        open_questions=sum(record['open_question'] for record in records),
        multiple_choice_questions=sum(not record['open_question'] for record in records),
        native_results=native, protocol=protocol, auxiliary_only=True,
        local_fused_judge_completed=False, main_cell_completion_written=False)
    save(destination / 'records.json', records)
    dump(frame, str(destination / 'native_results.xlsx'))
    save(destination / 'local_judge_failure_evidence.json', failure_evidence(folder, records, predictions))
    save(destination / 'score.json', result)
    after = preserved_files(folder)
    if before != after:
        raise ValueError('primary score files or main completion markers changed during auxiliary scoring')
    save(destination / 'preservation.json', dict(complete=True, unchanged=True, before=before, after=after))
    save(destination / 'complete.json', dict(complete=True, sample_count=420,
        scoring_policy='exact_matching', auxiliary_only=True, local_fused_judge_completed=False,
        protocol_sha256=fingerprint(protocol), records_sha256=fingerprint(records),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=[DATASET])
    parser.add_argument('method', choices=METHODS)
    args = parser.parse_args()
    result = score(args.method)
    print(json.dumps({key: result[key] for key in ('dataset', 'method', 'sample_count', 'accuracy_pct',
        'correct', 'option_matches', 'option_Z', 'prefetch_unmatched', 'open_questions', 'auxiliary_only')}, indent=2))


if __name__ == '__main__':
    main()
