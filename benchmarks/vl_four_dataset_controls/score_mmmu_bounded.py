"""Continue native MMMU judging, retaining unresolved extraction as UNKNOWN."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types

import score_native as shared


ROOT = shared.ROOT
DATASET = 'MMMU_DEV_VAL'
DIRECTORY = 'score_local_fused_native_bounded_2048'
SOURCE = Path(__file__).resolve().parent
ORIGINAL_SCORER = SOURCE / 'validation/judge_budget_8192_1794435547736921570/score_native.py.before'
ORIGINAL_SHA = '1e9ba67cc9124f4b081934118aaa927334e0894f7d63337cb5e8ce55a79264f9'
NATIVE_FILES = ('vlmeval/dataset/utils/multiple_choice.py', 'vlmeval/utils/matching_util.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(folder):
    paths = [folder / 'complete.json', folder / 'generation_complete.json']
    old = folder / 'score_local_fused_native'
    paths += [p for p in old.rglob('*') if p.is_file()] if old.exists() else []
    return {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, inode=p.stat().st_ino)
            if p.exists() else None for p in paths}


class UnknownExtraction(Exception):
    pass


@contextmanager
def forbid_random(module):
    previous = module.rd

    def reject(options):
        raise UnknownExtraction('native matcher exhausted all three attempts')

    module.rd = types.SimpleNamespace(choice=reject)
    try:
        yield
    finally:
        module.rd = previous


def frame_for(expected, predictions):
    import pandas as pd
    from vlmeval.dataset.utils.multiple_choice import MMMU_preproc

    frame = pd.DataFrame([{**row['metadata'], 'index': row['index'], 'prediction': prediction['prediction_text']}
                          for row, prediction in zip(expected, predictions)])
    opened = frame['A'].isna().tolist()
    frame = MMMU_preproc(frame)
    frame['GT'] = [answer if answer in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') else 'A' for answer in frame['answer']]
    return frame, opened


def classified(row, item, opened, option, log, origin):
    from vlmeval.dataset.utils.multiple_choice import build_choices

    if option is not None and option not in set(build_choices(item)) | {'Z'}:
        raise ValueError('native extraction returned an invalid option')
    hit = None if option is None else int(option == item['GT'])
    return dict(id=row['id'], index=row['index'], status='UNKNOWN' if hit is None else 'CORRECT' if hit else 'WRONG',
                hit=hit, option=option, ground_truth=item['GT'], open_question=opened,
                log=log, origin=origin, unknown_reason='native_matcher_exhausted' if hit is None else None)


def evaluate(judge, row, item, opened):
    from vlmeval.dataset.utils import multiple_choice as native

    judge.sample_id = row['id']
    try:
        with forbid_random(native):
            result = native.extract_answer_from_item(judge, item, dataset_name=DATASET)
        return classified(row, item, opened, result['opt'], result['log'], 'bounded_native_2048')
    except UnknownExtraction as error:
        return classified(row, item, opened, None, str(error), 'bounded_native_2048')


def import_previous(folder, expected, predictions, frame, opened, kit):
    from vlmeval.dataset.utils import multiple_choice as native

    old = folder / 'score_local_fused_native'
    if not (old / 'records.json').exists():
        return {}, dict(imported=0, source_directory=str(old))
    protocol = shared.read(old / 'protocol.json')
    required = dict(schema='native_local_fused_judge_v1', dataset=DATASET, method=folder.name,
        scorer_sha256=ORIGINAL_SHA, judge_model=shared.MODEL, judge_checkpoint='/data/fanya/models/' + shared.MODEL,
        judge_dtype='float16', judge_moe_strategy='fused', judge_batch_size=1, judge_max_tokens=2048,
        sample_count=len(expected), prediction_protocol='full_response_v1', empty_judge_is_failure=True,
        malformed_judge_is_failure=True, random_choice_fallback_forbidden=True,
        local_judge_differs_from_public_benchmark_judge=True,
        inputs_sha256=shared.fingerprint(dict(rows=expected, predictions=predictions)),
        native_sources={name: sha(kit / name) for name in NATIVE_FILES},
        kit_manifest=shared.read(kit / 'source_manifest.json'),
        native_open_question_rule='MMMU_preproc: A=reference answer, B=Other Answers, native option matching')
    if protocol != required or sha(ORIGINAL_SCORER) != ORIGINAL_SHA:
        raise ValueError('previous score provenance is incompatible; no partial records imported')
    previous = shared.read(old / 'records.json')
    ids = [row['id'] for row in expected]
    if list(previous) != ids[:len(previous)]:
        raise ValueError('previous native records must be the exact evaluation prefix')
    calls = shared.rows(old / 'judge_requests.jsonl') if (old / 'judge_requests.jsonl').exists() else []
    by_id = {}
    for call in calls:
        by_id.setdefault(call.get('id'), []).append(call)
    imported = {}
    for ordinal, row in enumerate(expected[:len(previous)]):
        value = previous[row['id']]
        log = value['log']
        if type(value['hit']) is not int or value['hit'] not in (0, 1) or 'randomly generate' in log:
            raise ValueError('previous score is not a deterministic native result')
        if not log.startswith('Match Log: ') or not log.endswith('. '):
            raise ValueError('previous native log has an unexpected format')
        raw = log[len('Match Log: '):-2]
        prompts = []

        class Replay:
            def generate(self, prompt):
                prompts.append(prompt)
                return raw

        with forbid_random(native):
            replay = native.extract_answer_from_item(Replay(), frame.iloc[ordinal], dataset_name=DATASET)
        if replay['log'] != raw or int(replay['opt'] == frame.iloc[ordinal]['GT']) != value['hit']:
            raise ValueError('previous hit/log does not reproduce under the frozen native matcher')
        if prompts:
            body = dict(model=shared.MODEL, messages=[dict(role='user', content=prompts[0])],
                        max_tokens=2048, temperature=0, stream=False)
            matching = [call for call in by_id.get(row['id'], []) if call.get('valid') is True
                        and call.get('request') == body
                        and call.get('response', {}).get('choices', [{}])[0].get('message', {}).get('content') == raw]
            if len(prompts) != 1 or not matching:
                raise ValueError('previous judge result lacks matching native prompt/2048 response evidence')
        imported[row['id']] = classified(row, frame.iloc[ordinal], opened[ordinal], replay['opt'], raw,
                                        'validated_prior_native_2048')
    return imported, dict(imported=len(imported), source_directory=str(old), protocol_sha256=sha(old / 'protocol.json'),
        records_sha256=sha(old / 'records.json'), judge_requests_sha256=sha(old / 'judge_requests.jsonl') if calls else None,
        original_scorer_sha256=ORIGINAL_SHA, original_scorer_snapshot=str(ORIGINAL_SCORER),
        current_shared_scorer_sha256=sha(shared.__file__), source_versions_identical=False,
        compatibility_basis='Original frozen native source, exact inputs, protocol and per-record native log replay; judge records require exact prompt/model/2048 request evidence. Current shared source adds optional cap/directory controls; it is not claimed byte-identical.')


def summary(records, total, protocol):
    values = list(records.values())
    counts = {status: sum(v['status'] == status for v in values) for status in ('CORRECT', 'WRONG', 'UNKNOWN')}
    correct, wrong, unknown = (counts[x] for x in ('CORRECT', 'WRONG', 'UNKNOWN'))
    pending = total - len(values)
    return dict(dataset=DATASET, method=protocol['method'], sample_count=total, processed=len(values),
        traversal_complete=len(values) == total, known_correct=correct, known_wrong=wrong, unknown=unknown,
        pending=pending, known_count=correct + wrong, score_coverage_pct=100 * (correct + wrong) / total,
        known_accuracy_pct=100 * correct / (correct + wrong) if correct + wrong else None,
        full_denominator_accuracy_lower_pct=100 * correct / total,
        full_denominator_accuracy_upper_pct=100 * (correct + unknown + pending) / total,
        strict_primary_complete=False, primary_cell_completion_written=False, protocol=protocol)


def validate_records(records, expected, frame):
    ids = [row['id'] for row in expected]
    if list(records) != ids[:len(records)]:
        raise ValueError('bounded records are not an exact evaluation prefix')
    for ordinal, value in enumerate(records.values()):
        hit = value['hit']
        if value['id'] != ids[ordinal] or value['ground_truth'] != frame.iloc[ordinal]['GT']:
            raise ValueError('bounded record identity/ground truth changed')
        if value['status'] == 'UNKNOWN':
            if hit is not None or value['option'] is not None or value['unknown_reason'] != 'native_matcher_exhausted':
                raise ValueError('unknown must not have a numeric score or guessed option')
        elif type(hit) is not int or hit not in (0, 1) or hit != int(value['option'] == value['ground_truth']):
            raise ValueError('known record score is inconsistent')
        elif value['status'] != ('CORRECT' if hit else 'WRONG'):
            raise ValueError('known record status is inconsistent')


def score(method, port):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.pop('VERBOSE', None)
    folder = ROOT / 'cells' / DATASET / method
    before = identities(folder)
    if not shared.read(folder / 'generation_complete.json').get('complete'):
        raise ValueError('generation is incomplete')
    expected = shared.rows(ROOT / 'prepare' / DATASET / 'evaluation.jsonl')
    predictions = shared.rows(folder / 'requests.jsonl')
    shared.check_coverage(expected, predictions)
    if len(expected) != 420:
        raise ValueError('MMMU must retain the full frozen 420 evaluation rows')
    destination = folder / DIRECTORY
    work = destination / 'work'
    work.mkdir(parents=True, exist_ok=True)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[name] = str(work)
    tempfile.tempdir = str(work)
    if not (ROOT / 'score_sources' / DATASET / 'source_manifest.json').exists():
        raise ValueError('existing frozen native source is required')
    kit = shared.bootstrap(DATASET)
    frame, opened = frame_for(expected, predictions)
    imported, provenance = import_previous(folder, expected, predictions, frame, opened, kit)
    protocol = dict(schema='native_local_fused_bounded_2048_v1', dataset=DATASET, method=method,
        sample_count=420, judge_model=shared.MODEL, judge_dtype='float16', judge_moe_strategy='fused',
        judge_batch_size=1, judge_max_tokens=2048, native_extraction_attempts=3, score_directory=DIRECTORY,
        prediction_protocol='full_response_v1', unknown_rule='Only exhausted native matcher becomes UNKNOWN; no random draw and no numeric score.',
        infrastructure_failure_rule='Fail closed; API/empty response exceptions do not become model errors or UNKNOWN.',
        strict_primary_complete=False, main_cell_completion_written=False,
        original_prompt_and_prediction_unchanged=True, random_fallback_forbidden=True,
        native_open_question_rule='MMMU_preproc: missing A becomes reference answer and B=Other Answers; non-letter GT becomes A',
        inputs_sha256=shared.fingerprint(dict(rows=expected, predictions=predictions)),
        local_judge_differs_from_public_benchmark_judge=True,
        scorer_sources={name: sha(SOURCE / name) for name in ('score_mmmu_bounded.py', 'score_native.py')},
        native_sources={name: sha(kit / name) for name in NATIVE_FILES},
        kit_source_manifest_sha256=sha(kit / 'source_manifest.json'), import_provenance=provenance)
    protocol_path = destination / 'protocol.json'
    if protocol_path.exists() and shared.read(protocol_path) != protocol:
        raise ValueError('bounded score protocol changed; preserve this directory')
    shared.save(protocol_path, protocol)
    checkpoint = destination / 'records.json'
    records = shared.read(checkpoint) if checkpoint.exists() else imported
    validate_records(records, expected, frame)
    if any(records.get(key) != value for key, value in imported.items()):
        raise ValueError('validated imported prefix changed')
    shared.save(checkpoint, records)
    judge = shared.StrictJudge(port, DATASET, destination, judge_max_tokens=2048)
    try:
        for ordinal, row in enumerate(expected):
            if row['id'] in records:
                continue
            records[row['id']] = evaluate(judge, row, frame.iloc[ordinal], opened[ordinal])
            shared.save(checkpoint, records)
            shared.save(destination / 'progress.json', summary(records, len(expected), protocol))
            print('BOUNDED_SCORE', method, len(records), len(expected), records[row['id']]['status'], flush=True)
        result = summary(records, len(expected), protocol)
        validate_records(records, expected, frame)
        shared.save(destination / 'score.json', result)
        if before != identities(folder):
            raise ValueError('original score or main completion files changed during bounded scoring')
        shared.save(destination / 'preservation.json', dict(unchanged=True, before=before, after=identities(folder)))
        shared.save(destination / 'traversal_complete.json', dict(complete=True, traversal_complete=True,
            strict_primary_complete=False, sample_count=420, unknown=result['unknown'],
            score_file=str(destination / 'score.json'), protocol_sha256=shared.fingerprint(protocol),
            records_sha256=shared.fingerprint(records), completed_utc=datetime.now(timezone.utc).isoformat()))
        return result
    except Exception as error:
        shared.save(destination / 'failure.json', dict(error=repr(error), processed=len(records),
            strict_primary_complete=False, original_files_unchanged=before == identities(folder)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=('fused', 'tablemoe', 'modes', 'schedule', 'table'))
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    result = score(args.method, args.port)
    print(json.dumps({key: value for key, value in result.items() if key != 'protocol'}, indent=2))
