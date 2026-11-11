"""Continue frozen Doc512 extraction without assigning scores to format failures."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import score_native as shared


ROOT = shared.ROOT
SOURCE = Path(__file__).resolve().parent
DATASET = 'MMLongBench_DOC'
DIRECTORY = 'score_local_fused_native_bounded_512'
PREVIOUS_DIRECTORY = 'score_local_fused_native_512'
SHARED_SHA = '11d91047ef7acafcafc87daa389a914058aa1c32e9d3d3d72127e35c1ea802c2'
NATIVE_FILE = 'vlmeval/dataset/mmlongbenchdoc.py'
FORMAT_ERRORS = ('Doc judge response lacks native answer markers', 'Doc judge extracted an empty answer')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(folder):
    paths = [folder / 'complete.json', folder / 'generation_complete.json']
    for directory in ('score_local_fused_native', PREVIOUS_DIRECTORY):
        old = folder / directory
        paths += [p for p in old.rglob('*') if p.is_file()] if old.exists() else []
    return {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, inode=p.stat().st_ino)
            if p.exists() else None for p in paths}


def body_for(prompt, temperature=0):
    return dict(model=shared.MODEL, messages=[dict(role='user', content=prompt)],
                max_tokens=512, temperature=temperature, stream=False)


def format_error(text):
    if 'Extracted answer:' not in text or 'Answer format:' not in text:
        return FORMAT_ERRORS[0]
    if not text.split('Answer format:')[0].split('Extracted answer:')[1].strip():
        return FORMAT_ERRORS[1]
    return None


def validate_format_attempts(calls, sample_id, body):
    if len(calls) != 3 or [call.get('attempt') for call in calls] != [0, 1, 2]:
        raise ValueError('UNKNOWN requires exactly this invocation\'s three ordered transport attempts')
    for call in calls:
        payload = call.get('response')
        if call.get('id') != sample_id or call.get('request') != body or call.get('valid') is not False:
            raise ValueError('UNKNOWN journal identity/request/validity mismatch')
        if not isinstance(payload, dict) or payload.get('error') or len(payload.get('choices', [])) != 1:
            raise ValueError('UNKNOWN cannot contain API or missing HTTP response evidence')
        choice = payload['choices'][0]
        text = choice.get('message', {}).get('content')
        tokens = payload.get('usage', {}).get('completion_tokens')
        if (not isinstance(text, str) or not text.strip() or 'Failed to obtain answer via API' in text
                or choice.get('finish_reason') not in ('stop', 'length')
                or type(tokens) is not int or not 0 < tokens <= 512):
            raise ValueError('UNKNOWN cannot contain empty/API/abnormal judge responses')
        reason = format_error(text)
        if reason is None or call.get('error') != repr(ValueError(reason)):
            raise ValueError('UNKNOWN accepts only proven pure native format extraction errors')
    return dict(attempts=3, all_nonempty_http_responses=True, pure_format_failures=True,
                journal_sha256=shared.fingerprint(calls), request_sha256=shared.fingerprint(body))


class UnknownDocExtraction(Exception):
    def __init__(self, evidence):
        super().__init__('all three 512-token transport attempts returned nonempty format-invalid responses')
        self.evidence = evidence


class BoundedJudge(shared.StrictJudge):
    def generate(self, prompt, **kwargs):
        journal = self.folder / 'judge_requests.jsonl'
        offset = journal.stat().st_size if journal.exists() else 0
        try:
            return super().generate(prompt, **kwargs)
        except RuntimeError:
            if not journal.exists():
                raise
            with journal.open('rb') as handle:
                handle.seek(offset)
                calls = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
            try:
                evidence = validate_format_attempts(calls, self.sample_id, body_for(prompt, kwargs.get('temperature', 0)))
            except (ValueError, KeyError, TypeError, AttributeError):
                raise RuntimeError('Doc judge failed with non-format or mixed transport evidence; not UNKNOWN')
            evidence.update(journal=str(journal), byte_start=offset, byte_end=journal.stat().st_size)
            raise UnknownDocExtraction(evidence)


def frame_for(expected, predictions):
    import pandas as pd
    return pd.DataFrame([{**row['metadata'], 'index': row['index'], 'prediction': prediction['prediction_text']}
                         for row, prediction in zip(expected, predictions)])


def known_record(row, item, extraction, origin):
    from vlmeval.dataset.mmlongbenchdoc import eval_score
    if not extraction['res'].strip() or not extraction['pred'].strip() or format_error(extraction['res']):
        raise ValueError('known Doc extraction lacks valid native markers or answer')
    error = None
    try:
        score = eval_score(item['answer'], extraction['pred'], item['answer_format'])
    except Exception as caught:
        score = 0.0
        error = repr(caught)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('native Doc score must remain in [0,1]')
    return dict(id=row['id'], index=row['index'], status='KNOWN', score=float(score), extraction=extraction,
                origin=origin, native_eval_exception_zero=error, unknown_reason=None)


def evaluate(judge, row, item):
    from vlmeval.dataset.mmlongbenchdoc import MMLongBench_auxeval
    judge.sample_id = row['id']
    try:
        return known_record(row, item, MMLongBench_auxeval(judge, item), 'bounded_native_512')
    except UnknownDocExtraction as error:
        return dict(id=row['id'], index=row['index'], status='UNKNOWN', score=None, extraction=None,
                    origin='bounded_native_512', unknown_reason='three_nonempty_pure_format_failures',
                    evidence=error.evidence)


def import_previous(folder, expected, predictions, frame, kit):
    from vlmeval.dataset.mmlongbenchdoc import MMLongBench_auxeval
    old = folder / PREVIOUS_DIRECTORY
    if not (old / 'records.json').exists():
        return {}, dict(imported=0, source_directory=str(old))
    protocol = shared.read(old / 'protocol.json')
    expected_protocol = dict(schema='native_local_fused_judge_v2', dataset=DATASET, method=folder.name,
        scorer_sha256=SHARED_SHA, judge_model=shared.MODEL, judge_checkpoint='/data/fanya/models/' + shared.MODEL,
        judge_dtype='float16', judge_moe_strategy='fused', judge_batch_size=1, judge_max_tokens=512,
        score_directory=PREVIOUS_DIRECTORY, score_output_path=str(ROOT / 'cells' / DATASET / folder.name / PREVIOUS_DIRECTORY),
        sample_count=len(expected), prediction_protocol='full_response_v1', empty_judge_is_failure=True,
        malformed_judge_is_failure=True, random_choice_fallback_forbidden=True,
        local_judge_differs_from_public_benchmark_judge=True,
        inputs_sha256=shared.fingerprint(dict(rows=expected, predictions=predictions)),
        native_sources={NATIVE_FILE: sha(kit / NATIVE_FILE)}, kit_manifest=shared.read(kit / 'source_manifest.json'))
    if protocol != expected_protocol or sha(shared.__file__) != SHARED_SHA:
        raise ValueError('previous Doc512 source/protocol is incompatible')
    previous = shared.read(old / 'records.json')
    if list(previous) != [row['id'] for row in expected[:len(previous)]]:
        raise ValueError('previous native records are not the exact frozen prefix')
    calls = shared.rows(old / 'judge_requests.jsonl')
    by_id = {}
    for call in calls:
        by_id.setdefault(call.get('id'), []).append(call)
    imported = {}
    for ordinal, row in enumerate(expected[:len(previous)]):
        value = previous[row['id']]
        requested = []
        class Replay:
            def generate(self, prompt, **kwargs):
                requested.append(body_for(prompt, kwargs.get('temperature', 0)))
                return value['res']
        replay = MMLongBench_auxeval(Replay(), frame.iloc[ordinal])
        if replay != value or len(requested) != 1:
            raise ValueError('previous extraction does not reproduce under native Doc auxeval')
        matching = [call for call in by_id.get(row['id'], []) if call.get('valid') is True
                    and call.get('request') == requested[0]
                    and call.get('response', {}).get('choices', [{}])[0].get('message', {}).get('content') == value['res']]
        if not matching:
            raise ValueError('previous extraction lacks matching original prompt/model/512 request and response')
        imported[row['id']] = known_record(row, frame.iloc[ordinal], replay, 'validated_prior_native_512')
    return imported, dict(imported=len(imported), source_directory=str(old), original_scorer_sha256=SHARED_SHA,
        protocol_sha256=sha(old / 'protocol.json'), records_sha256=sha(old / 'records.json'),
        judge_requests_sha256=sha(old / 'judge_requests.jsonl'), native_auxeval_replayed=True,
        every_imported_request_response_verified=True, continuous_native_scores_recomputed=True)


def summary(records, total, protocol):
    known = [value for value in records.values() if value['status'] == 'KNOWN']
    unknown = sum(value['status'] == 'UNKNOWN' for value in records.values())
    score_sum = sum(value['score'] for value in known)
    pending = total - len(records)
    return dict(dataset=DATASET, method=protocol['method'], sample_count=total, processed=len(records),
        traversal_complete=len(records) == total, known_count=len(known), unknown=unknown, pending=pending,
        known_score_sum=score_sum, score_coverage_pct=100 * len(known) / total,
        known_accuracy_pct=100 * score_sum / len(known) if known else None,
        full_denominator_accuracy_lower_pct=100 * score_sum / total,
        full_denominator_accuracy_upper_pct=100 * (score_sum + unknown + pending) / total,
        overall_f1=None, f1_status='not_reported_for_bounded_continuation', strict_primary_complete=False,
        primary_cell_completion_written=False, protocol=protocol)


def validate_records(records, expected):
    if list(records) != [row['id'] for row in expected[:len(records)]]:
        raise ValueError('bounded Doc checkpoint is not a frozen prefix')
    for key, value in records.items():
        if value['id'] != key:
            raise ValueError('bounded record ID mismatch')
        if value['status'] == 'UNKNOWN':
            if value['score'] is not None or value['extraction'] is not None or not value['evidence']['pure_format_failures']:
                raise ValueError('UNKNOWN cannot contain a numeric score or extracted answer')
        elif value['status'] != 'KNOWN' or not math.isfinite(value['score']) or not 0 <= value['score'] <= 1:
            raise ValueError('known Doc score must be continuous [0,1]')


def score(method, port):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    folder = ROOT / 'cells' / DATASET / method
    before = identities(folder)
    if not shared.read(folder / 'generation_complete.json').get('complete'):
        raise ValueError('generation is incomplete')
    expected = shared.rows(ROOT / 'prepare' / DATASET / 'evaluation.jsonl')
    predictions = shared.rows(folder / 'requests.jsonl')
    shared.check_coverage(expected, predictions)
    if len(expected) != 436 or sha(shared.__file__) != SHARED_SHA:
        raise ValueError('frozen Doc436/source contract changed')
    destination = folder / DIRECTORY
    work = destination / 'work'
    work.mkdir(parents=True, exist_ok=True)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[name] = str(work)
    tempfile.tempdir = str(work)
    if not (ROOT / 'score_sources' / DATASET / 'source_manifest.json').exists():
        raise ValueError('existing frozen native Doc source is required')
    kit = shared.bootstrap(DATASET)
    frame = frame_for(expected, predictions)
    imported, provenance = import_previous(folder, expected, predictions, frame, kit)
    protocol = dict(schema='native_doc_bounded_512_v1', dataset=DATASET, method=method, sample_count=436,
        judge_model=shared.MODEL, judge_checkpoint='/data/fanya/models/' + shared.MODEL,
        judge_dtype='float16', judge_moe_strategy='fused', judge_batch_size=1, judge_max_tokens=512,
        score_directory=DIRECTORY, native_auxeval_unchanged=True, native_transport_attempts=3,
        unknown_rule='Exactly three nonempty HTTP-success responses from one StrictJudge invocation; each independently proves pure native format extraction failure.',
        infrastructure_failure_rule='Any API, empty, transport, mixed, or incomplete attempt evidence remains fatal.',
        accuracy_rule='Native continuous eval_score in [0,1]; its exceptions follow native MMLongBench_acc zero rule. UNKNOWN is null and only contributes to the upper bound.',
        f1_rule='No exact F1 reported by this auxiliary bounded protocol.',
        inputs_sha256=shared.fingerprint(dict(rows=expected, predictions=predictions)),
        prediction_protocol='full_response_v1', original_prompt_and_prediction_unchanged=True,
        local_judge_differs_from_public_benchmark_judge=True, strict_primary_complete=False,
        main_cell_completion_written=False, import_provenance=provenance,
        scorer_sources={name: sha(SOURCE / name) for name in ('score_doc_bounded.py', 'score_native.py')},
        native_sources={NATIVE_FILE: sha(kit / NATIVE_FILE)}, kit_source_manifest_sha256=sha(kit / 'source_manifest.json'))
    protocol_path = destination / 'protocol.json'
    if protocol_path.exists() and shared.read(protocol_path) != protocol:
        raise ValueError('bounded Doc protocol changed; preserve this destination')
    shared.save(protocol_path, protocol)
    checkpoint = destination / 'records.json'
    records = shared.read(checkpoint) if checkpoint.exists() else imported
    validate_records(records, expected)
    if any(records.get(key) != value for key, value in imported.items()):
        raise ValueError('validated prior Doc prefix changed')
    shared.save(checkpoint, records)
    judge = BoundedJudge(port, DATASET, destination, judge_max_tokens=512)
    try:
        for ordinal, row in enumerate(expected):
            if row['id'] in records:
                continue
            records[row['id']] = evaluate(judge, row, frame.iloc[ordinal])
            shared.save(checkpoint, records)
            shared.save(destination / 'progress.json', summary(records, len(expected), protocol))
            print('DOC_BOUNDED_SCORE', method, len(records), len(expected), records[row['id']]['status'], flush=True)
        validate_records(records, expected)
        result = summary(records, len(expected), protocol)
        shared.save(destination / 'score.json', result)
        if before != identities(folder):
            raise ValueError('original Doc records or primary completion changed')
        shared.save(destination / 'preservation.json', dict(unchanged=True, before=before, after=identities(folder)))
        shared.save(destination / 'traversal_complete.json', dict(complete=True, traversal_complete=True,
            strict_primary_complete=False, sample_count=436, unknown=result['unknown'], score_file=str(destination / 'score.json'),
            protocol_sha256=shared.fingerprint(protocol), records_sha256=shared.fingerprint(records),
            completed_utc=datetime.now(timezone.utc).isoformat()))
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
