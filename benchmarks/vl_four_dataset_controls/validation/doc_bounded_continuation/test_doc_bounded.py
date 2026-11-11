from contextlib import contextmanager
import copy
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = HERE / ('cpu_' + str(time.time_ns()))
WORK.mkdir()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TMPDIR'] = str(WORK)
tempfile.tempdir = str(WORK)
sys.path.insert(0, str(ROOT))
import score_doc_bounded as bounded
import score_doc_bounded_with_server as wrapper
import score_native as shared
logging.disable(logging.CRITICAL)
kit = shared.bootstrap(bounded.DATASET)
from vlmeval.dataset.mmlongbenchdoc import build_mmlongbench_gpt4_prompt
checks = []

def check(name, value=True):
    assert value, name
    checks.append(name)

def rejects(fn, cls):
    try:
        fn()
    except cls as error:
        return error
    raise AssertionError('expected failure')

expected = shared.rows(ROOT / 'prepare/MMLongBench_DOC/evaluation.jsonl')
folder = ROOT / 'cells/MMLongBench_DOC/tablemoe'
predictions = shared.rows(folder / 'requests.jsonl')
original = bounded.identities(folder)
frame = bounded.frame_for(expected, predictions)
imported, provenance = bounded.import_previous(folder, expected, predictions, frame, kit)
check('all_real_339_records_replayed_with_original512_request_response_provenance', len(imported) == 339)
check('native_continuous_scores_preserved', any(0 < x['score'] < 1 for x in imported.values()))
fused = ROOT / 'cells/MMLongBench_DOC/fused'
fused_predictions = shared.rows(fused / 'requests.jsonl')
fused_frame = bounded.frame_for(expected, fused_predictions)
fused_records, _ = bounded.import_previous(fused, expected, fused_predictions, fused_frame, kit)
fused_score = shared.read(fused / 'score_local_fused_native_512/score.json')
fused_acc = 100 * sum(x['score'] for x in fused_records.values()) / 436
check('all_real_fused436_native_replay_reproduces_completed_ACC', len(fused_records) == 436 and abs(fused_acc - fused_score['accuracy_pct']) < 1e-7)
case = next(i for i, row in enumerate(expected) if row['id'] == 'MMLongBench_DOC:877')
prompt = build_mmlongbench_gpt4_prompt(frame.iloc[case])
body = bounded.body_for(prompt)
calls = [c for c in shared.rows(folder / 'score_local_fused_native_512/judge_requests.jsonl') if c['id'] == expected[case]['id']]
evidence = bounded.validate_format_attempts(calls, expected[case]['id'], body)
check('actual_877_three512_nonempty_pure_format_failures_proved', evidence['attempts'] == 3)
shared.save(HERE / 'actual_failure_evidence.json', dict(case_id=expected[case]['id'], validated=True,
    evidence=evidence, request_max_tokens=[x['request']['max_tokens'] for x in calls],
    attempt=[x['attempt'] for x in calls], errors=[x['error'] for x in calls],
    finish_reason=[x['response']['choices'][0]['finish_reason'] for x in calls],
    completion_tokens=[x['response']['usage']['completion_tokens'] for x in calls],
    response_characters=[len(x['response']['choices'][0]['message']['content']) for x in calls]))
for kind in ('empty', 'api', 'transport', 'mixed_error', 'wrong_id', 'wrong_cap', 'wrong_prompt', 'valid_format', 'two_attempts'):
    damaged = copy.deepcopy(calls)
    if kind == 'empty': damaged[0]['response']['choices'][0]['message']['content'] = ''
    elif kind == 'api': damaged[0]['response']['choices'][0]['message']['content'] = 'Failed to obtain answer via API.'
    elif kind == 'transport': damaged[0].pop('response')
    elif kind == 'mixed_error': damaged[0]['error'] = "TimeoutError('socket timed out')"
    elif kind == 'wrong_id': damaged[0]['id'] = 'different'
    elif kind == 'wrong_cap': damaged[0]['request']['max_tokens'] = 4096
    elif kind == 'wrong_prompt': damaged[0]['request']['messages'][0]['content'] += 'answer only'
    elif kind == 'valid_format': damaged[0]['response']['choices'][0]['message']['content'] = 'Extracted answer: x\nAnswer format: String'
    else: damaged.pop()
    check('unknown_rejects_' + kind, isinstance(rejects(lambda: bounded.validate_format_attempts(damaged, expected[case]['id'], body), ValueError), ValueError))

class Opener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
    def open(self, request, timeout):
        self.requests.append(json.loads(request.data))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return io.StringIO(json.dumps(value))

real_sleep = shared.time.sleep
shared.time.sleep = lambda seconds: None
isolated = WORK / 'wire'
isolated.mkdir()
journal = isolated / 'judge_requests.jsonl'
journal.write_text(json.dumps(dict(id='earlier_unrelated', attempt=2, error='old unrelated error')) + '\n')
judge = bounded.BoundedJudge(1, bounded.DATASET, isolated, judge_max_tokens=512)
judge.sample_id = expected[case]['id']
judge.opener = Opener([x['response'] for x in calls])
unknown_error = rejects(lambda: judge.generate(prompt, temperature=0), bounded.UnknownDocExtraction)
check('real_StrictJudge_three_attempts_are_isolated_by_journal_byte_offset', unknown_error.evidence['byte_start'] > 0
      and len(judge.opener.requests) == 3 and all(x == body for x in judge.opener.requests))
mixed = WORK / 'mixed'
mixed.mkdir()
judge = bounded.BoundedJudge(1, bounded.DATASET, mixed, judge_max_tokens=512)
judge.sample_id = expected[case]['id']
judge.opener = Opener([urllib.error.URLError('connection refused'), calls[1]['response'], calls[2]['response']])
error = rejects(lambda: judge.generate(prompt), RuntimeError)
check('real_StrictJudge_mixed_transport_plus_two_format_failures_remains_fatal', not isinstance(error, bounded.UnknownDocExtraction))
shared.time.sleep = real_sleep

known = next(x for x in imported.values() if 0 < x['score'] < 1)
unknown = dict(status='UNKNOWN', score=None)
summary = bounded.summary({'a': known, 'b': unknown}, 3, {'method': 'tablemoe'})
check('continuous_ACC_bounds_use_score_sum_and_unknown_pending', abs(summary['full_denominator_accuracy_lower_pct'] - 100 * known['score'] / 3) < 1e-10
      and abs(summary['full_denominator_accuracy_upper_pct'] - 100 * (known['score'] + 2) / 3) < 1e-10)
check('unknown_F1_not_reported_as_exact', summary['overall_f1'] is None and summary['strict_primary_complete'] is False)

fixture = WORK / 'fixture'
cell = fixture / 'cells/MMLongBench_DOC/tablemoe'
cell.mkdir(parents=True)
for name in ('generation_complete.json', 'requests.jsonl'):
    shutil.copyfile(folder / name, cell / name)
shutil.copytree(folder / bounded.PREVIOUS_DIRECTORY, cell / bounded.PREVIOUS_DIRECTORY)
prepare = fixture / 'prepare/MMLongBench_DOC'
prepare.mkdir(parents=True)
shutil.copyfile(ROOT / 'prepare/MMLongBench_DOC/evaluation.jsonl', prepare / 'evaluation.jsonl')
source = fixture / 'score_sources/MMLongBench_DOC'
source.mkdir(parents=True)
shutil.copyfile(kit / 'source_manifest.json', source / 'source_manifest.json')
bounded.ROOT = fixture
protocol_path = cell / bounded.PREVIOUS_DIRECTORY / 'protocol.json'
p = shared.read(protocol_path)
p['score_output_path'] = str(cell / bounded.PREVIOUS_DIRECTORY)
shared.save(protocol_path, p)
original_protocol = protocol_path.read_bytes()
p['judge_max_tokens'] = 4096
shared.save(protocol_path, p)
check('import_rejects_changed_judge_protocol', isinstance(rejects(lambda: bounded.import_previous(cell, expected, predictions, frame, kit), ValueError), ValueError))
protocol_path.write_bytes(original_protocol)
record_path = cell / bounded.PREVIOUS_DIRECTORY / 'records.json'
original_records = record_path.read_bytes()
p = shared.read(record_path)
p[next(iter(p))]['pred'] = 'tampered answer'
shared.save(record_path, p)
check('import_rejects_changed_extracted_answer', isinstance(rejects(lambda: bounded.import_previous(cell, expected, predictions, frame, kit), ValueError), ValueError))
record_path.write_bytes(original_records)
before = bounded.identities(cell)
judged = []
class MockJudge:
    def __init__(self, *args, **kwargs):
        assert kwargs['judge_max_tokens'] == 512
    def generate(self, prompt, **kwargs):
        judged.append(self.sample_id)
        if self.sample_id == expected[case]['id']:
            raise bounded.UnknownDocExtraction(evidence)
        return 'Extracted answer: Not answerable\nAnswer format: String'
real_judge = bounded.BoundedJudge
bounded.BoundedJudge = MockJudge
result = bounded.score('tablemoe', 1)
check('full436_mock_continuation_imports339_and_only_judges97', result['processed'] == 436 and len(judged) == 97
      and all(x not in imported for x in judged) and result['unknown'] == 1 and result['known_count'] == 435)
check('primary_files_immutable_no_false_strict_completion', bounded.identities(folder) == original
      and bounded.identities(cell) == before and not (cell / 'complete.json').exists()
      and not (cell / bounded.DIRECTORY / 'complete.json').exists()
      and (cell / bounded.DIRECTORY / 'traversal_complete.json').exists())
count = len(judged)
bounded.score('tablemoe', 1)
check('resumption_reuses_own_validated_prefix_without_judge_calls', len(judged) == count)
bounded.BoundedJudge = real_judge
wrapper.ROOT = fixture
wrapper.main('tablemoe', 0)
check('wrapper_recognizes_only_auxiliary436_marker')
server_events = []
@contextmanager
def server(*args):
    server_events.append('enter')
    try: yield
    finally: server_events.append('exit')
wrapper.running_server = server
wrapper.command = lambda *args: (['never_launch_server'], 1)
wrapper.environment = lambda gpu: {'CUDA_VISIBLE_DEVICES': ''}
wrapper.subprocess.run = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('worker failed'))
other = fixture / 'cells/MMLongBench_DOC/modes'
other.mkdir()
shutil.copyfile(folder / 'generation_complete.json', other / 'generation_complete.json')
check('wrapper_worker_error_exits_server_context', isinstance(rejects(lambda: wrapper.main('modes', 0), RuntimeError), RuntimeError)
      and server_events == ['enter', 'exit'])
shared.save(HERE / 'cpu_validation.json', dict(complete=True, cpu_only=True, gpu_started=False, real_judge_called=False,
    cases=len(checks), checks=checks, real_tablemoe_imported=339, real_fused_replayed=436,
    real_fused_replayed_ACC_pct=fused_acc, real_fused_published_ACC_pct=fused_score['accuracy_pct'],
    evidence_root=str(WORK), mock436_summary={k: v for k, v in result.items() if k != 'protocol'},
    sources={name: bounded.sha(ROOT / name) for name in ('score_doc_bounded.py', 'score_doc_bounded_with_server.py')},
    shared_sources_untouched={name: bounded.sha(ROOT / name) for name in ('score_native.py', 'score_with_server.py')}))
print(json.dumps({'passed':len(checks),'receipt':str(HERE / 'cpu_validation.json')}))
