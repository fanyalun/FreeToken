import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = HERE / ('cpu_' + str(time.time_ns()))
WORK.mkdir()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TMPDIR'] = str(WORK)
os.environ.pop('VERBOSE', None)
tempfile.tempdir = str(WORK)
sys.path.insert(0, str(ROOT))
import score_mmmu_bounded as bounded
import score_mmmu_bounded_with_server as wrapper
import score_native as shared
logging.disable(logging.CRITICAL)
kit = shared.bootstrap(bounded.DATASET)
from vlmeval.dataset.utils import multiple_choice as native
from vlmeval.utils.matching_util import can_infer, can_infer_option, can_infer_text
checks = []

def check(name, value=True):
    assert value, name
    checks.append(name)

def rejects(fn, error):
    try:
        fn()
    except error:
        return True
    raise AssertionError('expected exception was not raised')

expected = shared.rows(ROOT / 'prepare/MMMU_DEV_VAL/evaluation.jsonl')
folder = ROOT / 'cells/MMMU_DEV_VAL/fused'
predictions = shared.rows(folder / 'requests.jsonl')
original_identity = bounded.identities(folder)
frame, opened = bounded.frame_for(expected, predictions)
imported, provenance = bounded.import_previous(folder, expected, predictions, frame, opened, kit)
check('all_298_real_records_native_replayed_and_request_provenance_validated', len(imported) == 298)
check('old_and_new_source_versions_explicitly_distinct', provenance['source_versions_identical'] is False)
case = next(i for i, row in enumerate(expected) if row['id'] == 'MMMU_DEV_VAL:8084')
choices = native.build_choices(frame.iloc[case])
calls = [x for x in shared.rows(folder / 'score_local_fused_native/judge_requests.jsonl') if x['id'] == expected[case]['id']]
evidence = []
for call in calls:
    text = call['response']['choices'][0]['message']['content']
    normalized = text
    for character in '.()[],:;!*#{}':
        normalized = normalized.replace(character, ' ')
    labels = [c for c in choices if c in normalized.split()]
    assert labels == list('ABCD') and len(text) == 4651 and 2 * sum(len(str(x)) for x in choices.values()) == 60
    assert can_infer_option(text + '\nFinal answer: C', choices.copy()) is False
    assert can_infer_text(text + '\nFinal answer: C', choices.copy()) is False
    evidence.append(dict(labels=labels, response_characters=len(text), text_matching_maximum_characters=60,
        request_max_tokens=call['request']['max_tokens'], finish_reason=call['response']['choices'][0]['finish_reason'],
        completion_tokens=call['response']['usage']['completion_tokens'],
        response_sha256=hashlib.sha256(text.encode()).hexdigest()))
check('actual_three_greedy_2048_prefixes_cannot_be_repaired_by_appending_final_choice', len(evidence) == 3)
shared.save(HERE / '8192_static_rejection.json', dict(case_id=expected[case]['id'], attempts=evidence,
    reason='Native matcher counts distinct option labels across the entire prefix, then rejects answer text exceeding twice option text length. Appending a suffix cannot decrease either count or existing prefix length.',
    native_sources={name: bounded.sha(kit / name) for name in bounded.NATIVE_FILES},
    held_before_execution_receipt=str(ROOT / 'protocol_amendments/mmmu_judge_budget/held_before_execution.json'),
    retry_8192_directory_exists=(folder / 'score_local_fused_native_8192').exists(), executed_8192=False))

class Judge:
    sample_id = None
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
    def generate(self, prompt):
        self.calls.append(prompt)
        return self.answer

old_rd = native.rd
judge = Judge('A B C D, lengthy unresolved reasoning')
unknown = bounded.evaluate(judge, expected[case], frame.iloc[case], opened[case])
check('native_three_attempts_become_unknown_without_zero_or_random', len(judge.calls) == 3 and unknown['status'] == 'UNKNOWN'
      and unknown['hit'] is None and unknown['option'] is None and native.rd is old_rd)
judge = Judge('C')
known = bounded.evaluate(judge, expected[case], frame.iloc[case], opened[case])
check('native_known_answer_remains_correct', known['status'] == 'CORRECT' and len(judge.calls) == 1)
judge = Judge('A')
wrong = bounded.evaluate(judge, expected[case], frame.iloc[case], opened[case])
check('native_known_wrong_answer_remains_wrong', wrong['status'] == 'WRONG')
class Failed:
    def generate(self, prompt):
        raise RuntimeError('transport offline')
check('infrastructure_failure_is_not_converted_to_unknown_or_zero', rejects(lambda: bounded.evaluate(Failed(), expected[case], frame.iloc[case], opened[case]), RuntimeError) and native.rd is old_rd)
counts = bounded.summary({'a': known, 'b': wrong, 'c': unknown}, 4, {'method': 'fused'})
check('accuracy_bounds_include_unknown_and_pending_without_scoring_them', counts['full_denominator_accuracy_lower_pct'] == 25
      and counts['full_denominator_accuracy_upper_pct'] == 75 and counts['score_coverage_pct'] == 50)

fixture = WORK / 'fixture'
cell = fixture / 'cells/MMMU_DEV_VAL/fused'
cell.mkdir(parents=True)
for name in ('generation_complete.json', 'requests.jsonl'):
    shutil.copyfile(folder / name, cell / name)
shutil.copytree(folder / 'score_local_fused_native', cell / 'score_local_fused_native')
prepare = fixture / 'prepare/MMMU_DEV_VAL'
prepare.mkdir(parents=True)
shutil.copyfile(ROOT / 'prepare/MMMU_DEV_VAL/evaluation.jsonl', prepare / 'evaluation.jsonl')
source = fixture / 'score_sources/MMMU_DEV_VAL'
source.mkdir(parents=True)
shutil.copyfile(kit / 'source_manifest.json', source / 'source_manifest.json')
fixture_protocol = cell / 'score_local_fused_native/protocol.json'
original_protocol = fixture_protocol.read_bytes()
p = json.loads(original_protocol)
p['judge_max_tokens'] = 8192
shared.save(fixture_protocol, p)
check('incompatible_import_budget_rejected', rejects(lambda: bounded.import_previous(cell, expected, predictions, frame, opened, kit), ValueError))
fixture_protocol.write_bytes(original_protocol)
fixture_records = cell / 'score_local_fused_native/records.json'
original_records = fixture_records.read_bytes()
p = json.loads(original_records)
p[next(iter(p))]['hit'] = 1 - p[next(iter(p))]['hit']
shared.save(fixture_records, p)
check('tampered_import_hit_rejected', rejects(lambda: bounded.import_previous(cell, expected, predictions, frame, opened, kit), ValueError))
fixture_records.write_bytes(original_records)
bounded.ROOT = fixture
before = bounded.identities(cell)
by_id = {row['id']: i for i, row in enumerate(expected)}
judge_calls = []
class MockJudge:
    def __init__(self, port, dataset, destination, judge_max_tokens=None):
        assert judge_max_tokens == 2048
    def generate(self, prompt):
        judge_calls.append(self.sample_id)
        return 'A B C D unresolved' if self.sample_id == expected[case]['id'] else frame.iloc[by_id[self.sample_id]]['GT']
real_judge = shared.StrictJudge
shared.StrictJudge = MockJudge
result = bounded.score('fused', 1)
check('full_420_cpu_mock_continuation_imports_298_without_rejudging_them', result['processed'] == 420
      and all(x not in imported for x in judge_calls) and judge_calls.count(expected[case]['id']) == 3)
check('full_traversal_has_one_unknown_and_valid_bounds', result['unknown'] == 1 and result['known_count'] == 419
      and result['known_correct'] + result['known_wrong'] == 419
      and result['full_denominator_accuracy_upper_pct'] - result['full_denominator_accuracy_lower_pct'] > 0)
check('only_auxiliary_traversal_marker_written_no_primary_complete',
      (cell / bounded.DIRECTORY / 'traversal_complete.json').exists()
      and not (cell / bounded.DIRECTORY / 'complete.json').exists()
      and not (cell / 'complete.json').exists() and result['strict_primary_complete'] is False)
check('original_production_and_fixture_primary_files_immutable', bounded.identities(folder) == original_identity and bounded.identities(cell) == before)
count = len(judge_calls)
bounded.score('fused', 1)
check('resume_complete_continuation_makes_no_judge_calls', len(judge_calls) == count)
shared.StrictJudge = real_judge

wrapper.ROOT = fixture
wrapper.main('fused', 0)
check('wrapper_skips_valid_auxiliary_traversal_marker')
server_events = []
@contextmanager
def server(*args):
    server_events.append('enter')
    try:
        yield
    finally:
        server_events.append('exit')
wrapper.running_server = server
wrapper.command = lambda *args: (['never_launch_server'], 1)
wrapper.environment = lambda gpu: {'CUDA_VISIBLE_DEVICES': ''}
wrapper.subprocess.run = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('worker failed'))
other = fixture / 'cells/MMMU_DEV_VAL/modes'
other.mkdir()
shutil.copyfile(folder / 'generation_complete.json', other / 'generation_complete.json')
check('wrapper_failure_exits_owned_server_context', rejects(lambda: wrapper.main('modes', 0), RuntimeError)
      and server_events == ['enter', 'exit'])
shared.save(HERE / 'cpu_validation.json', dict(complete=True, cpu_only=True, gpu_started=False,
    real_judge_called=False, checks=checks, cases=len(checks), real_prior_records_validated=len(imported),
    mock_420_summary={k: v for k, v in result.items() if k != 'protocol'}, evidence_root=str(WORK),
    sources={name: bounded.sha(ROOT / name) for name in ('score_mmmu_bounded.py', 'score_mmmu_bounded_with_server.py')},
    shared_sources_untouched={name: bounded.sha(ROOT / name) for name in ('score_native.py', 'score_with_server.py')}))
print(json.dumps({'passed': len(checks), 'receipt': str(HERE / 'cpu_validation.json')}))
