"""CPU tests for separate native-judge output budgets and score directories."""

from contextlib import contextmanager
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = HERE / f'cpu_{time.time_ns()}'
WORK.mkdir()
os.environ['TMPDIR'] = str(WORK)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
tempfile.tempdir = str(WORK)
sys.path.insert(0, str(ROOT))

import score_native as native
import score_with_server as wrapper


def main():
    checks = []
    for name, cap in [('MMMU_DEV_VAL', 2048), ('BabyVision', 16), ('MMLongBench_DOC', 128)]:
        assert native.score_options(name) == (cap, 'score_local_fused_native')
    checks.append('original_default_caps_and_directory')
    for kwargs in [dict(judge_max_tokens=0), dict(judge_max_tokens=-1),
                   dict(judge_max_tokens=True), dict(score_directory='../old')]:
        try:
            native.score_options('MMMU_DEV_VAL', **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)
    checks.append('invalid_budget_and_directory_rejected')
    requests = []
    class Response(io.StringIO):
        pass
    class Opener:
        def open(self, request, timeout):
            requests.append(json.loads(request.data))
            return Response(json.dumps(dict(choices=[dict(message=dict(content='A'))])))
    for cap in (None, 8192):
        judge = native.StrictJudge(1, 'MMMU_DEV_VAL', WORK, judge_max_tokens=cap)
        judge.opener = Opener()
        assert judge.generate('unchanged native fixture prompt') == 'A'
    assert [request['max_tokens'] for request in requests] == [2048, 8192]
    assert requests[0]['messages'] == requests[1]['messages'] == [dict(role='user', content='unchanged native fixture prompt')]
    assert requests[0]['model'] == requests[1]['model'] == native.MODEL
    checks.append('strict_judge_wire_cap_only_prompt_and_model_unchanged')
    kit = native.bootstrap('MMMU_DEV_VAL')
    native.bootstrap = lambda name: kit
    native.ROOT = WORK / 'native'
    expected = [dict(id=f'fixture:{i}', index=i, metadata=dict(question='Choose a fixture.',
                    answer=answer, A='alpha', B='beta')) for i, answer in enumerate(('A', 'B'))]
    prepare = native.ROOT / 'prepare/MMMU_DEV_VAL'
    prepare.mkdir(parents=True)
    (prepare / 'evaluation.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in expected))
    calls = []
    def generate(self, prompt, **kwargs):
        calls.append(dict(id=self.sample_id, cap=self.max_tokens, prompt=prompt))
        return 'A' if self.sample_id == 'fixture:0' else 'B'
    native.StrictJudge.generate = generate
    def cell(method):
        target = native.ROOT / 'cells/MMMU_DEV_VAL' / method
        target.mkdir(parents=True)
        native.save(target / 'generation_complete.json', dict(complete=True))
        predictions = [dict(id=row['id'], prediction_text='reasoning without a final selection. ' * 100,
                            natural_eos=True, finish_reason='stop') for row in expected]
        (target / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in predictions))
        return target
    default = cell('default_fixture')
    result = native.score('MMMU_DEV_VAL', 'default_fixture', 1)
    assert result['scoring_denominator'] == 2 and result['protocol']['judge_max_tokens'] == 2048
    assert (default / 'score_local_fused_native/complete.json').is_file()
    checks.append('score_default_behavior_preserved')
    target = cell('variant_fixture')
    old = target / 'score_local_fused_native/records.json'
    native.save(old, {'fixture:0': dict(hit=0, log='old partial result')})
    old_bytes = old.read_bytes()
    calls.clear()
    result = native.score('MMMU_DEV_VAL', 'variant_fixture', 1, judge_max_tokens=8192,
                          score_directory='score_local_fused_native_8192')
    assert result['scoring_denominator'] == 2 and result['accuracy_pct'] == 100
    assert [call['id'] for call in calls] == [row['id'] for row in expected]
    assert all(call['cap'] == 8192 for call in calls)
    assert old.read_bytes() == old_bytes
    assert result['protocol']['schema'] == 'native_local_fused_judge_v2'
    assert result['protocol']['judge_max_tokens'] == 8192
    assert result['protocol']['score_directory'] == 'score_local_fused_native_8192'
    assert native.read(target / 'complete.json')['score_file'] == str(target / 'score_local_fused_native_8192/score.json')
    checks.append('variant_scores_every_sample_without_old_partial_reuse')
    calls.clear()
    native.score('MMMU_DEV_VAL', 'variant_fixture', 1, judge_max_tokens=8192,
                 score_directory='score_local_fused_native_8192')
    assert not calls
    checks.append('variant_completion_reused_in_its_own_directory')
    bad = cell('random_fixture')
    bad_calls = []
    def unmatched(self, prompt, **kwargs):
        bad_calls.append(self.max_tokens)
        return 'ambiguous reasoning without any selected option. ' * 20
    native.StrictJudge.generate = unmatched
    logging.disable(logging.CRITICAL)
    try:
        native.score('MMMU_DEV_VAL', 'random_fixture', 1, judge_max_tokens=8192,
                     score_directory='score_local_fused_native_8192')
    except RuntimeError as error:
        assert 'random guess rejected' in str(error)
    else:
        raise AssertionError('random fallback accepted')
    assert bad_calls == [8192, 8192, 8192]
    assert not (bad / 'complete.json').exists()
    checks.append('exactly_three_native_matcher_retries_random_fallback_rejected')
    wrapper.ROOT = WORK / 'wrapper'
    target = wrapper.ROOT / 'cells/MMMU_DEV_VAL/fused'
    native.save(target / 'generation_complete.json', dict(complete=True))
    native.save(target / 'score_local_fused_native/complete.json', dict(complete=True))
    native.save(target / 'score_local_fused_native/score.json', dict(protocol=dict(judge_max_tokens=2048)))
    events, commands = [], []
    @contextmanager
    def server(*args):
        events.append('enter')
        try:
            yield None
        finally:
            events.append('exit')
    wrapper.command = lambda *args: (['cpu-only-server-mock'], 1234)
    wrapper.environment = lambda *args: dict(CUDA_VISIBLE_DEVICES='')
    wrapper.running_server = server
    def subprocess_run(command, **kwargs):
        commands.append(command)
        assert command[command.index('--judge-max-tokens') + 1] == '8192'
        assert command[command.index('--score-directory') + 1] == 'score_local_fused_native_8192'
        return None
    wrapper.subprocess.run = subprocess_run
    wrapper.main('MMMU_DEV_VAL', ['fused'], 0)
    assert not events
    checks.append('wrapper_default_completed_directory_is_reused')
    wrapper.main('MMMU_DEV_VAL', ['fused'], 0, judge_max_tokens=8192,
                 score_directory='score_local_fused_native_8192')
    assert events == ['enter', 'exit'] and len(commands) == 1
    checks.append('wrapper_variant_does_not_skip_due_to_default_completion')
    native.save(target / 'score_local_fused_native_8192/complete.json', dict(complete=True))
    native.save(target / 'score_local_fused_native_8192/score.json', dict(protocol=dict(
        judge_max_tokens=8192, score_directory='score_local_fused_native_8192')))
    wrapper.main('MMMU_DEV_VAL', ['fused'], 0, judge_max_tokens=8192,
                 score_directory='score_local_fused_native_8192')
    assert len(commands) == 1
    checks.append('wrapper_variant_completed_directory_is_reused')
    try:
        wrapper.main('MMMU_DEV_VAL', ['fused'], 0, judge_max_tokens=8192)
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched completion accepted')
    checks.append('wrapper_completed_budget_mismatch_rejected')
    native.save(target / 'score_local_fused_native_8192/complete.json', dict(complete=False))
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    wrapper.subprocess.run = fail
    try:
        wrapper.main('MMMU_DEV_VAL', ['fused'], 0, judge_max_tokens=8192,
                     score_directory='score_local_fused_native_8192')
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError('worker failure disappeared')
    assert events[-2:] == ['enter', 'exit']
    checks.append('worker_failure_exits_server_context')
    output = dict(complete=True, cpu_only=True, gpu_started=False, real_judge_called=False,
        cases=len(checks), checks=checks, evidence_root=str(WORK),
        sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in ('score_native.py', 'score_with_server.py')})
    native.save(HERE / 'cpu_validation.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
