"""CPU fixtures for exact formal coverage and conservative failure classification."""

import importlib.util
import json
from pathlib import Path
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/fanya/vision2web')
from vision_formal_adapter_candidate import FormalCoverage, NO_START, freeze_policy


def load_adapter():
    path = ROOT / 'runtime_snapshot/benchmarks/bench_vision2web.py'
    spec = importlib.util.spec_from_file_location('fixture_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    evidence = ROOT / 'validation' / f'vision_formal_fixtures_{time.time_ns()}'
    names = [json.loads(line)['metadata']['task'] for line in
             (ROOT / 'prepare/Vision2Web_Level1/evaluation.jsonl').read_text().splitlines()]
    model = 'Qwen3-VL-30B-A3B-Instruct-fixture'
    checks = []
    cases = ('success', 'success_recovered_tool_error', 'success_source_error_string',
             'success_recovered_curl', 'success_final_http500', 'quality_zero', 'quality_zero_recovered_tool_error',
             'quality_zero_no_tools', 'no_tools_unparsed_text', 'no_tools_empty_final',
             'missing_terminal', 'thinking_parser', 'api_failure', 'prior_api_failure', 'tool_host_failure',
             'success_bad_attempt_protocol', 'conflicting_attempt_history',
             'sdk_api_error', 'cli_stderr_error',
             'unparsed_tool', 'judge_empty', 'browser_failure', 'container_failure')
    for case in cases:
        adapter = load_adapter()
        folder = evidence / case
        folder.mkdir(parents=True)
        config = dict(output=str(folder), projects=names, generation=dict(model=model),
                      datasets_dir='/data/fanya/datasets/vision2web_level1')
        adapter.save(folder / 'projects.json', [dict(name=name) for name in names])
        adapter.save(folder / 'inference_command.json', dict(command=['serve', '--model',
            '/data/fanya/models/Qwen3-VL-30B-A3B-Instruct', '--served-model-name', model,
            '--reasoning-parser', 'off', '--tool-call-parser', 'auto']))
        for i, name in enumerate(names):
            workspace = folder / 'results/webpage/claude_code' / model / name
            record = dict(project=name, name=name, model=model, framework='claude_code',
                status='success', error=None, logs=[], start_time='2026-11-11T20:00:00',
                end_time='2026-11-11T20:01:00', conversation=[
                    dict(type='assistant', message=dict(content=[dict(type='tool_use', id='tool1', name='Bash', input=dict(command='ls'))])),
                    dict(type='user', message=dict(content=[dict(type='tool_result', tool_use_id='tool1', content='prototypes resources')])),
                    dict(type='assistant', message=dict(content=[dict(type='text', text='I have finished the attempt.')])),
                    dict(type='result', subtype='success', is_error=False, terminal_reason='completed',
                         stop_reason='end_turn', api_error_status=None, result='I have finished the attempt.',
                         usage=dict(output_tokens=200), permission_denials=[]),
                ])
            no_start = i == 0 and case in ('quality_zero', 'quality_zero_recovered_tool_error', 'thinking_parser', 'api_failure',
                'prior_api_failure', 'conflicting_attempt_history', 'sdk_api_error',
                'unparsed_tool', 'container_failure', 'tool_host_failure',
                'quality_zero_no_tools', 'no_tools_unparsed_text', 'no_tools_empty_final')
            if no_start:
                record.update(status='failed', error=NO_START)
            if i == 0 and case in ('quality_zero_no_tools', 'no_tools_unparsed_text', 'no_tools_empty_final'):
                record['conversation'] = record['conversation'][2:]
                if case == 'no_tools_unparsed_text':
                    record['conversation'][0]['message']['content'][0]['text'] = '<tool_call>{}</tool_call>'
                if case == 'no_tools_empty_final':
                    record['conversation'][0]['message']['content'][0]['text'] = ''
                    record['conversation'][-1]['result'] = ''
            if i == 0 and case in ('success_recovered_tool_error', 'success_recovered_curl', 'quality_zero_recovered_tool_error', 'tool_host_failure'):
                record['conversation'][0:0] = [
                    dict(type='assistant', message=dict(content=[dict(type='tool_use', id='failed_tool', name='Bash', input=dict(command='identify missing'))])),
                    dict(type='user', message=dict(content=[dict(type='tool_result', tool_use_id='failed_tool', is_error=True,
                        content=('Tool host transport disconnected' if case == 'tool_host_failure' else
                                 'Exit code 7\ncurl: (7) Connection refused' if case == 'success_recovered_curl' else
                                 'Exit code 127\n/bin/bash: line 1: identify: command not found'))])),
                ]
            if i == 0 and case == 'success_source_error_string':
                record['conversation'][0]['message']['content'][0]['input'] = dict(
                    command='cat > app.js', source='const message = "API error";')
            if i == 0 and case == 'success_final_http500':
                record['conversation'][2]['message']['content'][0]['text'] = 'The page displays an HTTP500 error example.'
                record['conversation'][-1]['result'] = 'The page displays an HTTP500 error example.'
            if i == 0 and case == 'thinking_parser':
                record['conversation'][0]['message']['content'] = [dict(type='thinking', thinking='<tool_call>hidden</tool_call>')]
            if i == 0 and case == 'api_failure':
                record['conversation'][-1].update(is_error=True, subtype='error_during_execution', api_error_status=503)
            if i == 0 and case == 'sdk_api_error':
                record['conversation'].insert(0, dict(type='assistant', isApiErrorMessage=True,
                    message=dict(content=[dict(type='text', text='Request failed')]), error='api_error'))
            if i == 0 and case == 'cli_stderr_error':
                record['logs'] = ['API error: HTTP 503']
            if i == 0 and case == 'unparsed_tool':
                record['conversation'][2]['message']['content'][0]['text'] = '<tool_call>{}</tool_call>'
            if i == 0 and case == 'container_failure':
                record.update(status='error', error='Failed to create container', conversation=[])
            record.update(attempt_count=1, retry_count=0, attempts_history=[dict(
                attempt=1, status=record['status'], error=record['error'])])
            if i == 0 and case in ('prior_api_failure', 'success_bad_attempt_protocol'):
                record.update(attempt_count=2, retry_count=1)
                record['attempts_history'] = [dict(attempt=1, status='error', error='API error: HTTP 503'),
                    dict(attempt=2, status=record['status'], error=record['error'])]
            if i == 0 and case == 'conflicting_attempt_history':
                record['attempts_history'][0]['error'] = 'API error: HTTP 503'
            if not (i == 0 and case == 'missing_terminal'):
                adapter.save(workspace / 'results' / f'{name}.json', record)
            workspace.mkdir(parents=True, exist_ok=True)
            if no_start:
                continue
            (workspace / 'start.sh').write_text('# fixture only\n')
            adapter.save(workspace / 'evaluation_result.json', dict(status='error' if i == 0 and case == 'browser_failure' else 'success'))
            for viewport in ('desktop', 'tablet', 'mobile'):
                directory = workspace / 'test_results/prototypes'
                directory.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (2, 2)).save(directory / f'{viewport}_actual.png')
                scores = [] if i == 0 and case == 'judge_empty' else [dict(score=.5)]
                adapter.save(directory / f'{viewport}_scores.json', scores)
        policy = FormalCoverage(adapter, config, ROOT)
        strict_generation = adapter.generation_coverage(config)
        strict_evaluation = adapter.evaluation_coverage(config)
        policy.install()
        generation = policy.generation_coverage(config)
        evaluation = policy.evaluation_coverage(config)
        report = policy.report()
        expected = case in ('success', 'success_recovered_tool_error', 'success_source_error_string',
                           'success_recovered_curl', 'success_final_http500', 'quality_zero',
                           'quality_zero_recovered_tool_error', 'quality_zero_no_tools')
        assert report['complete'] is expected, (case, report)
        analysis = adapter.read(folder / 'official_analysis.json')
        stats = analysis['statistics'][f'webpage:claude_code:{model}']
        assert stats['total_count'] == 40 and len(stats['prototype_scores']) == 120
        if case in ('quality_zero', 'quality_zero_recovered_tool_error', 'quality_zero_no_tools'):
            assert strict_generation == [names[0]] and strict_evaluation
            assert not generation and not evaluation
            assert report['terminal_model_failure_zero_projects'] == [names[0]]
            assert stats['success_count'] == 39 and sum(stats['prototype_scores']) == 58.5
        elif not expected:
            assert generation or evaluation
            assert report['terminal_model_failure_zero_projects'] == []
        checks.append(dict(case=case, complete=report['complete'], expected=expected,
                           generation_missing=generation, evaluation_missing=evaluation,
                           denominator_projects=stats['total_count'], denominator_viewports=len(stats['prototype_scores'])))
    for count in (1, 39):
        config['projects'] = names[:count]
        try:
            FormalCoverage(adapter, config, ROOT)
        except ValueError:
            checks.append(dict(case=f'reject_{count}_project_manifest', complete=False))
        else:
            raise AssertionError('partial or smoke manifest was accepted')
    policy_path = evidence / 'frozen_policy.json'
    locked = dict(wrapper='candidate', wrapper_sha256='first', expected_projects=names,
                  frozen_attempt_count=1, frozen_retry_count=0, strict_smoke_unchanged=True)
    freeze_policy(adapter, policy_path, locked)
    before = policy_path.read_bytes()
    stamp = policy_path.stat().st_mtime_ns
    freeze_policy(adapter, policy_path, locked)
    assert policy_path.read_bytes() == before and policy_path.stat().st_mtime_ns == stamp
    checks.append(dict(case='same_policy_is_not_rewritten', complete=True))
    for key, value in [('wrapper_sha256', 'changed'), ('frozen_attempt_count', 2), ('expected_projects', names[::-1])]:
        try:
            freeze_policy(adapter, policy_path, {**locked, key: value})
        except ValueError:
            assert policy_path.read_bytes() == before and policy_path.stat().st_mtime_ns == stamp
            checks.append(dict(case=f'reject_policy_drift_{key}', complete=False))
        else:
            raise AssertionError('frozen policy drift was accepted')
    output = dict(complete=True, cases=len(checks), checks=checks, gpu_started=False,
                  official_analyzer_used=True, live_pipeline_modified=False, evidence_root=str(evidence))
    adapter.save(ROOT / 'validation/vision_formal_candidate_cpu.json', output)
    print(json.dumps(dict(complete=True, cases=len(checks), evidence_root=str(evidence))))


if __name__ == '__main__':
    main()
