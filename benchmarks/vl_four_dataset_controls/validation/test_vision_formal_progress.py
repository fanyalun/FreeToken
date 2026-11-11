"""CPU replay of native 40-task scheduling, early abort and synchronized partial progress."""

import asyncio
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision_formal_adapter import FormalCoverage, NO_START, terminal_reason
RUNTIME = Path('/data/fanya/tablemoe/vl_four_dataset_bs1_20261111/vision2web_runs_reasoning_off_20261111/smoke/fused/official_runtime')
sys.path.insert(0, str(RUNTIME))
from vision2web.inference.engine import InferenceEngine
from vision2web.analyze.analyzer import ResultAnalyzer


def adapter_module():
    spec = importlib.util.spec_from_file_location('progress_test_adapter', ROOT / 'runtime_snapshot/benchmarks/bench_vision2web.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def make_record(name, model, zero):
    status, error = ('failed', NO_START) if zero else ('success', None)
    return dict(project=name, name=name, task_type='webpage', model=model, framework='claude_code',
        status=status, error=error, logs=[], start_time='2026-11-11T20:00:00', end_time='2026-11-11T20:00:01',
        attempt_count=1, retry_count=0, attempts_history=[dict(attempt=1, status=status, error=error)],
        conversation=[dict(type='assistant', message=dict(content=[dict(type='text', text='Finished this attempt.')])),
            dict(type='result', subtype='success', is_error=False, terminal_reason='completed',
                stop_reason='end_turn', api_error_status=None, permission_denials=[],
                result='Finished this attempt.', usage=dict(output_tokens=20))])


async def exercise(folder, names, case):
    experiment = folder / case
    output = experiment / 'official/heldout/fused'
    adapter = adapter_module()
    split_path = experiment / 'prepare/Vision2Web_Level1/evaluation.jsonl'
    split_path.parent.mkdir(parents=True)
    split_path.write_text(''.join(json.dumps(dict(metadata=dict(task=name))) + '\n' for name in names))
    model = 'Qwen3-VL-30B-A3B-Instruct-fused'
    config = dict(output=str(output), projects=names, generation=dict(model=model),
                  datasets_dir='/data/fanya/datasets/vision2web_level1')
    adapter.save(output / 'projects.json', [dict(name=name) for name in names])
    adapter.save(output / 'inference_command.json', dict(command=['serve', '--model',
        '/data/fanya/models/Qwen3-VL-30B-A3B-Instruct', '--served-model-name', model,
        '--reasoning-parser', 'off', '--tool-call-parser', 'auto']))
    policy = FormalCoverage(adapter, config, experiment)
    policy.install()
    called = []

    class FakeProjectEngine(InferenceEngine):
        async def run_single_project(self, project):
            index = len(called)
            called.append(project.name)
            await asyncio.sleep(0)
            if case == 'cancelled' and index == 1:
                raise asyncio.CancelledError()
            zero = index != 0
            record = make_record(project.name, model, zero)
            workspace = output / 'results/webpage/claude_code' / model / project.name
            if case == 'infra_after_two' and index == 2:
                record['conversation'][-1].update(subtype='error_during_execution', is_error=True, api_error_status=503)
            if not (case == 'missing_terminal' and index == 1):
                adapter.save(workspace / 'results' / f'{project.name}.json', record)
            if not zero:
                workspace.mkdir(parents=True, exist_ok=True)
                (workspace / 'start.sh').write_text('# CPU fixture, never executed\n')
                adapter.save(workspace / 'evaluation_result.json', dict(status='success'))
                for viewport in ('desktop', 'tablet', 'mobile'):
                    path = workspace / 'test_results/prototypes'
                    path.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (2, 2)).save(path / f'{viewport}_actual.png')
                    adapter.save(path / f'{viewport}_scores.json', [dict(score=.5)])
            return record

    async def image_exists():
        return True

    engine = FakeProjectEngine.__new__(FakeProjectEngine)
    engine.config = SimpleNamespace(datasets_dir=Path(config['datasets_dir']), results_dir=output / 'results',
        inference=SimpleNamespace(max_workers=2 if case == 'wrong_concurrency' else 1, framework='claude_code', model=model))
    engine.logger = logging.getLogger('vision_formal_progress_fixture')
    engine.dataset_manager = SimpleNamespace(discover_projects=lambda **kwargs: [SimpleNamespace(name=name) for name in names])
    engine.sandbox_manager = SimpleNamespace(check_image=image_exists, image_name='unused-cpu-fixture')
    original = FakeProjectEngine.run_single_project
    restore = policy.install_inference_guard(FakeProjectEngine)
    try:
        await engine.run_all_projects(project_names=names, task_type='webpage')
    finally:
        restore()
        policy.audit()
        progress = policy.publish_progress()
    assert FakeProjectEngine.run_single_project is original
    expected_calls = dict(normal=40, infra_after_two=3, cancelled=2, missing_terminal=2, wrong_concurrency=0)[case]
    expected_completed = dict(normal=40, infra_after_two=2, cancelled=1, missing_terminal=1, wrong_concurrency=0)[case]
    assert len(called) == expected_calls, (case, called)
    assert progress['completed'] == expected_completed
    assert progress['attempted'] == expected_calls
    assert not set(called).intersection(progress['not_started_projects'])
    assert progress['complete'] is (case == 'normal')
    if case == 'infra_after_two':
        assert progress['attempted'] == 3 and len(progress['blocked_before_start']) == 37
        assert progress['quality_zero_projects'] == [names[1]]
    left = experiment / 'cells/Vision2Web_Level1/fused/progress.json'
    right = experiment / 'vision/fused/progress.json'
    assert left.read_bytes() == right.read_bytes()
    assert not (left.parent / 'generation_complete.json').exists()
    report = policy.report()
    assert report['complete'] is (case == 'normal')
    analysis = adapter.read(output / 'official_analysis.json')
    if case == 'normal':
        stats = analysis['statistics'][f'webpage:claude_code:{model}']
        assert stats['total_count'] == 40 and len(stats['prototype_scores']) == 120
        assert report['terminal_model_failure_zero_projects'] == names[1:]
    return dict(case=case, original_project_calls=len(called), completed=progress['completed'],
                attempted=progress['attempted'], blocked=len(progress['blocked_before_start']),
                strict_complete=report['complete'], dual_progress_identical=True)


def main():
    folder = ROOT / 'validation' / f'vision_formal_progress_{time.time_ns()}'
    folder.mkdir()
    names = [json.loads(line)['metadata']['task'] for line in
             (ROOT / 'prepare/Vision2Web_Level1/evaluation.jsonl').read_text().splitlines()]
    checks = [asyncio.run(exercise(folder, names, case)) for case in
              ('normal', 'infra_after_two', 'cancelled', 'missing_terminal', 'wrong_concurrency')]
    smoke = Path('/data/fanya/tablemoe/vl_four_dataset_bs1_20261111/vision2web_runs_reasoning_off_20261111/smoke/fused/results/webpage/claude_code/Qwen3-VL-30B-A3B-Instruct-fused/aihw')
    path = smoke / 'results/aihw.json'
    original = path.read_bytes()
    record = json.loads(original)
    assert terminal_reason(record, 'aihw', record['model']) is None
    native = ResultAnalyzer()._process_project(smoke, 'webpage', 'aihw', Path('/data/fanya/datasets/vision2web_level1'))
    assert native['prototypes'] == dict(desktop=0, tablet=0, mobile=0) and native['success'] is False
    assert path.read_bytes() == original and not (smoke / 'start.sh').exists()
    receipt = dict(complete=True, gpu_started=False, docker_started=False,
        native_run_all_projects_used=True, checks=checks, real_smoke_original_unchanged=True,
        real_smoke_native_zero=native['prototypes'], existing_smoke_success_relabelled=False,
        wrapper_sha256=hashlib.sha256((ROOT / 'vision_formal_adapter.py').read_bytes()).hexdigest(),
        evidence_root=str(folder))
    adapter_module().save(ROOT / 'validation/vision_formal_progress_cpu.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
