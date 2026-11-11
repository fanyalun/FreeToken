"""Check held-out coverage and the controlled MMMU lookup ablation."""

import importlib.util
import json
from pathlib import Path

import pytest


def load_runner():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/mmmu_answer_only_ab/run_mmmu_ab.py'
    spec = importlib.util.spec_from_file_location('mmmu_answer_only_test_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation_rows():
    return [dict(index=1, metadata=dict(question_type='multiple-choice', split='dev')),
            dict(index=2, metadata=dict(question_type='open', split='validation'))]


def test_overall_includes_validation_hits():
    runner = load_runner()
    scores = runner.aggregate_hits(evaluation_rows(), [dict(index=1, hit=0), dict(index=2, hit=1)])
    assert scores['overall'] == dict(correct=1, samples=2, accuracy_pct=50.)
    assert scores['dev']['accuracy_pct'] == 0.
    assert scores['validation']['accuracy_pct'] == 100.
    assert scores['multiple_choice']['samples'] == scores['open']['samples'] == 1


def test_duplicate_native_index_is_rejected():
    runner = load_runner()
    with pytest.raises(ValueError, match='indices'):
        runner.aggregate_hits(evaluation_rows(), [dict(index=1, hit=1), dict(index=1, hit=1)])


def test_lookup_ablation_changes_only_lookup_flag(tmp_path):
    runner = load_runner()
    runner.ROOT = tmp_path
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(inputs=[dict(
        dataset='MMMU_DEV_VAL', pca_dir='/preserved/pca', cache_dir='/preserved/cache')])))
    lookup, lookup_port = runner.command('mmmu_lookup', 0)
    schedule, schedule_port = runner.command('mmmu_schedule', 0)
    for command in (lookup, schedule):
        position = command.index('--tablemoe-report-path')
        command.pop(position + 1)
        command.pop(position)
    assert schedule.pop() == '--tablemoe-skip-lookup'
    assert lookup == schedule
    assert lookup_port == schedule_port
    assert '--tablemoe-no-renormalize-weights' in lookup
    assert lookup[lookup.index('--tablemoe-decode-top-h') + 1] == '8'
    assert lookup[lookup.index('--moe-cache-size') + 1] == '6144'
