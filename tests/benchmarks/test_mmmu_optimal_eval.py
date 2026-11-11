"""Protect alias-aware judging, complete aggregation, and isolated calibration sampling."""

import importlib.util
from pathlib import Path

import pytest


def common():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/mmmu_optimal_eval/common.py'
    spec = importlib.util.spec_from_file_location('mmmu_optimal_common_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_open_reference_aliases_are_individual_answers():
    module = common()
    assert module.answer_aliases("['$MgS$', 'MgS']") == ['$MgS$', 'MgS']
    assert module.answer_aliases("['24/7', '3.429']") == ['24/7', '3.429']
    assert module.answer_aliases('9.20') == ['9.20']
    with pytest.raises(ValueError, match='empty'):
        module.answer_aliases('')


def test_judge_requires_boolean_not_ambiguous_prose_or_number():
    module = common()
    assert module.parse_judgment('{"correct": true}') == 1
    assert module.parse_judgment('```json\n{"correct": false}\n```') == 0
    for value in ['{"correct": "false"}', '{"correct": 1}', 'true', '{"correct":true,"reason":"ok"}']:
        with pytest.raises(ValueError):
            module.parse_judgment(value)


def test_overall_combines_question_types_and_splits_without_separate_reports():
    module = common()
    expected = [dict(index=1, metadata=dict(question_type='multiple-choice', split='dev')),
                dict(index=2, metadata=dict(question_type='open', split='validation'))]
    result = module.aggregate(expected, [dict(index=1, hit=0), dict(index=2, hit=1)])
    assert result == dict(correct=1, scoring_denominator=2, accuracy_pct=50.)
    with pytest.raises(ValueError, match='coverage'):
        module.aggregate(expected, [dict(index=1, hit=1), dict(index=1, hit=1)])


def row(index, category, group=None):
    return dict(id=str(index), index=index, group_id=group or str(index),
                images=[dict(path=f'{index}.jpg')], metadata=dict(category=category))


def test_calibration_sampling_is_seeded_stratified_and_group_isolated():
    module = common()
    calibration = [row(f'{category}{i}', category) for category in ['a', 'b'] for i in range(5)]
    selected, record = module.select_calibration(calibration, [row('heldout', 'a')], per_category=2)
    assert record['category_counts'] == {'a': 2, 'b': 2}
    assert module.select_calibration(calibration, [row('heldout', 'a')], per_category=2)[0] == selected
    with pytest.raises(ValueError, match='overlap'):
        module.select_calibration(calibration, [row('heldout', 'a', group='a0')])
    grouped = [row('a0', 'a', 'shared'), row('b0', 'b', 'shared')]
    selected, record = module.select_calibration(grouped, [row('heldout', 'a')], per_category=1)
    assert len(selected) == 2 and record['category_counts'] == {'a': 1, 'b': 1}


def test_schedule_uses_the_same_selected_parameters_and_only_disables_lookup():
    module = common()
    table, schedule = module.job('mmmu_tablemoe'), module.job('mmmu_schedule')
    assert (table['alpha'], table['h'], table['slots']) == (.8, 4, 3072)
    for key in ['id', 'skip_lookup']:
        table.pop(key)
        schedule.pop(key)
    assert table == schedule
    assert module.job('mmmu_modes')['method'] == 'modes'
