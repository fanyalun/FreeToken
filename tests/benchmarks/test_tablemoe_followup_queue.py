"""Keep sensitivity reports tied to their requested alpha and baseline."""

import importlib.util
import json
from pathlib import Path


def queue_module():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/tablemoe_sensitivity_controls/run_followup_queue.py'
    spec = importlib.util.spec_from_file_location('tablemoe_followup_queue_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_labels_alpha_and_includes_baseline_while_results_are_pending(tmp_path):
    module = queue_module()
    baseline = tmp_path / 'baseline.json'
    pending = tmp_path / 'pending.json'
    module.save(baseline, dict(complete=True, quality_metric=77.))
    protocol = dict(alpha=.6, decode_top_h=4, expert_slots=3072,
                    comparison_records=dict(tablemoe=str(pending)), baseline_record=str(baseline),
                    comparison_rows=[dict(label='lossless baseline', alpha=None, record=str(baseline)),
                                     dict(label='TableMoE', alpha=.6, record=str(pending))])
    module.report(tmp_path, protocol)
    result = json.loads((tmp_path / 'comparison.json').read_text())
    assert result['alpha'] == .6 and not result['complete']
    assert result['baseline_score'] == 77.
    assert result['comparison_rows'][0]['score'] == 77.
    assert result['comparison_rows'][1]['score'] is None
    assert 'lossless baseline' in (tmp_path / 'comparison.md').read_text()
    module.save(pending, dict(complete=True, quality_metric=75.))
    module.report(tmp_path, protocol)
    result = json.loads((tmp_path / 'comparison.json').read_text())
    assert result['complete'] and result['scores'] == dict(tablemoe=75.)
    assert result['comparison_rows'][1]['difference_from_baseline'] == -2.


def test_legacy_alpha07_queue_keeps_its_report_metadata(tmp_path):
    module = queue_module()
    record = tmp_path / 'score.json'
    module.save(record, dict(complete=True, quality_metric=74.))
    module.report(tmp_path, dict(comparison_records=dict(tablemoe=str(record))))
    result = json.loads((tmp_path / 'comparison.json').read_text())
    assert result['complete'] and result['alpha'] == .7
    assert result['decode_top_h'] == 4 and result['expert_slots'] == 3072
