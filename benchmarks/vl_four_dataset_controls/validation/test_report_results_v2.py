import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('report_results', SOURCE / 'report_results.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def test_native_score_binding_visual_units_and_auxiliary_bounds(self):
        with tempfile.TemporaryDirectory(dir=SOURCE / 'validation', prefix='report_fixture_') as name:
            root = Path(name)

            def save(relative, data):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))

            score = root / 'cells/MMLongBench_DOC/fused/score_local_fused_native_512/score.json'
            save(score.relative_to(root), {'accuracy_pct': 25.0})
            save('cells/MMLongBench_DOC/fused/complete.json', {'complete': True, 'score_file': str(score)})
            save('cells/Vision2Web_Level1/fused/complete.json', {'complete': True, 'visual_score_pct': 12.5})
            save('cells/Vision2Web_Level1/tablemoe/progress.json', {'completed': 3, 'total': 40})
            bounded = 'cells/MMMU_DEV_VAL/fused/score_local_fused_native_bounded_2048/'
            save(bounded + 'traversal_complete.json', {'complete': True, 'traversal_complete': True,
                                                      'strict_primary_complete': False})
            save(bounded + 'score.json', {'known_count': 419, 'unknown': 1, 'pending': 0,
                                         'full_denominator_accuracy_lower_pct': 40.0,
                                         'full_denominator_accuracy_upper_pct': 40.24})
            doc_bounded = 'cells/MMLongBench_DOC/tablemoe/score_local_fused_native_bounded_512/'
            save(doc_bounded + 'traversal_complete.json', {'complete': True, 'traversal_complete': True,
                                                          'strict_primary_complete': False})
            save(doc_bounded + 'score.json', {'known_count': 435, 'unknown': 1, 'pending': 0,
                                             'full_denominator_accuracy_lower_pct': 12.0,
                                             'full_denominator_accuracy_upper_pct': 12.23})
            previous = report.ROOT
            try:
                report.ROOT = root
                report.main()
            finally:
                report.ROOT = previous
            summary = json.loads((root / 'summary.json').read_text())
            rows = {(row['dataset'], row['method']): row for row in summary['rows']}
            self.assertEqual(rows['MMLongBench_DOC', 'fused']['accuracy_pct'], 25.0)
            self.assertEqual(rows['Vision2Web_Level1', 'fused']['visual_score_pct'], 12.5)
            self.assertEqual(rows['Vision2Web_Level1', 'tablemoe']['completed_samples'], 3)
            mm = rows['MMMU_DEV_VAL', 'fused']
            self.assertFalse(mm['scoring_complete'])
            self.assertIsNone(mm['accuracy_pct'])
            self.assertTrue(mm['bounded_judging_complete'])
            self.assertEqual(mm['bounded_known_count'], 419)
            self.assertEqual(mm['bounded_unknown'], 1)
            self.assertEqual(mm['accuracy_lower_pct'], 40.0)
            self.assertEqual(mm['accuracy_upper_pct'], 40.24)
            self.assertEqual(summary['completed_scored_cells'], 2)
            doc = rows['MMLongBench_DOC', 'tablemoe']
            self.assertFalse(doc['scoring_complete'])
            self.assertIsNone(doc['accuracy_pct'])
            self.assertEqual(doc['accuracy_lower_pct'], 12.0)
            self.assertEqual(doc['accuracy_upper_pct'], 12.23)
            self.assertEqual(summary['completed_bounded_judging_cells'], 2)
            self.assertFalse(summary['complete'])


if __name__ == '__main__':
    unittest.main()
