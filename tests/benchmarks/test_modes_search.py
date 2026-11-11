from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from modes_search import validate_full_target


def test_full_validation_keeps_measured_candidate_within_target_tolerance():
    measurements = iter([
        (.006230660957802077, .9798201190035383),
        (.00018617259629157465, .9262490807310952),
        (2.1159369074060425e-5, .39147034168518063),
        (4.320208217241846e-6, .23158874737719082),
        (1.5789757578469715e-5, .18681954815683371),
        (2.2675708528652194e-5, .2058911637044351),
    ])
    evaluated = []

    def evaluate(text, visual):
        evaluated.append(dict(text=text, visual=visual))
        return next(measurements)

    result = validate_full_target(dict(text=.001, visual=.002), dict(text=1., visual=1.),
                                  evaluate, .2, max_evaluations=8, tolerance=.01)

    assert len(result['full_validation']) == 6
    assert result['target_tolerance_met']
    assert result['tau'] == evaluated[-1]
    assert result['skip_proportion'] == pytest.approx(.2058911637044351)
    assert result['best_val'] == pytest.approx(2.2675708528652194e-5)


def test_full_validation_reports_outside_tolerance_when_no_qualified_candidate_exists():
    measurements = iter([(1e-5, .3), (2e-5, .25)])
    result = validate_full_target(dict(text=.001, visual=.002), dict(text=1., visual=1.),
                                  lambda text, visual: next(measurements), .2,
                                  max_evaluations=2, tolerance=.01)

    assert not result['target_tolerance_met']
    assert result['skip_proportion'] == .3
    assert result['best_val'] == 1e-5


@pytest.mark.parametrize('continue_evaluation', [lambda: True, lambda: False])
def test_full_validation_rejects_missing_feasible_measurements(continue_evaluation):
    with pytest.raises(ValueError, match='did not find a feasible threshold'):
        validate_full_target(dict(text=.001, visual=.002), dict(text=1., visual=1.),
                             lambda text, visual: (1., .1), .2, max_evaluations=1,
                             tolerance=.01, continue_evaluation=continue_evaluation)
