import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from bench_speculative_hybrid import warmup_runner


@pytest.mark.parametrize("sd", [False, True])
def test_warmup_covers_nonterminal_sd_reconcile_widths_and_restores_depth(sd):
    class Runner:
        draft_width = 4
        def __init__(self):
            self.calls = []
        def generate(self, prompt, *, max_tokens):
            self.calls.append((self.draft_width, max_tokens))
    runner = Runner()
    warmup_runner(runner, [1, 2], 256, sd=sd)
    expected = [(4, cap) for cap in (2, 3, 4, 5, 256)]
    if sd:
        expected = [(width, 6) for width in range(1, 5)] + expected
        assert all(cap - 1 > width for width, cap in runner.calls[:4])
    assert runner.calls == expected
    assert runner.draft_width == 4


def test_warmup_restores_sd_depth_on_failure():
    class Runner:
        draft_width = 4
        def generate(self, prompt, *, max_tokens):
            raise RuntimeError("warmup failed")
    runner = Runner()
    with pytest.raises(RuntimeError, match="warmup failed"):
        warmup_runner(runner, [1], 256, sd=True)
    assert runner.draft_width == 4
