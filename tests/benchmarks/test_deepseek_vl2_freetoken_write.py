from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from bench_deepseek_vl2_freetoken_write import METHODS, policy_options


EXPECTED = {
    "baseline": dict(tablemoe_prefill_keep_rate=1., tablemoe_decode_keep_k=6),
    "tablemoe_06_top3": dict(tablemoe_prefill_keep_rate=.6, tablemoe_decode_keep_k=3),
    "tablemoe_07_top4": dict(tablemoe_prefill_keep_rate=.7, tablemoe_decode_keep_k=4),
    "tablemoe_06_top3_512_top4": dict(tablemoe_prefill_keep_rate=.6, tablemoe_decode_keep_k=3,
                                   tablemoe_decode_switch_after=512, tablemoe_decode_late_top_h=4),
    "adapmoe_gating_top4": dict(tablemoe_prefill_keep_rate=1., tablemoe_decode_keep_k=4,
                               deepseek_vl2_router_top_k=4),
}


@pytest.mark.parametrize("method", EXPECTED)
def test_deepseek_top6_comparison_uses_requested_policy(method):
    assert policy_options(method) == EXPECTED[method]


def test_deepseek_matrix_contains_exactly_the_requested_five_methods():
    assert METHODS == tuple(EXPECTED)
