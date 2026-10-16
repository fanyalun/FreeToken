import importlib.util
import sys
from pathlib import Path

import torch
import pytest


def load():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    spec = importlib.util.spec_from_file_location("sparse_verify_motivation", root / "benchmarks/bench_sparse_verify_motivation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_rank_and_all_hits_are_preserved():
    module = load()
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    hits = torch.tensor([[True, False, True, False, False, False, False, False]])
    keep = module.retained_mask(weights, hits)
    assert keep.tolist() == [[True, False, True, False, True, True, True, True]]


def test_correction_is_proposed_but_not_accepted_without_target():
    module = load()
    assert module.candidate_tokens([9, 10, 11, 12], [10, 99, 88, 77], 3) == ([10, 99], 1)
    assert module.accepted_prefix([10, 99], [10, 11, 12]) == 1
    assert module.candidate_tokens([9, 10, 11, 12], [10, 11, 12, 13], 3) == ([10, 11, 12], 3)
    assert module.candidate_tokens([9, 10, 11, 12], [99, 11, 12, 13], 3) == ([99], 0)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("rows", (1, 2, 3, 4))
def test_equal_weights_preserve_original_route_order(device, rows):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = load()
    weights = torch.tensor([[0.20, 0.17, 0.12, 0.10, 0.10, 0.10, 0.09, 0.08]], device=device).repeat(rows, 1)
    hits = torch.tensor([[True, True, True, False, True, False, False, True]], device=device).repeat(rows, 1)
    assert module.retained_mask(weights, hits).tolist() == [[True, True, True, True, True, False, False, True]] * rows


def test_deadline_queue_and_prefetch_pollution():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("sparse_analysis", root / "benchmarks/analyze_sparse_verify_motivation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.schedule_copy(0, 2, 0, [(1, 3), (4, 5)]) == 7
    assert module.schedule_copy(0, 1, 0, [(1, 3)]) == 1
    baseline = module.lru_replay([1, 2], [1, 2], [3], [1, 2], 2, False)
    predicted = module.lru_replay([1, 2], [1, 2], [3], [1, 2], 2, True)
    assert baseline["demand_misses"] == 0
    assert predicted["demand_misses"] == 2
    assert predicted["unused_prefetches"] == 1
