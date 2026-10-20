import os

import pytest
import torch

from freetoken.moe.cpu_target import CpuTargetExperts


@pytest.mark.parametrize("rows", range(1, 34))
@pytest.mark.parametrize("threads", [1, 2, 4])
def test_grouped_experts_equal_original_row_dots(rows, threads):
    from freetoken.kernel import _cpu_moe

    torch.manual_seed(42)
    cores = sorted(os.sched_getaffinity(0))[:threads]
    gate_up = [torch.randn(16, 64, 64).bfloat16() * 0.1]
    down = [torch.randn(16, 64, 32).bfloat16() * 0.1]
    serial = CpuTargetExperts(gate_up, down, 8, cores, rows, grouped=False, backend="legacy")
    grouped = CpuTargetExperts(gate_up, down, 8, cores, rows, grouped=True)
    try:
        x = torch.randn(rows, 64).bfloat16()
        ids = torch.stack([torch.randperm(16)[:8] for _ in range(rows)])
        weights = torch.rand(rows, 8)
        expected = torch.cat([serial(0, x[i:i + 1], weights[i:i + 1], ids[i:i + 1]) for i in range(rows)])
        assert torch.equal(grouped(0, x, weights, ids).view(torch.int16), expected.view(torch.int16))
        ids[:, 3] = -1
        ids[:, 4] = 16
        weights[:, 2] = 0
        assert torch.equal(grouped(0, x, weights, ids).view(torch.int16),
                           serial(0, x, weights, ids).view(torch.int16))
        ids[:] = torch.arange(8)
        assert torch.equal(grouped(0, x, weights, ids).view(torch.int16),
                           serial(0, x, weights, ids).view(torch.int16))
        assert len(grouped.tasks) == rows
        assert len({entry[0].untyped_storage().data_ptr() for entry in grouped.io.values()}) == 1
    finally:
        serial.close()
        grouped.close()


def test_cpu_only_expert_entry_does_not_initialize_cuda():
    import subprocess
    import sys

    code = """
import os
import torch
from freetoken.moe.cpu_target import CpuTargetExperts
assert not torch.cuda.is_initialized()
cores = sorted(os.sched_getaffinity(0))[:2]
ex = CpuTargetExperts([torch.zeros(8, 32, 32, dtype=torch.bfloat16)],
    [torch.zeros(8, 32, 16, dtype=torch.bfloat16)], 8, cores, 2)
ex(0, torch.ones(2, 32, dtype=torch.bfloat16), torch.ones(2, 8), torch.arange(8).repeat(2, 1))
ex.close()
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("rows", range(1, 34))
@pytest.mark.parametrize("k", [7, 16, 35, 129])
def test_projection_width_threads_and_argmax_are_exact(rows, k):
    from freetoken.kernel import _cpu_moe

    torch.manual_seed(17)
    x = torch.randn(rows, k).bfloat16()
    weight = torch.randn(517, k)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected = torch.cat([_cpu_moe.target_linear(row[None], weight, False) for row in x])
        for threads in (1, 2, 4):
            torch.set_num_threads(threads)
            actual = _cpu_moe.target_linear(x, weight, False)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(_cpu_moe.target_linear(x, weight, True), expected.argmax(-1))
        tied = torch.zeros_like(weight)
        assert _cpu_moe.target_linear(x, tied, True).eq(0).all()
    finally:
        torch.set_num_threads(previous)


def test_backend_rejects_ungrouped_optimized_before_allocation():
    with pytest.raises(ValueError, match="ungrouped"):
        CpuTargetExperts([], [], 8, [0], 33, grouped=False)


def test_native_parallel_loop_uses_requested_threads():
    from freetoken.kernel import _cpu_moe

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        assert _cpu_moe.target_parallel_threads() == 4
    finally:
        torch.set_num_threads(previous)
