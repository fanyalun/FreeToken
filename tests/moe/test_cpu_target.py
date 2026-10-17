import os

import pytest
import torch

from freetoken.moe.cpu_target import CpuTargetExperts


@pytest.mark.parametrize("rows", [1, 4, 9, 33])
def test_grouped_experts_equal_original_row_dots(rows):
    from freetoken.kernel import _cpu_moe

    if not hasattr(_cpu_moe.CpuMoeExecutor, "set_expert_grouped"):
        pytest.skip("requires rebuilt expert grouping extension")
    torch.manual_seed(42)
    cores = sorted(os.sched_getaffinity(0))[:2]
    gate_up = [torch.randn(16, 64, 64).bfloat16() * 0.1]
    down = [torch.randn(16, 64, 32).bfloat16() * 0.1]
    serial = CpuTargetExperts(gate_up, down, 8, cores, rows, grouped=False)
    grouped = CpuTargetExperts(gate_up, down, 8, cores, rows, grouped=True)
    try:
        x = torch.randn(rows, 64).bfloat16()
        ids = torch.stack([torch.randperm(16)[:8] for _ in range(rows)])
        weights = torch.rand(rows, 8)
        expected = torch.cat([serial(0, x[i:i + 1], weights[i:i + 1], ids[i:i + 1]) for i in range(rows)])
        assert torch.equal(grouped(0, x, weights, ids), expected)
        ids[:, 3] = -1
        assert torch.equal(grouped(0, x, weights, ids), serial(0, x, weights, ids))
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
    [torch.zeros(8, 32, 16, dtype=torch.bfloat16)], 8, cores, 2, grouped=False)
ex(0, torch.ones(2, 32, dtype=torch.bfloat16), torch.ones(2, 8), torch.arange(8).repeat(2, 1))
ex.close()
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", code], check=True)
