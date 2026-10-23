import os

import pytest
import torch

from freetoken.moe.cpu_target import CpuTargetExperts


@pytest.mark.parametrize("rows", range(1, 34))
@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("schedule", ["tile", "expert"])
def test_grouped_experts_equal_original_row_dots(rows, threads, schedule):
    from freetoken.kernel import _cpu_moe

    torch.manual_seed(42)
    cores = sorted(os.sched_getaffinity(0))[:threads]
    gate_up = [torch.randn(16, 64, 64).bfloat16() * 0.1]
    down = [torch.randn(16, 64, 32).bfloat16() * 0.1]
    serial = CpuTargetExperts(gate_up, down, 8, cores, rows, grouped=False, backend="legacy")
    grouped = CpuTargetExperts(gate_up, down, 8, cores, rows, grouped=True, schedule=schedule)
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


@pytest.mark.parametrize("hidden,inter", [(7, 15), (65, 17), (129, 67)])
def test_reused_expert_inputs_preserve_tails_and_repeated_calls(hidden, inter):
    torch.manual_seed(142)
    cores = sorted(os.sched_getaffinity(0))[:2]
    gate_up = [torch.randn(16, 2 * inter, hidden).bfloat16() * 0.1]
    down = [torch.randn(16, hidden, inter).bfloat16() * 0.1]
    serial = CpuTargetExperts(gate_up, down, 8, cores, 7, grouped=False, backend="legacy")
    reused = CpuTargetExperts(gate_up, down, 8, cores, 7, schedule="expert")
    try:
        for rows in (7, 1, 3, 7):
            x = torch.randn(rows, hidden).bfloat16()
            ids = torch.arange(8).repeat(rows, 1)
            weights = torch.rand(rows, 8)
            expected = serial(0, x, weights, ids)
            assert torch.equal(reused(0, x, weights, ids).view(torch.int16), expected.view(torch.int16))
    finally:
        serial.close()
        reused.close()


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


@pytest.mark.parametrize("options", [dict(schedule="unknown"),
    dict(schedule="expert", backend="legacy"), dict(schedule="expert", packed=True)])
def test_expert_schedule_rejects_unsupported_modes_before_allocation(options):
    with pytest.raises(ValueError, match="expert schedule"):
        CpuTargetExperts([], [], 8, [0], 33, **options)


@pytest.mark.parametrize("rows", [1, 7, 33])
def test_numa_expert_queues_cover_both_and_empty_halves(rows):
    from pathlib import Path

    nodes = {}
    for core in sorted(os.sched_getaffinity(0)):
        for path in Path(f"/sys/devices/system/cpu/cpu{core}").glob("node[0-9]*"):
            nodes.setdefault(path.name, core)
    if len(nodes) < 2:
        pytest.skip("two NUMA nodes required")
    cores = [nodes[node] for node in sorted(nodes)[:2]]
    torch.manual_seed(62)
    gate_up = [torch.randn(16, 64, 64).bfloat16() * 0.1]
    down = [torch.randn(16, 64, 32).bfloat16() * 0.1]
    old = CpuTargetExperts(gate_up, down, 8, cores, rows)
    new = CpuTargetExperts(gate_up, down, 8, cores, rows, schedule="expert_numa")
    try:
        x = torch.randn(rows, 64).bfloat16()
        weights = torch.rand(rows, 8)
        for start in (0, 8, 4, 0):
            ids = torch.arange(start, start + 8).repeat(rows, 1)
            assert torch.equal(old(0, x, weights, ids).view(torch.int16),
                               new(0, x, weights, ids).view(torch.int16))
    finally:
        old.close()
        new.close()


def test_native_parallel_loop_uses_requested_threads():
    from freetoken.kernel import _cpu_moe

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        assert _cpu_moe.target_parallel_threads() == 4
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("outputs", [1, 31, 256, 511, 1025])
def test_projection_partition_keeps_first_tie_and_nan(outputs):
    from freetoken.kernel import _cpu_moe

    x = torch.ones(17, 35, dtype=torch.bfloat16)
    weight = torch.zeros(outputs, 35)
    previous = torch.get_num_threads()
    try:
        for value in (1.0, float("inf"), float("nan")):
            weight.zero_()
            first = outputs // 3
            weight[first, 0] = value
            weight[-1, 0] = value
            for threads in (1, 2, 4):
                torch.set_num_threads(threads)
                logits = _cpu_moe.target_linear(x, weight, False)
                prediction = _cpu_moe.target_linear(x, weight, True)
                assert prediction.eq(first).all()
                assert torch.equal(prediction, logits.argmax(-1))
    finally:
        torch.set_num_threads(previous)
