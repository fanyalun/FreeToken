"""Hybrid decode's bandwidth-matched fetch split.

Covers the profile reader of --moe-hybrid-max-fetch auto, which turns the `ft bench bw`
row of each MoE layer kind into a fetch fraction.
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import PROFILE_VERSION, default_profile_path, fetch_fraction, layer_kind_key, load_layer_kinds, locked_fetch_fraction


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rows,valid,fraction,cap,preverify,open_slots", [
    (1, 1, .25, 1, False, 16),
    (4, 4, .413, 1, True, 16),
    (17, 11, .413, 1, False, 16),
    (4, 2, 0., 1, True, 16),
    (4, 4, 0., 0, False, 16),
    (17, 17, 1., 1, False, 16),
    (4, 4, .5, 1, False, 2),
    (4, 0, .5, 1, True, 16),
])
def test_three_level_bandwidth_partition_matches_cpu_oracle(rows, valid, fraction, cap, preverify, open_slots):
    import triton
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.moe.speculative_hybrid import _partition

    experts, slots, top_k = 64, 64, 4
    cache = OffloadMoeCache(num_layers=1, num_experts=experts, cache_size=slots,
                           device=torch.device("cuda"), decode_target="hybrid")
    owners, usage = cache.pool_state(0)
    owners[:2] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cache.slot_for_id[0, :2] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    host_ids = torch.arange(rows * top_k, dtype=torch.int32).reshape(rows, top_k) % experts
    host_weights = torch.full((rows, top_k), .25)
    if preverify:
        host_weights[:, 1:] = 0
    active = (torch.arange(rows)[:, None] < valid) & (host_weights != 0)
    counts = torch.bincount(host_ids[active].long(), minlength=experts)
    missing = sorted((e for e in range(2, experts) if counts[e]), key=lambda e: (-int(counts[e]), e))
    fixed = round(fraction * 65536)
    if fixed:
        target = len(missing) * fixed / 65536
        candidates = {int(target), min(len(missing), int(target) + 1)}
        chosen = min(candidates, key=lambda q: (max(q * (1-fixed/65536), (len(missing)-q) * fixed/65536), q))
    else:
        chosen = cap
    available = open_slots - sum(bool(counts[e]) for e in (0, 1))
    chosen = min(chosen, len(missing), available)
    fetched = set(missing[:chosen])
    expected_gpu = active & torch.tensor([[int(e) < 2 or int(e) in fetched for e in row] for row in host_ids])
    ids, weights = host_ids.cuda(), host_weights.cuda()
    cpu_ids, cpu_weights, gpu_slots = torch.empty_like(ids), torch.empty_like(weights), torch.empty_like(ids)
    refs = torch.zeros(experts, dtype=torch.int32, device="cuda")
    pending = torch.zeros(slots, dtype=torch.int32, device="cuda")
    pending[open_slots:] = 1
    demand = counts.to(device="cuda", dtype=torch.int32)
    plan = [torch.empty(experts, dtype=torch.int32, device="cuda") for _ in range(3)]
    plan_count = torch.zeros(1, dtype=torch.int64, device="cuda")
    stats = torch.zeros(32, dtype=torch.int64, device="cuda")
    valid_rows = torch.tensor([valid], dtype=torch.int32, device="cuda")
    _partition[(1,)](cache.slot_for_id, owners, usage, cache.step, refs, pending,
        ids, weights, cpu_ids, cpu_weights, gpu_slots, demand, *plan, plan_count, stats,
        0, experts, slots, rows, top_k, rows*top_k, cap, fixed, 0, experts, slots,
        triton.next_power_of_2(rows*top_k), preverify, valid_rows)
    assert int(plan_count) == chosen and set(plan[2][:chosen].cpu().tolist()) == fetched
    assert torch.equal(weights.cpu(), torch.where(expected_gpu, host_weights, 0))
    assert torch.equal(cpu_weights.cpu(), torch.where(active & ~expected_gpu, host_weights, 0))
    assert torch.equal(cpu_ids.cpu(), torch.where(active & ~expected_gpu, host_ids, -1))
    assert not bool((weights.cpu() * cpu_weights.cpu()).any())
    assert torch.equal((weights + cpu_weights).cpu(), torch.where(active, host_weights, 0))


def test_layer_kind_rows_give_fetch_fractions(tmp_path):
    bf16 = layer_kind_key("bf16", {}, 2048, 768, 8, 12)
    gguf = layer_kind_key("gguf", {"gate_up_type": 12, "down_type": 14}, 2048, 768, 8, 12)
    assert gguf == "gguf[down_type=14,gate_up_type=12]:2048x768:top8:t12"
    prof = {
        "version": PROFILE_VERSION,
        "gpu": {"name": "FAKE GPU"},
        "layer_kinds": {
            bf16: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            gguf: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0, "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    rows = load_layer_kinds(path=str(path))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert fetch_fraction(rows[bf16]) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert fetch_fraction(rows[gguf]) == pytest.approx(0.25)
    # a locked layer's fetch adds a memcpy at the CPU MoE rate before the gather: p / (2p + c), from the same pair
    assert locked_fetch_fraction(rows[bf16]) == pytest.approx(40 / 180)
    assert locked_fetch_fraction(rows[gguf]) == pytest.approx(30 / 150)
    # another thread count is another kind
    assert layer_kind_key("bf16", {}, 2048, 768, 8, 6) not in rows
    # a profile from different hardware is ignored
    assert load_layer_kinds(gpu_name="OTHER", path=str(path)) is None
    # so is one of another version
    path.write_text(json.dumps({**prof, "version": PROFILE_VERSION - 1}))
    assert load_layer_kinds(path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, row):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"version": PROFILE_VERSION, "gpu": {"name": name}, "layer_kinds": {"k": row}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "legacy")
    assert load_layer_kinds(gpu_name="FAKE GPU", gpu_uuid=uuid) == {"k": "legacy"}
    assert load_layer_kinds(gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "card")
    assert load_layer_kinds(gpu_name="FAKE GPU", gpu_uuid=uuid) == {"k": "card"}


@pytest.mark.parametrize("hybrid", [False, True])
def test_prefetch_budget_fits_the_measured_copy_window(hybrid):
    from freetoken.moe.bench_profile import prefetch_rates, prefetch_count

    row = dict(top_k=8, pcie_gather_gbs=24.0, pcie_gather_overlap_gbs=16.0,
               cpu_moe_gbs=64.0, cpu_moe_overlap_gbs=32.0,
               pcie_copy_launch_us=4.0, pcie_copy_expert_us=200.0,
               gpu_moe=dict(version=1, rows=[1, 16], us=[28.0, 148.0]))
    size = 6 * 2**20
    copy, launch, fixed, route, cpu = prefetch_rates(row, size, hybrid=hybrid)
    assert fixed + 8 * route == pytest.approx(28.0)
    assert fixed + 16 * 8 * route == pytest.approx(148.0)
    assert copy == pytest.approx(size / ((16.0 if hybrid else 24.0) * 1000))
    assert cpu == pytest.approx(size / 32000 if hybrid else 0.0)
    for window in (0.0, 3.0, 800.0, 3200.0):
        count = prefetch_count(window, copy, launch, 256)
        if count:
            assert launch + count * copy <= window
        if window >= launch:
            assert count == 256 or launch + (count + 1) * copy > window
    assert prefetch_count(3200.0, copy, launch, 2) == 2
    assert prefetch_count(3200.0, copy, launch, 0) == 0
    assert prefetch_rates({k: v for k, v in row.items() if k != "gpu_moe"}, size) is None
    assert prefetch_rates({**row, "pcie_gather_gbs": 0}, size) is None


@pytest.mark.parametrize("cpu,queued", [(False, 0), (True, 0), (True, 3)])
@pytest.mark.parametrize("graph", [False, True])
def test_prefetch_device_budget_accounts_for_the_remaining_cpu_window(cpu, queued, graph):
    import torch
    from freetoken.moe.speculative_gpu import _prefetch_budget
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    weights = torch.tensor([0.5, 0, 0.3, 0, 0, 0.2, 0, 0], device="cuda")
    ids = torch.tensor([-1, 2, -1, 2, 3, -1, 4, 5], dtype=torch.int32, device="cuda") if cpu else None
    copies = torch.tensor([queued], dtype=torch.int32, device="cuda")
    budget = torch.zeros(1, dtype=torch.int32, device="cuda")
    launch = lambda: _prefetch_budget[(1,)](weights, ids, copies, budget,
        8, 8, 8, 8, 10.0, 2.0, 5.0, 4.0, 20.0, 10.0, 8)
    launch()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            launch()
        captured.replay()
    gpu_time = 5 + 3 * 4
    cpu_time = max(0, 4 * 20 - queued * 10 - 2) if cpu else 0
    count = budget.item()
    assert count * 10 + 2 <= max(gpu_time, cpu_time)
    assert count == 8 or (count + 1) * 10 + 2 > max(gpu_time, cpu_time)


@pytest.mark.parametrize("rows,fraction,cap,capacity", [
    (1, 0.5, 0, 6), (5, 0.398, 1, 6), (17, 0.398, 1, 20), (68, 0.398, 1, 6),
    (68, 1.0, 1, 6), (5, 0.0, 0, 6), (5, 0.0, 1, 6), (17, 0.5, 1, 6),
])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("ragged", [False, True])
def test_speculative_hybrid_bandwidth_partition_preserves_routes(rows, fraction, cap, capacity, graph, ragged):
    import torch
    import triton

    from freetoken.moe.speculative_hybrid import _partition

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    experts, topk, layer = 16, 4, 1
    valid_count = max(1, rows - 1)
    host_ids = torch.arange(rows * topk).reshape(rows, topk).remainder(experts).int()
    host_weights = torch.linspace(0.1, 0.9, rows * topk).reshape(rows, topk)
    host_active = torch.arange(rows) < valid_count
    if ragged:
        host_active &= torch.arange(rows) % 3 != 1
    counts = torch.bincount(host_ids[host_active].flatten().long(), minlength=experts)
    hits = {0, 1} & set(host_ids[host_active].flatten().tolist())
    missing = sorted((e for e in range(experts) if counts[e] and e not in hits), key=lambda e: (-int(counts[e]), e))
    available = capacity - len(hits) - 1
    if fraction:
        f = round(fraction * 65536) / 65536
        fetch = min(range(len(missing) + 1), key=lambda q: max(q * (1 - f), (len(missing) - q) * f))
    else:
        fetch = cap
    expected_fetch = min(fetch, len(missing), available)
    expected_gpu = hits | set(missing[:expected_fetch])
    host_mapping = torch.full((2 * experts,), -1, dtype=torch.int32)
    host_mapping[experts:experts + 2] = torch.arange(2, dtype=torch.int32)
    host_mapping[2] = 2
    host_owners = torch.full((capacity,), -1, dtype=torch.int32)
    host_owners[:3] = torch.tensor([experts, experts + 1, 2])
    host_pending = torch.zeros(capacity, dtype=torch.int32)
    host_pending[-1] = 1
    mapping, owners, pending = (t.cuda() for t in (host_mapping, host_owners, host_pending))
    usage = torch.zeros(capacity, dtype=torch.int64, device="cuda")
    clock = torch.zeros(1, dtype=torch.int64, device="cuda")
    refs = torch.zeros(2 * experts, dtype=torch.int32, device="cuda")
    refs[2] = 1
    ids, weights = host_ids.cuda(), host_weights.cuda()
    cpu_ids, cpu_weights = torch.empty_like(ids), torch.empty_like(weights)
    gpu_slots = torch.empty_like(ids)
    demand = counts.int().cuda()
    plans = [torch.empty(experts, dtype=torch.int32, device="cuda") for _ in range(3)]
    plan_count = torch.zeros(1, dtype=torch.int32, device="cuda")
    stats = torch.zeros(31, dtype=torch.int64, device="cuda")
    valid_rows = torch.tensor([valid_count], dtype=torch.int32, device="cuda")
    conflict = torch.zeros_like(plan_count)
    row_mask = host_active.cuda() if ragged else None
    launch = lambda: _partition[(1,)](mapping, owners, usage, clock, refs, pending, ids, weights,
        cpu_ids, cpu_weights, gpu_slots, demand, *plans, plan_count, stats,
        layer, experts, capacity, rows, topk, rows * topk, cap, round(fraction * 65536), 0,
        experts, triton.next_power_of_2(capacity), triton.next_power_of_2(rows * topk),
        True, valid_rows, conflict, row_mask)
    launch()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            launch()
    mapping.copy_(host_mapping)
    owners.copy_(host_owners)
    pending.copy_(host_pending)
    demand.copy_(counts.int())
    weights.copy_(host_weights)
    stats.zero_()
    conflict.zero_()
    captured.replay() if graph else launch()
    assert plan_count.item() == expected_fetch
    assert plans[2][:expected_fetch].tolist() == missing[:expected_fetch]
    assert owners[-1].item() == host_owners[-1].item() and pending[-1].item() == 1
    active = host_active[:, None]
    gpu = active & torch.tensor([[e in expected_gpu for e in row] for row in host_ids.tolist()])
    cpu = active & ~gpu
    torch.testing.assert_close(weights.cpu(), torch.where(gpu, host_weights, 0))
    torch.testing.assert_close(cpu_weights.cpu(), torch.where(cpu, host_weights, 0))
    torch.testing.assert_close(cpu_ids.cpu(), torch.where(cpu, host_ids, -1))
    resolved = mapping[experts:][host_ids.cuda().long()].cpu()
    torch.testing.assert_close(gpu_slots.cpu()[gpu], resolved[gpu])
