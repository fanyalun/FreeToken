"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def test_verify_small_cache_falls_back_to_per_token_fetch_cap(monkeypatch):
    from types import SimpleNamespace
    import freetoken.core as core
    from freetoken.core import Batch, Context
    from freetoken.layers.moe import OffloadMoELayer

    cpu_routes, fetched = [], []

    class Executor:
        def decode_submit(self, layer_id, hidden, weights, ids):
            cpu_routes.append(ids.clone())
            return torch.zeros_like(hidden)

        def decode_sync(self, pending):
            return pending

    class Cache:
        cpu_executor = Executor()
        collect_stats = False
        cache_size = 8

        def ensure_experts_hybrid(self, layer_id, ids):
            fetched.append(ids[0, 0].item())
            ids[:, 0] = 0
            ids[:, 1:] = -1

        def copy_missing(self):
            pass

        def bank_views(self):
            return ()

        def alphas_for_slots(self, layer_id):
            return None

    layer = SimpleNamespace(layer_id=0, _expert_gemm=lambda cache, hidden, *args, **kwargs: hidden)
    layer._decode_hybrid = lambda *args: OffloadMoELayer._decode_hybrid(layer, *args)
    ctx = Context(1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    batch = Batch([], "decode")
    batch.rowwise_linear = True
    hidden = torch.arange(8).reshape(4, 2).float()
    ids = torch.arange(12).reshape(4, 3).int()
    with ctx.forward_batch(batch):
        actual = layer._decode_hybrid(Cache(), hidden, torch.ones(4, 3), ids.clone())
    assert fetched == [0, 3, 6, 9]
    assert torch.equal(torch.cat(cpu_routes), torch.tensor([[-1, 1, 2], [-1, 4, 5], [-1, 7, 8], [-1, 10, 11]]))
    torch.testing.assert_close(actual, hidden)


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fraction_gpu_matches_cpu_reference():
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fixed_cap_unchanged():
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("frac", [0.0, 0.415, 1.0])
@pytest.mark.parametrize("cache_size", [32, 64, 4961])
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_window_plan_matches_union_reference_and_weight_bytes(frac, cache_size, rows):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    torch.manual_seed(13)
    experts, top_k = 32, 8
    sources = {name: [alloc_pinned_tensor(experts, 64, dtype=torch.bfloat16).copy_(
        torch.randn(experts, 64)) for _ in range(2)] for name in ("gate_up", "down")}
    def make():
        cache = OffloadMoeCache(2, experts, cache_size, torch.device("cuda"),
                                decode_target="hybrid", hybrid_max_fetch=1,
                                hybrid_fetch_fraction=frac)
        cache.set_bank_sources(sources)
        cache.collect_stats = True
        cache.collect_decode_freq = True
        return cache
    window, sequential = make(), make()
    sequential.hybrid_max_fetch *= rows
    for step in range(12):
        layer = step % 2
        ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(rows)]).int().cuda()
        expected = ids.clone()
        plans = []
        sequential.ensure_experts_hybrid(layer, expected.view(-1))
        sequential.record_decode_stats_hybrid(layer)
        count = sequential.num_indices.item()
        plans.append((sequential.evict_slots[:count].clone(), sequential.src_indices[:count].clone()))
        sequential.copy_missing()
        actual = ids.clone()
        window.ensure_experts_hybrid_window(layer, actual)
        window.copy_missing()
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)
        for name in ("slot_for_id", "id_of_slot", "usage", "step", "expert_recency",
                     "decode_freq", "active_mask", "num_missing_full", "stat_missing",
                     "stat_fetched", "stat_active", "stat_calls", "stat_missing_layer",
                     "stat_fetched_layer", "stat_active_layer", "stat_steps_layer"):
            assert torch.equal(getattr(window, name), getattr(sequential, name)), name
        count = window.num_indices.item()
        assert count == sum(p[0].numel() for p in plans)
        assert torch.equal(window._pending_copy_plan[0][:count], torch.cat([p[0] for p in plans]))
        assert torch.equal(window._pending_copy_plan[1][:count], torch.cat([p[1] for p in plans]))
        for name in sources:
            assert torch.equal(window.bank_caches[name], sequential.bank_caches[name])
        for row, raw in zip(actual, ids):
            live = row >= 0
            assert torch.equal(window.id_of_slot[row[live].long()], layer * experts + raw[live])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_window_plan_graph_replay_tracks_new_routes():
    from freetoken.kernel.pinned import alloc_pinned_tensor

    cache = OffloadMoeCache(1, 32, 32, torch.device("cuda"), decode_target="hybrid", hybrid_max_fetch=1)
    source = alloc_pinned_tensor(32, 64, dtype=torch.bfloat16).copy_(torch.randn(32, 64))
    cache.set_bank_sources({"gate_up": [source], "down": [source]})
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ids = torch.empty(4, 8, dtype=torch.int32, device="cuda")
        ids.copy_(torch.arange(32, device="cuda").reshape(4, 8))
        cache.ensure_experts_hybrid_window(0, ids)
        cache.copy_missing()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cache.ensure_experts_hybrid_window(0, ids)
            cache.copy_missing()
        for seed in range(8):
            torch.manual_seed(seed)
            raw = torch.stack([torch.randperm(32)[:8] for _ in range(4)]).int().cuda()
            ids.copy_(raw)
            graph.replay()
            stream.synchronize()
            for slots, experts in zip(ids, raw):
                live = slots >= 0
                assert torch.equal(cache.id_of_slot[slots[live].long()], experts[live])
                assert torch.equal(cache.bank_caches["gate_up"][slots[live].long()].cpu(), source[experts[live].long().cpu()])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_window_scratch_is_shared_across_layers_and_matches_budget():
    cache = OffloadMoeCache(2, 32, 32, torch.device("cuda"), decode_target="hybrid", hybrid_max_fetch=1)
    for width in (2, 3, 4):
        ids = torch.arange(width * 8, dtype=torch.int32, device="cuda").reshape(width, 8)
        cache.ensure_experts_hybrid_window(0, ids.clone())
        pointers = tuple(t.data_ptr() for t in cache._hybrid_window_plans[(width, 8)])
        cache.ensure_experts_hybrid_window(1, ids.clone())
        assert pointers == tuple(t.data_ptr() for t in cache._hybrid_window_plans[(width, 8)])
    assert sum(t.numel() * t.element_size() for plan in cache._hybrid_window_plans.values() for t in plan) == 624


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rows", [2, 3, 4])
@pytest.mark.parametrize("grouped", [False, True])
def test_hybrid_window_matches_union_partials(monkeypatch, rows, grouped):
    from types import SimpleNamespace
    import freetoken.core as core
    import freetoken.layers.moe as moe
    from freetoken.core import Batch, Context
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl
    from freetoken.kernel.pinned import alloc_pinned_tensor
    import freetoken.moe.fused as fused

    monkeypatch.setattr(fused, "_GROUPED_DECODE", grouped)

    torch.manual_seed(rows)
    experts, hidden_size, intermediate, top_k = 32, 512, 256, 8
    sources = {
        "gate_up": [alloc_pinned_tensor(experts, 2 * intermediate, hidden_size, dtype=torch.bfloat16).copy_(
            torch.randn(experts, 2 * intermediate, hidden_size) * 0.03)],
        "down": [alloc_pinned_tensor(experts, hidden_size, intermediate, dtype=torch.bfloat16).copy_(
            torch.randn(experts, hidden_size, intermediate) * 0.03)],
    }
    caches = []
    for _ in range(2):
        cache = OffloadMoeCache(1, experts, experts, torch.device("cuda"), decode_target="hybrid", hybrid_max_fetch=1)
        cache.set_bank_sources(sources)
        cache.cpu_executor = CpuMoeExecutor(cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
                                           num_threads=4, max_tokens=rows, device=torch.device("cuda"))
        caches.append(cache)
    ctx = Context(1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    batch = Batch([], "decode")
    batch.rowwise_linear = True
    def gemm(cache, hidden, weights, ids, **kwargs):
        return fused_experts_decode_impl(hidden, *kwargs["views"], weights, ids)
    layer = SimpleNamespace(layer_id=0, _expert_gemm=gemm)
    layer._decode_hybrid = lambda *args: moe.OffloadMoELayer._decode_hybrid(layer, *args)
    caches[0].hybrid_max_fetch *= rows
    try:
        with ctx.forward_batch(batch):
            for _ in range(6):
                hidden = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
                ids = torch.stack([torch.randperm(experts, device="cuda")[:top_k] for _ in range(rows)]).int()
                weights = torch.rand(rows, top_k, device="cuda")
                reference = caches[0]
                slots = ids.clone()
                reference.ensure_experts_hybrid(0, slots.view(-1))
                on_gpu = slots >= 0
                cpu_ids = torch.where(on_gpu, -1, ids).contiguous()
                cpu = reference.cpu_executor.decode(0, hidden, weights, cpu_ids)
                reference.copy_missing()
                gpu_weights = torch.where(on_gpu, weights, 0).contiguous()
                gpu = fused_experts_decode_impl(hidden, *reference.bank_views(), gpu_weights, slots.clamp_min(0))
                expected = gpu + cpu
                monkeypatch.setattr(moe, "_HYBRID_WINDOW", True)
                actual = layer._decode_hybrid(caches[1], hidden, weights, ids.clone())
                torch.cuda.synchronize()
                assert torch.equal(actual, expected)
    finally:
        torch.cuda.synchronize()
        for cache in caches:
            cache.cpu_executor.close()
