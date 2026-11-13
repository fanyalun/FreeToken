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
    frac = torch.tensor([fixed], dtype=torch.int32, device="cuda")
    valid_rows = torch.tensor([valid], dtype=torch.int32, device="cuda")
    _partition[(1,)](cache.slot_for_id, owners, usage, cache.step, refs, pending,
        ids, weights, cpu_ids, cpu_weights, gpu_slots, demand, *plan, plan_count, frac, stats,
        0, experts, slots, rows, top_k, rows*top_k, cap, experts, slots,
        triton.next_power_of_2(rows*top_k), preverify, 0, valid_rows)
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
