"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.utils import div_ceil
from .speculative_window import three_level_max_rows

if TYPE_CHECKING:
    import torch


CPU_ASYNC_REPLAY_ROWS = 8


def speculative_split_workspace_bytes(config) -> int:
    prefix = getattr(config, "cpu_async_prefix_layers", None)
    if prefix is None:
        return 0
    mc = config.model_config
    group = mc.linear_attention_group()
    rows = config.cpu_async_loops * (config.speculative_mtp_depth + 1) + 1
    linear = sum(mc.is_linear_layer(i) for i in range(prefix))
    suffix = sum(mc.is_linear_layer(i) for i in range(prefix, mc.num_layers))
    activation = config.dtype.itemsize
    channels = 2 * group.num_key_heads * group.key_head_dim + group.num_value_heads * group.value_head_dim
    history = group.conv_kernel_dim - 1
    cpu_replay = linear * rows * group.num_value_heads * (group.key_head_dim + group.value_head_dim + 2) * 4
    state = cpu_replay + linear * channels * (history + rows) * activation
    kv = (prefix - linear) * rows * mc.num_kv_heads * mc.head_dim * activation * 2
    replay = suffix * (channels * (history + rows - 1) * activation + rows * (
        group.num_key_heads * group.key_head_dim * activation
        + group.num_value_heads * group.value_head_dim * activation + group.num_value_heads * 8))
    hidden = rows * (mc.hidden_size * activation + 4) + rows * 12
    # The suffix plan and hints coexist with the frozen prediction and admission queue.
    plans = mc.num_moe_layers * mc.num_experts * (3 * 4 + 3 * 4) + mc.num_moe_layers * 8
    outer = max(0, rows - max(CPU_ASYNC_REPLAY_ROWS, config.speculative_mtp_depth + 1))
    control = outer * (12 + mc.num_moe_layers * mc.num_experts_per_tok * 4)
    return state + kv + replay + hidden + plans + control


def speculative_gdn_workspace_bytes(config) -> int:
    depth = getattr(config, "speculative_mtp_depth", 0) or 0
    if not depth or getattr(config, "speculative_reference_math", False):
        return 0
    group = config.model_config.linear_attention_group()
    if group is None:
        return 0
    from freetoken.kvcache.linear_state_pool import ssm_state_dtype
    from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState

    if not QwenGDNVerifyState.chunk_enabled:
        return 0

    inner = depth + 1
    three_level = bool(getattr(config, "speculative_three_level", False))
    cpu_async = getattr(config, "speculative_mode", None) == "cpu_async"
    if cpu_async:
        width = max(CPU_ASYNC_REPLAY_ROWS, inner)
    elif three_level:
        maximum = three_level_max_rows(depth, config.speculative_max_loops)
        width = min(getattr(config, "speculative_submit_tokens", 0) or maximum, maximum)
    else:
        width = inner
    if not cpu_async:
        width = min(width, getattr(config, "speculative_max_output_tokens", None) or width)
    tp = config.tp_info.size
    costs = [QwenGDNVerifyState.chunk_temporary_bytes_for_geometry(
        rows, div_ceil(group.num_key_heads, tp), div_ceil(group.num_value_heads, tp),
        group.key_head_dim, group.value_head_dim, activation_dtype=config.dtype,
        state_dtype=ssm_state_dtype())
        for rows in range(1, min(width, QwenGDNVerifyState.chunk_max_width) + 1)]
    if not costs:
        return 0
    pooled = three_level and getattr(config, "speculative_graph_implementation", "reference") in ("coarse", "full")
    eager = getattr(config, "cuda_graph_max_bs", None) == 0 or getattr(config, "cuda_graph_bs", None) == []
    # Full three-level graphs share a pool; legacy per-width graphs retain separate pools.
    if eager or pooled:
        return max(costs)
    return sum(costs) + (sum(costs[:inner]) if three_level else 0)


def speculative_hybrid_workspace_bytes(config, *, include_cpu_partials=True) -> int:
    if (getattr(config, "moe_strategy", "offload") != "hybrid"
            or getattr(config, "speculative_expert_scheduler", "host") != "gpu"
            or not getattr(config, "speculative_hybrid_phases", ())):
        return 0
    mc = config.model_config
    inner = config.speculative_mtp_depth + 1
    loops = config.cpu_async_loops if getattr(config, "speculative_mode", None) == "cpu_async" else config.speculative_max_loops
    maximum = loops * inner + 1
    if (getattr(config, "speculative_three_level", False)
            and getattr(config, "speculative_mode", None) != "cpu_async"):
        maximum = three_level_max_rows(config.speculative_mtp_depth, config.speculative_max_loops)
    outer = min(config.speculative_submit_tokens or maximum, maximum)
    limit = getattr(config, "speculative_max_output_tokens", None) or outer
    inner, outer = min(inner, limit), min(outer, limit)
    maximum = max(inner, outer)
    if getattr(config, "speculative_mode", None) == "cpu_async":
        # Recovery and suffix controllers reserve their full widths even for short outputs.
        maximum = max(maximum, CPU_ASYNC_REPLAY_ROWS, loops * (config.speculative_mtp_depth + 1) + 1)
    widths = maximum * (maximum + 1) // 2
    routes = maximum * mc.num_experts_per_tok
    tables = 4 * (2 * (maximum + 1) * (routes + 1) + 4 * (maximum + 1) + routes + 1)
    routing = widths * mc.num_experts_per_tok * 8
    # The CPU branch joins before merge; the next layer forks after that merge.
    partials = 2 * widths * mc.hidden_size if include_cpu_partials else 0
    return tables + routing + partials + 7 * 8


def layer_slot_bytes(sources: dict[str, "list[torch.Tensor]"]) -> list[int]:
    """Bytes one expert slot of each layer occupies on GPU: its row bytes summed over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4/v4_pool.py.
    num_layers = len(next(iter(sources.values())))
    return [sum(t[l][0].numel() * t[l].element_size() for t in sources.values()) for l in range(num_layers)]


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU, rounded up to the mean over layers when their bank layouts differ."""
    per_layer = layer_slot_bytes(sources)
    return -(-sum(per_layer) // len(per_layer))


def expert_pool_bytes(sources, pools, sizes, row_tail_bytes=None, layer_tags=None):
    from freetoken.moe.legacy_format import canonical_role
    from freetoken.moe.offload_cache import bank_layouts

    layouts = bank_layouts(sources, layer_tags)
    tails = {canonical_role(name): values for name, values in (row_tail_bytes or {}).items()}
    total = 0
    for name, layers in sources.items():
        pads = {}
        for layout, tail in zip(layouts, tails.get(canonical_role(name), [0] * len(layers))):
            pads[layout] = max(pads.get(layout, 0), -(-tail // 16) * 16)
        total += max(pads.values())
        for pool, size in zip(pools, sizes):
            layer = pool.layers[0]
            total += size * layers[layer][0].numel() * layers[layer].element_size() + pads[layouts[layer]]
    return total


def minimum_staging_slots(sources, pools, num_experts, *, buffers, counts=None, row_tail_bytes=None, layer_tags=None):
    """Smallest split-slot allocation that fits every bank's complete prefill buffers."""
    from freetoken.moe.legacy_format import canonical_role
    from freetoken.moe.offload_cache import bank_layouts
    from freetoken.moe.pool_policy import split_slots

    layouts = bank_layouts(sources, layer_tags)
    tails = {canonical_role(name): values for name, values in (row_tail_bytes or {}).items()}
    banks = []
    for name, layers in sources.items():
        rows = [t[0].numel() * t.element_size() for t in layers]
        pads_by_layout = {}
        for layout, tail in zip(layouts, tails.get(canonical_role(name), [0] * len(layers))):
            pads_by_layout[layout] = max(pads_by_layout.get(layout, 0), -(-tail // 16) * 16)
        pads = [pads_by_layout[layout] for layout in layouts]
        banks.append((rows, pads, buffers * (num_experts * max(rows) + max(pads))))

    def fits(total):
        sizes = split_slots(total, pools, counts)
        return all(sum(size * rows[pool.layers[0]] + pads[pool.layers[0]]
                       for size, pool in zip(sizes, pools)) + max(pads) >= required
                   for rows, pads, required in banks)

    low = max(num_experts, sum(pool.min_num_slots for pool in pools))
    high = sum(pool.max_num_slots for pool in pools)
    if low > high or not fits(high):
        raise ValueError(f"expert slot pools cannot hold {buffers} complete prefill staging buffers")
    # Both split policies allocate a fixed ordered sequence of slots, so each bank grows monotonically.
    while low < high:
        middle = (low + high) // 2
        if fits(middle):
            high = middle
        else:
            low = middle + 1
    return low


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = max(remaining // cache_per_page, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = moe_cache_size * per_expert_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    max_slots: int | None = None,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    ``max_slots`` is the expert kernel's addressable slot limit; the plan never exceeds it.

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = total_experts if max_slots is None else min(max_slots, total_experts)
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )
