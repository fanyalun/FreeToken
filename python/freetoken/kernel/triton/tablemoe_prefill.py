"""Compact expert-group Prefill kernels with stable route-indexed outputs."""

import triton
import triton.language as tl


@triton.jit
def compact_group_blocks(
    sorted_ids, num_padded, route_slots, route_groups, group_counts, group_offsets,
    group_counters, group_blocks,
    ROUTES: tl.constexpr, MAX_BLOCKS: tl.constexpr, BLOCK_M: tl.constexpr,
    GROUPS: tl.constexpr, HIST: tl.constexpr, BLOCK: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK)
    count = tl.zeros((HIST,), tl.int32)
    padded = tl.load(num_padded)
    for start in range(tl.cdiv(MAX_BLOCKS, BLOCK)):
        block = start * BLOCK + lanes
        first = tl.load(sorted_ids + block * BLOCK_M,
                        (block < MAX_BLOCKS) & (block * BLOCK_M < padded), ROUTES)
        valid = first < ROUTES
        slot = tl.load(route_slots + first, valid, -1)
        group = tl.load(route_groups + first, valid, -2) + 1
        valid &= (slot >= 0) & (group >= 0) & (group < GROUPS)
        count += tl.histogram(tl.where(valid, group, HIST - 1), HIST)
    groups = tl.arange(0, HIST)
    count = tl.where(groups < GROUPS, count, 0)
    offset = tl.cumsum(count, 0) - count
    tl.store(group_counts + groups, count, groups < GROUPS)
    tl.store(group_offsets + groups, offset, groups < GROUPS)
    tl.store(group_counters + groups, 0, groups < GROUPS)
    # One CTA owns the entire plan, so this barrier also orders scatter counters.
    tl.debug_barrier()
    for start in range(tl.cdiv(MAX_BLOCKS, BLOCK)):
        block = start * BLOCK + lanes
        first = tl.load(sorted_ids + block * BLOCK_M,
                        (block < MAX_BLOCKS) & (block * BLOCK_M < padded), ROUTES)
        valid = first < ROUTES
        slot = tl.load(route_slots + first, valid, -1)
        group = tl.load(route_groups + first, valid, -2) + 1
        valid &= (slot >= 0) & (group >= 0) & (group < GROUPS)
        base = tl.load(group_offsets + group, valid, 0)
        rank = tl.atomic_add(group_counters + group, 1, valid)
        tl.store(group_blocks + base + rank, block, valid)


@triton.jit(do_not_specialize=["group_id"])
def grouped_expert_gemm(
    a_ptr, b_ptr, c_ptr, sorted_token_ids_ptr, route_slots_ptr,
    group_blocks_ptr, group_counts_ptr, group_offsets_ptr, group_id,
    N: tl.constexpr, K: tl.constexpr, num_valid_routes: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_be: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, TOP_K: tl.constexpr, EVEN_K: tl.constexpr,
):
    num_pid_m = tl.load(group_counts_ptr + group_id + 1)
    block_offset = tl.load(group_offsets_ptr + group_id + 1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # A bounded persistent grid avoids launching a full-route grid for empty groups.
    for pid in range(tl.program_id(0), num_pid_m * num_pid_n, tl.num_programs(0)):
        scheduling_group = pid // num_pid_in_group
        first_pid_m = scheduling_group * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        group_block = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m
        pid_m = tl.load(group_blocks_ptr + block_offset + group_block)
        first_route = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M)
        routes = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M +
                         tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
        valid = routes < num_valid_routes
        ns = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        ks = tl.arange(0, BLOCK_SIZE_K)
        expert = tl.load(route_slots_ptr + first_route).to(tl.int64)
        a = a_ptr + routes[:, None] // TOP_K * stride_am + ks[None, :] * stride_ak
        b = b_ptr + expert * stride_be + ks[:, None] * stride_bk + ns[None, :] * stride_bn
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
        for block in range(tl.cdiv(K, BLOCK_SIZE_K)):
            if EVEN_K:
                av = tl.load(a, valid[:, None], 0)
                bv = tl.load(b)
            else:
                av = tl.load(a, valid[:, None] & (ks[None, :] < K - block * BLOCK_SIZE_K), 0)
                bv = tl.load(b, ks[:, None] < K - block * BLOCK_SIZE_K, 0)
            acc += tl.dot(av, bv)
            a += BLOCK_SIZE_K * stride_ak
            b += BLOCK_SIZE_K * stride_bk
        output_ns = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        dst = c_ptr + routes[:, None] * stride_cm + output_ns[None, :] * stride_cn
        tl.store(dst, acc.to(tl.float16), valid[:, None] & (output_ns[None, :] < N))


@triton.jit(do_not_specialize=["group_id"])
def grouped_silu(
    gate_up, activated, sorted_token_ids, group_blocks, group_counts, group_offsets, group_id,
    ROUTES: tl.constexpr, INTER: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    count = tl.load(group_counts + group_id + 1)
    offset = tl.load(group_offsets + group_id + 1)
    cols = tl.cdiv(INTER, BLOCK_N)
    for pid in range(tl.program_id(0), count * cols, tl.num_programs(0)):
        block = tl.load(group_blocks + offset + pid // cols)
        routes = tl.load(sorted_token_ids + block * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        ns = (pid % cols) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (routes[:, None] < ROUTES) & (ns[None, :] < INTER)
        gate = tl.load(gate_up + routes[:, None] * (2 * INTER) + ns[None, :], mask, 0).to(tl.float32)
        up = tl.load(gate_up + routes[:, None] * (2 * INTER) + INTER + ns[None, :], mask, 0).to(tl.float32)
        value = gate / (1 + tl.exp(-gate)) * up
        tl.store(activated + routes[:, None] * INTER + ns[None, :], value.to(tl.float16), mask)
