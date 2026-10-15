import triton
import triton.language as tl


@triton.jit
def fused_moe_decode_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    total_routes: int,
    N: int,
    K: int,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_ck,
    stride_cn,
    stride_tw_m,
    stride_tw_k,
    stride_tid_m,
    stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_row = route_id if A_ROW_IS_ROUTE else token_id

    expert_id = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k)
    route_weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
    a_ptrs = a_ptr + a_row * stride_am + offs_k * stride_ak
    b_ptrs = (
        b_ptr
        + expert_id.to(tl.int64) * stride_be
        + offs_n[None, :] * stride_bn
        + offs_k[:, None] * stride_bk
    )

    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    if (expert_id >= 0) & (route_weight != 0):
        for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if even_Ks:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
            else:
                k_mask = offs_k < K - k_start * BLOCK_SIZE_K
                a = tl.load(a_ptrs, mask=k_mask, other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
            accumulator += tl.sum(a[:, None] * b, axis=0)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        accumulator = accumulator * route_weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(
        c_ptrs,
        accumulator.to(compute_type),
        mask=(route_id < total_routes) & (offs_n < N),
    )


@triton.jit
def group_decode_routes(ids, weights, group_slots, group_routes, group_count,
                        ROWS: tl.constexpr, TOP_K: tl.constexpr, BLOCK: tl.constexpr,
                        TABLE: tl.constexpr):
    r = tl.arange(0, BLOCK)
    expert = tl.load(ids + r, r < ROWS * TOP_K, -1)
    w = tl.load(weights + r, r < ROWS * TOP_K, 0)
    valid = (r < ROWS * TOP_K) & (expert >= 0) & (w != 0)
    same = (expert[:, None] == expert[None, :]) & valid[:, None] & valid[None, :]
    earlier = r[None, :] < r[:, None]
    # Pair the nth occurrence in each token. This also preserves duplicate routes
    # within a token while bounding each shared-weight group to one route per row.
    occurrence = tl.sum((same & earlier & ((r[:, None] // TOP_K) == (r[None, :] // TOP_K))).to(tl.int32), 1)
    same = same & (occurrence[:, None] == occurrence[None, :])
    first = valid & (tl.sum((same & earlier).to(tl.int32), 1) == 0)
    compact = tl.cumsum(first.to(tl.int32)) - 1
    head = tl.min(tl.where(same, r[None, :], BLOCK - 1), 1)
    group = tl.gather(compact, head, 0)
    off = tl.arange(0, TABLE)
    tl.store(group_routes + off, -1, off < ROWS * ROWS * TOP_K)
    tl.debug_barrier()
    tl.store(group_slots + compact, expert, first)
    tl.store(group_routes + group * ROWS + r // TOP_K, r, valid)
    tl.store(group_count, tl.sum(first.to(tl.int32)))


@triton.jit
def fused_moe_decode_grouped_kernel(
    a, b, c, weights, ids, group_slots, group_routes, group_count,
    N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr,
    stride_be: tl.constexpr, stride_bn: tl.constexpr,
    ROWS: tl.constexpr, TOP_K: tl.constexpr, A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    group = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    original_id = tl.load(ids + group)
    original_weight = tl.load(weights + group)
    if (original_id < 0) | (original_weight == 0):
        tl.store(c + group * N + n, 0, n < N)
    if group < tl.load(group_count):
        expert = tl.load(group_slots + group)
        # One weight tile serves every token routed to this expert. Padding only
        # the register tile keeps tensor-core accumulation independent of grouping.
        m = tl.arange(0, 16)
        routes = tl.load(group_routes + group * ROWS + m, m < ROWS, -1)
        input_rows = routes if A_ROW_IS_ROUTE else routes // TOP_K
        accumulator = tl.zeros((16, BLOCK_N), tl.float32)
        for start in range(tl.cdiv(K, BLOCK_K)):
            k = start * BLOCK_K + tl.arange(0, BLOCK_K)
            inputs = tl.load(a + input_rows[:, None] * stride_am + k[None, :],
                             (routes[:, None] >= 0) & (k[None, :] < K), 0)
            tile = tl.load(b + expert.to(tl.int64) * stride_be + n[None, :] * stride_bn + k[:, None],
                           (k[:, None] < K) & (n[None, :] < N), 0)
            accumulator = tl.dot(inputs, tile, accumulator)
        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(weights + routes, routes >= 0, 0)[:, None]
        tl.store(c + routes[:, None] * N + n[None, :], accumulator,
                 (routes[:, None] >= 0) & (n[None, :] < N))
