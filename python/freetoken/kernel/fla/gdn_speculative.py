# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Fold and deduplicated conv windows adapted from SGLang e9400c5a8f3a."""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_verify_kernel(
    raw, weight, state, windows, out,
    raw_row: tl.constexpr, raw_col: tl.constexpr,
    weight_row: tl.constexpr, weight_col: tl.constexpr,
    state_slot: tl.constexpr, state_col: tl.constexpr, state_time: tl.constexpr,
    CHANNELS: tl.constexpr, WIDTH: tl.constexpr, HISTORY: tl.constexpr,
    SLOT: tl.constexpr, BLOCK: tl.constexpr, FP32_PRODUCTS: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = c < CHANNELS
    base = SLOT * state_slot + c * state_col
    for j in tl.static_range(HISTORY - 1):
        old = tl.load(state + base + (j + 1) * state_time, valid, other=0)
        tl.store(windows + c * (HISTORY + WIDTH - 1) + j, old, valid)
    for t in tl.static_range(WIDTH):
        x = tl.load(raw + t * raw_row + c * raw_col, valid, other=0)
        tl.store(windows + c * (HISTORY + WIDTH - 1) + HISTORY - 1 + t, x, valid)
        acc = tl.zeros((BLOCK,), tl.float32)
        for j in tl.static_range(HISTORY + 1):
            source = t + j - HISTORY
            if source < 0:
                value = tl.load(state + base + (source + HISTORY) * state_time, valid, other=0)
            else:
                value = tl.load(raw + source * raw_row + c * raw_col, valid, other=0)
            w = tl.load(weight + c * weight_row + j * weight_col, valid, other=0)
            if FP32_PRODUCTS:
                acc += value.to(tl.float32) * w.to(tl.float32)
            else:
                acc += value * w
        acc = acc / (1 + tl.exp(-acc))
        tl.store(out + t * CHANNELS + c, acc, valid)


@triton.jit
def _conv_commit_kernel(
    state, windows, count,
    state_layer: tl.constexpr, state_slot: tl.constexpr,
    state_col: tl.constexpr, state_time: tl.constexpr,
    CHANNELS: tl.constexpr, WIDTH: tl.constexpr, HISTORY: tl.constexpr,
    SLOT: tl.constexpr, BLOCK: tl.constexpr,
):
    n = tl.load(count)
    if n <= 0 or n > WIDTH:
        return
    layer = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = c < CHANNELS
    for j in tl.static_range(HISTORY):
        offset = (layer * CHANNELS + c) * (HISTORY + WIDTH - 1) + n - 1 + j
        value = tl.load(windows + offset, valid, other=0)
        tl.store(state + layer * state_layer + SLOT * state_slot + c * state_col + j * state_time, value, valid)


@triton.jit
def _ssm_fold_kernel(
    state, raw_k, raw_v, g, beta, count,
    state_layer: tl.constexpr, state_slot: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, WIDTH: tl.constexpr,
    SLOT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    ROUND_STATE_EACH_STEP: tl.constexpr,
):
    n = tl.load(count)
    if n <= 0 or n > WIDTH:
        return
    layer = (tl.program_id(1) // HV).to(tl.int64)
    hv = tl.program_id(1) % HV
    h = hv // (HV // H)
    k = tl.arange(0, BK)
    v = tl.program_id(0) * BV + tl.arange(0, BV)
    mask = (k[:, None] < K) & (v[None, :] < V)
    address = state + layer * state_layer + SLOT * state_slot + hv * K * V + k[:, None] + v[None, :] * K
    s = tl.load(address, mask, other=0).to(tl.float32)
    for t in range(0, n):
        key = tl.load(raw_k + ((layer * WIDTH + t) * H + h) * K + k, k < K, other=0).to(tl.float32)
        value = tl.load(raw_v + ((layer * WIDTH + t) * HV + hv) * V + v, v < V, other=0).to(tl.float32)
        decay = tl.load(g + (layer * WIDTH + t) * HV + hv)
        b = tl.load(beta + (layer * WIDTH + t) * HV + hv)
        key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
        s *= tl.exp(decay)
        value -= tl.sum(s * key[:, None], 0)
        value *= b
        s += key[:, None] * value[None, :]
        if ROUND_STATE_EACH_STEP:
            s = s.to(state.dtype.element_ty).to(tl.float32)
    tl.store(address, s.to(state.dtype.element_ty), mask)


def conv_verify(raw, weight, state, windows, slot, *, fp32_products=True):
    width, channels = raw.shape
    history = state.shape[-1]
    if windows.shape != (channels, history + width - 1):
        raise ValueError("conv verification window has the wrong shape")
    out = torch.empty_like(raw, memory_format=torch.contiguous_format)
    _conv_verify_kernel[(triton.cdiv(channels, 128),)](
        raw, weight, state, windows, out,
        *raw.stride(), *weight.stride(), *state.stride(),
        CHANNELS=channels, WIDTH=width, HISTORY=history, SLOT=slot, BLOCK=128,
        FP32_PRODUCTS=fp32_products,
        num_warps=4,
    )
    return out


@triton.jit
def _prepare_chunk_verify_kernel(
    mixed, a, b, a_log, dt_bias, q_normalized, k_normalized,
    raw_k, raw_v, decay, beta,
    MIXED_STRIDE: tl.constexpr, A_STRIDE: tl.constexpr, B_STRIDE: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
):
    row, hv = tl.program_id(0), tl.program_id(1)
    kh = hv // (HV // H)
    if hv % (HV // H) == 0:
        key_offsets = tl.arange(0, BK)
        query = tl.load(mixed + row * MIXED_STRIDE + kh * K + key_offsets,
                        key_offsets < K, 0).to(tl.float32)
        key = tl.load(mixed + row * MIXED_STRIDE + H * K + kh * K + key_offsets,
                      key_offsets < K, 0).to(tl.float32)
        base = (row * H + kh) * K + key_offsets
        tl.store(raw_k + base, key, key_offsets < K)
        query /= tl.sqrt(tl.sum(query * query) + 1e-6)
        key /= tl.sqrt(tl.sum(key * key) + 1e-6)
        tl.store(q_normalized + base, query, key_offsets < K)
        tl.store(k_normalized + base, key, key_offsets < K)
    value_offsets = tl.arange(0, BV)
    value = tl.load(mixed + row * MIXED_STRIDE + 2 * H * K + hv * V + value_offsets,
                    value_offsets < V, 0)
    tl.store(raw_v + (row * HV + hv) * V + value_offsets, value, value_offsets < V)
    av = tl.load(a + row * A_STRIDE + hv).to(tl.float32)
    bv = tl.load(b + row * B_STRIDE + hv).to(tl.float32)
    log = tl.load(a_log + hv).to(tl.float32)
    bias = tl.load(dt_bias + hv).to(tl.float32)
    x = av + bias
    softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    tl.store(decay + row * HV + hv, -tl.exp(log) * softplus)
    tl.store(beta + row * HV + hv, 1.0 / (1.0 + tl.exp(-bv)))


def chunk_verify(mixed, a, b, layer, state_source, indices, replay_inputs):
    from freetoken.kernel.fla.chunk import chunk_gated_delta_rule_fwd

    width = mixed.shape[0]
    h, hv, k, v = layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim
    q_normalized = torch.empty((1, width, h, k), device=mixed.device, dtype=mixed.dtype)
    k_normalized = torch.empty_like(q_normalized)
    raw_k, raw_v, decay, beta = replay_inputs
    _prepare_chunk_verify_kernel[(width, hv)](
        mixed, a, b, layer.A_log, layer.dt_bias, q_normalized, k_normalized,
        raw_k, raw_v, decay, beta,
        MIXED_STRIDE=mixed.stride(0), A_STRIDE=a.stride(0), B_STRIDE=b.stride(0),
        H=h, HV=hv, K=k, V=v, BK=triton.next_power_of_2(k), BV=triton.next_power_of_2(v),
        num_warps=4,
    )
    # Equal-length single-sequence metadata avoids host chunk-index construction in capture.
    return chunk_gated_delta_rule_fwd(
        q=q_normalized, k=k_normalized, v=raw_v.unsqueeze(0),
        g=decay.unsqueeze(0), beta=beta.unsqueeze(0), scale=k ** -0.5,
        initial_state=state_source, initial_state_indices=indices, inplace_update=False,
    )[1][0]


def commit_gdn(state, conv, inputs, windows, count, slot, width):
    if count.numel() != 1 or count.dtype != torch.int32 or count.device != state.device:
        raise ValueError("GDN commit requires one device int32 input count")
    raw_k, raw_v, g, beta = inputs
    layers, _, heads, key_dim, value_dim = state.shape
    _ssm_fold_kernel[(triton.cdiv(value_dim, 32), layers * heads)](
        state, raw_k, raw_v, g, beta, count,
        *state.stride()[:2], H=raw_k.shape[2], HV=heads,
        K=key_dim, V=value_dim, WIDTH=width, SLOT=slot,
        BK=triton.next_power_of_2(key_dim), BV=32,
        ROUND_STATE_EACH_STEP=state.dtype != torch.float32,
        num_warps=1, num_stages=3,
    )
    _conv_commit_kernel[(layers, triton.cdiv(conv.shape[2], 128))](
        conv, windows, count, *conv.stride(), CHANNELS=conv.shape[2],
        WIDTH=width, HISTORY=conv.shape[-1], SLOT=slot, BLOCK=128,
        num_warps=4,
    )


@triton.jit
def _copy_cpu_conv_windows_kernel(
    sources, destination,
    destination_layer: tl.constexpr, destination_channel: tl.constexpr,
    CHANNELS: tl.constexpr, CAPACITY: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    layer = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    channel, position = offset // CAPACITY, offset % CAPACITY
    valid = (channel < CHANNELS) & (position < WIDTH)
    source = tl.load(sources + layer).to(tl.pointer_type(tl.bfloat16))
    value = tl.load(source + offset, valid, other=0)
    tl.store(destination + layer * destination_layer + channel * destination_channel + position, value, valid)


def copy_cpu_conv_windows(destination, sources, width):
    if (destination.dtype != torch.bfloat16 or destination.ndim != 3
            or sources.dtype != torch.int64 or sources.shape != destination.shape[:1]
            or not 1 <= width <= destination.shape[-1]):
        raise ValueError("invalid mapped CPU convolution windows")
    layers, channels, capacity = destination.shape
    if layers:
        _copy_cpu_conv_windows_kernel[(layers, triton.cdiv(channels * capacity, 1024))](
            sources, destination, *destination.stride()[:2], CHANNELS=channels,
            CAPACITY=capacity, WIDTH=width, BLOCK=1024, num_warps=4)


@triton.jit
def _cpu_ssm_fold_kernel(
    state, replay, count,
    state_layer: tl.constexpr, state_slot: tl.constexpr,
    replay_layer: tl.constexpr, replay_row: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    SLOT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, V2: tl.constexpr,
):
    layer = (tl.program_id(1) // HV).to(tl.int64)
    head = tl.program_id(1) % HV
    k = tl.arange(0, BK)
    v = tl.program_id(0) * BV + tl.arange(0, BV)
    mask = (k[:, None] < K) & (v[None, :] < V)
    address = state + layer * state_layer + SLOT * state_slot + head * V * K + v[None, :] * K + k[:, None]
    s = tl.load(address, mask, other=0)
    for row in range(count):
        item = replay + layer * replay_layer + row * replay_row + head * (K + V + 2)
        key = tl.load(item + k, k < K, other=0)
        value = tl.load(item + K + v, v < V, other=0)
        decay = tl.load(item + K + V)
        beta = tl.load(item + K + V + 1)
        s = s * decay
        if V2:
            # Match the CPU's 16-lane FMA accumulation and AVX-512 horizontal reduction.
            lane = tl.arange(0, 16)
            partial = tl.full((16, BV), 0, tl.float32)
            for block in tl.static_range(K // 16):
                positions = (block * 16 + lane)[:, None] + tl.full((1, BV), 0, tl.int32)
                sv = tl.gather(s, positions, 0)
                kv = tl.load(item + block * 16 + lane)
                partial = tl.fma(sv, kv[:, None], partial)
            half = tl.arange(0, 8)[:, None] + tl.full((1, BV), 0, tl.int32)
            partial = tl.gather(partial, half + 8, 0) + tl.gather(partial, half, 0)
            quarter = tl.arange(0, 4)[:, None] + tl.full((1, BV), 0, tl.int32)
            partial = tl.gather(partial, quarter + 4, 0) + tl.gather(partial, quarter, 0)
            pair = tl.arange(0, 2)[:, None] + tl.full((1, BV), 0, tl.int32)
            partial = tl.gather(partial, pair + 2, 0) + tl.gather(partial, pair, 0)
            first = tl.full((1, BV), 0, tl.int32)
            dot = tl.reshape(tl.gather(partial, first, 0) + tl.gather(partial, first + 1, 0), (BV,))
            for index in tl.static_range((K // 16) * 16, K):
                sv = tl.reshape(tl.gather(s, tl.full((1, BV), index, tl.int32), 0), (BV,))
                kv = tl.load(item + index)
                dot = tl.fma(sv, kv, dot)
        else:
            # v1 adds products in key order; a tree reduction changes accepted CPU state.
            dot = tl.full((BV,), 0, tl.float32)
            product = s * key[:, None]
            for index in tl.static_range(K):
                term = tl.reshape(tl.gather(product, tl.full((1, BV), index, tl.int32), 0), (BV,))
                dot = dot + term
        delta = (value - dot) * beta
        if V2:
            s = tl.fma(key[:, None], delta[None, :], s)
        else:
            s = s + key[:, None] * delta[None, :]
    tl.store(address, s, mask)


@triton.jit
def _cpu_conv_commit_kernel(
    conv, windows, count,
    conv_layer: tl.constexpr, conv_slot: tl.constexpr,
    window_layer: tl.constexpr, window_channel: tl.constexpr,
    CHANNELS: tl.constexpr, HISTORY: tl.constexpr, SLOT: tl.constexpr, BLOCK: tl.constexpr,
):
    layer = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    for time in tl.static_range(HISTORY):
        value = tl.load(windows + layer * window_layer + channel * window_channel + count + time,
                        channel < CHANNELS, other=0)
        tl.store(conv + layer * conv_layer + SLOT * conv_slot + channel * HISTORY + time,
                 value, channel < CHANNELS)


def commit_cpu_gdn(state, conv, replay, windows, count, slot, *, v2=False):
    if (state.ndim != 5 or conv.ndim != 4 or replay.ndim != 4 or windows.ndim != 3
            or state.dtype != torch.float32 or replay.dtype != torch.float32
            or conv.dtype != torch.bfloat16 or windows.dtype != torch.bfloat16):
        raise ValueError("CPU GDN fold requires FP32 state/replay and BF16 convolution windows")
    layers, slots, heads, key_dim, value_dim = state.shape
    if (replay.shape[0] != layers or replay.shape[2:] != (heads, key_dim + value_dim + 2)
            or conv.shape[:2] != (layers, slots) or windows.shape[:2] != (layers, conv.shape[2])
            or type(count) is not int or not 1 <= count <= replay.shape[1]
            or not 0 <= slot < slots or windows.shape[-1] < conv.shape[-1] + count):
        raise ValueError("CPU GDN fold geometry or accepted input count mismatch")
    if layers:
        _cpu_ssm_fold_kernel[(triton.cdiv(value_dim, 16), layers * heads)](
            state, replay, count, *state.stride()[:2], *replay.stride()[:2],
            HV=heads, K=key_dim, V=value_dim, SLOT=slot,
            BK=max(16, triton.next_power_of_2(key_dim)), BV=16, V2=v2,
            num_warps=4, enable_fp_fusion=False)
        _cpu_conv_commit_kernel[(layers, triton.cdiv(conv.shape[2], 128))](
            conv, windows, count, *conv.stride()[:2], *windows.stride()[:2],
            CHANNELS=conv.shape[2], HISTORY=conv.shape[-1], SLOT=slot, BLOCK=128, num_warps=4)
