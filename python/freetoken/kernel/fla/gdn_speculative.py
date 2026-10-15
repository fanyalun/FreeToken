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
