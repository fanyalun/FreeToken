from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import triton
import triton.language as tl

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)


@triton.jit
def _store_cache_masked(k_cache, v_cache, indices, k, v, valid_rows,
                        K_STRIDE: tl.constexpr, V_STRIDE: tl.constexpr,
                        CACHE_STRIDE: tl.constexpr, FEATURES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row < tl.load(valid_rows):
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(indices + row)
        key = tl.load(k + row * K_STRIDE + col, col < FEATURES, 0)
        value = tl.load(v + row * V_STRIDE + col, col < FEATURES, 0)
        tl.store(k_cache + slot * CACHE_STRIDE + col, key, col < FEATURES)
        tl.store(v_cache + slot * CACHE_STRIDE + col, value, col < FEATURES)


def store_cache_masked(k_cache, v_cache, indices, k, v, valid_rows):
    k = k.reshape(k.shape[0], -1)
    v = v.reshape(v.shape[0], -1)
    features = k.shape[1]
    _store_cache_masked[(k.shape[0], triton.cdiv(features, 256))](
        k_cache, v_cache, indices, k, v, valid_rows,
        k.stride(0), v.stride(0), features, features, 256)
