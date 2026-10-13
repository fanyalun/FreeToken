"""Fixed-shape MoDES modality classification and route selection."""

import triton
import triton.language as tl


@triton.jit
def _select(weights, tokens, request_slots, modality, special_ids, importance, tau, keep,
            TOP_K: tl.constexpr, SPECIALS: tl.constexpr, DUMMY: tl.constexpr,
            N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = r // TOP_K
    token = tl.load(tokens + row, r < N, 0)
    mod = tl.load(modality + row, r < N, 1)
    special = tl.full((BLOCK,), False, tl.int1)
    for i in range(SPECIALS):
        special |= token == tl.load(special_ids + i)
    imp = tl.load(importance + mod)
    threshold = tl.load(tau + mod).to(tl.float16).to(tl.float32)
    w = tl.load(weights + r, r < N, 0).to(tl.float16).to(tl.float32)
    score = (w * imp).to(tl.float16).to(tl.float32)
    valid = tl.load(request_slots + row, r < N, DUMMY) != DUMMY
    exact = ((score >= threshold) | (special & (mod == 1))) & valid
    tl.store(keep + r, exact, r < N)
