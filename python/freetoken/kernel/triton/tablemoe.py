import torch
import triton
import triton.language as tl


@triton.jit
def _offline_search(query, keys, counts, ids, modality, reuse, result,
                    TOP_K: tl.constexpr, DIM: tl.constexpr, CLUSTERS: tl.constexpr,
                    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
                    PREFILL: tl.constexpr, VISION_THRESHOLD: tl.constexpr,
                    TEXT_THRESHOLD: tl.constexpr, DECODE_THRESHOLD: tl.constexpr = None):
    slot = tl.program_id(0)
    if not tl.load(reuse + slot):
        tl.store(result + slot, -1)
        return
    row = slot // TOP_K
    expert = tl.load(ids + slot)
    mod = tl.load(modality + row)
    count = tl.load(counts + expert * 2 + mod)
    ds = tl.arange(0, BLOCK_D)
    ks = tl.arange(0, BLOCK_K)
    q = tl.load(query + row * DIM + ds, ds < DIM, 0).to(tl.float32)
    if not PREFILL and DECODE_THRESHOLD is not None:
        q = q / tl.maximum(tl.sqrt(tl.sum(q * q, 0)), 1e-6)
    key = tl.load(keys + ((expert * 2 + mod) * CLUSTERS + ks[:, None]) * DIM + ds[None, :],
                  (ks[:, None] < count) & (ds[None, :] < DIM), 0).to(tl.float32)
    scores = tl.sum(key * q[None, :], 1)
    scores = tl.where(ks < count, scores, -float("inf"))
    best, index = tl.max(scores, 0, return_indices=True)
    if PREFILL:
        hit = best >= tl.where(mod == 0, VISION_THRESHOLD, TEXT_THRESHOLD)
    elif DECODE_THRESHOLD is not None:
        hit = best >= DECODE_THRESHOLD
    else:
        hit = best > 0.0
    tl.store(result + slot, tl.where(hit & (count > 0),
                                    (expert * 2 + mod) * CLUSTERS + index, -1))


def offline_search(query, keys, counts, ids, modality, reuse, *, prefill,
                   vision_threshold=0.6, text_threshold=0.7, decode_threshold=None):
    result = torch.empty_like(ids, dtype=torch.int64)
    _offline_search[(ids.numel(),)](
        query, keys, counts, ids, modality, reuse, result,
        ids.shape[1], query.shape[1], keys.shape[2],
        triton.next_power_of_2(query.shape[1]), triton.next_power_of_2(keys.shape[2]),
        prefill, vision_threshold, text_threshold, decode_threshold,
    )
    return result


@triton.jit
def _prepare_routes(ids, slots, keep, cpu_ids, cache_hit, slot_for_id, request_slots,
                    TOP_K: tl.constexpr, KEEP_K: tl.constexpr, EXPERTS: tl.constexpr,
                    LAYER: tl.constexpr, DUMMY: tl.constexpr, PREFILL: tl.constexpr,
                    HAS_CACHE: tl.constexpr, BLOCK: tl.constexpr, N: tl.constexpr,
                    EXTERNAL_MASK: tl.constexpr = False):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    eid = tl.load(ids + r, r < N, 0).to(tl.int32)
    req = tl.load(request_slots + r // TOP_K, r < N, DUMMY)
    exact = (tl.load(keep + r, r < N, False) if PREFILL or EXTERNAL_MASK else r % TOP_K < KEEP_K) & (req != DUMMY)
    tl.store(keep + r, exact, r < N)
    hit = tl.full((BLOCK,), False, tl.int1)
    if HAS_CACHE:
        hit = exact & (tl.load(slot_for_id + LAYER * EXPERTS + eid, r < N, -1) >= 0)
    tl.store(cache_hit + r, hit, r < N)
    tl.store(slots + r, tl.where(exact, eid, -1), r < N)
    tl.store(cpu_ids + r, tl.where(exact, eid, -1), r < N)


@triton.jit
def _split_routes(ids, keep, slots, cpu_ids, cache_hit, counts, num_indices,
                  N: tl.constexpr, TOP_K: tl.constexpr, GPU_K: tl.constexpr,
                  HAS_CACHE: tl.constexpr, STATS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    row, k = r // TOP_K, r % TOP_K
    exact = tl.load(keep + r, r < N, False)
    gpu = tl.full((BLOCK,), False, tl.int1)
    if HAS_CACHE:
        s = tl.load(slots + row * GPU_K + k, (r < N) & (k < GPU_K), -1)
        gpu = exact & (k < GPU_K) & (s >= 0)
    eid = tl.load(ids + r, r < N, -1).to(tl.int32)
    tl.store(cpu_ids + r, tl.where(exact & ~gpu, eid, -1), r < N)
    if STATS:
        tl.atomic_add(counts, tl.sum(gpu.to(tl.int64), 0))
        tl.atomic_add(counts + 1, tl.sum(tl.load(cache_hit + r, r < N, False).to(tl.int64), 0))
        if HAS_CACHE:
            tl.atomic_add(counts + 2, tl.load(num_indices).to(tl.int64))


@triton.jit
def _gather_online_keys(keys, gathered, request_slots, sizes,
                        H: tl.constexpr, CAP: tl.constexpr, DUMMY: tl.constexpr,
                        BLOCK: tl.constexpr):
    row, token = tl.program_id(0), tl.program_id(1)
    ds = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(request_slots + row)
    size = tl.load(sizes + req)
    val = tl.load(keys + (req * CAP + token).to(tl.int64) * H + ds,
                  (req != DUMMY) & (token < size) & (ds < H), 0)
    tl.store(gathered + (row * CAP + token) * H + ds, val, ds < H)


@triton.jit
def _online_select(scores, online_ids, online_values, sizes, request_slots, ids, keep,
                   outputs, hits, TOP_K: tl.constexpr, H: tl.constexpr,
                   CAP: tl.constexpr, DUMMY: tl.constexpr,
                   BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
                   COSINE_THRESHOLD: tl.constexpr = None):
    route = tl.program_id(0)
    row = route // TOP_K
    req = tl.load(request_slots + row)
    ds = tl.arange(0, BLOCK_H)
    hit = tl.full((), False, tl.int1)
    value = tl.full((BLOCK_H,), 0, tl.float16)
    if (req != DUMMY) & ~tl.load(keep + route):
        size = tl.load(sizes + req)
        cs = tl.arange(0, BLOCK_C)
        s = tl.load(scores + row * CAP + cs, cs < size, -float("inf")).to(tl.float32)
        best, token = tl.max(s, 0, return_indices=True)
        ks = tl.arange(0, BLOCK_K)
        eid = tl.load(ids + route)
        match = tl.load(online_ids + (req * CAP + token) * TOP_K + ks,
                        ks < TOP_K, -1) == eid
        slot = tl.min(tl.where(match & (ks < TOP_K), ks, 2147483647), 0)
        accepted = best > 0 if COSINE_THRESHOLD is None else best >= COSINE_THRESHOLD
        hit = (size > 0) & accepted & (slot < TOP_K)
        value = tl.load(online_values + ((req * CAP + token).to(tl.int64) * TOP_K + slot) * H + ds,
                        hit & (ds < H), 0)
    tl.store(outputs + route * H + ds, value, ds < H)
    tl.store(hits + route, hit)


@triton.jit
def _mapped_offline(query, keys, counts, offsets, host_addr, ids, modality, keep,
                    online_hits, request_slots, outputs, hits, descriptor,
                    TOP_K: tl.constexpr, H: tl.constexpr, DIM: tl.constexpr,
                    CLUSTERS: tl.constexpr, DUMMY: tl.constexpr,
                    BLOCK_D: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
                    PREFILL: tl.constexpr, VISION: tl.constexpr, TEXT: tl.constexpr,
                    DECODE_THRESHOLD: tl.constexpr = None):
    route = tl.program_id(0)
    row = route // TOP_K
    hit = tl.full((), False, tl.int1)
    index = tl.full((), -1, tl.int64)
    req = tl.load(request_slots + row)
    if (req != DUMMY) & ~tl.load(keep + route) & ~tl.load(online_hits + route):
        eid = tl.load(ids + route).to(tl.int32)
        mod = tl.load(modality + row) if PREFILL else 1
        count = tl.load(counts + eid * 2 + mod)
        ds, cs = tl.arange(0, BLOCK_D), tl.arange(0, BLOCK_C)
        q = tl.load(query + row * DIM + ds, ds < DIM, 0).to(tl.float32)
        if not PREFILL and DECODE_THRESHOLD is not None:
            q = q / tl.maximum(tl.sqrt(tl.sum(q * q, 0)), 1e-6)
        key = tl.load(keys + ((eid * 2 + mod) * CLUSTERS + cs[:, None]) * DIM + ds[None, :],
                      (cs[:, None] < count) & (ds[None, :] < DIM), 0).to(tl.float32)
        # Match the reference offline kernel's FP32 reduction and lowest-index tie.
        score = tl.sum(key * q[None, :], 1)
        score = tl.where(cs < count, score, -float("inf"))
        best, cluster = tl.max(score, 0, return_indices=True)
        if PREFILL:
            accepted = best >= tl.where(mod == 0, VISION, TEXT)
        elif DECODE_THRESHOLD is not None:
            accepted = best >= DECODE_THRESHOLD
        else:
            accepted = best > 0
        hit = (count > 0) & accepted
        offset = tl.load(offsets + eid * 2 + mod).to(tl.int64)
        index = tl.where(hit, offset + cluster.to(tl.int64), -1)
        hs = tl.arange(0, BLOCK_H)
        ptr = host_addr.to(tl.uint64).to(tl.pointer_type(tl.float16))
        value = tl.load(ptr + index * H + hs.to(tl.int64), hit & (hs < H), 0)
        tl.store(outputs + route * H + hs, value, hs < H)
    tl.store(hits + route, hit)
    tl.store(descriptor + route, index)


@triton.jit
def _merge_update(x, ids, weights, keep, cpu_ids, gpu_slots, gpu_values, cpu_addr,
                   table_values, online_hits, offline_hits, request_slots, update_rows,
                   online_keys, online_ids, online_values, sizes, pointers, result,
                   counters, TOP_K: tl.constexpr, GPU_K: tl.constexpr,
                   H: tl.constexpr, CAP: tl.constexpr, DUMMY: tl.constexpr,
                   HAS_GPU: tl.constexpr, HAS_CPU: tl.constexpr, STATS: tl.constexpr,
                   BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
                   RENORMALIZE: tl.constexpr = True):
    row, block = tl.program_id(0), tl.program_id(1)
    req = tl.load(request_slots + row)
    hs = block * BLOCK_H + tl.arange(0, BLOCK_H)
    ks = tl.arange(0, BLOCK_K)
    route = row * TOP_K + ks
    exact = tl.load(keep + route, ks < TOP_K, False)
    on = tl.load(online_hits + route, ks < TOP_K, False)
    off = tl.load(offline_hits + route, ks < TOP_K, False)
    cpu = tl.load(cpu_ids + route, ks < TOP_K, -1) >= 0
    survivor = (exact | on | off) & (req != DUMMY) & (ks < TOP_K)
    w = tl.load(weights + route, ks < TOP_K, 0).to(tl.float32)
    w = tl.where(survivor, w, 0)
    denom = tl.sum(w, 0)
    normalized = tl.where(denom > 0, w / tl.maximum(denom, 1e-30), 0) if RENORMALIZE else w
    mask = (ks[:, None] < TOP_K) & (hs[None, :] < H)
    val = tl.load(table_values + route[:, None] * H + hs[None, :], mask, 0)
    if HAS_GPU:
        slot = tl.load(gpu_slots + row * GPU_K + ks, ks < GPU_K, -1)
        gv = tl.load(gpu_values + (row * GPU_K + ks[:, None]) * H + hs[None, :],
                     mask & (ks[:, None] < GPU_K) & (slot[:, None] >= 0), 0)
        val = tl.where(exact[:, None] & ~cpu[:, None], gv, val)
    if HAS_CPU:
        ptr = cpu_addr.to(tl.uint64).to(tl.pointer_type(tl.float16))
        cv = tl.load(ptr + route[:, None].to(tl.int64) * H + hs[None, :], mask & cpu[:, None], 0)
        val = tl.where(cpu[:, None], cv, val)
    val = tl.where(survivor[:, None], val, 0)
    out = tl.sum(val.to(tl.float32) * normalized[:, None], 0)
    tl.store(result + row * H + hs, out, hs < H)
    if CAP > 0:
        update = tl.load(update_rows + row)
        if (req != DUMMY) & (update >= 0):
            pos = (tl.load(pointers + req) + update) % CAP
            dst = (req * CAP + pos).to(tl.int64)
            xv = tl.load(x + row * H + hs, hs < H, 0)
            tl.store(online_keys + dst * H + hs, xv, hs < H)
            tl.store(online_values + (dst * TOP_K + ks[:, None]) * H + hs[None, :], val, mask)
            if block == 0:
                eid = tl.load(ids + route, ks < TOP_K, -1)
                tl.store(online_ids + dst * TOP_K + ks, tl.where(survivor, eid, -1), ks < TOP_K)
    if STATS:
        if (block == 0) & HAS_CPU:
            tl.atomic_add(counters + 7, H * 2 + TOP_K * 4)
        if (block == 0) & (req != DUMMY):
            tl.atomic_add(counters, tl.sum(cpu.to(tl.int64), 0))
            tl.atomic_add(counters + 1, tl.sum(on.to(tl.int64), 0))
            nh = tl.sum(off.to(tl.int64), 0)
            tl.atomic_add(counters + 2, nh)
            tl.atomic_add(counters + 3, TOP_K - tl.sum(survivor.to(tl.int64), 0))
            tl.atomic_add(counters + 4, (tl.sum(survivor.to(tl.int32), 0) == 0).to(tl.int64))
            tl.atomic_add(counters + 5, nh * H * 2)
            tl.atomic_add(counters + 6, tl.sum(cpu.to(tl.int64), 0) * H * 2)


@triton.jit
def _advance_online(sizes, pointers, request_slots, update_counts,
                    CAP: tl.constexpr, DUMMY: tl.constexpr):
    row = tl.program_id(0)
    req = tl.load(request_slots + row)
    n = tl.load(update_counts + row)
    if (req != DUMMY) & (n > 0):
        size = tl.load(sizes + req)
        pos = tl.load(pointers + req)
        tl.store(sizes + req, tl.minimum(CAP, size + n))
        tl.store(pointers + req, (pos + n) % CAP)


@triton.jit
def _audit_routes(ids, keep, cpu_ids, cache_hit, online, offline, request_slots, modality,
                  table_counts, counts, shared, TOP_K: tl.constexpr, H: tl.constexpr,
                  LAYER: tl.constexpr, LAYERS: tl.constexpr, PHASE: tl.constexpr,
                  DUMMY: tl.constexpr, HAS_CACHE: tl.constexpr, HAS_CPU: tl.constexpr,
                  ONLINE_ENABLED: tl.constexpr, BYPASS: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    req = tl.load(request_slots + row)
    if req == DUMMY:
        if HAS_CPU:
            tl.atomic_add(shared + 2, H * 2)
            tl.atomic_add(shared + 3, TOP_K * 4)
        return
    ks = tl.arange(0, BLOCK_K)
    mask = ks < TOP_K
    r = row * TOP_K + ks
    exact = tl.load(keep + r, mask, False)
    cpu = tl.load(cpu_ids + r, mask, -1) >= 0
    on = tl.load(online + r, mask, False)
    off = tl.load(offline + r, mask, False)
    hit = tl.load(cache_hit + r, mask, False)
    eq = exact & HAS_CACHE
    oq = mask & ~exact & ONLINE_ENABLED
    fq = mask & ~exact & ~on & (not BYPASS)
    missing = tl.full((BLOCK_K,), False, tl.int1)
    mod = tl.load(modality + row) if PHASE == 0 else 1
    if not BYPASS:
        eid = tl.load(ids + r, mask, 0).to(tl.int32)
        missing = fq & (tl.load(table_counts + eid * 2 + mod, mask, 0) == 0)
    nc = tl.sum(cpu.to(tl.int64), 0)
    ne = tl.sum(exact.to(tl.int64), 0)
    non = tl.sum(on.to(tl.int64), 0)
    noff = tl.sum(off.to(tl.int64), 0)
    nfq = tl.sum(fq.to(tl.int64), 0)
    nhit = tl.sum(hit.to(tl.int64), 0)
    neq = tl.sum(eq.to(tl.int64), 0)
    noq = tl.sum(oq.to(tl.int64), 0)
    nm = tl.sum(missing.to(tl.int64), 0)
    offsets = tl.arange(0, 32)
    value = tl.full((32,), 0, tl.int64)
    value = tl.where(offsets == 0, 1, value)
    value = tl.where(offsets == 1, TOP_K, value)
    value = tl.where(offsets == 2, nc, value)
    value = tl.where(offsets == 3, ne - nc, value)
    value = tl.where(offsets == 4, neq, value)
    value = tl.where(offsets == 5, nhit, value)
    value = tl.where(offsets == 6, neq - nhit, value)
    value = tl.where(offsets == 7, noq, value)
    value = tl.where(offsets == 8, non, value)
    value = tl.where(offsets == 9, noq - non, value)
    value = tl.where(offsets == 10, nfq, value)
    value = tl.where(offsets == 11, noff, value)
    value = tl.where(offsets == 12, nfq - noff, value)
    value = tl.where(offsets == 13, nm, value)
    value = tl.where(offsets == 14, nfq - noff - nm, value)
    value = tl.where(offsets == 15, TOP_K - ne - non - noff, value)
    value = tl.where((offsets == 16) & HAS_CPU, H * 2, value)
    value = tl.where((offsets == 17) & HAS_CPU, TOP_K * 4, value)
    value = tl.where(offsets == 18, nc * H * 2, value)
    value = tl.where(offsets == 19, noff * H * 2, value)
    value = tl.where(offsets == 20, (ne + non + noff == 0).to(tl.int64), value)
    base = (((req * LAYERS + LAYER) * 2 + PHASE) * 2 + mod) * 21
    tl.atomic_add(counts + base + offsets, value, offsets < 21)


@triton.jit
def _audit_shared(shared, num_indices, EXPERT_BYTES: tl.constexpr, HAS_CACHE: tl.constexpr,
                  FLAG_SYNC: tl.constexpr):
    if HAS_CACHE:
        fetched = tl.load(num_indices).to(tl.int64)
        tl.atomic_add(shared, fetched)
        tl.atomic_add(shared + 1, fetched * EXPERT_BYTES)
    if FLAG_SYNC:
        tl.atomic_add(shared + 4, 8)
        tl.atomic_add(shared + 5, 4)
