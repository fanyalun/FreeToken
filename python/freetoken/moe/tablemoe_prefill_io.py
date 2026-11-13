"""Sparse staging and independent copy descriptors for TableMoE Prefill."""

import torch
import triton

from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
from freetoken.kernel.triton.tablemoe_prefill_io import (
    _accumulate_fill_scores, _compact_stage_experts, _map_route_groups, _mark_stage_experts,
    _plan_prefill_fill, _publish_prefill_fill, _snapshot_copy_plan,
)


class PrefillCacheFill:
    def __init__(self, cache, stage_pointers, stats=None):
        self.cache = cache
        self.stage_pointers = stage_pointers
        self.stats = (stats if stats is not None else
                      torch.zeros(cache.num_layers, 4, device=cache.device, dtype=torch.int64))
        self.scores = torch.zeros(cache.num_layers, cache.num_experts, device=cache.device, dtype=torch.float32)
        self.candidates = torch.zeros_like(self.scores, dtype=torch.int32)
        self.activated = torch.zeros_like(self.candidates)
        self.old_slots = torch.empty_like(cache.slot_for_id)
        # A fixed layer capacity keeps snapshot memory independent of cache resizing.
        self.old_owners = torch.empty(cache.num_layers, cache.num_experts, device=cache.device, dtype=torch.int32)
        self.old_usage = torch.empty_like(self.old_owners, dtype=torch.int64)
        self.sources = torch.empty(cache.num_experts, device=cache.device, dtype=torch.int32)
        self.destinations = torch.empty_like(self.sources)
        self.copy_count = torch.zeros(1, device=cache.device, dtype=torch.int64)
        self.restore_count = torch.zeros_like(self.copy_count)
        self.admission_count = torch.zeros_like(self.copy_count)
        covered = cache._inval[0]
        covered = (range(*covered.indices(cache.cache_size)) if isinstance(covered, slice)
                   else covered.cpu().tolist())
        covered = set(covered)
        self.deferred_pools = frozenset(
            pool for pool, (base, size) in enumerate(zip(cache.pool_bases, cache.pool_sizes))
            if any(slot in covered for slot in range(base, base + size)))
        self.active = False

    @property
    def buffers(self):
        return (self.stats, self.scores, self.candidates, self.activated, self.old_slots, self.old_owners,
                self.old_usage, self.sources, self.destinations, self.copy_count,
                self.restore_count, self.admission_count)

    def reset(self):
        self.active = False
        self.scores.zero_()
        self.candidates.zero_()
        self.activated.zero_()

    def record(self, layer, ids, keep, weights, attention, *, request_slots=None, dummy_slot=-1):
        cache = self.cache
        if layer == 0:
            self.old_slots.copy_(cache.slot_for_id)
            self.old_owners.view(-1)[:cache.cache_size].copy_(cache.id_of_slot)
            self.old_usage.view(-1)[:cache.cache_size].copy_(cache.usage)
            self.reset()
            self.active = True
        if not self.active:
            raise RuntimeError("TableMoE Prefill cache fill must begin at the first MoE layer")
        _accumulate_fill_scores[(triton.cdiv(ids.numel(), 256),)](
            ids.contiguous(), keep.contiguous(), weights.contiguous(),
            attention.contiguous() if attention is not None else weights,
            request_slots.contiguous() if request_slots is not None else ids,
            self.scores[layer], self.candidates[layer], self.activated[layer],
            ids.numel(), ids.shape[1], cache.num_experts, dummy_slot,
            request_slots is not None, attention is not None, 256)

    def _copy(self, layer, host):
        cache = self.cache
        pool = cache.pool_of_layer[layer]
        if host and layer in cache._unpinned_layers:
            count = int(self.copy_count.item())
            for expert, slot in zip(self.sources[:count].tolist(), self.destinations[:count].tolist()):
                for (banks, _), destination in zip(cache.banks, cache.pool_bank_views(layer)):
                    destination[slot].copy_(banks[layer][expert])
        elif cache._copy_fused_ok:
            fast_index_copy_multi_jit(
                cache._copy_dst_ptrs_pool[pool], cache._copy_src_ptrs[layer] if host else self.stage_pointers,
                cache._copy_feat_bytes, self.destinations, self.sources, self.copy_count)
        else:
            from freetoken.kernel import fast_index_copy_jit

            sources = ([banks[layer] for banks, _ in cache.banks] if host else cache.stage_views(layer, 0))
            for source, destination in zip(sources, cache.pool_bank_views(layer)):
                fast_index_copy_jit(destination, self.destinations, source, self.sources, self.copy_count)

    def _fill(self, layer, host):
        cache = self.cache
        pool = cache.pool_of_layer[layer]
        size, base = cache.pool_sizes[pool], cache.pool_bases[pool]
        block = triton.next_power_of_2(max(cache.num_experts, size))
        _plan_prefill_fill[(1,)](
            self.scores[layer], self.candidates[layer], self.activated[layer],
            self.old_slots[layer], self.old_owners, cache.slot_for_id, cache.id_of_slot, cache.usage,
            self.sources, self.destinations, self.copy_count, self.restore_count, self.admission_count,
            cache.num_experts, size, base, host, block)
        self._copy(layer, host)
        _publish_prefill_fill[(1,)](
            self.sources, self.destinations, self.copy_count, self.restore_count, self.admission_count,
            cache.slot_for_id, cache.id_of_slot, cache.usage, self.old_usage,
            cache.expert_recency, cache.step, self.stats,
            layer, cache.num_experts, base, host, block)

    def finish_layer(self, layer):
        cache = self.cache
        if cache.pool_of_layer[layer] not in self.deferred_pools:
            self._fill(layer, False)
        cache.release_prefill_layer(layer)
        if layer == cache.num_layers - 1:
            # Later layers overwrite staging, so overlapping pools can only publish here.
            for previous in range(cache.num_layers):
                if cache.pool_of_layer[previous] in self.deferred_pools:
                    self._fill(previous, True)
            self.active = False


class SparsePrefillStage:
    def __init__(self, cache, stats=None):
        self.cache = cache
        self.layer = 0
        self.bank_pointers = tuple(bank.data_ptr() for bank in cache.bank_caches.values())
        views = cache.stage_views(0, buffer_id=0)
        self.destination_ptrs = (torch.tensor([view.data_ptr() for view in views],
                                             device=cache.device, dtype=torch.int64)
                                 if cache._copy_fused_ok else None)
        self.cache_fill = PrefillCacheFill(cache, self.destination_ptrs, stats)

    @property
    def views(self):
        return self.cache.stage_views(self.layer, buffer_id=0)

    @property
    def buffers(self):
        return (self.destination_ptrs, *self.cache_fill.buffers)

    @property
    def startup_h2d_buffers(self):
        return (self.destination_ptrs,)

    def prepare(self, layer, route_slots):
        cache = self.cache
        self.layer = layer
        # Staging borrows arena rows; their old decode tags cannot survive an overwrite.
        cache._invalidate_prefill_buffer(0)
        cache.active_mask.zero_()
        _mark_stage_experts[(triton.cdiv(route_slots.numel(), 256),)](
            route_slots, cache.active_mask, route_slots.numel(), cache.num_experts, 256)
        _compact_stage_experts[(1,)](
            cache.active_mask, cache.src_indices, cache.evict_slots, cache.num_indices,
            cache.num_experts, triton.next_power_of_2(cache.num_experts))
        return self

    def copy(self):
        cache = self.cache
        if self.layer in cache._unpinned_layers:
            for expert in cache.src_indices[:int(cache.num_indices.item())].tolist():
                for (sources, _), destination in zip(cache.banks, self.views):
                    destination[expert].copy_(sources[self.layer][expert])
        elif cache._copy_fused_ok:
            fast_index_copy_multi_jit(
                self.destination_ptrs, cache._copy_src_ptrs[self.layer], cache._copy_feat_bytes,
                cache.evict_slots, cache.src_indices, cache.num_indices)
        else:
            from freetoken.kernel import fast_index_copy_jit

            for (sources, _), destination in zip(cache.banks, self.views):
                fast_index_copy_jit(destination, cache.evict_slots, sources[self.layer],
                                    cache.src_indices, cache.num_indices)


class GroupPrefillCopyPlan:
    def __init__(self, cache, layer, route_slots, group_size, *, stage=None):
        if group_size < 1:
            raise ValueError("prefill expert group size must be positive")
        if not cache._copy_fused_ok:
            raise ValueError("grouped TableMoE prefill requires mapped, aligned expert banks")
        self.cache = cache
        self.stage = stage
        self.group_size = group_size
        self.num_groups = triton.cdiv(cache.num_experts, group_size)
        self.sources = torch.empty(self.num_groups, group_size, device=route_slots.device, dtype=torch.int32)
        self.destinations = torch.empty_like(self.sources)
        self.counts = torch.empty(self.num_groups, 1, device=route_slots.device, dtype=torch.int64)
        slot_count = cache.num_experts if stage is not None else cache.cache_size
        self.slot_groups = torch.empty(slot_count, device=route_slots.device, dtype=torch.int32)
        self.route_group_storage = torch.empty(route_slots.numel(), device=route_slots.device, dtype=torch.int32)
        self.route_groups = self.route_group_storage.view(route_slots.shape)
        self.prepare(layer, route_slots)

    @property
    def buffers(self):
        return dict(sources=self.sources, destinations=self.destinations, counts=self.counts,
                    slot_groups=self.slot_groups, route_groups=self.route_group_storage)

    def prepare(self, layer, route_slots):
        cache = self.cache
        if route_slots.numel() > self.route_group_storage.numel():
            raise ValueError("grouped prefill routes exceed the prepared copy-plan capacity")
        slot_count = cache.num_experts if self.stage is not None else cache.cache_size
        if slot_count != self.slot_groups.numel():
            raise ValueError("grouped prefill copy plan must be recreated after cache resizing")
        if route_slots.dtype != torch.int32 or not route_slots.is_contiguous():
            raise ValueError("grouped prefill routes must be contiguous int32 cache slots")
        if route_slots.device != self.route_group_storage.device:
            raise ValueError("grouped prefill routes must use the prepared copy-plan device")
        self.route_groups = self.route_group_storage[:route_slots.numel()].view(route_slots.shape)
        self.layer = layer
        self.slot_groups.fill_(-1)
        capacity = self.sources.numel()
        _snapshot_copy_plan[(triton.cdiv(capacity, 256),)](
            cache.src_indices, cache.evict_slots, cache.num_indices,
            self.sources, self.destinations, self.counts, self.slot_groups,
            self.group_size, capacity, slot_count, 256)
        _map_route_groups[(triton.cdiv(route_slots.numel(), 256),)](
            route_slots, self.slot_groups, self.route_groups, route_slots.numel(), slot_count, 256)
        return self

    def copy_group(self, group_id):
        if not 0 <= group_id < self.num_groups:
            raise ValueError("prefill copy group is out of range")
        cache = self.cache
        fast_index_copy_multi_jit(
            self.stage.destination_ptrs if self.stage is not None else cache._copy_dst_ptrs,
            cache._copy_src_ptrs[self.layer], cache._copy_feat_bytes,
            self.destinations[group_id], self.sources[group_id], self.counts[group_id])
