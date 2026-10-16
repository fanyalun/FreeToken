from __future__ import annotations

from collections import Counter, OrderedDict

import torch


class NoEvictableSlot(RuntimeError):
    pass


class ExpertLeases:
    def __init__(self, owners):
        self.owners = list(owners)
        self.slots = {owner: slot for slot, owner in enumerate(owners) if owner >= 0}
        if len(self.slots) != sum(owner >= 0 for owner in owners):
            raise ValueError("duplicate expert owners in cache")
        self.lru = OrderedDict((slot, None) for slot in range(len(owners)))
        self.refs = Counter()
        self.rows = {}
        self.pending = {}
        self.readers = {}
        self.epoch = None

    def begin(self, epoch):
        if self.epoch is not None or self.pending or self.readers or self.rows:
            raise RuntimeError("previous speculative window still owns resources")
        self.epoch = epoch

    def observe(self, epoch, loop, layer, rows, experts):
        if epoch != self.epoch:
            raise RuntimeError("stale speculative window")
        for index, ids in enumerate(rows):
            key = loop, layer, index
            if key in self.rows:
                raise RuntimeError("duplicate route observation")
            owners = {layer * experts + expert for expert in ids}
            self.rows[key] = owners
            self.refs.update(owners)

    def decide(self, epoch, loop, consumed):
        if epoch != self.epoch:
            raise RuntimeError("stale speculative window")
        if consumed < 1:
            raise RuntimeError("invalid provisional decision")
        self._release(lambda key: key[0] == loop and key[2] >= consumed)

    def release_layer(self, layer):
        self._release(lambda key: key[1] == layer)

    def _release(self, predicate):
        for key in list(self.rows):
            if predicate(key):
                for owner in self.rows.pop(key):
                    self.refs[owner] -= 1
                    if not self.refs[owner]:
                        del self.refs[owner]

    def reap(self):
        for table in (self.pending, self.readers):
            for slot, event in list(table.items()):
                if event.query():
                    del table[slot]

    def ready(self, owner):
        self.reap()
        return owner in self.slots and self.slots[owner] not in self.pending

    def capacity(self):
        self.reap()
        blocked = sum(not self.refs[self.owners[slot]] for slot in self.pending.keys() | self.readers.keys())
        return {"protected_experts": len(self.refs), "blocked_slots": blocked,
                "available_slots": len(self.owners) - len(self.refs) - blocked}

    def allocate(self, owner, active=()):
        self.reap()
        if owner in self.slots:
            slot = self.slots[owner]
            self.lru.move_to_end(slot)
            return slot, False
        active = set(active)
        slot = next((slot for slot in self.lru
                     if not self.refs[self.owners[slot]] and self.owners[slot] not in active
                     and slot not in self.pending and slot not in self.readers), None)
        if slot is None:
            raise NoEvictableSlot("no evictable expert slot; protected experts cannot be displaced")
        previous = self.owners[slot]
        self.slots.pop(previous, None)
        self.owners[slot] = owner
        self.slots[owner] = slot
        self.lru.move_to_end(slot)
        return slot, True

    def end(self):
        if self.pending or self.readers:
            raise RuntimeError("wait for all expert readers and copies before ending the window")
        self._release(lambda _: True)
        self.epoch = None


class SpeculativeExpertCache:
    def __init__(self, cache, stream, *, prefetch=True, use_graph=False, audit=False,
                 layer_quota=4, gap_capacity=0):
        if cache.quant_format != "bf16" or cache.cpu_layer_ids or cache._unpinned_layers:
            raise ValueError("three-level decoding requires BF16 experts with pinned GPU-offload banks")
        from freetoken.layers.moe import _HYBRID_OVERLAP, _HYBRID_WINDOW
        if cache.decode_target == "hybrid" and not _HYBRID_WINDOW:
            raise ValueError("three-level hybrid decoding requires FREETOKEN_HYBRID_WINDOW=1")
        self.hybrid_overlap = _HYBRID_OVERLAP
        self.cache, self.stream = cache, stream
        self.prefetch, self.use_graph, self.audit = prefetch, use_graph, audit
        self.copy_stream = torch.cuda.Stream(device=cache.device)
        self.layer_quota, self.gap_capacity = layer_quota, gap_capacity
        self.input_offset = 0
        self.profiler = None
        self.queued = OrderedDict()
        self.ledger = None
        self.graphs = {}
        self.graph_replays = 0
        self.active = False
        self.records = []

    def begin(self, epoch):
        if self.active or getattr(self.cache, "speculative_controller", None) is not None:
            raise RuntimeError("expert cache already has an active speculative owner")
        owners = self.cache.id_of_slot.cpu().tolist()
        usage = self.cache.usage.cpu().tolist()
        self.ledger = ExpertLeases(owners)
        self.ledger.lru = OrderedDict((slot, None) for slot in sorted(range(len(owners)), key=lambda i: (usage[i], i)))
        self.ledger.begin(epoch)
        self.epoch, self.phase, self.loop = epoch, "preverify", 0
        self.records, self.stats = [], Counter()
        self.queued.clear()
        self.transfers, self.consumers = [], []
        self.first_demands = []
        self.origin = torch.cuda.Event(enable_timing=True) if self.audit else None
        if self.origin is not None:
            self.origin.record(self.stream)
            self.copy_stream.wait_event(self.origin)
        self.recency = self.cache.expert_recency.flatten().cpu().tolist()
        self.initial = set(owners) - {-1}
        self.prefetched = set()
        self.used_prefetch = set()
        self.clock = int(self.cache.step.item())
        self.cache.speculative_controller = self
        self.active = True

    def set_phase(self, phase, loop=0):
        self.phase, self.loop = phase, loop

    def _cancel_queued(self, *, all_tasks=False):
        for owner in list(self.queued):
            if all_tasks or not self.ledger.refs[owner]:
                del self.queued[owner]
                self.stats["discarded_queued_prefetch" if all_tasks else "cancelled_queued_prefetch"] += 1

    def _pump_prefetch(self, limit, *, layer=None):
        self.ledger.reap()
        submitted = 0
        for owner in sorted(self.queued, key=lambda owner: (*self.queued[owner], owner)):
            if layer is not None and self.queued[owner][0] != layer:
                continue
            if owner in self.ledger.slots:
                del self.queued[owner]
                continue
            if submitted >= limit:
                break
            try:
                self._copy(owner, (), prefetch=True)
            except NoEvictableSlot:
                break
            del self.queued[owner]
            submitted += 1
        return submitted

    def _queue_prefetch(self, layer, rows, demanded):
        for index, ids in enumerate(rows):
            for column, expert in enumerate(ids):
                owner = layer * self.cache.num_experts + expert
                if owner not in demanded and owner not in self.ledger.slots:
                    priority = layer, self.input_offset + index, column
                    self.queued[owner] = min(priority, self.queued.get(owner, priority))
        self.stats["queued_prefetch_peak"] = max(self.stats["queued_prefetch_peak"], len(self.queued))
        self.stats["layer_prefetch_loads"] += self._pump_prefetch(self.layer_quota, layer=layer)

    def finish_loop(self, continuing):
        if continuing and self.prefetch:
            self.stats["gap_prefetch_loads"] += self._pump_prefetch(max(0, self.gap_capacity - 1))
        self._cancel_queued(all_tasks=True)

    def _copy(self, owner, active, *, prefetch):
        ledger, cache = self.ledger, self.cache
        if not prefetch:
            self.queued.pop(owner, None)
        try:
            slot, missing = ledger.allocate(owner, active)
        except NoEvictableSlot:
            blocked = next((slot for slot in ledger.lru
                            if not ledger.refs[ledger.owners[slot]] and ledger.owners[slot] not in active
                            and (slot in ledger.pending or slot in ledger.readers)), None)
            if prefetch or blocked is None:
                raise
            for table in (ledger.pending, ledger.readers):
                if blocked in table:
                    table[blocked].synchronize()
            self.stats["demand_slot_waits"] += 1
            slot, missing = ledger.allocate(owner, active)
        if missing:
            layer, expert = divmod(owner, cache.num_experts)
            # One IO stream preserves demand-before-background submission order.
            stream = self.copy_stream
            with torch.cuda.stream(stream):
                begin = torch.cuda.Event(enable_timing=True) if self.audit else None
                if begin is not None:
                    begin.record(stream)
                for sources, bank in cache.banks:
                    bank[slot].copy_(sources[layer][expert], non_blocking=True)
                event = torch.cuda.Event(enable_timing=self.audit)
                event.record(stream)
            if begin is not None:
                self.transfers.append((owner, slot, "cache", self.phase, self.loop, prefetch, begin, event))
            ledger.pending[slot] = event
            self.stats["prefetch_loads" if prefetch else "demand_loads"] += 1
            if prefetch:
                self.prefetched.add(owner)
            else:
                self.prefetched.discard(owner)
        if not prefetch and slot in ledger.pending:
            self.stream.wait_event(ledger.pending[slot])
        return slot

    def _compute(self, layer, hidden, weights, slots):
        cache = self.cache
        def run(x, w, ids):
            return layer._expert_gemm(cache, x, w, ids, views=cache.bank_views(), n=None,
                                      alphas=cache.alphas_for_slots(layer.layer_id), is_prefill=False)
        if not self.use_graph:
            return run(hidden, weights, slots)
        key = layer.layer_id, hidden.shape[0]
        if key not in self.graphs:
            x, w, ids = hidden.clone(), weights.clone(), slots.clone()
            run(x, w, ids)
            self.stream.synchronize()
            x.copy_(hidden)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                output = run(x, w, ids)
            self.graphs[key] = graph, x, w, ids, output
        graph, x, w, ids, output = self.graphs[key]
        x.copy_(hidden)
        w.copy_(weights)
        ids.copy_(slots)
        graph.replay()
        self.graph_replays += 1
        return output

    def route(self, layer, hidden, weights, ids):
        cache, ledger = self.cache, self.ledger
        rows = ids.cpu().tolist()
        li, experts = layer.layer_id, cache.num_experts
        if self.audit and self.phase == "preverify" and li == 0:
            event = torch.cuda.Event(enable_timing=True)
            event.record(self.stream)
            self.first_demands.append((self.loop, event))
        profile = self.profiler
        if profile is not None and self.phase == "preverify":
            profile.route_start(self.loop, li, len(rows), self.stream)
        ledger.reap()
        ready_owners = {owner for owner, slot in ledger.slots.items() if slot not in ledger.pending}
        if self.phase == "preverify":
            ledger.observe(self.epoch, self.loop, li, rows, experts)
        else:
            # Routing is now exact; this layer's predicted experts need no further lease.
            ledger.release_layer(li)
            self._cancel_queued()
        ready = [[li * experts + expert in ready_owners for expert in row] for row in rows]
        resident = torch.tensor(ready, dtype=torch.bool, device=ids.device)
        if self.phase == "preverify":
            rank = weights.argsort(dim=-1, descending=True, stable=True).argsort(dim=-1)
            keep = resident | (rank < min(4, ids.shape[1]))
        else:
            keep = torch.ones_like(resident)
        kept = keep.cpu().tolist()
        demanded = {li * experts + expert for row, mask in zip(rows, kept)
                    for expert, retained in zip(row, mask) if retained}
        pending_before = {owner for owner in demanded if owner in ledger.slots
                          and ledger.slots[owner] in ledger.pending}
        if self.phase == "verify":
            actual = {li * experts + expert for row in rows for expert in row}
            timely = actual & self.prefetched & ready_owners
            self.used_prefetch.update(timely)
            self.stats["target_unique_experts"] += len(actual)
            self.stats["target_initial_missing"] += len(actual - self.initial)
            self.stats["target_timely_prefetch"] += len(timely - self.initial)
        gpu = set(demanded)
        if cache.decode_target == "hybrid":
            from .offload_kernels import _HYBRID_FETCH_BY_RECENCY
            missing = sorted((owner for owner in demanded if owner not in ledger.slots),
                             key=lambda owner: (-self.recency[owner], owner) if _HYBRID_FETCH_BY_RECENCY else (0, owner))
            fraction = min(65536, max(0, round(cache.hybrid_fetch_fraction * 65536)))
            fetch = cache.hybrid_max_fetch * hidden.shape[0]
            if fraction:
                lo = len(missing) * fraction // 65536
                cost_lo = max(lo * (65536 - fraction), (len(missing) - lo) * fraction)
                cost_hi = max((lo + 1) * (65536 - fraction), (len(missing) - lo - 1) * fraction)
                fetch = lo if cost_lo <= cost_hi else lo + 1
            gpu.difference_update(missing[max(0, fetch):])
        try:
            for owner in sorted(gpu):
                self._copy(owner, demanded, prefetch=False)
        except NoEvictableSlot as error:
            self.stats["capacity_failures"] += 1
            raise NoEvictableSlot(f"{self.phase} layer {li} cannot fit its exact GPU working set with R=0; "
                                 f"capacity={ledger.capacity()}, demanded={len(gpu)}") from error
        mapping = ledger.slots
        if profile is not None and self.phase == "preverify":
            profile.demand_done(self.loop, li, len(rows), self.stream)
        if self.phase == "preverify" and self.prefetch:
            self._queue_prefetch(li, rows, gpu)
        gpu_mask = torch.tensor([[li * experts + expert in gpu and retained
                                  for expert, retained in zip(row, mask)] for row, mask in zip(rows, kept)],
                                dtype=torch.bool, device=ids.device)
        # A zero-weight route must not read a slot concurrently overwritten by prefetch.
        fallback = mapping[min(gpu)] if gpu else 0
        gpu_ids = torch.tensor([[mapping[li * experts + expert] if retained and li * experts + expert in gpu else fallback
                                 for expert, retained in zip(row, mask)] for row, mask in zip(rows, kept)],
                               dtype=ids.dtype, device=ids.device)
        gpu_weights = torch.where(gpu_mask, weights, 0).contiguous()
        pending_cpu = None
        cpu_result = None
        if cache.decode_target == "hybrid":
            cpu_ids = torch.where(keep & ~gpu_mask, ids, -1).contiguous()
            pending_cpu = cache.cpu_executor.decode_submit(li, hidden, weights, cpu_ids)
            if not self.hybrid_overlap:
                cpu_result = cache.cpu_executor.decode_sync(pending_cpu)
        if self.audit and self.phase == "verify":
            consumer = torch.cuda.Event(enable_timing=True)
            consumer.record(self.stream)
            self.consumers.append((li, consumer))
        if profile is not None and gpu and self.phase == "preverify":
            profile.capture_probe(layer, cache, hidden, gpu_weights, gpu_ids)
        result = self._compute(layer, hidden, gpu_weights, gpu_ids) if gpu else torch.zeros_like(hidden)
        reader = torch.cuda.Event()
        reader.record(self.stream)
        for owner in gpu:
            ledger.readers[ledger.slots[owner]] = reader
        if pending_cpu is not None:
            result = result + (cpu_result if cpu_result is not None else cache.cpu_executor.decode_sync(pending_cpu))
        self.clock += 1
        for owner in demanded:
            self.recency[owner] = self.clock
        self.stats["pending_demand_experts"] += len(pending_before)
        self.stats[f"{self.phase}_native_routes"] += ids.numel()
        self.stats[f"{self.phase}_retained_routes"] += sum(sum(mask) for mask in kept)
        self.stats["protected_peak"] = max(self.stats["protected_peak"], len(ledger.refs))
        self.records.append({"phase": self.phase, "loop": self.loop, "layer": li, "ids": rows,
                             "ready": ready} if self.audit else
                            {"phase": self.phase, "loop": self.loop, "layer": li, "ids": rows})
        return result

    def decide(self, loop, consumed):
        self.ledger.decide(self.epoch, loop, consumed)
        self._cancel_queued()

    def transfer_audit(self):
        return {"scope": "CUDA event times relative to window start; expert copies include all weight banks",
                "transfers": [{"owner": owner, "slot": slot, "storage": storage, "phase": phase, "loop": loop,
                               "prefetch": prefetch, "start_ms": self.origin.elapsed_time(begin),
                               "end_ms": self.origin.elapsed_time(end)}
                              for owner, slot, storage, phase, loop, prefetch, begin, end in self.transfers],
                "target_consumers": [{"layer": li, "start_ms": self.origin.elapsed_time(event)}
                                     for li, event in self.consumers],
                "preverify_first_demands": [{"loop": loop, "start_ms": self.origin.elapsed_time(event)}
                                            for loop, event in self.first_demands]}

    def close(self):
        if not self.active:
            return
        self._cancel_queued(all_tasks=True)
        self.copy_stream.synchronize()
        self.stream.synchronize()
        ledger, cache = self.ledger, self.cache
        ledger.reap()
        ledger.end()
        cache.slot_for_id.fill_(-1)
        owners = torch.tensor(ledger.owners, device=cache.device, dtype=cache.id_of_slot.dtype)
        cache.id_of_slot.copy_(owners)
        mapping = [-1] * (cache.num_layers * cache.num_experts)
        for owner, slot in ledger.slots.items():
            mapping[owner] = slot
        cache.slot_for_id.copy_(torch.tensor(mapping, device=cache.device, dtype=cache.slot_for_id.dtype).reshape_as(cache.slot_for_id))
        stamps = [0] * cache.cache_size
        for stamp, slot in enumerate(ledger.lru, self.clock + 1):
            stamps[slot] = stamp
        cache.usage.copy_(torch.tensor(stamps, device=cache.device, dtype=cache.usage.dtype))
        cache.step.fill_(max(stamps))
        cache.expert_recency.copy_(torch.tensor(self.recency, device=cache.device, dtype=cache.expert_recency.dtype).reshape_as(cache.expert_recency))
        cache.speculative_controller = None
        expert_bytes = sum(bank[0].numel() * bank.element_size() for _, bank in cache.banks)
        self.stats["expert_bytes"] = expert_bytes
        for key in ("prefetch_loads", "demand_loads", "target_timely_prefetch"):
            self.stats[key + "_bytes"] = self.stats[key] * expert_bytes
        self.active = False
