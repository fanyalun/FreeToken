"""Mapped CPU values and fixed request state for TableMoE eager/graph execution."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
from freetoken.kernel.triton import tablemoe as kernels
from freetoken.moe.tablemoe import TableMoeRuntime, prefill_keep_mask
from freetoken.utils import init_logger

logger = init_logger(__name__)


def gpu_storage_bytes(objects, excluded=()) -> int:
    seen = {t.untyped_storage().data_ptr() for t in excluded if t.is_cuda}

    def visit(obj):
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            storage = obj.untyped_storage()
            key = storage.data_ptr()
            if key not in seen:
                seen.add(key)
                return storage.nbytes()
        elif isinstance(obj, dict):
            return sum(visit(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return sum(visit(v) for v in obj)
        return 0

    return visit(objects)


class TableMoeOptimizedRuntime(TableMoeRuntime):
    def __init__(self, config):
        super().__init__(config)
        self._init_optimized_state()

    def _prepare_gpu_assets(self):
        self.gpu_keys, self.gpu_counts, self.host_values = {}, {}, {}
        self.value_offsets, self.value_addresses = {}, {}
        if getattr(self, "bypass_tables", False):
            return
        for layer in range(self.num_layers):
            data = [(key, torch.load(path, map_location="cpu", weights_only=True, mmap=True))
                    for key, path in self.offline_paths.items() if key[0] == layer]
            clusters = max([1] + [d["key"].shape[0] for _, d in data])
            width = self.pca[layer, "text"][0].shape[1]
            keys = torch.zeros(self.num_experts, 2, clusters, width, dtype=torch.float16)
            counts = torch.zeros(self.num_experts, 2, dtype=torch.int32)
            offsets = torch.full((self.num_experts, 2), -1, dtype=torch.int64)
            total = sum(d["value"].shape[0] for _, d in data)
            try:
                values = alloc_pinned_tensor(max(1, total), self.hidden, dtype=torch.float16)
                address = device_ptr(values)
            except RuntimeError as exc:
                raise RuntimeError("optimized TableMoE requires device-mapped pinned CPU memory") from exc
            cursor = 0
            for (_, expert, modality), d in data:
                mod = 0 if modality == "vision" else 1
                n = d["key"].shape[0]
                keys[expert, mod, :n] = F.normalize(d["key"], dim=-1, eps=1e-6)
                counts[expert, mod], offsets[expert, mod] = n, cursor
                values[cursor:cursor + n].copy_(d["value"])
                cursor += n
            self.gpu_keys[layer], self.gpu_counts[layer] = keys.to(self.device), counts.to(self.device)
            self.value_offsets[layer] = offsets.to(self.device)
            self.host_values[layer], self.value_addresses[layer] = values, address
            for mod in ("vision", "text"):
                path = Path(self.config.tablemoe_pca_dir) / f"layer_{layer}" / f"L{layer}_{mod}_pca.pt"
                d = torch.load(path, map_location="cpu", weights_only=True)
                proj = d["proj"].to(device=self.device, dtype=torch.float16)
                mean = d["mean"].to(device=self.device, dtype=torch.float16)
                self.pca[layer, mod] = (proj, -mean @ proj)

    def _init_optimized_state(self):
        self.top_k = self.config.model_config.num_experts_per_tok
        self.dummy_slot = self.config.max_running_req
        self.capacity = self.config.tablemoe_online_cache_size
        self.stats_enabled = self.config.tablemoe_stats or bool(getattr(self.config, "tablemoe_report_path", None))
        self.skip_lookup = getattr(self.config, "tablemoe_skip_lookup", False)
        self.external_mask = getattr(self.config, "moe_policy", "native") != "native"
        self.prefill_uses_tablemoe = (not self.external_mask or
                                    getattr(self.config, "modes_prefill_policy", "modes") == "tablemoe")
        self.bypass_tables = self.skip_lookup or (not self.external_mask and self.config.tablemoe_prefill_keep_rate == 1 and
                              self.config.tablemoe_decode_keep_k == self.top_k)
        self._owners, self._uid_slots, self.workspaces = {}, {}, {}
        self.batch_slots = torch.empty(max(128, self.config.max_extend_tokens + self.dummy_slot),
                                       device=self.device, dtype=torch.int32)
        self.graph_slots = None
        self.fixed_online = {}
        cap = self.capacity if self.external_mask else max(1, self.capacity)
        slots = self.dummy_slot + 1
        for layer in range(self.num_layers):
            self.fixed_online[layer] = dict(
                keys=torch.empty(slots, cap, self.hidden, device=self.device, dtype=torch.float16),
                ids=torch.empty(slots, cap, self.top_k, device=self.device, dtype=torch.int64),
                values=torch.empty(slots, cap, self.top_k, self.hidden, device=self.device, dtype=torch.float16),
                size=torch.zeros(slots, device=self.device, dtype=torch.int32),
                pointer=torch.zeros(slots, device=self.device, dtype=torch.int32),
            )
        self._route_counts = torch.zeros(8, device=self.device, dtype=torch.int64)
        self.audit = None
        if self.stats_enabled:
            from freetoken.moe.tablemoe_stats import RequestAudit

            self.audit = RequestAudit(self)
        self.collector = None
        if path := getattr(self.config, "tablemoe_collect_config", None):
            from freetoken.moe.tablemoe_builder import InputCollector
            from freetoken.moe.tablemoe_artifacts import read_json

            collection = read_json(path)
            self.collector = InputCollector(collection["output"], collection["builder"], self.hidden, self.num_experts,
                                            identity=collection["input_sha256"])
        self.lookup_stream = torch.cuda.Stream(device=self.device)
        self.lookup_ready, self.lookup_done = torch.cuda.Event(), torch.cuda.Event()
        prefill_size = 128
        if self.prefill_uses_tablemoe:
            prefill_size = getattr(self.config, "tablemoe_prefill_chunk_size", 0) or self.config.max_extend_tokens
        if self.expert_mode == "cpu":
            prefill_size = min(prefill_size, self.executor.max_tokens)
        sizes = {1, 2, 4, 128, prefill_size}
        from freetoken.engine.graph import _determine_cuda_graph_bs

        sizes.update(_determine_cuda_graph_bs(self.config.cuda_graph_bs,
                                              getattr(self.config, "cuda_graph_max_bs", self.dummy_slot), 0))
        for bs in sorted(sizes):
            self._workspace(bs)
        for bs in range(1, self.dummy_slot + 1):
            self._workspace(bs)

    @property
    def graph_enabled(self):
        return self.expert_mode != "cpu" and (self.executor is None or self.executor._flag_sync)

    def resident_gpu_bytes(self, target_cache_size=None):
        objects = [self.pca, self.gpu_keys, self.gpu_counts, self.value_offsets,
                   self.fixed_online, self.workspaces, self._route_counts, self._gpu_counts,
                   self.batch_slots, self.graph_slots]
        if self.audit is not None:
            objects.extend((self.audit.counts, self.audit.shared))
        excluded = ()
        if self.expert_cache is not None:
            objects.append(vars(self.expert_cache))
            excluded = tuple(self.expert_cache.bank_caches.values())
        total = gpu_storage_bytes(objects, excluded)
        if target_cache_size is not None and self.expert_cache is not None:
            # id_of_slot, usage, evict_slots and src_indices each scale with cache_size.
            total += (target_cache_size - self.expert_cache.cache_size) * 20
        return total

    def _workspace(self, bs):
        if bs in self.workspaces:
            return self.workspaces[bs]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TableMoE workspace must be prepared before graph capture")
        if self.executor is not None and bs <= self.executor.max_tokens:
            self.executor._io_for(bs)
            for layer in range(self.num_layers):
                self.executor._task_for(layer, bs)
        parent_size = min((size for size in self.workspaces if size > bs), default=None)
        if parent_size is not None:
            parent = self.workspaces[parent_size]
            ws = {}
            for name, value in parent.items():
                if isinstance(value, dict):
                    ws[name] = {layer: tensor[:bs] for layer, tensor in value.items()}
                elif name == "gpu_workspace":
                    gk = self.top_k if self.external_mask else max(1, self.config.tablemoe_decode_keep_k)
                    ws[name] = (value[0][:bs], value[1][:bs * gk], value[2][:bs])
                else:
                    ws[name] = value[:bs]
            self.workspaces[bs] = ws
            return ws
        def empty(*shape, dtype=torch.float16):
            return torch.empty(shape, device=self.device, dtype=dtype)
        k, h = self.top_k, self.hidden
        gk = self.top_k if self.external_mask else max(1, self.config.tablemoe_decode_keep_k)
        inter = self.config.model_config.moe_intermediate_size
        ws = dict(
            slots=empty(bs, gk, dtype=torch.int32), full_slots=empty(bs, k, dtype=torch.int32),
            keep=empty(bs, k, dtype=torch.bool), cpu_ids=empty(bs, k, dtype=torch.int32),
            cache_hit=empty(bs, k, dtype=torch.bool), on=empty(bs, k, dtype=torch.bool),
            off=empty(bs, k, dtype=torch.bool), descriptor=empty(bs, k, dtype=torch.int64),
            values=empty(bs, k, h), result=empty(bs, h),
            request_slots=empty(bs, dtype=torch.int32), update_rows=empty(bs, dtype=torch.int32),
            update_counts=empty(bs, dtype=torch.int32), modality=torch.ones(bs, device=self.device, dtype=torch.int32),
            gathered=empty(bs, max(1, self.capacity), h), scores=empty(bs, 1, max(1, self.capacity)),
            query={layer: empty(bs, self.pca[layer, "text"][0].shape[1]) for layer in range(self.num_layers)
                   if not self.bypass_tables},
            vision_query={layer: empty(bs, self.pca[layer, "text"][0].shape[1]) for layer in range(self.num_layers)
                          if not self.bypass_tables},
            gpu_weights=torch.ones(bs, gk, device=self.device, dtype=torch.float32),
            gpu_workspace=(empty(bs, gk, 2 * inter), empty(bs * gk, inter), empty(bs, gk, h)),
        )
        self.workspaces[bs] = ws
        return ws

    def prepare_graph_workspace(self, bs):
        self._workspace(bs)
        if self.executor is not None and any((layer, bs) not in self.executor._flag_slots
                                             for layer in range(self.num_layers)):
            raise RuntimeError("optimized TableMoE hybrid Graph requires a CPU flag handshake slot "
                               "for every captured batch size; reduce --cuda-graph-bs or use eager")

    def prepare_batch(self, batch, *, capture=False):
        if capture:
            return
        reqs = batch.padded_reqs
        slots = []
        for req in reqs:
            slot = self.dummy_slot if req.uid == -1 else req.table_idx
            if not 0 <= slot <= self.dummy_slot or (req.uid != -1 and slot == self.dummy_slot):
                raise ValueError("invalid TableMoE request slot")
            if req.uid != -1 and self._owners.get(slot) != req.uid:
                if slot in self._owners:
                    raise RuntimeError("TableMoE request slot reused before release")
                self._owners[slot], self._uid_slots[req.uid] = req.uid, slot
                if self.audit is not None:
                    self.audit.acquire(slot, req.uid)
                for cache in self.fixed_online.values():
                    cache["size"][slot:slot + 1].zero_()
                    cache["pointer"][slot:slot + 1].zero_()
            slots.extend([slot] * (req.extend_len if batch.is_prefill else 1))
        if len(slots) > self.batch_slots.numel():
            raise ValueError("TableMoE batch exceeds the prepared request-slot capacity")
        batch.tablemoe_slots = self.batch_slots[:len(slots)]
        batch.tablemoe_slots.copy_(torch.tensor(slots, device="cpu", dtype=torch.int32), non_blocking=True)
        if self.audit is not None:
            self.audit.slots_io(slots, batch.is_prefill)

    def reset_capture_state(self):
        for cache in self.fixed_online.values():
            cache["size"].zero_()
            cache["pointer"].zero_()
        self._route_counts.zero_()
        self._gpu_counts.zero_()
        self.counters.clear()
        if self.audit is not None:
            self.audit.reset()

    def release_request(self, uid, *, aborted=False):
        slot = self._uid_slots.get(uid)
        if self.collector is not None:
            self.collector.flush(uid)
        if slot is not None and self.audit is not None:
            self.audit.release(slot, uid, aborted=aborted)
        slot = self._uid_slots.pop(uid, None)
        if slot is not None:
            self._owners.pop(slot, None)
            for cache in self.fixed_online.values():
                cache["size"][slot:slot + 1].zero_()
                cache["pointer"][slot:slot + 1].zero_()
        if self.stats_enabled:
            logger.info("TableMoE request %s released; cumulative routes: %s", uid, self.stats())

    def close(self):
        # Mapped allocations cannot be freed until both expert and lookup streams drain.
        torch.cuda.synchronize(self.device)
        if self.collector is not None:
            self.collector.close()
        if self.audit is not None:
            self.audit.close()
        self.host_values.clear()

    def stats(self):
        if not self.stats_enabled:
            return {"collected": False}
        torch.cuda.synchronize(self.device)
        route = self._route_counts.cpu().tolist()
        gpu = self._gpu_counts.cpu().tolist()
        result = dict(zip(("cpu_exact_routes", "online_hits", "offline_hits", "dropped_routes",
                           "all_dropped_tokens", "table_value_mapped_read_bytes",
                           "cpu_route_output_mapped_read_bytes", "activation_d2h_bytes"), route))
        result.update(collected=True, gpu_exact_routes=gpu[0], gpu_cache_hit_routes=gpu[1],
                      fetched_experts=gpu[2], expert_weight_h2d_bytes=gpu[2] * self.expert_bytes,
                      table_value_h2d_bytes=0, cpu_route_output_h2d_bytes=0,
                      lookup_metadata_d2h_bytes=0, lookup_host_waits=0)
        if self.audit is not None:
            result["request_audit_totals"] = self.audit.totals
            result["last_request_audit"] = getattr(self.audit, "last_request", None)
        return result

    def forward(self, layer_id, hidden_states, weights, ids, batch):
        if batch.tablemoe_slots is None:
            self.prepare_batch(batch)
        if self.collector is not None:
            self.collector.snapshot(layer_id, hidden_states, ids, batch)
        if batch.is_prefill:
            return self._prefill(layer_id, hidden_states, weights, ids, batch)
        ws = self._workspace(hidden_states.shape[0])
        ws["request_slots"].copy_(batch.tablemoe_slots)
        ws["update_rows"].zero_()
        ws["update_counts"].fill_(1)
        return self._run(layer_id, hidden_states, weights, ids, ws, False)

    def uses_modes_routing(self, prefill):
        return self.external_mask and not (prefill and self.prefill_uses_tablemoe)

    def lookup_enabled(self, prefill):
        if self.bypass_tables:
            return False
        return self.prefill_uses_tablemoe if prefill else getattr(self.config, "moe_policy", "native") != "modes"

    def _prefill(self, layer, x, weights, ids, batch):
        exact = torch.empty_like(ids, dtype=torch.bool)
        update = torch.full((x.shape[0],), -1, device=self.device, dtype=torch.int32)
        advance = torch.zeros_like(update)
        offset = 0
        for req in batch.padded_reqs:
            n = req.extend_len
            exact[offset:offset + n] = True if self.bypass_tables and not self.skip_lookup else prefill_keep_mask(
                ids[offset:offset + n], weights[offset:offset + n],
                batch.tablemoe_attn_scores[layer][offset:offset + n],
                self.num_experts, self.config.tablemoe_prefill_keep_rate)
            count = min(n, self.capacity)
            if count and req.uid != -1:
                update[offset + n - count:offset + n] = torch.arange(count, device=self.device)
                advance[offset + n - 1] = count
            offset += n
        output = torch.empty_like(x)
        chunk_size = getattr(self.config, "tablemoe_prefill_chunk_size", 0) or max(offset, 1)
        if self.expert_mode == "cpu":
            chunk_size = min(chunk_size, self.executor.max_tokens)
        for start in range(0, offset, chunk_size):
            end = min(start + chunk_size, offset)
            ws = self._workspace(end - start)
            ws["keep"].copy_(exact[start:end])
            ws["request_slots"].copy_(batch.tablemoe_slots[start:end])
            ws["update_rows"].copy_(update[start:end])
            ws["update_counts"].copy_(advance[start:end])
            ws["modality"].fill_(1)
            if batch.mm_rows is not None:
                rows = batch.mm_rows[(batch.mm_rows >= start) & (batch.mm_rows < end)] - start
                ws["modality"][rows.long()] = 0
            output[start:end].copy_(self._run(layer, x[start:end], weights[start:end], ids[start:end], ws, True))
        return output

    def _run(self, layer, x, weights, ids, ws, prefill):
        bs, k = ids.shape
        cache = self.expert_cache
        online = self.fixed_online[layer]
        gpu_k = k if prefill or self.external_mask else max(1, self.config.tablemoe_decode_keep_k)
        slots = ws["full_slots"]
        kernels._prepare_routes[(triton.cdiv(ids.numel(), 256),)](
            ids, slots, ws["keep"], ws["cpu_ids"], ws["cache_hit"],
            cache.slot_for_id if cache is not None else ids, ws["request_slots"],
            k, self.config.tablemoe_decode_keep_k, self.num_experts, layer, self.dummy_slot,
            prefill, cache is not None, 256, ids.numel(), self.external_mask)
        if not prefill:
            ws["slots"].copy_(slots[:, :gpu_k])
            slots = ws["slots"]
        if cache is not None:
            cache.hybrid_max_fetch = self.num_experts if prefill or self.expert_mode == "offload" else self.hybrid_max_fetch
            cache.ensure_experts_hybrid(layer, slots)
        kernels._split_routes[(1,)](
            ids, ws["keep"], slots, ws["cpu_ids"], ws["cache_hit"], self._gpu_counts,
            cache.num_indices if cache is not None else ids,
            ids.numel(), k, gpu_k, cache is not None, self.stats_enabled,
            triton.next_power_of_2(ids.numel()))
        pending = None
        if self.expert_mode == "cpu" or (self.expert_mode == "hybrid" and not prefill):
            pending = self.executor.decode_submit(layer, x, weights, ws["cpu_ids"], mapped_output=True)
        lookup = self.lookup_enabled(prefill)
        if not lookup:
            ws["on"].zero_()
            ws["off"].zero_()
        else:
            if prefill or self.capacity == 0:
                ws["on"].zero_()
                ws["values"].zero_()
            else:
                kernels._gather_online_keys[(bs, self.capacity, triton.cdiv(self.hidden, 256))](
                    online["keys"], ws["gathered"], ws["request_slots"], online["size"],
                    self.hidden, self.capacity, self.dummy_slot, 256)
                threshold = getattr(self.config, "tablemoe_decode_online_cosine_threshold", None)
                online_query = x
                if threshold is not None:
                    online_query = F.normalize(x.float(), dim=-1, eps=1e-6).to(x.dtype)
                    ws["gathered"].copy_(F.normalize(ws["gathered"].float(), dim=-1, eps=1e-6))
                torch.bmm(online_query.view(bs, 1, self.hidden), ws["gathered"].transpose(1, 2), out=ws["scores"])
                kernels._online_select[(ids.numel(),)](
                    ws["scores"], online["ids"], online["values"], online["size"], ws["request_slots"],
                    ids, ws["keep"], ws["values"], ws["on"], k, self.hidden, self.capacity,
                    self.dummy_slot, triton.next_power_of_2(self.capacity),
                    triton.next_power_of_2(self.hidden), triton.next_power_of_2(k), threshold)
            proj, bias = self.pca[layer, "text"]
            query = ws["query"][layer]
            torch.addmm(bias, x, proj, out=query)
            if prefill:
                proj, bias = self.pca[layer, "vision"]
                vq = ws["vision_query"][layer]
                torch.addmm(bias, x, proj, out=vq)
                torch.where((ws["modality"] == 0)[:, None], vq, query, out=query)
                F.normalize(query, dim=-1, eps=1e-6, out=query)
            self.lookup_ready.record()
            self.lookup_stream.wait_event(self.lookup_ready)
            with torch.cuda.stream(self.lookup_stream):
                keys = self.gpu_keys[layer]
                kernels._mapped_offline[(ids.numel(),)](
                    query, keys, self.gpu_counts[layer], self.value_offsets[layer], self.value_addresses[layer],
                    ids, ws["modality"], ws["keep"], ws["on"], ws["request_slots"], ws["values"],
                    ws["off"], ws["descriptor"], k, self.hidden, query.shape[1], keys.shape[2], self.dummy_slot,
                    triton.next_power_of_2(query.shape[1]), triton.next_power_of_2(keys.shape[2]),
                    triton.next_power_of_2(self.hidden), prefill,
                    self.config.tablemoe_offline_cosine_threshold_vision,
                    self.config.tablemoe_offline_cosine_threshold_text,
                    getattr(self.config, "tablemoe_decode_offline_cosine_threshold", None))
                self.lookup_done.record()
        gpu_values = ws["values"]
        if cache is not None:
            cache.copy_missing()
            if prefill:
                from freetoken.moe.fused import fused_experts_impl
                gpu_values = fused_experts_impl(x.contiguous(), cache.bank_caches["gate_up"], cache.bank_caches["down"],
                                               torch.ones_like(slots, dtype=torch.float32), slots, return_routes=True)
            else:
                from freetoken.moe.fused import fused_experts_decode_impl
                gpu_values = fused_experts_decode_impl(
                    x, cache.bank_caches["gate_up"], cache.bank_caches["down"], ws["gpu_weights"], slots,
                    return_routes=True, workspace=ws["gpu_workspace"])
        cpu_addr = 0
        if pending is not None:
            cpu_values = self.executor.decode_sync(pending)
            cpu_addr = device_ptr(cpu_values)
        if lookup:
            torch.cuda.current_stream().wait_event(self.lookup_done)
        kernels._merge_update[(bs, triton.cdiv(self.hidden, 256))](
            x, ids, weights, ws["keep"], ws["cpu_ids"], slots, gpu_values, cpu_addr,
            ws["values"], ws["on"], ws["off"], ws["request_slots"], ws["update_rows"],
            online["keys"], online["ids"], online["values"], online["size"], online["pointer"],
            ws["result"], self._route_counts, k, gpu_k, self.hidden,
            0 if self.bypass_tables else self.capacity, self.dummy_slot, cache is not None,
            pending is not None, self.stats_enabled, 256, triton.next_power_of_2(k),
            getattr(self.config, "tablemoe_renormalize_weights", True) and not self.uses_modes_routing(prefill))
        if self.capacity and not self.bypass_tables:
            kernels._advance_online[(bs,)](online["size"], online["pointer"], ws["request_slots"],
                                           ws["update_counts"], self.capacity, self.dummy_slot)
        if self.audit is not None:
            self.audit.record(layer, ids, ws, slots, gpu_k, prefill, pending is not None)
        return ws["result"]
