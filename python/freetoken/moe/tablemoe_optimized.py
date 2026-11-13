"""Mapped CPU values and fixed request state for TableMoE eager/graph execution."""

from __future__ import annotations

from contextlib import nullcontext
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
        if self._uses_prefill_staging():
            self._prefill_stage_for()
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
            prefill_size = self.config.max_extend_tokens
        sizes = {1, 2, 4, 128, prefill_size}
        from freetoken.engine.graph import _determine_cuda_graph_bs

        sizes.update(_determine_cuda_graph_bs(self.config.cuda_graph_bs,
                                              getattr(self.config, "cuda_graph_max_bs", self.dummy_slot), 0))
        for bs in sorted(sizes):
            self._workspace(bs)
        for bs in range(1, self.dummy_slot + 1):
            self._workspace(bs)
        self.prefill_copy_stream = None
        self.prefill_copy_plan = None
        self.prefill_plan_ready = None
        self.prefill_group_ready = []
        if self._prefill_overlap_enabled():
            self._prefill_copy_plan_for(0, self._workspace(self.config.max_extend_tokens)["full_slots"])

    def _prefill_overlap_enabled(self):
        return (self.prefill_uses_tablemoe and self.expert_mode in ("offload", "hybrid") and
                getattr(self.config, "tablemoe_prefill_experts_per_group", 0) > 0 and
                self.expert_cache._copy_fused_ok and not self.expert_cache._unpinned_layers)

    def _prefill_copy_plan_for(self, layer, slots):
        from freetoken.moe.tablemoe_prefill_io import GroupPrefillCopyPlan

        group_size = self.config.tablemoe_prefill_experts_per_group
        stage = self._prefill_stage_for() if self._uses_prefill_staging() else None
        slot_count = self.num_experts if stage is not None else self.expert_cache.cache_size
        plan = self.prefill_copy_plan
        if (plan is None or plan.cache is not self.expert_cache or plan.group_size != group_size or
                plan.stage is not stage or plan.slot_groups.numel() != slot_count or
                plan.route_group_storage.numel() < slots.numel()):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("TableMoE prefill copy plan must be prepared before graph capture")
            if self.prefill_copy_stream is None:
                self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
                self.prefill_plan_ready = torch.cuda.Event()
            capacity = max(len(slots), self.config.max_extend_tokens)
            self.prefill_copy_plan = plan = GroupPrefillCopyPlan(
                self.expert_cache, layer, self._workspace(capacity)["full_slots"], group_size, stage=stage)
            self.prefill_group_ready = [torch.cuda.Event() for _ in range(plan.num_groups)]
        return plan.prepare(layer, slots)

    def _start_prefill_experts(self, layer, x, ids, slots, ws):
        from freetoken.moe.tablemoe_prefill import GroupedPrefillExperts

        plan = self._prefill_copy_plan_for(layer, slots)
        gate_up, down = (plan.stage.views if plan.stage is not None else
                         self.expert_cache.pool_bank_views(layer))
        runner = GroupedPrefillExperts(
            x, gate_up, down,
            slots, plan.route_groups, workspace=ws["gpu_workspace"],
            logical_ids=ids, num_experts=self.num_experts)
        self.prefill_plan_ready.record()
        self.prefill_copy_stream.wait_event(self.prefill_plan_ready)
        with torch.cuda.stream(self.prefill_copy_stream):
            plan.copy_group(0)
            self.prefill_group_ready[0].record()
        return plan, runner

    def _run_prefill_experts(self, pipeline):
        plan, runner = pipeline
        runner.run(-1)
        for group in range(plan.num_groups):
            if group + 1 < plan.num_groups:
                with torch.cuda.stream(self.prefill_copy_stream):
                    plan.copy_group(group + 1)
                    self.prefill_group_ready[group + 1].record()
            torch.cuda.current_stream().wait_event(self.prefill_group_ready[group])
            runner.run(group)
        return runner.outputs

    @property
    def graph_enabled(self):
        return self.expert_mode != "cpu" and (self.executor is None or self.executor._flag_sync)

    def resident_gpu_bytes(self, target_cache_size=None):
        objects = [self.pca, self.gpu_keys, self.gpu_counts, self.value_offsets,
                   self.fixed_online, self.workspaces, self._route_counts, self._gpu_counts,
                   self.batch_slots, self.graph_slots]
        if self.prefill_copy_plan is not None:
            objects.append(self.prefill_copy_plan.buffers)
        if getattr(self, "_prefill_stage_plan", None) is not None:
            objects.append(self._prefill_stage_plan.buffers)
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
            if self.prefill_copy_plan is not None and self.prefill_copy_plan.stage is None:
                total += (target_cache_size - self.prefill_copy_plan.slot_groups.numel()) * 4
        return total

    def on_expert_cache_rebuild(self):
        super().on_expert_cache_rebuild()
        self.prefill_copy_plan = None
        if self._prefill_overlap_enabled():
            self._prefill_copy_plan_for(0, self._workspace(self.config.max_extend_tokens)["full_slots"])

    def _workspace(self, bs):
        if self.expert_mode == "cpu" and bs > self.executor.max_tokens:
            raise ValueError("TableMoE batch exceeds CPU executor capacity; increase --max-extend-tokens")
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
                    gk = self.top_k
                    ws[name] = (value[0][:bs], value[1][:bs * gk], value[2][:bs])
                else:
                    ws[name] = value[:bs]
            self.workspaces[bs] = ws
            return ws
        def empty(*shape, dtype=torch.float16):
            return torch.empty(shape, device=self.device, dtype=dtype)
        k, h = self.top_k, self.hidden
        # Cache hits can occupy any route, including positions after the leading h.
        gk = self.top_k
        inter = self.config.model_config.moe_intermediate_size
        ws = dict(
            full_slots=empty(bs, k, dtype=torch.int32),
            keep=empty(bs, k, dtype=torch.bool), cpu_ids=empty(bs, k, dtype=torch.int32),
            cache_hit=empty(bs, k, dtype=torch.bool), on=empty(bs, k, dtype=torch.bool),
            off=empty(bs, k, dtype=torch.bool), descriptor=empty(bs, k, dtype=torch.int64),
            values=empty(bs, k, h), result=empty(bs, h),
            request_slots=empty(bs, dtype=torch.int32), update_rows=empty(bs, dtype=torch.int32),
            update_counts=empty(bs, dtype=torch.int32), modality=torch.ones(bs, device=self.device, dtype=torch.int32),
            scores=empty(bs, 1, max(1, self.capacity)),
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
        stage = getattr(self, "_prefill_stage_plan", None)
        if stage is not None:
            stage.cache_fill.reset()
            stage.cache_fill.stats.zero_()
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
        # Mapped allocations must outlive expert compute, grouped copies and lookup.
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
        result.update(self._prefill_cache_fill_counters())
        result["expert_weight_total_h2d_bytes"] = (result["expert_weight_h2d_bytes"] +
                                                   result["prefill_cache_fill_h2d_bytes"])
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
                advance[offset + n - 1:offset + n].fill_(count)
            offset += n
        if not offset:
            return torch.empty_like(x)
        ws = self._workspace(offset)
        ws["keep"].copy_(exact)
        ws["request_slots"].copy_(batch.tablemoe_slots)
        ws["update_rows"].copy_(update)
        ws["update_counts"].copy_(advance)
        ws["modality"].fill_(1)
        if batch.mm_rows is not None:
            ws["modality"].index_fill_(0, batch.mm_rows.long(), 0)
        return self._run(layer, x, weights, ids, ws, True,
                         attention_scores=batch.tablemoe_attn_scores.get(layer)).clone()

    def _run(self, layer, x, weights, ids, ws, prefill, attention_scores=None):
        try:
            return self._run_impl(layer, x, weights, ids, ws, prefill, attention_scores)
        except BaseException:
            # An interrupted layer must drain side streams before its buffers are reused.
            if prefill and self.prefill_copy_stream is not None:
                self.prefill_copy_stream.synchronize()
                self.lookup_stream.synchronize()
            if prefill and getattr(self, "_prefill_stage_plan", None) is not None:
                self._prefill_stage_plan.cache_fill.reset()
            raise

    def _run_impl(self, layer, x, weights, ids, ws, prefill, attention_scores=None):
        bs, k = ids.shape
        cache = self.expert_cache
        online = self.fixed_online[layer]
        gpu_k = k
        lookup = self.lookup_enabled(prefill)
        parallel_lookup = lookup and not prefill and self.expert_mode == "offload"
        slots = ws["full_slots"]
        kernels._prepare_routes[(triton.cdiv(ids.numel(), 256),)](
            ids, slots, ws["keep"], ws["cpu_ids"], ws["cache_hit"],
            cache.slot_for_id if cache is not None else ids, ws["request_slots"],
            k, self.config.tablemoe_decode_keep_k, self.num_experts, layer, self.dummy_slot,
            prefill, cache is not None, 256, ids.numel(), self.external_mask)
        if parallel_lookup:
            self.lookup_ready.record()
        if cache is not None:
            if prefill and self._uses_prefill_staging():
                stage = self._prefill_stage_for()
                if self._prefill_cache_fill_enabled():
                    stage.cache_fill.record(layer, ids, ws["keep"], weights, attention_scores,
                                            request_slots=ws["request_slots"], dummy_slot=self.dummy_slot)
                stage.prepare(layer, slots)
            else:
                cache.hybrid_max_fetch = self.num_experts if prefill or self.expert_mode == "offload" else self.hybrid_max_fetch
                cache.ensure_experts_hybrid(layer, slots)
        kernels._split_routes[(1,)](
            ids, ws["keep"], slots, ws["cpu_ids"], ws["cache_hit"], self._gpu_counts,
            cache.num_indices if cache is not None else ids,
            ids.numel(), k, gpu_k, cache is not None, self.stats_enabled,
            triton.next_power_of_2(ids.numel()))
        pipeline = (self._start_prefill_experts(layer, x, ids, slots, ws)
                    if prefill and self._prefill_overlap_enabled() else None)
        pending = None
        if self.expert_mode == "cpu" or (self.expert_mode == "hybrid" and not prefill):
            pending = self.executor.decode_submit(layer, x, weights, ws["cpu_ids"], mapped_output=True)
        if not lookup:
            ws["on"].zero_()
            ws["off"].zero_()
        else:
            if parallel_lookup:
                self.lookup_stream.wait_event(self.lookup_ready)
            # Keep the CPU handshake's existing Graph branch layout in hybrid mode.
            with torch.cuda.stream(self.lookup_stream) if parallel_lookup else nullcontext():
                if prefill or self.capacity == 0:
                    ws["on"].zero_()
                    ws["values"].zero_()
                else:
                    threshold = getattr(self.config, "tablemoe_decode_online_cosine_threshold", None)
                    kernels._online_scores[(bs, self.capacity)](
                        x, online["keys"], ws["scores"], ws["keep"], ws["request_slots"], online["size"],
                        self.hidden, k, self.capacity, self.dummy_slot,
                        triton.next_power_of_2(self.hidden), triton.next_power_of_2(k), threshold is not None)
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
                if not parallel_lookup:
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
            if pipeline is not None:
                gpu_values = self._run_prefill_experts(pipeline)
            elif prefill:
                from freetoken.moe.fused import fused_experts_impl
                if self._uses_prefill_staging():
                    stage = self._prefill_stage_for()
                    stage.copy()
                    gate_up, down = stage.views
                else:
                    cache.copy_missing()
                    gate_up, down = cache.pool_bank_views(layer)
                gpu_values = fused_experts_impl(x.contiguous(), gate_up, down,
                                               torch.ones_like(slots, dtype=torch.float32), slots, return_routes=True)
            else:
                from freetoken.moe.fused import fused_experts_decode_impl
                cache.copy_missing()
                gate_up, down = cache.pool_bank_views(layer)
                gpu_values = fused_experts_decode_impl(
                    x, gate_up, down, ws["gpu_weights"], slots,
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
        if prefill and cache is not None and self._uses_prefill_staging():
            self._finish_prefill_cache_fill(layer)
        return ws["result"]
