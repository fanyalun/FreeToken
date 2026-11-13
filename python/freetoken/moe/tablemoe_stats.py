"""Request-owned device counters with shared fetch and padding IO accounting."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import triton

from freetoken.kernel.triton.tablemoe import _audit_routes, _audit_shared


COUNTERS = ("tokens", "routes", "cpu_exact_routes", "gpu_exact_routes", "gpu_cache_queries",
            "gpu_cache_hit_routes", "gpu_cache_miss_routes", "online_queries", "online_hits",
            "online_misses", "offline_queries", "offline_hits", "offline_misses", "missing_tables",
            "threshold_rejections", "dropped_routes", "activation_d2h_bytes", "routing_metadata_d2h_bytes",
            "cpu_route_output_mapped_read_bytes", "table_value_mapped_read_bytes", "all_dropped_tokens")
SHARED = ("fetched_experts", "expert_weight_h2d_bytes", "padding_activation_d2h_bytes",
          "padding_routing_metadata_d2h_bytes", "cpu_flag_mapped_write_d2h_bytes",
          "cpu_flag_mapped_read_h2d_bytes", "padding_request_slot_h2d_bytes")
PREFILL_CACHE_FILL = ("prefill_cache_fill_d2d_experts", "prefill_cache_fill_h2d_experts",
                      "prefill_cache_new_admissions", "prefill_cache_restored_experts")


def reconcile(c, lookup_enabled=True):
    exact = c["cpu_exact_routes"] + c["gpu_exact_routes"]
    checks = [c["routes"] == exact + c["online_hits"] + c["offline_hits"] + c["dropped_routes"],
              c["offline_queries"] == (c.get("lookup_eligible_routes", c["routes"] - exact)
                                       - c["online_hits"] if lookup_enabled else 0),
              c["offline_queries"] == c["offline_hits"] + c["offline_misses"],
              c["online_queries"] == c["online_hits"] + c["online_misses"],
              c["gpu_cache_queries"] == c["gpu_cache_hit_routes"] + c["gpu_cache_miss_routes"],
              c["offline_misses"] == c["missing_tables"] + c["threshold_rejections"]]
    return all(checks) and all(v >= 0 for v in c.values())


def io_rows(c):
    components = {
        "activation_d2h_bytes": ("activation", "d2h", "explicit_copy"),
        "routing_metadata_d2h_bytes": ("routing_metadata", "d2h", "explicit_copy"),
        "cpu_route_output_mapped_read_bytes": ("cpu_route_output", "h2d", "mapped_read"),
        "table_value_mapped_read_bytes": ("offline_table_value", "h2d", "mapped_read"),
        "expert_weight_h2d_bytes": ("expert_weight", "h2d", "mapped_read"),
        "prefill_cache_fill_h2d_bytes": ("expert_weight_cache_fill", "h2d", "mapped_read"),
        "prefill_cache_fill_d2d_bytes": ("expert_weight_cache_fill", "d2d", "device_copy"),
        "padding_activation_d2h_bytes": ("activation", "d2h", "explicit_copy"),
        "padding_routing_metadata_d2h_bytes": ("routing_metadata", "d2h", "explicit_copy"),
        "cpu_flag_mapped_write_d2h_bytes": ("cpu_flag", "d2h", "mapped_write"),
        "cpu_flag_mapped_read_h2d_bytes": ("cpu_flag", "h2d", "mapped_read"),
        "request_slot_h2d_bytes": ("request_slot", "h2d", "explicit_copy"),
        "padding_request_slot_h2d_bytes": ("request_slot", "h2d", "explicit_copy"),
    }
    return [dict(component=component, direction=direction, mechanism=mechanism,
                 bytes=c.get(name, 0), gb=c.get(name, 0) / 1e9, padding=name.startswith("padding_"))
            for name, (component, direction, mechanism) in components.items() if c.get(name, 0)]


def prefill_cache_fill_counters(values, expert_bytes):
    counters = dict(zip(PREFILL_CACHE_FILL, values))
    counters["prefill_cache_fill_d2d_bytes"] = values[0] * expert_bytes
    counters["prefill_cache_fill_h2d_bytes"] = values[1] * expert_bytes
    return counters


class RequestAudit:
    def __init__(self, runtime):
        self.runtime = runtime
        path = getattr(runtime.config, "tablemoe_report_path", None)
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path is not None and self.path.exists():
            raise ValueError(f"audit path must be fresh: {self.path}")
        self.counts = torch.zeros(runtime.dummy_slot + 1, runtime.num_layers, 2, 2, len(COUNTERS),
                                  device=runtime.device, dtype=torch.int64)
        self.shared = torch.zeros(runtime.num_layers, 2, len(SHARED), device=runtime.device, dtype=torch.int64)
        self.metadata, self.totals, self.active = {}, {}, set()
        self.peak = 0
        self.last_shared = torch.zeros_like(self.shared, device="cpu")
        fill = getattr(runtime, "prefill_cache_fill_stats", None)
        self.last_prefill_cache_fill = (torch.zeros_like(fill, device="cpu") if fill is not None else
                                        torch.zeros(runtime.num_layers, len(PREFILL_CACHE_FILL), dtype=torch.int64))
        startup = sum(t.numel() * t.element_size() for mapping in
                      (runtime.gpu_keys, runtime.gpu_counts, runtime.value_offsets) for t in mapping.values())
        startup += sum(proj.numel() * proj.element_size() + runtime.hidden * 2
                       for proj, _ in runtime.pca.values())
        cache = runtime.expert_cache
        if cache is not None and cache._copy_fused_ok:
            descriptors = (cache._copy_dst_ptrs, cache._copy_feat_bytes, *cache._copy_src_ptrs,
                           *getattr(cache, "_copy_dst_ptrs_pool", ()))
            stage = getattr(runtime, "_prefill_stage_plan", None)
            if stage is not None:
                descriptors += getattr(stage, "startup_h2d_buffers", (stage.destination_ptrs,))
            startup += sum(t.numel() * t.element_size() for t in
                           {t.data_ptr(): t for t in descriptors if t is not None}.values())
        staging = runtime._uses_prefill_staging()
        fill_enabled = runtime._prefill_cache_fill_enabled()
        self.write(dict(type="startup", scope="moe", explicit_copy_h2d_bytes=startup,
                        gpu_expert_slots=cache.cache_size if cache is not None else 0,
                        gpu_cache_scope="per-layer" if staging else getattr(runtime.config, "moe_cache_scope", "unified"),
                        gpu_cache_pool_sizes=getattr(cache, "pool_sizes", []),
                        gpu_cache_pool_of_layer=getattr(cache, "pool_of_layer", []),
                        prefill_gpu_execution=("cpu" if cache is None else
                                               "sparse_staging" if staging else "decode_cache"),
                        prefill_staging_admits_decode_cache=fill_enabled if staging else None,
                        prefill_cache_fill_policy=("exact_attention_priority_preserve_activated_residents_reuse_inactive_slots" if fill_enabled else None),
                        prefill_cache_fill_preserve=("old_resident_intersect_all_real_router_activated" if fill_enabled else None),
                        prefill_cache_fill_reusable_slots="old_inactive_and_empty" if fill_enabled else None,
                        prefill_cache_fill_new_admissions=("new_exact_including_inactive_slot_replacements" if fill_enabled else None),
                        prefill_cache_fill_no_real_routes="preserve_residents" if fill_enabled else None,
                        prefill_cache_fill_priority="sum(router_weight*last_token_attention)" if fill_enabled else None,
                        total_io_semantics="logical_host_device_transfers_excluding_d2d",
                        gpu_cache_hit_semantics="persistent_cache_before_staging_or_fetch",
                        requested_graph_max_bs=getattr(runtime.config, "cuda_graph_max_bs", None),
                        asset_provenance=getattr(runtime, "asset_provenance", "full_expert_bypass"),
                        asset_fingerprints=getattr(runtime, "asset_fingerprints", None),
                        moe_policy=getattr(runtime.config, "moe_policy", "native"),
                        decode_policy="modes" if runtime.external_mask else "top_h_plus_gpu_cache_hits",
                        decode_top_h=None if runtime.external_mask else runtime.config.tablemoe_decode_keep_k,
                        modes_prefill_policy=getattr(runtime.config, "modes_prefill_policy", "modes"),
                        modes_calibration=getattr(runtime, "calibration", None),
                        physical_pcie_bytes=None, coverage="FP16 optimized MoE logical transfers; excludes attention/encoder/KV and allocator setup"))

    def write(self, record):
        if self.path is None:
            return
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    def acquire(self, slot, uid):
        self.counts[slot].zero_()
        self.metadata[slot] = [0, 0]
        self.active.add(uid)
        self.peak = max(self.peak, len(self.active))

    def slots_io(self, slots, prefill):
        phase = 0 if prefill else 1
        for slot in slots:
            if slot == self.runtime.dummy_slot:
                self.shared[0, phase, -1] += 4
            else:
                self.metadata[slot][phase] += 4

    def reset(self):
        self.counts.zero_()
        self.shared.zero_()
        self.last_shared.zero_()
        fill = getattr(self.runtime, "prefill_cache_fill_stats", None)
        if fill is not None:
            fill.zero_()
        self.last_prefill_cache_fill.zero_()
        self.totals.clear()

    def record(self, layer, ids, ws, slots, gpu_k, prefill, has_cpu):
        rt, phase = self.runtime, 0 if prefill else 1
        bypass = not rt.lookup_enabled(prefill)
        shared = self.shared[layer, phase]
        _audit_routes[(ids.shape[0],)](
            ids, ws["keep"], ws["cpu_ids"], ws["cache_hit"], ws["on"], ws["off"],
            ws["request_slots"], ws["modality"], rt.gpu_counts[layer] if not rt.bypass_tables else ids,
            self.counts, shared, rt.top_k, rt.hidden, layer, rt.num_layers, phase, rt.dummy_slot,
            rt.expert_cache is not None, has_cpu, not prefill and rt.capacity > 0 and not bypass,
            bypass, triton.next_power_of_2(rt.top_k))
        flag = has_cpu and rt.executor._flag_sync and (layer, len(ids)) in rt.executor._flag_slots
        _audit_shared[(1,)](shared, rt.expert_cache.num_indices if rt.expert_cache is not None else ids,
                            rt.expert_bytes, rt.expert_cache is not None, flag)

    def shared_delta(self, scope):
        values = self.shared.cpu()
        delta = values - self.last_shared
        rows = []
        for layer in range(self.runtime.num_layers):
            for phase in range(2):
                counters = dict(zip(SHARED, delta[layer, phase].tolist()))
                if any(counters.values()):
                    rows.append(dict(layer=layer, phase=("prefill", "decode")[phase],
                                     expert_fetch_target=("prefill_staging" if phase == 0 and
                                                          self.runtime._uses_prefill_staging() else "decode_cache"),
                                     counters=counters, io=io_rows(counters)))
        self.last_shared.copy_(values)
        fill = getattr(self.runtime, "prefill_cache_fill_stats", None)
        if fill is not None:
            fill_values = fill.cpu()
            if fill_values.shape != self.last_prefill_cache_fill.shape:
                self.last_prefill_cache_fill = torch.zeros_like(fill_values)
            fill_delta = fill_values - self.last_prefill_cache_fill
            for layer, values in enumerate(fill_delta.tolist()):
                if any(values):
                    counters = prefill_cache_fill_counters(values, self.runtime.expert_bytes)
                    rows.append(dict(layer=layer, phase="prefill", expert_fetch_target="decode_cache_fill",
                                     counters=counters, io=io_rows(counters)))
            self.last_prefill_cache_fill.copy_(fill_values)
        occupancy = self.cache_occupancy()
        if rows or occupancy:
            self.write(dict(type="batch_shared", export_scope=scope, active_response_ids=[f"chatcmpl-{uid}" for uid in sorted(self.active)],
                            rows=rows, gpu_cache_occupancy=occupancy,
                            total_io_bytes=sum(v["bytes"] for row in rows for v in row["io"] if v["direction"] != "d2d"),
                            total_device_io_bytes=sum(v["bytes"] for row in rows for v in row["io"] if v["direction"] == "d2d")))

    def cache_occupancy(self):
        cache = self.runtime.expert_cache
        if cache is None:
            return []
        tags = cache.id_of_slot.cpu().tolist()
        resident = [0] * cache.num_layers
        occupied, base = [], 0
        for size in cache.pool_sizes:
            pool_tags = tags[base : base + size]
            occupied.append(sum(owner >= 0 for owner in pool_tags))
            for owner in pool_tags:
                if owner >= 0:
                    resident[owner // cache.num_experts] += 1
            base += size
        return [dict(layer=layer, pool=pool, pool_slots=cache.pool_sizes[pool], resident_experts=resident[layer],
                     pool_occupied_slots=occupied[pool], pool_empty_slots=cache.pool_sizes[pool] - occupied[pool])
                for layer, pool in enumerate(cache.pool_of_layer)]

    def release(self, slot, uid, *, aborted=False):
        rt = self.runtime
        # Slot lifetime ends only after both streams have stopped accessing its counters.
        rt.lookup_stream.synchronize()
        torch.cuda.synchronize(rt.device)
        values = self.counts[slot].cpu()
        rows, total = [], dict.fromkeys(COUNTERS, 0)
        for layer in range(rt.num_layers):
            for phase in range(2):
                for mod in range(2):
                    c = dict(zip(COUNTERS, values[layer, phase, mod].tolist()))
                    if not c["tokens"]:
                        continue
                    for name, value in c.items():
                        total[name] += value
                    lookup = rt.lookup_enabled(phase == 0)
                    rows.append(dict(layer=layer, phase=("prefill", "decode")[phase], modality=("vision", "text")[mod],
                                     gpu_execution=("sparse_staging" if phase == 0 and rt._uses_prefill_staging()
                                                    else "decode_cache"),
                                     routing_policy="modes" if rt.uses_modes_routing(phase == 0) else "tablemoe",
                                     counters=c, lookup_enabled=lookup,
                                     reconciled=reconcile(c, lookup), io=io_rows(c)))
        slot_phase_bytes = self.metadata.pop(slot)
        slot_bytes = sum(slot_phase_bytes)
        total["request_slot_h2d_bytes"] = slot_bytes
        if getattr(rt, "external_mask", False):
            total["lookup_eligible_routes"] = sum(
                row["counters"]["routes"] - row["counters"]["cpu_exact_routes"] - row["counters"]["gpu_exact_routes"]
                for row in rows if row["lookup_enabled"])
            total["modes_skipped_routes"] = sum(
                row["counters"]["routes"] - row["counters"]["cpu_exact_routes"] - row["counters"]["gpu_exact_routes"]
                for row in rows if row["routing_policy"] == "modes")
            total["modes_recovered_routes"] = sum(
                row["counters"]["online_hits"] + row["counters"]["offline_hits"]
                for row in rows if row["routing_policy"] == "modes")
            total["table_recovered_routes"] = total["online_hits"] + total["offline_hits"]
        for name, value in total.items():
            self.totals[name] = self.totals.get(name, 0) + value
        prompt = values[0, 0, :, 0].tolist()
        self.write(dict(type="request", uid=uid, response_id=f"chatcmpl-{uid}", aborted=aborted, counters=total,
                        rows=rows, lookup_enabled=not getattr(rt, "skip_lookup", False),
                        prompt_vision_tokens=prompt[0], prompt_nonvision_tokens=prompt[1],
                        reconciled=all(row["reconciled"] for row in rows), io=io_rows(total),
                        request_slot_io_by_phase=[dict(phase=phase, io=io_rows(dict(request_slot_h2d_bytes=value)))
                                                  for phase, value in zip(("prefill", "decode"), slot_phase_bytes)],
                        total_io_bytes=sum(row["bytes"] for row in io_rows(total))))
        self.last_request = dict(uid=uid, counters=total, rows=rows)
        self.shared_delta(f"release:{uid}")
        self.active.discard(uid)

    def close(self):
        self.shared_delta("shutdown")
        shared = {name: int(self.shared[..., i].sum().cpu()) for i, name in enumerate(SHARED)}
        shared.update(prefill_cache_fill_counters(self.last_prefill_cache_fill.sum(dim=0).tolist(), self.runtime.expert_bytes))
        requests_io = sum(v["bytes"] for v in io_rows(self.totals))
        shared_io = sum(v["bytes"] for v in io_rows(shared) if v["direction"] != "d2d")
        shared_device_io = sum(v["bytes"] for v in io_rows(shared) if v["direction"] == "d2d")
        self.write(dict(type="run_summary", request_counters=self.totals, shared_counters=shared,
                        request_io_bytes=requests_io, shared_io_bytes=shared_io,
                        total_io_bytes=requests_io + shared_io, total_device_io_bytes=shared_device_io,
                        peak_active_requests=self.peak,
                        unreleased_response_ids=[f"chatcmpl-{uid}" for uid in sorted(self.active)]))
