"""Independent audit hooks, enabled only through TABLEMOE_AUDIT_DIR."""

import os

if os.environ.get("TABLEMOE_AUDIT_DIR"):
    import json
    import hashlib
    import sys
    from pathlib import Path

    import torch
    from freetoken.engine.engine import Engine
    try:
        from freetoken.moe.tablemoe import TableMoeRuntime
        from freetoken.moe.tablemoe_optimized import TableMoeOptimizedRuntime
    except ImportError:
        TableMoeOptimizedRuntime = None

    root = Path(os.environ["TABLEMOE_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit_source_sha256.txt").write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if TableMoeOptimizedRuntime is not None:
        import inspect

        source_path = Path(inspect.getfile(TableMoeOptimizedRuntime))
        (root / "runtime_source_sha256.txt").write_text(hashlib.sha256(source_path.read_bytes()).hexdigest())
    if TableMoeOptimizedRuntime is None:
        original_init = Engine.__init__

        def init(self, config, *args, **kwargs):
            object.__setattr__(config, "moe_collect_stats", True)
            return original_init(self, config, *args, **kwargs)

        Engine.__init__ = init
    decode_calls = 0
    checked = []
    decode_requests = []
    profile_start = int(os.environ.get("TABLEMOE_PROFILE_START", "64"))
    profile_steps = int(os.environ.get("TABLEMOE_PROFILE_STEPS", "32"))
    profile_end = profile_start + profile_steps - 1
    final_decode_step = int(os.environ.get("TABLEMOE_FINAL_DECODE_STEP", "2595"))
    if TableMoeOptimizedRuntime is not None:
        original_run = TableMoeOptimizedRuntime._run

        def run(self, layer, x, weights, ids, ws, prefill):
            global decode_calls
            if layer == 0 and not prefill and not torch.cuda.is_current_stream_capturing():
                decode_calls += 1
            check = (not prefill and 45 < decode_calls <= 53
                     and not self.bypass_tables and not torch.cuda.is_current_stream_capturing())
            if check:
                slot = int(ws["request_slots"][0].item())
                cache = self.fixed_online[layer]
                size = int(cache["size"][slot].item())
                saved = {name: cache[name][slot].clone() for name in ("keys", "ids", "values", "size", "pointer")}
                snapshot = dict(x=x.cpu(), ids=ids.cpu(), weights=weights.cpu(),
                                online={name: value.cpu() for name, value in saved.items()},
                                layer=layer, config=dict(keep_k=self.config.tablemoe_decode_keep_k,
                                                         capacity=self.capacity))
                self.online[0][layer] = dict(keys=saved["keys"], ids=saved["ids"], values=saved["values"], size=size)
                reference_values, hit = TableMoeRuntime._online_search(self, 0, layer, x, ids)
                keep = torch.arange(ids.shape[1], device=x.device)[None, :] < self.config.tablemoe_decode_keep_k
                hit &= ~keep
                descriptor = TableMoeRuntime._search(self, layer, x, ids, None, ~keep & ~hit, False)
                clusters = self.gpu_keys[layer].shape[2]
                packed = self.value_offsets[layer].reshape(-1)[(descriptor.clamp_min(0) // clusters).long()]
                packed = torch.where(descriptor >= 0, packed + descriptor % clusters, -1)
                indices = packed.cpu()
                offline_values = self.host_values[layer][indices.clamp_min(0)].to(x.device)
                reference_values = torch.where((packed >= 0)[:, :, None], offline_values, reference_values)
            result = original_run(self, layer, x, weights, ids, ws, prefill)
            if check:
                torch.cuda.synchronize()
                routes = keep | hit | (packed >= 0)
                computed = self.fixed_online[layer]["values"][slot, (int(saved["pointer"]) % self.capacity)]
                reference_values = torch.where(keep[:, :, None], computed[None], reference_values)
                surviving_weights = weights.masked_fill(~routes, 0)
                normalized = surviving_weights / surviving_weights.sum(-1, keepdim=True).clamp_min(1e-30)
                reference_output = (reference_values.float() * normalized[:, :, None]).sum(1).half()
                selected = hit | (packed >= 0)
                torch.testing.assert_close(ws["values"][selected], reference_values[selected], atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(result, reference_output, atol=2e-2, rtol=2e-2)
                record = dict(step=decode_calls, layer=layer, online_equal=torch.equal(hit, ws["on"]),
                              offline_equal=torch.equal(packed, ws["descriptor"]),
                              mapped_values_close=True, merged_output_close=True)
                checked.append(record)
                if layer == 0:
                    torch.save(snapshot, root / f"layer_input_{decode_calls}.pt")
                (root / "retrieval_parity.json").write_text(json.dumps(checked, indent=2))
                self.online[0].pop(layer, None)
                if not all((record["online_equal"], record["offline_equal"])):
                    raise ValueError(f"TableMoE retrieval choice changed: {record}")
            return result

        TableMoeOptimizedRuntime._run = run

        if os.environ.get("TABLEMOE_DECODE_COUNTERS"):
            original_release = TableMoeOptimizedRuntime.release_request

            def release(self, uid):
                baseline, start_step, ordinal = self._audit_prefill_baselines.pop(uid)
                end = self.stats()
                decode_requests.append(dict(uid=uid, request_ordinal=ordinal, decode_steps=steps - start_step,
                    counters={k: v - baseline[k] for k, v in end.items()
                              if isinstance(v, int) and not isinstance(v, bool)}))
                (root / "request_decode_counters.json").write_text(json.dumps(decode_requests, indent=2))
                return original_release(self, uid)

            TableMoeOptimizedRuntime.release_request = release

    if os.environ.get("TABLEMOE_NSYS") or os.environ.get("TABLEMOE_VERIFY") or os.environ.get("TABLEMOE_DECODE_COUNTERS") or TableMoeOptimizedRuntime is None:
        original_forward = Engine.forward_batch
        steps = 0
        prefills = 0
        observed_batches = set()

        def native_counts(self):
            cache = self.moe_offload_cache
            torch.cuda.synchronize()
            return dict(lru_stats=cache.lru_stats.cpu().tolist(),
                        missing=int(cache.stat_missing.item()), active=int(cache.stat_active.item()),
                        fetched=int(cache.stat_fetched.item()), calls=int(cache.stat_calls.item()),
                        actual_cache_slots=cache.cache_size, mode=cache.decode_target,
                        decode_real_tokens=steps, cpu_exact_routes=(steps * 48 * 8 if cache.decode_target == "cpu" else None))

        def forward(self, batch, *args, **kwargs):
            global steps, prefills
            if not batch.is_prefill:
                steps += 1
            else:
                prefills += 1
            if TableMoeOptimizedRuntime is None and prefills == 6 and batch.is_prefill:
                (root / "warmup_native_counters.json").write_text(json.dumps(native_counts(self), indent=2))
            if os.environ.get("TABLEMOE_NSYS") and not batch.is_prefill and steps == profile_start:
                if TableMoeOptimizedRuntime is not None:
                    (root / "profile_start_counters.json").write_text(json.dumps(self.tablemoe_runtime.stats(), indent=2))
                torch.cuda.cudart().cudaProfilerStart()
                (root / "profile_window.json").write_text(json.dumps(dict(start=profile_start, end=profile_end,
                    decode_steps=profile_steps, warmup_decode_steps=45), indent=2))
            if not batch.is_prefill and profile_start <= steps <= profile_end:
                with torch.cuda.nvtx.range(f"decode.step_{steps}"):
                    result = original_forward(self, batch, *args, **kwargs)
                    if steps == profile_end:
                        torch.cuda.synchronize()
            else:
                result = original_forward(self, batch, *args, **kwargs)
            if os.environ.get("TABLEMOE_VERIFY") and not batch.is_prefill:
                observed_batches.add((batch.size, batch.padded_size, self.graph_runner.can_use_cuda_graph(batch)))
                (root / "observed_batches.json").write_text(json.dumps(sorted(observed_batches), indent=2))
            if os.environ.get("TABLEMOE_DECODE_COUNTERS") and TableMoeOptimizedRuntime is not None and batch.is_prefill:
                runtime = self.tablemoe_runtime
                baseline = runtime.stats()
                if not hasattr(runtime, "_audit_prefill_baselines"):
                    runtime._audit_prefill_baselines = {}
                for req in batch.padded_reqs:
                    if req.uid != -1:
                        runtime._audit_prefill_baselines[req.uid] = (baseline, steps, prefills)
            if os.environ.get("TABLEMOE_NSYS") and not batch.is_prefill and steps == profile_end:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                if TableMoeOptimizedRuntime is not None:
                    start_counts = json.loads((root / "profile_start_counters.json").read_text())
                    end_counts = self.tablemoe_runtime.stats()
                    (root / "profile_counters.json").write_text(json.dumps({
                        k: v - start_counts[k] for k, v in end_counts.items()
                        if isinstance(v, int) and not isinstance(v, bool)}, indent=2))
            if TableMoeOptimizedRuntime is None and steps == final_decode_step and not batch.is_prefill:
                (root / "native_counters.json").write_text(json.dumps(native_counts(self), indent=2))
            return result

        Engine.forward_batch = forward
