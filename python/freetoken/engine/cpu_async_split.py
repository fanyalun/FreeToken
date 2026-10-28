from __future__ import annotations

import torch

from .qwen_cpu_target import CpuVerifyResult, PrefixCheckpoint
from .speculative import SpeculativeFrontier


class PrefixStateTransfer:
    def __init__(self, engine, worker):
        from freetoken.kernel.pinned import device_ptr, host_register, host_unregister
        from freetoken.kernel.fast_index_copy import (_jit_fast_index_copy_multi_module,
                                                      _skip_fast_index_copy_enabled)

        if _skip_fast_index_copy_enabled():
            raise ValueError("CPU prefix state installation cannot skip fast_index_copy")

        self.engine = engine
        self.stream = torch.cuda.Stream(device=engine.device)
        self.registered = []
        self._unregister = host_unregister
        self.closed = False
        self.pending = None
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        self.buffers = {}
        rows = engine.config.cpu_async_loops * (engine.config.speculative_mtp_depth + 1) + 1
        self.rows = rows
        self.sources = {}
        self.conv_sources = {}
        self.metadata = []
        try:
            for slot in worker.slots:
                for name in ("hidden", "keys", "values", "prefix_replay", "prefix_conv"):
                    tensor = slot[name]
                    if tensor.numel():
                        host_register(tensor.data_ptr(), tensor.numel() * tensor.element_size())
                        self.registered.append(tensor)
            with torch.cuda.stream(engine.stream):
                bank_fields, destinations, features = [], [], []
                for name in ("keys", "values", "replay"):
                    field = "prefix_replay" if name == "replay" else name
                    source = worker.slots[0][field]
                    shape = (source.shape[0], rows, *source.shape[2:])
                    target = self.buffers[name] = torch.empty(shape, dtype=source.dtype, device=engine.device)
                    for layer in range(source.shape[0]):
                        feature = source[layer, 0].numel() * source.element_size()
                        if feature % 16 or target[layer].data_ptr() % 16:
                            raise ValueError("CPU prefix mapped-copy banks require 16-byte aligned rows")
                        bank_fields.append((field, layer))
                        destinations.append(target[layer].data_ptr())
                        features.append(feature)
                self.buffers["conv"] = torch.empty_like(worker.slots[0]["prefix_conv"], device=engine.device)
                self.destinations = torch.tensor(destinations, dtype=torch.int64, device=engine.device)
                self.features = torch.tensor(features, dtype=torch.int64, device=engine.device)
                self.indices = torch.arange(rows, dtype=torch.int32, device=engine.device)
                self.count = torch.zeros(1, dtype=torch.int64, device=engine.device)
                self.metadata.extend((self.destinations, self.features, self.indices, self.count))
                for slot in worker.slots:
                    pointers = [device_ptr(slot[field][layer]) for field, layer in bank_fields]
                    if any(pointer % 16 for pointer in pointers):
                        raise ValueError("CPU prefix mapped-copy sources require 16-byte alignment")
                    table = torch.tensor(pointers, dtype=torch.int64, device=engine.device)
                    bases = tuple(slot[field].data_ptr() for field in
                                  ("keys", "values", "prefix_replay", "prefix_conv"))
                    self.sources[slot["hidden"].data_ptr()] = table, bases
                    self.metadata.append(table)
                    conv_table = torch.tensor([device_ptr(layer) for layer in slot["prefix_conv"]],
                                              dtype=torch.int64, device=engine.device)
                    self.conv_sources[slot["hidden"].data_ptr()] = conv_table
                    self.metadata.append(conv_table)
                self.copy_kernel = _jit_fast_index_copy_multi_module(num_threads=1024, blocks_per_bank=1)
                if destinations:
                    self.copy_kernel.launch(self.destinations, next(iter(self.sources.values()))[0],
                        self.features, self.indices, None, self.count, 0)
        except BaseException:
            for tensor in reversed(self.registered):
                host_unregister(tensor.data_ptr())
            self.registered.clear()
            raise

    @property
    def pinned_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.registered)

    @property
    def device_bytes(self):
        return sum(t.numel() * t.element_size() for t in (*self.buffers.values(), *self.metadata))

    def stage(self, prefix):
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if _skip_fast_index_copy_enabled():
            raise ValueError("CPU prefix state installation cannot skip fast_index_copy")
        if self.closed or self.pending is not None:
            raise RuntimeError("CPU prefix transfer is closed or still owns an uninstalled window")
        if not 1 <= prefix.width <= self.rows:
            raise ValueError("CPU prefix state transfer exceeds its row capacity")
        source = self.sources.get(prefix.hidden.data_ptr())
        bases = tuple(getattr(prefix, name).data_ptr() for name in ("keys", "values", "replay", "conv"))
        if source is None or source[1] != bases:
            raise ValueError("CPU prefix state transfer must read its registered shared result slot")
        self.pending = prefix
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(self.engine.stream)
            self.events[0].record(self.stream)
            self.count.fill_(prefix.width)
            if self.destinations.numel():
                self.copy_kernel.launch(self.destinations, source[0], self.features,
                    self.indices, None, self.count, 0)
            from freetoken.kernel.fla.gdn_speculative import copy_cpu_conv_windows

            copy_cpu_conv_windows(self.buffers["conv"], self.conv_sources[prefix.hidden.data_ptr()],
                                  prefix.conv.shape[-1])
            self.events[1].record(self.stream)
        return self.events[1]

    def install(self, window, prefix, verified):
        if self.pending is not prefix:
            raise RuntimeError("CPU prefix transfer acceptance belongs to a different window")
        if (prefix.request_epoch != window.epoch or verified.start != window.canonical_length
                or verified.start != prefix.start or not 1 <= verified.input_count <= prefix.width):
            raise ValueError("split acceptance does not extend the canonical prefix")
        e, r = self.engine, window.runner
        count = verified.input_count
        end = prefix.start + count
        if end > window.output_end:
            raise ValueError("split acceptance exceeds request capacity")
        with torch.cuda.stream(e.stream):
            e.stream.wait_event(self.events[1])
            r._reserve(end)
            layers = e.config.cpu_async_prefix_layers
            attention = [i for i in range(layers) if not e.config.model_config.is_linear_layer(i)]
            for index, layer in enumerate(attention):
                e.kv_cache.k_cache(layer)[prefix.start:end, 0].copy_(self.buffers["keys"][index, :count])
                e.kv_cache.v_cache(layer)[prefix.start:end, 0].copy_(self.buffers["values"][index, :count])
            pool = e.linear_state_pool
            linear = self.buffers["replay"].shape[0]
            if pool is not None and linear:
                from freetoken.kernel.fla.gdn_speculative import commit_cpu_gdn

                v2 = (getattr(e.config, "cpu_async_kernel_profile", "v1") == "ggml_v2"
                      and "gdn" in (getattr(e.config, "cpu_async_kernel_components", None) or ()))
                commit_cpu_gdn(pool.recurrent_states[:linear], pool.conv_states[:linear],
                               self.buffers["replay"], self.buffers["conv"], count, 1, v2=v2)
            window.state_adapter.fork(window.private_pool)
            for offset in range(0, count, window.replay_rows):
                width = min(window.replay_rows, count - offset)
                window._refresh("cpu_async_confirmed_mtp", verified.tokens[offset:offset + width],
                                verified.hidden[offset:offset + width], prefix.start + offset,
                                offset + width == count)
            window.canonical_length, window.canonical_anchor = end, verified.tokens[-1]
            r.frontier = SpeculativeFrontier(1, end, end, r._reserved_len)
            checkpoint = PrefixCheckpoint(end, verified.tokens[-1], self.buffers["keys"][:, :count],
                self.buffers["values"][:, :count], pool.recurrent_states[:linear, 1],
                pool.conv_states[:linear, 1])
            result = CpuVerifyResult(prefix.request_epoch, prefix.block_id, prefix.start,
                verified.accepted, count, verified.tokens, verified.terminal, checkpoint,
                verified.hidden[:count], verified.predictions)
            reader = window.reader_events[window.reader_generation % 2]
            window.reader_generation += 1
            reader.record(e.stream)
        self.pending = None
        return result, reader

    def close(self):
        if self.closed:
            return
        self.stream.synchronize()
        self.engine.stream.synchronize()
        for tensor in reversed(self.registered):
            self._unregister(tensor.data_ptr())
        self.registered.clear()
        self.closed = True
