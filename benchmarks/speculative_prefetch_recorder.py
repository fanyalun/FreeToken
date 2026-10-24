from __future__ import annotations

from collections import defaultdict

import torch


class PrefetchWindowRecorder:
    """Diagnostic graph events, read only after the existing control drain."""

    def __init__(self, context_boundaries=(512, 2048, 5120)):
        self.context_boundaries = tuple(context_boundaries)
        self.events = {}
        self.current = None
        self.windows = []
        self.stages = []
        self.probes = {}
        self.enabled = False
        self.pending_gap = False
        self.pending_tail = None
        self._wrapped = set()

    def context_bucket(self, tokens):
        lower = 1
        for upper in self.context_boundaries:
            if tokens <= upper:
                return f"{lower}:{upper}"
            lower = upper + 1
        raise ValueError("context exceeds prefetch calibration coverage")

    def attach(self, engine, controller):
        self.engine, self.controller = engine, controller
        self.events.clear()
        self.probes.clear()
        self.pending_gap = False
        self.pending_tail = None
        self.gap_start = torch.cuda.Event(enable_timing=True, external=True)
        self.gap_copy_end = torch.cuda.Event(enable_timing=True, external=True)
        self.gap_copy_recorded = False
        cache = controller.cache
        self.probe_layers = {layout: max(i for i, value in enumerate(cache.layer_layout) if value == layout)
                             for layout in set(cache.layer_layout)}
        target_layers = getattr(getattr(engine.model.model, "layers", None), "op_list", ())
        draft_layers = getattr(getattr(engine.mtp, "layers", None), "op_list", ())
        layers = [(False, i, layer) for i, layer in enumerate(target_layers)]
        layers += [(True, i, layer) for i, layer in enumerate(draft_layers)]
        for draft, index, layer in layers:
            if id(layer) in self._wrapped:
                continue
            original = layer.forward

            def forward(*args, _original=original, _index=index, _draft=draft, **kwargs):
                if self.current is None:
                    return _original(*args, **kwargs)
                stage = "draft_layer" if _draft else "layer"
                marker = (self.draft_index, _index) if _draft else _index
                with torch.cuda.nvtx.range(f"PREFETCH_LAYER/{self.current}/{stage}={marker}"):
                    self.mark(stage + "_start", marker)
                    result = _original(*args, **kwargs)
                    self.mark(stage + "_end", marker)
                    return result

            layer.forward = forward
            self._wrapped.add(id(layer))

    def begin(self, phase, rows, suffix):
        self.current = phase, rows, suffix
        self.draft_index = 0
        self.mark("begin")

    def end(self):
        self.mark("end")
        self.current = None

    def mark(self, name, layer=-1, stream=None):
        if self.current is None:
            return
        key = (*self.current, name, layer)
        event = self.events.get(key)
        if event is None:
            event = torch.cuda.Event(enable_timing=True, external=True)
            self.events[key] = event
        with torch.cuda.nvtx.range(f"PREFETCH_CALIB/{'/'.join(map(str, key))}"):
            event.record(self.engine.stream if stream is None else stream)

    def draft_begin(self):
        if self.current is None:
            return
        self.mark("draft_start", self.draft_index)

    def draft_end(self):
        if self.current is None:
            return
        self.mark("draft_end", self.draft_index)
        self.draft_index += 1

    def record_compute(self, layer, hidden, selected, slots, ids):
        if self.current is None or self.current[2] != "body":
            return
        phase, rows, _ = self.current
        layout = self.controller.cache.layer_layout[layer.layer_id]
        if layer.layer_id != self.probe_layers[layout]:
            return
        key = phase, layout, rows
        probe = self.probes.get(key)
        if probe is None:
            probe = dict(phase=phase, layer=layer, graph_width=rows, valid_rows=rows,
                         compute_rows=hidden.shape[0], last_tile=True,
                         hidden=torch.empty_like(hidden), selected=torch.empty_like(selected),
                         slots=torch.empty_like(slots), ids=torch.empty_like(ids))
            self.probes[key] = probe
        for name, source in (("hidden", hidden), ("selected", selected), ("slots", slots), ("ids", ids)):
            probe[name].copy_(source)

    def fork_gap(self, continuing):
        self.pending_gap = self.enabled and continuing
        self.gap_copy_recorded = False
        if self.pending_gap:
            self.gap_start.record(self.engine.stream)

    def gap_copied(self, stream):
        if self.pending_gap:
            self.gap_copy_end.record(stream)
            self.gap_copy_recorded = True

    def tail_submitted(self, phase, rows):
        self.pending_tail = (phase, rows, "tail") if self.enabled else None

    def _elapsed(self, key, start, end):
        first = self.events.get((*key, *start))
        last = self.events.get((*key, *end))
        if first is None or last is None:
            raise ValueError(f"missing calibration event {key}: {start} -> {end}")
        value = first.elapsed_time(last)
        if value < 0:
            raise ValueError("prefetch calibration event ordering is invalid")
        return value

    def body_finished(self, phase, rows, context_tokens):
        if not self.enabled:
            self.pending_gap = False
            self.pending_tail = None
            return
        key = phase, rows, "body"
        context = self.context_bucket(context_tokens)
        layers = self.controller.cache.num_layers
        for source in range(layers - 1):
            target = source if phase == "preverify" else source + 1
            row = dict(phase=phase, layer=target, rows=rows, context=context,
                context_tokens=context_tokens, graph_width=rows, valid_rows=rows,
                source_layer=source, duration_ms=self._elapsed(key, ("fork_ready", source),
                                                               ("route_ready", source + 1)))
            copy_done = self.events.get((*key, "prefetch_copy_done", source))
            if copy_done is not None:
                join_ready = self.events[(*key, "route_ready", source + 1)]
                row["residual_copy_wait_ms"] = max(0., join_ready.elapsed_time(copy_done))
            self.windows.append(row)
        for layer in range(layers):
            self.stages.append(dict(phase=phase, layer=layer, rows=rows, context=context,
                context_tokens=context_tokens,
                layer_ms=self._elapsed(key, ("layer_start", layer), ("layer_end", layer)),
                attention_and_router_ms=self._elapsed(key, ("layer_start", layer), ("route_ready", layer)),
                demand_and_plan_ms=self._elapsed(key, ("route_ready", layer), ("fork_ready", layer)),
                expert_compute_ms=self._elapsed(key, ("fork_ready", layer), ("compute_done", layer))))
        if self.pending_gap and phase == "preverify":
            join = self.events[(*key, "gap_join", -1)]
            row = dict(phase="draft_gap", layer=None, rows=rows, context=context,
                context_tokens=context_tokens, graph_width=rows, valid_rows=rows,
                duration_ms=self.gap_start.elapsed_time(join))
            if self.gap_copy_recorded:
                row["residual_copy_wait_ms"] = max(0., join.elapsed_time(self.gap_copy_end))
            self.windows.append(row)
        self.pending_gap = False
        keys = [key] + ([self.pending_tail] if self.pending_tail else [])
        for stage_key in keys:
            indices = sorted(k[-1] for k in self.events if k[:3] == stage_key and k[3] == "draft_start")
            for index in indices:
                self.stages.append(dict(phase="draft", owner_phase=stage_key[0], suffix=stage_key[2],
                    rows=stage_key[1], context=context, draft_call=index,
                    duration_ms=self._elapsed(stage_key, ("draft_start", index), ("draft_end", index))))
            layers = sorted(k[-1] for k in self.events if k[:3] == stage_key and k[3] == "draft_layer_start")
            for call, layer in layers:
                self.stages.append(dict(phase="draft_layer", owner_phase=stage_key[0], suffix=stage_key[2],
                    rows=stage_key[1], context=context, draft_call=call, layer=layer,
                    duration_ms=self._elapsed(stage_key, ("draft_layer_start", (call, layer)),
                                             ("draft_layer_end", (call, layer)))))
        self.pending_tail = None

    def clear_samples(self):
        self.windows.clear()
        self.stages.clear()
        self.reset_request()

    def reset_request(self):
        self.pending_gap = False
        self.pending_tail = None
        self.gap_copy_recorded = False

    def coverage_counts(self):
        counts = defaultdict(int)
        for row in self.windows:
            counts[row["phase"], row["rows"], row["context"]] += 1
        return {"/".join(map(str, key)): value for key, value in sorted(counts.items())}
