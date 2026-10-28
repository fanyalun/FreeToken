from __future__ import annotations

import torch


class PrefetchWindowRecorder:
    """Read Graph timing events at the existing control synchronization."""

    def __init__(self):
        self.events = {}
        self.current = None
        self.windows, self.stages = [], []
        self.enabled = False
        self.pending_tail = None

    def attach(self, engine, controller):
        self.engine, self.controller = engine, controller
        for index, layer in enumerate(engine.model.model.layers.op_list):
            original = layer.forward

            def forward(*args, _original=original, _index=index, **kwargs):
                self.mark("layer_start", _index)
                result = _original(*args, **kwargs)
                self.mark("layer_end", _index)
                return result

            layer.forward = forward

    def begin(self, phase, rows, suffix):
        self.current = phase, rows, suffix
        self.draft_index = self.copy_index = 0
        self.mark("begin")

    def end(self):
        self.mark("end")
        self.current = None

    def mark(self, name, layer=-1, stream=None):
        if self.current is None:
            return
        key = (*self.current, name, layer)
        if key not in self.events:
            self.events[key] = torch.cuda.Event(enable_timing=True, external=True)
        self.events[key].record(self.engine.stream if stream is None else stream)

    def draft_begin(self):
        self.mark("draft_start", self.draft_index if self.current is not None else -1)

    def draft_end(self):
        if self.current is not None:
            self.mark("draft_end", self.draft_index)
            self.draft_index += 1

    def copy_begin(self, layer):
        self.mark("copy_start", (layer, self.copy_index))

    def copy_end(self, layer):
        self.mark("copy_end", (layer, self.copy_index))
        self.copy_index += 1

    def tail_submitted(self, phase, rows):
        self.pending_tail = (phase, rows, "tail") if self.enabled else None

    def _elapsed(self, key, name, index):
        return self.events[(*key, name + "_start", index)].elapsed_time(
            self.events[(*key, name + "_end", index)])

    def body_finished(self, phase, rows, context_tokens, *, capture_rows=None, suffix="body", draft_tail_count=0):
        if not self.enabled:
            self.pending_tail = None
            return
        key = phase, rows if capture_rows is None else capture_rows, suffix
        self.stages.append(dict(phase=phase, rows=rows, duration_ms=self._elapsed(key, "target", -1)))
        if draft_tail_count:
            self.stages.append(dict(phase="draft_tail", rows=draft_tail_count,
                                    duration_ms=self._elapsed(key, "proposal", -1)))
        layers = self.controller.cache.num_layers
        for layer in range(layers):
            copies = [k[-1] for k in self.events if k[:3] == key and k[3] == "copy_start" and k[-1][0] == layer]
            io_ms = sum(self._elapsed(key, "copy", index) for index in copies)
            layer_ms = self._elapsed(key, "layer", layer)
            self.windows.append(dict(phase=phase, layer=layer, rows=rows, context_tokens=context_tokens,
                duration_ms=layer_ms, io_ms=io_ms, compute_ms=max(0., layer_ms - io_ms)))
        draft_ms = []
        for stage_key in [key] + ([self.pending_tail] if self.pending_tail else []):
            indices = sorted(k[-1] for k in self.events if k[:3] == stage_key and k[3] == "draft_start")
            for index in indices:
                elapsed = self._elapsed(stage_key, "draft", index)
                draft_ms.append(elapsed)
                self.stages.append(dict(phase="draft", rows=stage_key[1], duration_ms=elapsed))
        if (phase == "preverify" or draft_tail_count) and self.pending_tail is not None:
            self.stages.append(dict(phase="draft_gap", rows=rows, duration_ms=sum(draft_ms)))
        self.pending_tail = None

    def clear_samples(self):
        self.windows.clear()
        self.stages.clear()
        self.reset_request()

    def reset_request(self):
        self.pending_tail = None
