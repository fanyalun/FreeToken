"""Decode text score histograms on the production MoDES selection path."""

import os
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _histogram(weights, tokens, slots, special_ids, importance, keep, histogram, skipped,
               TOP_K: tl.constexpr, SPECIALS: tl.constexpr, DUMMY: tl.constexpr,
               N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = r // TOP_K
    token = tl.load(tokens + row, r < N, 0)
    special = tl.full((BLOCK,), False, tl.int1)
    for i in range(SPECIALS):
        special |= token == tl.load(special_ids + i)
    valid = (r < N) & (tl.load(slots + row, r < N, DUMMY) != DUMMY) & ~special
    w = tl.load(weights + r, r < N, 0).to(tl.float16).to(tl.float32)
    score = (w * tl.load(importance)).to(tl.float16)
    bits = score.to(tl.uint16, bitcast=True).to(tl.int32)
    tl.atomic_add(histogram + bits, 1, valid)
    exact = tl.load(keep + r, r < N, True)
    tl.atomic_add(skipped, tl.sum((valid & ~exact).to(tl.int32), 0))


def install():
    from freetoken.engine.engine import Engine
    from freetoken.moe.modes_runtime import ModesRuntime
    from freetoken.moe.tablemoe_artifacts import write_json
    from freetoken.scheduler.scheduler import Scheduler

    root = Path(os.environ["MODES_DECODE_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    active, records = {}, []
    original_init = ModesRuntime.__init__
    original_select = ModesRuntime._select
    original_forward = Engine.forward_batch
    original_free = Scheduler._free_req_resources

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.decode_histogram = torch.zeros((self.num_layers, 65536), device=self.device, dtype=torch.int32)
        self.decode_skipped = torch.zeros(self.num_layers, device=self.device, dtype=torch.int32)

    def select(self, layer, weights, token_ids, ws, is_prefill=False):
        original_select(self, layer, weights, token_ids, ws, is_prefill=is_prefill)
        if not is_prefill:
            _histogram[(triton.cdiv(weights.numel(), 256),)](
                weights, token_ids, ws["request_slots"], self.special_ids,
                self.decode_importance[layer, 1:], ws["keep"], self.decode_histogram[layer], self.decode_skipped[layer:],
                weights.shape[1], self.special_ids.numel(), self.dummy_slot, weights.numel(), 256)

    def forward(self, batch, *args, **kwargs):
        runtime = self.tablemoe_runtime
        if batch.is_prefill:
            for req in batch.reqs:
                if req.uid not in active:
                    if active:
                        raise ValueError("decode calibration requires one active request")
                    runtime.decode_histogram.zero_()
                    runtime.decode_skipped.zero_()
                    active[req.uid] = dict(steps=0, graph_steps=0)
        else:
            if len(active) != 1:
                raise ValueError("decode request boundary is missing")
            state = next(iter(active.values()))
            state["steps"] += 1
            state["graph_steps"] += int(self.graph_runner.can_use_cuda_graph(batch))
        return original_forward(self, batch, *args, **kwargs)

    def free(self, req):
        if req.table_idx != -1 and req.uid in active:
            state = active.pop(req.uid)
            runtime = self.engine.tablemoe_runtime
            hist = runtime.decode_histogram.cpu().long()
            skipped = runtime.decode_skipped.cpu().long()
            counts = hist.sum(1)
            if not torch.all(counts == counts[0]):
                raise ValueError("decode layer coverage differs")
            path = root / f"histogram_{len(records):03d}.pt"
            torch.save(dict(histogram=hist, skipped=skipped), path)
            records.append(dict(response_id=f"chatcmpl-{req.uid}", ordinal=len(records),
                histogram=path.name, text_routes=int(counts.sum()), skipped_text_routes=int(skipped.sum()),
                text_tokens=int(counts[0]) // runtime.top_k, **state))
            write_json(root / "requests.json", records)
        return original_free(self, req)

    ModesRuntime.__init__ = init
    ModesRuntime._select = select
    Engine.forward_batch = forward
    Scheduler._free_req_resources = free
