from __future__ import annotations

from types import SimpleNamespace

import torch

from .preverify_policy import merge_identity


class ResidentExpertController:
    gpu_schedule = True
    cross_layer_prefetch = False
    compact_control = False
    prefetch = False
    implementation = "resident"
    stage_candidates = False
    graph_replays = 0

    def __init__(self, engine, policy, *, audit=False):
        mc = engine.config.model_config
        self.policy, self.audit = policy, audit
        self.copy_stream = engine.stream
        self.cache = SimpleNamespace(num_layers=mc.num_moe_layers, num_experts=mc.num_experts,
                                     cache_size=mc.num_moe_layers * mc.num_experts)
        self.ledger = SimpleNamespace(capacity=lambda: dict(protected_experts=0, blocked_slots=0,
                                                          available_slots=self.cache.cache_size))
        self.graphs, self.stats = {}, {}
        self.prefetched, self.used_prefetch = set(), set()
        self.active, self.phase, self.loop = False, "inactive", 0
        self.records, self.device_records = [], []
        self.ablation_layer = None
        self.signals = None
        self.signal_offset = 0
        for li, block in enumerate(engine.model.model.layers.op_list):
            block.mlp.experts.preverify_controller = self
            block.mlp.experts.preverify_layer_id = li

    def begin(self, epoch):
        if self.active:
            raise RuntimeError("resident window already active")
        self.active = True
        self.phase = "inactive"
        self.records, self.device_records = [], []

    def set_phase(self, phase, loop=0):
        self.phase, self.loop = phase, loop

    def route(self, layer, hidden, weights, ids):
        from .fused import fused_experts_impl

        keep = (self.policy.mask(weights, layer.preverify_layer_id) if self.phase == "preverify"
                else torch.ones_like(weights, dtype=torch.bool))
        if self.phase == "preverify" and self.ablation_layer is not None:
            keep = torch.full_like(keep, layer.preverify_layer_id != self.ablation_layer)
        identity = self.policy.identity_term(hidden, weights, keep) if self.phase == "preverify" else None
        if self.signals is not None and self.phase == "preverify":
            li, offset = layer.preverify_layer_id, self.signal_offset
            self.signals[0][li, offset:offset + weights.shape[0]].copy_(weights)
            self.signals[1][li, offset:offset + weights.shape[0]].copy_(keep)
        if self.audit:
            self.device_records.append(dict(phase=self.phase, loop=self.loop,
                layer=layer.preverify_layer_id, ids=ids.clone(), keep=keep.clone()))
        if (self.phase != "preverify" or
                (self.ablation_layer is None and self.policy.kind == "fixed_topk" and self.policy.top_k == 8) or
                (self.ablation_layer is not None and self.ablation_layer != layer.preverify_layer_id)):
            return layer._resident_gemm(hidden, weights, ids)
        selected = torch.where(keep, weights, 0).contiguous()
        compute_ids = torch.where(keep, ids, -1).contiguous()
        result = fused_experts_impl(hidden, layer.gate_up_proj, layer.down_proj, selected, compute_ids,
            layer.activation, layer.apply_router_weight_on_input, float(layer.alpha),
            float("inf") if layer.limit is None else layer.limit,
            skip_invalid_routes=True)
        return merge_identity(result, identity)

    def export_records(self):
        self.records = [{**r, "ids": r["ids"].cpu().tolist(), "keep": r["keep"].cpu().tolist()}
                        for r in self.device_records]

    def decide(self, loop, consumed):
        pass

    def finish_loop(self, reconcile):
        pass

    def check_error(self):
        pass

    def close(self, **kwargs):
        self.phase, self.active = "inactive", False

    def transfer_audit(self):
        return {"logical_bytes": 0, "loads": 0, "resident": True}
