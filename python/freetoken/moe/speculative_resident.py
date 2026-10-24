from __future__ import annotations

from types import SimpleNamespace

import torch

from .preverify_policy import merge_identity


class ResidentExpertController:
    gpu_schedule = True
    cross_layer_prefetch = False
    compact_control = True
    prefetch = False
    implementation = "resident"
    stage_candidates = True
    optimize_admission = True
    graph_replays = 0

    def __init__(self, engine, policy, *, audit=False):
        mc = engine.config.model_config
        config = engine.config
        self.policy, self.audit = policy, audit
        self.stream = engine.stream
        self.copy_stream = engine.stream
        self.cache = SimpleNamespace(num_layers=mc.num_moe_layers, num_experts=mc.num_experts,
                                     cache_size=mc.num_moe_layers * mc.num_experts,
                                     banks=(), pool_sizes=(), pool_of_layer=(), reset=lambda: None)
        self.max_loops = config.speculative_max_loops
        self.inner_rows = config.speculative_mtp_depth + 1
        self.max_rows = min(config.speculative_submit_tokens or config.speculative_max_output_tokens,
                            self.max_loops * self.inner_rows + 1, config.speculative_max_output_tokens)
        self.layer_quota = self.gap_capacity = 0
        self.keep_hits = self.renormalize = self.collect_signals = False
        self.verify_prefetch_enabled = False
        self.verify_layer_quotas = (0,) * mc.num_moe_layers
        self.cpu_stream = None
        self.valid_rows = self.eos_mask = None
        self.gap_active = self.gap_planned = False
        self.candidate_buffer = torch.empty(self.max_rows, dtype=torch.int32, device=engine.device)
        size = max(self.inner_rows, self.max_rows) + 8
        self.control_device = torch.empty(size, dtype=torch.int64, device=engine.device)
        self.control_host = torch.empty(size, dtype=torch.int64, pin_memory=True)
        self.control_event = torch.cuda.Event()
        self.error = torch.zeros(1, dtype=torch.int32, device=engine.device)
        self.cache_conflict = torch.zeros_like(self.error)
        self.protected = torch.zeros(mc.num_moe_layers, dtype=torch.int32, device=engine.device)
        self.pending = torch.zeros(self.cache.cache_size, dtype=torch.int32, device=engine.device)
        self.ledger = SimpleNamespace(capacity=lambda: dict(protected_experts=0, blocked_slots=0,
                                                          available_slots=self.cache.cache_size))
        self.graphs, self.stats = {}, {}
        self.prefetched, self.used_prefetch = set(), set()
        self.active, self.phase, self.loop = False, "inactive", 0
        self.records, self.device_records = [], []
        self.ablation_layer = None
        self.signals = None
        self.signal_offset = 0
        from freetoken.layers.moe import iter_moe_layers
        for li, experts in enumerate(iter_moe_layers(engine.model)):
            experts.preverify_controller = self
            experts.preverify_layer_id = li

    def begin(self, epoch):
        if self.active:
            raise RuntimeError("resident window already active")
        self.active = True
        self.phase = "inactive"
        self.records, self.device_records = [], []

    def set_phase(self, phase, loop=0):
        self.phase, self.loop = phase, loop

    def route(self, layer, hidden, weights, ids):
        keep = (self.policy.mask(weights, layer.preverify_layer_id) if self.phase == "preverify"
                else torch.ones_like(weights, dtype=torch.bool))
        if self.phase == "preverify" and self.ablation_layer is not None:
            keep = torch.full_like(keep, layer.preverify_layer_id != self.ablation_layer)
        if self.valid_rows is not None:
            keep &= torch.arange(weights.shape[0], device=weights.device)[:, None] < self.valid_rows
        identity = self.policy.identity_term(hidden, weights, keep) if self.phase == "preverify" else None
        if self.signals is not None and self.phase == "preverify":
            li, offset = layer.preverify_layer_id, self.signal_offset
            self.signals[0][li, offset:offset + weights.shape[0]].copy_(weights)
            self.signals[1][li, offset:offset + weights.shape[0]].copy_(keep)
        if self.audit:
            self.device_records.append(dict(phase=self.phase, loop=self.loop,
                layer=layer.preverify_layer_id, ids=ids.clone(), keep=keep.clone()))
        if self.phase != "preverify" and self.valid_rows is None:
            return layer._resident_gemm(hidden, weights, ids)
        selected = torch.where(keep, weights, 0).contiguous()
        compute_ids = ids
        if self.phase == "preverify" and self.ablation_layer is None and self.policy.kind == "fixed_topk":
            # A resident bank has no ready-hit exception: only these h routes run.
            columns = weights.argsort(dim=-1, descending=True, stable=True)[:, :self.policy.top_k]
            selected = selected.gather(1, columns).contiguous()
            compute_ids = ids.gather(1, columns).contiguous()
        result = layer._resident_gemm(hidden, selected, compute_ids)
        return merge_identity(result, identity)

    def write_control(self, tokens, commit, *, inner, loop=None, candidates=0, candidate_limit=0,
                      remaining=0, depth=3, context=None, eos_mask=None):
        from .speculative_gpu import _control
        _control[(1,)](tokens, commit.input_count, commit.accepted_drafts, commit.terminal,
            self.error, self.protected, self.pending, self.control_device, self.candidate_buffer,
            candidates, candidate_limit, remaining, 0 if loop is None else loop + 1,
            tokens.numel(), self.cache.num_layers, self.cache.cache_size, 0,
            depth, self.max_loops, inner, self.audit or context is not None, True, context,
            CPU_FALLBACK=True, cache_conflict=self.cache_conflict, RUNTIME_ADMISSION=True,
            eos_mask=self.eos_mask if eos_mask is None else eos_mask)

    def drain_control(self, rows):
        size = rows + 8
        self.control_host[:size].copy_(self.control_device[:size], non_blocking=True)
        self.control_event.record(self.stream)
        self.control_event.synchronize()
        return self.control_host[:size].tolist()

    def read_control(self, tokens, commit, *, loop=None, candidates=0, candidate_limit=0,
                     remaining=0, depth=3):
        self.write_control(tokens, commit, inner=loop is not None, loop=loop, candidates=candidates,
                           candidate_limit=candidate_limit, remaining=remaining, depth=depth)
        return self.drain_control(tokens.numel())

    def retire_gap(self):
        pass

    def join_gap(self):
        pass

    def retire_layers(self):
        pass

    def retire_verify(self):
        pass

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

    def reset_request_metrics(self):
        pass

    def request_metrics(self):
        return {"available": True, "scope": "gpu_decode", "logical_total_bytes": 0,
                "logical_total_loads": 0, "resident": True}
