from __future__ import annotations


def prepare_prefetch(config):
    active = (config.speculative_prefetch and config.moe_strategy != "fused"
              and config.speculative_mode == "three_level_gpu" and not config._speculative_legacy)
    fields = ("speculative_prefetch_layer_experts", "speculative_verify_prefetch_layer_experts",
              "speculative_prefetch_gap_capacity")
    if not active:
        for name in fields:
            if getattr(config, name) is None:
                object.__setattr__(config, name, 0)


def validate_prefetch(config):
    for name in ("speculative_prefetch_layer_experts", "speculative_verify_prefetch_layer_experts",
                 "speculative_prefetch_gap_capacity"):
        value = getattr(config, name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    layers = config.model_config.num_moe_layers
    for name in ("speculative_prefetch_layer_quotas", "speculative_verify_prefetch_layer_quotas"):
        values = getattr(config, name)
        if values is not None and (len(values) != layers or any(type(q) is not int or q < 0 for q in values)):
            raise ValueError(f"{name} must have one nonnegative integer per MoE layer")
    verify = config.speculative_verify_prefetch_layer_quotas
    if verify is not None and verify[0] != 0:
        raise ValueError("Full Verify prefetch must start with zero")


def resolve_prefetch_rates(engine):
    from freetoken.moe.bench_profile import layer_kind_key, load_layer_kinds, prefetch_rates
    from freetoken.utils import init_logger
    from .engine import _profile_gpu, offload_layer_methods

    cache, config = engine.moe_offload_cache, engine.config
    methods = offload_layer_methods(engine.model)
    gpu_name, gpu_uuid = _profile_gpu(engine.device.index)
    rows = load_layer_kinds(gpu_name=gpu_name, gpu_uuid=gpu_uuid) or {}
    phases, missing = [], False
    for phase in ("preverify", "verify"):
        values = []
        for li, method in enumerate(methods):
            fmt, opts = method.cpu_spec()
            c = method.cfg
            threads = cache.cpu_executor.num_threads if cache.cpu_executor is not None else config.moe_cpu_threads
            key = layer_kind_key(fmt, opts, c.hidden, c.local_intermediate, c.top_k, threads)
            row = rows.get(key)
            if row is None and cache.cpu_executor is None:
                prefix = key.rsplit(":t", 1)[0] + ":t"
                matches = [v for k, v in rows.items() if k.startswith(prefix)]
                row = max(matches, key=lambda v: v.get("measured_at", ""), default=None)
            size = sum(bank[0].numel() * bank.element_size() for bank in cache.pool_bank_views(li))
            rates = prefetch_rates(row, size, hybrid=phase in config.speculative_hybrid_phases
                                   and cache.decode_target == "hybrid")
            missing |= rates is None
            values.append(rates)
        phases.append(tuple(values))
    if missing:
        init_logger(__name__).warning_rank0(
            "Automatic prefetch needs GPU/PCIe probes from `ft bench bw --model <checkpoint>`; "
            "unmeasured layers use zero automatic prefetch. Explicit quotas still apply.")
    return tuple(phases)


class DraftTimings:
    """Bounded observations of normal MTP work; event queries never wait for the GPU."""

    def __init__(self):
        self.events = {}
        self.pending = {}
        self.us_per_row = {}

    def _collect(self, stage):
        entry = self.pending.get(stage)
        if entry is not None:
            start, end, rows = entry
            if end.query():
                value = start.elapsed_time(end) * 1000 / rows
                previous = self.us_per_row.get(stage, value)
                self.us_per_row[stage] = previous * 0.75 + value * 0.25
                del self.pending[stage]

    def begin(self, stage, rows, stream):
        import torch
        if torch.cuda.is_current_stream_capturing():
            return None
        self._collect(stage)
        if stage in self.pending:
            return None
        if stage not in self.events:
            self.events[stage] = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start, end = self.events[stage]
        start.record(stream)
        return stage, start, end, rows, stream

    def end(self, sample):
        if sample is not None:
            stage, start, end, rows, stream = sample
            end.record(stream)
            self.pending[stage] = start, end, rows

    def window(self, reconcile_rows, draft_rows):
        for stage in ("reconcile", "draft"):
            self._collect(stage)
        return (self.us_per_row.get("reconcile", 0.0) * reconcile_rows
                + self.us_per_row.get("draft", 0.0) * draft_rows)
