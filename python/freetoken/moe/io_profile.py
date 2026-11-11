"""Inference-only MoE transfer payloads, grouped by scheduler forward phase."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

COMPONENTS = {
    "expert_weight_h2d_bytes": ("h2d", "explicit_copy"),
    "expert_weight_mapped_read_bytes": ("h2d", "mapped_read"),
    "expert_weight_d2d_bytes": ("d2d", "device_copy"),
    "activation_d2h_bytes": ("d2h", "explicit_copy"),
    "routing_metadata_d2h_bytes": ("d2h", "explicit_copy"),
    "routing_metadata_h2d_bytes": ("h2d", "explicit_copy"),
    "cpu_output_h2d_bytes": ("h2d", "explicit_copy"),
    "cpu_output_mapped_read_bytes": ("h2d", "mapped_read"),
    "offline_table_value_h2d_bytes": ("h2d", "explicit_copy"),
    "offline_table_value_mapped_read_bytes": ("h2d", "mapped_read"),
    "lookup_metadata_d2h_bytes": ("d2h", "explicit_copy"),
    "request_slot_h2d_bytes": ("h2d", "explicit_copy"),
    "cpu_flag_mapped_write_d2h_bytes": ("d2h", "mapped_write"),
    "cpu_flag_mapped_read_h2d_bytes": ("h2d", "mapped_read"),
}


class MoeIoProfiler:
    def __init__(self, config, device):
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if _skip_fast_index_copy_enabled():
            raise ValueError("profile-io requires expert copies; disable FREETOKEN_SKIP_FAST_INDEX_COPY")
        destination = config.profile_io
        rank, world = config.tp_info.rank, config.tp_info.size
        if destination == "auto":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            destination = f"artifacts/profile_io/{stamp}_pid{os.getpid()}_rank{rank}.json"
        path = Path(destination).expanduser().resolve()
        if world > 1 and config.profile_io != "auto":
            path = path.with_name(f"{path.stem}_rank{rank}{path.suffix}")
        if path == Path("/tmp") or Path("/tmp") in path.parents:
            raise ValueError("profile-io output must use a persistent directory outside /tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"profile-io output must be fresh: {path}")
        self.path = path
        self.device = torch.device(device)
        self.counters = torch.zeros(len(COMPONENTS), device=self.device, dtype=torch.int64)
        self.indices = {name: i for i, name in enumerate(COMPONENTS)}
        self.phases = {name: dict(steps=0, real_rows=0, padded_rows=0,
                                 component_bytes=dict.fromkeys(COMPONENTS, 0))
                       for name in ("prefill", "decode")}
        self.metadata = dict(
            checkpoint=config.model_path, moe_strategy=config.moe_strategy,
            moe_policy=getattr(config, "moe_policy", "native"),
            tablemoe_lookup_mode=("none" if getattr(config, "tablemoe_skip_lookup", False)
                                 else getattr(config, "tablemoe_lookup_mode", "all")),
            tablemoe_skip_lookup=getattr(config, "tablemoe_skip_lookup", False),
            dtype=str(config.dtype), rank=rank, world_size=world,
        )
        self.failed_forwards = 0
        self.closed = False
        self.write("running")
        logger.info("MoE profile-io enabled: %s; synchronizes each forward for byte accounting", path)

    def add(self, component, amount):
        if isinstance(amount, torch.Tensor):
            amount = amount.reshape(())
        self.counters[self.indices[component]].add_(amount)

    def snapshot(self):
        if self.device.type == "cuda":
            # Copies may run on the prefill or table lookup side stream.
            torch.cuda.synchronize(self.device)
        return self.counters.cpu().tolist()

    @contextmanager
    def batch(self, batch):
        if self.closed:
            raise RuntimeError("profile-io is closed")
        before = self.snapshot()
        try:
            yield
        except BaseException:
            self.failed_forwards += 1
            self.write("incomplete")
            raise
        after = self.snapshot()
        phase = self.phases["prefill" if batch.is_prefill else "decode"]
        delta = [end - start for start, end in zip(before, after)]
        if any(value < 0 for value in delta):
            raise RuntimeError("profile-io counters were reset during a forward")
        for name, value in zip(COMPONENTS, delta):
            phase["component_bytes"][name] += value
        phase["steps"] += 1
        rows = sum(req.extend_len if batch.is_prefill else 1 for req in batch.reqs)
        phase["real_rows"] += rows
        phase["padded_rows"] += batch.input_ids.numel()
        self.write("running")

    def report(self, status="running"):
        result = dict(schema="freetoken_profile_io_v1", status=status,
                      scope="moe", measurement="logical_transfer_payload",
                      step_definition="one scheduler batch forward across all MoE layers",
                      total_definition="h2d_bytes + d2h_bytes; d2d_bytes is separate",
                      excluded=["startup_weights", "graph_capture", "internal_warmup",
                                "attention", "encoder", "kv_cache", "sampling",
                                "general_control_io", "profiler_telemetry", "pcie_protocol_overhead"],
                      physical_pcie_bytes=None, failed_forwards=self.failed_forwards,
                      includes_padding=True, metadata=self.metadata)
        for name, phase in self.phases.items():
            counts = phase["component_bytes"].copy()
            by_direction = {direction: sum(counts[key] for key, spec in COMPONENTS.items()
                                           if spec[0] == direction)
                            for direction in ("h2d", "d2h", "d2d")}
            total = by_direction["h2d"] + by_direction["d2h"]
            steps = phase["steps"]
            result[name] = dict(
                steps=steps, real_rows=phase["real_rows"], padded_rows=phase["padded_rows"],
                total_bytes=total, total_mib=total / 2**20,
                h2d_bytes=by_direction["h2d"], d2h_bytes=by_direction["d2h"],
                d2d_bytes=by_direction["d2d"], component_bytes=counts,
                mean_bytes_per_step=total / steps if steps else None,
                mean_mib_per_step=total / steps / 2**20 if steps else None,
                mean_component_bytes_per_step={key: value / steps if steps else None
                                               for key, value in counts.items()},
            )
        return result

    def write(self, status):
        pending = self.path.with_suffix(self.path.suffix + ".partial")
        pending.write_text(json.dumps(self.report(status), indent=2, allow_nan=False) + "\n")
        pending.replace(self.path)

    def close(self):
        if not self.closed:
            self.write("incomplete" if self.failed_forwards else "complete")
            self.closed = True
