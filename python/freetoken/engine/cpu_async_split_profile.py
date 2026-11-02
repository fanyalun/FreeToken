from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = 4
CPU_PREFIX_PROTOCOL = "cpu_prefix_compact_replay_v1"
TIMINGS = ("cpu_prefix_seconds", "gpu_generation_seconds")


def validate_suffix_gpu_experts(value, native_top_k=None):
    if type(value) is not int or value < 1 or (native_top_k is not None and value >= native_top_k):
        raise ValueError("CPU split requires 1 <= cpu_async_suffix_gpu_experts < native routing top-k")


def select_split(measurements, num_layers, *, margin_seconds=0.0):
    if type(num_layers) is not int or num_layers < 2:
        raise ValueError("split calibration requires at least two Target layers")
    if not math.isfinite(margin_seconds) or margin_seconds < 0:
        raise ValueError("split calibration margin must be finite and nonnegative")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError("split calibration requires measured windows")
    grouped = {}
    for row in measurements:
        if not isinstance(row, dict):
            raise ValueError("invalid split measurement")
        layers = row.get("prefix_layers")
        if type(layers) is not int or not 1 <= layers < num_layers:
            raise ValueError("invalid calibrated CPU prefix layer count")
        for field in ("context_tokens", "rows"):
            if type(row.get(field)) is not int or row[field] < 1:
                raise ValueError(f"split measurement requires positive {field}")
        for field in TIMINGS:
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"split measurement requires finite nonnegative {field}")
        if row["gpu_generation_seconds"] == 0:
            raise ValueError("GPU generation measurement must be positive")
        grouped.setdefault(layers, []).append(row)
    request_mode = any("sample_id" in row for row in measurements)
    if request_mode and any(not isinstance(row.get("sample_id"), str) or not row.get("input_sha256")
                            for row in measurements):
        raise ValueError("request calibration requires sample IDs and frozen input hashes")
    workloads = ({(row["sample_id"], row["input_sha256"]) for row in next(iter(grouped.values()))}
                 if request_mode else Counter((row["context_tokens"], row["rows"])
                                              for row in next(iter(grouped.values()))))
    summaries = []
    for layers, rows in sorted(grouped.items()):
        actual = ({(row["sample_id"], row["input_sha256"]) for row in rows} if request_mode
                  else Counter((row["context_tokens"], row["rows"]) for row in rows))
        if actual != workloads:
            raise ValueError("every candidate split must measure the same samples or same context and row workloads")
        gaps = []
        for row in rows:
            cpu = row["cpu_prefix_seconds"]
            gpu = row["gpu_generation_seconds"]
            gaps.append(gpu - cpu - margin_seconds)
        mean_cpu = sum(row["cpu_prefix_seconds"] for row in rows) / len(rows)
        mean_gpu = sum(row["gpu_generation_seconds"] for row in rows) / len(rows)
        summaries.append(dict(prefix_layers=layers, windows=len(rows), covered=mean_gpu > mean_cpu + margin_seconds,
                              mean_cpu_prefix_seconds=mean_cpu, mean_gpu_generation_seconds=mean_gpu,
                              minimum_coverage_margin_seconds=min(gaps),
                              uncovered_windows=sum(gap < 0 for gap in gaps),
                              mean_coverage_gap_seconds=sum(gaps) / len(gaps)))
    covered = [row for row in summaries if row["covered"]]
    if not covered:
        raise ValueError("no measured CPU prefix fits GPU generation; measure a smaller prefix")
    selected = max(covered, key=lambda row: row["prefix_layers"])
    return dict(prefix_layers=selected["prefix_layers"], margin_seconds=margin_seconds,
                objective="maximum_mean_covered_prefix_layers", aggregation="pooled_window_arithmetic_mean",
                candidates=summaries)


def build_split_identity(config, device=None, *, hardware=None):
    from .speculative_prefetch_profile import _checkpoint_identity, _hardware_identity, _runtime_identity
    from freetoken.layers.embedding import _FUSED_GREEDY
    from freetoken.moe.fused import _GROUPED_DECODE
    from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState

    fields = ("dtype", "moe_strategy", "attention_backend", "gpu_memory_budget_bytes", "memory_ratio",
              "moe_cache_auto", "moe_cache_scope", "speculative_depth", "speculative_graph",
              "cpu_async_loops", "cpu_async_recovery", "cpu_async_core_ids", "moe_cpu_core_ids",
              "cpu_async_recovery_loops", "cpu_async_suffix_prefetch_wait",
              "cpu_async_stop_when_ready",
              "cpu_async_kernel_profile", "cpu_async_kernel_components",
              "speculative_preverify_top_h", "speculative_max_input_tokens", "speculative_max_output_tokens",
              "cpu_async_suffix_prefetch_experts", "cpu_async_suffix_gpu_experts", "moe_hybrid_max_fetch")
    settings = {name: getattr(config, name) for name in fields}
    settings["dtype"] = str(config.dtype)
    settings.update(requested_moe_cache_size=config.moe_cache_size,
                    cpu_prefix_protocol=CPU_PREFIX_PROTOCOL,
                    kernel_profile=dict(fused_greedy=_FUSED_GREEDY, grouped_decode=_GROUPED_DECODE,
                                        chunk_gdn=QwenGDNVerifyState.chunk_enabled),
                    max_rows=config.cpu_async_loops * (config.speculative_depth + 1) + 1,
                    max_context=config.speculative_max_input_tokens + config.speculative_max_output_tokens,
                    num_layers=config.model_config.num_layers, tp_size=config.tp_info.size)
    identity = dict(checkpoint=_checkpoint_identity(config.model_path), settings=settings,
                    runtime_sha256=_runtime_identity(),
                    hardware=hardware if hardware is not None else _hardware_identity(device))
    return json.loads(json.dumps(identity, sort_keys=True))


def build_split_profile(identity, measurements, num_layers, *, margin_seconds=0.0):
    return dict(schema_version=SCHEMA_VERSION, state_protocol=CPU_PREFIX_PROTOCOL, identity=copy.deepcopy(identity),
                measurements=copy.deepcopy(measurements),
                selection=select_split(measurements, num_layers, margin_seconds=margin_seconds),
                num_layers=num_layers)


def load_split_profile(path, identity=None):
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("CPU split profile must be an existing JSON file")
    data = json.loads(Path(path).expanduser().read_bytes())
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported CPU split profile schema; recalibrate for maximum covered prefix")
    if data.get("state_protocol") != CPU_PREFIX_PROTOCOL:
        raise ValueError("CPU split profile state protocol mismatch; recalibrate for compact replay")
    if not isinstance(data.get("identity"), dict) or not data["identity"]:
        raise ValueError("CPU split profile requires model, runtime and hardware identity")
    if identity is not None and data["identity"] != identity:
        raise ValueError("CPU split profile identity mismatch; recalibrate for current model, cores and geometry")
    selection = data.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("CPU split profile requires a measured selection")
    expected = select_split(data.get("measurements"), data.get("num_layers"),
                            margin_seconds=selection.get("margin_seconds", 0.0))
    if selection != expected:
        raise ValueError("CPU split profile selection does not match measured windows")
    return data


def prepare_split_profile(config, device=None):
    layers = config.cpu_async_prefix_layers
    if layers is None:
        return None
    mc = config.model_config
    if (getattr(mc, "model_type", None) not in ("qwen3_5_moe", "qwen3_5_moe_text")
            or type(layers) is not int or not 1 <= layers < mc.num_layers):
        raise ValueError("CPU prefix split requires native Qwen3.5/3.6 MoE and 1 <= L < Target layers")
    validate_suffix_gpu_experts(config.cpu_async_suffix_gpu_experts, mc.num_experts_per_tok)
    if not config.cpu_async_core_ids or not config.moe_cpu_core_ids:
        raise ValueError("configure disjoint CPU prefix and Hybrid executor cores before split preparation")
    path = config.cpu_async_split_profile
    receipt = dict(source="explicit", prefix_layers=layers, validated=True)
    if path is not None:
        identity = build_split_identity(config, device)
        data = load_split_profile(path, identity)
        if data["selection"]["prefix_layers"] != layers or data["num_layers"] != mc.num_layers:
            raise ValueError("CPU split profile layer geometry changed after configuration")
        receipt.update(source="profile", profile_path=str(Path(path).expanduser().resolve()),
                       profile_sha256=hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest(),
                       identity=identity, selection=copy.deepcopy(data["selection"]))
    object.__setattr__(config, "_cpu_async_split_receipt", receipt)
    return receipt
