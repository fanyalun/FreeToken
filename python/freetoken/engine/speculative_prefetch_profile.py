from __future__ import annotations

import copy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


SCHEMA_VERSION = 2
QUOTA_FIELDS = ("speculative_prefetch_layer_experts", "speculative_prefetch_gap_capacity",
                "speculative_verify_prefetch_layer_experts")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def profile_key(identity):
    return hashlib.sha256(_canonical(identity).encode()).hexdigest()


def profile_directory():
    root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return root / "freetoken" / "speculative_prefetch"


def _checkpoint_identity(path):
    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    if not path.exists():
        raise ValueError(f"prefetch profile checkpoint does not exist: {path}")
    if path.suffix == ".gguf":
        stem = path.name.split("-00001-of-")[0]
        weights = sorted(directory.glob(stem + "-*-of-*.gguf")) or [path]
    else:
        weights = (sorted(directory.glob("*.safetensors")) + sorted(directory.glob("*.ftw"))
                   + sorted(directory.glob("*.gguf")))
    return dict(path=str(path), weights=[dict(name=p.name, bytes=p.stat().st_size,
        mtime_ns=p.stat().st_mtime_ns) for p in weights], metadata_sha256={p.name:
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.json"))},
        weight_identity="file sizes and nanosecond mtimes; full weight hashes not computed")


def _runtime_identity():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(p for directory in ("engine", "moe", "layers", "attention", "kernel", "models", "kvcache")
                   for p in (root / directory).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".cpp", ".cu", ".cuh", ".h", ".so"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _hardware_identity(device):
    import torch
    from freetoken.gpu_select import gpu_identity

    gpu = gpu_identity(device.index if device.index is not None else torch.cuda.current_device())
    gpu.pop("index", None)
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version,pci.bus_id",
        "--format=csv,noheader,nounits"], text=True)
    selected = [line.split(",") for line in query.splitlines() if line.split(",")[0].strip() == gpu["uuid"]]
    if len(selected) != 1:
        raise ValueError("cannot establish GPU driver and PCIe identity for prefetch calibration")
    gpu.update(driver=selected[0][1].strip(), pci_bus_id=selected[0][2].strip())
    bus = gpu["pci_bus_id"].lower().replace("00000000:", "0000:")
    pci = Path("/sys/bus/pci/devices") / bus
    gpu["pcie"] = {name: (pci / name).read_text().strip() for name in
                   ("numa_node", "local_cpulist", "max_link_speed", "max_link_width") if (pci / name).is_file()}
    cpu = Path("/proc/cpuinfo").read_text()
    model = next((line.split(":", 1)[1].strip() for line in cpu.splitlines()
                  if line.startswith("model name")), "unknown")
    packages = {}
    for name in ("triton", "flashlib", "flashinfer-python"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return dict(gpu=gpu, cpu=model, cpu_affinity=sorted(os.sched_getaffinity(0)),
                threads={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                torch=str(torch.__version__), cuda=torch.version.cuda, packages=packages)


def build_prefetch_identity(config, device=None, *, hardware=None):
    from .speculative_config import resolve_draft_method

    fields = ("dtype", "hf_overrides", "use_dummy_weight", "moe_strategy", "quant_backend", "attention_backend", "memory_ratio",
              "gpu_memory_budget_bytes", "moe_cache_scope", "moe_cache_split", "moe_cache_policy",
              "moe_prefill_overlap", "moe_prefill_hit_d2d", "page_size", "speculative_mode",
              "max_seq_len_override", "num_page_override", "num_token_override",
              "speculative_depth", "speculative_graph", "speculative_max_loops", "cpu_async_loops",
              "speculative_submit_tokens", "speculative_max_input_tokens", "speculative_max_output_tokens",
              "speculative_preverify_policy", "speculative_preverify_top_h", "speculative_preverify_replacement",
              "speculative_execution_implementation", "speculative_graph_implementation",
              "speculative_prefetch_demand_fence", "speculative_reference_math", "speculative_hybrid_phases",
              "speculative_hybrid_window_policy", "speculative_hybrid_cpu_backend", "moe_hybrid_max_fetch",
              "moe_cpu_threads", "cpu_async_core_ids", "moe_cpu_core_ids")
    values = {name: getattr(config, name) for name in fields}
    values["dtype"] = str(config.dtype)
    values["tp_size"] = config.tp_info.size
    values["max_running_req"] = config.max_running_req
    values["draft_method"] = resolve_draft_method(config)
    assets = {}
    for name in ("speculative_modes_calibration", "speculative_hybrid_profile"):
        path = getattr(config, name)
        if path:
            assets[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if config.moe_cache_split != "equal":
        path = Path(config.moe_cache_split.removeprefix("profile:"))
        assets["moe_cache_split"] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = dict(target=_checkpoint_identity(config.model_path),
        draft=_checkpoint_identity(config.speculative_draft_model) if config.speculative_draft_model else "native_mtp",
        runtime_sha256=_runtime_identity(), hardware=hardware if hardware is not None else _hardware_identity(device),
        settings=values, assets=assets)
    return json.loads(_canonical(identity))


def _nonnegative(value):
    return type(value) is int and value >= 0


def validate_profile_data(data, identity=None):
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported speculative prefetch profile schema")
    if not isinstance(data.get("identity"), dict):
        raise ValueError("prefetch profile is missing identity")
    if identity is not None and data["identity"] != identity:
        raise ValueError("prefetch profile identity mismatch; recalibrate for the current configuration")
    evidence = data.get("evidence", {})
    if (evidence.get("status") != "calibrated" or evidence.get("samples") != 1
            or evidence.get("method") != "single_sample_mean"):
        raise ValueError("prefetch profile requires single-sample mean calibration")
    artifacts = evidence.get("artifact_sha256")
    if (not isinstance(artifacts, dict) or not artifacts
            or any(not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                   or any(c not in "0123456789abcdef" for c in digest) for name, digest in artifacts.items())):
        raise ValueError("calibrated prefetch profile requires measurement hashes")
    quotas, geometry = data.get("quotas", {}), data.get("geometry", {})
    preverify, verify = quotas.get("preverify"), quotas.get("verify")
    if (not isinstance(preverify, list) or not preverify or not all(map(_nonnegative, preverify))
            or not isinstance(verify, list) or len(verify) != len(preverify)
            or not all(map(_nonnegative, verify)) or verify[0] != 0
            or not _nonnegative(quotas.get("gap_capacity"))):
        raise ValueError("invalid prefetch profile quotas; verify must start with zero")
    pools, owners = geometry.get("pool_sizes"), geometry.get("pool_of_layer")
    if (not isinstance(pools, list) or not pools or any(not _nonnegative(n) or n == 0 for n in pools)
            or not _nonnegative(geometry.get("total_slots")) or sum(pools) != geometry["total_slots"]
            or not isinstance(owners, list) or len(owners) != len(preverify)
            or any(not _nonnegative(p) or p >= len(pools) for p in owners)
            or sorted(set(owners)) != list(range(len(pools)))):
        raise ValueError("invalid prefetch profile pool geometry")
    for name in ("layer_layout", "expert_bytes_by_layer"):
        values = geometry.get(name)
        if not isinstance(values, list) or len(values) != len(preverify) or not all(map(_nonnegative, values)):
            raise ValueError(f"invalid prefetch profile {name}")
    layouts = geometry.get("bank_layouts")
    if not isinstance(layouts, list) or len(layouts) != len(preverify) or any(not isinstance(v, list) for v in layouts):
        raise ValueError("invalid prefetch profile bank_layouts")
    return data


def build_prefetch_profile(identity, quotas, geometry, evidence):
    data = dict(schema_version=SCHEMA_VERSION, identity=copy.deepcopy(identity), quotas=copy.deepcopy(quotas),
                geometry=copy.deepcopy(geometry), evidence=copy.deepcopy(evidence))
    return validate_profile_data(data)


def _requested(config):
    return dict(preverify=config.speculative_prefetch_layer_experts,
                preverify_layers=copy.deepcopy(config.speculative_prefetch_layer_quotas),
                gap_capacity=config.speculative_prefetch_gap_capacity,
                verify=config.speculative_verify_prefetch_layer_experts,
                verify_layers=copy.deepcopy(config.speculative_verify_prefetch_layer_quotas))


def _effective(config, layers):
    preverify = (list(config.speculative_prefetch_layer_quotas) if config.speculative_prefetch_layer_quotas is not None
                 else [config.speculative_prefetch_layer_experts or 0] * layers)
    verify = (list(config.speculative_verify_prefetch_layer_quotas)
              if config.speculative_verify_prefetch_layer_quotas is not None
              else [0] + [config.speculative_verify_prefetch_layer_experts or 0] * (layers - 1))
    gap = config.speculative_prefetch_gap_capacity or 0
    if not config.speculative_prefetch or config.moe_strategy == "fused" or config.speculative_mode == "sd":
        preverify, gap, verify = [0] * layers, 0, [0] * layers
    elif config.speculative_mode == "cpu_async":
        verify = [0] * layers
    return dict(preverify=preverify, gap_capacity=gap, verify=verify,
                max_gap_expert_loads=max(0, gap - 1))


def _calibration_hint(config):
    model_type = getattr(config.model_config, "model_type", None)
    if (config.speculative_mode != "three_level_gpu" or config.moe_strategy != "offload"
            or config.speculative_graph != "full"
            or model_type not in ("qwen3_5_moe", "qwen3_5_moe_text", "qwen3_moe")):
        return ("This mode/backend needs its own independently calibrated profile; "
                "the offline calibrator currently supports Qwen MTP/EAGLE3 three_level_gpu/offload/full. "
                "Alternatively specify all applicable prefetch quotas explicitly.")
    repo = Path(__file__).resolve().parents[3]
    root = Path.home() / "freetoken_benchmark_results/speculative_modes_v1"
    output = root / f"prefetch_calibration_{time.time_ns()}"
    command = [sys.executable, str(repo / "benchmarks/bench_speculative_modes.py"), "--model", config.model_path,
        "--mode", "three_level_gpu", "--backend", "offload", "--graph", "full",
        "--max-loops", str(config.speculative_max_loops), "--depth", str(config.speculative_depth),
        "--preverify-top-h", str(config.speculative_preverify_top_h),
        "--max-input-tokens", str(config.speculative_max_input_tokens),
        "--max-output-tokens", str(config.speculative_max_output_tokens),
        "--tokens", str(min(128, config.speculative_max_output_tokens)),
        "--memory-ratio", str(config.memory_ratio), "--samples", "1",
        "--calibrate-prefetch", "--purpose", "diagnostic", "--output", str(output)]
    if config.gpu_memory_budget_bytes is not None:
        command += ["--budget-gib", str(config.gpu_memory_budget_bytes / 2**30)]
    if config.speculative_draft_model:
        command += ["--draft-model", config.speculative_draft_model]
    temporary = root / "tmp"
    return ("Calibrate one sample with matching settings, then reuse its mean-time quotas: "
            f"mkdir -p {shlex.quote(str(temporary))} && TMPDIR={shlex.quote(str(temporary))} {shlex.join(command)}")


def prepare_prefetch_profile(config, device):
    put = lambda name, value: object.__setattr__(config, name, value)
    requested = _requested(config)
    inactive = (not config.speculative_prefetch or config.moe_strategy == "fused"
                or config.speculative_mode not in ("three_level_gpu", "cpu_async"))
    if config._speculative_legacy or inactive:
        for name in QUOTA_FIELDS:
            if getattr(config, name) is None:
                put(name, 0)
        receipt = dict(source="legacy" if config._speculative_legacy else "disabled", requested=requested,
                       validated=True, geometry=None)
        put("_speculative_prefetch_receipt", receipt)
        return receipt
    complete = ((requested["preverify"] is not None or requested["preverify_layers"] is not None)
                and requested["gap_capacity"] is not None
                and (requested["verify"] is not None or requested["verify_layers"] is not None))
    selection = config.speculative_prefetch_profile
    model_default = False
    if not complete and selection in (None, "auto") and config.speculative_mode == "three_level_gpu":
        model = config.model_config
        shape = tuple(getattr(model, name, None) for name in
                      ("hidden_size", "moe_intermediate_size", "num_experts", "num_moe_layers"))
        if (getattr(model, "model_type", None) in ("qwen3_5_moe", "qwen3_5_moe_text")
                and shape == (2048, 512, 256, 40)):
            if requested["preverify"] is None and requested["preverify_layers"] is None:
                put("speculative_prefetch_layer_experts", 1)
                put("speculative_prefetch_layer_quotas", [1] * 39 + [0])
            if requested["verify"] is None and requested["verify_layers"] is None:
                put("speculative_verify_prefetch_layer_experts", 2)
            if requested["gap_capacity"] is None:
                put("speculative_prefetch_gap_capacity", 10)
            complete, model_default = True, True
    if complete and selection in (None, "auto"):
        for name in QUOTA_FIELDS:
            if getattr(config, name) is None:
                put(name, 0)
        receipt = dict(source="model_default" if model_default else "manual", requested=requested,
                       validated=True, geometry=None)
        put("_speculative_prefetch_receipt", receipt)
        return receipt
    if selection is None:
        raise ValueError("prefetch requires a calibrated profile or all applicable explicit quotas. " + _calibration_hint(config))
    identity = build_prefetch_identity(config, device)
    path = profile_directory() / (profile_key(identity) + ".json") if selection == "auto" else Path(selection).expanduser()
    if not path.is_file():
        raise ValueError(f"prefetch profile missing: {path}. " + _calibration_hint(config))
    raw = path.read_bytes()
    try:
        data = validate_profile_data(json.loads(raw), identity)
    except ValueError as exc:
        raise ValueError(f"{exc}. {_calibration_hint(config)}") from exc
    geometry, quotas = data["geometry"], data["quotas"]
    if config.moe_cache_rate is not None or config.moe_cache_mem is not None:
        raise ValueError("prefetch profile fixes expert slots; remove cache rate or memory overrides. " + _calibration_hint(config))
    if config.moe_cache_size not in (0, geometry["total_slots"]):
        raise ValueError("prefetch profile expert slots mismatch. " + _calibration_hint(config))
    layers = config.model_config.num_moe_layers
    if len(quotas["preverify"]) != layers:
        raise ValueError("prefetch profile MoE layer count mismatch. " + _calibration_hint(config))
    if requested["preverify"] is None and requested["preverify_layers"] is None:
        put("speculative_prefetch_layer_quotas", list(quotas["preverify"]))
    if requested["verify"] is None and requested["verify_layers"] is None:
        put("speculative_verify_prefetch_layer_quotas", list(quotas["verify"]))
    if requested["gap_capacity"] is None:
        put("speculative_prefetch_gap_capacity", quotas["gap_capacity"])
    for name in QUOTA_FIELDS:
        if getattr(config, name) is None:
            put(name, 0)
    overridden = any(value is not None for value in requested.values())
    effective = _effective(config, layers)
    zero_profile = not any(effective["preverify"]) and not any(effective["verify"]) and not effective["max_gap_expert_loads"]
    if zero_profile:
        put("speculative_prefetch", False)
    receipt = dict(source="profile_override" if overridden else "profile", requested=requested,
        profile_path=str(path.resolve()), profile_sha256=hashlib.sha256(raw).hexdigest(),
        profile_key=profile_key(identity), profile_status=data["evidence"]["status"],
        identity=identity, geometry=copy.deepcopy(geometry), effective=_effective(config, layers),
        reason="calibrated_zero_profile" if zero_profile else "matched_profile", validated=False,
        inactive_phases=["gpu_full_verify"] if config.speculative_mode == "cpu_async" else [])
    put("_speculative_prefetch_receipt", receipt)
    return receipt


def prefetch_geometry(engine):
    cache = engine.moe_offload_cache
    if cache is None:
        return None
    sources = cache.bank_sources
    payload = [sum(tensors[layer][0].numel() * tensors[layer].element_size()
                   for tensors in sources.values()) for layer in range(cache.num_layers)]
    return dict(total_slots=cache.cache_size, pool_sizes=list(cache.pool_sizes),
        pool_of_layer=list(cache.pool_of_layer), layer_layout=list(cache.layer_layout),
        expert_bytes_by_layer=payload, bank_layouts=[
            [dict(name=name, dtype=str(tensors[layer].dtype), shape=list(tensors[layer].shape[1:]))
             for name, tensors in sources.items()] for layer in range(cache.num_layers)])


def validate_prefetch_profile(engine):
    config = engine.config
    receipt = getattr(config, "_speculative_prefetch_receipt", None)
    if receipt is None:
        return
    if receipt["source"] == "legacy":
        return receipt
    geometry = prefetch_geometry(engine)
    expected = receipt.get("geometry")
    if expected is not None and geometry != expected:
        raise ValueError("prefetch profile expert layout or pool geometry mismatch. " + _calibration_hint(config))
    layers = engine.config.model_config.num_moe_layers
    effective = _effective(config, layers)
    if expected is not None and effective != receipt["effective"]:
        raise ValueError("prefetch profile configuration changed after resolution; rebuild the engine")
    if len(effective["preverify"]) != layers or len(effective["verify"]) != layers:
        raise ValueError("prefetch quotas must have one value per MoE layer")
    receipt.update(effective=effective, actual_geometry=geometry, validated=True)
    receipt["inactive_phases"] = ["gpu_full_verify"] if config.speculative_mode == "cpu_async" else []
    return receipt


def prefetch_receipt(config):
    return copy.deepcopy(getattr(config, "_speculative_prefetch_receipt", None))
