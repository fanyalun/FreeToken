from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint(root):
    root = Path(root)
    suffixes = {".py", ".cpp", ".cc", ".c", ".cu", ".cuh", ".h", ".hpp", ".so", ".json"}
    paths = {p for p in (root / "python/freetoken").rglob("*") if p.is_file() and p.suffix in suffixes}
    paths.update(root / "benchmarks" / name for name in (
        "bench_qwen_mtp.py", "audit_qwen_mtp.py", "run_qwen_mtp_matrix.py", "qwen_mtp_config.py",
    ))
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def prepare_experiment(model, data, root, samples=16):
    import torch
    from transformers import AutoTokenizer

    model, data, root = Path(model).resolve(), Path(data).resolve(), Path(root).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    rows = [json.loads(line) for line in data.read_text().splitlines()][:samples]
    if len(rows) != samples:
        raise ValueError("dataset has fewer requested samples")
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": row["question"]}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False,
    ) for row in rows]
    index = model / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    weights = {}
    for name in shards:
        stat = (model / name).stat()
        weights[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    metadata = [p for p in model.iterdir() if p.is_file() and p.suffix in (".json", ".jinja", ".model", ".txt")]
    metadata.extend((model / "chat_templates").glob("*.jinja"))
    config = {
        "fingerprint_version": 1,
        "model": str(model), "data": str(data), "source_root": str(root),
        "source_sha256": source_fingerprint(root),
        "commit": (json.loads((root / "source_identity.json").read_text())["commit"]
                   if (root / "source_identity.json").exists() else
                   subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()),
        "model_config_sha256": digest(model / "config.json"),
        "weight_index_sha256": digest(index),
        "checkpoint_metadata_sha256": {str(p.relative_to(model)): digest(p) for p in sorted(metadata)},
        # Avoid rereading all weight shards before each cell. Metadata signatures
        # reject changed local files; they are not cryptographic weight hashes.
        "weight_shard_stat": weights,
        "data_sha256": digest(data),
        "prompt_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "sample_ids": list(range(samples)),
        "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "environment": {k: v for k, v in sorted(os.environ.items()) if (k.startswith("FREETOKEN_") or k in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        )) and not any(secret in k for secret in ("API_KEY", "PASSWORD", "SECRET", "AUTH_TOKEN", "ACCESS_TOKEN"))},
    }
    return prompts, config


def benchmark_config(args, experiment):
    return {
        **{key: value for key, value in vars(args).items() if key != "moe_slots"}, **experiment,
        "requested_moe_slots": getattr(args, "moe_slots", None),
        "max_active_requests": 1, "target_dtype": "bfloat16", "draft_dtype": "bfloat16",
        "depth": 3 if args.mode == "mtp" else 0,
        "ignore_eos": True, "temperature": 0, "prompt_chunk_size": getattr(args, "prompt_chunk_size", None),
        "max_input_tokens": getattr(args, "max_input_tokens", 3840),
        "max_output_tokens": args.tokens,
        "hybrid_window_policy": getattr(args, "hybrid_window_policy", "balanced"),
        "hybrid_cpu_backend": getattr(args, "hybrid_cpu_backend", "grouped"),
        "hybrid_profile": getattr(args, "hybrid_profile", None),
        "hybrid_profile_sha256": digest(args.hybrid_profile) if getattr(args, "hybrid_profile", None) else None,
        "ar_implementation": "engine_graph_runner_and_sampler" if args.mode == "ar" else None,
        "round_metrics_version": 2,
        "timing_scope": "complete programmatic single-request generation through resource drain",
        "ttft_scope": "generation start through first target token materialization",
        "warmups": 5,
        "expert_cache_start": "cold metadata after kernel warmups, before each complete 16-request pass",
    }


def validate_cell_config(directory, expected):
    path = Path(directory) / "config.json"
    try:
        actual = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot reuse {directory}: missing or invalid config.json; use a new --output directory") from exc
    if not isinstance(actual, dict):
        raise ValueError(f"cannot reuse {directory}: invalid config.json; use a new --output directory")
    # Output paths and Git metadata do not alter execution; content hashes do.
    mismatches = [key for key, value in expected.items()
                  if key not in ("output", "source_root", "commit") and (key not in actual or actual[key] != value)]
    if mismatches:
        raise ValueError(f"cannot reuse {directory}: experiment configuration mismatch: {', '.join(mismatches)}; "
                         "use a new --output directory")
    return actual
