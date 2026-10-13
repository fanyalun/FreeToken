"""Whole-layer 8-route vs 4-route cost on saved real decode inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
os.environ["PATH"] = str(REPO / ".venv/bin") + ":/usr/local/cuda-13.0/bin:" + os.environ["PATH"]

import torch
from safetensors import safe_open

from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.moe.cpu_executor import CpuMoeExecutor
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.tablemoe_optimized import TableMoeOptimizedRuntime


def runtime(args, mode, keep_k, gate, down):
    rt = TableMoeOptimizedRuntime.__new__(TableMoeOptimizedRuntime)
    rt.hidden, rt.num_layers, rt.num_experts = 2048, 1, 128
    rt.device = torch.device("cuda", args.gpu)
    rt.expert_mode = mode
    rt.config = SimpleNamespace(
        tablemoe_pca_dir=args.pca, tablemoe_cache_dir=args.cache, tablemoe_online_cache_size=64,
        tablemoe_prefill_keep_rate=1 if keep_k == 8 else .6, tablemoe_decode_keep_k=keep_k,
        tablemoe_offline_cosine_threshold_vision=.6, tablemoe_offline_cosine_threshold_text=.7,
        tablemoe_stats=False, max_running_req=4, max_extend_tokens=128, cuda_graph_bs=[1, 2, 4],
        model_config=SimpleNamespace(num_experts_per_tok=8, moe_intermediate_size=768),
    )
    rt.pca, rt.offline_paths, rt.online, rt.counters = {}, {}, defaultdict(dict), defaultdict(int)
    rt._load_assets()
    rt._prepare_gpu_assets()
    rt.expert_cache = OffloadMoeCache(1, 128, 3072, rt.device, quant_format="fp16")
    rt.expert_cache.set_bank_sources({"gate_up": [gate], "down": [down]})
    rt.hybrid_max_fetch = 1
    rt.expert_bytes = 3 * 2048 * 768 * 2
    rt.executor = None
    if mode == "hybrid":
        banks = SimpleNamespace(num_layers=1, num_experts=128,
                                bank_sources={"gate_up": [gate], "down": [down]})
        rt.executor = CpuMoeExecutor(banks, top_k=8, activation="silu", apply_router_weight_on_input=False,
                                     num_threads=16, max_tokens=128, device=rt.device, fmt="fp16_routes")
    rt._gpu_counts = torch.zeros(3, device=rt.device, dtype=torch.int64)
    rt._stats_event = torch.cuda.Event()
    rt._init_optimized_state()
    return rt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--model", type=Path, default=Path("/data/fanya/models/Qwen3-VL-30B-A3B-Instruct"))
    parser.add_argument("--pca", default="/home/fanya/tablemoe/offline_table/qwen_fp16/clustering_results/MMBench_DEV_EN_V11_LayerPCA_256")
    parser.add_argument("--cache", default="/home/fanya/tablemoe/offline_table/qwen_fp16/offline_table/MMBench_DEV_EN_V11_LayerPCA_256")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.gpu)
    traces = sorted(args.inputs.glob("layer_input_*.pt"))
    if len(traces) < 4:
        raise ValueError("at least four audited real decode inputs are required")
    inputs = [torch.load(path, weights_only=True) for path in traces[:4]]
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    banks = {}
    for role, leaf in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
        key = f"model.language_model.layers.0.mlp.experts.{leaf}"
        with safe_open(args.model / index[key], framework="pt", device="cpu") as source:
            tensor = source.get_tensor(key).transpose(-1, -2).contiguous()
            banks[role] = alloc_pinned_tensor(*tensor.shape, dtype=torch.float16)
            banks[role].copy_(tensor)
    manifest = dict(inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in traces[:4]},
                    model=str(args.model), layer=0, gpu=args.gpu, model_dtype="BF16 -> FP16",
                    device=torch.cuda.get_device_name(args.gpu), expert_slots=3072,
                    measurement="complete MoE layer; state/cache restore outside measurement; CUDA events plus synchronized wall time")
    manifest["source"] = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (REPO / "python/freetoken").rglob("*.py")}
    manifest["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["compiled_cpu_moe_sha256"] = hashlib.sha256(next((REPO / "python/freetoken/kernel").glob("_cpu_moe*.so")).read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    records = []
    for mode in ("offload", "hybrid"):
        for keep_k in (8, 4):
            rt = runtime(args, mode, keep_k, banks["gate_up"], banks["down"])
            for bs in (1, 2, 4):
                x = torch.cat([t["x"] for t in inputs[:bs]]).to(rt.device)
                ids = torch.cat([t["ids"] for t in inputs[:bs]]).to(rt.device)
                weights = torch.cat([t["weights"] for t in inputs[:bs]]).to(rt.device)
                batch = SimpleNamespace(is_prefill=False, tablemoe_slots=torch.arange(bs, device=rt.device, dtype=torch.int32))
                saved = {name: torch.stack([t["online"][name] for t in inputs[:bs]]).to(rt.device)
                         for name in ("keys", "ids", "values", "size", "pointer")}
                rt.forward(0, x, weights, ids, batch)
                stream = torch.cuda.Stream(device=rt.device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    rt.forward(0, x, weights, ids, batch)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    rt.forward(0, x, weights, ids, batch)
                for cache_state in ("all_hit", "cold_miss"):
                    for execution in ("eager", "graph"):
                        event_ms, wall_ms = [], []
                        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        for _ in range(args.iterations):
                            if cache_state == "cold_miss":
                                rt.expert_cache.reset()
                            else:
                                active = ids.int().clone()
                                rt.expert_cache.hybrid_max_fetch = 128
                                rt.expert_cache.ensure_experts_hybrid(0, active)
                                rt.expert_cache.copy_missing()
                            cache = rt.fixed_online[0]
                            for name, value in saved.items():
                                cache[name][:bs].copy_(value)
                            torch.cuda.synchronize()
                            begin = time.perf_counter()
                            start_event.record()
                            if execution == "graph":
                                graph.replay()
                            else:
                                rt.forward(0, x, weights, ids, batch)
                            end_event.record()
                            end_event.synchronize()
                            wall_ms.append((time.perf_counter() - begin) * 1000)
                            event_ms.append(start_event.elapsed_time(end_event))
                        record = dict(mode=mode, keep_k=keep_k, bs=bs, cache=cache_state, execution=execution,
                                      iterations=args.iterations, mean_cuda_ms=statistics.mean(event_ms),
                                      median_cuda_ms=statistics.median(event_ms), mean_wall_ms=statistics.mean(wall_ms))
                        records.append(record)
                        print(json.dumps(record), flush=True)
                del graph
            rt.close()
            del rt
            torch.cuda.empty_cache()
    with (args.output / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "complete.json").write_text(json.dumps(dict(complete=True, cells=len(records)), indent=2))


if __name__ == "__main__":
    main()
