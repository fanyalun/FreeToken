"""Export valid GPU prefixes for separate long-context CPU diagnostics."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from bench_cpu_verify_feasibility import BenchmarkConfig
from freetoken.distributed import DistributedInfo
from freetoken.engine.engine import Engine
from freetoken.engine.qwen_async_three_level import QwenAsyncGpuWindow
from freetoken.mm.config import MultimodalConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--prompt-chunk-size", type=int, default=128)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--audit-logits", action="store_true")
    args = parser.parse_args()
    if not args.lengths or min(args.lengths) < 1 or max(args.lengths) > 4096 or args.prompt_chunk_size < 1:
        parser.error("prefix lengths must be 1..4096 and chunk size positive")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.prefix_run / "config.json").read_text())
    prompts = json.loads((args.prefix_run / "prompts.json").read_text())
    capacity = max(*args.lengths, *(len(p) for p in prompts)) if args.include_original else max(args.lengths)
    if capacity > 4096:
        parser.error("original prompts must fit the 4096-token diagnostic capacity")
    frozen = [json.loads(line) for line in (args.prefix_run / "measurements.jsonl").read_text().splitlines()]
    config.update(samples=len(prompts) * (len(args.lengths) + int(args.include_original)), purpose="diagnostic",
        diagnostic="repeat each frozen GSM8K prompt to a fixed context length; arbitrary frozen candidates",
        prompt_chunk_size=args.prompt_chunk_size, source_prefix_run=str(args.prefix_run),
        max_input_tokens=capacity, max_output_tokens=128, output=str(args.output), repeats=1,
        include_original=args.include_original, audit_logits=args.audit_logits,
        command=[sys.executable, *sys.argv], python=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        measurement_scope="valid long-prefix GPU export, not performance timing", formal_timing=False)
    for key in ("cpu_runtime", "gpu_initialization_seconds", "resource_start", "backend", "ungrouped"):
        config.pop(key, None)
    repo = Path(__file__).resolve().parents[1]
    paths = list((repo / "python/freetoken").rglob("*.py"))
    paths += list((repo / "python/freetoken/kernel/csrc/cpu_moe").glob("*"))
    paths += list((repo / "python/freetoken/kernel").glob("*.so"))
    paths += [Path(__file__).resolve(), repo / "setup.py"]
    config["source_root"] = str(repo)
    config["source_sha256"] = {}
    for path in paths:
        if path.is_file():
            relative = str(path.relative_to(repo))
            config["source_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            copied = args.output / "source" / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(path.read_bytes())
    (args.output / "complete.json").write_text(json.dumps(dict(validity="incomplete")))
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(1)
    engine = Engine(BenchmarkConfig(rendezvous_file=str(args.output / "rendezvous"),
        model_path=config["model"], tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=1, attention_backend="triton", moe_strategy="offload",
        moe_cache_size=4907, moe_cache_auto=False, num_page_override=4300,
        max_seq_len_override=capacity + 128, gpu_memory_budget_bytes=64 << 30,
        speculative_max_input_tokens=capacity, speculative_max_output_tokens=128,
        speculative_mtp_depth=3, cuda_graph_max_bs=0, cache_type="naive",
        max_extend_tokens=args.prompt_chunk_size,
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio")))))
    window = QwenAsyncGpuWindow(engine.speculative_runner)
    identities = []
    try:
        for original_sample, prompt in enumerate(prompts):
            for requested in ([None] if args.include_original else []) + args.lengths:
                length = len(prompt) if requested is None else requested
                tokens = prompt if requested is None else (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length]
                checkpoint = window.initialize(tokens, chunk_size=args.prompt_chunk_size)
                index = len(identities)
                path = args.output / f"gpu_prefix_sample{index}.pt"
                torch.save(asdict(checkpoint), path)
                identities.append(dict(sample_id=index, original_sample=original_sample, length=length,
                    input_tokens=tokens, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    original_prompt=requested is None, prefill_wall_seconds=window.prefill_wall_seconds,
                    prefill_gpu_seconds=window.prefill_seconds))
                if args.audit_logits:
                    torch.save(window.audit_gpu_anchor(checkpoint.anchor), args.output / f"gpu_logits_sample{index}.pt")
                with (args.output / "measurements.jsonl").open("a") as stream:
                    for row in frozen:
                        if row["sample_id"] == original_sample and row["repeat"] == 0:
                            stream.write(json.dumps({**row, "sample_id": index}) + "\n")
                window.validate_budget()
                window.close_request()
                print(json.dumps(dict(sample=index, length=length, anchor=checkpoint.anchor)), flush=True)
        (args.output / "prefix_identities.json").write_text(json.dumps(identities, indent=2))
        (args.output / "complete.json").write_text(json.dumps(dict(validity="valid", prefix_count=len(identities))))
    finally:
        window.close_request()
        engine.shutdown()


if __name__ == "__main__":
    main()
