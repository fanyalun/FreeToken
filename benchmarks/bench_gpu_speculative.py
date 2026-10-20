from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
from pathlib import Path
import time

import torch

from bench_three_level_mtp import BenchmarkConfig
from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.qwen_three_level import ThreeLevelCoordinator
from freetoken.mm.config import MultimodalConfig


PROFILE_ACTIVE = [False]


def profile_ranges():
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.speculative_gpu import GPUExpertCache
    def wrap(cls, method, label, capture_only=False):
        original = getattr(cls, method)
        @functools.wraps(original)
        def call(self, *args, **kwargs):
            if not PROFILE_ACTIVE[0] and not capture_only:
                return original(self, *args, **kwargs)
            torch.cuda.nvtx.range_push(label(self, *args, **kwargs))
            try:
                return original(self, *args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
        setattr(cls, method, call)
    wrap(ThreeLevelCoordinator,"step",lambda s,*a,**k:f"OUTER/epoch={s.epoch+1}")
    wrap(ThreeLevelCoordinator,"_forward",lambda s,inputs,*a,**k:f"FORWARD/{s.experts.phase}/loop={s.experts.loop}/rows={len(inputs)}")
    wrap(GPUExpertCache,"route",lambda s,layer,*a,**k:f"GPU_ROUTE/layer={layer.layer_id:02d}",True)
    wrap(GPUExpertCache,"_allocate",lambda s,layer,*a,**k:f"GPU_ADMIT/layer={layer}/prefetch={k.get('prefetch',False)}",True)
    wrap(GPUExpertCache,"_copy",lambda s,layer,*a,**k:f"GPU_COPY/layer={layer:02d}",True)
    wrap(OffloadMoELayer,"_expert_gemm",lambda s,*a,**k:f"GPU_GEMM/layer={s.layer_id:02d}",True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--data", default="/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2)+"\n")
    write("complete.json", dict(status="running"))
    root = Path(__file__).resolve().parents[1]
    prompts, identity = prepare_experiment(args.model, args.data, root, args.samples)
    config = dict(**identity, slots=4907, kv_pages=4300, budget_gib=64, max_loops=10,
        prefetch_layer_experts=4, prefetch_gap_capacity=12, tokens=args.tokens, samples=args.samples,
        repeats=args.repeats, graph=True, warmup="complete passes until target graph widths stabilize; max five",
        timing="complete programmatic request including prefill and resource drain",
        reset="cache metadata before every warmup, timing, and independent audit pass",
        order="rotate sd, host, gpu each repeat", profiling=args.profile)
    write("config.json", config)
    manifest = {}
    for path in sorted((root/"python/freetoken").rglob("*.py")) + [Path(__file__),root/"benchmarks/qwen_mtp_config.py"]:
        data=path.read_bytes();target=out/"source"/path.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        manifest[str(path.relative_to(root))]=hashlib.sha256(data).hexdigest()
    write("source_manifest.json",manifest)
    engine = Engine(BenchmarkConfig(rendezvous_file=str((out/"rendezvous").resolve()),
        model_path=args.model,tp_info=DistributedInfo(0,1),dtype=torch.bfloat16,
        max_running_req=1,attention_backend="triton",moe_strategy="offload",
        moe_cache_auto=False,moe_cache_size=4907,kv_reserve_tokens=4096,
        max_seq_len_override=4096,gpu_memory_budget_bytes=64<<30,num_page_override=4300,
        speculative_mtp_depth=3,speculative_three_level=True,speculative_max_loops=10,
        speculative_max_input_tokens=max(map(len, prompts)), speculative_max_output_tokens=args.tokens,
        speculative_execution_implementation="optimized",
        speculative_expert_scheduler="gpu",speculative_prefetch_layer_experts=4,
        speculative_prefetch_gap_capacity=12,cuda_graph_max_bs=4,cache_type="naive",
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision","audio")))))
    runner=engine.speculative_runner
    gpu=runner.three_level
    object.__setattr__(engine.config,"speculative_expert_scheduler","host")
    host=ThreeLevelCoordinator(runner)
    object.__setattr__(engine.config,"speculative_expert_scheduler","gpu")
    controls={"sd":None,"host":host,"gpu":gpu}
    if args.profile:
        profile_ranges()
    config.update(actual_slots=engine.moe_offload_cache.cache_size,actual_kv_pages=engine.num_pages)
    write("config.json",config)
    results={};warmups=[]
    def reset():
        engine.moe_offload_cache.reset()
        torch.cuda.synchronize(engine.device)
    def pass_requests(mode, label, audit=False):
        runner.three_level=controls[mode]
        if controls[mode] is not None:
            controls[mode].experts.audit=audit
        reset()
        rows=[]
        with (out/(label+".jsonl")).open("x") as file:
            for index,prompt in enumerate(prompts):
                if args.profile and label=="profiled":
                    torch.cuda.nvtx.range_push(f"REQUEST/sample={index}")
                before=time.perf_counter()
                result=runner.generate(prompt,max_tokens=args.tokens)
                wall=time.perf_counter()-before
                if args.profile and label=="profiled":
                    torch.cuda.nvtx.range_pop()
                assert len(result.token_ids)==args.tokens
                row=dict(sample_id=index,**dataclasses.asdict(result),complete_wall_seconds=wall)
                file.write(json.dumps(row)+"\n");file.flush();rows.append(row)
                print(json.dumps(dict(mode=mode,pass_name=label,sample=index,wall=wall,
                    decode=wall-result.ttft_seconds,graphs=len(runner.graphs))),flush=True)
        return rows
    def trajectory(rows):
        return [(r["token_ids"],[(w["drafted"],w["accepted"],w["returned"],w.get("loop_count")) for w in r["rounds"]]) for r in rows]
    try:
        for mode in controls:
            for attempt in range(5):
                before=set(runner.graphs)
                pass_requests(mode,f"warmup_{mode}_{attempt}")
                after=set(runner.graphs)
                warmups.append(dict(mode=mode,attempt=attempt,new_graph_keys=[str(k) for k in after-before]))
                write("warmups.json",warmups)
                if after==before:
                    break
            else:
                raise RuntimeError(f"{mode} graph widths did not stabilize")
        for repeat in range(args.repeats):
            modes=list(controls)
            modes=modes[repeat%3:]+modes[:repeat%3]
            for mode in modes:
                before=set(runner.graphs)
                rows=pass_requests(mode,f"timing_{mode}_{repeat}")
                assert set(runner.graphs)==before,"timed graph capture"
                results.setdefault(mode,[]).extend(rows)
        checks={}
        for mode in controls:
            audit=pass_requests(mode,"audit_"+mode,True)
            expected=results[mode][:args.samples]
            checks[mode]=dict(independent_token_acceptance_parity=trajectory(audit)==trajectory(expected),
                timing_repeat_parity=all(trajectory(results[mode][i:i+args.samples])==trajectory(expected)
                    for i in range(0,len(results[mode]),args.samples)))
        if args.profile:
            runner.three_level=gpu
            gpu.experts.audit=False
            reset()
            torch.cuda.profiler.start()
            PROFILE_ACTIVE[0]=True
            profiled=pass_requests("gpu","profiled")
            PROFILE_ACTIVE[0]=False
            torch.cuda.profiler.stop()
            checks["profiled"]=dict(parity=trajectory(profiled)==trajectory(results["gpu"][:args.samples]))
        metrics={}
        for mode,rows in results.items():
            wall=sum(r["complete_wall_seconds"] for r in rows)
            decode=sum(r["complete_wall_seconds"]-r["ttft_seconds"] for r in rows)
            windows=[w for r in rows for w in r["rounds"]]
            inner=[l for w in windows for l in w.get("loops",[])]
            metrics[mode]=dict(requests=len(rows),returned_tokens=len(rows)*args.tokens,
                wall_seconds=wall,tokens_per_second=len(rows)*args.tokens/wall,
                tpot_ms=1000*decode/(len(rows)*(args.tokens-1)),
                mean_ttft_ms=1000*sum(r["ttft_seconds"] for r in rows)/len(rows),
                mean_loops=sum(w.get("loop_count",0) for w in windows)/len(windows),
                inner_accepted=sum(l["accepted"] for l in inner),inner_drafted=sum(l["drafted"] for l in inner),
                outer_accepted=sum(w["accepted"] for w in windows),outer_drafted=sum(w["drafted"] for w in windows))
        write("metrics.json",dict(methods=metrics,validation=checks,
            gpu_host_token_parity=[a["token_ids"]==b["token_ids"] for a,b in zip(results["gpu"],results["host"])]))
        assert all(v["independent_token_acceptance_parity"] and v["timing_repeat_parity"] for k,v in checks.items() if k!="profiled"),checks
        write("complete.json",dict(status="measurements_completed",validated=True,requests_per_mode=args.samples*args.repeats))
    finally:
        runner.three_level=gpu
        runner.invalidate()
        engine.shutdown()


if __name__=="__main__":
    main()
