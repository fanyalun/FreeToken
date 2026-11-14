"""Capture real prefill and full-Graph decode, with one layer attributed by NVTX."""

import argparse
import json
import os
from pathlib import Path
import time


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def canonical_mm_input(result, image_pad):
    ids = result.input_ids.tolist()
    canonical = list(ids)
    visual = 0
    for item in result.mm_items:
        for start, end in item.offsets:
            if ids[start:end] != [item.pad_value] * (end-start):
                raise ValueError("visual span does not contain its declared content pad ID")
            canonical[start:end] = [image_pad] * (end-start)
            visual += end-start
    return canonical, visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=("offload", "tablemoe"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--cold-decode", action="store_true")
    parser.add_argument("--cached-kernels-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "this experiment is restricted to GPU1"

    import torch
    from freetoken.core import SamplingParams
    from freetoken.engine.engine import Engine
    from freetoken.llm import LLM
    from freetoken.mm.config import MultimodalConfig
    from qwen3vl_profile_hooks import fp16_native_banks, install

    if args.cached_kernels_only:
        from triton.runtime.jit import JITFunction
        import subprocess

        def blocked_compile(self, *a, **kw):
            save(args.output / "compile_required.json", dict(kernel=self.__name__,
                 reason="new JIT compilation requires confirmation; no dependency is installed"))
            raise RuntimeError(f"JIT cache miss requires confirmation: {self.__name__}")

        JITFunction._do_compile = blocked_compile
        original_popen = subprocess.Popen

        def cached_only_popen(command, *a, **kw):
            words = command if isinstance(command, (list, tuple)) else [command]
            text = " ".join(str(x) for x in words)
            if "--version" not in words and any(tool in text for tool in ("nvcc", "ninja", "build_ext")):
                save(args.output / "compile_required.json", dict(command=text,
                     reason="compiler invocation blocked pending confirmation"))
                raise RuntimeError(f"compiler invocation requires confirmation: {text[:180]}")
            return original_popen(command, *a, **kw)

        subprocess.Popen = cached_only_popen
    else:
        from triton.runtime.jit import JITFunction
        original_compile = JITFunction._do_compile

        def record_compile(self, *a, **kw):
            started = time.monotonic()
            try:
                return original_compile(self, *a, **kw)
            finally:
                with (args.output / "jit_compiles.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(kernel=self.__name__,elapsed_s=time.monotonic()-started)) + "\n")

        JITFunction._do_compile = record_compile
    torch.manual_seed(42)
    torch.set_num_threads(config["cpu_threads"])
    fp16_native_banks()
    if args.profile:
        install(config["layer"])
    state = dict(active=False, request="", prefill_chunks=0, decode_step=0, capture=None)
    original_forward = Engine.forward_batch
    observed = []

    def stop_capture():
        with torch.cuda.nvtx.range("PROFILE/BoundarySync"):
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        state["capture"] = None

    def forward(self, batch, *a, **kw):
        if not state["active"]:
            return original_forward(self, batch, *a, **kw)
        if batch.is_prefill:
            chunk = state["prefill_chunks"]
            state["prefill_chunks"] += 1
            label = f"{state['request']}/PREFILL/chunk{chunk:02d}"
            if args.profile and not args.decode_only and state["capture"] is None:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                state["capture"] = "prefill"
        else:
            state["decode_step"] += 1
            step = state["decode_step"]
            if args.profile and state["capture"] == "prefill":
                stop_capture()
            if args.profile and state["request"] == "MEASURE" and step == config["decode_step"] - 1:
                torch.cuda.cudart().cudaProfilerStart()
                state["capture"] = "decode"
            if args.cold_decode and state["request"] == "MEASURE" and step == config["decode_step"]:
                with torch.cuda.nvtx.range("DIAGNOSTIC/ResetCachesBeforeTargetForward"):
                    torch.cuda.synchronize()
                    cache.reset()
                    if args.backend == "tablemoe":
                        for online in self.tablemoe_runtime.fixed_online.values():
                            online["size"].zero_()
                            online["pointer"].zero_()
                    torch.cuda.synchronize()
                    assert bool((cache.slot_for_id == -1).all().item())
                save(args.output / "cold_decode_reset.json", dict(expert_cache_cleared=True,
                     online_cache_cleared=args.backend == "tablemoe", before_decode_step=step,
                     capacity_unchanged=cache.cache_size, graph_recaptured=False,
                     expected_layer_expert_misses=8 if args.backend == "offload" else 4,
                     logical_expert_payload_bytes=(8 if args.backend == "offload" else 4)*9437184,
                     hardware_pcie_bytes_measured=False))
            label = f"{state['request']}/DECODE/step{step:03d}"
        use_graph = self.graph_runner.can_use_cuda_graph(batch)
        observed.append(dict(label=label, prefill=batch.is_prefill, input_tokens=batch.input_ids.numel(),
                             batch_size=batch.size, padded_batch_size=batch.padded_size, cuda_graph=use_graph))
        with torch.cuda.nvtx.range(label):
            result = original_forward(self, batch, *a, **kw)
        if args.profile and state["capture"] == "decode" and state["decode_step"] == config["decode_step"]:
            stop_capture()
        return result

    Engine.forward_batch = forward
    options = dict(model_path=config["model"], dtype=torch.float16, moe_strategy=args.backend,
        moe_cache_size=config["expert_slots"], max_running_req=1, cuda_graph_max_bs=1,
        num_page_override=config["num_pages"], page_size=1, max_seq_len_override=config["max_seq_len"],
        cache_type="naive", max_extend_tokens=config["prefill_chunk_tokens"], attention_backend="triton",
        mm=MultimodalConfig(encoder_weights="gpu", processor_kwargs=config["processor"]),
        moe_collect_stats=False)
    if args.backend == "tablemoe":
        options.update(tablemoe_expert_mode="offload", tablemoe_implementation="optimized",
            tablemoe_pca_dir=config["pca_dir"], tablemoe_cache_dir=config["cache_dir"],
            tablemoe_convert_bf16_weights=True, tablemoe_prefill_keep_rate=.6, tablemoe_decode_keep_k=4,
            tablemoe_online_cache_size=128, tablemoe_renormalize_weights=False,
            tablemoe_prefill_experts_per_group=16, tablemoe_offline_cosine_threshold_vision=.6,
            tablemoe_offline_cosine_threshold_text=.7, tablemoe_decode_online_cosine_threshold=.7,
            tablemoe_decode_offline_cosine_threshold=.7)
    save(args.output / "launch.json", dict(pid=os.getpid(), physical_gpu=1, backend=args.backend,
        profile=args.profile, cold_decode=args.cold_decode, decode_only=args.decode_only,
        options={k:repr(v) for k,v in options.items()}, cached_kernels_only=args.cached_kernels_only))
    llm = LLM(**options)
    cache = llm.engine.moe_offload_cache if args.backend == "offload" else llm.engine.tablemoe_runtime.expert_cache
    assert cache.cache_size == 3072 and cache.num_layers == 48 and cache.num_experts == 128
    assert all(t.dtype == torch.float16 for t in cache.bank_caches.values())
    assert set(llm.engine.graph_runner.graph_map) == {1}
    inputs = json.loads(Path(config["input"]).read_text())
    reference, row = inputs["reference"], inputs["sample"]
    images = [Path(item["path"]).read_bytes() for item in row["images"]]
    original_process = llm._mm_processor.apply
    input_audits = []
    last_input = []

    def checked_process(*a, **kw):
        result = original_process(*a, **kw)
        canonical, visual = canonical_mm_input(result, llm.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        assert canonical == reference["expanded_input_ids"]
        last_input[:] = result.input_ids.tolist()
        input_audits.append(dict(request=state["request"],tokens=len(canonical),image_items=len(result.mm_items),
              visual_tokens=visual,text_and_visual_positions_match=True,
              content_pad_ids=[item.pad_value for item in result.mm_items]))
        save(args.output / "input_validation.json",input_audits)
        return result

    llm._mm_processor.apply = checked_process
    save(args.output / "resolved_runtime.json", dict(config=repr(llm.engine.config),
         bank_dtypes={k:str(v.dtype) for k,v in cache.bank_caches.items()}, gpu=1,
         layer=config["layer"], expert_slots=cache.cache_size, graph_bs=[1],
         native_fp16_layout_hook=True, full_prompt_expected_tokens=reference["prompt_tokens"],
         offline_files=len(llm.engine.tablemoe_runtime.offline_paths) if args.backend=="tablemoe" else 0))
    results = []
    try:
        for name, length in (("WARMUP",16),("MEASURE",config["decode_step"]+1)):
            state.update(active=True,request=name,prefill_chunks=0,decode_step=0,capture=None)
            started=time.monotonic()
            output=llm.generate([reference["chat_template"]], SamplingParams(max_tokens=length,temperature=0,ignore_eos=True),images=[images])[0]
            torch.cuda.synchronize()
            elapsed=time.monotonic()-started
            assert len(output["token_ids"])==length
            assert llm.status_map[0].input_ids==last_input
            assert state["decode_step"]==length-1
            results.append(dict(request=name,output_tokens=length,token_ids=output["token_ids"],text=output["text"],
                 elapsed_s=elapsed,prefill_chunks=state["prefill_chunks"],decode_forwards=state["decode_step"]))
            save(args.output / "requests.json",results)
        assert all(x["cuda_graph"] for x in observed if not x["prefill"])
        save(args.output / "observed_batches.json",observed)
        save(args.output / "complete.json",dict(complete=True,physical_gpu=1,backend=args.backend,
             profile=args.profile,layer=config["layer"],measured_decode_step=config["decode_step"],
             full_prompt_identical=True,cold_decode=args.cold_decode,formal_performance_eligible=False))
    finally:
        state["active"]=False
        llm.engine.shutdown()


if __name__=="__main__":
    main()
