"""Check candidate survival and fixed-token rolling independently of timing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from bench_speculative_modes import (BenchmarkConfig, checkpoint_identity, memory_report,
                                    prompt_inputs, repository_identity, resolve_model, source_identity)
from cpu_async_algorithm_common import AlgorithmSession
from cpu_async_algorithm_pairs import run_pairs
from cpu_async_algorithm_roll import run_rolling
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.cpu_async import CpuAsyncRuntime
from freetoken.mm.config import MultimodalConfig
from freetoken.utils.hf import load_tokenizer


def write(path, value):
    temporary = Path(str(path) + ".new")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def wait_for_gpu(output, gpu, idle_seconds, timeout):
    started, quiet = time.monotonic(), None
    while True:
        query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True)
        rows = [[int(value.strip()) for value in line.split(",")] for line in query.splitlines()]
        target = next(row for row in rows if row[0] == gpu)
        quiet = quiet or time.monotonic() if target[1] == 0 and target[2] == 0 else None
        write(output / "gpu_idle_gate.json", dict(time=time.time(), physical_gpu=gpu,
            readings=rows, required_idle_seconds=idle_seconds, waiting=quiet is None))
        if quiet is not None and time.monotonic() - quiet >= idle_seconds:
            return
        if timeout and time.monotonic() - started > timeout:
            raise TimeoutError("GPU did not become idle within the requested timeout")
        time.sleep(1)


def summarize(rows):
    rolling = [row["rolling"] for row in rows]
    handoffs = [handoff for row in rolling for handoff in row["handoffs"]]
    eligible = [row for row in rows if row["pairs"] and row["pairs"]["experiment2"].get("eligible")]
    return dict(samples=len(rows), algorithm_only=True, performance="not_measured",
        paired_experiments_run=sum(row["pairs"] is not None for row in rows),
        paired_eligible_samples=len(eligible),
        paired_skipped_samples=sum(row["pairs"] is not None for row in rows) - len(eligible),
        cpu_serial_output_exact=all(row["cpu_serial_output_exact"] for row in rolling),
        returned_tokens=sum(row["returned_tokens"] for row in rolling),
        handoffs=len(handoffs), nonterminal_handoffs=sum(not row["terminal"] for row in handoffs),
        retained_handoffs=sum(row["retained_handoffs"] for row in rolling),
        regenerated_handoffs=sum(row["regenerated_handoffs"] for row in rolling),
        longest_success_chains=[row["longest_success_chain"] for row in rolling],
        discarded_ahead_candidates=sum(row["discarded_ahead_candidates"] for row in rolling),
        cpu_accepted_old_ahead_tokens=sum(row["cpu_accepted_old_ahead_tokens"] for row in rolling),
        slide_cell_attempts=sum(row["slide_cells"]["attempts"] for row in rolling),
        zero_slide_attempts=sum(row["slide_cells"]["zero_slide_attempts"] for row in rolling),
        at_least_two_slide_attempts=sum(row["slide_cells"]["at_least_two"] for row in rolling),
        output_censored_chains=sum(row["slide_cells"]["output_censored"] for row in rolling),
        initial_slide_cells=[row["slide_cells"]["episodes"][0] for row in rolling],
        slide_cell_histogram={key: sum(row["slide_cells"]["histogram_observed"].get(key, 0) for row in rolling)
            for key in sorted({key for row in rolling for key in row["slide_cells"]["histogram_observed"]}, key=int)},
        handoff_reasons={reason: sum(row["reason"] == reason for row in handoffs)
                         for reason in sorted({row["reason"] for row in handoffs})},
        cold_prefill_checkpoint_equal=all(row["resets_equal"] for row in rows),
        cpu_state_audits=sum(row["cpu_state_audits"] for row in rows))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--idle-seconds", type=float, default=3)
    parser.add_argument("--idle-timeout", type=float, default=0)
    parser.add_argument("--top-h", type=int, choices=(4, 6), default=4)
    parser.add_argument("--recovery", choices=("fixed_replay", "recheck"), default="fixed_replay")
    parser.add_argument("--rolling-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.tokens < 64 or args.idle_seconds < 0 or args.idle_timeout < 0:
        parser.error("positive samples, at least 64 output tokens, and nonnegative idle limits required")
    out = args.output.resolve()
    if out == Path("/tmp") or Path("/tmp") in out.parents:
        parser.error("use a persistent output directory outside /tmp")
    out.mkdir(parents=True, exist_ok=False)
    (out / "tmp").mkdir()
    os.environ["TMPDIR"] = str(out / "tmp")
    import tempfile
    tempfile.tempdir = str(out / "tmp")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    write(out / "complete.json", dict(status="incomplete", valid=False, algorithm_only=True))
    engine = session = None
    root = Path(__file__).resolve().parents[1]
    source = source_identity(root)
    write(out / "source_manifest.json", source)
    try:
        wait_for_gpu(out, args.gpu, args.idle_seconds, args.idle_timeout)
        model = resolve_model(args.model)
        tokenizer = load_tokenizer(model)
        prompt_args = argparse.Namespace(data=[args.data], samples=args.samples, input_tokens=None,
                                        max_input_tokens=args.max_input_tokens)
        prompts = prompt_inputs(prompt_args, tokenizer)
        identities = dict(target=checkpoint_identity(model),
                          draft=checkpoint_identity(args.draft_model) if args.draft_model else "native_mtp")
        write(out / "protocol.json", dict(arguments=vars(args), prompts=prompts, checkpoints=identities,
            sampling="greedy", ignore_eos=True, loops_per_block=4, top_h=args.top_h, prefetch=False,
            recovery=args.recovery, rolling_only=args.rolling_only,
            recheck_policy="first mismatch correction, no refill, cap at original B length; partial B is not a full slide",
            mode="serial_algorithm_simulation", performance="not_measured", cpu_backend="optimized",
            cpu_profile="default_v1_or_gguf", reference="serial one-row CPU after identical GPU prefill",
            data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
            reset="cold expert cache, zero canonical/private recurrent state, repeated identical prefill",
            memory_ratio=.9, gpu_memory_budget_bytes=None, **repository_identity(root)))
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        config = BenchmarkConfig(model_path=model, rendezvous_file=str(out / "rendezvous"),
            tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, speculative_mode="cpu_async",
            speculative_draft_model=args.draft_model, speculative_depth=3, speculative_graph="eager",
            speculative_preverify_top_h=args.top_h, speculative_prefetch=False,
            speculative_prefetch_layer_experts=0, speculative_prefetch_gap_capacity=0,
            speculative_verify_prefetch_layer_experts=0, cpu_async_loops=4, cpu_async_recovery=args.recovery,
            moe_strategy="offload", memory_ratio=.9, gpu_memory_budget_bytes=None,
            speculative_max_input_tokens=args.max_input_tokens, speculative_max_output_tokens=args.tokens,
            mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))))
        engine = Engine(config)
        runtime = CpuAsyncRuntime(engine)
        engine.cpu_async_runtime = runtime
        runtime._prepare_worker()
        if runtime.window.policy.top_k != args.top_h:
            raise RuntimeError("resolved Pre-Verify expert count differs from the requested top-h")
        write(out / "resolved_preverify_policy.json", asdict(runtime.window.policy))
        write(out / "resolved_config.json", asdict(config))
        write(out / "memory_plan.json", memory_report(engine))
        write(out / "cpu_target_runtime.json", runtime.worker.metadata)
        rows = []
        for prompt in prompts:
            sample_out = out / f"sample_{prompt['sample']:02d}"
            sample_out.mkdir()
            write(out / "state.json", dict(status="cpu_serial_reference", sample=prompt["sample"],
                                            completed_samples=len(rows)))
            session = AlgorithmSession(engine, runtime.window, runtime.worker, prompt["input_ids"],
                                       output_tokens=args.tokens, output=sample_out)
            session.build_reference()
            write(sample_out / "reference_tokens.json", session.reference_tokens)
            pairs = None
            if not args.rolling_only:
                write(out / "state.json", dict(status="paired_experiments", sample=prompt["sample"],
                                                completed_samples=len(rows)))
                pairs = run_pairs(session)
                write(sample_out / "pairs.json", pairs)
            write(out / "state.json", dict(status="rolling", sample=prompt["sample"],
                                            completed_samples=len(rows)))
            rolling = run_rolling(session, recovery=args.recovery)
            write(sample_out / "rolling.json", rolling)
            session.close()
            row = dict(sample=prompt["sample"], input_tokens=len(prompt["input_ids"]), pairs=pairs,
                       rolling=rolling, resets_equal=True, resets=session.resets,
                       cpu_state_audits=session.audit_count)
            rows.append(row)
            write(out / "summary.json", summarize(rows))
            write(sample_out / "complete.json", dict(status="complete", valid=True,
                cpu_serial_output_exact=rolling["cpu_serial_output_exact"], returned_tokens=rolling["returned_tokens"]))
            print(json.dumps(dict(sample=prompt["sample"], completed_samples=len(rows),
                retained=rolling["retained_handoffs"], longest_chain=rolling["longest_success_chain"])), flush=True)
            session = None
        if source != source_identity(root):
            raise RuntimeError("source changed during the algorithm experiment")
        if identities != dict(target=checkpoint_identity(model),
                draft=checkpoint_identity(args.draft_model) if args.draft_model else "native_mtp"):
            raise RuntimeError("checkpoint changed during the algorithm experiment")
        engine.shutdown()
        engine = None
        write(out / "state.json", dict(status="complete", completed_samples=len(rows)))
        write(out / "complete.json", dict(status="complete", valid=True, algorithm_only=True,
            samples=len(rows), returned_tokens=args.samples * args.tokens, source_unchanged=True,
            cpu_serial_output_exact=True, cpu_state_audits=sum(row["cpu_state_audits"] for row in rows),
            shutdown_complete=True, performance="not_measured", strict_gpu_ar_parity="not_established"))
    except BaseException as exc:
        write(out / "complete.json", dict(status="failed", valid=False, algorithm_only=True,
            error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            if engine is not None:
                engine.shutdown()


if __name__ == "__main__":
    main()
