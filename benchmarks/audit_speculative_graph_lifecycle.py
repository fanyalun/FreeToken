from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading

import torch

from bench_three_level_mtp import BenchmarkConfig
from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.qwen_mtp import QwenMTPRunner
from freetoken.layers.moe import OffloadMoELayer
from freetoken.mm.config import MultimodalConfig
from freetoken.moe.speculative_cache import NoEvictableSlot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-implementation", choices=("coarse", "full"), default="coarse")
    parser.add_argument("--budget-gib", type=int, default=64)
    parser.add_argument("--slots", type=int, default=4907)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    write("complete.json", dict(status="incomplete", validated=False))
    root = Path(__file__).resolve().parents[1]
    prompts, identity = prepare_experiment("/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        "/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl", root, 1)
    identity["source_sha256"][str(Path(__file__).relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write("config.json", dict(**identity, purpose="diagnostic", tokens=32, depth=3,
        slots=args.slots, kv_pages=4300, budget_gib=args.budget_gib, graph_implementation=args.graph_implementation,
        layer_quota=2, gap_capacity=12))
    for name in identity["source_sha256"]:
        if name.endswith(".py"):
            target = out / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    engine = Engine(BenchmarkConfig(rendezvous_file=str(out / "rendezvous"), model_path=identity["model"],
        tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, max_running_req=1,
        attention_backend="triton", moe_strategy="offload", moe_cache_auto=False, moe_cache_size=args.slots,
        num_page_override=4300, max_seq_len_override=4096, gpu_memory_budget_bytes=args.budget_gib << 30,
        speculative_mtp_depth=3, speculative_three_level=True, speculative_max_loops=6,
        speculative_max_input_tokens=max(map(len, prompts)), speculative_max_output_tokens=32,
        speculative_expert_scheduler="gpu", speculative_execution_implementation="optimized",
        speculative_graph_implementation=args.graph_implementation, speculative_prefetch_layer_experts=2,
        speculative_prefetch_gap_capacity=12, cuda_graph_max_bs=4, cache_type="naive",
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio")))))
    runner, cache = engine.speculative_runner, engine.moe_offload_cache
    graphs, control = runner.coarse_graphs, runner.three_level.experts
    def clean():
        torch.cuda.synchronize(engine.device)
        assert not runner.active and not runner._request_lock.locked()
        assert not control.active and cache.speculative_controller is None
        assert not control.pending.any() and not control.refs.any()
        assert not engine.page_table[0].any()
    def generate(cancel=None):
        cache.reset()
        return engine.generate_speculative(prompts[0], max_tokens=32, cancel=cancel).token_ids
    expected = generate()
    clean()
    checks = {}
    cancel = threading.Event()
    original_tail = graphs.tail
    def cancelling_tail(inner, continuing):
        result = original_tail(inner, continuing)
        if inner and continuing:
            cancel.set()
        return result
    graphs.tail = cancelling_tail
    try:
        cancelled = generate(cancel)
        assert cancel.is_set() and len(cancelled) < 32
    finally:
        graphs.tail = original_tail
    clean()
    assert generate() == expected
    clean()
    checks["cancel_with_gap_in_flight"] = True
    snapshots = []
    original_body = graphs.body
    def failing_body(*a, **kw):
        if not kw["inner"]:
            pool = engine.linear_state_pool
            snapshots.extend((pool.recurrent_states[:, 1:2].clone(), pool.conv_states[:, 1:2].clone()))
            control.error.fill_(1)
        return original_body(*a, **kw)
    graphs.body = failing_body
    try:
        try:
            generate()
        except NoEvictableSlot:
            pass
        else:
            raise AssertionError("injected admission failure was ignored")
    finally:
        graphs.body = original_body
    clean()
    pool = engine.linear_state_pool
    torch.testing.assert_close(pool.recurrent_states[:, 1:2], snapshots[0], atol=0, rtol=0)
    torch.testing.assert_close(pool.conv_states[:, 1:2], snapshots[1], atol=0, rtol=0)
    assert generate() == expected
    clean()
    checks["invalid_verify_never_commits_canonical_state"] = True
    original_gemm = OffloadMoELayer._expert_gemm
    injected = []
    def capture_failure(layer, *a, **kw):
        current = getattr(cache, "speculative_controller", None)
        if (torch.cuda.is_current_stream_capturing() and layer.layer_id == 5
                and current is not None and current.phase == "preverify"):
            injected.append(True)
            raise RuntimeError("injected coarse capture failure")
        return original_gemm(layer, *a, **kw)
    OffloadMoELayer._expert_gemm = capture_failure
    try:
        try:
            QwenMTPRunner(engine)
        except RuntimeError as exc:
            assert "injected coarse capture failure" in str(exc)
        else:
            raise AssertionError("capture fault was not injected")
    finally:
        OffloadMoELayer._expert_gemm = original_gemm
    assert injected and not torch.cuda.is_current_stream_capturing()
    clean()
    assert generate() == expected
    clean()
    checks["capture_failure_joins_layer_copy_forks"] = True
    expected_graphs = 2 * (graphs.width + graphs.outer_width) if graphs.exact else 4
    assert len(graphs.graphs) == graphs.captures == expected_graphs
    write("validation.json", dict(checks=checks, graph=graphs.report(), recovery_tokens=expected))
    for name, digest in identity["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"source changed during audit: {name}")
    write("complete.json", dict(status="complete", validated=True, checks=checks))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
