from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("mode", ["fixed_replay", "recheck"])
@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("delayed", [False, True])
def test_recovery_keeps_cpu_and_gpu_async_until_cpu_completion(monkeypatch, mode, reject, delayed):
    from types import SimpleNamespace
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import bench_cpu_recovery as benchmark
    from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, QwenAsyncGpuWindow
    from freetoken.engine.speculative import SpeculativeFrontier

    calls, jobs, polled = [], [], set()
    w = QwenAsyncGpuWindow.__new__(QwenAsyncGpuWindow)
    w.epoch, w.canonical_length, w.canonical_anchor, w.output_end = 1, 0, 0, 12
    w.engine = SimpleNamespace(config=SimpleNamespace(speculative_mtp_depth=1),
        stream=SimpleNamespace(synchronize=lambda: None))
    w.runner = SimpleNamespace(frontier=SpeculativeFrontier(1, 0, 0, 12))
    w.experts = SimpleNamespace(counters=torch.zeros(4), cache=SimpleNamespace(banks=[]))
    w._record_timing = lambda _: None
    w.replay = lambda *args: pytest.fail("fixed recovery must not replay old B")

    def generate(anchor, loops, *, candidate_cap=None):
        start = w.runner.frontier.target_valid_len
        calls.append(("generate", start, anchor))
        n = min(2 * loops, candidate_cap, w.output_end - start)
        block = GpuCandidateBlock(start, anchor, tuple(range(anchor + 1, anchor + n + 1)))
        w.runner.frontier = SpeculativeFrontier(1, start + n, start + n, 12)
        return block, []

    def recheck(block, remaining, **kwargs):
        calls.append(("recheck", block.start))
        n = min(len(block.candidates) + 1, remaining)
        tokens = tuple(range(block.anchor + 1, block.anchor + n + 1))
        w.runner.frontier = SpeculativeFrontier(1, block.start + n, block.start + n, 12)
        return SimpleNamespace(block=GpuCandidateBlock(block.start, block.anchor, tokens),
            stop_reason="exhausted", matched_old_tokens=min(len(block.candidates), n),
            dropped_old_tokens=0, bonus_token=tokens[-1], private_end=block.start + n, segments=())

    def install(result):
        calls.append(("install", result.block_id))
        w.canonical_length, w.canonical_anchor = result.checkpoint.length, result.tokens[-1]
        w.runner.frontier = SpeculativeFrontier(1, w.canonical_length, w.canonical_length, 12)
        return object()

    def submit(job):
        assert job.start == w.canonical_length and job.anchor == w.canonical_anchor
        assert job.start + len(job.candidates) + 1 <= 12
        jobs.append(job)
        calls.append(("submit", job.block_id, job.start))

    def poll():
        job = jobs[-1]
        calls.append(("poll", job.block_id))
        if delayed and job.block_id not in polled:
            polled.add(job.block_id)
            return None
        accepted = 0 if reject and job.block_id == 1 else len(job.candidates)
        n = min(accepted + 1, job.remaining)
        tokens = tuple(range(job.anchor + 1, job.anchor + n + 1))
        if reject and job.block_id == 1:
            tokens = (999,)
        result = SimpleNamespace(start=job.start, block_id=job.block_id, terminal=n == job.remaining,
            accepted=accepted, input_count=n, predictions=torch.zeros(len(job.candidates) + 1),
            checkpoint=SimpleNamespace(length=job.start + n), tokens=tokens)
        return result, dict(compute_seconds=0, publish_seconds=0)

    w.generate_block, w.recheck, w.install = generate, recheck, install
    worker = SimpleNamespace(submit=submit, poll=poll, release=lambda *args: None)
    measured = benchmark.run_pair(w, worker, SimpleNamespace(length=0, anchor=0), 1, mode,
        handoff_limit=0)
    assert measured["request_complete"] and measured["confirmed_tokens"] == 12
    assert measured["actual_handoffs"] > 2 and len(measured["output_tokens"]) == 12
    assert all(h["deadline_miss"] == delayed for h in measured["handoffs"])
    for i, call in enumerate(calls):
        if call[0] == "submit":
            assert calls[i + 1][0] == "generate" and calls[i + 2][0] == "poll"
    retained = [h for h in measured["handoffs"] if h["recovery"]["retained"]]
    assert retained
    if mode == "fixed_replay":
        assert all(h["recovery"]["reason"] == "install_only" for h in retained)
        first = retained[0]
        next_handoff = measured["handoffs"][first["block_id"] + 1]
        assert next_handoff["ahead"]["start"] == next_handoff["submitted"]["start"]
    else:
        assert any(c[0] == "recheck" for c in calls)


@pytest.fixture
def prefix_audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import audit_cpu_verify_prefixes
    return audit_cpu_verify_prefixes


@pytest.mark.parametrize("profile,components,expected", [
    ("v1", None, ()), ("ggml_v2", None, ("attention",)),
    ("ggml_v2", ["gdn", "attention"], ("attention", "gdn")),
])
def test_prefix_audit_restores_arithmetic_and_capacity(prefix_audit, profile, components, expected):
    options = prefix_audit.worker_options(dict(backend="optimized", kernel_profile=profile,
        components=components, ungrouped=True, bind_openmp=True, loops=[1, 12],
        max_input_tokens=4096, max_output_tokens=256))
    assert options == dict(backend="optimized", kernel_profile=profile, components=expected,
        grouped=False, bind_openmp=True, max_rows=49, max_context=4352, frozen_root=None)


def test_prefix_audit_requires_matching_runtime_metadata(prefix_audit):
    config = dict(backend="optimized", kernel_profile="ggml_v2", components=["attention", "gdn"])
    options = prefix_audit.worker_options(config)
    metadata = dict(backend="optimized", kernel_profile="ggml_v2", v2_components=["attention", "gdn"],
                    arithmetic_version="cpu_target_ggml_v2:attention,gdn", extension_sha256="original",
                    cuda_initialized=False)
    with pytest.raises(ValueError, match="lacks CPU runtime identity"):
        prefix_audit.validate_worker(config, 23, metadata, options)
    config["cpu_runtime"] = {"23": dict(metadata)}
    prefix_audit.validate_worker(config, 23, metadata, options)
    for change in (dict(kernel_profile="v1"), dict(v2_components=["attention"]),
                   dict(extension_sha256="different"), dict(cuda_initialized=True)):
        with pytest.raises(ValueError):
            prefix_audit.validate_worker(config, 23, dict(metadata, **change), options)


def test_prefix_audit_restores_frozen_runtime(prefix_audit, tmp_path):
    config = dict(backend="optimized", frozen_cpu_root=str(tmp_path))
    options = prefix_audit.worker_options(config)
    assert options["frozen_root"] == str(tmp_path)
    metadata = dict(backend="optimized", kernel_profile="v1", v2_components=[], cuda_initialized=False,
                    frozen_v1_runtime=True, runtime_source_root=str(tmp_path),
                    runtime_files={"target": "original"}, frozen_manifest_sha256="manifest")
    config["cpu_runtime"] = {"23": dict(metadata)}
    prefix_audit.validate_worker(config, 23, metadata, options)
    for change in (dict(runtime_files={"target": "different"}), dict(frozen_manifest_sha256="different"),
                   dict(runtime_source_root="/wrong"), dict(frozen_v1_runtime=False)):
        with pytest.raises(ValueError):
            prefix_audit.validate_worker(config, 23, dict(metadata, **change), options)
    with pytest.raises(ValueError, match="optimized v1"):
        prefix_audit.worker_options(dict(config, kernel_profile="ggml_v2"))


def test_frozen_prefix_audit_checks_wrapper_not_unused_local_target(prefix_audit, tmp_path):
    import hashlib
    name = "python/freetoken/engine/qwen_cpu_worker.py"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text("worker")
    config = dict(frozen_cpu_root="/frozen", source_sha256={name: hashlib.sha256(b"worker").hexdigest()})
    prefix_audit.validate_sources(config, tmp_path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="drifted"):
        prefix_audit.validate_sources(config, tmp_path)


@pytest.mark.parametrize("mismatch", [False, True])
def test_prefix_audit_main_uses_factory_and_records_actual_runtime(prefix_audit, monkeypatch, tmp_path, mismatch):
    import hashlib
    import json
    from types import SimpleNamespace
    import torch

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"text_config": {}}')
    metadata = dict(backend="optimized", kernel_profile="ggml_v2", v2_components=["attention", "gdn"],
                    arithmetic_version="cpu_target_ggml_v2:attention,gdn", extension_sha256="binary",
                    cuda_initialized=False, core_ids=[1])
    config = dict(model=str(model), model_config_sha256=hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        weight_shard_stat={}, threads=[1], samples=1, backend="optimized", kernel_profile="ggml_v2",
        components=["attention", "gdn"], cpu_runtime={"1": metadata})
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(dict(length=2, anchor=3, **{k: torch.zeros(1) for k in ("keys", "values", "recurrent", "conv")}),
               tmp_path / "gpu_prefix_sample0.pt")
    calls = []
    worker = SimpleNamespace(metadata=dict(metadata, kernel_profile="v1") if mismatch else metadata,
        wait_ready=lambda: None, initialize=lambda *args: None, audit=lambda *args: calls.append("audit"),
        submit=lambda *args: None, cancel=lambda *args: None, close=lambda: calls.append("close"),
        owners=[None], process=SimpleNamespace(is_alive=lambda: False, exitcode=0))
    def factory(*args, **kwargs):
        calls.append(kwargs)
        return worker
    def wait_for(worker, kind):
        if kind == "result":
            raise RuntimeError("canonical pending anchor")
        return dict(checks=["parity"], cuda_initialized=False)
    monkeypatch.setattr(prefix_audit, "create_cpu_worker", factory)
    monkeypatch.setattr(prefix_audit, "validate_sources", lambda *args: None)
    monkeypatch.setattr(prefix_audit, "wait_for", wait_for)
    monkeypatch.setattr(prefix_audit.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["audit_cpu_verify_prefixes.py", str(tmp_path), "--output", str(output)])
    if mismatch:
        with pytest.raises(ValueError, match="arithmetic differs"):
            prefix_audit.main()
        assert not output.exists() and "audit" not in calls
    else:
        prefix_audit.main()
        report = json.loads(output.read_text())
        assert report["status"] == "validated"
        assert report["cpu_runtime"]["1"] == metadata
    assert calls[0]["kernel_profile"] == "ggml_v2"
    assert calls[0]["components"] == ("attention", "gdn")
    assert calls[-1] == "close"


def test_commit_diagnostics_preserve_result_and_restore_trace(monkeypatch):
    import torch

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    monkeypatch.syspath_prepend(str(root / "tests"))
    from engine.test_qwen_cpu_target import tiny_target
    from bench_cpu_target_fixed import Components
    from freetoken.engine.qwen_cpu_target import VerifyJob, bitwise_equal

    previous_threads = torch.get_num_threads()
    target = None
    try:
        torch.set_num_threads(2)
        target, checkpoint = tiny_target("ggml_v2")
        job = VerifyJob(7, 0, checkpoint.length, checkpoint.anchor, (4, 5), 1)
        expected = target.verify(job)
        target.initialize(checkpoint, 7)
        meter = Components(target)
        actual = target.verify(job)
        assert sys.gettrace() is None and meter.seconds["commit_and_result"] > 0
        assert actual.tokens == expected.tokens and actual.accepted == expected.accepted
        assert bitwise_equal(actual.hidden, expected.hidden)
        assert bitwise_equal(actual.predictions, expected.predictions)
        for field in ("keys", "values", "recurrent", "conv"):
            assert bitwise_equal(getattr(actual.checkpoint, field), getattr(expected.checkpoint, field))
    finally:
        if target is not None:
            target.close()
        torch.set_num_threads(previous_threads)


def test_cycle_estimate_pairs_each_sample_before_aggregation():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    try:
        from bench_cpu_verify_feasibility import summarize

        rows = [dict(threads=23, loops=1, cpu_publish_seconds=cpu,
                     gpu_generation_seconds=gen, install_replay_seconds=restore,
                     candidate_count=4, deadline_miss=True)
                for cpu, gen, restore in ((10, 1, 1), (1, 10, 1), (1, 1, 10))]
        result = summarize(rows)[0]
        assert result["cycle_estimate_seconds"] == 11
        assert result["cycle_deadline_misses"] == 1
        assert not result["cycle_estimate_feasible"]
        assert result["coverage_margin_seconds"] == 0
        assert result["coverage_deadline_misses"] == 1
        assert not result["coverage_feasible"]
    finally:
        sys.path.pop(0)


def test_coverage_excludes_gpu_replay_and_separates_loop_horizons(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from bench_cpu_verify_feasibility import summarize

    rows = [dict(threads=23, loops=8, ahead_loops=ahead, cpu_publish_seconds=0.2,
                 gpu_generation_seconds=generation, install_replay_seconds=0.2,
                 candidate_count=31, deadline_miss=False)
            for ahead, generation in ((4, 0.3), (8, 0.5))]
    short, long = summarize(rows)
    assert short["cycle_estimate_feasible"]
    assert short["coverage_feasible"]
    assert short["ahead_loops"] == 4
    assert short["coverage_margin_seconds"] == pytest.approx(0.1)
    assert short["window_estimate_seconds"] == pytest.approx(0.5)
    assert long["ahead_loops"] == 8
    assert long["coverage_feasible"]
    assert long["coverage_margin_seconds"] == pytest.approx(0.3)


def test_four_loop_coverage_does_not_hide_cpu_wait_inside_gpu_replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from bench_cpu_verify_feasibility import summarize

    row = dict(threads=46, loops=4, ahead_loops=4, cpu_publish_seconds=0.4,
        gpu_generation_seconds=0.3, install_replay_seconds=0.2,
        candidate_count=16, deadline_miss=True)
    result = summarize([row])[0]
    assert not result["coverage_feasible"]
    assert not result["generation_only_no_wait"]
    assert result["coverage_margin_seconds"] == pytest.approx(-0.1)
    assert result["window_estimate_seconds"] == pytest.approx(0.6)
    assert result["generation_plus_replay_feasible"]
    assert result["generation_plus_replay_margin_seconds"] == pytest.approx(0.1)


def test_rejected_cpu_prefix_discards_gpu_tail_in_real_replay_budget(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from bench_cpu_verify_feasibility import replay_block_for_result, GpuCandidateBlock

    result = SimpleNamespace(start=4, terminal=False, accepted=1, predictions=SimpleNamespace(numel=lambda: 4),
        input_count=2, checkpoint=SimpleNamespace(length=6), tokens=(11, 12))
    future = GpuCandidateBlock(7, 13, (21, 22, 23))
    block, retained = replay_block_for_result(result, future)
    assert not retained and block.start == 6 and block.anchor == 12 and block.candidates == ()
    diagnostic, retained = replay_block_for_result(result, future, "forced_diagnostic")
    assert not retained and diagnostic.candidates == future.candidates
    result.accepted, result.input_count = 3, 4
    result.checkpoint.length, result.tokens = 8, (11, 12, 13, 21)
    block, retained = replay_block_for_result(result, future)
    assert retained and block == GpuCandidateBlock(8, 21, (22, 23))


def test_resource_interference_distinguishes_own_gpu_and_cpu_workers():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    try:
        from bench_cpu_verify_feasibility import resource_interference
        snapshot = dict(gpu_processes="pid, uuid, memory\n10, GPU-own, 100\n20, GPU-other, 10\n",
                        processes="PID PPID PSR PCPU COMM\n10 1 0 90 python\n11 10 1 2300 python\n12 10 0 100 helper\n30 1 2 200 other\n")
        report = resource_interference(snapshot, {10, 11})
        assert report["external_gpu_pids"] == [20]
        assert [r["pid"] for r in report["external_cpu_processes"]] == [30]
        assert report["detected"]
        assert not resource_interference(snapshot, {10, 11, 20, 30})["detected"]
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("lifetime_percent,tick_delta,detected", [(90, 1, False), (1, 80, True)])
def test_cpu_interference_uses_measured_interval(monkeypatch, lifetime_percent, tick_delta, detected):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from bench_cpu_verify_feasibility import process_cpu_interval, resource_interference

    before = dict(monotonic=10.0, ticks_per_second=100,
        processes={"30": dict(ppid=1, ticks=5000, start_ticks=100, command="indexer")})
    after = dict(monotonic=11.0, ticks_per_second=100,
        processes={"30": dict(ppid=1, ticks=5000 + tick_delta, start_ticks=100, command="indexer")})
    snapshot = dict(gpu_processes="pid, uuid, memory\n",
        processes=f"PID PPID PSR PCPU COMM\n30 1 2 {lifetime_percent} indexer\n",
        process_cpu_counters=after, process_cpu_interval=process_cpu_interval(before, after))
    report = resource_interference(snapshot, {10})
    assert report["detected"] is detected
    assert report["cpu_usage_basis"] == "proc_stat_interval"
    snapshot["process_cpu_interval"] = None
    assert resource_interference(snapshot, {10}, previous={"process_cpu_counters": before}) == report


def test_cpu_interval_accounts_for_new_and_reused_processes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from bench_cpu_verify_feasibility import process_cpu_interval

    before = dict(monotonic=10.0, ticks_per_second=100,
        processes={"30": dict(ppid=1, ticks=5000, start_ticks=100, command="old")})
    after = dict(monotonic=11.0, ticks_per_second=100, processes={
        "30": dict(ppid=1, ticks=80, start_ticks=1010, command="replacement"),
        "40": dict(ppid=1, ticks=60, start_ticks=1020, command="new")})
    result = process_cpu_interval(before, after)
    assert {p["pid"]: p["cpu_percent"] for p in result["processes"]} == {30: 80, 40: 60}
    with pytest.raises(ValueError, match="advancing time"):
        process_cpu_interval(before, before)


def test_cpu_window_budget_reserves_late_buffers_before_choosing_slots():
    from freetoken.engine.baseline_sd import plan_measured_memory

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    try:
        from bench_cpu_verify_feasibility import plan_cpu_window_memory

        mib = 1 << 20
        native = plan_measured_memory(capacity_tokens=2176, kv_pages=2176,
            budget_bytes=1024*mib, measured_peak_bytes=600*mib, initial_slots=40,
            expert_slot_bytes=10*mib, minimum_slots=10, maximum_slots=200,
            automatic=True)
        actual = plan_cpu_window_memory(native, 800*mib, 40, 10, 200)
        assert actual.final_slots == 56
        assert actual.final_slots < native.final_slots
        actual.validate(actual.final_slots)
        import pytest
        with pytest.raises(ValueError, match="budget insufficient"):
            actual.validate(actual.final_slots + 1)
        with pytest.raises(ValueError, match="budget insufficient"):
            plan_cpu_window_memory(native, 2000*mib, 40, 10, 200)
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("auto_slots", [False, True])
def test_cpu_window_epochs_follow_gpu_after_sizing(monkeypatch, auto_slots):
    from types import SimpleNamespace
    import torch

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import bench_cpu_verify_feasibility as benchmark

    class Window:
        def __init__(self, *args, **kwargs):
            self.epoch = 0
            self.active = False

        def initialize(self, prompt, chunk_size=None):
            assert not self.active
            self.active = True
            self.epoch += 1
            return SimpleNamespace(anchor=7, prompt=prompt)

        def audit_gpu_anchor(self, anchor):
            pass

        def warm_widths(self, anchor):
            pass

        def close_request(self):
            self.active = False

    class Worker:
        def __init__(self):
            self.epochs = []

        def require_drained(self):
            pass

        def initialize(self, checkpoint, epoch):
            assert epoch == window.epoch
            assert not self.epochs or epoch > self.epochs[-1]
            self.epochs.append(epoch)

        def poll(self):
            return dict(kind="initialized", epoch=self.epochs[-1])

    window = Window()
    if auto_slots:
        cache = SimpleNamespace(cache_size=512, num_layers=40, num_experts=256, max_slots=0)
        engine = SimpleNamespace(moe_offload_cache=cache, baseline_sd_memory_plan=None,
            config=SimpleNamespace(model_config=SimpleNamespace(num_experts=256), moe_prefill_overlap=True),
            speculative_runner=None, device="cpu", _baseline_allocated=0, _physical_baseline_free=1000,
            rebuild_runtime_cache=lambda **kw: setattr(cache, "cache_size", kw["moe_cache_size"]))
        monkeypatch.setattr(benchmark, "QwenAsyncGpuWindow", Window)
        monkeypatch.setattr(benchmark, "plan_cpu_window_memory", lambda *a: SimpleNamespace(final_slots=2000))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 100)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 100)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (900, 1000))
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        window, _ = benchmark.prepare_auto_window(engine, [7], None, False)
        assert window.epoch == 1
    worker = Worker()
    for prompt in ([7], [0], [0], [1], [1]):
        checkpoint = benchmark.initialize_cpu_window(window, worker, prompt, None)
        assert checkpoint.prompt == prompt
        window.close_request()
    assert worker.epochs == list(range(2 if auto_slots else 1, 7 if auto_slots else 6))


def test_feasibility_entrypoint_allows_runtime_inference_tensor_updates(tmp_path, monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import bench_cpu_verify_feasibility as benchmark

    with torch.inference_mode():
        runtime_buffer = torch.empty(1)

    def prepare(*args, **kwargs):
        runtime_buffer.fill_(1)
        raise RuntimeError("inference scope verified")

    monkeypatch.setattr(benchmark, "prepare_experiment", prepare)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--model", "unused", "--data", "unused",
                                    "--output", str(tmp_path / "run")])
    with pytest.raises(RuntimeError, match="inference scope verified"):
        benchmark.main()
    assert not torch.is_inference_mode_enabled()


@pytest.mark.parametrize("mode", ["layerwise", "layerwise_barrier"])
def test_streamed_window_binds_barrier_policy_before_candidates(monkeypatch, mode):
    from types import SimpleNamespace
    from freetoken.engine.prefix_transfer import PrefixTicket

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import bench_cpu_verify_feasibility as benchmark

    ticket = PrefixTicket(1, 2, 3)
    transfer = object()
    calls = []

    def initialize(prompt, *, chunk_size, prefix_transfer, prefix_mode):
        assert prefix_transfer is transfer and prefix_mode == mode
        return ticket

    worker = SimpleNamespace(require_drained=lambda: None,
        initialize_streamed=lambda value, **kw: calls.append((value, kw)),
        poll=lambda: dict(kind="initialized", epoch=1))
    window = SimpleNamespace(initialize=initialize)
    assert benchmark.initialize_cpu_window(window, worker, [1, 2], None, transfer, mode) is ticket
    assert calls == [(ticket, dict(wait_all=mode == "layerwise_barrier"))]


def test_prefix_analysis_rejects_mixed_barrier_semantics(monkeypatch, tmp_path):
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import analyze_prefix_transfer as analysis

    for name, stage in (("legacy", None), ("new", "before_cpu_verify")):
        root = tmp_path / name
        root.mkdir()
        (root / "config.json").write_text(json.dumps(dict(budget_gib=24, prompt_chunk_size=512)))
        row = dict(sample_id=0, prefix_transfer="layerwise_barrier")
        if stage is not None:
            row["prefix_barrier_stage"] = stage
        (root / "measurements.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="cannot combine"):
        analysis.summarize(tmp_path)


def test_prefix_matrix_preserves_initialization_failure(monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import run_prefix_transfer_matrix as matrix
    import qwen_mtp_config

    baseline = tmp_path / "baseline"
    (baseline / "budget_24g_retained").mkdir(parents=True)
    (baseline / "budget_24g_retained/config.json").write_text(json.dumps(dict(actual_slots=2428, actual_kv_pages=2176)))
    command = [sys.executable, "benchmark.py", "--budget-gib", "24", "--auto-slots", "--formal-timing", "--output", "unused"]
    (baseline / "retained_matrix_complete.json").write_text(json.dumps(dict(jobs=[dict(command=command)])))
    output = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", ["matrix", "--baseline", str(baseline), "--output", str(output), "--phase", "audit"])
    monkeypatch.setattr(qwen_mtp_config, "source_fingerprint", lambda _: {})

    def fail(command, **kwargs):
        directory = Path(command[command.index("--output") + 1])
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "complete.json").write_text('{"status": "incomplete"}')
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(matrix.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="gate failed"):
        matrix.main()
    result = json.loads((output / "manifest.json").read_text())
    assert result["jobs"][0]["status"] == "incomplete_or_invalid"
    assert result["jobs"][0]["returncode"] == 1
