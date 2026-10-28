from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import torch

from .qwen_async_three_level import GpuCandidateBlock, QwenAsyncGpuWindow
from .qwen_cpu_target import VerifyJob
from .qwen_cpu_worker import CpuVerifierWorker
from .qwen_mtp import GenerationResult


def configure_cpu_resources(config):
    from freetoken.moe.cpu_executor import physical_core_cpus

    physical = physical_core_cpus()
    allowed = set(os.sched_getaffinity(0))
    available = physical[2:]
    target = config.cpu_async_core_ids
    hybrid = config.moe_cpu_core_ids if config.moe_strategy == "hybrid" else ()
    if target is None and hybrid is None:
        if len(available) < 2:
            raise ValueError("CPU async Hybrid needs two execution cores plus two coordinator cores")
        split = max(1, min(len(available) - 1, 3 * len(available) // 4))
        target, hybrid = available[:split], available[split:]
    elif target is None:
        target = [cpu for cpu in available if cpu not in hybrid]
    elif hybrid is None:
        hybrid = [cpu for cpu in available if cpu not in target]
    target, hybrid = tuple(target), tuple(hybrid)
    if not target or (config.moe_strategy == "hybrid" and not hybrid):
        raise ValueError("CPU async executor affinity is empty")
    groups = []
    for cpus in (target, hybrid):
        if len(set(cpus)) != len(cpus) or not set(cpus) <= allowed:
            raise ValueError("CPU async affinity contains duplicates or CPUs outside the process affinity")
        siblings = set()
        for cpu in cpus:
            path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            siblings.add(path.read_text().strip() if path.exists() else str(cpu))
        if len(siblings) != len(cpus):
            raise ValueError("CPU async affinity must use distinct physical cores")
        groups.append(siblings)
    if groups[0] & groups[1]:
        raise ValueError("CPU Target and Hybrid executor cannot share physical cores")
    coordinator = None
    if hybrid:
        for cpu in physical:
            path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            siblings = path.read_text().strip() if path.exists() else str(cpu)
            if siblings not in groups[0] | groups[1]:
                coordinator = cpu
                break
        if coordinator is None:
            raise ValueError("CPU async Hybrid requires a spare physical core for its coordinator")
    object.__setattr__(config, "_cpu_moe_coordinator_core", coordinator)
    object.__setattr__(config, "cpu_async_core_ids", target)
    object.__setattr__(config, "moe_cpu_core_ids", hybrid or None)
    if hybrid:
        object.__setattr__(config, "moe_cpu_threads", len(hybrid))


class _Cancelled(Exception):
    pass


class _Finished(Exception):
    pass


class CpuAsyncRuntime:
    def __init__(self, engine):
        self.engine = engine
        self.runner = engine.speculative_runner
        self.window = QwenAsyncGpuWindow(self.runner, instrument_components=False)
        self.split = None
        self.prefix_transfer = None
        if engine.config.cpu_async_prefix_layers is not None:
            from .qwen_split_verify import QwenSplitVerifier
            self.split = QwenSplitVerifier(self.window, engine.config.cpu_async_prefix_layers)
        self.worker = None
        self.lock = threading.Lock()

    def _prepare_worker(self):
        if self.worker is not None:
            return
        e, c = self.engine, self.engine.config
        hf = getattr(c.hf_config, "text_config", c.hf_config)
        if e.draft_method == "eagle3":
            from freetoken.models.qwen3_moe.gguf import _hf_config
            hf = _hf_config(hf)
        config = hf.to_dict() if hasattr(hf, "to_dict") else dict(vars(hf))
        mc = c.model_config
        config.update(hidden_size=mc.hidden_size, head_dim=mc.head_dim,
                      num_hidden_layers=mc.num_layers, num_key_value_heads=mc.num_kv_heads)
        config.setdefault("layer_types", ["linear_attention" if mc.is_linear_layer(i) else "full_attention"
                                          for i in range(mc.num_layers)])
        config["draft_feature_size"] = e.draft_feature_size
        if e.draft_method == "eagle3":
            config["eagle3_aux_layer_boundaries"] = tuple(e.mtp.config.aux_layer_boundaries)
        shared = None
        if e.moe_offload_cache is not None:
            from freetoken.layers.moe import iter_moe_layers
            from freetoken.moe.shared_banks import export_cpu_banks
            layers = sorted(iter_moe_layers(e.model), key=lambda layer: layer.layer_id)
            cache = e.moe_offload_cache
            prefix = c.cpu_async_prefix_layers
            sources = ({name: banks[:prefix] for name, banks in cache.bank_sources.items()}
                       if prefix is not None else cache.bank_sources)
            formats = [layer.quant_method.cpu_spec() for layer in layers[:prefix]]
            shared = export_cpu_banks(sources, formats,
                                      cache.num_experts)
        worker = CpuVerifierWorker(c.model_path, config, c.cpu_async_core_ids,
            max_context=e.max_seq_len, max_rows=c.cpu_async_loops * (c.speculative_mtp_depth + 1) + 1,
            grouped=True, backend="optimized", bind_openmp=True, shared_banks=shared,
            kernel_profile=c.cpu_async_kernel_profile, components=c.cpu_async_kernel_components,
            cpu_prefix_layers=c.cpu_async_prefix_layers)
        try:
            worker.wait_ready(c.cpu_async_timeout)
            if self.split is not None:
                from .cpu_async_split import PrefixStateTransfer
                self.prefix_transfer = PrefixStateTransfer(e, worker)
                self.window.validate_budget()
        except BaseException:
            try:
                if self.prefix_transfer is not None:
                    self.prefix_transfer.close()
                    self.prefix_transfer = None
            finally:
                worker.close()
            raise
        self.worker = worker

    def _wait(self, kind, cancel):
        deadline = time.monotonic() + self.engine.config.cpu_async_timeout
        while True:
            if cancel():
                raise _Cancelled()
            result = self.worker.poll()
            if result is not None:
                actual = "result" if isinstance(result, tuple) else result["kind"]
                if actual != kind:
                    raise RuntimeError(f"expected CPU {kind}, received {actual}")
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(f"CPU verifier {kind} deadline exceeded")
            time.sleep(0.001)

    @torch.inference_mode()
    def generate(self, input_ids, *, max_tokens, eos_ids=(), cancel=None, prompt_chunk_size=None):
        c, e, window = self.engine.config, self.engine, self.window
        if type(max_tokens) is not int or not 1 <= max_tokens <= c.speculative_max_output_tokens:
            raise ValueError("max_tokens must be positive and within the configured output capacity")
        eos_ids = tuple(eos_ids)
        if any(type(token) is not int or not 0 <= token < c.model_config.vocab_size for token in eos_ids):
            raise ValueError("EOS token outside vocabulary")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("CPU async supports one request at a time")
        cancel = cancel.is_set if hasattr(cancel, "is_set") else cancel or (lambda: False)
        outputs, rounds = [], []
        started = time.perf_counter()
        first = None
        prefilled = False
        failed = False
        self.runner.graph_replays, self.runner.graph_fallbacks = {}, {}
        try:
            if cancel():
                raise _Cancelled()
            self.window.prepare_graphs()
            self.runner.graph_replays, self.runner.graph_fallbacks = {}, {}
            self._prepare_worker()
            started = time.perf_counter()
            if cancel():
                raise _Cancelled()
            checkpoint = window.initialize(input_ids, chunk_size=prompt_chunk_size, eos_ids=eos_ids)
            prefilled = True
            outputs.append(checkpoint.anchor)
            first = time.perf_counter()
            if checkpoint.anchor in eos_ids or max_tokens == 1:
                raise _Finished()
            self.worker.initialize(checkpoint, window.epoch)
            self._wait("initialized", cancel)
            cap = min(c.cpu_async_loops * (c.speculative_mtp_depth + 1), max_tokens - len(outputs) - 1)
            block, inner = window.generate_block(checkpoint.anchor, c.cpu_async_loops, candidate_cap=max(0, cap))
            block_id = 0
            while len(outputs) < max_tokens:
                if cancel():
                    raise _Cancelled()
                remaining = max_tokens - len(outputs)
                block = GpuCandidateBlock(block.start, block.anchor, block.candidates[:max(0, remaining - 1)])
                submitted = time.perf_counter()
                job = VerifyJob(window.epoch, block_id, block.start, block.anchor,
                                block.candidates, remaining, tuple(eos_ids))
                hints = self.split.capture_hints() if self.split is not None else None
                if self.split is None:
                    self.worker.submit(job)
                else:
                    self.worker.submit_prefix(job)
                    self.split.begin_prefetch(hints)
                ahead_cap = max(0, min(c.cpu_async_loops * (c.speculative_mtp_depth + 1),
                    checkpoint.length + max_tokens - 1 - self.runner.frontier.target_valid_len))
                ahead_started = time.perf_counter()
                ahead, ahead_inner = window.generate_ahead(block, c.cpu_async_loops, candidate_cap=ahead_cap)
                e.stream.synchronize()
                generation_seconds = time.perf_counter() - ahead_started
                waiting = time.perf_counter()
                result, timing = self._wait("result", cancel)
                wait_seconds = time.perf_counter() - waiting
                if result.start != window.canonical_length or result.block_id != block_id:
                    raise RuntimeError("CPU result does not belong to the submitted canonical window")
                split_timing = {}
                install_started = time.perf_counter()
                if self.split is None:
                    reader = window.install(result)
                    self.worker.release(result, reader)
                else:
                    prefix = result
                    hidden = self.split.stage_hidden(prefix)
                    self.prefix_transfer.stage(prefix)
                    suffix_started = time.perf_counter()
                    verified = self.split.verify(prefix, job, hidden=hidden)
                    suffix_seconds = time.perf_counter() - suffix_started
                    self.worker.commit_prefix(prefix, verified.input_count, verified.tokens[-1])
                    result, reader = self.prefix_transfer.install(window, prefix, verified)
                    split_timing = dict(cpu_prefix_layers=c.cpu_async_prefix_layers,
                        gpu_generation_seconds=generation_seconds,
                        gpu_suffix_host_seconds=suffix_seconds,
                        prefix_state_transfer_bytes=sum(getattr(prefix, name).numel()
                            * getattr(prefix, name).element_size() for name in ("keys", "values", "replay", "conv")),
                        suffix_prefetch_only=True, verify_prefetch=False)
                outputs.extend(result.tokens)
                continuing = not result.terminal and len(outputs) < max_tokens
                recovery_started = time.perf_counter()
                next_block, recovery = window.recover(result, ahead, c.cpu_async_recovery, c.cpu_async_loops,
                    refill=continuing, candidate_cap=max(0, max_tokens - len(outputs) - 1), eos_ids=eos_ids)
                recovery_enqueued = time.perf_counter()
                if self.split is not None:
                    commit_reply = self._wait("prefix_committed", cancel)
                    self.worker.release(prefix, reader)
                    split_timing.update(cpu_commit_seconds=commit_reply.get("commit_seconds"),
                                        cpu_commit_wait_seconds=time.perf_counter() - recovery_enqueued)
                rounds.append(dict(mode="cpu_async", block_id=block_id, start=result.start,
                    candidates=len(block.candidates), matched_drafts=result.accepted,
                    returned_drafts=min(result.accepted, len(result.tokens)),
                    committed_input_rows=result.input_count, returned_tokens=len(result.tokens),
                    terminal=result.terminal, inner=inner, ahead_inner=ahead_inner, recovery=recovery,
                    cpu_wait_seconds=wait_seconds, handoff_seconds=time.perf_counter() - submitted,
                    cpu_compute_seconds=timing["compute_seconds"], cpu_publish_seconds=timing["publish_seconds"],
                    cpu_verify_phases=timing.get("verify_phases"),
                    timestamps=dict(submit=submitted, cpu_start=timing.get("cpu_started_at"),
                        cpu_compute_end=timing.get("cpu_compute_ended_at"), cpu_publish=timing.get("published_at"),
                        gpu_start=ahead_started, gpu_end=waiting, result_received=install_started,
                        install_start=install_started, recovery_start=recovery_started,
                        recovery_enqueued=recovery_enqueued, handoff_end=time.perf_counter()),
                    ahead_start=ahead.start, ahead_candidates=list(ahead.candidates),
                    proposed_candidates=list(block.candidates), next_proposal_start=next_block.start if next_block else None,
                    next_proposal_candidates=list(next_block.candidates) if next_block else [],
                    **split_timing))
                if not continuing:
                    break
                if next_block is None:
                    raise RuntimeError("CPU async recovery produced no candidate window")
                block, inner = next_block, recovery.get("inner", ())
                block_id += 1
        except _Finished:
            pass
        except _Cancelled:
            failed = getattr(self, "split", None) is not None
            if self.worker is not None and self.worker.epoch is not None:
                self.worker.cancel(self.worker.epoch)
                try:
                    self._wait("cancelled", lambda: False)
                except BaseException:
                    failed = True
                    raise
        except BaseException:
            failed = True
            raise
        finally:
            try:
                self._finish_request(failed)
            finally:
                self.lock.release()
        metrics = (window.experts.request_metrics() if prefilled else
                   dict(available=True, scope="gpu_decode", cancelled_before_prefill=True,
                        logical_total_loads=0, logical_total_bytes=0))
        metrics.update(cpu_target_core_ids=list(c.cpu_async_core_ids),
                       hybrid_core_ids=list(c.moe_cpu_core_ids or ()),
                       hybrid_coordinator_core=c._cpu_moe_coordinator_core,
                       recovery=c.cpu_async_recovery,
                       canonical_target="cpu_prefix_gpu_suffix" if getattr(self, "split", None) is not None else "cpu",
                       cpu_prefix_layers=getattr(c, "cpu_async_prefix_layers", None),
                       gpu_suffix_execution="eager_hybrid" if getattr(self, "split", None) is not None else None,
                       gpu_suffix_partition="fixed_route_rank" if getattr(self, "split", None) is not None else None,
                       gpu_suffix_gpu_experts=(c.cpu_async_suffix_gpu_experts
                           if getattr(self, "split", None) is not None else None),
                       graph_scope="gpu_generation_and_recovery" if getattr(self, "split", None) is not None else "gpu_window",
                       split_state_pinned_bytes=self.prefix_transfer.pinned_bytes if getattr(self, "prefix_transfer", None) else 0,
                       split_state_device_bytes=self.prefix_transfer.device_bytes if getattr(self, "prefix_transfer", None) else 0,
                       expert_io_scope="decode_after_prefill")
        wall = time.perf_counter() - started
        return GenerationResult(outputs, wall,
            (first - started) if first is not None else wall, rounds,
            graph_replays=dict(self.runner.graph_replays), graph_fallbacks=dict(self.runner.graph_fallbacks),
            graph_captures=dict(self.runner.graph_captures), metrics=metrics)

    def _close_worker(self):
        worker, self.worker = self.worker, None
        try:
            if getattr(self, "prefix_transfer", None) is not None:
                self.prefix_transfer.close()
                self.prefix_transfer = None
        finally:
            if worker is not None:
                worker.close()

    def _finish_request(self, failed):
        try:
            self.window.close_request()
        except BaseException:
            failed = True
            raise
        finally:
            if failed:
                self._close_worker()
            elif self.worker is not None:
                try:
                    self.worker.require_drained()
                except BaseException:
                    self._close_worker()
                    raise

    def close(self):
        try:
            self.window.close()
        finally:
            self._close_worker()


def generate_cpu_async(engine, input_ids, **kwargs):
    runtime = getattr(engine, "cpu_async_runtime", None)
    if runtime is None or runtime.runner is not engine.speculative_runner:
        if runtime is not None:
            runtime.close()
        runtime = engine.cpu_async_runtime = CpuAsyncRuntime(engine)
    return runtime.generate(input_ids, **kwargs)
