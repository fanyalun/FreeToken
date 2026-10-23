"""BF16 task-slot executor for speculative Hybrid and its arithmetic compatibility mode."""

from __future__ import annotations

import threading
import time
import weakref

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.moe.cpu_executor import _ACT_IDS, _FLAG_SYNC, physical_core_cpus, resolve_threads_and_affinity, logger

_FLAG_SLOTS_PER_LAYER = 16


class SpeculativeBf16Executor:
    """BF16 grouped experts with graph-stable task slots and ordered reduction."""

    def __init__(
        self,
        cache,
        *,
        top_k: int,
        activation: str,
        apply_router_weight_on_input: bool,
        num_threads: int,
        max_tokens: int,
        device: torch.device,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float | None = None,
        layer_formats: list[tuple[str, dict[str, int]]] | None = None,
        grouped_widths: tuple[int, ...] = (),
        grouped_compat: bool = False,
    ) -> None:
        from freetoken.kernel import _cpu_moe
        from freetoken.moe.legacy_format import canonical_role

        formats = ([(cache.quant_format, {})] * int(cache.num_layers)
                   if layer_formats is None else layer_formats)
        if len(formats) != cache.num_layers or any(name != "bf16" or opts for name, opts in formats):
            raise ValueError("speculative CPU experts require uniform BF16 layers")
        fmt = "bf16"
        if activation not in _ACT_IDS:
            raise NotImplementedError(f"CPU MoE backend: unsupported activation {activation!r}")

        self.num_layers = int(cache.num_layers)
        self.num_experts = int(cache.num_experts)
        self.top_k = int(top_k)
        self.quant_format = fmt
        self.device = device
        self.max_tokens = int(max_tokens)
        self.grouped_widths = tuple(sorted(set(grouped_widths)))
        self.grouped_compat = grouped_compat
        if grouped_compat and not hasattr(_cpu_moe.SpeculativeBf16Executor, "create_grouped_compat_task"):
            raise RuntimeError("rebuild _cpu_moe for baseline-compatible grouped tasks")
        if self.grouped_widths and (fmt != "bf16" or self.grouped_widths[0] < 1
                                   or self.grouped_widths[-1] > self.max_tokens):
            raise ValueError("grouped CPU MoE requires BF16 widths within max_tokens")
        if self.grouped_widths and not hasattr(_cpu_moe.SpeculativeBf16Executor, "prepare_grouped"):
            raise RuntimeError("rebuild _cpu_moe with `python setup.py build_ext --inplace` for grouped BF16")
        self._poisoned = False
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)
        self.activation_signature = [activation, self.apply_router_weight_on_input, float(swiglu_alpha),
                                     swiglu_limit]
        # The per-layer tensors and their pointer tables must outlive the executor
        # (C++ holds raw addresses into both).
        self._banks: list[torch.Tensor] = []
        ptrs, (self.H, self.I) = self._resolve_banks(
            {canonical_role(name): per_layer for name, per_layer in cache.bank_sources.items()}, fmt
        )

        # Decide the flag handshake up front (env + device + a functional stream-memop
        # probe): its coordinator needs a core of its own, which the auto thread sizing
        # below reserves (a coordinator time-slicing against the GEMV workers measurably
        # destabilizes throughput on fully-subscribed boxes).
        self._flag_sync = _FLAG_SYNC and device.type == "cuda"
        self._cpu_moe = _cpu_moe  # module ref for the decode-path memop calls
        if self._flag_sync:
            probe_scratch = alloc_pinned_tensor(1, dtype=torch.int64)
            probe_scratch.zero_()
            if not _cpu_moe.speculative_memops_probe(
                torch.cuda.current_stream().cuda_stream, probe_scratch.data_ptr()
            ):
                logger.info_rank0(
                    "cpu-moe flag handshake unavailable: CUDA stream memory operations "
                    "are not supported here (Windows WDDM / vGPU / old driver); using "
                    "the cudaLaunchHostFunc sync"
                )
                self._flag_sync = False
        if self.grouped_widths and not self._flag_sync:
            raise RuntimeError("speculative Hybrid requires CUDA Graph stream memops; callbacks are unsupported")

        nthreads, core_ids = resolve_threads_and_affinity(num_threads)
        coord_core = -1
        if self._flag_sync and num_threads == 0 and nthreads > 2:
            # Auto sizing: give the coordinator the last physical core instead of
            # oversubscribing (workers drop from N to N-1).
            coord_core = core_ids[-1]
            nthreads -= 1
            core_ids = core_ids[:-1]
        self._coord_core = coord_core
        self._ext = _cpu_moe.SpeculativeBf16Executor(
            num_threads=nthreads,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.H,
            inter_size=self.I,
            max_tokens=self.max_tokens,
            activation_id=_ACT_IDS[activation],
            apply_router_weight_on_input=1 if apply_router_weight_on_input else 0,
            swiglu_alpha=float(swiglu_alpha),
            swiglu_limit=float(swiglu_limit) if swiglu_limit is not None else float("inf"),
            core_ids=core_ids,
            **ptrs,
        )
        self.num_threads = nthreads
        if self.grouped_widths:
            self._ext.prepare_grouped(self.max_tokens)
        self.core_ids = core_ids
        self.isa = self._ext.isa_name()

        spare = len(physical_core_cpus()) - nthreads - (1 if coord_core >= 0 else 0) - 1
        clamp = max(1, min(torch.get_num_threads(), spare))
        if clamp < torch.get_num_threads():
            logger.info_rank0(
                f"torch intra-op threads: {torch.get_num_threads()} -> {clamp} "
                "(cores reserved for the pinned CPU MoE pool)"
            )
            torch.set_num_threads(clamp)

        self._io: dict[int, dict[str, torch.Tensor]] = {}
        self._tasks: dict[tuple[int, int], int] = {}

        # Flag-based handshake: mapped-pinned ready/done/err int64 arrays (one slot per
        # (MoE layer, decode batch size) pair, allocated as tasks are created) + a
        # persistent CPU coordinator that polls ready[], runs the slot's task on the
        # pool, and sets done[]. Binary per-step protocol (GPU memops: done=0, ready=1;
        # coordinator: consume ready, run, done=1; GPU waits done>=1 -- the WAIT
        # immediate is constant, so CUDA-graph replays are safe). err[] is raised by the
        # WATCHDOG thread when a ready flag stays unanswered (dead coordinator): it
        # poisons done to unblock the stream and raise_if_unhealthy() turns the step
        # into a loud error instead of silent stale output. Buffers are kept alive on
        # self so the coordinator's pinned pointers stay valid for the executor's
        # lifetime (flag_sync itself was decided above, before thread sizing).
        self._ready = self._done = self._err = None
        self._flag_slots: dict[tuple[int, int], int] = {}  # (layer_id, bs) -> slot
        self._flag_capacity = self.num_layers * (_FLAG_SLOTS_PER_LAYER + len(self.grouped_widths))
        if self._flag_sync:
            self._ready = alloc_pinned_tensor(self._flag_capacity, dtype=torch.int64)
            self._done = alloc_pinned_tensor(self._flag_capacity, dtype=torch.int64)
            self._err = alloc_pinned_tensor(self._flag_capacity, dtype=torch.int64)
            self._ready.zero_()
            self._done.zero_()
            self._err.zero_()
            self._ext.start_flag_coordinator(
                self._ready.data_ptr(), self._done.data_ptr(), self._flag_capacity,
                self._coord_core,
            )
            self._watchdog_stop = False
            # The thread target holds a WEAKREF and re-derefs it each tick: a bound
            # method would strong-reference the executor forever (the loop never ends
            # on its own), pinning the C++ worker pool and the pinned banks against GC
            # in build-many-engines scenarios. NB: the stop flag / weakref death is
            # observed only between 2 s sleeps, so teardown of the THREAD can lag up to
            # ~2 s -- it is a daemon, so neither GC of the executor (weakref breaks the
            # cycle) nor process exit waits on it.
            self._watchdog = threading.Thread(
                target=_watchdog_main,
                args=(weakref.ref(self),),
                name="cpu-moe-flag-watchdog",
                daemon=True,
            )
            self._watchdog.start()

        for bs in self.grouped_widths:
            for layer_id in range(self.num_layers):
                self._task_for(layer_id, bs, grouped=True)

        logger.info_rank0(
            f"CPU MoE executor ready: threads={nthreads} (pinned to cores "
            f"{core_ids[0]}..{core_ids[-1]}) isa={self.isa} fmt={fmt} "
            f"H={self.H} I={self.I} experts={self.num_experts} layers={self.num_layers} "
            f"top_k={self.top_k} act={activation} max_tokens={self.max_tokens}"
        )

    def _make_table(self, layers: list[torch.Tensor]) -> torch.Tensor:
        """Build a CPU int64 tensor of per-layer base addresses for one bank.

        ``layers`` is one ``[num_experts, ...]`` tensor per layer (the per-layer host
        bank contract). The C++ side stores this table's pointer and indexes
        ``tbl[layer_id]`` at call time; both the table and the layer tensors are kept
        on ``self._banks`` (GC guard) so the raw pointers stay valid for the
        executor's lifetime.
        """
        assert len(layers) == self.num_layers, (len(layers), self.num_layers)
        table = torch.tensor([t.data_ptr() for t in layers], dtype=torch.int64)
        self._banks.append(table)
        self._banks.extend(layers)
        return table

    def validate_grouped_graph(self, stream):
        if not self.grouped_widths or not self._flag_sync:
            raise RuntimeError("grouped CPU MoE requires memop Graph support")
        with torch.cuda.stream(stream):
            for width in self.grouped_widths:
                hidden = torch.zeros(width, self.H, device=self.device, dtype=torch.bfloat16)
                ids = torch.full((width, self.top_k), -1, device=self.device, dtype=torch.int32)
                weights = torch.zeros(width, self.top_k, device=self.device)
                self.decode_sync(self.decode_submit(0, hidden, weights, ids, grouped=True))
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = self.decode_sync(self.decode_submit(0, hidden, weights, ids, grouped=True))
                graph.replay()
                stream.synchronize()
                self.raise_if_unhealthy()
                if bool(output.count_nonzero()):
                    raise RuntimeError("grouped memop Graph handshake returned stale output")

    def _resolve_banks(self, banks: dict, fmt: str) -> tuple[dict, tuple[int, int]]:
        gate_up, down = banks["gate_up"], banks["down"]
        hidden, inter = int(gate_up[0].shape[2]), int(gate_up[0].shape[1] // 2)
        if len(gate_up) != self.num_layers or len(down) != self.num_layers or any(
            tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu" or not tensor.is_contiguous()
            for tensor in [*gate_up, *down]
        ):
            raise ValueError("speculative CPU experts require contiguous CPU BF16 banks")
        if any(tuple(t.shape) != (self.num_experts, 2 * inter, hidden) for t in gate_up) or any(
            tuple(t.shape) != (self.num_experts, hidden, inter) for t in down
        ):
            raise ValueError("inconsistent speculative CPU expert geometry")
        return dict(gate_up_ptr=self._make_table(gate_up).data_ptr(),
                    down_ptr=self._make_table(down).data_ptr()), (hidden, inter)

    def _io_for(self, bs: int) -> dict[str, torch.Tensor]:
        io = self._io.get(bs)
        if io is None:
            io = {
                "x": alloc_pinned_tensor(bs, self.H, dtype=torch.bfloat16),
                "ids": alloc_pinned_tensor(bs, self.top_k, dtype=torch.int32),
                "w": alloc_pinned_tensor(bs, self.top_k, dtype=torch.float32),
                "y": alloc_pinned_tensor(bs, self.H, dtype=torch.bfloat16),
            }
            if self.grouped_compat or bs in self.grouped_widths:
                io["gpu_out"] = torch.empty((bs, self.H), dtype=torch.bfloat16, device=self.device)
            self._io[bs] = io
        return io

    def _task_for(self, layer_id: int, bs: int, *, grouped: bool = False) -> int:
        key = (layer_id, bs, "grouped") if grouped else (layer_id, bs)
        task = self._tasks.get(key)
        if task is None:
            if grouped:
                if bs not in self.grouped_widths or not 0 <= layer_id < self.num_layers:
                    raise ValueError("unregistered grouped CPU MoE geometry")
                if len(self._flag_slots) >= self._flag_capacity:
                    raise RuntimeError("grouped CPU MoE flag capacity exhausted")
            io = self._io_for(bs)
            create = (self._ext.create_grouped_compat_task if self.grouped_compat else
                      self._ext.create_grouped_task) if grouped else self._ext.create_task
            task = create(
                layer_id,
                bs,
                io["x"].data_ptr(),
                io["ids"].data_ptr(),
                io["w"].data_ptr(),
                io["y"].data_ptr(),
            )
            self._tasks[key] = task
            # Allocate this (layer, bs) combo a flag slot and register its task with the
            # coordinator. Combos past the slot capacity keep the host-func path.
            if self._flag_sync and key not in self._flag_slots:
                slot = len(self._flag_slots)
                if slot < self._flag_capacity:
                    self._flag_slots[key] = slot
                    self._ext.register_flag_task(slot, task)
        return task

    def decode(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One MoE layer of decode on the CPU. Returns a GPU [bs, H] tensor.

        All ops go on the current CUDA stream so the whole thing is captured into
        the active CUDA graph (the two host nodes carry the data dependency on the
        pinned buffers, which hold this step's real routing on replay)."""
        pending = self.decode_submit(layer_id, hidden_states, topk_weights, topk_ids,
                                     grouped=self.grouped_compat)
        return self.decode_sync(pending)

    def decode_submit(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        grouped: bool = False,
    ) -> tuple:
        """Issue the D2H copies + the CPU-pool submit host node, then return without
        waiting. Lets a caller (the hybrid backend) enqueue GPU work between this and
        :meth:`decode_sync` so the CPU compute overlaps the GPU GEMM / PCIe fetch.

        ``topk_ids`` may carry ``-1`` entries (the C++ kernel skips them), so the CPU
        computes only the routes assigned to it. Returns an opaque handle to pass to
        :meth:`decode_sync`. The output tensor is allocated here so it stays live (and
        distinct from the interleaved GPU work) across the overlap window."""
        if self._ext is None:
            raise RuntimeError("CPU MoE executor is closed")
        if self._poisoned:
            raise RuntimeError("CPU MoE executor is poisoned; rebuild the executor")
        bs = hidden_states.shape[0]
        if grouped and (bs not in self.grouped_widths or (layer_id, bs, "grouped") not in self._flag_slots):
            raise ValueError("grouped CPU MoE task must be registered before capture")
        io = self._io_for(bs)

        # D2H: ship this step's activations + routing to pinned host memory.
        io["x"].copy_(hidden_states, non_blocking=True)
        io["ids"].copy_(topk_ids.to(torch.int32), non_blocking=True)
        io["w"].copy_(topk_weights.to(torch.float32), non_blocking=True)

        task = self._task_for(layer_id, bs, grouped=grouped)
        if self.grouped_compat or grouped:
            out = io["gpu_out"]
        else:
            out = torch.empty_like(hidden_states)
        key = (layer_id, bs, "grouped") if grouped else (layer_id, bs)
        slot = self._flag_slots.get(key) if self._flag_sync else None
        if slot is not None:
            # Front-end memops: done[slot]=0 then ready[slot]=1 (the coordinator's
            # doorbell). No kernel launched; no host-func round trip.
            self._cpu_moe.speculative_memop_submit(
                torch.cuda.current_stream().cuda_stream,
                self._done.data_ptr(), self._ready.data_ptr(), slot,
            )
        else:
            stream = torch.cuda.current_stream().cuda_stream
            self._ext.submit_with_cuda_stream(stream, task)
        return (bs, task, out, slot)

    def decode_sync(self, pending: tuple, add_to: torch.Tensor | None = None) -> torch.Tensor:
        """Issue the CPU-pool sync + the H2D result copy for a prior :meth:`decode_submit`,
        and return the GPU output tensor. With flag-sync the wait is a front-end stream
        memop on done[slot] (set by the CPU coordinator); otherwise a cudaLaunchHostFunc."""
        bs, task, out, slot = pending
        if slot is not None:
            # Front-end WAIT(done[slot] >= 1): blocks this stream's later nodes without
            # occupying an SM, so GPU utilization stays truthful during the CPU window.
            self._cpu_moe.speculative_memop_sync(
                torch.cuda.current_stream().cuda_stream, self._done.data_ptr(), slot,
            )
        else:
            stream = torch.cuda.current_stream().cuda_stream
            self._ext.sync_with_cuda_stream(stream, task)
        io = self._io[bs]
        out.copy_(io["y"], non_blocking=True)
        return out if add_to is None else add_to + out

    def _watchdog_tick(self, suspects: dict) -> None:
        """One watchdog sampling round (called every 2 s by ``_watchdog_main``).

        A slot is only declared dead when THREE things hold across >=10 s: its doorbell
        is still pending (ready==1 && done==0), it was already pending when first
        suspected, and the coordinator has served NOTHING on it since (flag_served_count
        unchanged). The served-count criterion kills the false-positive window: two
        point samples can land on the same slot's (independent, us-scale) pending
        windows under heavy external load, but a coordinator that made progress in
        between is alive by definition. ``suspects`` maps slot -> (first_seen,
        served_at_first_sight) and persists across ticks."""
        if self._poisoned:
            self._done.fill_(1)
            return
        stuck = (self._ready == 1) & (self._done == 0)
        active = getattr(self._ext, "flag_active_slot", lambda: -1)()
        if active >= 0 and self._done[active] == 0:
            stuck[active] = True
        if not bool(stuck.any()):
            suspects.clear()
            return
        now = time.monotonic()
        pending = set(stuck.nonzero().flatten().tolist())
        for slot in list(suspects):
            if slot not in pending:
                del suspects[slot]
        dead = []
        for slot in pending:
            served = self._ext.flag_served_count(slot)
            first_seen, served_then = suspects.get(slot, (None, None))
            if first_seen is None or served != served_then:
                suspects[slot] = (now, served)  # new suspect, or alive-but-loaded: rearm
                continue
            if now - first_seen >= 10.0:
                dead.append(slot)
        if not dead:
            return
        logger.error(
            f"cpu-moe flag watchdog: slots {dead} unanswered for >10s with no coordinator "
            "progress (wedged/dead); poisoning done[] and failing the next step"
        )
        for i in dead:
            self._err[i] = 1
        self._poisoned = True
        for i in dead:
            self._done[i] = 1  # after err: unblock the stream into a checked failure
            suspects.pop(i, None)

    def raise_if_unhealthy(self) -> None:
        """Raise if the flag watchdog fired (a doorbell stayed unanswered because the
        coordinator never responded). Called by the engine once per forward -- a single
        pinned read -- so a dead coordinator surfaces as a loud error on the next step
        instead of silently shipping stale expert outputs."""
        if self._err is not None and bool((self._err != 0).any()):
            self._poisoned = True
            raise RuntimeError(
                "CPU MoE flag-handshake watchdog fired: a decode step's doorbell was "
                "never answered by the coordinator thread (its outputs cannot be "
                "trusted). This indicates a wedged/killed coordinator; restart the "
                "engine, or set FREETOKEN_CPU_MOE_FLAG_SYNC=0 to use the "
                "cudaLaunchHostFunc sync."
            )

    def close(self) -> None:
        if self._ext is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._watchdog_stop = True
        watchdog = getattr(self, "_watchdog", None)
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join()
        # The C++ destructor joins its coordinator while the mapped flags still exist.
        self._ext = None
        self._tasks.clear()
        self._io.clear()
        self._flag_slots.clear()
        self._ready = self._done = self._err = None


def _watchdog_main(executor_ref) -> None:
    """Watchdog daemon body: weakref-deref per tick so the thread never keeps a dead
    executor alive (see the start site in ``SpeculativeBf16Executor.__init__``)."""
    suspects: dict = {}
    while True:
        time.sleep(2.0)
        executor = executor_ref()
        if executor is None or executor._watchdog_stop or executor._ready is None:
            return
        try:
            executor._watchdog_tick(suspects)
        finally:
            del executor  # drop the strong ref before the next sleep
