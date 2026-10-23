"""Initial prefix transport for the single-request eager CPU experiment."""

from dataclasses import dataclass
from contextlib import nullcontext
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time

import torch


class PrefixAborted(RuntimeError):
    pass


def prefix_range(name):
    return torch.cuda.nvtx.range(name) if os.getenv("FREETOKEN_CPU_VERIFY_NVTX") == "1" else nullcontext()


@dataclass(frozen=True)
class PrefixTicket:
    epoch: int
    length: int
    anchor: int


def gpu_local_affinity(device):
    properties = torch.cuda.get_device_properties(device)
    pci = f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0"
    node = int(Path(f"/sys/bus/pci/devices/{pci}/numa_node").read_text())
    local = set()
    if node >= 0:
        for part in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
            bounds = list(map(int, part.split("-")))
            local.update(range(bounds[0], bounds[-1] + 1))
    local &= os.sched_getaffinity(0)
    if not local:
        raise RuntimeError("prefix receiver requires an allowed GPU-local NUMA CPU")
    return node, min(local)


def prefix_shapes(config, capacity):
    kinds = config["layer_types"]
    h, k, v = (config[name] for name in
               ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
    channels = 2 * config["linear_num_key_heads"] * k + h * v
    kv = (kinds.count("full_attention"), capacity, config["num_key_value_heads"], config["head_dim"])
    return dict(keys=(kv, torch.bfloat16), values=(kv, torch.bfloat16),
                recurrent=((kinds.count("linear_attention"), h, v, k), torch.float32),
                conv=((kinds.count("linear_attention"), channels, config["linear_conv_kernel_dim"] - 1),
                      torch.bfloat16))


class PrefixReceiver:
    """Only CPU tensors and spawn-safe synchronization objects cross processes."""

    def __init__(self, config, capacity, *, device=None):
        self.capacity = capacity
        self.kinds = tuple(config["layer_types"])
        self.indices = tuple(self.kinds[:i].count(kind) for i, kind in enumerate(self.kinds))
        self.numa_node = None
        affinity, threads = os.sched_getaffinity(0), torch.get_num_threads()
        try:
            if device is not None:
                self.numa_node, cpu = gpu_local_affinity(device)
                os.sched_setaffinity(0, [cpu])
                torch.set_num_threads(1)
            self.tensors = {name: torch.zeros(shape, dtype=dtype).share_memory_()
                            for name, (shape, dtype) in prefix_shapes(config, capacity).items()}
        finally:
            if device is not None:
                os.sched_setaffinity(0, affinity)
                torch.set_num_threads(threads)
        context = mp.get_context("spawn")
        self.condition = context.Condition()
        self.epoch = context.Value("q", -1, lock=False)
        self.length = context.Value("q", 0, lock=False)
        self.aborted = context.Value("b", False, lock=False)
        self.consumers = context.Value("i", 0, lock=False)
        self.ready = context.Array("q", len(self.kinds), lock=False)

    def begin(self, epoch, length):
        with self.condition:
            if self.consumers.value or epoch <= self.epoch.value or not 0 < length <= self.capacity:
                raise ValueError("prefix begin requires a new epoch, valid length and no consumers")
            self.epoch.value, self.length.value, self.aborted.value = epoch, length, False
            self.ready[:] = [0] * len(self.kinds)

    def check(self, epoch):
        if epoch != self.epoch.value or self.aborted.value:
            raise PrefixAborted("prefix cancelled or stale epoch")

    def acquire(self, ticket):
        with self.condition:
            self.check(ticket.epoch)
            if ticket.length != self.length.value or self.consumers.value:
                raise ValueError("invalid prefix ticket or duplicate consumer")
            self.consumers.value += 1

    def release(self, epoch):
        with self.condition:
            if epoch != self.epoch.value or self.consumers.value != 1:
                raise RuntimeError("stale prefix consumer release")
            self.consumers.value = 0
            self.condition.notify_all()

    def publish(self, epoch, layer, start, end):
        with self.condition:
            if epoch != self.epoch.value or self.aborted.value:
                return False
            if not 0 <= start < end <= self.length.value or self.ready[layer] != start:
                raise ValueError("prefix completion has a gap, overlap or duplicate")
            self.ready[layer] = end
            self.condition.notify_all()
            return True

    def wait(self, epoch, layer, timeout=60):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                self.check(epoch)
                if self.ready[layer] == self.length.value:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.aborted.value = True
                    self.condition.notify_all()
                    raise PrefixAborted(f"prefix layer {layer} timed out")
                self.condition.wait(remaining)

    def abort(self):
        with self.condition:
            self.aborted.value = True
            self.condition.notify_all()

    def wait_all(self, epoch, timeout=60):
        deadline = time.monotonic() + timeout
        for layer in range(len(self.kinds)):
            self.wait(epoch, layer, timeout=max(0, deadline - time.monotonic()))


class PrefixTransfer:
    def __init__(self, receiver, engine, *, chunk_size=None):
        from freetoken.kernel.pinned import host_register, host_unregister, _load_pinned_extension

        if not hasattr(_load_pinned_extension(), "host_unregister"):
            raise RuntimeError("rebuild the pinned extension with host_unregister before layerwise transfer")

        self.receiver, self.engine = receiver, engine
        self._unregister = host_unregister
        self.registered = []
        self.pending = queue.Queue()
        self.records = []
        self.error = None
        self.closed = False
        self.accepting = False
        self.submitted = [0] * len(receiver.kinds)
        self.stream = torch.cuda.Stream(device=engine.device)
        chunk = receiver.capacity if chunk_size is None else chunk_size
        if type(chunk) is not int or chunk < 1:
            raise ValueError("positive prefix chunk capacity required")
        copies = (receiver.kinds.count("linear_attention") + receiver.kinds.count("full_attention")
                  * ((receiver.capacity + chunk - 1) // chunk))
        self.events = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(copies)]
        for pair in self.events:
            for event in pair:
                event.record(self.stream)
        self.stream.synchronize()
        self.numa_node, cpu = gpu_local_affinity(engine.device)
        affinity = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, [cpu])
            for tensor in receiver.tensors.values():
                if not tensor.numel():
                    continue
                tensor.zero_()
                host_register(tensor.data_ptr(), tensor.numel() * tensor.element_size())
                self.registered.append(tensor)
        except BaseException:
            for tensor in reversed(self.registered):
                host_unregister(tensor.data_ptr())
            raise
        finally:
            os.sched_setaffinity(0, affinity)
        self.thread = threading.Thread(target=self._complete, name="prefix_d2h_complete", daemon=True)
        self.thread.start()

    @property
    def pinned_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.registered)

    def begin(self, epoch, prompt_length):
        self.drain()
        if self.closed:
            raise RuntimeError("prefix transfer is closed")
        self.receiver.begin(epoch, prompt_length)
        self.submitted = [0] * len(self.receiver.kinds)
        self.records = []
        self.event_cursor = 0
        self.accepting = True

    def enqueue_layer(self, layer_id, start, end, final_chunk):
        r, e = self.receiver, self.engine
        if not self.accepting or self.closed:
            raise RuntimeError("prefix transfer is not accepting copies")
        r.check(r.epoch.value)
        if not 0 <= start < end <= r.length.value or final_chunk != (end == r.length.value):
            raise ValueError("invalid prefix chunk")
        index = r.indices[layer_id]
        linear = r.kinds[layer_id] == "linear_attention"
        if linear and not final_chunk:
            return
        begin = 0 if linear else start
        if self.submitted[layer_id] != begin:
            raise ValueError("prefix submission has a gap or duplicate")
        if linear:
            pool = e.linear_state_pool
            sources = dict(recurrent=pool.recurrent_states[index, 1], conv=pool.conv_states[index, 1])
            targets = {key: r.tensors[key][index] for key in sources}
        else:
            sources = dict(keys=e.kv_cache.k_cache(layer_id)[start:end, 0],
                           values=e.kv_cache.v_cache(layer_id)[start:end, 0])
            targets = {key: r.tensors[key][index, start:end] for key in sources}
        if self.event_cursor >= len(self.events):
            raise ValueError("prefix chunks exceed preallocated event capacity")
        ready, done = self.events[self.event_cursor]
        self.event_cursor += 1
        ready.record(e.stream)
        submitted_at = time.perf_counter()
        with prefix_range(f"Prefix D2H submit layer {layer_id} rows {begin}:{end}"), torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            for key, source in sources.items():
                if source.shape != targets[key].shape or source.dtype != targets[key].dtype:
                    raise ValueError("prefix source layout or dtype mismatch")
                targets[key].copy_(source, non_blocking=True)
            done.record(self.stream)
        self.submitted[layer_id] = end
        record = dict(epoch=r.epoch.value, layer=layer_id, start=begin, end=end,
                      submitted_at=submitted_at,
                      bytes=sum(t.numel() * t.element_size() for t in sources.values()))
        self.records.append(record)
        # Keep views and both events alive until DMA finishes.
        self.pending.put((done, ready, sources, record))

    def _complete(self):
        with torch.cuda.device(self.engine.device):
            while True:
                item = self.pending.get()
                try:
                    if item is None:
                        return
                    done, _, _, record = item
                    done.synchronize()
                    record["completed_at"] = time.perf_counter()
                    self.receiver.publish(record["epoch"], record["layer"], record["start"], record["end"])
                except BaseException as exc:
                    self.error = exc
                    self.receiver.abort()
                finally:
                    self.pending.task_done()

    def drain(self):
        self.pending.join()
        if self.error is not None:
            raise RuntimeError("prefix completion failed") from self.error

    def finish(self):
        self.accepting = False
        if any(n != self.receiver.length.value for n in self.submitted):
            self.receiver.abort()
            raise RuntimeError("prefix did not submit every layer")

    def abort(self):
        self.accepting = False
        self.receiver.abort()
        try:
            self.drain()
        finally:
            # Submission can fail after a copy but before its completion is queued.
            self.stream.synchronize()

    def close(self):
        if self.closed:
            return
        self.accepting = False
        self.receiver.abort()
        try:
            self.drain()
        finally:
            self.stream.synchronize()
            self.pending.put(None)
            self.thread.join()
            for tensor in reversed(self.registered):
                self._unregister(tensor.data_ptr())
            self.registered.clear()
            self.closed = True
