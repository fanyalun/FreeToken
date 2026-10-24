from collections import deque
import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.qwen_cpu_target import VerifyJob
from freetoken.engine.qwen_cpu_worker import CpuVerifierWorker
from freetoken.engine import qwen_cpu_worker


@pytest.mark.parametrize("bind_openmp", [False, True])
@pytest.mark.parametrize("start_fails", [False, True])
def test_spawn_inherits_worker_affinity_and_restores_parent(monkeypatch, bind_openmp, start_fails):
    parent_cores, worker_cores = {0, 24}, [1, 2, 25, 26]
    affinity = set(parent_cores)

    def set_affinity(pid, cores):
        nonlocal affinity
        assert pid == 0
        affinity = set(cores)

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(affinity))
    monkeypatch.setattr(os, "sched_setaffinity", set_affinity)
    monkeypatch.setenv("GOMP_CPU_AFFINITY", "0 24")
    monkeypatch.delenv("OMP_PROC_BIND", raising=False)
    monkeypatch.setattr(torch.Tensor, "share_memory_", lambda tensor: tensor)

    def start():
        assert affinity == (set(worker_cores) if bind_openmp else parent_cores)
        assert os.environ["GOMP_CPU_AFFINITY"] == ("1 2 25 26" if bind_openmp else "0 24")
        assert os.environ.get("OMP_PROC_BIND") == ("TRUE" if bind_openmp else None)
        if start_fails:
            raise RuntimeError("spawn failed")

    context = SimpleNamespace(Pipe=lambda: (Pipe(), SimpleNamespace(close=lambda: None)),
        Process=lambda **kwargs: SimpleNamespace(start=start))
    monkeypatch.setattr(qwen_cpu_worker.mp, "get_context", lambda method: context)
    config = dict(layer_types=["linear_attention", "full_attention"], linear_num_value_heads=1,
        linear_key_head_dim=2, linear_value_head_dim=2, linear_num_key_heads=1,
        num_key_value_heads=1, head_dim=2, linear_conv_kernel_dim=2, hidden_size=4, vocab_size=8)
    if start_fails:
        with pytest.raises(RuntimeError, match="spawn failed"):
            CpuVerifierWorker("unused", config, worker_cores, max_context=4, max_rows=2,
                bind_openmp=bind_openmp)
    else:
        CpuVerifierWorker("unused", config, worker_cores, max_context=4, max_rows=2,
            bind_openmp=bind_openmp)
    assert affinity == parent_cores
    assert os.environ["GOMP_CPU_AFFINITY"] == "0 24"
    assert "OMP_PROC_BIND" not in os.environ


class Pipe:
    def __init__(self):
        self.messages, self.sent = deque(), []

    def poll(self):
        return bool(self.messages)

    def recv(self):
        return self.messages.popleft()

    def send(self, message):
        self.sent.append(message)


def worker():
    w = CpuVerifierWorker.__new__(CpuVerifierWorker)
    w.prefix_receiver = None
    w.pipe = Pipe()
    w.closed, w.ready, w.initialized = False, True, True
    w.epoch = w.latest_epoch = 3
    w.inflight = None
    w.owners, w.readers, w.generations = [None, None], [None, None], [0, 0]
    w.process = SimpleNamespace(is_alive=lambda: True, exitcode=None)
    w.slots = [{"keys": torch.zeros(1, 16, 1, 2), "values": torch.zeros(1, 16, 1, 2),
        "recurrent": torch.zeros(1, 2, 2, 2), "conv": torch.zeros(1, 8, 3),
        "hidden": torch.zeros(8, 4), "predictions": torch.zeros(8, dtype=torch.int64)} for _ in range(2)]
    return w


@pytest.mark.parametrize("broken_pipe", [False, True])
def test_close_stops_child_after_reader_failure(monkeypatch, broken_pipe):
    events = []
    w = worker()
    def fail_reader():
        events.append("reader")
        raise RuntimeError("failed CUDA reader")
    def send(message):
        events.append(message["kind"])
        if broken_pipe:
            raise BrokenPipeError()
    w.readers[0] = SimpleNamespace(synchronize=fail_reader)
    w.pipe = SimpleNamespace(send=send, close=lambda: events.append("pipe_close"))
    w.process = SimpleNamespace(is_alive=lambda: True,
        join=lambda timeout: events.append("join"), terminate=lambda: events.append("terminate"))
    retained = []
    monkeypatch.setattr(qwen_cpu_worker, "_failed_reader_workers", retained)
    with pytest.raises(RuntimeError, match="failed CUDA reader"):
        w.close()
    assert events == ["reader", "close", "join", "terminate", "join", "pipe_close"]
    assert retained == [w] and w.closed
    w.close()
    assert len(events) == 6


def publish(w, slot=0, block=0, epoch=3):
    w.pipe.messages.append(dict(kind="result", slot=slot, generation=w.generations[slot],
        request_epoch=epoch, block_id=block, start=4, accepted=1, input_count=2,
        tokens=(5, 6), terminal=False, width=2))


@pytest.mark.parametrize("wait_all", [False, True])
def test_streamed_initialization_sends_barrier_policy_without_waiting(wait_all):
    from freetoken.engine.prefix_transfer import PrefixTicket

    w = worker()
    w.prefix_receiver = object()
    w.initialize_streamed(PrefixTicket(4, 2, 3), wait_all=wait_all)
    assert w.pipe.sent == [dict(kind="initialize_streamed",
        ticket=dict(epoch=4, length=2, anchor=3), wait_all=wait_all)]
    assert w.inflight == "initialize" and not w.initialized


def test_cancel_during_streamed_initialization_waits_for_ack():
    w = worker()
    aborted = []
    w.prefix_receiver = SimpleNamespace(abort=lambda: aborted.append(True))
    w.inflight, w.initialized = "initialize", False
    w.cancel(3)
    assert aborted == [True]
    w.pipe.messages.append(dict(kind="initialize_aborted", epoch=3))
    w.pipe.messages.append(dict(kind="cancelled", epoch=3))
    assert w.poll() is None
    assert w.poll()["kind"] == "cancelled"
    w.require_drained()
    assert not w.initialized


def test_slots_wait_for_last_reader_and_do_not_reuse_published_result():
    w = worker()
    job = VerifyJob(3, 0, 4, 2, (5,), 10)
    w.submit(job)
    with pytest.raises(RuntimeError, match="drained"):
        w.require_drained()
    with pytest.raises(RuntimeError, match="busy"):
        w.submit(job)
    publish(w)
    result, _ = w.poll()
    done = SimpleNamespace(ready=False)
    event = SimpleNamespace(query=lambda: done.ready)
    w.release(result, event)
    w.submit(VerifyJob(3, 1, 6, 6, (), 10))
    assert w.inflight[0] == 1
    assert w.generations == [1, 1]
    with pytest.raises(RuntimeError, match="still have readers"):
        w._free_slot()
    done.ready = True
    assert w._free_slot() == 0
    assert w.owners[0] is None


def test_cancel_discards_late_result_and_waits_for_cancel_ack():
    w = worker()
    w.submit(VerifyJob(3, 0, 4, 2, (5,), 10))
    w.cancel(3)
    publish(w)
    assert w.poll() is None
    assert w.inflight == "cancel" and w.owners[0] is None
    w.pipe.messages.append(dict(kind="initialized", epoch=3))
    assert w.poll() is None and w.inflight == "cancel"
    w.pipe.messages.append(dict(kind="cancelled", epoch=3))
    assert w.poll()["kind"] == "cancelled"
    assert w.inflight is None and not w.initialized
    with pytest.raises(RuntimeError):
        w.submit(VerifyJob(3, 1, 4, 2, (), 10))


def test_generation_mismatch_and_worker_failure_fail_closed():
    w = worker()
    w.submit(VerifyJob(3, 0, 4, 2, (), 10))
    publish(w)
    w.generations[0] += 1
    w.owners[0] = (3, 0, 2)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        w.poll()
    w.pipe.messages.append(dict(kind="error", traceback="test worker exception"))
    with pytest.raises(RuntimeError, match="test worker exception"):
        w.poll()
    w.process = SimpleNamespace(is_alive=lambda: False, exitcode=17)
    with pytest.raises(RuntimeError, match="code 17"):
        w.poll()


def test_slow_worker_poll_does_not_publish_provisional_output():
    w = worker()
    w.submit(VerifyJob(3, 0, 4, 2, (), 10))
    for _ in range(5):
        assert w.poll() is None
    assert w.inflight is not None
